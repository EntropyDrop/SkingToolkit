# 08 推理管线与路由

本章是"生产视角"的完整推理流程：从输入图片到最终 `pred_uv.png`，
中间每一步解决什么问题、有哪些可调参数。对应源码：`infer.py`、
`foreground.py`、`utils.py`（路由与 splat）、`simple_inpainting.py`、
`inference_config.py`、`run_infer.sh`。

## 8.1 输入与加载

```bash
COMBINED=/path/to/front_back.png ./run_infer.sh
```

- 输入可以是**并排拼接图**（`COMBINED`，宽度按视图数等分）、
  单独文件列表（`--view_images`）或 `FRONT`/`BACK` 两张图；
- `infer.py::load_parser` 从 checkpoint 重建模型：视图列表、
  `geometry_fit`（要求 3 类路由）、各语义维度、先验尺寸、
  占用头等全部从 `model_config` 推断，并自动挂载冻结 SigLIP2
  运行时（推理**不读训练缓存**，语义特征由运行时骨干在线提取）；
- 渲染器从 `mappings_dir` 加载与 checkpoint 视图匹配的映射文件；
- 检查点选择：`run_infer.sh` 自动取最高版本 `runs/dense_uv_parser_vN/best.pt`
  （`latest.pt` 兜底），可用 `PARSER_CHECKPOINT` 覆盖。

## 8.2 前景提取与输入构造（foreground.py）

模型对"背景"敏感，所以推理先做**确定性的前景提取**：

- `estimate_top_left_flood_foreground`：以每张视图**左上角像素**的
  颜色为种子，做**四连通 flood fill**，只移除"与种子同色且与角
  连通"的像素（容差 `FOREGROUND_FLOOD_TOLERANCE=0.03`，
  梯度容差 0.05，最大种子容差 0.20）。被角色完全包围的同色
  区域（例如裙摆内的空隙）会被保留——这正是 flood fill 而非
  全局阈值的原因。
- 产物：`foreground_probability.png`、`foreground_mask_raw.png`、
  `foreground_mask.png`、透明 `foreground_cutout.png`。
- **自适应背景**：`select_adaptive_background` 从 8 个候选纯色中
  挑一个"离前景边界最远"的颜色（距离按边界像素的 10% 分位数），
  把背景替换为它，得到 `foreground_parser_input.png` 喂给模型。
  这保证 parser 输入与训练分布一致（纯色背景），且不泄露原背景
  颜色。`FOREGROUND_METHOD=legacy` 可跳过这步。
- 原始 RGB 始终保留，用于最终的 UV 取色（取色用原图，路由用
  替换背景后的图）。

## 8.3 模型前向与辅助头

```python
outputs = parser_model(parser_rendered, view_ids=view_ids,
                       semantic_foreground=observed_foreground)
```

- `view_ids = [0, 1]` 对应 checkpoint 的视图顺序；
- 之后 `attach_projected_outer_uv_occupancy` 与
  `attach_projected_head_outer_structure` 把可选头的预测投影到
  图集（`utils.py`），供救援/修复使用；
- 若开启文本原型，还会打印每个视图 top-3 提示词分数（诊断
  "模型认为这是什么"）。

## 8.4 路由与门控（核心）

`splat_parser_predictions_to_uv_conditioning` 是推理的心脏
（`utils.py` 中约 1400 行的函数，内部调用
`_routing_from_geometry_outputs`）。它把每个"前景像素"决策为
**内层/外层/次级**，并映射到图集纹素。生产默认值见
`inference_config.py::PRODUCTION_SPLAT_DEFAULTS`，要点：

### 8.4.1 逐像素决策

1. 前景阈值：`fg_threshold=0.5`；非前景像素丢弃。
2. **路由角色**：网络输出的 3 类概率取 argmax（结合
   `route_confidence` 的几何平均，次级还结合 surface 概率）。
3. **置信度门控**：
   - 内层：默认不设门（`0.0/0.0`）；
   - 外层：严格门 `outer_route_confidence_threshold=0.80` +
     `outer_route_margin_threshold=0.55`（置信度与"次高分差"
     都要达标），外加 **纹素覆盖** `outer_uv_min_coverage=0.25`
     与**最少源像素** `outer_uv_min_source_pixels=15`（防止
     残余背景碎片变成持久的外层证据）；
   - **救援路径**：几何救援（外层独有轮廓/精确次级槽位证明）
     与语义救援（`outer_semantic_presence_threshold=0.80` 等）
     可把门放宽到 `0.60/0.25`、覆盖 0.10。内/外层重叠区域仍用
     严格门。
4. **投影纹素共识（texel consensus）**：中心加权软融合——每个
   纹素最终外层分数 = `0.40×本地概率 + 0.60×纹素级证据`
   （`geometry_route_texel_consensus_weight=0.60`）。饱和的局部
   外层预测不能绕过融合后的门。`routing_filter` 日志会报告被
   此门拒绝的外层预测。
5. **跨视角一致性（cross-view veto）**：对共享纹素，若一个视角
   强烈支持外层、另一视角看到明确背景或强烈支持内层，则**否决**
   外层（仅冲突时否决，弱证据/单视角纹素不动）：
   `geometry_cross_view_outer_consistency=true`，正/负证据门槛
   0.70/0.20，背景覆盖上限 0.25。
6. **次级像素**：单独预测精确表面槽位，映射回图集（不丢、
   不强塞到最近直接表面）。

### 8.4.2 颜色提取（grid_mode）

颜色**从不平均**：`grid_mode` 对每个图集纹素收集其"安全路由
像素"，取**出现次数最多的 8-bit RGB**（平局才用 UV 中心质量
打破）。取色 mask 与路由占用 mask 分离：优先内部前景样本，只有
与检测到的源背景相差 ≤ 8/255 的**边界样本**才被排除——真实的
内部皮肤纹素允许恰好等于背景色，而抗锯齿背景边沿不能赢下纹素。

### 8.4.3 输出 conditioning

路由结果写入 **10 通道**（或 12 通道）张量：

```text
[inner RGBA, inner evidence, outer RGBA, outer evidence]
（12 通道时每层再 +1 通道 confidence）
```

`conditioning_to_pred_uv` 把它转成"仅可见证据"的 64×64 RGBA：
未知纹素保持透明 → `parser_pred_uv.png`。

## 8.5 确定性修复（simple_inpainting.py）

`simple_symmetry_nearest_inpaint` 处理 `parser_pred_uv` 中
**内层**的缺失纹素（外层缺失保持透明——修复**不创造**外层）：

- 每个部位独立处理；
- 前/后面：矩形环，从外圈向中心；
- 左/右面：逐行、每行从两边向中间，优先**同行**源；
- 顶/底面：环遍历；
- 每缺一个纹素：先取**可用镜像**（02 章的 `mirrored_texel`），
  否则取同一部位在规范 3D 空间中**最近的定义纹素**；
- 透明残留 RGB 永不当作证据；已定义的内层与全部外层**逐字节
  保留**。

另有头部外层结构完成（`_complete_head_outer_structure`）：当
模型高置信（阈值 0.65 等）且存在 ≥2 个锚点种子时，按物理邻接
图传播补全帽檐/王冠；对称完成与"开顶环"规则（`open_top` 系列
阈值）防止把开放帽顶错误闭合。所有修复产物：

- `outputs/parser_pred_uv.png`：仅证据；
- `outputs/parser_pred_uv_simple_inpainting.png`：修复产物；
- `outputs/pred_uv.png`：最终 UV（与修复产物相同，经 alpha
  后处理 `finalize_minecraft_alpha`：内层强制不透明）。

## 8.6 诊断输出

`run_infer.sh` 默认还输出（全部在 `outputs/`）：

| 文件 | 内容 |
| --- | --- |
| `parser_conditioning.png` | 10/12 通道条件张量预览（内层 RGB 行 + 外层 RGB 行） |
| `parser_debug.png` / `parser_debug_overlay.png` | 路由/网格可视化与叠加 |
| `parser_debug_inner/outer/secondary.png` | 各路由角色 cutout |
| `parser_debug_face_raw.png`、`parser_debug_layer_face_raw.png` | **原始输入坐标**下的语义头预测（未做仿射/路由过滤） |
| `parser_debug_observed_canonical.png` | 仿射后、路由前的规范前景 mask |
| `parser_debug_geometry_grid/fill/overlay/routed_overlay.png` | 拟合几何网格与路由叠加 |
| `parser_debug_color_source.png` | 取色源预览 |
| `foreground_*.png` | 前景提取各阶段 |

> `PARSER_ONLY=true` 跳过修复，只导出解析证据与诊断；
> `SIMPLE_INPAINT_OUTPUT=` 只关掉额外副本，`pred_uv.png` 仍修复。

## 8.7 关键默认参数速查（PRODUCTION_SPLAT_DEFAULTS）

```text
fg_threshold                     0.5
outer confidence/margin 严格门   0.80 / 0.55
outer 覆盖 / 最少源像素           0.25 / 15
救援门(几何/语义)               0.60 / 0.25，覆盖 0.10
texel 共识权重                   0.60（40% 本地 + 60% 纹素级）
跨视角否决                       开启（正/负 0.70/0.20，背景覆盖上限 0.25）
颜色聚合                         grid_mode（众数，不平均）
affine_refine                    false（不移动规范几何）
```

## 8.8 推理性能与确定性

- 全部路由/修复都是**确定性**的（flood fill、argmax、众数投票、
  最近邻）；随机性仅存在于训练。
- SigLIP2 特征在线提取时按批（`SEMANTIC_RUNTIME_BATCH_SIZE=32`），
  若 checkpoint 有缓存可跳过（推理一般不重建缓存）。
- 大量门控是为"保守"服务的：**宁缺毋滥**——错误的外层颜色
  比透明缺口更糟。`ROUTING_PROFILE=balanced` 可换取更高外层
  recall。

## 本章要点

1. 推理 = 洪水填充前景 + 自适应背景 → 模型 → 置信度门控路由 →
   纹素众数取色 → 确定性修复。
2. 外层是精度优先（严格门 + 覆盖 + 源像素数 + 共识 + 跨视角
   否决），内层默认不设门；救援路径用几何/语义证据放宽门。
3. 颜色提取用"众数投票"而非平均，且取色 mask 与路由 mask 分离。
4. 修复只补内层、只在本部位内、优先镜像、其次 3D 最近邻；
   外层缺失保持透明。
5. 所有产物均可视化，便于分离"几何对齐 / 路由 / 取色"三类错误。

## 思考题

1. 为什么外层要同时要求"置信度 + 边距 + 覆盖 + 最少源像素"
   四道门？每道门分别防什么？
2. grid_mode 为什么取众数而不是平均？提示：残余背景、抗锯齿
   边缘、以及"内部纹素恰好等于背景色"的合法情况。
3. 跨视角否决为什么只在"强证据冲突"时生效，而不是硬求交集？
4. 修复算法为什么不允许补外层？如果允许，会引入什么风险？

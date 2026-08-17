# 03 系统总览与代码地图

本章把整个系统画成一张图，并给出"文件 → 职责"的地图，让你在
读任何一个大文件之前先知道它为什么存在。

## 3.1 端到端数据流

**训练阶段**（`run_dense_uv_parser_training.sh` → `train.py`）：

```text
skins/ 目录下的 64×64 皮肤 PNG
   │  SkinUVDataset 读取、瘦模型归一化(alice_to_steve)、背景填充
   ▼
渲染器 DifferentiableRenderer（固定视角、固定相机）
   │  forward_view: grid_sample + alpha 合成
   ▼
渲染图 + 精确稠密标签（build_dense_parser_batch）
   │  foreground / layer / route_role / part / face / surface / uv
   ▼
冻结 SigLIP2 特征（预缓存为 mmap 数组，或在线提取）
   │  global 768-d + spatial 768×14×8（FP16）
   ▼
DenseUVParserNet 前向
   │  每个像素: 前景、路由角色(内/外/次级)、表面槽位、置信度、…
   ▼
损失 = 分类损失 + 可微渲染分支（soft splat → 重渲染 → RGB/alpha 损失）
   │
   ▼
runs/dense_uv_parser_vN/best.pt（按 loss_hard_uv_color_selection 选优）
```

**推理阶段**（`run_infer.sh` → `infer.py`）：

```text
front_back.png（正面+背面并排，或两个单独文件）
   │  前景提取: 四连通 flood fill（从左上角种子色）
   │  自适应背景替换 → parser 输入（RGB，alpha=1）
   ▼
DenseUVParserNet（加载 best.pt + 冻结 SigLIP2 运行时）
   │  前景/路由角色/表面/置信度 + 文本原型相似度
   ▼
路由与门控（splat_parser_predictions_to_uv_conditioning）
   │  置信度门控、纹素共识、跨视角否决、几何/语义救援、颜色投票
   ▼
10/12 通道 conditioning 张量（内层 RGBA+证据, 外层 RGBA+证据）
   │
   ├─► parser_pred_uv.png（仅可见证据，未知纹素透明）
   ▼
确定性拓扑修复（simple_inpainting.py）
   │  镜像优先 → 3D 最近邻；每部位独立；绝不动外层
   ▼
pred_uv.png（最终 64×64 RGBA 皮肤）
```

## 3.2 代码地图

所有文件位于 `SkingToolkit/dense_uv_parser/`（行数为撰写时的大致规模，
供参考——**大文件是常态**，用下面"如何读"的路线图进入）。

| 文件 | 规模 | 职责 | 学习章节 |
| --- | --- | --- | --- |
| `uv_layout.py` | ~170 行 | 图集矩形布局、mask、alpha 后处理 | 02 |
| `uv_topology.py` | ~290 行 | 图集拓扑、3D 坐标、镜像、外层图、填充顺序 | 02 |
| `SkingToolkit/renderer.py` | ~320 行 | 可微渲染器（正向过程） | 02 |
| `model.py` | ~1170 行 | `DenseUVParserNet` 与所有子模块 | 04 |
| `skin_dataset.py` | ~75 行 | 皮肤数据集（读取 + 归一化） | 05 |
| `semantic_targets.py` | ~115 行 | 图集级语义标签（外层存在/覆盖率、头部占用） | 05 |
| `semantic_backbone.py` | ~545 行 | 冻结 SigLIP2 / TIPSv2 视觉塔封装 | 06 |
| `semantic_cache.py` | ~135 行 | mmap 语义特征缓存的读取端 | 06 |
| `cache_semantic_features.py` | ~270 行 | 语义特征缓存的构建端（离线编码） | 06 |
| `semantic.py` | ~145 行 | 运行时骨干的构建与挂载 | 06 |
| `losses.py` | ~725 行 | 核心损失：focal、route swap、texel 一致性、prior 正则 | 07 |
| `train.py` | ~5480 行 | 训练主循环、全部其余损失、指标、checkpoint、CLI | 07 |
| `run_dense_uv_parser_training.sh` | ~690 行 | 训练启动器：锁、版本目录、缓存、全部超参 env 默认值 | 07 |
| `inference_config.py` | ~115 行 | 生产推理默认值（预处理 + splat 门控） | 08 |
| `foreground.py` | ~140 行 | 前景 mask、自适应背景选择 | 08 |
| `utils.py` | ~6200 行 | 路由、splat、调试可视化、批构建、占位头挂载（大杂烩） | 05/08 |
| `infer.py` | ~2800 行 | 推理 CLI：加载、前处理、路由调用、输出 | 08 |
| `simple_inpainting.py` | ~545 行 | 确定性拓扑修复（含头部外层结构完成） | 08 |
| `run_infer.sh` | ~610 行 | 推理启动器：checkpoint 选择、输出路径、env 覆盖 | 08 |
| `runtime.py` | 7 行 | 设备选择 | — |
| `test_foreground.py` / `test_semantic_parser.py` / `test_affine_routing.py` | — | 单元测试（`unittest`） | 09 |

## 3.3 如何阅读这个代码库

代码总量约 2 万行，直接从头读会迷失。建议路线：

1. **先跑通再读**：用 `test_imgs/` 跑一次 `run_infer.sh`，对照
   08 章的输出文件清单，建立"黑盒"直觉。
2. **读小文件建立概念**：`uv_layout.py` → `uv_topology.py` →
   `skin_dataset.py` → `foreground.py`。这些是纯逻辑，没有深度
   学习复杂度。
3. **读渲染器**：`SkingToolkit/renderer.py`，理解正向过程。
4. **读模型结构**：`model.py` 的 `__init__` 和 `forward`（先跳过
   各融合模块的细节，用 04 章的图对照）。
5. **再读训练/推理的主干**：`train.py::main()` 和
   `infer.py::main()`。这两个函数虽然长，但结构是线性的：
   参数校验 → 加载 → 循环。用 grep 跳着读，例如：
   ```bash
   grep -n "def run_epoch\|def main\|optimizer = \|dataset =" train.py
   ```
6. **最后啃 `utils.py`**：它包含推理路由的完整实现
   （`splat_parser_predictions_to_uv_conditioning` →
   `_routing_from_geometry_outputs`），是 08 章的实战代码。
   不要一次读完，按函数名定位。

> **工程经验**：`train.py` / `infer.py` / `utils.py` 都是数千行的
> 大文件，这是本项目逐步演进的结果（主 README 记录了每个特性的
> 历史缘由）。阅读时把它当作"文档 + 代码"混合体：先用
> `grep "^(def |class )"` 拿到函数清单，再按需深入。

## 3.4 两条关键约定

1. **输入永远是 RGB 语义**：parser 输入总是"不透明"张量
   （alpha 固定为 1），背景颜色随机化/替换，因此 PNG/JPEG 输入
   格式一致。真实 alpha 只用于训练标签与输出皮肤。
2. **视图顺序不可变**：模型以视图索引为条件，checkpoint 记录了
   训练时的视图顺序（如 `front_left,back_left`）。推理必须保持
   相同顺序，否则前/后映射会互换。

## 3.5 单元测试

从包含 `SkingToolkit/` 的目录运行：

```bash
python -m unittest discover -s SkingToolkit/dense_uv_parser -p 'test_*.py'
```

测试覆盖前景提取、语义解析、仿射路由等，是理解"某个函数到底
返回什么形状/语义"的最快途径（测试里充满了形状断言与数值断言）。

## 本章要点

1. 系统 = 渲染器（正向，固定）+ 解析器（逆向，可训练）+ 确定性修复。
2. 训练与推理共享同一套几何映射与路由策略（validation 与推理
   使用相同配置）。
3. 阅读顺序建议：小文件 → 渲染器 → 模型 → 主循环 → utils 路由。
4. 所有超参通过环境变量注入 shell 启动器，再透传给 `train.py` /
   `infer.py` 的 argparse。

## 思考题

1. 为什么验证集必须使用与推理完全相同的路由门控配置？
2. 训练时背景颜色随机化，但推理时把背景替换为自适应纯色，
   这两者如何配合避免模型"记住"灰色背景？
3. 用 `grep -n "def " utils.py | wc -l` 数一数 `utils.py` 的函数
   数量，猜一猜哪些函数属于"训练标签生成"，哪些属于"推理路由"。

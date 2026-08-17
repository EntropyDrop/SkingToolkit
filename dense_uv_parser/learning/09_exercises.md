# 09 练习与实验

本章练习分为四类：**运行练习**（动手跑）、**阅读练习**（在代码里
找答案）、**消融实验**（理解设计取舍）、**扩展项目**（开放题）。
难度从 ★（基础）到 ★★★（进阶）。

## 9.1 运行练习

### R1（★）跑通推理

用仓库自带的测试图跑一次完整推理：

```bash
cd SkingToolkit/dense_uv_parser
FRONT=../test_imgs/walk_front_both_layer_ortho_pyvista.png \
BACK=../test_imgs/walk_back_both_layer_ortho_pyvista.png \
./run_infer.sh
```

（也可以换成 `FRONT=../test_imgs/banana_front.png BACK=../test_imgs/banana_back.png`，
或对并排图 `COMBINED=../test_imgs/banana_output1.png` 使用。）

对照 08 章的输出清单，回答：

1. `parser_pred_uv.png` 与 `pred_uv.png` 的差别是什么？哪些区域
   是修复出来的？
2. 打开 `parser_debug_inner.png` / `parser_debug_outer.png` /
   `parser_debug_secondary.png`，说出"次级"像素在图中出现在哪些
   部位（提示：帽子下的脸、透明孔）。
3. 在日志里找到 `routing_filter` 与 `foreground_filter=` 两段
   JSON，解释每个计数的含义。

### R2（★）输入变化的影响

- 用 `FOREGROUND_METHOD=legacy` 再跑一次，比较
  `parser_pred_uv.png` 与默认 flood 版本的差异；
- 换一张纯色背景的输入（可以用脚本把背景涂成白色），观察
  flood fill 与自适应背景是否仍然正确；
- 用 `PARSER_ONLY=true` 跑一次，确认 `pred_uv.png` 不再生成。

### R3（★★）把推理代码当库用

写一个小脚本，加载 checkpoint 与渲染器，对一张 64×64 皮肤
**先渲染再解析**，计算解析结果与 GT 皮肤的像素级差异
（例如内层区域的平均 RGB MAE）。这一步验证"正向→逆向"闭环。

```python
# 骨架（参考 infer.py::load_parser / load_view_images 的用法）
import torch
from SkingToolkit.renderer import DifferentiableRenderer
from SkingToolkit.dense_uv_parser.infer import load_parser
from SkingToolkit.dense_uv_parser.runtime import get_device

device = get_device("auto")
model, args = load_parser("runs/dense_uv_parser_vN/best.pt", device)
views = args["views"].split(",")
renderer = DifferentiableRenderer(mappings_dir=args["mappings_dir"]).to(device)
# 1) 读皮肤 -> tensor (1,4,64,64)
# 2) renderer.forward_view(skin, view) 得到正/背渲染
# 3) 参考 infer.py main() 中 flood/parser_input/forward/splat 的调用顺序
```

> 提示：`infer.py` 的 `main()` 里从 `rendered = load_view_images(...)`
> 到 `splat_parser_predictions_to_uv_conditioning(...)` 之间的代码
> 就是完整步骤，可以直接照抄。

## 9.2 阅读练习

### C1（★）UV 布局

在 `uv_layout.py::minecraft_layer_rects` 中找出：

1. 头部外层（帽子）的 6 个矩形坐标；
2. 左右腿的 `decor_offset` 为什么不同；
3. `is_slim=True` 时哪些矩形发生变化。

用 `build_part_layer_masks` 生成 mask，`save_image` 存成图片
肉眼验证。

### C2（★★）拓扑推导

阅读 `uv_topology.py`，回答：

1. `_surface_coordinate(face=0, u, v, ...)` 与 `face=1` 的坐标
   为什么 x 要取反？
2. 外层膨胀 `expansion` 为什么头部是 1.0、其他部位 0.5？
3. `mirrored_texel` 用 `torch.cdist` 找最近邻；如果改用图集坐标
   平移，会在哪些部位出错？写一个 5 行脚本验证你的推断。
4. 外层邻接图阈值 1.20 的依据是什么？（提示：1.125 与 √2。）

### C3（★★）渲染合成

阅读 `SkingToolkit/renderer.py::forward_view`，手写伪代码描述
`trust_composite` 分支的决策条件，并用一个"透明帽子"的皮肤
渲染图验证你的理解。

### C4（★★★）路由代码追踪

在 `utils.py` 中找到 `_routing_from_geometry_outputs`，画出它的
执行流程（前景 → 角色 → 门控 → 救援 → 共识 → 否决 → 取色），
并标出与 08 章 8.4 小节各参数对应的代码行。

### C5（★★）训练主循环

在 `train.py::run_epoch` 中回答：

1. 梯度累积如何实现（`gradient_accumulation_steps`）？
2. 混合精度用哪个 API？bf16 与 fp16 路径有何不同
   （`autocast_context` / `build_grad_scaler`）？
3. `best.pt` 的更新条件写在哪里？

## 9.3 消融实验

> 消融实验需要能训练的环境（GPU）。若条件有限，可以只读
> 主 README 中对应的段落 + 本节"预期结果"，然后对照
> `run_dense_uv_parser_training.sh` 的默认值推理原因。

### A1（★）几何 only 消融（无语义）

```bash
SEMANTIC_BACKBONE=none CACHE_SIGLIP_FEATURES=false \
  RUN_NAME=dense_uv_parser_ablation_nosem \
  MAX_SAMPLES=20000 EPOCHS=1 ./run_dense_uv_parser_training.sh
```

预期：路由角色准确率下降，尤其是"外层 vs 次级"（透明孔）；
对比验证集 `acc_route_role` 与 `loss_primary_route_swap`。

### A2（★）文本提示词消融

```bash
SIGLIP_TEXT_PROMPT_FUSION=false \
  RUN_NAME=dense_uv_parser_ablation_notext \
  MAX_SAMPLES=20000 EPOCHS=1 ./run_dense_uv_parser_training.sh
```

预期：帽子类外层 recall 下降（文本原型正是为"配饰"设计的）；
对比 `hard_recall_outer` 与 `precision_trusted_route`。

### A3（★★）路由先验消融

```bash
ROUTE_ROLE_SPATIAL_PRIOR=false \
  RUN_NAME=dense_uv_parser_ablation_noprior \
  MAX_SAMPLES=20000 EPOCHS=1 ./run_dense_uv_parser_training.sh
```

预期：常见结构（刘海/帽檐位置）的稳定区域出现更多零星错误；
对比 `acc_route_role` 与 debug 预览中的
`route_role_prior` 图（A3 无此图）。

### A4（★★）门控策略对比（推理侧，无需训练）

对同一 checkpoint，比较：

```bash
ROUTING_PROFILE=conservative ./run_infer.sh     # 默认
ROUTING_PROFILE=balanced ./run_infer.sh
```

统计 `outputs/pred_uv.png` 中外层纹素数量与 `routing_filter`
日志中"被拒外层"数量，解释两种配置的取舍。

### A5（★★★）背景增强消融

```bash
BACKGROUND_AUGMENT=false RUN_NAME=... MAX_SAMPLES=20000 EPOCHS=1 \
  ./run_dense_uv_parser_training.sh
```

预期：模型把固定灰背景当特征，推理遇到其他背景时前景/路由
退化。这解释了 README 中"输入仍是 RGB、背景随机化"的设计。

## 9.4 扩展项目

### P1（★★）添加新视图

修改训练/推理的 `VIEWS`，加入第三个视角（如侧面），需要：

1. 在 `differentiable_minecraft_renderer` 中为该视角生成映射文件；
2. 修改 `VIEWS`/`PRIVILEGED_VIEWS` 并保持视图顺序一致；
3. 评估跨视角一致性是否提升 `cross_view_outer_precision`。

观察 `MultiViewSemanticFusion` 的 transformer 是否利用新视图
信息。这是一个很好的"系统集成"练习。

### P2（★★★）颜色聚合器对比

在 `utils.py` 的 `SPLAT_COLOR_AGGREGATIONS` 中新增一种聚合方式
（例如"中位数 + 色度离群剔除"），在推理侧接入并对比
`hard_rgb_mae_inner/outer` 与主观质量。

### P3（★★★）稀疏外层改进

`PREDICT_OUTER_UV_OCCUPANCY=true` 默认关闭。开启它重新训练
（注意：不能续训旧 checkpoint，参数集不同），把
`OUTER_UV_OCCUPANCY_ROUTING=true` 加入推理，评估外层占用头的
precision/recall 以及它作为"救援路径"的价值。

### P4（★★★）可复现性工程

用 `REPRODUCIBLE=true SEED=1234` 与 `SEED=5678` 各训一个小模型，
比较两轮训练的指标差异，讨论"训练可复现到什么程度"（注意
`EPOCHS=1`、数据顺序、cudnn benchmark 等影响因素）。

### P5（开放）逆渲染泛化

本系统只支持标准 Steve 手臂（`infer.py::load_parser` 直接拒绝
非 Steve）。研究 `uv_layout.py` 中的 `is_slim` 参数，列出让解析器
支持 Alex（瘦手臂）模型需要修改的代码位置（布局、拓扑、渲染器、
数据集、模型配置），并说明哪些模块可以复用。

## 9.5 综合自测题

1. 用一段话向同学解释"几何锚定 + 语义条件 + 确定性修复"。
2. 画出从 `COMBINED` 输入到 `pred_uv.png` 的完整数据流，标出
   每个箭头对应的函数名。
3. 列举 5 个"推理时宁可拒绝也不犯错"的设计，说明各自代价。
4. 如果训练数据里 90% 的皮肤没有帽子，模型会如何表现？
   哪些损失项在防止"模型干脆不预测外层"？
5. 解释为什么 `best.pt` 的指标包含 RGB MAE 而不只是占用
   IoU（提示：稀疏图集陷阱）。

## 参考答案提示

- C1: `minecraft_layer_rects()` 头部 6 面的 `decor_offset` 均为
  `(32,0)`；腿部两部位分别为 `(-16,0)` 与 `(0,16)`。
- C2.3: 左右腿/手臂在图集上的朝向不同（镜像部位），简单平移
  会错位；`mirrored_texel` 用 3D 反射 + 最近邻。
- C2.4: 外层最大膨胀步长 1.125（部位深度膨胀），同面斜对角
  ≥√2≈1.414，阈值 1.20 位于两者之间。
- A1–A3 的预期差异都写在主 README 对应小节，实验后请回来对照。

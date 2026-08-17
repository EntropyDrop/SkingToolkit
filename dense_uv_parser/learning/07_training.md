# 07 训练流程与损失设计

本章讲训练：启动器做了什么、损失函数为什么这样设计、看哪些指标、
checkpoint 怎么选。对应源码：`run_dense_uv_parser_training.sh`、
`train.py`、`losses.py`。

## 7.1 启动器：run_dense_uv_parser_training.sh

启动器是一个"环境变量即超参"的 bash 脚本，依次做：

1. **防重入锁**：`flock`（不可用时跳过，macOS 没有 flock）防止
   两个训练进程竞争同一个 run 目录。
2. **run 版本管理**：自动选择 `runs/dense_uv_parser_vN`（N 自增），
   或用 `RESUME=latest` 续训最高版本；`RUN_NAME` 可覆盖。
3. **解析资源**：`DATA_DIR`（默认 `../skins`，即
   `SkingToolkit/skins/`）、`MAPPINGS_SIZE`（默认 256×512）、
   自动定位 `differentiable_minecraft_renderer/mappings_256x512`。
4. **读取全部超参默认值**：核心默认值（可在命令行前缀覆盖）：
   - 数据：`MAX_SAMPLES=180000`，`VIEWS=front_left,back_left`，
     `PRIVILEGED_VIEWS=front_right,back_right`（训练共 4 视图）；
   - 优化：`EPOCHS=1`（默认只训 1 个 epoch！），`BATCH_SIZE=16`，
     `GRADIENT_ACCUMULATION_STEPS=2`（有特权视图时），
     `LR=2e-4`，`LR_SCHEDULE=cosine`，`MIN_LR_RATIO=0.05`，
     `--weight_decay=1e-4`（AdamW），`MIXED_PRECISION=bf16`，
     `SEED=1234`；
   - 语义：`SEMANTIC_BACKBONE=siglip2`，
     `SIGLIP_MODEL=google/siglip2-base-patch16-224`，
     `CACHE_SIGLIP_FEATURES=true`，`SIGLIP_CACHE_SPATIAL=true`，
     `SIGLIP_TEXT_PROMPT_FUSION=true`；
   - 结构：`ROUTE_ROLE_SPATIAL_PRIOR=true`（32×16、cap 1.5、
     dropout 0.10），`PREDICT_HEAD_OUTER_STRUCTURE=true`（projected
     模式、input version 4），`PREDICT_OUTER_UV_OCCUPANCY=false`；
   - 路由门控：与推理一致的生产默认（outer 0.80/0.55、
     coverage 0.25、min source 15、texel consensus 0.60、
     cross-view consistency 等，详见 08 章）；
   - 损失权重：一大堆 `LAMBDA_*`（见 7.3）。
5. **构建/复用语义缓存**：调用 `cache_semantic_features.py`
   （首次运行构建，后续复用，见 06 章）。
6. **执行 `python train.py`**，透传所有参数。

> 注意两个不同的 LR 默认：启动器设 `LR=2e-4` 并传给 train.py；
> train.py 里 argparse 的默认 `--lr` 是 1e-4。以启动器为准。
> 另外默认 `EPOCHS=1`：这是"单 epoch 大数据集"配方（18 万皮肤
> 一 epoch 已足够），主 README 里提到过"one-epoch recipe"。

## 7.2 训练主循环：train.py

`main()` 的流程（线性，建议对照源码）：

1. 参数校验（大量 `ValueError`，把互斥/依赖的参数组合挡在训练前）；
2. `seed_everything`（可复现模式还设置 `CUBLAS_WORKSPACE_CONFIG`）；
3. 建数据集 + `random_split`（`val_split=0.1`，固定种子）；
4. 建语义缓存或运行时骨干；编码文本提示词；
5. 构造 `DenseUVParserNet`（`geometry_fit` 模式）；
6. 优化器：**AdamW**（`lr=args.lr, weight_decay=1e-4`）；
   可选 GradScaler（fp16 时）；
7. `run_epoch` 循环（训练/验证），内含：
   - 渲染批次 → 构建标签（05 章）→ 可选背景随机化 → 语义特征
     （缓存或在线）→ 前向 → 计算损失 → 梯度裁剪
     （`clip_grad_norm_`，默认 `--grad_clip=1.0`）→ 反向；
   - 学习率：**绝对 epoch 余弦调度**（从 checkpoint epoch 计算，
     不存调度器状态，因此可安全续训），终点为 `LR×0.05`；
   - 每 `log_every` 步打印指标；验证集使用与推理完全相同的
     路由与门控；
8. 保存 checkpoint：`latest.pt` 每次覆盖，`best.pt` 按
   `loss_hard_uv_color_selection` 取最优；**原子写入**（先写临时
   文件再 rename），训练中启动的推理进程永远读到完整 checkpoint；
9. 保存预览图到 `runs/<run>/previews/`：
   - `epoch_XXXX.png`：预测内/外层 RGB 行 + GT 内/外层 RGB 行；
   - `epoch_XXXX_debug.png`：路由角色诊断、先验图、拟合网格等；
   - `epoch_XXXX_outer_occupancy.png`（若开启占用头）。

## 7.3 损失函数设计

损失分为三大类：**逐像素分类损失**、**语义辅助损失**、以及
**可微渲染分支损失**。

### 7.3.1 逐像素分类损失（losses.py）

- **前景 BCE**（`LAMBDA_FOREGROUND=1.0`）。
- **路由角色交叉熵**（`LAMBDA_LAYER=1.0`）：3 类（内/外/次级），
  类别权重有上下限（`ROUTE_CLASS_WEIGHT_FLOOR=0.75`，
  `ROUTE_OUTER_CLASS_WEIGHT_CAP=0.90`）——外层是稀有类，权重
  不能无限放大。
- **外层 false-positive / false-negative focal 损失**
  （`outer_false_positive_loss` / `outer_false_negative_loss`，
  `LAMBDA_OUTER_FALSE_POSITIVE=1.0`、`LAMBDA_OUTER_FALSE_NEGATIVE=0.75`，
  gamma 3.0 / 2.0）：互补的两项 focal 项，**精度优先**——把
  "内层误判为外层"（代价高：会在皮肤上画出错误装饰）惩罚得比
  "外层漏判"更重。
- **主路由 swap 损失**（`primary_route_swap_loss`，
  `LAMBDA_PRIMARY_ROUTE_SWAP=1.0`，gamma 2.0）：内层与外层
  **宏平衡**的 swap 项，避免像素数不平衡掩盖内↔外误判。
- **投影纹素一致性损失**（`projected_texel_consistency_loss`，
  `LAMBDA_ROUTE_TEXEL_CONSISTENCY=0.25`）：多个源像素若映射到
  同一个 GT 层/纹素，它们的路由概率应当一致。
- **置信度损失**（`LAMBDA_ROUTE_CONFIDENCE=0.25`）：训练
  `route_confidence` 头预测"路由是否正确"（校准）。
- **路由先验正则**（`route_prior_regularization`）：L2 + TV，
  见 04 章。
- 可选 `LAMBDA_UV/LAMBDA_PART/LAMBDA_FACE/...` 只在非 geometry
  模式下使用（本系统生产模式不用）。

### 7.3.2 语义辅助损失（train.py 内实现）

- 外层存在性/覆盖率：`LAMBDA_SEMANTIC_PRESENCE=0.25`、
  `LAMBDA_SEMANTIC_COVERAGE=0.25`（对 `outer_presence` /
  `outer_coverage` 头）。
- 文本原型路由：`LAMBDA_TEXT_PROMPT_ROUTE=0.35`（默认开启时）。
- 跨视角外层可见性（`cross_view_outer_visibility_loss`，
  `LAMBDA_CROSS_VIEW_OUTER_VISIBILITY=0.25`）：把每个外层纹素的
  预测 alpha 与 GT alpha 比较（含透明候选——那些暴露内层的
  透明外层纹素是"硬负样本"，`OUTER_VISIBILITY_HARD_NEGATIVE_FRACTION=0.20`、
  `WEIGHT=0.75`）。
- 特权视角蒸馏（`privileged_view_outer_distillation_loss`，
  `LAMBDA_PRIVILEGED_VIEW_DISTILLATION=0.25`）：主视角的高置信
  外层预测作为教师，教特权视角。
- 头部外层结构（一堆 `LAMBDA_HEAD_OUTER_*`）：存在/覆盖/8×8
  占用/拓扑/对称性/组件召回/开顶环形状/配饰分类/路由连通性。
- 可选外层占用（`LAMBDA_OUTER_UV_OCCUPANCY=0.0` 默认关闭）：
  平衡 BCE + Dice（`OUTER_UV_OCCUPANCY_DICE_WEIGHT=0.50`）。

### 7.3.3 可微渲染分支（differentiable_geometry_losses）

这是训练里最有意思的部分：**把预测"渲染回去"再算损失**。

1. 路由角色与表面 logits → softmax 概率；
2. 通过固定的渲染器 UV 候选做**软 splat**（`soft_splat_geometry_
   predictions_to_uv`）：每个纹素按概率加权收集 RGB/alpha，得到
   一张"预测的临时皮肤"；
3. 把这张皮肤**渲染回所有配置视图**；
4. 与真实渲染比较：软 UV 的 RGB/alpha 误差 + 多视图重渲染的
   RGB/alpha 误差（`LAMBDA_SOFT_UV_RGB=0.25`、
   `LAMBDA_SOFT_UV_ALPHA=0.35`、`LAMBDA_RENDER_RGB=0.20`、
   `LAMBDA_RENDER_ALPHA=0.25`），其中 **alpha 权重大于 RGB**
   （错误的实心外层比漏掉不确定的外层更有害）；
5. 层召回损失（`focused_visible_layer_recall_loss`，
   `LAMBDA_SOFT_UV_INNER_RECALL=0.50` / `OUTER=0.50`）：只统计
   **实际可见**的 GT 投影，且其中一半权重集中在最差的 10% 纹素
   （`SOFT_UV_RECALL_HARD_FRACTION=0.10`）——防止眼睛、脸部等
   小区域在平均值里消失。

六个 `LAMBDA_*` 全设 0 即退化为纯分类训练。这套损失让"错误的
内外层选择"同时收到颜色与轮廓梯度，而不只是交叉熵梯度。

## 7.4 指标与 checkpoint 选择

训练日志里值得关注的核心指标（主 README 有完整清单）：

- 硬指标：`hard_iou_inner`、`hard_precision_outer`、
  `hard_recall_outer`、`hard_iou_outer`、`hard_rgb_mae_inner`、
  `hard_rgb_mae_outer`；
- 路由质量：`loss_primary_route_swap`、`loss_route_texel_
  consistency`、`acc_route_role`、`precision_secondary`、
  `recall_secondary`、`precision_trusted_route`、
  `coverage_trusted_route`、`confidence_mae`；
- 跨视角：`cross_view_outer_precision` / `recall`、
  `loss_cross_view_outer_visibility`；
- 占用头（若开启）：`outer_uv_occupancy_precision` / `recall`。

**checkpoint 选择**：`best.pt` 取验证集上
`loss_hard_uv_color_selection` 最低的 epoch——该指标 = 硬内/外层
占用分数 + 硬内/外层 RGB MAE。这样"稀疏但颜色正确"的图集不会被
"占用精确但颜色错误"的图集挤掉。`latest.pt` 只是最新权重，
推理必须用 `best.pt`。

> 可复现性：`REPRODUCIBLE=true` 会设置 `PYTHONHASHSEED` 与
> `CUBLAS_WORKSPACE_CONFIG`；`STRICT_DETERMINISM` 进一步约束。
> 完全可复现需要固定数据顺序、关闭 cudnn benchmark 等，启动器
> 已提供相应开关。

## 7.5 常见训练问题速查

| 现象 | 排查方向 |
| --- | --- |
| 外层全透明 / 稀疏 | 外层 recall 权重、`OUTER_UV_MIN_SOURCE_PIXELS`、置信度门控过严 |
| 外层误涂（把内层当外层） | `LAMBDA_OUTER_FALSE_POSITIVE`、外层置信度门控 |
| 训练/验证差距增大 | `FEATURE_DROPOUT=0.10`（默认开启） |
| 小区域（眼睛）丢失 | `SOFT_UV_RECALL_HARD_FRACTION` 硬尾权重 |
| 续训旧 checkpoint 报错 | 新特性（如 route prior、占用头）要求全新训练 |
| 显存不足 | 减小 `BATCH_SIZE` / `SEMANTIC_RUNTIME_BATCH_SIZE` / 关特权视图 |

## 本章要点

1. 一切超参都是环境变量；启动器负责锁、版本、缓存与默认值。
2. 优化器 AdamW + 绝对 epoch 余弦 LR + bf16 + 梯度裁剪。
3. 损失 = 分类（focal、swap、texel 一致、置信度校准）+
   语义辅助（存在/覆盖/跨视角/头部结构）+ 可微渲染分支（软
   splat 重渲染，alpha 优先）。
4. 验证集用与推理相同的路由配置；`best.pt` 按硬 UV 颜色选择
   指标选取；checkpoint 原子写入。
5. 默认单 epoch、18 万样本、4 视图（2 主 + 2 特权）。

## 思考题

1. 为什么"错误的实心外层"比"漏掉的外层"更严重？请从最终皮肤
   的视觉效果回答。
2. 软 splat 分支为什么能向"颜色"传导梯度？提示：bilinear
   grid_sample 对 UV 坐标可微。
3. 如果 checkpoint 选择只看占用 IOU 而不看颜色，会发生什么？
   （答案藏在 `loss_hard_uv_color_selection` 的定义里。）
4. 特权视角蒸馏的教师是"主视角的高置信预测"，为什么置信度
   门槛很重要？

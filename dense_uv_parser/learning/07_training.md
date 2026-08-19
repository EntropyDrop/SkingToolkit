# 07 训练流程与损失设计

本章系统讲解训练启动脚本、损失函数体系、监控指标与 Checkpoint 选优机制。
对应源码：`run_dense_uv_parser_training.sh`、`train.py`、`losses.py`。

## 7.1 启动器：run_dense_uv_parser_training.sh

启动脚本采用"环境变量即超参数"的设计，核心配置如下：

- **数据与视角**：
  - `MAX_SAMPLES=180000`（18 万高品质 Minecraft 皮肤）。
  - `VIEWS=front_left,back_left`（主视角）。
  - `PRIVILEGED_VIEWS=front_right,back_right`（特权训练视角，共 4 视角联合蒸馏）。
- **优化与精度**：
  - `EPOCHS=1`（单 epoch 大规模数据集配方）。
  - `BATCH_SIZE=16`，`GRADIENT_ACCUMULATION_STEPS=2`。
  - `LR=2e-4`，`LR_SCHEDULE=cosine`，`MIN_LR_RATIO=0.05`。
  - `MIXED_PRECISION=bf16`（BFloat16 AMP 混合精度，显存占用减半且数值稳定）。
- **3D 结构与特征**：
  - `CROSS_VIEW_SPATIAL_FUSION=true`（启用 `UVMultiViewSpatialFusion` 3D UV 空间特征投影融合）。
  - `PREDICT_OUTER_UV_OCCUPANCY=true`（开启 64×64 3D UV 外层直接占有率监督）。
  - `OUTER_SILHOUETTE_MIN_PIXELS=1`（放宽细小突起阈值，保护皇冠尖角与猫耳等 1-pixel 饰品）。
  - `OUTER_UV_MIN_SOURCE_PIXELS=4`（保护单视角高对比度微小文字与细节）。

## 7.2 综合损失函数体系

训练损失由 **逐像素路由分类**、**3D UV 外层真值占有率** 与 **可微重渲染重构** 三大支柱构成：

### 1. 逐像素路由与焦点损失（Pixel-wise Route Losses）

- **路由角色交叉熵（`LAMBDA_LAYER=1.0`）**：监督内层、外层与次级表面。
- **外层精度优先 Focal 损失（`outer_false_positive_loss` / `outer_false_negative_loss`）**：
  - `LAMBDA_OUTER_FALSE_POSITIVE=1.0`（$\gamma=3.0$）：极重度惩罚将内层误判为外层（防止下巴面罩或浮空杂色）。
  - `LAMBDA_OUTER_FALSE_NEGATIVE=0.75`（$\gamma=2.0$）：适度惩罚外层漏判。
- **主路由交换对称损失（`primary_route_swap_loss`, `LAMBDA_PRIMARY_ROUTE_SWAP=1.0`）**：
  - 维持内外层分类的宏观平衡，避免大面积内层掩盖稀疏外层。
- **投影纹素一致性损失（`LAMBDA_ROUTE_TEXEL_CONSISTENCY=0.25`）**：
  - 约束投射到同一图集纹素的所有 2D 像素保持路由概率一致。

### 2. 多任务稠密语义辅助监督（Dense Multi-Task Semantic Losses）

- **15 类开集语义焦点损失（`dense_semantic_supervision_loss`, `LAMBDA_DENSE_SEMANTICS=0.30`）**：
  - 直接监督 2D 逐像素分类为眼镜、帽子、连帽衫、面部五官等 15 类语义原型。
  - **类别逆频重平衡（Inverse Frequency Class Balancing）**：针对微小配件（如眼镜框仅占画面 ~1% 像素）容易被 99% 的大面积背景与衣服像素淹没的问题，计算有效像素的类别频次逆平方根权重。
  - **外层微配件 2.50 倍梯度强力加权（Accessory Gradient Boost）**：对所有外层饰品类别（类别 0..7）施加 2.50 倍的梯度反传倍率，强力惩罚将眼镜框、皇冠等微小突起漏判或切断为内层皮肤的行为。
- **开集提示词路由正则化（`LAMBDA_TEXT_PROMPT_ROUTE=0.35`）**：
  - 约束提示词注意力响应与几何路由的联合一致性。

### 3. 3D UV 外层直接真值监督（Outer UV Occupancy Losses）

- **全局 64×64 占有率损失（`LAMBDA_OUTER_UV_OCCUPANCY=0.55`）**：
  - 结合 BCE 与 Dice Loss（`OUTER_UV_OCCUPANCY_DICE_WEIGHT=0.50`），直接用 GT 外层 Alpha 图监督 3D UV 融合特征。
- **难负样本惩罚（`OUTER_HARD_NEGATIVE_WEIGHT=0.50`）**：
  - 重点惩罚易产生浮空杂色的非饰品区域。
- **路由-占有率一致性约束（`LAMBDA_ROUTE_OCCUPANCY_AGREEMENT=0.30`）**：
  - 强制 2D 像素路由判决与 3D UV 全局占有率预测高度对齐。

### 4. 可微重渲染分支（Differentiable Re-rendering Losses）

通过 `soft_splat_geometry_predictions_to_uv` 将预测概率软映射回 64×64 临时皮肤，并调用 `DifferentiableRenderer` 重新投射为 2D 渲染图：

- **Soft UV 重构损失（`LAMBDA_SOFT_UV_RGB=0.50`, `LAMBDA_SOFT_UV_ALPHA=0.60`）**：
  - 监督临时 UV 贴图与真实贴图的颜色及透明度。
- **多视角 2D 重渲染损失（`LAMBDA_RENDER_RGB=0.40`, `LAMBDA_RENDER_ALPHA=0.50`）**：
  - 强力惩罚重渲染画面与输入图像的像素级残差。
- **高难度纹素聚焦召回（`LAMBDA_SOFT_UV_INNER/OUTER_RECALL=0.75`）**：
  - 对误差最大的 10% 纹素加权，防止眼睛、徽标等小面积特征被平均化。

## 7.3 指标监控与 Checkpoint 选优

- **核心硬指标**：`hard_iou_inner`、`hard_precision_outer`、`hard_recall_outer`、`hard_rgb_mae_inner`。
- **选优标准（`best.pt`）**：
  - 系统以验证集上的 `loss_hard_uv_color_selection` 最低点作为最佳权重。
  - 该指标兼顾了内外层占有率精确度（IOU）与色彩绝对平均误差（RGB MAE），确保选出的模型既无浮空假外层，又具有高饱和度锐利原色。

## 本章要点

1. 启用 `UVMultiViewSpatialFusion` 与 3D UV 外层直接占有率监督（`PREDICT_OUTER_UV_OCCUPANCY=true`）。
2. 可微重渲染 Loss 直接以 2D 输入画面像素对齐为导向，倒逼高保真逆向。
3. 降低细小轮廓阈值，完整保留皇冠尖角等 1 像素级立体特征。

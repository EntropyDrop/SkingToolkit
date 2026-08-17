# 04 网络架构（model.py）

本章拆解 `DenseUVParserNet`。读之前请先理解 01 章的哲学：**网络不学
部位/面/UV 坐标，只学"这个像素应该路由到哪个候选表面"以及"路由
是否正确"**。

## 4.1 总体结构

`model.py` 中 `DenseUVParserNet.__init__` 的默认配置（`geometry_fit`
模式，即生产配置）构成如下主干：

```text
输入 (B, 4, H, W) RGBA + view one-hot 常量平面 (B, V, H, W)
   │  ┌────────────────────────────────────────────┐
   ▼  │             U-Net（base_channels=32）        │
stem (3×3 conv block) ──► down1 ──► down2 ──► down3 ──► mid
   ▲                                    │
   │              up0 ◄── up1 ◄── up2 ◄──┘ (+skip)
   │                                     │
   └─────────────────────────────────────┘
                │
           共享特征图 x（B, 32, H, W）
                │
   ┌────────────┼─────────────────────────────────────────┐
   ▼            ▼            ▼                ▼            ▼
 foreground  layer(3类)   part(6) face(6)  layer_face(12) uv(2)
 (1通道)   内/外/次级    (仅非geometry模式)         uv_x/uv_y(64类)
                │
            route_confidence(1)   surface(N槽)   affine(3)
            （可选，默认开）        （可选）        （可选）
```

`ConvBlock` = `Conv3×3 → GroupNorm → SiLU` 两次；`DownBlock` 用
stride-2 卷积下采样（无池化）；`UpBlock` 双线性上采样后与 skip
拼接。这是一个标准 U-Net，没有任何花哨的主干改动。

**视图条件**：当 `view_classes > 0` 时，每个样本的 `view_ids` 被
编码为 one-hot，并作为常量平面拼在输入通道后面（`forward` 中
`torch.cat([x, view_one_hot.expand(...)], dim=1)`）。这让同一个
网络可以处理"正面"与"背面"两种视图而不混淆。

## 4.2 语义融合：三条旁路，全部"零初始化起步"

模型最重要的设计是：**所有语义旁路的最终投影都用零初始化**，因此
训练开始时它们对输出的贡献为零，网络等价于旧的纯几何解析器；
语义修正必须通过监督信号"学出来"，而不是一开始就扰动稳定的几何
解。三条旁路是：

### (1) `MultiViewSemanticFusion` — 全局特征 FiLM 调制

- 输入：每个视图的 pooled 全局语义特征（768 维原始 SigLIP2 特征，
  见 06 章），按视图分组。
- 结构：`LayerNorm → Linear(768→128) → GELU` 投影；加上可学习的
  `view_embedding`；经过 1 层、4 头的 transformer encoder（视图之间
  互相注意，得到"前/后视图互相参考"的上下文）；每视图 token 与
  池化摘要拼接后，经 `modulation` 输出 **FiLM 的 scale 与 shift**
  （各 128 维），在 U-Net 瓶颈处对特征做仿射调制：
  `x = x * (1 + scale) + shift`。
- 初始化：`modulation` 最后一层 Linear 权重与偏置置零 ⇒ 初始
  scale=shift=0，即恒等变换。

### (2) `SpatialSemanticFusion` — 空间特征适配器

- 输入：逐 patch 的 SigLIP2 空间特征（768 通道的 14×8 特征图，
  因为 256×512 输入 letterbox 到 224×224 后是 14×14，裁剪后为
  14×8——见 06 章）。
- 结构：`LayerNorm → 1×1 Conv(768→64) → GELU → 1×1 Conv(64→瓶颈)`，
  残差加到 U-Net 特征上，双线性插值到特征图尺寸。
- 初始化：输出投影置零 ⇒ 初始无贡献。

### (3) `TextPromptRouteFusion` — 文本原型路由证据

- 输入：空间特征 + 全局特征 + 一组**固定的文本提示词嵌入**
  （完整提示词见 `semantic_backbone.py::DEFAULT_SIGLIP_ROUTE_PROMPTS`，
  描述"凸起的头发/帽子/兜帽/眼镜/面罩/耳机/动物耳朵/围巾高领/
  外套袖子/四肢装饰/平面头发/平面五官纹理"等；在训练启动时由
  冻结的 SigLIP2 文本塔编码，随 checkpoint 保存，推理不再运行
  文本塔）。
- 计算：全局特征与提示词嵌入做余弦相似度
  `einsum("nc,pc->np")`，乘 logit scale 加 bias；空间 patch 特征
  与提示词嵌入在 32 维共享子空间里算逐位置相似度
  `einsum("nchw,pc->nphw")`，得到 `prompt_count` 张相似度图。
  全局分数只保留一个**有界残差**（0.10·tanh(标准化后的分数)），
  因为冻结的 pooled 分数是"弱配饰分类器"，不能让它压制可训练的
  局部证据。局部 + 全局残差 → 1×1 卷积混合 → `route_classes`
  个通道的路由 logits，加到主干 `layer` 头输出上。
- 初始化：最后的卷积置零 ⇒ 初始为无操作。
- 该分支回答的是"帽檐 vs 刘海 vs 透明孔"这类**需要语义**的路由
  难题。`SIGLIP_TEXT_PROMPT_FUSION=false` 可做消融。

## 4.3 输出头（heads）

| 头 | 输出 | 说明 |
| --- | --- | --- |
| `foreground` | (B,1,H,W) | 前景概率 |
| `layer` | (B,3,H,W) | **路由角色**：0=内层(inner)，1=外层(outer)，2=次级/背面(secondary)。`geometry_fit` 用 3 类（旧 checkpoint 是 2 类，加载时会拒绝） |
| `route_confidence` | (B,1,H,W) | 该像素路由正确的概率（用于推理门控） |
| `part` | (B,6,H,W) | 部位（仅非 geometry 模式；生产模式不用） |
| `face` | (B,6,H,W) | 面（同上） |
| `layer_face` | (B,12,H,W) | 层×面联合（同上） |
| `uv` | (B,2,H,W) | UV 回归（sigmoid 归一化，非 geometry 模式） |
| `uv_x` / `uv_y` | (B,64,H,W) | UV 分类（64 类，非 geometry 模式） |
| `surface` | (B,N,H,W) | 精确表面槽位（映射文件里 N 个候选表面，如复合层/几何回退层） |
| `affine` | (B,3) | [tx, ty, log_scale] 全局仿射残差（默认关闭/置零，推理时默认 `affine_refine=false`） |
| `outer_presence` / `outer_coverage` | (B,6) | 从语义摘要预测每个部位的外层存在性与覆盖率 |
| 头部外层结构头 | 多个 | 6 面的 8×8 占用、对称性、开顶环等（见 4.5） |
| `outer_uv_occupancy` | (B,1,64,64) | 可选：整张图集的外层 alpha 占用（默认关闭） |

## 4.4 学习到的固定视图路由先验（route role spatial prior）

`route_role_spatial_prior=True`（默认）时，模型含一个可学习参数
`route_role_prior`，形状 `(view_classes, 3, 32, 16)`：对每个视图、
每个路由角色，是一张低分辨率 logit 图。它学习的是**统计规律**——
"在正面视图的这块区域，通常是刘海/帽檐/后脑勺"这类常见结构位置。

使用方式（`forward`）：

- 按视图索引取出对应先验，`tanh` 截断到 ±`route_prior_logit_cap`(1.5)
  （防止先验过强）；
- 训练时以 `route_prior_dropout`(0.10) 的概率整张丢弃（防止模型
  过度依赖先验而忽略图像）；
- 双线性插值到特征图尺寸后**加到** `layer` logits 上；
- 训练用 L2 + 全变差（TV）正则（`LAMBDA_ROUTE_PRIOR_REGULARIZATION`
  =0.001，`ROUTE_PRIOR_TV_WEIGHT`=1.0）保持先验平滑且弱。

它只是"软统计偏置"：图像条件的 CNN 输出可以覆盖它。几何增强
（平移/缩放）与它不兼容，因为先验假设规范坐标。

## 4.5 外层占用 GNN 与头部结构头

### `ProjectedOuterUVTopologyHead`（可选外层占用头）

当 `predict_outer_uv_occupancy=True` 时，模型把共享特征图上每个
像素"投影"到外层图集纹素，聚合出每个外层纹素的证据向量
（投影特征均值 + 前景覆盖率 + outer 概率统计 + 支持视图数），
然后在**外层纹素的物理邻接图**（02 章的 `outer_edge_index`）上做
消息传递：

```text
节点特征 = 证据投影 + part embedding + face embedding + 局部UV位置 + 全局上下文
   ──► 3~4 层 OuterUVGraphBlock（自投影 + 邻居均值 + 残差 + LayerNorm）
   ──► 线性头 → 每个外层纹素的 alpha logit（bias 初始化为 -1，偏稀疏）
```

`OuterUVGraphBlock` 用 `index_add_` 做邻居聚合、按度数归一化——
这是最朴素的 GCN 风格消息传递，值得作为 GNN 入门案例阅读。

> 该头是**辅助**的：输入特征从主干**分离（detach）**，训练只更新
> 该头自身，不能扰动主路由表示；推理默认不参与路由
> （`OUTER_UV_OCCUPANCY_ROUTING=false`），仅作为诊断/可选证据。

### 头部外层结构头（predict_head_outer_structure）

帽子/王冠/帽檐是最难的部分：它们常常只在一两个视角可见、形状小、
且"帽子是否闭合"决定修复算法能不能补。模型从语义摘要预测：

- 6 个头部面各自的 8×8 占用图（`head_outer_face_occupancy_head`，
  或投影模式 `head_outer_projected_head` 在头部外层的子图上
  做 GNN）；
- 面存在性、覆盖率、左右对称性分数、以及"闭合侧环 / 开顶环"
  结构标志（`head_outer_accessory_head` 等）。

这些预测**不门控**推理路由（只作为修复阶段的可信度证据），但
训练时允许梯度回传改进主干。`simple_inpainting.py` 里的
`_complete_head_outer_structure` 用它们决定是否把候选外层纹素
"补全"成完整结构（需要 ≥2 个种子锚点）。

## 4.6 forward 的数据流小结

```text
x = cat(RGBA, view_one_hot)
s0..s3 = stem/down 塔
bottleneck = mid(s3)
bottleneck += SpatialSemanticFusion(spatial_features)     # 空间旁路
scale, shift = MultiViewSemanticFusion(global, view_ids)  # FiLM
bottleneck = bottleneck * (1+scale) + shift
x = up 塔（带 skip）
x = features(x)  （+ Dropout2d(0.10) 训练时）
layer_logits = layer(x)
layer_logits += TextPromptRouteFusion(spatial, global)    # 文本旁路
layer_logits += route_prior(view_id)                      # 空间先验
输出 dict：foreground / layer / confidence / 各辅助头
```

## 4.7 参数与规模

`count_parameters(model)` 统计可训练参数量。生产配置下主干
`base_channels=32`，瓶颈 256 通道；语义融合 128 通道、4 头、
1 层 transformer；空间适配器 64 通道；文本原型 32 通道。
完整模型为千万级参数，其中绝大部分在 U-Net 主干。

> 动手验证：`python -c "from SkingToolkit.dense_uv_parser.model
> import DenseUVParserNet, count_parameters; m = DenseUVParserNet(...);
> print(count_parameters(m))"`（需要合适的参数，见
> `infer.py::load_parser` 中的构造方式）。

## 本章要点

1. 主干 = 标准 U-Net + 视图 one-hot 条件。
2. 三条语义旁路（全局 FiLM / 空间适配器 / 文本原型）全部零初始化，
   保证"先几何、后语义修正"的训练行为。
3. 路由角色 3 类（内/外/次级）是核心分类任务；部位/面/UV 由固定
   几何提供，网络不学。
4. 可学习视图先验提供"常见结构出现在哪里"的软统计偏置。
5. 外层占用与头部结构是辅助头：前者 detach 自主干，后者允许
   回传；推理阶段都不直接门控路由。

## 思考题

1. 为什么三条旁路要零初始化？如果随机初始化，训练初期会发生什么？
2. `TextPromptRouteFusion` 为什么把全局提示词分数只当作 0.10 的
   tanh 残差，而不是直接相加？
3. 视图 one-hot 作为常量平面拼进输入，与"分视图训练两个网络"
   相比有什么优缺点？
4. 为什么外层占用头要 detach 主干特征，而头部结构头不 detach？
   提示：考虑稀疏占用目标对主表示的稳定性影响，以及头部结构
   目标的信息量。

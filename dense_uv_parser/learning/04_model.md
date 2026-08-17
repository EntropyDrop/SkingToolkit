# 04 网络架构（model.py）

本章拆解 `DenseUVParserNet` 的核心架构与前向传播数据流。
核心设计理念：**网络不回归坐标，而是结合语义先验与 3D UV 空间跨视角对齐，推断每个可见像素的路由角色（内层/外层/次级）与置信度**。

## 4.1 总体架构

`DenseUVParserNet` 的主体由 **多任务 U-Net 主干** 与 **UVMultiViewSpatialFusion 3D 跨视角融合模块** 构成：

```text
输入各视角图片 (B, 4, H, W) + 视图 one-hot 条件
   │
   ▼
U-Net 编码器 (Stem ──► Down1 ──► Down2 ──► Down3 ──► Bottleneck)
   │
   ├─► SpatialSemanticFusion: 注入 SigLIP2 空间 Patch 语义特征
   ├─► MultiViewSemanticFusion: 全局语义特征 FiLM 仿射调制 (Scale & Shift)
   │
   ▼
U-Net 解码器 (Up2 ──► Up1 ──► Up0)
   │
   ▼
UVMultiViewSpatialFusion (3D UV 空间跨视角投影融合)
   │  1. 收集各视角的 2D 特征图
   │  2. 沿 3D 视线反向 Scatter 投射到 64×64 UV 空间
   │  3. 在 3D UV 空间做跨视角全局多尺度融合 (Conv2D + GroupNorm + GELU)
   │  4. 沿 3D 映射将融合后的 UV 上下文 Gather 广播回各视角 2D 像素
   ▼
稠密预测头 (Heads)
   │  • foreground: 前景分割掩码
   │  • layer: 路由角色 (0: Inner, 1: Outer, 2: Secondary)
   │  • surface: 精确复合表面与几何回退表面槽位
   │  • route_confidence: 路由置信度
   │  • outer_uv_occupancy: 64×64 全局外层占有率预测图
```

## 4.2 3D UV 空间跨视角融合（UVMultiViewSpatialFusion）

传统 2D 图像网络在处理多视角输入时，通常只能在 2D 屏幕空间做简单的通道拼接或全局平均，但这破坏了 3D 物理一致性（例如正面左上角与背面右上角在 2D 上距离极远，但物理上同属于左耳）。

`UVMultiViewSpatialFusion` 彻底解决了这一问题：
1. **反向投影（Scatter to UV）**：利用预计算的 `static_mappings`（包含每个像素对应的 UV 索引 `flat_uv`），将前视角、后视角等所有输入的 2D 局部特征，精确投射到 64×64 的 UV 图集上。
2. **3D UV 空间特征聚合**：在物理 UV 空间中，无论来自哪个视角的特征，都在其对应的身体纹素处汇聚。通过两层残差卷积与 GroupNorm，网络在 3D 身体展开面上提取 360° 无缝环绕的上下文表征。
3. **正向反投（Gather to 2D Views）**：将聚合了全局 3D 信息的 UV 特征，重新依据坐标查表回填给各个视角的 2D 像素，极大地增强了网络对遮挡、层级和立体饰品的辨识能力。

## 4.3 语义旁路：SigLIP2 多模态先验

模型引入了三条语义旁路（均采用零初始化，确保初始状态不扰乱几何主干）：

1. **`MultiViewSemanticFusion`（全局特征 FiLM 调制）**：
   - 提取 SigLIP2 768 维全局语义向量。
   - 经过 1 层 4 头 Transformer 编码器实现视角交互，输出 Scale 与 Shift 在 U-Net 瓶颈层进行仿射调制：$x = x \cdot (1 + \text{scale}) + \text{shift}$。
2. **`SpatialSemanticFusion`（空间 Patch 适配器）**：
   - 提取 SigLIP2 14×8 空间语义 Patch 特征，通过 1×1 卷积与双线性插值残差注入 U-Net 特征。
3. **`TextPromptRouteFusion`（文本提示词原型）**：
   - 使用固定的文本提示词嵌入（如 "3d protruding hat", "flat facial texture", "outer hoodie collar" 等）。
   - 计算图像空间特征与文本提示词的余弦相似度图，为 `layer` 路由头提供显式的语义证据。

## 4.4 核心输出头详解

| 输出头 | 输出形状 | 物理含义 |
| :--- | :--- | :--- |
| `foreground` | `(B, 1, H, W)` | 像素属于角色的前景概率 |
| `layer` | `(B, 3, H, W)` | 3 类路由角色：0: 内层 (Inner), 1: 外层 (Outer), 2: 次级表面 (Secondary) |
| `surface` | `(B, N, H, W)` | $N$ 个预计算表面槽位分类（用于穿透透明孔洞映射深层表面） |
| `route_confidence` | `(B, 1, H, W)` | 路由预测的标定置信度，供推理阶段硬门控使用 |
| `outer_uv_occupancy` | `(B, 1, 64, 64)` | 64×64 3D UV 图集上的全局外层存在性与占有率预测 |

## 本章要点

1. U-Net 主干结合视图 one-hot 条件处理多视角输入。
2. `UVMultiViewSpatialFusion` 通过 3D 射线在 64×64 UV 空间实现跨视角特征物理对齐与融合。
3. 冻结 SigLIP2 视觉-语言模型通过全局 FiLM、空间适配器与文本原型提供强先验。

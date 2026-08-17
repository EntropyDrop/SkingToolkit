# 02 UV 图集与固定几何

本章讲清楚"正向"过程：皮肤图集长什么样、Minecraft 角色的几何是什么、
渲染器如何把图集变成渲染图。这是理解后续所有监督信号的基石。
对应源码：`uv_layout.py`、`uv_topology.py`、`SkingToolkit/renderer.py`。

## 2.1 64×64 皮肤图集（skin atlas）

Minecraft 的经典皮肤是一张 **64×64 RGBA 图片**，其中每个像素叫一个
**纹素（texel）**。整个图集按"展开的长方体"方式排布：角色由 6 个
长方体（cuboid）组成，每个长方体展开成 6 个面（face），每面是一块
矩形区域。

6 个部位（part；命名与
`differentiable_minecraft_renderer/config.py::full_part` 一致）：

| 索引 | 部位 |
| --- | --- |
| 0 | 头（head，8×8×8） |
| 1 | 躯干（body，8×12×4） |
| 2 | 左臂（left_arm，4×12×4） |
| 3 | 右臂（right_arm，4×12×4） |
| 4 | 左腿（left_leg，4×12×4） |
| 5 | 右腿（right_leg，4×12×4） |

> "左/右"按角色的视角命名（与渲染器配置一致）；对玩家而言画面
> 上左右会互换，理解时以配置为准即可。

6 个面（face）及其在规范 3D 空间中的位置（见 `uv_topology.py` 的
`_surface_coordinate`，w/h/d 为部位的宽/高/深）：

| 索引 | 面 | 坐标特征 |
| --- | --- | --- |
| 0 | 前面 | z = +depth/2 |
| 1 | 后面 | z = −depth/2，且 x 取反 |
| 2 | 右面 | x = +width/2 |
| 3 | 左面 | x = −width/2 |
| 4 | 底面 | y = −height/2 |
| 5 | 顶面 | y = +height/2 |

> "右/左"按代码的 ±x 约定命名；不同工具对"角色右/观察者右"的
> 命名可能相反，理解时以坐标符号为准。

**内层与外层（inner/outer layer）**：每个部位有两套矩形：内层
（base，皮肤本体）和外层（decor，帽子、外套等装饰层）。外层矩形
由内层坐标加上固定的偏移 `decor_offset` 得到。例如头部前面
`(8,8)` 起 8×8 的内层，其外层在 `(40,8)`（偏移 `(32,0)`），
也就是经典布局中"帽子位于头部区域右侧"的那块 8×8 区域。

经典布局（64×64，宽方向为 x，高方向为 y，坐标原点在左上角；
仅画出头部区域作为示例，括号内为面索引）：

```text
y=0  ┌───────────┬───────────┐
     │ 头-底面(4) │ 头-顶面(5) │
     │ (8,0)     │ (16,0)    │
     ├───────────┼───────────┼───────────┬───────────┤
y=8  │ 头-左面(3) │ 头-前面(0) │ 头-右面(2) │ 头-后面(1) │
     │ (0,8)     │ (8,8)     │ (16,8)    │ (24,8)    │
     │           │           │           │           │
     │ 头外层：外层 = 内层 + decor_offset(32,0)           │
     │ (32,8)    │ (40,8)    │ (48,8)    │ (56,8)    │
y=16 ────────────────────────────────────────────────────
     │ 躯干(20,20 起) … 手臂、腿 … 各自的 6 面矩形           │
     ...
```

> 精确的矩形列表在 `uv_layout.py::minecraft_layer_rects()` 中，每个
> 部位返回 6 个面 `(内层x, 内层y, 宽, 高, 外层偏移dx, 外层偏移dy)`。
> `is_slim=True` 时手臂宽度从 4 变为 3（Alex 模型），但本项目的
> 解析器只支持标准 Steve 手臂（见 `infer.py::load_parser` 的检查）。

`uv_layout.py` 提供：

- `minecraft_layer_rects(is_slim=False)`：返回 6×6=36 个面的矩形与
  外层偏移。
- `build_part_layer_masks(is_slim=False)`：把矩形画成
  `(6 部位, 1, 64, 64)` 的内层/外层 mask 张量。
- `build_uv_masks()`：合并所有部位，得到整张图集的
  inner/outer 有效区域 mask。
- `finalize_minecraft_alpha()`：把模型输出的 alpha 阈值化，并强制
  内层区域不透明（`enforce_base_alpha`）——因为皮肤内层永远存在，
  只有外层才允许透明。

## 2.2 固定几何拓扑（uv_topology.py）

`build_simple_uv_topology()` 用上面的矩形信息，离线构建一份完整的
图集元数据（`SimpleUVTopology`，用 `lru_cache` 缓存只构建一次），
包括：

- `valid`：64×64 中哪些纹素属于图集（`True`）——图集不是全满的，
  有大量空白区域。
- `layer / part / face`：每个纹素的层、部位、面标签；
  无效纹素用 `INVALID_LAYER = 2`、`INVALID_PART = 6`、
  `INVALID_FACE = 6` 填充（即"等于类别数"的哨兵值）。
- `local_uv`：每个纹素在**所在面内**的归一化坐标 `(u, v) ∈ [0,1]`。
- `world_position`：每个纹素中心在**规范 3D 空间**中的坐标
  （单位：纹素）。`_surface_coordinate(face, u, v, w, h, d)` 把面上
  的 `(u,v)` 换算成 3D 坐标——例如面 0（前面）是
  `(x, y, +depth/2)`，面 1（后面）是 `(-x, y, -depth/2)` 等。
  外层纹素还要按部位做**向外膨胀**（`expansion = 1.0`（头部）或
  `0.5`（其他部位）），因为外层几何比内层大一圈——这正是"帽子比
  头大"的几何本质。
- `mirrored_texel`：每个纹素关于 X 轴对称的镜像纹素（左右手/左右腿
  互映）。实现方式：把自身世界坐标的 x 取反，然后找镜像部位中
  **3D 距离最近**的纹素（`torch.cdist` + `argmin`），而不是简单
  按图集坐标平移——因为图集上的"左右"方向在展开后并不整齐。
- `inner_fill_order`：内层纹素的**确定性填充顺序**（供 08 章的修复
  算法使用）：前面/后面/顶面/底面按"从外圈到中心"的环（ring）
  顺序；左面/右面按行、每行从两边向中间。
- `outer_flat_indices` + `outer_edge_index`：外层纹素集合以及
  **物理邻接图**。关键设计：邻接关系**不能**用图集上的像素邻接
  （UV islands 在 PNG 里相邻不代表 3D 相邻，而相邻的立方体面在
  图集上常常不相邻），而是在 3D 空间中用 `torch.cdist` 找距离
  ≤ 1.20 的纹素对（最大外层步长是 1.125 纹素，同面对角线至少
  √2，阈值留出了间隙）。这张图被外层占用头（04 章）和头部结构
  修复（08 章）使用。

`build_outer_uv_graph()` 返回外层图集的扁平索引与节点坐标下的边；
`build_head_outer_face_indices()/build_head_outer_face_graph()` 只取
头部外层的 6×8×8 纹素（face-major 顺序）。

> **读代码提示**：`_surface_coordinate` 和 `_inward_ring_key` 是
> 纯几何函数，值得逐行推一遍。`_inward_ring_key` 用 `min(u, v,
> w-1-u, h-1-v)` 计算"第几环"，再用边与偏移量排序——这就是
> "从外圈到中心"的环遍历。

## 2.3 可微渲染器（SkingToolkit/renderer.py）

`DifferentiableRenderer` 是一个 `nn.Module`，把 `(B, 4, 64, 64)`
的皮肤批渲染成任意已注册视角的 `(B, 4, H, W)` 图片。它不学任何
参数，所有映射都是**预计算的离线数据**：

- 每个视角对应一个 `*_mapping.pt` 文件（默认来自
  `differentiable_minecraft_renderer/mappings_256x512/`），其中
  `inner_uv_map / outer_uv_map` 是每个屏幕像素 → 图集 UV 坐标的
  映射表，`inner_mask / outer_mask` 是该视角下内/外层可见性 mask。
- 渲染时把 UV 坐标归一化到 [-1,1]，用 `F.grid_sample` 从图集采样
  （`sampling_mode` 默认 bilinear，可选 nearest），然后做 alpha
  合成：`final_rgb = outer_alpha·outer_rgb + (1-outer_alpha)·(inner_alpha·inner_rgb + (1-inner_alpha)·bg)`。

> 注意：图集 alpha 是二值的（数据集强制，不存在半透明纹素），但
> 默认的 bilinear 采样会在内外层交界处把 alpha 插值成 0~1 的分数值，
> 因此渲染图上依然会出现"内外层颜色混合"的抗锯齿边缘像素（轮廓
> 边缘还会与背景混合）；`nearest` 模式可避免这种"发明颜色"的过渡
> （主 README 提到严格导出预览会用它）。
- 除基础的内/外两层外，映射文件还可能带**复合表面层**
  （`composite_uv_layers`，例如把帽子+头作为一个整体）和
  **几何回退层**（`geometry_uv_layers`，针对头部装饰的深度排序
  表面）。`forward_view` 会按层从后到前合成，并仅在"首可见层是
  装饰层"或"首可见层索引 ≤ 1"时信任复合结果（`trust_composite`），
  否则回退到普通两层合成。这些表面槽位在训练时被用来生成
  `surface` 标签（见 05 章），让模型能把"透过透明帽子看到的深处
  表面"映射到正确的图集纹素。

**视角（views）**：每个视角是一个命名相机。系统默认使用正交
（orthographic）前/后视图：

- 推理契约（checkpoint 元数据中的 `views`）：
  `walk_front_both_layer_ortho,walk_back_both_layer_ortho`；
- 训练启动脚本默认用 `front_left,back_left`，另加训练专用的
  "特权视角" `front_right,back_right`（用于跨视角一致性蒸馏，
  不参与验证）。
- 默认视图尺寸为 256×512。

模型必须知道每个输入属于哪个视角（`view_ids`），因为正面与背面
渲染即使内容相同，UV 映射也不同。

## 2.4 从渲染图到"每个像素属于哪个纹素"

训练时，正向渲染 + 预计算映射可以直接给出**精确的稠密标签**
（详见 05 章 `build_dense_parser_batch`）：对每个屏幕像素，从
映射表查到它对应的表面槽位、层、部位、面、UV 坐标，再对照图集
alpha 判断它是否真的可见。这套"渲染器自己给自己出题"的机制是
整个系统不需要人工标注的原因。

推理时，映射表同样被用来做**反向路由**：网络预测每个像素的
"表面槽位/层"后，从映射表反查它对应的图集纹素，把颜色"投"回
图集（splat，见 08 章）。

## 本章要点

1. 图集 = 6 部位 × 6 面 × 2 层（内层/外层）的固定矩形排布。
2. 拓扑 = 每个纹素的层/部位/面/局部 UV/3D 坐标/镜像纹素/填充顺序，
   全部由固定几何离线推导。
3. 外层邻接图必须用 3D 距离构建，不能用图集像素邻接。
4. 渲染器 = 预计算 UV 映射表 + `grid_sample` + alpha 合成 +
   复合/几何回退表面。
5. 正向过程完全确定，因此可以自动生成训练标签。

## 思考题

1. 为什么头部外层膨胀系数是 1.0 而四肢是 0.5？提示：想一想帽子
   要包住头，而袖子只是比手臂"厚一圈"。
2. `mirrored_texel` 为什么不用图集坐标平移实现？找一张皮肤图，
   观察左右手臂在图集上的朝向，验证你的推理。
3. 如果渲染器用 `nearest` 采样而不是 `bilinear`，对"从渲染图
   反推纹素颜色"有什么影响？对训练标签呢？

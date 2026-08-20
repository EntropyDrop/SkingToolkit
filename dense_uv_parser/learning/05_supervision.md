# 05 数据与监督信号

本章回答一个问题：**训练标签从哪里来？** 答案是：从渲染器自己来。
这是本项目最优雅的设计——不需要任何人工标注。

## 5.1 数据集：skin_dataset.py

`SkinUVDataset` 读取目录下的 64×64 皮肤图片（PNG/WebP/JPG）：

- `load_skin()`：转 RGBA；**瘦模型（Alex）归一化**——检查
  像素 `(47, 52)` 的 alpha，若透明则判定为 Alex 模型，调用
  `mc_skin_utils.alice_to_steve` 转成 Steve 布局（因为解析器的
  固定几何是标准 Steve 手臂）；
- 半透明像素的 alpha 置为 255（皮肤语义上只有"有/无"）；
- 透明区域填充背景色（训练默认 (128,128,128)），返回
  `(4, 64, 64)` 的 RGBA 张量。

没有数据增强（`__getitem__` 里只有读取），因为系统要求规范几何：
训练、验证、推理使用完全相同的渲染坐标。**背景颜色随机化**
（`randomize_render_background`，概率 0.9，随机纯色）是唯一的数据
变化，因为它不移动角色像素，只是让模型不要依赖背景颜色。

## 5.2 自动生成稠密标签：utils.py::build_dense_parser_batch

训练时每个批次做如下事情（这是 05 章的核心，建议对照源码阅读）：

1. **渲染**：`renderer.forward_view(skins, view)` 得到
   `(B,4,H,W)` 渲染图。
2. **构建静态表面路由表**：`build_static_surface_routing(renderer,
   view)` 把映射文件里所有候选表面（直接内层、直接外层、复合层、
   几何回退层）整理成统一的 grids/masks/flat_uv/layer/part/face
   张量——即"屏幕每个像素可能属于哪些表面槽位"。
3. **采样 alpha**：`_sample_surface_alpha` 用每个表面的 UV 网格对
   图集 alpha 采样，得到"如果该像素属于表面 s，则该处纹素是否
   不透明"。
4. **决定每个像素的真实表面**：
   - 直接可见判定：`sampled_alpha[:, :2] > 0.5`（前两个表面是
     直接内层与直接外层）；外层可见 ⇒ surface=1，否则内层可见
     ⇒ surface=0；
   - **复合层**：若首可见复合层是装饰层（或索引 ≤ 1）则信任复合
     结果（`trust_composite`），因为帽子类装饰在两层合成里会
     错误地透出内层；
   - **几何回退层**：当前面装饰层完全透明但几何层有内容时
     （`geometry_fallback`），用几何层填充——例如"透明帽子边缘
     露出的后脑勺"这种情形。
5. **分类路由角色**：`classify_route_role(static, layer, flat_uv,
   valid)` 检查该像素选出的 `(层, 图集纹素)` 是否与**直接映射表**
   一致：
   - 一致 ⇒ 角色 = 该层本身（inner=0 或 outer=1）——"主路由"
     （primary）；
   - 不一致（例如像素属于复合表面，或属于背面的面）⇒ 角色 =
     `ROUTE_SECONDARY`（2）——**次级/背面表面**。
   
   这正是"直接可映射"与"次级"的精确含义：次级像素仍然有精确的
   surface 标签可以映射回图集，只是它不在直接内/外映射表里
   （例如透过透明外孔看到的更深立方体面或背面）。
6. **输出 targets**：

   ```python
   {
     "foreground": valid 前景 mask,          # (B,1,H,W)
     "layer":      直接层标签(0/1)或 IGNORE,  # (B,H,W)
     "route_role": 0/1/2 或 IGNORE,          # (B,H,W)
     "part":       部位标签或 IGNORE,         # (B,H,W)
     "face":       面标签或 IGNORE,           # (B,H,W)
     "surface":    精确表面槽位或 IGNORE,     # (B,H,W)
     "uv":         图集 UV01 坐标或 0,        # (B,2,H,W)
   }
   ```

   无效像素用 `IGNORE_INDEX = 255` 标记，损失计算时被屏蔽
   （`_deterministic_cross_entropy` 只统计有效像素）。

**关键理解**：网络看到的输入是"像素颜色"，标签是"这个像素由哪个
表面、哪个纹素产生"。前者是图像证据，后者是几何真值。网络学的是
从图像证据到几何槽位的映射——即**路由**。

## 5.3 训练时的多视角组合

- 主视角：`front_left, back_left`（2 个）；
- 特权视角：`front_right, back_right`（2 个，仅训练，不参与验证
  与推理），用于 `privileged_view_outer_distillation_loss`：把
  主视角学到的高置信外层预测"蒸馏"给特权视角，让模型对外层有
  多视角一致的表示；
- 语义特征按"完整视图组"组织，`MultiViewSemanticFusion` 要求
  样本数能被视图数整除，且视图顺序是规范的。

## 5.4 2D 稠密拓扑语义标签生成：semantic_targets.py

默认的 version 2 标签不再根据 UV 坐标猜测“眼镜、皇冠、衣服”等物体名称。那种 15 类伪标签会把矩形区域里的透明空洞也当成饰品，并把位置当作语义，容易反过来污染内外层路由。当前训练从真实 UV alpha 和固定渲染映射自动生成四类逐像素真值：

1. `head_top_accessory`：与头部顶面或四个侧面顶部两行物理连通的真实头部外层 texel；
2. `other_outer`：其余当前视角可见、alpha 有效的真实外层 texel；
3. `inner`：当前视角可见的内层表面；
4. `background`：角色轮廓之外的背景。

其中 `head_top_accessory` 不是一个 2D 包围框。系统在头部六个 8×8 外层面上建立物理邻接图，从顶面和侧面顶部开始，只沿真实 alpha texel 做连通分量传播。因此皇冠尖端、帽沿、头盔和顶部立体头发可以跨 UV 接缝获得完整监督，透明孔洞和不相连的眼睛、脸部像素不会被填入外层标签。

四类 logits 同时接受逐像素 focal loss；头顶部件额外使用 Dice、难正样本召回和图像级存在性损失。语义路由分支以零增益初始化，并用类别置信度控制注入强度：训练开始时完全保留既有 parser，只有经过真值校准的语义证据才逐渐影响 inner/outer logits。旧 15 类 schema 仅为读取历史 checkpoint 保留，不再是默认训练配置。

## 5.5 图集级语义软标签

除 2D 逐像素多任务标签外，还有**图集级/部位级**的软标签（从 GT 皮肤直接算出）：

- `build_semantic_attribute_targets`：每个部位的外层**存在性**（`outer_presence`，覆盖率 > 0）与**覆盖率**（`outer_coverage`，外层 alpha 加权面积 / 部位外层总面积）；每个部位的 inner/outer 平均颜色（`part_colors`，用于诊断与可能的颜色先验）。
- `build_head_outer_face_targets`：头部 6 个外层面各自的 **8×8 占用图**（alpha > 0.5），以及由此派生的结构标签：
  - `closed_ring_rows` / `closed_side_ring`：侧环是否闭合（侧面 4 个面每行 ≥4 个可见纹素）；
  - `open_top_rim`：侧环闭合但顶面只有外圈可见（"开顶帽"，修复时不允许闭合顶面）。

## 5.6 验证集与 checkpoint 选择

- 数据按 `--val_split`（默认 0.1）用**固定种子**的 `random_split`
  切分训练/验证；
- 验证集也是规范几何（无背景随机化），因此 `best.pt` 的选择与
  推理条件一致；
- checkpoint 选择指标默认 `loss_hard_uv_color_selection`：硬内/
  外层占用分数 + 硬内/外层 RGB MAE 的组合（见 07 章），防止
  "只稀疏、不准确"的图集赢得选择。

## 本章要点

1. 标签全部由渲染器 + 映射表自动生成（自监督式稠密标注），
   无人工标注。
2. 逐像素标签：foreground / layer / route_role / part / face /
   surface / uv，以及四类拓扑语义；无效像素用 IGNORE_INDEX 屏蔽。
3. route_role 的"次级"指：不属于直接内/外映射表的可见表面
   （复合层、几何回退层、背面），它仍有精确的 surface 槽位。
4. 训练用 4 个视角（2 主 + 2 特权），特权视角用于外层知识蒸馏。
5. 图集级软标签（外层存在/覆盖率、头部 8×8 占用与结构标志）
   支持语义辅助头。

## 思考题

1. 为什么 `trust_composite` 只在"首可见层是装饰层"时成立？
   提示：普通两层合成什么时候会给出错误颜色？
2. `classify_route_role` 中"一致"的判断是 `selected_flat_uv ==
   expected_uv`，为什么需要这个额外检查而不是只看 surface？
3. 如果某个外层纹素在正面可见、背面不可见，它的"证据"如何
   跨视角使用？（提示：结合 07 章 cross-view outer visibility
   与 08 章的跨视角否决。）
4. 背景颜色随机化算不算数据增强？它为什么是"安全"的？

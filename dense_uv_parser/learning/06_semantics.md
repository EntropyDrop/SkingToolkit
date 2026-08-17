# 06 冻结语义骨干（SigLIP2）

路由"内层 vs 外层 vs 次级"需要语义：帽檐、刘海、透明孔、分层衣服。
这些很难只用局部颜色判断。系统的做法是：把一张**冻结的视觉-语言
模型**（SigLIP2）当作特征提取器，提供两类特征——全局（整张图的
语义摘要）与空间（逐 patch 的语义特征图），再叠加**文本提示词
原型**来直接回答"这是不是帽子"之类的问题。

对应源码：`semantic_backbone.py`、`semantic_cache.py`、
`cache_semantic_features.py`、`semantic.py`。

## 6.1 SigLIP2 视觉塔封装：semantic_backbone.py

`SigLIP2VisionBackbone`（默认模型 `google/siglip2-base-patch16-224`）
做了四件事：

1. **加载**：用 Hugging Face `transformers` 的 `AutoModel` 加载，
   只取 `vision_model` 塔；`requires_grad_(False)` + `eval()`，
   **永不训练**（`train()` 被重写：投影适配器可以训练，视觉塔
   保持 eval）。
2. **可微 letterbox 预处理**：`_letterbox()` 按"短边对齐 224、
   保持宽高比、bilinear + antialias"缩放并居中填充。用
   **FixRes 模型**（固定分辨率）是有意为之：预处理完全可微，
   因此"预测的 UV → 重渲染 → SigLIP2 → 语义损失"这条梯度链可以
   一直回传到 UV 输出（用于 07 章的可微渲染分支）。
3. **特征提取**：
   - `encode_global()` → pooled 全局特征（768 维，可再投影到
     `semantic_channels=128`）；
   - `encode_dense()` → 逐 patch 空间特征图：224×224 输入、16×16
     patch ⇒ 14×14 token；再按 letterbox 的内容矩形**裁剪**
     （`_valid_token_indices`），得到 **768×14×8** 的特征图
     （对应 256×512 输入）。
   - 投影头 `token_projection` / `global_projection`（LayerNorm +
     Linear）是**可训练的**（只有它们是），把 768 维压到 128 维。
4. **文本提示词编码**：`encode_siglip2_text_prompts()` 用同一模型
   的文本塔编码固定提示词库 `DEFAULT_SIGLIP_ROUTE_PROMPTS`
   （`semantic_backbone.py`，描述"凸起的头发/帽子/兜帽/眼镜/面罩/
   耳机/动物耳朵/围巾高领/外套袖子/四肢装饰/平面头发/平面五官
   纹理"等），返回归一化嵌入与**SigLIP 校准标量**（logit scale /
   bias）。这些直接存进 parser checkpoint，推理时**不运行文本塔**。

> TIPSv2（`TIPSv2VisionBackbone`）是另一个可选骨干，仅在显式选择
> 时作为在线消融使用（`SEMANTIC_BACKBONE=tipsv2`），不是生产默认。

## 6.2 运行时挂载：semantic.py

`attach_semantic_runtime()` 把冻结骨干挂到 parser 模型上，但用
`object.__setattr__` 绕过 `nn.Module` 注册——**骨干不进 checkpoint
的 state_dict**，只按名字引用（`_runtime_semantic_backbone`）。
这样 checkpoint 保持小巧，且推理端可以自由选择骨干实现。

`DenseUVParserNet._runtime_semantic_features()` 在推理时若没有缓存
特征，则在线编码：先用前景 mask 把背景替换为**中性灰 0.5**
（与训练缓存一致，防止背景颜色变成路由证据），再在
`torch.no_grad()` 下编码。

## 6.3 语义特征缓存：cache_semantic_features.py + semantic_cache.py

**问题**：训练没有几何增强，冻结塔的输入每 epoch 完全相同，在线
重复编码浪费大量 GPU 时间。

**方案**：预计算一次，存成**内存映射（mmap）numpy 数组**，训练时
按需读取当前批次。缓存目录结构：

```text
cache/semantic_dense_parser_siglip2_spatial_256x512_180000_privileged/
├── metadata.json            # version、views、siglip_model、data_dir、filenames、维度
├── embeddings.npy           # (N, V, 768)  FP32  pooled 全局特征
└── spatial_embeddings.npy   # (N, V, 768, 14, 8)  FP16  空间特征
```

- 构建端（`cache_semantic_features.py::main`）：渲染每张皮肤的两
  个（或四个）视图 → letterbox → 编码 → 写入数组。只裁剪掉
  letterbox 的确定性填充列，**语义特征维度不压缩**（保持 768 全
  维度，因为下游适配器要学自己的子空间）。
- 规模：180,000 皮肤 × 4 视图 × (768×14×8×2 字节) ≈ **115 GiB**
  （含全局特征）。这就是必须用 mmap 的原因——`np.load(..., 
  mmap_mode="r")` 只把文件映射进虚拟内存，按索引取切片时才真正
  读盘，训练进程的常驻内存不会暴涨。
- 复用检查：`cache_is_reusable()` 校验 `metadata.json` 中的版本、
  视图列表、模型名、数据目录、以及文件名列表与数据集完全一致，
  任一不符就重建。`SIGLIP_CACHE_SPATIAL=false` 可退化为只有全局
  特征的旧缓存；`CACHE_SIGLIP_FEATURES=false` 则完全在线提取。
- 读取端（`semantic_cache.py::SigLIPGlobalCache`）：`get()` /
  `get_spatial()` 返回当前样本的张量；`semantic.py::cached_semantic_batch`
  把一批文件名对应的特征堆叠成 `{"raw_global": ..., "raw_spatial": ...}`，
  直接喂给 04 章的融合模块。

## 6.4 为什么"冻结 + 缓存"是合理的工程选择

1. **数据分布稳定**：输入是"规范几何 + 纯色背景"的渲染图，不是
   开放世界照片，冻结特征已经足够表达"帽子/刘海/透明孔"。
2. **训练目标明确**：语义特征只用来**条件化**路由选择，最终
   行为由稠密路由标签监督；微调塔反而有灾难性遗忘风险且成本高。
3. **可复现**：缓存是确定性的（同一输入同一输出），多轮训练
   结果可比。

代价是：推理必须能拿到与训练一致的编码（同样的 letterbox、
同样的灰背景替换），所以推理代码里 `_runtime_semantic_features`
会复用同一套预处理。

## 6.5 消融路径（为什么有这么多开关）

主 README 里反复出现的三个开关是理解语义贡献的实验工具：

| 环境变量 | 效果 |
| --- | --- |
| `SEMANTIC_BACKBONE=none CACHE_SIGLIP_FEATURES=false` | 纯几何解析器（无任何语义旁路） |
| `SIGLIP_TEXT_PROMPT_FUSION=false` | 关掉文本原型分支，保留视觉特征 |
| `SEMANTIC_BACKBONE=tipsv2` | 换一个冻结骨干做对照 |

## 本章要点

1. 骨干 = 冻结 SigLIP2 视觉塔（768 维全局 + 768×14×8 空间特征），
   仅两个投影头可训练。
2. 文本提示词库在启动时编码一次，嵌入与校准标量存进 checkpoint，
   推理零文本塔开销。
3. 特征缓存 = mmap numpy 数组（全局 FP32 / 空间 FP16），
   180k×4 视图 ≈ 115 GiB，按批读取。
4. 挂载用 `object.__setattr__` 绕过注册，骨干不进 state_dict。
5. 所有预处理（letterbox、灰背景替换）训练/推理必须一致。

## 思考题

1. 空间特征为什么是 768×14×8 而不是 768×14×14？提示：letterbox
   后裁剪。
2. 如果推理时背景是纯黑色而训练缓存用中性灰，会发生什么？
   推理代码如何避免？（提示：`_runtime_semantic_features` 的
   foreground 替换。）
3. mmap 缓存为什么能支持"180k 样本但内存占用小"？请解释
   虚拟内存与按页加载。
4. 文本提示词融合为什么不直接使用全局余弦分数，而要加一个
   "0.10·tanh 残差"？复习 04 章。

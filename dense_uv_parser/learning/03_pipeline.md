# 03 系统总览与代码地图

本章给出系统端到端的数据流全景图，并列出"文件 → 职责"的代码地图。

## 3.1 端到端数据流

### 训练阶段（`run_dense_uv_parser_training.sh` → `train.py`）：

```text
skins/ 目录下的 64×64 皮肤 PNG
   │  SkinUVDataset 读取、瘦模型归一化、背景随机化
   ▼
可微渲染器 DifferentiableRenderer（多视角正向渲染）
   │  生成 front_left, back_left 及特权视角 front_right, back_right
   ▼
精确稠密几何标签生成（build_dense_parser_batch）
   │  foreground / layer / route_role / part / face / surface / uv
   ▼
冻结 SigLIP2 视觉塔 + 文本提示词（mmap 内存映射快速读取）
   ▼
DenseUVParserNet 前向计算
   │  1. UNet 提取各视角 2D 卷积特征
   │  2. UVMultiViewSpatialFusion: 2D 特征反向投影至 64×64 3D UV 空间融合，再广播回各视角
   │  3. 输出稠密路由、表面槽位、置信度以及 64×64 全局外层占有率
   ▼
综合损失监督
   │  • 语义分类损失（Focal Loss / CrossEntropy）
   │  • 3D 外层占有率与拓扑损失（BCE + Dice + Occupancy Agreement）
   │  • 可微重渲染损失（Soft UV Splatting → Re-render → RGB/Alpha Loss）
   ▼
保存最优模型 checkpoint（best.pt）
```

### 推理阶段（`run_infer.sh` → `infer.py`）：

```text
输入 2D 视角图（front_back.png 或多张单视角图）
   │  前景分割（Flood Fill）与自适应背景融合
   ▼
DenseUVParserNet 推理
   │  输出各像素路由概率、置信度与 3D UV 外层先验
   ▼
路由与中心加权取色（splat_parser_predictions_to_uv_conditioning）
   │  基于 texel_center_score 加权众数投票，提取锐利高频原色
   ▼
极简内层确定性修复（simple_inpainting.py）
   │  • 内层：左右镜像补全 + 同部位 2D 近邻保底（Alpha=1.0）
   │  • 外层：未观测区域严格保持透明（Alpha=0.0，RGB=0）
   ▼
可微渲染假说检验与裁决（differentiable_hypothesis_refiner.py）
   │  • Analysis-by-Synthesis 假说对比
   │  • 下巴/面部防遮挡裁决（消除蓝色领口面罩伪影）
   │  • 部件级 3D 轮廓可微残差检验（确认真实皇冠/饰品）
   ▼
输出最终高保真 64×64 RGBA 皮肤（pred_uv.png）
```

## 3.2 代码地图

所有核心源码位于 `SkingToolkit/dense_uv_parser/`：

| 文件 | 规模 | 核心职责 |
| --- | --- | --- |
| `uv_layout.py` | ~170 行 | 64×64 皮肤图集矩形排布、部位 Mask、Alpha 阈值化处理 |
| `simple_inpainting.py` | ~150 行 | 极简内层盲区修复（左右镜像 + 同部位近邻填充，外层零脑补） |
| `differentiable_hypothesis_refiner.py` | ~200 行 | 可微渲染假说检验器（Analysis-by-Synthesis 消除内外层误判） |
| `SkingToolkit/renderer.py` | ~320 行 | 可微 3D Minecraft 皮肤渲染器（正向渲染与假说评估） |
| `model.py` | ~1200 行 | `DenseUVParserNet` 主干网络、`UVMultiViewSpatialFusion` 模块 |
| `skin_dataset.py` | ~75 行 | 皮肤数据集加载与格式标准化 |
| `semantic_targets.py` | ~115 行 | 图集级语义标签生成（外层存在率/覆盖率） |
| `semantic_backbone.py` / `semantic_cache.py` | ~680 行 | 冻结 SigLIP2 视觉-语言模型封装与 mmap 高速特征缓存 |
| `losses.py` | ~725 行 | 核心损失函数（Focal Loss、Route Swap Loss、Texel Consensus 等） |
| `train.py` | ~5500 行 | 训练循环、外层 3D 占有率监督、可微重渲染 Loss、验证指标 |
| `run_dense_uv_parser_training.sh` | ~700 行 | 训练启动脚本（环境变量配置、特权视角蒸馏、参数透传） |
| `infer.py` | ~2850 行 | 推理主流程（前处理、网络预测、加权取色、假说检验裁决） |
| `run_infer.sh` | ~610 行 | 推理启动脚本（自动选择最优 checkpoint、参数覆盖） |
| `inference_config.py` | ~115 行 | 生产级推理配置与默认阈值 |

## 3.3 核心设计原则

1. **输入一致性**：输入网络的内容统一在 RGB 空间处理，背景自适应泛化，避免模型死记单一纯色背景。
2. **3D UV 空间对齐**：视角不是割裂的 2D 平面，所有视角的特征通过 `UVMultiViewSpatialFusion` 在物理 3D UV 空间汇合。
3. **物理闭环裁决**：2D 空间无法确定的内外层歧义，通过 `differentiable_hypothesis_refiner` 在 3D 物理渲染空间直接以真实残差判定。
4. **外层绝不脑补**：外层只有真实观测证据才能存在，未观测部分永远保持透明，从根源消除脏色伪影。

## 本章要点

1. 系统 = 正向可微渲染器 + 3D UV 跨视角逆向解析器 + 极简内层修复 + 可微假说裁决。
2. 训练端支持 4 视角特权蒸馏与 3D UV 占有率直接监督；推理端支持 2 视角极速高保真逆向。

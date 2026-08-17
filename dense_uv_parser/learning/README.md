# Dense UV Parser 教程（面向计算机专业学生）

本目录是为计算机专业学生编写的 **Dense UV Parser** 系统学习教程。Dense UV
Parser 是 `SkingToolkit` 中唯一可训练的核心模块：它把两张或多张固定视角的
Minecraft 角色渲染图（如正面 + 背面）**逆渲染**回原始的 64×64 皮肤 UV 图集。
这个系统深度结合了计算机图形学（3D 可微渲染、UV 映射、Analysis-by-Synthesis 假说检验）、
深度学习（U-Net、3D UV 空间跨视角投影融合、视觉-语言模型 SigLIP2）、
以及确定性工程系统（内存映射缓存、极简内层修复、物理防遮挡裁决）。

## 读者对象与先修知识

- 熟悉 Python 与 PyTorch 基础（`nn.Module`、张量操作、`DataLoader`）。
- 了解 CNN 基本概念（卷积、下采样、U-Net 结构、交叉熵损失）。
- 不需要 Minecraft 游戏经验；教程会在需要时解释所有图形学概念。
- 建议具备基本 Linux 命令行与 shell 环境变量知识（本项目用环境变量控制所有超参数）。

## 目录结构

| 章节 | 内容 | 主要涉及源码 |
| :--- | :--- | :--- |
| [01 问题定义与背景](01_introduction.md) | 逆渲染问题、皮肤图集、内外层物理歧义与设计哲学 | — |
| [02 UV 图集与固定几何](02_uv_geometry.md) | 64×64 图集布局、内/外层、固定 Steve 几何、可微渲染器 | `uv_layout.py`, `simple_inpainting.py`, `renderer.py` |
| [03 系统总览与代码地图](03_pipeline.md) | 端到端数据流、3D UV 空间融合、模块职责地图 | 全部文件 |
| [04 网络架构](04_model.md) | U-Net 主干、UVMultiViewSpatialFusion 3D 融合、SigLIP2 语义旁路 | `model.py` |
| [05 数据与监督信号](05_supervision.md) | 数据集、渲染器自动生成稠密标签、route role 定义 | `skin_dataset.py`, `semantic_targets.py` |
| [06 冻结语义骨干](06_semantics.md) | SigLIP2 视觉塔、文本提示词库、FP16 内存映射缓存 | `semantic_backbone.py`, `semantic_cache.py` |
| [07 训练流程与损失设计](07_training.md) | 启动脚本、3D UV 外层占有率 Loss、可微重渲染损失 | `train.py`, `losses.py`, `run_dense_uv_parser_training.sh` |
| [08 推理管线与路由](08_inference.md) | 中心加权取色、极简内层修复、Analysis-by-Synthesis 假说裁决 | `infer.py`, `simple_inpainting.py`, `differentiable_hypothesis_refiner.py` |
| [09 练习与实验](09_exercises.md) | 阅读题、运行实验、消融实验、扩展项目 | 全部 |
| [10 术语表](10_glossary.md) | 中英对照术语速查表 | — |

## 建议学习路径

1. **快速跑通**：读 [03 系统总览](03_pipeline.md) 和 [08 推理管线](08_inference.md)，用 `run_infer.sh` 跑一次推理体验端到端逆渲染。
2. **正向理解**：按 01 → 02 → 03 顺序读，搞懂"皮肤图集 → 3D 长方体 → 2D 渲染图"的正向物理过程。
3. **逆向理解**：读 04 → 05 → 06，理解 3D UV 空间融合（`UVMultiViewSpatialFusion`）与语义先验。
4. **训练与裁决细节**：读 07 → 08，掌握可微重渲染 Loss 与 Analysis-by-Synthesis 假说检验的物理闭环。
5. **动手实践**：做 [09 练习与实验](09_exercises.md) 中的消融实验。

## 如何运行代码

仓库结构：

```text
SkingToolkit/
├── renderer.py                  # 可微 3D Minecraft 皮肤渲染器（正向过程）
├── dense_uv_parser/             # 逆向解析器
│   ├── train.py                 # 训练入口
│   ├── infer.py                 # 推理入口
│   ├── run_dense_uv_parser_training.sh
│   ├── run_infer.sh
│   └── ...
└── skins/                       # 训练数据集（64×64 皮肤 PNG）
```

训练：
```bash
./run_dense_uv_parser_training.sh
```

推理：
```bash
COMBINED=/path/to/front_back.png ./run_infer.sh
```

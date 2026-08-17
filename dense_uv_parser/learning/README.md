# Dense UV Parser 教程（面向计算机专业学生）

本目录是为计算机专业学生编写的 **Dense UV Parser** 系统学习教程。Dense UV
Parser 是 `SkingToolkit` 中唯一可训练的核心模块：它把两张固定视角的
Minecraft 角色渲染图（正面 + 背面）**逆渲染**回原始的 64×64 皮肤 UV 图集。
这个系统同时涉及计算机图形学（渲染、UV 映射、alpha 合成）、深度学习
（CNN、视觉-语言模型、损失设计）、以及工程系统（内存映射缓存、确定性修复、
可靠性门控），非常适合作为"把一门课的知识组合成一个真实系统"的案例。

## 读者对象与先修知识

- 熟悉 Python 与 PyTorch 基础（`nn.Module`、张量操作、`DataLoader`）。
- 了解 CNN 基本概念（卷积、下采样、U-Net 结构、交叉熵损失）。
- 不需要 Minecraft 游戏经验；但如果你玩过 Minecraft，会更容易理解皮肤
  图集布局。教程会在需要时解释所有图形学概念。
- 建议具备基本 Linux 命令行与 shell 环境变量知识（本项目用环境变量控制
  几乎所有超参数）。

## 目录结构

| 章节 | 内容 | 主要涉及源码 |
| --- | --- | --- |
| [01 问题定义与背景](01_introduction.md) | 逆渲染问题、皮肤图集、为什么这个问题很难 | — |
| [02 UV 图集与固定几何](02_uv_geometry.md) | 64×64 图集布局、内/外层、固定 Steve 几何、可微渲染器 | `uv_layout.py`, `uv_topology.py`, `renderer.py` |
| [03 系统总览与代码地图](03_pipeline.md) | 端到端数据流、模块职责、阅读代码的路线图 | 全部文件 |
| [04 网络架构](04_model.md) | U-Net 主干、视图条件、语义融合、各输出头、GNN 占用头 | `model.py` |
| [05 数据与监督信号](05_supervision.md) | 数据集、渲染器自动生成稠密标签、route role 定义 | `skin_dataset.py`, `utils.py`（批构建部分）, `semantic_targets.py` |
| [06 冻结语义骨干](06_semantics.md) | SigLIP2 视觉塔、文本提示词库、FP16 内存映射缓存 | `semantic_backbone.py`, `semantic_cache.py`, `cache_semantic_features.py`, `semantic.py` |
| [07 训练流程与损失设计](07_training.md) | 启动脚本、损失函数、指标、checkpoint 选择 | `train.py`, `losses.py`, `run_dense_uv_parser_training.sh` |
| [08 推理管线与路由](08_inference.md) | 前景提取、置信度门控、UV splat、确定性修复、输出文件 | `infer.py`, `foreground.py`, `simple_inpainting.py`, `inference_config.py` |
| [09 练习与实验](09_exercises.md) | 阅读题、运行实验、消融实验、扩展项目 | 全部 |
| [10 术语表](10_glossary.md) | 中英对照术语 | — |

## 建议学习路径

1. **快速跑通**（可选，约 15 分钟）：先读
   [03 系统总览](03_pipeline.md) 和 [08 推理管线](08_inference.md)，
   用 `run_infer.sh` 对 `test_imgs/` 里的样例跑一次推理，亲眼看到
   `pred_uv.png` 输出。这能让你对"输入是什么、输出是什么"建立直觉。
2. **正向理解**（第 1–3 天）：按 01 → 02 → 03 顺序读，弄清楚"皮肤
   图集 → 渲染图"这个**正向**过程。正向过程是后面所有监督信号的来源，
   必须最先理解。
3. **逆向理解**（第 2–4 天）：读 04 → 05 → 06，理解网络结构、标签如何
   自动生成、以及冻结的语义骨干如何提供先验。
4. **训练与推理细节**（第 4–6 天）：读 07 → 08，理解损失为什么这样设计、
   推理时那些数量众多的门控各自解决什么问题。
5. **动手**（第 6 天起）：做 [09 练习与实验](09_exercises.md) 中的练习。
   强烈建议至少做一次消融实验（例如关掉 SigLIP2 语义特征），
   亲身体会语义信息对内外层路由的贡献。

## 如何运行代码

仓库结构（从工作区根目录，即包含 `SkingToolkit/` 的目录）：

```text
SkingToolkit/
├── renderer.py                  # 可微 Minecraft 皮肤渲染器（正向渲染）
├── dense_uv_parser/             # 本教程的主角
│   ├── train.py                 # 训练入口
│   ├── infer.py                 # 推理入口
│   ├── run_dense_uv_parser_training.sh
│   ├── run_infer.sh
│   └── ...（见 03 章代码地图）
└── skins/                       # 训练数据集（64×64 皮肤 PNG）
```

环境要求（见 `SkingToolkit/README.md`）：

```bash
pip install torch torchvision tqdm pillow numpy
pip install -U transformers sentencepiece safetensors
```

训练（在 `SkingToolkit/dense_uv_parser/` 下）：

```bash
./run_dense_uv_parser_training.sh
```

推理：

```bash
COMBINED=/path/to/front_back.png ./run_infer.sh
```

> 注意：训练需要 GPU（默认 bf16 混合精度）和较大的显存；如果只有 CPU，
> 建议只做推理实验和"阅读代码"类练习。推理默认会自动选择
> `runs/dense_uv_parser_vN/best.pt` 中最高版本且兼容的 checkpoint。

## 阅读约定

- 教程中引用的文件路径均相对于 `SkingToolkit/dense_uv_parser/`，
  除非另行说明。
- 代码中的关键标识符（函数名、类名、环境变量名）保持英文原文，
  讲解用中文。
- 每章末尾有"本章要点"与"思考题"，答案分散在后续章节或练习章。

## 与主 README 的关系

`dense_uv_parser/README.md` 是一份面向维护者的、按特性组织的
**参考文档**（记录了每个开关与损失的历史缘由）；本教程则是按
**学习顺序**组织的**教学文档**。两者互补：遇到某个环境变量不明白，
可以先查本教程对应章节，再回主 README 看更完整的语义。

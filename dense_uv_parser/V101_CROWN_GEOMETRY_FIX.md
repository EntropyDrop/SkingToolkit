# v101 皇冠顶面修复（2026-09-06）

`img46_d55e72215c00` 的皇冠侧面和齿尖已分到外层，但最终外层头顶 UV 仍有 11 个不应存在的水平面格。本次移除了这些格，保留皇冠侧面、竖直齿尖和内层头发。版本继续为 v101。

## 原因与实现

旧流程把外层皇冠像素强制投到主表面。相机射线首先碰到的候选顶面可能实际上透明，真正可见的皇冠侧壁在它后面。将像素语义直接写到第一个 UV 命中，会制造水平横板。原来的“另一视图有高置信头发则否决”没有评估实际遮挡，只清除了部分多余格；即使输入人工合成的正确像素标签，这个问题也能复现。

新流程启用 `crown_top_geometry_mode: rendered_semantics`。在皇冠整体检测通过后，固定其他几何，逐格比较“保留／删除外层头顶格”的渲染结果，以两个视图的皇冠概率和前景轮廓误差决定删除。调用最终输出使用的同一个渲染器，包含次级表面与遮挡回退。每删一个格重新计算可见性，证据不充分或删除不能改善误差时保留。没有颜色阈值、针对 img46 的判断、固定皇冠轮廓或“顶面必须全空”的规则。

这次是推理几何修复，训练步数为 0。模型的 386 个张量与前一版完全相同；发布文件仅更新配套推理配置与来源记录。已有的帽带整件分层、皇冠整体外层、眼镜额头隔离、耳机颜色隔离及训练后的去背景模型继续使用。

## 验证

- 134 项单元／集成检查通过，包含新增的真实齿尖顶面保留、多余横板移除、颜色无关、渲染裁剪一致性及多人物隔离测试。
- 32 个合成皇冠：16 个有真实顶面、16 个无顶面。多余顶面格 **491 → 14**，减少 **97.15%**；真实顶面保留 **178/182** 前后不变，可见皇冠覆盖率 **99.4567%** 前后不变。所有非顶面 UV 在这组几何检查中完全相同。
- 本例最终顶面格 **11 → 0**；皇冠侧面和齿尖的 alpha 不变；内层金色皇冠格仍为 0。
- 重跑 14 张真实回归图：身体 UV、前景掩码和头顶以外的 alpha 全部不变。帽带外层强红格仍为 0，帽沿仍为四面各 8 格；眼镜额头格 alpha 为 0；耳机内层强绿格为 0。
- 其余 13 张真实图中，11 张 UV 完全相同，2 张各只有一个颜色通道相差 1/255。皇冠在几何修改后的材质拟合会调整少量头部颜色。

32 个合成样本使用曾用于前版测试的来源皮肤和新的固定程序种子，因此是回归证据，不作为全新独立泛化测试。仍有 14 个多余合成顶面格和原有的 4 个缺失有效顶面格；本次只删除有证据的多余顶面，不补造缺失侧面。

## 使用与结果

默认入口读取 `dense_uv_parser/v101_release.json`：

```bash
bash dense_uv_parser/batch_infer.sh --inputs /absolute/path/image.png
```

发布包：`dense_uv_parser/runs/v101_crown_geometry_release_20260906/`。

- `parser.pt` SHA-256: `a8aa3d8cd51cc6fec28d7c525aa1b012976d7cb1206e8b00c707de76bfb205dc`
- `pipeline.json` SHA-256: `3e14b98437eba9c5189a3344f560e7183f8daad44b9c40c55e15f7e0fda27685`
- `previous_release.json` 保留上一发布入口；前版权重与历史结果未覆盖。
- `final_checks.json`、`frozen_check.json` 保存验收与模型张量一致性记录。

重跑的 14 张图：`dense_uv_parser/output_history/v101_crown_geometry_20260906/`。

本例：`img46_9b5a1c089bde/parser_pred_uv_simple_inpainting.png`，同目录有 `simple_inpaint_render.png`、`crown_geometry.json`、`crown_removed_top_uv.png`。其中日志记录的是从原始投影删除的 40 格；上一发布已删除其中 29 格，所以相对用户指出的上一结果，本次额外去掉 11 格。

复现：`python dense_uv_parser/run_local.py test_crown_geometry`；完整 32 例几何检查入口为 `dense_uv_parser/regression/check_crown_geometry.py`。使用项目的 `sking-v61-worker` 环境和已有来源分割、检查点。

实现提交：`28f548a`。备份分支：`backup/v101_before_crown_geometry_20260906`。

默认入口实测：`output_history/v101_crown_geometry_default_smoke_20260906/img46_1f4bee6b7302` 的 UV 与审查结果逐字节相同；新发布包的皇冠、帽带两例同样逐字节一致。

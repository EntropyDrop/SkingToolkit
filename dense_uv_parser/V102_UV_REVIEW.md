# v102：胡子 UV 对称复查与参考修正

2026-09-06。用户再次否定上一轮胡子与刘海结果。自动模型没有通过这次复查；本次没有重新训练，也没有更换 checkpoint。

输入为 `SKING_DDJ_v61/1ZW63H5Z8YNV57MT_edited.png`，头发参考为用户指定的同目录 `1ZW63H5Z8YNV57MT_result.png`。完整路径、校验值、掩码、量化结果及备份位置见 `regression/v102_uv_symmetry_review.json`。

## UV 证据

上一轮外层正面胡子的第 7 行左端缺一格，第 8 行右端缺一格；侧脸底边也没有正确配对。UV 的侧脸不能用整张 PNG 水平翻转来比较，必须使用立方体表面的镜像对应关系，例如外层 `(39,15)` 对应 `(48,15)`，`(38,15)` 对应 `(49,15)`。

在这张图明确标注的胡子检查范围内：

| 检查 | 原自动结果 | 参考修正结果 |
| --- | ---: | ---: |
| 内层透明度镜像不一致格数 | 0 | 0 |
| 内层可见 RGB 镜像不一致格数 | 24 | 0 |
| 外层透明度镜像不一致格数 | 8 | 0 |
| 外层可见 RGB 镜像不一致格数 | 20 | 0 |
| 所选刘海范围相对旧版的透明度差异格数 | 14 | 0 |

统计数为不一致的格子数量，镜像对的两个位置分别计数。允许胡子内外层并存，所选区域使用对应内层基底与匹配纹理；嘴部和脸颊的皮肤缺口保留。刘海、下方发际线及所选头顶区域逐字节恢复为旧参考图。未标注区域及身体 UV 逐字节不变。

## 修正范围与使用

`uv_reference_repair.py` 是显式参考和掩码驱动的标注修正工具，不在自动推理链路中。它不会按输入文件名启用规则，也不会把所有人物胡子强制对称。颜色来源、胡子支持区域和头发参考范围均由本次标注明确指定；这不是对模型语义泛化能力的证明。

最终结果写回数据目录的 `1ZW63H5Z8YNV57MT_result_v102.png`，并附 `1ZW63H5Z8YNV57MT_result_v102.uv_review.json`。原始自动结果保存在 `runs/v102_uv_symmetry_20260906/before.png`。参考原图和 v101 结果没有改动。

```bash
PYTHON=/home/ds/miniconda3/envs/sking-v61-worker/bin/python
OMP_NUM_THREADS=4 "$PYTHON" dense_uv_parser/run_local.py uv_reference_repair \
  --spec /home/ds/llms/SkingToolkitDev/dense_uv_parser/runs/v102_uv_symmetry_20260906/repair_spec.json \
  --audit-only
# 去掉 --audit-only 并提供尚不存在的 --output 路径即可复现修正。
OMP_NUM_THREADS=4 "$PYTHON" dense_uv_parser/run_local.py test_uv_reference_repair
```

输出记录位于 `output_history/v102_uv_reference_20260906/1ZW63H5Z8YNV57MT_edited/`，包含 UV、渲染、四个头部视角、输入清单和检查报告。数据目录的修正结果可能被将来的自动批量推理覆盖；发布记录中的已知失败仍然有效，不能把它标记为模型自动通过。

## 验证与后续自动模型要求

4 项单元测试检查侧脸镜像映射、混合层对齐、修改范围，以及无内层基底时拒绝添加外层胡子。额外用原自动 UV 和修正 UV 验证回归检查器能分别报失败和通过。

`train_v102.real_review` 现在对有显式审查标注的输入记录最终 UV 胡子对称性与刘海透明度差异。输入 SHA 仅用于选择开发集标注，检查结果不修改推理，不参与梯度，不构成独立测试集。当前 checkpoint 在本例仍然失败，已在 `v102_release.json` 标记。

下一次模型候选必须先在自动生成的 UV 上通过这一检查，再检查多视图轮廓和原有皇冠、鼻子、耳机、帽沿回归。不能用这张经过参考修正的输出替代自动模型结果，也不能把开发样例上的通过推广到全部新图片。

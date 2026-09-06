# v103 训练恢复与结果

2026-09-07。6000 步参数更新已完成，最终状态为 `complete_with_validation_warnings`。训练状态和断点恢复流程已修复；底座颜色细化的数值漂移仍有记录，候选模型也未通过效果验收，未发布、未覆盖现有皮肤。

## 中断原因与修复

原运行在第 200 步完整推理检查时中断：UV 坐标 `(30,12)` 的绿色通道在缓存评估和重新推理之间相差 1/255，alpha 一致，损失/梯度均正常。复现显示语义特征和进入 `refine_head_material` 前的 UV 完全一致，48 步颜色细化后产生小幅浮点漂移，身体没有差异。

- 模型、优化器、步数/学习率、Python/NumPy/CPU/CUDA 随机状态在验证前写入同一个原子恢复文件。验证不再改变后续训练的随机序列。
- 新增 `--resume`，检查底座、缓存、batch 和总步数匹配；已在真实 GPU 训练中验证第 200 步暂停/续训，并从完整第 6000 步状态恢复最终检查。
- 序列化、相同输入的完整推理、alpha、身体、语义证据保持严格检查。RGB 验收容差为浮点不超过 1/255 且 PNG 通道差值不超过 1。
- 第 6000 步曾出现 0.003980 的 RGB 漂移，超出浮点容差，报告保留 `passed: false`。仅此类严格检查全通过、底座输入颜色发生变化的情况可记为警告并保存训练结果；不能把它当作验收通过。恢复后也保留历史警告，不通过重复运行消除失败记录。

原第 200 步没有优化器状态，因此复用完成的数据缓存重新计算了前 200 步；此后是包含优化器和随机状态的真实断点恢复。旧失败目录和日志完整保留。

代码在 `v103` 分支，修复提交 `3302e20`、`7d774304df9da4b5dab5c62c4acf2d6d987cf539`。10 项当前模块测试和 14 项 v102 回归测试通过。核对确认最终文件保留全部第 6000 步训练权重、优化器矩估计和原 v102 底座张量。

## 效果检查

32 个与训练源身份隔离的合成验证样本：

| 指标 | v102 底座 | v103 第 6000 步 |
|---|---:|---:|
| 外层 IoU | 74.63% | 85.89% |
| 外层误增格子 | 293 | 195 |
| 外层漏检格子 | 496 | 230 |
| 可见颜色 MAE（0–1） | 0.06852 | 0.04994 |

真实样例尚未通过验收：

- 胡子训练样例的外层轮廓镜像差异从 8 格降到 0 格；选定头发区域与参考的 alpha 差异从 14 格降到 0 格。
- 胡子颜色仍有内层 22 格、外层 20 格镜像差异，最大通道差值分别为 46、21。轮廓改善不代表颜色对称性已解决。
- 蓝色皇冠顶面延伸从 12 格降到 11 格；金色皇冠头顶新增 6 格。
- 帽沿四面原为 `[8,8,8,8]` 格，变成 `[7,8,6,0]`，存在明显回退。
- 17 个样例的身体 UV 与各自缓存底座完全一致。鼻子、眼镜额头、耳机内层强绿色泄漏、帽带分层的已有检查通过。

这 17 个结果基于固定缓存生成；胡子另外通过完整加载推理检查。胡子身份已进入训练，只能说明拟合情况；其余 16 个是反复使用的开发样例，不能当作全新独立测试集。没有任何检查点满足全部现有质量检查。

[第 6000 步胡子 UV](runs/v103_final_uv_retrain_20260907/training_completed/real_6000/beard/uv.png) · [正背面渲染](runs/v103_final_uv_retrain_20260907/training_completed/real_6000/beard/render.png) · [完整检查数据](runs/v103_final_uv_retrain_20260907/training_completed/recovered_review.json)

## 服务器产物

服务器 `ds@192.168.0.111`，运行根目录：

`/home/ds/llms/SkingToolkitDev/dense_uv_parser/runs/v103_final_uv_retrain_20260907/`

- 当前产物入口：`current_training.json`。
- 最终目录：`training_completed/`。
- 最终模型：`training_completed/step_6000.pt`。
- 完整恢复状态：`training_completed/training_state_latest.pt`。
- 当前状态：`training_completed/status.json`。
- 历史稳定性警告：`training_completed/validation_warnings.json`。
- 权重和优化器核对：`training_completed/completion_verification.json`。
- 全部质量检查：`training_completed/recovered_review.json`。
- 最终评估日志：`training_completed.log`；实际第 200–6000 步训练日志保留在 `training_recovered.log`。

最终模型 SHA-256：`6919f171dc192fa347a590fb5e016878577cf49a52ee8d7736d879502332e466`。

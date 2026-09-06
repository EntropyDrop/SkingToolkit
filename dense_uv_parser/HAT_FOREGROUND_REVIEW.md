# 帽沿与背景改进，2026-09-06

当前 v101 正式训练已完成 20,000 步，原 best.pt 来自第 11,000 步。本轮更改在 `fix/v101-hat-foreground` 分支，原 v101 分支和权重保留。

## 已定位的问题

用户指定 `PU418PLGERMFC487_edited_1c82c0cdd702` 的权重 SHA-256 为 `91651ac9fdc971fefc571a207d2951f588d492b96e9121e28b3704b1b26acc23`。原图帽沿中，两个视角的帽子类别概率均值约 1e-9，语义类别命中率为零；前景覆盖为 100%，所以帽沿缺失不能归咎于背景掩码。最终外层保留率为 72.7% / 24.0%。这些数字基于助手目视描绘的开发样例帽沿，去掉 1 像素边界，不能视作独立测试准确率。

原帽子样本的四面深度及色带分别随机生成，无法提供环绕一致的监督；帽冠基本都和帽沿同样向外凸出，也缺少用户样例中较窄帽冠和外凸帽沿的台阶。原始皮肤存储在内层的帽子/头发还可能被作为语义负样本。UV 层位置是几何存储事实，不能直接等同于物体语义。

## 本轮修改和验证

- 改为四面共享帽冠深度、帽沿高度和色带材料；补充较窄帽冠、完整外层帽沿的样式；增加深色帽子比例。
- UV 结构损失单独计算并加强跨面接缝项。短训设置 `--replay-weight 0`，不再将未知原始内层头部直接作为语义负样本。旧回放选项仍保留兼容性；`clean_added_outer_false_positive_rate` 这个旧指标实际只是与存储层的分歧代理值，并非完整的语义误报标注。
- 仅修正环绕样式的 800 步短训没有改善真实帽沿；补充帽冠/帽沿的台阶结构后，1,200 步短训将帽沿保留率提高到 **90.7% / 98.2%**。眼镜开发回归的整副外层保留率 93.2%、语义 precision 91.4%，通过原有检查点准入。
- **候选还不能视作最终成品**：目视检查发现帽沿连续性改善，但部分红色带被黑色外层遮挡或改色。当前像素级分层指标不能保证 UV 材质保真，因此没有用候选覆盖正式 v101。
- 14 张候选图片完成全流程推理和头部目视检查。测试共 118 项通过：配饰 7、前景 8、几何路由 89、语义 14。

候选权重：`/home/ds/llms/SkingToolkitDev/dense_uv_parser/runs/v101_hat_brim_pilot_20260906/best.pt`

候选结果：`/home/ds/llms/SkingToolkitDev/dense_uv_parser/output_history/v101_hat_refinement/`

下一步帽子训练需要将整个帽子的身份、帽冠/帽沿等部件、外层占用和材质颜色分别监督，并在联合 UV 图中检查四面接缝和环绕完整性。验收还应包括红色带的颜色还原，而不能只看“更多像素被分到外层”。不要用“看到帽子就强制补满一圈”的规则替代训练。

## 背景：先接入预训练模型与可靠取色，再决定专项微调

当前默认是左上角颜色容差的四连通 flood。它没有软 alpha 或前景色估计，渐变背景、封闭背景孔洞以及抗锯齿混色会漏入 UV。新分支保留默认兼容行为，新增显式的外部分割概率接口。

已用服务器缓存的 BiRefNet，在帽子、眼镜和 img28 三张真实图上对比。直接替换掩码后仍有紫色边缘，而且会改变原几何/轮廓判断；例如帽沿保留率变为 61.1% / 16.5%，说明不能未经联合验收就替换默认掩码。

新增可靠取色通道将两件事分开：概率 ≥0.5 仍作为前景形状；只有概率 ≥0.98 且位于前景内部 1 像素的来源参与 UV 取色。它不按 RGB 色相删除真实配饰，也不把分割置信度当作真实 alpha 来反算颜色。测试验证关闭取色来源不会改变路由前景范围。

在 img28 上，最终渲染中背景样紫色像素诊断计数从 flood 的 10,398，降为仅替换掩码的 6,059，再降为可靠取色的 0。诊断定义为 R>G+0.04 且 B>G+0.08，只适用于这张角色本身没有紫色材质的图，不代表所有图片均无背景污染。

建议先使用现成 BiRefNet 模块，加可靠取色和前景色/alpha 的边缘处理；若在更多 Minecraft 画风中仍漏细结构，再基于现成权重专项微调，不必从零训练。渲染器可以提供精确前景、alpha 和干净 RGB；训练时合成渐变、纹理背景及缩放/JPEG/抗锯齿污染，并补充人工确认的真实边缘样本。使用与训练身份隔离的验证集，联合检查形状、颜色污染、帽沿和眼镜。

BiRefNet 是前景分割模型，默认输出的 sigmoid 置信度不等于物理 alpha。细边混色的完整处理还需要 matting/foreground-color estimation；不能只保存一张透明 PNG 就认为颜色污染已经被消除。

依据：[BiRefNet 官方实现](https://github.com/ZhengPeng7/BiRefNet)、[PyMatting 前景色估计文档](https://pymatting.github.io/foreground.html)、[前景色与 alpha 联合估计论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hou_Context-Aware_Image_Matting_for_Simultaneous_Foreground_and_Alpha_Estimation_ICCV_2019_paper.pdf)。

## 可复现入口

在服务器仓库根目录执行：

```bash
# 现有 comfy 环境有 BiRefNet 所需依赖；不改动 v61 worker 环境。
/home/ds/miniconda3/envs/comfy/bin/python dense_uv_parser/foreground_provider.py \
  --model-dir /home/ds/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet/snapshots/6a62b7dcfa18a3829087877fb16c8006831e4220 \
  --inputs /absolute/path/combined.png --output-dir dense_uv_parser/runs/foreground

# 把上一步返回的 probability.png 显式传入；输入图片和概率图须一一对应。
bash dense_uv_parser/batch_infer.sh --inputs /absolute/path/combined.png \
  --foreground-probabilities /absolute/path/probability.png
```

分割模块只加载本地、已审阅的模型快照；输出记录模型权重、代码和输入的校验值。解析器按概率图内容、阈值及取色设置生成新指纹，保存概率、前景、可靠取色来源供检查。未传概率图时保留原 flood 流程。

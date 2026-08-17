# 10 术语表

中英对照速查。按主题分组；括号内给出首次出现的教程章节或源码位置。

## 图形学与数据

| 术语 | 英文 | 说明 |
| :--- | :--- | :--- |
| 皮肤图集 | skin atlas / UV map | 64×64 RGBA 皮肤纹理图，角色所有表面的贴图拼在一起（01/02） |
| 纹素 | texel | 图集上的一个像素单元（01） |
| UV 映射 | UV mapping | 表面坐标 ↔ 纹理坐标的对应关系（01） |
| 部位 | part | 角色的 6 个长方体：头、躯干、双臂、双腿（02） |
| 面 | face | 一个部位展开的 6 个矩形面（02） |
| 内层 / 外层 | inner layer / outer layer (decor) | 皮肤本体层与装饰层（帽子/外套），外层 alpha 覆盖内层（02） |
| 展开偏移 | decor offset | 外层矩形相对内层矩形的固定位移（02） |
| 合成 | alpha compositing | 按 alpha 把外层叠在内层、内层叠在背景上（02） |
| 复合表面 / 几何回退表面 | composite / geometry fallback surface | 映射文件中额外的表面槽位，用于正确处理帽子等装饰（02/05） |
| 视角 / 视图 | view | 命名相机，如 `front_left`、`back_left`（02） |
| 映射文件 | mapping file | 每视角一个 `*_mapping.pt`，记录屏幕像素 → UV 的预计算表（02） |
| 正/逆渲染 | forward / inverse rendering | 纹理→图像 与 图像→纹理（01） |
| 规范坐标 | canonical coordinates | 训练/推理统一使用的标准几何坐标（03） |

## 模型与学习

| 术语 | 英文 | 说明 |
| :--- | :--- | :--- |
| 3D UV 跨视角融合 | UVMultiViewSpatialFusion | 将 2D 特征反向投影到 64×64 UV 空间进行 360° 全局聚合并广播回各视角（04） |
| 假说检验 | Analysis-by-Synthesis | 生成多套 UV 假说并用可微渲染器重渲染对比真实残差进行物理裁决（01/08） |
| 可微假说裁决器 | Differentiable Hypothesis Refiner | 执行 Analysis-by-Synthesis 消除内外层误判与遮挡面罩的推理模块（08） |
| 路由角色 | route role | 像素属于内层(0)/外层(1)/次级表面(2) 的三分类（01/04） |
| 次级表面 | secondary surface | 不属于直接内/外映射的可见表面（透过透明孔看到的深处/背面）（05） |
| 表面槽位 | surface slot | 映射文件中的候选表面编号，训练/推理用它精确定位纹素（05） |
| 冻结骨干 | frozen backbone | 不训练的特征提取器（SigLIP2 视觉塔）（06） |
| 文本原型 | text prompt prototype | 固定提示词的嵌入，与图像特征做相似度（04/06） |
| FiLM | Feature-wise Linear Modulation | 用全局特征生成 scale/shift 调制特征图（04） |
| 聚焦损失 | focal loss | 用 gamma 加权难例的损失（07） |
| 3D 外层占有率 | Outer UV Occupancy | 64×64 3D UV 图集上的全局外层存在性预测（04/07） |

## 工程与流程

| 术语 | 英文 | 说明 |
| :--- | :--- | :--- |
| 中心加权取色 | center-weighted mode splat | 按距离 UV 纹素中心的距离得分加权众数投票，提取锐利原色（08） |
| 极简内层修复 | pure inner-only inpainting | 仅补全内层死角盲区（左右镜像 + 同部位近邻），外层未观测严格透明（08） |
| 内存映射 | memory-mapped file (mmap) | 不整体载入内存、按页读取的大数组缓存（06） |
| checkpoint | checkpoint | 保存模型权重与配置的 `.pt` 文件（07） |
| 原子写入 | atomic write | 先写临时文件再 rename，避免读到半成品（07） |
| 洪水填充 | flood fill | 从种子像素连通扩散的前景提取（08） |
| 自适应背景 | adaptive background | 从候选色中选"离前景最远"的纯色替换背景（08） |
| 特权视角 | privileged view | 仅训练用的额外视角，用于跨视角蒸馏（05/07） |
| 可微重渲染分支 | render loss cycle | 把预测软映射为临时皮肤并重渲染对比输入图像的训练损失（07） |

## 常见缩写

| 缩写 | 全称 | 说明 |
| :--- | :--- | :--- |
| RGBA | Red Green Blue Alpha | 颜色 + 透明度通道 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| VLM | Vision-Language Model | 视觉-语言模型（SigLIP2） |
| FiLM | Feature-wise Linear Modulation | 特征级线性调制 |
| IoU | Intersection over Union | 交并比（占用质量指标） |
| MAE | Mean Absolute Error | 平均绝对误差（颜色指标） |
| BCE | Binary Cross Entropy | 二分类交叉熵 |
| mmap | memory map | 内存映射文件 |
| bf16 / fp16 | bfloat16 / float16 | 混合精度训练的低精度格式 |
| LR | learning rate | 学习率 |
| GT | ground truth | 真值标签 |

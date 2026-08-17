# 10 术语表

中英对照速查。按主题分组；括号内给出首次出现的教程章节或源码
位置，方便回查。

## 图形学与数据

| 术语 | 英文 | 说明 |
| --- | --- | --- |
| 皮肤图集 | skin atlas / UV map | 64×64 RGBA 皮肤纹理图，角色所有表面的贴图拼在一起（01/02） |
| 纹素 | texel | 图集上的一个像素单元（01） |
| UV 映射 | UV mapping | 表面坐标 ↔ 纹理坐标的对应关系（01） |
| 部位 | part | 角色的 6 个长方体：头、躯干、双臂、双腿（02） |
| 面 | face | 一个部位展开的 6 个矩形面（02） |
| 内层 / 外层 | inner layer / outer layer (decor) | 皮肤本体层与装饰层（帽子/外套），外层 alpha 覆盖内层（02） |
| 展开偏移 | decor offset | 外层矩形相对内层矩形的固定位移（02） |
| 合成 | alpha compositing | 按 alpha 把外层叠在内层、内层叠在背景上；图集 alpha 是二值的，但双线性采样会在层边界产生分数 alpha（02） |
| 分数 alpha | fractional alpha | 双线性采样在二值 alpha 边界插值出的 0~1 中间值，产生内外层颜色混合的抗锯齿边缘像素（01/02） |
| 复合表面 / 几何回退表面 | composite / geometry fallback surface | 映射文件中额外的表面槽位，用于正确处理帽子等装饰（02/05） |
| 视角 / 视图 | view | 命名相机，如 `front_left`、`walk_front_both_layer_ortho`（02） |
| 映射文件 | mapping file | 每视角一个 `*_mapping.pt`，记录屏幕像素 → UV 的预计算表（02） |
| 正/逆渲染 | forward / inverse rendering | 纹理→图像 与 图像→纹理（01） |
| 病态问题 | ill-posed problem | 解不唯一/信息不足的问题（01） |
| 背面 | backface | 相机看不到的面；其纹素只能靠修复或镜像（01/05） |
| 规范坐标 | canonical coordinates | 训练/推理统一使用的标准几何坐标（无平移缩放）（03） |

## 模型与学习

| 术语 | 英文 | 说明 |
| --- | --- | --- |
| 路由角色 | route role | 像素属于内层(0)/外层(1)/次级表面(2) 的三分类（01/04） |
| 次级表面 | secondary surface | 不属于直接内/外映射的可见表面（透过透明孔看到的深处/背面）（05） |
| 表面槽位 | surface slot | 映射文件中的候选表面编号，训练/推理用它精确定位纹素（05） |
| 条件张量 | conditioning tensor | 10/12 通道分层 UV 证据张量 `[inner RGBA+evidence, outer RGBA+evidence(+confidence)]`（01/08） |
| splat | splat | 把像素颜色/证据"投"回图集纹素的过程（08） |
| 证据 | evidence | 图集纹素是否有可信路由像素的标志（08） |
| 路由先验 | route role spatial prior | 可学习的"常见结构位置"统计偏置（04） |
| 冻结骨干 | frozen backbone | 不训练的特征提取器（SigLIP2 视觉塔）（06） |
| 文本原型 | text prompt prototype | 固定提示词的嵌入，与图像特征做相似度（04/06） |
| FiLM | Feature-wise Linear Modulation | 用全局特征生成 scale/shift 调制特征图（04） |
| 消融 | ablation | 关闭某组件观察其贡献的实验（09） |
| 硬负样本 | hard negative | 高置信但错误的负例，训练中加重惩罚（07） |
| 聚焦损失 | focal loss | 用 gamma 加权难例的损失（07） |
| 置信度校准 | confidence calibration | 让预测概率反映真实正确率（07） |
| 宏平衡 | macro balance | 各类别等权，避免像素数不平衡主导（07） |
| 图神经网络 | GNN | 外层占用头在外层纹素邻接图上的消息传递（04） |

## 工程与流程

| 术语 | 英文 | 说明 |
| --- | --- | --- |
| 内存映射 | memory-mapped file (mmap) | 不整体载入内存、按页读取的大数组（06） |
| checkpoint | checkpoint | 保存模型权重与配置的 `.pt` 文件（07） |
| 原子写入 | atomic write | 先写临时文件再 rename，避免读到半成品（07） |
| 洪水填充 | flood fill | 从种子像素连通扩散的前景提取（08） |
| 自适应背景 | adaptive background | 从候选色中选"离前景最远"的纯色替换背景（08） |
| 置信度门控 | confidence gate | 置信度+边距阈值过滤路由决策（08） |
| 纹素共识 | texel consensus | 本地概率与纹素级证据的加权融合（08） |
| 跨视角否决 | cross-view veto | 多视角强证据冲突时否决外层（08） |
| 救援路径 | rescue path | 几何/语义证据支持下放宽门控（08） |
| 众数取色 | grid mode / majority vote | 取出现最多的 8-bit RGB，不平均（08） |
| 确定性修复 | deterministic inpainting | 镜像优先、3D 最近邻的规则补全（08） |
| 特权视角 | privileged view | 仅训练用的额外视角，用于蒸馏（05/07） |
| 蒸馏 | distillation | 用主视角高置信预测训练特权视角（07） |
| 硬指标 | hard metric | 按最终推理决策（非软概率）统计的指标（07） |
| 渲染循环 | render cycle | 把预测渲染回去再算损失的训练分支（07） |

## 常见缩写

| 缩写 | 全称 | 说明 |
| --- | --- | --- |
| RGBA | Red Green Blue Alpha | 颜色 + 透明度通道 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| VLM | Vision-Language Model | 视觉-语言模型（SigLIP2） |
| FiLM | Feature-wise Linear Modulation | 特征级线性调制 |
| GNN | Graph Neural Network | 图神经网络 |
| GAN / VAE | — | 生成式模型（本系统刻意不用） |
| IoU | Intersection over Union | 交并比（占用质量指标） |
| MAE | Mean Absolute Error | 平均绝对误差（颜色指标） |
| BCE | Binary Cross Entropy | 二分类交叉熵 |
| TV | Total Variation | 全变差（平滑正则） |
| mmap | memory map | 内存映射文件 |
| bf16 / fp16 | bfloat16 / float16 | 混合精度训练的低精度格式 |
| LR | learning rate | 学习率 |
| GT | ground truth | 真值标签 |
| IGNORE_INDEX | — | 无效像素标签哨兵值 255（05） |

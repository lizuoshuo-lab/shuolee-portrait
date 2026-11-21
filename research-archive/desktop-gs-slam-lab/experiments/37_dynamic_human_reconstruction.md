---
id: "gslab-e37"
slug: "dynamic-human-reconstruction-benchmark"
title: "动态人物重建基准与评测约定"
order: 37
status: "temporal_validation_complete_appearance_refinement_negative"
updated: "2026-10-15"
sources:
  - "https://github.com/JonathonLuiten/Dynamic3DGaussians/tree/7dbbd4dec404308524ff402756bdb8143a2589b0"
  - "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip"
  - "https://github.com/ziyc/drivestudio/tree/e59bda4fa681f829dbb1d65f0de582b0f633c450"
---

# 动态人物重建基准与评测约定

主线目标为大场景内移动人物的清晰重建。首先在公开同步多视角人物运动数据上建立静态与动态对照，再进入街景。上一阶段的物理仿真与控制结果保留，不作为本任务的重建证据。

理论入口：[大场景中的动态人物重建](../knowledge/32_dynamic_human_scene_reconstruction.md)。

## 数据与版本

采用 Dynamic 3D Gaussians 作者清理发布的 Panoptic 篮球序列。已从官方压缩包按字节范围提取并校验 8,703 个文件；每个压缩成员通过 CRC 校验，并保存解压后的 SHA-256。只提取本次序列，不把尚未下载的整个压缩包标记为已完成。

原发布序列共有 150 个时刻，分辨率为 640×360；27 个训练相机和 4 个测试相机。第一阶段固定为帧 0～30，测试相机为 0、10、15、30。初始化点云有 194,105 个点，其中 135,141 个标为前景。相机采用作者提供的 OpenCV 坐标约定和内外参。

动态代码固定为 `7dbbd4dec404308524ff402756bdb8143a2589b0`，大场景候选 DriveStudio 固定为 `e59bda4fa681f829dbb1d65f0de582b0f633c450`。原仓库文件保持不变，适配代码单独保存。

## 本地适配与训练条件

复用 RTX 5080 已可用的 Torch 与 gsplat 环境。以 gsplat 的经典光栅化模式替代原项目 CUDA 渲染器，以 SciPy 最近邻替代 Open3D；保留作者的图像、分割、局部刚性、旋转、等距、地面、背景与颜色连续性损失。屏幕位置梯度由像素坐标转换到原密度控制所用的归一化坐标量纲。本结果需称为本地兼容适配，不能称为原环境逐位复现。

初始相机的 20 步可微检查通过：损失从 1.5452 降至 1.0997，位置与屏幕梯度有限且非零。这仅表明渲染与优化能工作，不是人物重建质量评测。

动态组首帧优化 10,000 次，其后 30 帧各 2,000 次；静态组使用相同 31 帧随机采样，共 70,000 次更新。两组均从同一作者点云初始化、使用相同 27 个训练相机和种子 73。静态组没有时间条件，也没有跨时刻动态正则；动态组遵循作者逐时刻优化方式。两者匹配总更新次数，不代表参数自由度、计算量或存储量完全相同。

保存首帧中间模型、逐时刻模型、优化器、随机数状态及恢复检查点。每组设置 120 万高斯的预先资源上限，超过时停止并保留失败信息，不悄悄改变方法后合并成绩。

## 冻结的评测

固定 4 个测试相机、全部 31 个时刻，共每种方法 124 个视图。测试图像不参与优化，也不为测试相机额外拟合曝光补偿。人物矩形区域在查看模型测试渲染之前，依据原始首尾帧固定。

报告全图与人物区域的 PSNR、SSIM、相对原图的梯度误差，以及人物区域的时间图像变化误差。后者比较相邻帧差分，不是经过光流对齐的闪烁指标，也不是三维轨迹真值。人物区域有部分背景，不能称为人体像素掩码。

展示固定相机 10 的帧 0、15、30，以及完整连续动画；同时保存所有测试视角，检查轮廓、肢体、衣物、遮挡和重影。原始观测、静态结果与动态结果使用相同区域和显示比例，不通过锐化或生成式修复改善展示效果。

## 实际状态

两组均完成全部 70,000 次更新，并完成每组 124 个留出视图的冻结评测。静态组有 300,145 个高斯，动态组有 368,203 个高斯。人物区域 PSNR 为 20.5289 → 28.3295 dB，SSIM 为 0.78129 → 0.88533；四个相机各自的平均 PSNR 改善为 7.58、7.95、7.72、7.96 dB。全部 124 个视图的这两项指标均改善。

已检查相机 0、10、15、30 的帧 0、15、30，以及按动态组最低人物区域 PSNR 选出的相机 15、帧 21。正面姿态与轮廓明显恢复，侧后视角衣物存在斑点和破碎外观，因此验收状态为“动态重建优于静态，但清晰度仍有未通过项”。全部帧已数值评测，未声称逐帧均已人工视觉验收。

固定首、中、末帧对比：[相机 0](../outputs/dynamic-human-reconstruction-003/review/camera0-frames000-015-030.png)、[相机 10](../outputs/dynamic-human-reconstruction-003/review/camera10-frames000-015-030.png)、[相机 15](../outputs/dynamic-human-reconstruction-003/review/camera15-frames000-015-030.png)、[相机 30](../outputs/dynamic-human-reconstruction-003/review/camera30-frames000-015-030.png)。

独立从保存的 PNG 复算全部人物区域 PSNR，与未量化浮点渲染最大相差 0.007668 dB，小于预设 0.08 dB 量化核验阈值。原始输出保存在桌面学习目录的动态人物重建第 003 阶段，未锐化、超分或生成式修复图片。

## 颜色连续性的单变量诊断

对原代码的实际时序初始化函数进行独立内存探测，确认上一帧颜色参照与当前参数指向相同存储。对当前颜色施加 0.25 的受控变化后，原正则项仍为零，梯度也为零；独立保存的参照产生非零损失。此探测没有修改任何已保存模型。

原始动态模型的颜色参数范围由首帧的约 −3.08～4.48 扩展到末帧的 −39.53～46.26。越界颜色可能在已见视角相互抵消，并在新视角产生伪影，但目前尚不能把全部伪影归因于此。对 4 个相机的首、中、末及较差时刻检查，没有发现处在 0.2～1 米之间、投影在图像内的前景高斯中心，故这些样本中的人体缺损不能用 1 米近裁剪阈值直接解释。

第 004 阶段的两组模型从同一首帧高斯开始，以相同的零 Adam 动量继续训练帧 1～30，每帧 2,000 次；一组保留原参照别名，一组只改为独立冻结上一帧颜色。首帧参数、点数、抽样种子及其余损失均相同。两组均已完成训练和 124 个视图的评测；人物区域 PSNR 为 28.2881 → 28.6110 dB，SSIM 为 0.88454 → 0.89798。这是针对已观察问题的诊断对照，不是新数据集上的独立测试。

已核对四个相机的首、中、末帧，以及相机 15 的第 21 帧。部分背景和彩色噪点减少，侧背面衣物的大块亮斑和破碎外观仍然存在；相同首帧就已有近处衣物误差。这说明修正颜色参照有效，但不能据此判定几何和外观均已重建清楚。

后续第 006 阶段固定为帧 31～60，27 个训练相机、原 4 个留出相机，每组再更新 60,000 次；完整继承对应第 004 阶段的优化器和随机数状态。评测规则在该区间训练前固定。根据第 60 帧原图扩大人物矩形范围以容纳运动，两组使用同一范围；不同区间的区域均值不直接比较。这是同一场景后续时刻的新视角验证，不是预测尚未观察到的人体运动，也不是跨场景泛化。

第 006 阶段现已完成。人物区域平均 PSNR 为 28.8866 → 29.3277 dB，SSIM 为 0.89223 → 0.90819；120 组配对视图的 PSNR 均改善。四个相机的平均 PSNR 改善分别为 0.3773、0.5103、0.3920、0.4850 dB。240 张保存 PNG 的独立复算与浮点 PSNR 最大相差 0.00875 dB，封存代码与协议哈希均一致。

视觉检查覆盖四个相机的帧 31、45、60。大体姿态与轮廓保持，但相机 15 的近处衣袖和手臂仍有严重彩色碎片，相机 30 的躯干仍斑驳。该结果支持颜色参照修正在同场景后续区间有效，不能宣布人物清晰度验收完成。时间图像变化误差均值从 0.01226 降至 0.01150，仍不是运动补偿后的闪烁或三维运动误差。

对照证据：[全部视图指标](../outputs/dynamic-human-color-continuity-004/diagnostic-metrics.json)、[视觉检查与图片复算](../outputs/dynamic-human-color-continuity-004/visual-review.json)、[后续区间冻结协议](../outputs/dynamic-human-temporal-validation-006/evaluation-protocol.json)、[后续区间指标](../outputs/dynamic-human-temporal-validation-006/validation-metrics.json)、[后续区间独立核验](../outputs/dynamic-human-temporal-validation-006/independent-audit.json)。后续区间真实图像：[相机 0](../outputs/dynamic-human-temporal-validation-006/assets/camera0-frames031-045-060.png)、[相机 10](../outputs/dynamic-human-temporal-validation-006/assets/camera10-frames031-045-060.png)、[相机 15](../outputs/dynamic-human-temporal-validation-006/assets/camera15-frames031-045-060.png)、[相机 30](../outputs/dynamic-human-temporal-validation-006/assets/camera30-frames031-045-060.png)。

## 首帧外观与前景覆盖诊断

读取原动态组首帧训练的 2,000、4,000、6,000、8,000、10,000 次更新模型，不改动已保存参数。每个模型在全部 4 个留出相机分别渲染原始颜色、将每个高斯颜色截断到 $[0,1]$ 的诊断结果，以及前景标签的合成贡献；核对三种渲染的累积不透明度逐像素相同。

依据相机 15 的首帧原图，在诊断渲染之前固定衣物内部矩形 $[275,200,365,310]$。10,000 次更新模型在此区域的前景贡献最小值为 0.9813，中位数为 0.99986，低于 0.9 的像素比例为零。颜色截断只使该区域 PSNR 从 29.2613 变为 29.3075 dB；斑块仍明显。这不支持把该内部区域的问题简单归为透明孔洞或越界颜色，也不等于几何已经正确。区域外的手臂、腰部和轮廓仍需各自检查。

![同一几何下的第一帧外观诊断](../outputs/dynamic-human-first-frame-diagnosis-007/assets/initial-10000-camera15.png)

另对 5 个首帧模型的全部 27 个训练相机，用作者训练前景掩膜检查拟合。相机颜色校正后的平均训练前景 PSNR 从 29.4330 增至 30.1507 dB；最终平均前景 IoU 为 0.98707。已检查训练相机 14、16 的真实对照图，它们的衣物外观明显优于留出相机 15；这些编号不表示物理相邻。训练掩膜与留出矩形区域的统计支持不同，不直接相减为泛化差距。当前证据要求继续核对多视角覆盖、几何和颜色拟合，尚未证明单一根因。

![训练相机的真实拟合及既有相机颜色校正](../outputs/dynamic-human-first-frame-diagnosis-007/assets/train-camera14-initial-10000.png)

诊断证据：[参数范围](../outputs/dynamic-human-first-frame-diagnosis-007/parameter-audit.json)、[全部留出相机诊断](../outputs/dynamic-human-first-frame-diagnosis-007/render-diagnostics.json)、[全部训练相机拟合](../outputs/dynamic-human-first-frame-diagnosis-007/training-view-audit.json)、[视觉检查范围](../outputs/dynamic-human-first-frame-diagnosis-007/visual-review.json)。这组已知数据上的事后诊断不计为新的独立测试成绩。

## 固定几何的颜色再训练对照

第 008 阶段从同一个 368,203 点首帧模型出发，固定位置、旋转、尺度、不透明度、前景标签和相机颜色校正，两支仅训练 RGB，分别更新 2,000 次。使用相同零 Adam 动量、0.0025 学习率、种子 73 和逐步完全相同的训练相机序列。测试 RGB 不进入损失。第一支只有原 0.8 L1 与 0.2 SSIM 图像损失；第二支额外加入权重 0.01 的前景空间颜色一致性正则。8 近邻由固定三维中心计算，距离权重逐点归一化，颜色差用平滑绝对值惩罚。只测试这一个预定权重，未按留出成绩搜索参数。

两支训练完成后，所有固定参数与源模型逐元素相同，四个留出视角的不透明度图逐像素相同。四相机人物区域平均 PSNR：原模型 28.8173 dB，纯图像再训练 28.6045 dB，空间正则 28.8391 dB；对应 SSIM 为 0.90463、0.89381、0.90161。12 张保存渲染的独立 PSNR 复算与浮点结果最大差异为 0.00560 dB。

![固定几何下颜色正则的真实对照](../outputs/dynamic-human-appearance-refinement-008/assets/camera15-person.png)

相机 15 已有衣物内部区域的 PSNR 为 29.2613、25.0277、29.2302 dB。纯图像再训练放大了彩色碎片；空间正则抑制这种恶化，但没有恢复原模型缺失的衣物结构。四个相机的对照均已检查。相对原模型，空间正则平均 PSNR 仅高 0.0218 dB，SSIM 更低，故按“没有恢复清晰衣物”的负结果保存，不接入动态主基线。

这是已知场景、已检查相机上的诊断性干预，不是新场景泛化结果。前景仍包含篮球；欧氏近邻可能跨越邻近但不同的表面，正则也可能平滑真实纹理。该有限实验不证明几何是唯一原因，但排除了把这一次固定几何颜色再训练直接当作清晰度修复的做法。进一步工作应核查表面几何、可见性或人体结构约束。

证据：[封存训练协议](../outputs/dynamic-human-appearance-refinement-008/protocol.json)、[全部指标](../outputs/dynamic-human-appearance-refinement-008/evaluation.json)、[独立复算与视觉结论](../outputs/dynamic-human-appearance-refinement-008/independent-audit.json)。

## 证据入口与边界

- [源代码版本](../outputs/dynamic-human-reconstruction-003/source-receipts.json)、[冻结评测约定](../outputs/dynamic-human-reconstruction-003/evaluation-protocol.json)。
- [动态训练协议](../outputs/dynamic-human-reconstruction-003/dynamic/protocol.json)、[静态训练协议](../outputs/dynamic-human-reconstruction-003/static/protocol.json)、[可微检查](../outputs/dynamic-human-reconstruction-003/smoke/checks.json)。
- [方向与阶段状态](../manifests/dynamic-human-work.json)。
- [留出指标与逐视图记录](../outputs/dynamic-human-reconstruction-003/test-metrics.json)、[图片数值复算](../outputs/dynamic-human-reconstruction-003/saved-output-audit.json)、[视觉检查范围](../outputs/dynamic-human-reconstruction-003/visual-review.json)。
- [实现版本哈希](../outputs/dynamic-human-reconstruction-003/implementation-receipt.json)、[颜色参照探测](../outputs/dynamic-human-reconstruction-003/color-reference-audit.json)。
- [未修正参照的训练协议](../outputs/dynamic-human-color-continuity-004/aliased/protocol.json)、[冻结参照的训练协议](../outputs/dynamic-human-color-continuity-004/frozen/protocol.json)。

## 街景数据衔接

已对 DriveStudio 官方示例列表中的 Argoverse 2 首个街景进行输入筛选。官方公共存储可访问；单个日志包含 3,035 个对象，共 1,263,003,345 字节。标注中有 76 条行人轨迹；按至少 20 次标注观测、补偿自车运动后首尾位移至少 1 米的初筛条件，得到 65 条候选。该计数是数据筛选结果，不是重建或跟踪成绩。

最近的候选轨迹距自车约 3.82 米，持续 13.8 秒，世界坐标首尾位移约 18.41 米。已核对同一附近时刻的 7 个环视相机原图，右侧相机可见近处行人躯干、腿部和衣物；完整单场景的 3,035 个文件已下载、核对远端对象大小并保存 SHA-256。原数据的面部与车牌匿名化保留，后续以人体轮廓、肢体和衣物重建为目标。

此数据按 Argoverse 2 的 CC BY-NC-SA 4.0 用于本地研究与学习，来源为 © 2021 Argo AI, LLC。数据取得时尚无街景训练结果；后续已完成无人体模板的街景基线，官方 SMPL 模板接入与人体分支验证见文末。

输入证据：[行人轨迹初筛](../outputs/dynamic-human-reconstruction-003/street-preflight/human-track-screening.json)、[单场景下载与文件哈希](../outputs/dynamic-human-reconstruction-003/street-preflight/street-log-data-receipt.json)。

提供点云的构建相机子集未公开说明，故初始化的信息来源边界需保留。数据本身的分辨率和室内采集范围也限制结论：本阶段验证人物动态重建，街景规模、远距人物和真实遮挡仍需在后续公开街景片段上完成。

## 街景预处理与人体姿态接口核验

固定作者 AV2 SDK 分支版本 `4bbc40c12ecc1f9df31df7b999c813e26419c15f`。复用 SDK 的相机模型、时间配对、姿态、激光与三维标注读取；原样执行 DriveStudio 的标定、激光、位姿和粗掩膜导出函数。相较原处理程序，图片直接复制原始 JPEG 字节，避免再次有损压缩；不加载本次导出不需要的地图栅格和 GPU 可视化依赖。

157 个原始激光时刻中，第一个不满足全部 7 路相机与对应位姿齐全的条件，按原规则移除。其余 156 个时刻生成 1,092 张原图、156 帧点云、124 个动态实例（76 条行人轨迹）和 3,276 张粗掩膜。粗掩膜保持作者的三维框投影规则，要求八个角点全部可见；其中 human 组还包含其他非刚体类别，不能当作精确人体分割。

全部配对用整数纳秒独立复查，与 SDK 最近时间配对一致，最大差 24.1244 毫秒，中位绝对差 12.4709 毫秒。相机拍摄时刻与激光时刻的相机位置最大相差 0.2231 米。在两种投影都完整落在图像内的 7,700 个行人框观测上，仅改变自车位姿时间后，八角点平均位移的总体均值为 3.8227 像素，95 分位数为 16.5149 像素。该量是两种投影之间的差异，不是相对于二维真值的误差改善。

![近处行人原图与标注框的时间补偿核查](../outputs/dynamic-human-street-005/assets/street-camera-time-review.png)

已检查最近行人在右侧相机第 134 帧的完整图像与原生像素裁图。自车时间补偿后的框仍需人物运动补偿；不能把三维包围框边界当作人体轮廓。额外保存的 `camera_pose_capture` 为相机到城市坐标的变换。新增可选择启用的适配器同时修改照片渲染与人体全局朝向转换使用的两个相机位姿入口，保持以区间首个激光时刻自车位姿为参考，与物体和点云坐标一致。原仓库保持不变。

使用真实标定数据、原作者标定函数和适配器检查了全部 1,092 个相机位姿，另外覆盖两个非零起始时间窗口与两种加载分辨率，共 42 种相机／窗口／尺度组合。内参缩放和畸变参数保持原值，照片渲染与人体朝向使用的相机变换逐元素一致，缺少拍摄时刻位姿时明确报错。这一轮为隔离标定函数检查；随后在 Linux 中导入完整训练模块，并在原相机类构造函数中验证了帧 0、134 的全部 7 个相机，共 14 个真实图像／掩膜／射线案例。仍需区分已通过的加载检查和未完成的人体网格、联合训练；物体标注仍处于激光时刻，尚无物体运动或滚动快门补偿。

原作者人体姿态归档下载大小为 65,709,897 字节，只提取当前场景。使用限制为三类允许全局对象的读取器，将张量载入 CPU；另存为无需 pickle 的数组归档。27 个行人、2,366 个有效姿态观测全部通过帧编号、类别、相机编号、有限值与旋转检查。每帧包含 1 个全局朝向矩阵、23 个局部关节旋转矩阵和 10 个体形系数。旋转正交性和行列式最大误差均为约 $4.77\times10^{-7}$。

近处主目标对应实例 79，139 次三维框标注中有 100 帧有效人体姿态，来自相机 0、2、4、6。归档只有作者数字实例编号，不含原始 UUID；接口验证证明其与本地导出的编号、类别及有效帧一致。随后已接入官方 SMPL 模板并检查真实投影，仍有明显初始偏移，见文末。

语义掩膜采用 NVIDIA 发布的 SegFormer B5 Cityscapes 权重，版本 `2c6f153e4c23c229e2fa2b188eb250607e030cd8`。按照原配置的 1024×1024 窗口、768 像素步长在原生图像上滑窗，重叠区域平均 logits。使用 Transformers 与 FP16 推理，未复制旧 MMSeg 的整图缩放增强流程，因此明确作为本地预处理适配。全部 1,092 张图片已完成推理，6,552 张输出通过独立的原生尺寸、哈希及类别映射逐像素核对；1,067 张图片含预测 person 像素。这里的计数不衡量分割准确率。

![真实街景与预测的人物天空掩膜](../outputs/dynamic-human-street-005/assets/semantic-preview-134_4.png)

检查了预先固定的相机／帧 0／000、2／078、4／134、6／155。近处衣物区域可以提取，但遮挡人物会连在一起，手提袋部分被归入人物，匿名化头部有漏分；远处小人物也不能逐个保证完整。这些是语义预测，不能作为人物实例真值，也不能用于虚构分割精度。原图匿名化不变。

新增接口与输出证据：[拍摄时刻位姿适配器核验](../outputs/dynamic-human-street-005/capture-time-adapter-audit.json)、[整段语义推理](../outputs/dynamic-human-street-005/semantic-completion.json)、[全部掩膜独立核验](../outputs/dynamic-human-street-005/semantic-output-audit.json)、[掩膜视觉检查](../outputs/dynamic-human-street-005/semantic-visual-review.json)。

证据：[预处理版本与时间核查](../outputs/dynamic-human-street-005/preprocessing-receipt.json)、[人体姿态接口核验](../outputs/dynamic-human-street-005/input-audit.json)、[语义预测协议](../outputs/dynamic-human-street-005/semantic-protocol.json)、[单图推理结果](../outputs/dynamic-human-street-005/semantic-smoke.json)。

## 街景 Linux 环境与实际 CUDA 运算

独立环境位于本地 WSL 的 dynamic-human 目录，保持此前 SplaTAM 环境不变。采用 Torch 2.7.1 + CUDA 12.8、PyTorch3D 0.7.9、nvdiffrast 0.4.0、gsplat 1.3.0，针对 RTX 5080 的计算能力编译。最初缺少 cuSPARSE 头文件搜索路径，已将当前环境中官方 NVIDIA 包的 include 与 lib 目录加入构建环境；原失败日志保留。完整 DriveStudio 数据、人体节点和训练模块导入通过。

实际 CUDA 核验包括：PyTorch3D 最近邻距离与独立张量计算的前向差异最大约 $1.49\times10^{-8}$，反向梯度一致；nvdiffrast 的三角形插值与重心坐标表达一致，顶点和属性梯度有限且非零；gsplat 单高斯渲染满足黑背景下颜色等于不透明度乘高斯颜色，颜色梯度与解析值一致。这些是小规模算子检查，不是街景重建成绩。

完整相机类检查实际载入了原图、预测人物／动态物体／天空掩膜，并生成相机射线。图片与作者缩放规则一致；fine_dynamic_masks 被实际选中；拍摄时刻位姿、整数纳秒时间戳、单位射线方向和独立中心像素投影均通过。覆盖两帧、七相机和四倍下采样，完整 1,092 位姿范围由前述矩阵审计覆盖。

整段数据随后通过原 DrivingDataset 构造：156 帧、7 相机共 1,092 张照片连同激光、对象与姿态全部加载；按每 10 帧留出的规则形成 987 张训练图与 105 张测试图。27 个行人的 2,366 个有效姿态逐个核对了世界坐标根旋转、23 个局部关节旋转和三维框中心平移，主目标 79 保留 100 个有效姿态。原始姿态归档使用已验证的限制解码器；这仍不是人体网格与图像的视觉对齐。

这次完整加载发现评测边界问题：作者原流程在划分训练／测试之前，用所有图像为激光点赋色。因此最初的完整数据构造不宣称初始化未见测试 RGB。后续增加仅使用训练时刻照片与激光点的适配器，并完成下述真实输入干预；已知三维姿态和相机先验的可用范围仍在街景协议中单独声明。

证据：[扩展构建与模块导入](../outputs/dynamic-human-street-005/linux-extension-build-completion.json)、[实际 CUDA 前后向核验](../outputs/dynamic-human-street-005/linux-cuda-smoke.json)、[完整相机加载器](../outputs/dynamic-human-street-005/full-camera-loader-audit.json)、[整段数据与人体姿态核验](../outputs/dynamic-human-street-005/full-scene-dataset-audit.json)。

## 测试照片与初始化隔离

采用原场景第 120～140 帧、全部七个相机，共 147 张真实图片；按原相对时间步长划分，133 张用于训练，第 130、140 帧的 14 张照片留出。第二次构造只将这 14 张测试照片全部替换为纯品红色。训练照片不变，相机、轨迹、激光与随机种子不变。

未修改的作者赋色流程在干预后产生不同的点云颜色，构成检测信息混入的有效对照。新增适配器只允许训练时刻的照片赋色与标记可见点，最终保留 1,865,811 个训练时刻激光点。两次构造的训练 RGB、点云几何／颜色／时间、背景采样种子、7 个刚体实例和 41 个可变形实例的全部种子与轨迹逐项哈希一致；测试时刻不再向初始化提供激光点。此检查针对初始化，不能单凭它宣称任意后续训练代码都不会读测试图。

稀疏深度还有独立的可见性处理：多个激光点落入同一像素时，原实现按索引覆盖，重复像素写入不保证稳定的最近点。适配器采用最小正相机深度，即保留最近命中；训练和测试照片均保留独立的深度图，后者只用于评测。147 张稀疏深度图全部与 NumPy 的独立最小值归约逐像素一致。两次构造分别有 86,914 和 87,875 个像素的深度与原覆盖结果不同，这是碰撞处理变更的计数，不是对真实深度误差的改善量。

原始图片、点云、作者仓库和最初失败日志均保留。新增适配器在进程内显式启用。相机拍摄时刻位姿、三维框和轨迹仍是已知输入；隔离测试 RGB 与测试时刻点云不等于未知轨迹预测，也没有解决人物自身运动造成的跨传感器时间差。

证据：[真实输入干预、原实现对照与种子核验](../outputs/dynamic-human-street-005/training-only-initialization-audit.json)。

## 街景联合重建对照的运行范围

完整 156 帧街景按作者默认三分之一图像尺寸载入，7 路相机形成 987 张训练图、105 张留出图。模型包含静态背景、刚体和可变形节点；省略 SMPL 人体节点后，移动人物由可变形节点表示。相机采用已知拍摄时刻位姿并固定，不优化逐图片相机校正和曝光仿射参数。其余训练与加密配置保留原 OmniRe 配置，预算固定为 30,000 次更新，形变网络在第 3000 步之后参与。

正式训练前单独完成三个被丢弃的优化步骤，分别检查初始分辨率、完整加载分辨率和形变网络启用后的运行路径，并渲染三个留出画面。这里跳过中间训练步，只检查运行条件，不能当作已经训练 3000 步。完整模型初始化有 1,011,651 个背景高斯、81,278 个刚体高斯及 157,178 个可变形高斯；实际最高分配显存为 3,725,888,000 字节。这一数值是该检查时模型分配量，不包含桌面等其他程序，也不代表后续高斯加密的最终显存上限。

人物评测区域在正式更新前由三维人物标注投影固定：保留正深度、裁剪后面积不低于原投影面积一半、宽至少 12 像素、高至少 24 像素且面积至少 512 像素的框，共 435 个人物区域，其中主目标 79 有 13 个区域。该区域含背景及可能被遮挡的人物，不能称为精确人体掩膜。正式训练完成后评测全部 105 张留出图，保存完整渲染和人物原生裁图；中途仅保存预定近处人物视图，不据此改变模型参数。训练、初始化与评测均不使用增强或生成式图像修复。

这轮无人体模板的可变形对照已完成，训练结果与视觉核查见下一节；它不能直接视为完整 OmniRe 的复现成绩。人体关节模型另行验证，街景人物清晰度尚未达到目标。

首个无显存上限的运行在已记录 2,001 次更新后出现明显停滞：模型最高分配约 4.16 GB，但分配器最高保留约 17.27 GB。该运行被中断，500 次更新的模型、完整优化器状态和真实渲染均保留。新运行设置 10 GB 的进程分配上限并定期清理空闲缓存，保持同一种子、模型配置、训练照片与总迭代预算，从头开始；不把先前运行的更新次数累加到新运行。此调整针对内存使用，不是提高重建质量的新方法。

证据：[被丢弃的运行检查](../outputs/dynamic-human-street-joint-009/preflight/completion.json)、[完整初始化规模](../outputs/dynamic-human-street-joint-009/preflight/initialization.json)、[初次运行中止记录](../outputs/dynamic-human-street-joint-009/baseline/termination.json)、[正式训练协议](../outputs/dynamic-human-street-joint-009/baseline-bounded/protocol.json)。

## 街景检查点、完整状态恢复与人物清晰度

10 GB 上限的运行完成并保存 10,000 次更新，后续加密时发生 CUDA 显存分配失败，失败日志与中止原因均保留。新的恢复运行将分配上限调为 12 GB 并启用可扩展内存段；数据、模型、损失、学习率计划与总预算不变。恢复检查逐项核对模型、Adam 状态、随机数与加密累计量，三张预定视图与原 10,000 次检查点保存的 PNG 逐像素一致，首个继续训练的图片编号也一致。前次从头启动的运行与此次完整状态恢复是两个不同概念，不能混称为全部过程逐步相同。

![近处侧视人物的观测与训练至三万次的检查点](../outputs/dynamic-human-street-joint-009/final-image-audit/frame130-camera4-person79-comparison.png)

![多人重叠和画面边缘处的人物对照](../outputs/dynamic-human-street-joint-009/final-image-audit/frame140-camera4-person79-comparison.png)

![后视相机中的同一人物对照](../outputs/dynamic-human-street-joint-009/final-image-audit/frame150-camera6-person79-comparison.png)

三图均采用训练前规定的人物框、同样的最近邻显示放大，未进行锐化。基线已完成 30,000 次更新。帧 130 的人物框 PSNR 从 10,000 次时的 18.48 dB 提高到 20.24 dB，但衣物褶皱、手和双腿仍不清晰；帧 140 包含遮挡人物及画面边界，不是孤立人物误差；帧 150 的人物框 PSNR 达到 27.09 dB，脚部和衣物仍模糊。不能把人物框 PSNR 或整个背景更干净等同于人体重建完成，也不能从训练过程中多个参数同时变化的检查点对照推断某个模块的独立因果作用。

最终全部 105 张留出图与 435 个人物框已完成数值评测和独立核验；连同早期检查点，共核对 114 张完整渲染与 471 个区域。保存的目标图与原始 JPEG 采用相同双线性缩放后逐像素一致。PNG 与浮点 PSNR 最大差为 0.07273 dB，全部位于由每通道半个灰度级量化误差推导的界内。最终全图 PSNR 均值为 22.41 dB，主目标 13 个框均值为 19.07 dB；这些是逐图／逐框 dB 值的平均，不是合并全部像素后的 PSNR，也不是精确人体分割指标。

主目标全部 13 个留出区域均已看图，包含人物框 PSNR 最低的帧 80／相机 2。近处主要问题仍是躯干边缘扩散、衣物纹理缺失和腿部混合。另检查了全图 PSNR 最低的帧 130／相机 6：原图中的自车引擎盖未被重建，道路与人物也有误差，因此该全图分数不能直接归因于人体。当前结论是基线训练与评测完成，清晰人物重建未通过。

完整人物区域对照：[前段五组](../outputs/dynamic-human-street-joint-009/final-image-audit/person79-all-heldout-1.png)、[近处五组](../outputs/dynamic-human-street-joint-009/final-image-audit/person79-all-heldout-2.png)、[后视三组](../outputs/dynamic-human-street-joint-009/final-image-audit/person79-all-heldout-3.png)。

证据：[显存失败记录](../outputs/dynamic-human-street-joint-009/baseline-bounded/termination.json)、[恢复协议](../outputs/dynamic-human-street-joint-009/resume-10000-expandable/protocol.json)、[恢复逐项核验](../outputs/dynamic-human-street-joint-009/resume-10000-expandable/recovery-audit.json)、[最终图像独立复算](../outputs/dynamic-human-street-joint-009/final-image-audit/independent-audit.json)、[实际视觉检查](../outputs/dynamic-human-street-joint-009/final-image-audit/visual-review.json)、[统计口径与汇总](../outputs/dynamic-human-street-joint-009/final-image-audit/summary.json)。早期检查点核查另行保留：[原数值记录](../outputs/dynamic-human-street-joint-009/checkpoint-image-audit/independent-audit.json)。

## 官方 SMPL 模板接入与初始对齐

用户从官网下载的 Python 2.7 v1.1.0 压缩包为 330,607,275 字节，ZIP CRC 检查通过。仅提取中性模板和许可说明，保留原始文件。模板虽然使用旧版带有“10”的文件名，实际提供 300 个体形基；兼容转换保留全部基和其他数组的逐字节数值，由作者 SMPLLayer 默认选取前 10 个。Windows 与 WSL 使用相同的兼容文件，原模板及转换后的文件均不进入网站公共资源。

实际调用 DriveStudio 的 SMPLTemplate，采用作者 DA 参考姿态、线性混合蒙皮和姿态网格包围盒中心修正，逐帧生成目标人物 79 的全部 100 个有效网格。每个网格有 6,890 个顶点和 13,776 个三角面；关节数为 24。全局朝向从姿态来源相机转换到世界参考系，对象平移沿用激光时刻标注，未额外拟合测试图像。

![原始街景与 SMPL 初始网格的真实投影，近处人物仍有明显偏移](../outputs/dynamic-human-smpl-010/frame130-camera4-person79-pair.png)
> 图源：Argoverse 2 原始图像，CC BY-NC-SA 4.0；SMPL 模型由 Max Planck Institute for Intelligent Systems 提供，Loper 等（2015）· 本地非商业研究核查。左为观测，右为初始网格叠加，未恢复匿名化面部。

已检查帧／相机 60／0、90／2、120／2、130／4、140／4、140／6、150／6 共七组图。蓝色网格大致落在目标行人附近，但近处头部经常高于或偏离真实头部，躯干、手臂和双腿位置也不一致。因此当前结论是模板可用、姿态能运行，初始视觉对齐不通过。该叠加没有场景遮挡合成，也不表示衣物已经重建；下一步需核对并优化人体位置、关节与场景外观。

证据：[下载与安装](../outputs/dynamic-human-smpl-010/template-installation.json)、[无损数值转换](../outputs/dynamic-human-smpl-010/template-conversion.json)、[网格前向检查](../outputs/dynamic-human-smpl-010/mesh-projection-audit.json)、[视觉检查范围与问题](../outputs/dynamic-human-smpl-010/visual-review.json)。

随后使用真实人物 79 的第 128～132 帧在 CPU 上构造实际 SMPLNodes，保留作者体素蒙皮变形器设置。节点初始化、中心与四元数变换、平移／根旋转／关节旋转的非零有限梯度，以及留出时刻相邻姿态插值均通过。该检查只使用一个人物、五个时刻与被丢弃的代数梯度探针，未进行图像优化，不能替代完整街景前后向和对齐验证。[实际人体节点检查](../outputs/dynamic-human-smpl-010/smpl-node-smoke.json)。

进一步保持体形和关节不变，仅沿城市竖直方向把网格最低点对齐到标注框最低点。100 帧所需下移量约为 0.20～0.28 米。帧 130 的脚部高度更接近路面，但水平偏移仍在；帧 120 的躯干、手臂与腿仍错位；帧 150 的头部与肢体也未对齐。标注框底部本身不是独立的足部接触真值，因此未将此规则纳入人体训练对照。见[真实投影](../outputs/dynamic-human-smpl-010/box-bottom-diagnostic/frame130-camera4-person79-pair.png)、[诊断记录](../outputs/dynamic-human-smpl-010/box-bottom-diagnostic/mesh-projection-audit.json)、[视觉结论](../outputs/dynamic-human-smpl-010/box-bottom-diagnostic/visual-review.json)。

## 完整街景人体关节模型的运行检查与对照范围

保留基线的数据划分、三分之一载入分辨率、相机和曝光固定设置、背景及其他对象配置，恢复作者 SMPL 人体节点及其自身正则。此对照同时改变人体几何、拓扑与正则，不是单一损失项或同容量消融。两支现均完成 30,000 次正式更新，并使用原先固定的全部 105 张留出图和 435 个人物框完成评测。

原始数据对主目标只有第 56～155 帧的 100 个有效人体姿态，但三维标注从第 17 帧开始，共 139 帧。若按“存在人体姿态文件”直接移除整条可变形轨迹，会漏掉前 39 帧。因此人体姿态有效时由 SMPL 节点表示，缺姿态时保留可变形节点。实际初始化逐轨迹核对两组有效帧不重叠、并集等于原标注；主目标明确满足 139 = 100 + 39。该覆盖修正不等于时间连续性已经通过。

完整模型初始化含 179,140 个人体高斯，对应 26 个采用人体模板的运动实例。分别在日程位置 0、1001、3001、10001 进行了四次检查更新，实际只更新四次，权重全部丢弃。各步损失有限，人体平移、根旋转、关节旋转及高斯参数均获得非零有限梯度；三张固定留出画面完成渲染。检查阶段最高实际分配显存约 5.45 GB，不代表高斯加密后的正式训练峰值。

正式运行已在新的进程中重新初始化，未使用检查阶段的模型或优化器。作者估计的姿态与体形包含留出时刻；未证明其上游姿态估计过程完全排除了留出 RGB。当前图像优化和激光／颜色初始化只使用训练时刻。初始网格存在偏移，最终渲染对照也未通过清晰度验收，不能由梯度非零推断人物清晰度改善。

证据：[运行检查](../outputs/dynamic-human-street-smpl-joint-011/preflight/completion.json)、[实际模型与逐轨迹覆盖](../outputs/dynamic-human-street-smpl-joint-011/preflight/initialization.json)、[正式训练协议](../outputs/dynamic-human-street-smpl-joint-011/articulated/protocol.json)。

## 人体颜色、几何对齐与遮挡对应的中途诊断

500、3,000、10,000 次更新均按相同预算比较两种表示。每阶段三张预定完整图、12 个人物框及主目标原生裁图完成独立核验，目标图片与原始 JPEG 的同流程缩放逐像素一致。人体分支的初始随机杂色逐步减少，但到 10,000 次仍有衣物模糊、轮廓破碎与肢体不清。三个人物框的 PSNR 均值为可变形模型 21.37 dB、人体关节模型 21.25 dB；这一小范围中途结果不能推断最终优劣。

![一万次更新时相同人物区域的真实对照](../outputs/dynamic-human-street-smpl-joint-011/paired-updates-10000-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实测 · 相同更新预算、区域和显示缩放，未锐化 · 原始数据 CC BY-NC-SA 4.0。

此外在 CPU 上独立恢复实际 3,000 次人体检查点，所有保存张量逐项精确一致。采用训练后的高斯中心和原始 SMPL 拓扑显示投影；这是一项几何诊断，不是带衣物的纹理网格。帧 129／相机 4 和帧 151／相机 6 均属于训练输入，仍能看到头部、躯干和腿部偏移；留出帧 130 也有类似问题。由此只能确认采样训练视图中也存在几何偏差，不能声称已经确定全部误差来源。

![训练帧中实际保存的人体几何仍有偏移](../outputs/dynamic-human-street-smpl-joint-011/geometry-03000-cpu/frame129-camera4-person79.png)
> 图源：Argoverse 2 原图、本站实际模型投影 · SMPL：Loper 等（2015），MPI · 仅作本地研究诊断，无场景遮挡合成。

对四张训练照片另用 Torchvision 官方 COCO Keypoint R-CNN 估计二维人体关节，没有读入留出 RGB，也没有据此修改训练。检查发现：帧 129、131 的部分手臂点落在旁人的手臂上，帧 131 的双踝预测挤在同一只前脚附近；后视图交叉双腿也存在歧义。因此未把整组预测当作姿态监督。人物框重叠率和较高检测分数不足以保证每个关节属于同一人。

读取 500 与 15,000 次检查点时，主目标 90 个有效训练时刻的平移改变量中位数约为 6.02 厘米，根旋转改变量中位数约为 0.30°，局部关节旋转改变量中位数约为 0.90°。这些是相对早期检查点的实际参数变化，不是真实位姿误差；变化小或大都不能直接证明对齐正确，仍须结合原图投影与留出重建判断。

证据：[一万次配对图片核验](../outputs/dynamic-human-street-smpl-joint-011/paired-updates-10000-audit/independent-audit.json)、[三组视觉检查](../outputs/dynamic-human-street-smpl-joint-011/paired-updates-10000-audit/visual-review.json)、[实际几何恢复](../outputs/dynamic-human-street-smpl-joint-011/geometry-03000-cpu/receipt.json)、[二维关节原图检查](../outputs/dynamic-human-training-alignment-013/frame129-camera4-keypoints.png)、[关节对应结论](../outputs/dynamic-human-training-alignment-013/keypoint-visual-review.json)、[实际参数变化](../outputs/dynamic-human-training-alignment-013/parameter-updates-through15000.json)。


## 三万次人体表示对照：平均改善与近处退步并存

两种表示均已完成 30,000 次更新，各自 105 张完整留出渲染、435 个人物区域全部独立核验。主目标的全部 13 组原始观测／可变形／人体关节裁图逐一查看。目标 PNG 与原始 JPEG 经规定缩放的像素完全一致，区域边界相同，保存图片的 PSNR 均处于浮点评测量化误差界内。此比较同时改变表示、拓扑及正则，不是单因素消融。

![最终近处侧面原始观测与两种人体表示](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实测 · 帧 130、相机 4 · 同预算、同区域、未锐化 · 原始数据 CC BY-NC-SA 4.0。

主目标区域 PSNR 均值从 19.07 提高到 19.70 dB，SSIM 从 0.566 提高到 0.580；全图平均 PSNR 则为 22.41 与 22.39 dB。帧 130／相机 4 的人物区域从 20.24 下降到 19.16 dB，原图清楚的衣褶、手提袋边界和分开的双腿仍未恢复。帧 140／相机 6 的区域分数提高较多，但含大量背景，人物本身仍模糊。不能把总体均值改善解释为各视角均清晰。

![最终侧后方人物对照](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/frame150-camera6-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实测 · 帧 150、相机 6 · 同区域显示，未进行图像修复。

人体分支的全图 PSNR 最低视图也是帧 130／相机 6。核对发现自车引擎盖未重建，道路、植被与行人也存在误差，所以最低全图分数并非独立的人体质量。当前结果是两支训练和数值评测完成，清晰人物重建未通过。

证据：[最终配对核验与全部指标](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/independent-audit.json)、[十三组实际看图范围](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/visual-review.json)。

## 连续运动、遮挡与人体表示切换

四个窗口在渲染前固定，共 80 个不同的帧／相机组合。两支均从最终模型恢复，每个保存张量精确一致；与已保存留出图重叠的视图重新渲染，每通道差异不超过一个 8 位灰度级。共 160 张新渲染和 152 组相邻差分完成独立复算。视频保持约 10 Hz 的原拍摄节奏，包含训练时刻和留出时刻，不能称为 80 个全新视图。

四段人物固定区域对照：[姿态起始与表示切换](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/pose-onset-fixed-union.mp4)、[人物接近](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/approach-fixed-union.mp4)、[侧面遮挡](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/side-occlusion-fixed-union.mp4)、[后方遮挡](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/rear-occlusion-fixed-union.mp4)。完整画面对应为：[起始](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/pose-onset-full-image.mp4)、[接近](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/approach-full-image.mp4)、[侧面](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/side-occlusion-full-image.mp4)、[后方](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/rear-occlusion-full-image.mp4)。

每个窗口抽看四张对照，共 16 组；视频全部帧通过解码检查，但尚不等于每一帧均逐帧视觉验收。近处侧面在训练帧中也可见破碎轮廓和不清晰肢体。该窗口固定区域平均 PSNR 从 24.19 降到 22.94 dB，相邻图像变化残差从 0.04394 升到 0.04840；其他窗口表现不一。固定区域会包含背景与其他人物，差分也含自车运动、人物运动及光照变化，不能直接称作闪烁指标。远处第 56 帧的表示切换只能检查该距离下的连续性，不能证明近处衣物清晰。

证据：[预先固定的窗口](../outputs/dynamic-human-street-temporal-012/protocol.json)、[全部图像与差分独立核验](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/independent-audit.json)、[视觉检查边界](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/visual-review.json)。MP4 是有损观看副本，所有数值来自原 PNG，未锐化或生成式增强。

## 区分姿态误差与高斯表面偏移

![同一训练帧的原始观测、仅保留优化姿态的参考人体、实际训练高斯中心](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/frame129-camera4-decomposition.png)
> 图源：Argoverse 2 原图与实际 15,000 次模型的 CPU 投影 · SMPL：Loper 等（2015），MPI · 几何诊断，没有场景遮挡合成。

左列是原始训练照片，中列保留实际优化的根平移、根旋转和关节旋转，恢复固定参考顶点和固定蒙皮权重，右列使用实际训练高斯中心及学习到的蒙皮。帧 129／相机 4、帧 151／相机 6 的中列仍有头部、腿部偏移，右列出现额外的锯齿状边缘。右列用原拓扑连接高斯中心仅作诊断，不能称作衣物表面网格。该分解同时去掉位置偏移和体素蒙皮修正，不能据此单独量化某一项的因果作用。

因此下一步优先进行训练图像上的姿态拟合，再检查高斯偏移约束和衣物细节。已发现二维关节预测混入旁人肢体，不能直接当作真值加入监督，也不继续以增加背景高斯或延长总训练替代对齐。

证据：[几何分解计算](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/receipt.json)、[实际看图范围](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/visual-review.json)。

## 从姿态更新到有效拟合

以完成 30,000 次训练的人体模型为共同起点，冻结所有高斯属性、体素蒙皮、背景、其他人物、相机与曝光，只更新主目标 79 的 90 个训练时刻位置和关节。按同一投影框规则得到 101 个人物训练区域，留出图像仍不进入优化；留出姿态条目保持不变，渲染继续使用原有相邻姿态插值。此处是问题诊断，增加了额外优化预算，不是与原模型同预算的算法比较。

第一种单图随机优化完成 3,000 次更新后，训练区域平均 PSNR 从 19.79 降到 18.92 dB，平均绝对误差从 0.07479 升到 0.07984；留出目标区域从 19.70 降到 19.19 dB。三组预定近处图仍有破碎衣物、模糊双腿，后视图手臂及提袋也有偏移，因此未替换原模型。仅从这个负结果不能判断姿态对齐无用，因为训练拟合本身已经恶化。[协议](../outputs/dynamic-human-street-pose-refinement-014/protocol.json)、[独立图像复算](../outputs/dynamic-human-street-pose-refinement-014/independent-audit/independent-audit.json)、[实际视觉检查](../outputs/dynamic-human-street-pose-refinement-014/independent-audit/visual-review.json)。

第二种从原始 30,000 次模型重新开始，以全部 101 个训练区域的平均颜色与结构误差为目标，采用 L-BFGS 和线搜索，附加相对原姿态的变化约束及相邻时刻修正量平滑。20 次优化迭代共进行 21 次完整训练目标计算，对应 2,121 次区域反向计算，不能与单图随机更新次数等同。模型选择只依据训练目标；目标从 0.13425 降到 0.12777，训练区域平均 PSNR 提高至 20.06 dB，平均绝对误差降至 0.07257。

![原始观测、原人体模型与全训练区域姿态优化后的真实留出对照](../outputs/dynamic-human-street-pose-batch-015/independent-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原图与本站实测 · 帧 130、相机 4 · 同区域、未锐化；第三列增加姿态优化预算。

独立复算全部 105 张留出图和 435 个人物区域后，主目标 13 个框的平均 PSNR 从 19.70 提高到 19.88 dB，SSIM 从 0.580 提高到 0.596。另直接比较源模型和新模型：80 个其他模型张量条目精确不变，非目标人物以及全部留出时刻的姿态值也精确不变。位置变化中位数约 1.03 厘米，根旋转约 2.14°，局部关节约 0.65°；这是参数变化，并不是真实位姿误差。

实际查看帧／相机 110／2、130／4、140／4、150／6 四组，衣褶、手和双腿仍不清晰，也尚未验证新候选的连续遮挡画面。小幅指标改善只说明该姿态更新有助于当前图像拟合，尚未完成清晰重建。20 次迭代结束时训练目标仍在下降，姿态变化尚未触及限制，因此进一步保持同一目标与冻结范围检验收敛，再决定是否调整表面自由度。

证据：[全训练区域优化协议](../outputs/dynamic-human-street-pose-batch-015/protocol.json)、[仅训练目标的选择记录](../outputs/dynamic-human-street-pose-batch-015/training-selection.json)、[独立模型参数检查](../outputs/dynamic-human-street-pose-batch-015/independent-state-audit.json)、[独立图像复算](../outputs/dynamic-human-street-pose-batch-015/independent-audit/independent-audit.json)、[四组视觉检查与边界](../outputs/dynamic-human-street-pose-batch-015/independent-audit/visual-review.json)。

### 延长姿态优化后，衣物模糊仍然存在

保持相同起点、全部训练区域、目标函数与冻结范围，把上限延长到 120 次 L-BFGS 迭代，共计算 123 次完整目标、12,423 次区域反向传播。这是从原始模型重新开始的同目标长预算验证，并非恢复前一轮优化器状态。训练目标从 0.13425 降到 0.12260，训练人物区域 PSNR 从 19.79 提高到 20.40 dB，平均绝对误差从 0.07479 降到 0.06970。

最终使用训练目标最优且满足预设变化范围的第 120 次目标计算结果；其后试探点超过局部关节变化上限而未采用。训练目标仍在下降，不能将达到迭代上限称为已收敛。相对源模型的位置变化中位数为 3.24 厘米，根旋转为 5.65°，局部关节为 5.12°，最大局部关节变化约 44.89°。这些是参数变化量，没有真实关节误差的含义。

![长预算姿态优化的完整近处侧面留出对照](../outputs/dynamic-human-street-pose-convergence-016/independent-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实测 · 帧 130、相机 4 · 同区域、未锐化；第三列仅改变训练时刻姿态。

独立复算全部 105 张留出图、435 个人物区域，并检查全部模型张量后，主目标 13 个框的平均 PSNR 为 19.94 dB、SSIM 为 0.595；其他人物、背景、高斯属性与留出姿态条目保持精确不变。此次已逐一查看全部 13 组裁图，接近、侧面和后方画面仍有模糊衣物与不清晰的腿部。长预算比 20 次迭代仅增加约 0.07 dB，SSIM 还略有下降，不能根据单一指标把它选为质量已达标的模型。

结论是姿态拟合有一定帮助，但还不足以解释或解决表面模糊。原始基线与两轮诊断模型均保留；高斯尺度与遮挡贡献见下节，当前候选后续已纳入[连续运动对照](#表面拟合后的连续运动对照)。

证据：[长预算协议](../outputs/dynamic-human-street-pose-convergence-016/protocol.json)、[训练选择记录](../outputs/dynamic-human-street-pose-convergence-016/training-selection.json)、[独立模型核验](../outputs/dynamic-human-street-pose-convergence-016/independent-state-audit.json)、[全部图像核验](../outputs/dynamic-human-street-pose-convergence-016/independent-audit/independent-audit.json)、[十三组实际视觉检查](../outputs/dynamic-human-street-pose-convergence-016/independent-audit/visual-review.json)。

## 高斯覆盖范围与遮挡合成的实际分解

固定长预算姿态候选的全部模型参数，对四个训练视图与两个留出视图做只读干预，没有执行训练。分别保存完整渲染、移除主目标、保留所有遮挡时的目标可见权重、目标颜色贡献、其他颜色贡献，以及仅将目标高斯尺度乘以 0.5 或 0.25 的结果。全部 48 张图片独立核验；原始观测与规定缩放的源照片逐像素一致，完整渲染也与已保存结果逐像素一致。

![训练视图的目标人物与其他场景贡献分解](../outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2/renders/frame129-camera4-contributions.png)
> 图源：Argoverse 2 原始观测与本站实际渲染 · 帧 129、相机 4 · 同一区域，贡献图按显示范围截断，未锐化；原始数据 CC BY-NC-SA 4.0。

上排右侧移除目标后显露的是邻近的另一个行人，不能直接称作目标残影。下排中间保留所有对象的不透明度，只显示目标对最终颜色的贡献，人物自身仍有柔软边界和不清晰肢体；下排右侧也保留了地面与背景模糊，不能把全图模糊全部归因于目标人物。下排左侧是模型的可见权重，不是人体真值掩码。

目标共有 6,890 个高斯，最大主轴标准差的中位数约 2.14 毫米、95 分位约 3.33 厘米、最大约 21.78 厘米。帧 130／相机 4 的光栅化包围半径中位数为 3 像素、最大 98 像素，21 个高斯超过 50 像素。包围半径只是计算覆盖范围，并不等于实际轮廓宽度，也不保证该高斯未被遮挡。

![目标尺度缩小后的真实留出渲染对照](../outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2/renders/frame130-camera4-scales.png)
> 图源：Argoverse 2 原始观测与本站只读诊断 · 帧 130、相机 4 · 未改变颜色、中心、不透明度和姿态；未训练、未锐化。

直接缩小所有目标高斯会暴露密集斑点与孔洞，未恢复原图衣褶。此图人物区域 PSNR 从 19.35 分别降到 18.67、18.08 dB；六个检查视图的 PSNR 与 SSIM 均下降。因此不采用统一缩小作为修复。它同时改变光学覆盖，又没有调整不透明度、采样密度与颜色；这个负结果不能证明尺度学习无用，而是说明表面位置、尺度、颜色和覆盖需要受约束地联合拟合，并保留同预算的颜色拟合对照。

最初尝试直接相加已截断的贡献图时，严格一致性检查失败，记录已保留。修正版保存截断前的浮点贡献，按原实现先截断高斯颜色、再加入天空的顺序重组；六个视图的重组误差均小于 0.00002，独立计算通过。网站不提供浮点数组、模型或 SMPL 模板。

证据：[事先固定的诊断协议](../outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2/protocol.json)、[实际尺度与逐视图测量](../outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2/completion.json)、[48 张图片独立核验](../outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2/independent-image-audit.json)、[六组看图范围与结论](../outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2/visual-review.json)。

## 固定姿态后的表面与颜色对照

以长预算姿态模型为共同起点，保持姿态、体素蒙皮、背景、其他人物、相机、曝光和 6,890 点拓扑不变。一组只拟合目标人物的球谐颜色系数；另一组同时拟合颜色、高斯中心、尺度、旋转和不透明度。两组均使用全部 101 个人物训练区域，每次取平均损失，各执行 100 次更新，对应各 10,100 次区域反向计算。参数数量与运行时间不同，不称为相同算力预算。模型仅按完整训练目标选择，所有 105 张留出图仍不参与优化。

表面组相对起点的中心位移限制为 3 厘米，主轴标准差比值限制为 0.5～2，旋转变化限制为 20°，并对变化幅度增加先验。这是多个高斯属性的联合拟合对照，不能将其变化全部归因于尺度或某一个正则项。

颜色组已完成并通过全部 105 张留出 PNG、435 个人物区域和模型状态的独立核验。训练人物区域平均 PSNR 从 20.40 提高到 20.67 dB；主目标 13 个留出框从 19.94 提高到 20.10 dB，SSIM 从 0.595 提高到 0.602。实际查看 110／2、130／4、140／4、150／6 四组近处裁图，衣物、手提袋与腿部仍模糊。81 个非颜色模型张量条目以及非目标人物的颜色精确不变，因此这轮小幅改善来自限定范围内的颜色拟合，尚未通过清晰度验收。

表面组也已完成相同的 100 次更新。训练人物区域平均 PSNR 达到 20.87 dB，主目标 13 个留出区域为 20.30 dB、SSIM 为 0.609；对应起点为 19.94 dB、0.595，颜色组为 20.10 dB、0.602。105 张留出 PNG 与 435 个区域通过独立误差复算；77 个冻结模型张量条目和所有非目标人物行精确不变。目标中心位移中位数为 7.17 毫米、最大 28.68 毫米，尺度比值在 0.508～1.987 内，最大旋转变化为 19.07°，均满足预设范围。这些量是相对起点的参数变化，不是真实表面误差。

![同一起点的颜色拟合与表面联合拟合，近处侧面仍缺少衣物细节](../outputs/dynamic-human-street-surface-refit-018/paired-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站真实留出渲染 · 帧 130、相机 4 · 后两组各额外 100 次全区域更新；同倍率，未锐化；原始数据 CC BY-NC-SA 4.0。

四列依次为原始观测、姿态优化后的起点、仅颜色拟合、表面与颜色联合拟合。左图清楚记录了衣物褶皱、提袋和腿部边界，右侧三种模型仍未重现这些结构。表面组改变了局部明暗与覆盖，却仍有柔软边缘、斑点和偏移，不能将更小的平均误差称为已经得到清晰衣物。

![人物接近时的真实留出对照，双腿与衣褶仍模糊](../outputs/dynamic-human-street-surface-refit-018/paired-audit/frame110-camera2-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站真实留出渲染 · 帧 110、相机 2 · 同一区域，保留原图匿名化，未锐化。

本次已逐一查看全部 13 组配对裁图，接近、侧面和后方均未通过人物清晰度验收；没有把最佳视角代替全部检查。两组均按训练目标选择第 100 次结果，训练目标仍在下降，不能称为已收敛。后续连续帧对照见下节。

首次表面尝试在任何更新前被起点一致性检查拦下；修正版保留四元数符号，并在实际检查点上验证初始精确一致、有限且非零梯度、物理范围与精确回退，颜色组结果未重跑或覆盖。修复的作用是保证对照从完全相同的模型开始，并非人物清晰度改善的证据。

证据：[共同训练协议](../outputs/dynamic-human-street-surface-refit-018/protocol.json)、[颜色组图像核验](../outputs/dynamic-human-street-surface-refit-018/appearance_only/independent-image-audit.json)、[颜色组冻结范围核验](../outputs/dynamic-human-street-surface-refit-018/appearance_only/independent-state-audit.json)、[四组实际看图范围](../outputs/dynamic-human-street-surface-refit-018/appearance_only/visual-review.json)、[起点检查失败记录](../outputs/dynamic-human-street-surface-refit-018/parameterization-initial-failure.json)、[修正后的参数检查](../outputs/dynamic-human-street-surface-refit-018/parameterization-sign-preserving-smoke.json)、[表面组重启协议](../outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/protocol.json)。

表面组新增证据：[全部图像与区域核验](../outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/surface_and_appearance/independent-image-audit.json)、[冻结范围与物理边界核验](../outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/surface_and_appearance/independent-state-audit.json)、[全部十三组实际视觉检查](../outputs/dynamic-human-street-surface-refit-018/paired-audit/visual-review.json)。

## 表面拟合后的连续运动对照

固定原有四个时间窗口，从姿态起点、颜色候选和表面候选分别恢复全部模型张量，各渲染 80 个相同的帧／相机组合。三支共 240 张图、228 组相邻差分通过独立复算；重叠留出图的重新渲染每通道差异不超过一个 8 位灰度级，渲染前后模型张量精确不变。此处没有再次训练。

人物固定区域视频：[表示切换前后](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/pose-onset-fixed-union.mp4)、[人物接近](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/approach-fixed-union.mp4)、[侧面与遮挡](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/side-occlusion-fixed-union.mp4)、[背面与遮挡](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/rear-occlusion-fixed-union.mp4)。对应完整画面：[切换](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/pose-onset-full-image.mp4)、[接近](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/approach-full-image.mp4)、[侧面](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/side-occlusion-full-image.mp4)、[背面](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/rear-occlusion-full-image.mp4)。四列保持同倍率、原拍摄约 10 Hz 节奏，没有锐化或生成式增强。

侧面固定区域的平均 PSNR 从起点 23.10 dB 提高到颜色组 23.25 dB、表面组 23.28 dB，相邻变化残差从 0.04759 分别变为 0.04701、0.04675。背面区域的表面组改善更小，PSNR 仅从 18.63 到 18.64 dB。这些区域覆盖背景与其他人物，不能与前述逐帧紧框的 13 视图均值直接比较；相邻差分还包含真实运动、相机移动和光照，不能把数值下降称为消除了闪烁。

四个窗口各抽看 4 帧，共 16 组：训练帧 108／117 的衣物与双腿仍模糊，侧面训练帧 124／136 和留出帧 130 也保留破碎轮廓、弥散提袋，背面没有恢复清楚的衣物表面。全部视频完成逐帧解码检查，但人工查看范围是这 16 组分帧对照图及上一节的全部 13 组留出紧框，未宣称每帧均完成视觉验收。第 56 帧之前的目标使用未改变的可变形分支，远处表示切换窗口不能证明近处衣物清晰。

这一轮确认了平均误差的有限改善，也保留了训练时刻仍不清晰的事实。下一步用固定的近处训练视图检查表面本身的拟合能力，再与跨时间共享表面的拟合比较；该诊断不能代替独立新视角和连续运动验证。

证据：[固定时间窗口](../outputs/dynamic-human-street-surface-temporal-019/protocol.json)、[模型恢复与只读渲染条件](../outputs/dynamic-human-street-surface-temporal-019/render-protocol.json)、[240 张图与 228 组差分独立核验](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/independent-audit.json)、[实际看图范围与结论](../outputs/dynamic-human-street-surface-temporal-019/independent-audit/visual-review.json)。

## 单帧与相邻双帧的表面拟合能力

从同一个长预算姿态模型出发，保留上一轮表面参数化、3 厘米位移约束、0.5～2 尺度比和 20° 旋转界限，冻结姿态、蒙皮、背景、其他人物与 6,890 点拓扑。一组仅拟合训练帧 129／相机 4，另一组拟合训练帧 129、131／相机 4；每组 1,000 次 Adam 更新，分别进行 1,000 与 2,000 次区域反向计算。两组更新次数与参数量相同，观测次数和计算量不同。模型分别按本组训练目标选择第 1,000 次与第 973 次状态，没有依据留出效果选择。

![局部训练画面的衣物拟合能力](../outputs/dynamic-human-street-local-capacity-020/paired-audit/frame129-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站真实训练视图 · 帧 129、相机 4 · 后两组各额外 1,000 次更新；未锐化。

第 129 帧人物区域 PSNR 从起点的 19.89 dB 提高到单帧组的 26.80 dB，衣褶、袖口与提袋边缘明显恢复。双帧组在第 129、131 帧分别达到 25.95、25.64 dB，两张训练画面均能显示更多服装结构。因此，当前表示并非完全不能表达这些细节；这不等于恢复了真实三维表面，脚部、地面与遮挡仍不完美。

![单帧拟合损害相邻画面，双帧拟合恢复被监督的另一帧](../outputs/dynamic-human-street-local-capacity-020/paired-audit/frame131-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实际渲染 · 帧 131、相机 4 · 此帧只参与最右列的局部拟合。

单帧组在未进入本次损失的第 131 帧出现彩色斑点与破碎肢体；该帧人物区域从 19.61 降到 17.75 dB。双帧组能改善两张被监督画面，但中间留出帧 130 仍有明显的衣物与提袋伪影。其 PSNR 虽从 19.35 提高到 21.31 dB，实际外观仍未通过，不能用平均颜色误差代替清晰度判断。

![两帧之间的留出时刻仍然失败](../outputs/dynamic-human-street-local-capacity-020/paired-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实际留出渲染 · 帧 130、相机 4 · 未参与局部训练。

全部 13 个人物留出框的平均 PSNR 从 19.94 dB 分别降到单帧组 16.71 dB、双帧组 15.77 dB；全部 101 个原训练区域的均值也从 20.40 分别降到 17.10、15.99 dB。查看 110／2、140／4、150／6 时，出现严重彩色与破碎表面伪影，因此两组都保留为诊断，不替换原模型。当前结论是局部观测可以拟合得清楚，而跨时刻共享仍失败；中心、尺度、旋转、不透明度和球谐颜色同时变化，尚不能单独确定哪个因素造成泛化退化。

两组所有 210 张留出 PNG、870 个区域，保存的训练图片和中途快照通过独立复算；77 个冻结模型张量条目、其他人物行与全部姿态精确不变。人工检查范围为 2 组训练画面和 4 组留出画面，不宣称全部 13 组或连续视频均已视觉验收。

训练及评测完成后，汇总程序因局部变量重名在末尾的源模型哈希检查处停止；两组检查点和完整评测均已保存。独立程序重新核对源哈希、模型边界、训练预算与图片后完成汇总，没有重训或覆盖候选，原失败记录保留。

证据：[事先固定的诊断协议](../outputs/dynamic-human-street-local-capacity-020/protocol.json)、[单帧图像核验](../outputs/dynamic-human-street-local-capacity-020/single_view/independent-image-audit.json)、[单帧模型核验](../outputs/dynamic-human-street-local-capacity-020/single_view/independent-state-audit.json)、[双帧图像核验](../outputs/dynamic-human-street-local-capacity-020/adjacent_pair/independent-image-audit.json)、[双帧模型核验](../outputs/dynamic-human-street-local-capacity-020/adjacent_pair/independent-state-audit.json)、[实际看图范围](../outputs/dynamic-human-street-local-capacity-020/paired-audit/visual-review.json)、[汇总失败与恢复](../outputs/dynamic-human-street-local-capacity-020/finalization-failure.json)。

## 局部拟合中的颜色、几何与不透明度

对源模型 016 和双帧候选 020 进行七种只读参数组合，评测相同的 2 张局部训练图与全部 13 张目标留出图，共保存 105 张全图和 105 张对应裁图。冻结所有人体姿态、蒙皮、背景、其他人物与相机；只允许目标 6,890 个高斯的指定属性来自候选。没有优化步骤，也没有选择替代模型。

![训练视图中的几何与颜色替换](../outputs/dynamic-human-street-capacity-components-022/paired-audit/frame129-camera4-geometry-color.png)
> 图源：Argoverse 2 与本站真实参数替换 · 帧 129、相机 4 · 保留匿名化；未训练、未锐化。

源模型的两张训练图均值为 19.75 dB，局部完整候选为 25.80 dB。只换候选颜色为 19.73 dB，只换几何与不透明度为 20.78 dB；后者已有部分衣褶结构，但两种拆分均不能保留完整候选的清晰外观。完整候选中仅恢复源不透明度后，训练均值降至 20.10 dB，说明遮挡权重与颜色、表面已经共同适配这两张观测。

![其他时刻不能仅靠恢复方向相关颜色修复](../outputs/dynamic-human-street-capacity-components-022/paired-audit/frame140-camera4-color-opacity.png)
> 图源：Argoverse 2 与本站真实留出渲染 · 帧 140、相机 4 · 相同区域与倍率。

13 张留出图均值：源模型 19.94 dB，完整候选 15.77 dB，只换颜色 17.24 dB，只换几何与不透明度 18.48 dB；两种拆分均比源模型差。完整候选恢复源方向相关颜色后为 16.66 dB，恢复源零阶颜色后为 17.31 dB，恢复源不透明度后为 16.11 dB。这里使用一阶球谐，不能将结果解释为三阶球谐过拟合；联合参数的替换也不是独立训练消融，不能把各组差值相加作为原因占比。

逐像素复核全部 105 张全图与裁图；原始目标按相同缩放流程逐像素一致，源模型及完整候选的 30 张重现图与既有结果最大差不超过 1 个 8 位灰度级。人工查看 7 张完整组合图，覆盖 129／4、131／4 两张训练图，以及 110／2、130／4、140／4、150／6 四张留出图。训练图中的拟合相互依赖、留出图中的彩色与破碎外观均可见；没有进行新的连续视频验收。

证据：[参数组合与固定范围](../outputs/dynamic-human-street-capacity-components-022/protocol.json)、[逐图指标](../outputs/dynamic-human-street-capacity-components-022/results.json)、[冻结状态与精确恢复](../outputs/dynamic-human-street-capacity-components-022/tensor-audit.json)、[独立图像复核](../outputs/dynamic-human-street-capacity-components-022/independent-image-audit.json)、[实际视觉检查](../outputs/dynamic-human-street-capacity-components-022/paired-audit/visual-review.json)。

## 共同优化姿态表面与颜色

从同一个长预算姿态模型出发，沿用前述表面组的 101 个训练区域、6,890 个目标高斯、图像损失、表面参数化和 100 次 Adam 更新；额外开放目标人物 90 个有效训练时刻的整体平移、整体旋转与关节旋转，并加入姿态幅度和修正变化的约束。表面组使用已完成的对照结果，没有重新训练。两组各执行 10,100 次区域反向计算；新增姿态参数与正则使计算量和参数量不同。

相对共同起点，平移修正限制为 10 厘米，整体旋转为 10°，关节旋转为 20°；高斯中心位移仍限制为 3 厘米，尺度比为 0.5～2，旋转为 20°。背景、其他人物、相机、曝光、体素蒙皮与点数保持不变。存储的留出姿态未被优化，但官方渲染器会在条件满足时由相邻训练姿态插值得到留出姿态，因此“留出姿态张量未变”不意味着留出画面的几何不变。

训练仅按完整训练目标选择第 100 次状态；总目标从 0.12235 降至 0.10731，末段仍在下降，不称为已经收敛。101 个训练区域平均 PSNR 为共同起点 20.40 dB、表面组 20.87 dB、联合组 21.32 dB。13 个目标留出区域对应为 19.94、20.30、20.43 dB；联合组相对表面组有 9 个区域提高、4 个下降，SSIM 均值从 0.6089 变为 0.6099。这是已多次检查过的探索性划分，不是未触碰的最终测试集。

![共同优化后近处训练画面的衣物仍不清楚](../outputs/dynamic-human-street-joint-pose-surface-024/paired-audit/frame129-camera4-training-comparison.png)
> 图源：Argoverse 2 与本站真实训练渲染 · 帧 129、相机 4 · 原图、共同起点、表面组、联合组；同倍率，未锐化。

![中间留出时刻的同区域比较](../outputs/dynamic-human-street-joint-pose-surface-024/paired-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 与本站真实留出渲染 · 帧 130、相机 4 · 保留原图匿名化；原始数据 CC BY-NC-SA 4.0。

逐一查看全部 13 组留出裁图和 129／4、131／4、149／6、151／6 四组训练裁图：近处袖口与局部明暗有所变化，但衣褶、手提袋和腿部边界仍然弥散，后方衣物也没有恢复清楚。远处样本本身像素较少，不能据此认定细节达标。本候选保留为联合优化诊断，没有作为清晰场景模型替换原模型。

全部 105 张留出 PNG、435 个人物区域及保存的 8 张优化前后训练画面通过独立复算；74 个冻结模型张量条目及全部非目标行精确不变。新增平移修正最大 4.15 厘米、整体旋转最大 3.67°、关节旋转最大 15.88°，均在预设界限内；这些数值是参数变化，不是真实姿态误差。初次尝试在 JSON 索引序列化阶段停止，当时尚未执行优化；原失败保留，修正后从同一起点重新运行。

按冻结的区域选择规则，90 个训练时刻中只有 11 个包含两个相机区域，另外 79 个只有一个。这限制了实际用于损失的同时刻视角约束，但没有证明人物在其他原图完全不可见，也未单独验证它是模糊的唯一原因。后续需要在真实同步多视角数据中检验共享表面、姿态与可见性，而不是把本次小幅指标提升当作目标完成。

证据：[训练协议](../outputs/dynamic-human-street-joint-pose-surface-024/protocol.json)、[模型选择](../outputs/dynamic-human-street-joint-pose-surface-024/training-selection.json)、[冻结参数与变化范围](../outputs/dynamic-human-street-joint-pose-surface-024/independent-state-audit.json)、[全部图像独立复核](../outputs/dynamic-human-street-joint-pose-surface-024/independent-image-audit.json)、[实际看图范围](../outputs/dynamic-human-street-joint-pose-surface-024/paired-audit/visual-review.json)、[训练视角覆盖](../outputs/dynamic-human-street-joint-pose-surface-024/training-view-support.json)。

### 联合优化后的连续运动检查

沿用原有四个窗口与全部 80 个原始视图，新增渲染联合组 80 张图，另复用已核验的起点和表面组共 160 张图；复用前检查对应模型与图像哈希，没有重训对照。三组共 240 张图、228 组相邻差分通过独立复算，8 段完整画面与固定区域视频均完成逐帧解码，按原采集约 10 Hz 播放。

侧面窗口固定区域的平均 PSNR 从表面组 23.28 dB 到联合组 23.56 dB，相邻变化残差从 0.04675 到 0.04586；背面窗口为 18.64 到 18.65 dB。这些矩形区域包含背景与遮挡者，不能与逐帧人物紧框均值混用，也不能把差分残差下降直接称为消除了闪烁。

实际查看四个窗口的固定区域及全图分帧对照，共 8 张图、16 个采样时刻：近处训练帧 108、117、124、136 的衣物与肢体仍模糊，留出帧 130 和背面 150 也没有恢复清晰提袋与衣褶。远处或人物正在出画的采样不用于证明细节达标；视频逐帧解码不等于已逐帧完成视觉验收。

固定区域视频：[表示切换前后](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/pose-onset-fixed-union.mp4)、[人物接近](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/approach-fixed-union.mp4)、[侧面与遮挡](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/side-occlusion-fixed-union.mp4)、[背面与遮挡](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/rear-occlusion-fixed-union.mp4)。对应完整画面：[切换](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/pose-onset-full-image.mp4)、[接近](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/approach-full-image.mp4)、[侧面](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/side-occlusion-full-image.mp4)、[背面](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/rear-occlusion-full-image.mp4)。

证据：[连续窗口](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/protocol.json)、[对照图像复用核验](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/cached-controls.json)、[图像与视频独立核验](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/independent-audit.json)、[实际视觉检查范围](../outputs/dynamic-human-street-joint-pose-surface-024/temporal/independent-audit/visual-review.json)。

## 更清晰的室外人物观测：数据接入状态

核对 AV2 官方说明后，确认其发布图像已经匿名化，不能把下载原始日志理解为获取未经模糊的人脸。ENeRF-Outdoor 的官方匿名下载此前受 Google Drive 配额限制；用户现已完成三组完整压缩包下载，实际大小与官方目录记录一致。三组均已完成 gzip 完整性检查、逐成员读取和哈希记录，共 130,252 个普通文件、约 37.1 GB。当前训练使用第一组 `actor1_4`；另外两组已解压，其中 `actor2_3` 的一个训练机位抽查可见抬腿、弯腰和转身，尚未完成相机与训练核验。

实际数据包含 18 个机位、每机位 1,200 张 1920×1080 图像、作者前景掩膜与标定。首段 60 帧共 1,080 张图像及相应掩膜已读取和校验。冻结划分为 810 张训练图、162 张留出机位图和 108 张留出时刻图：机位 00、06、12 不参与训练；时刻 5、15、25、35、45、55 的全部机位不参与训练。

标定文件中宽、高为 −1，依照 EasyVolcap 的读取约定表示未指定；实际尺寸来自图像，内参不缩放。图像和掩膜使用同一原生尺寸、同一内参去畸变，无效边缘不进入损失。25 对训练相机的前景 SIFT 匹配经极线距离、正深度、三角化夹角和重投影筛选，保留 6,453 个三维点；各对保留点的重投影误差中位数为 0.14～0.32 像素。这支持首帧相机投影的一致性，尚不独立证明物理尺度或全序列硬件同步。

本地数据接入记录保留在固定实验目录；原始数据不作为网站公开下载资源。按数据提供方要求引用 [4K4D](https://zju3dv.github.io/4k4d/) 与 [EasyVolcap](https://github.com/zju3dv/EasyVolcap)。另见[官方小样本说明](https://github.com/zju3dv/EasyVolcap#examples)与[完整数据访问条件](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md)。

### 原生分辨率首帧重建

先用 15 个训练机位构建轮廓交集，允许膨胀 3 像素后的掩膜在一个机位不一致；以间隔 0.012 个发布坐标单位的体素表面初始化 39,745 个高斯。初次包围盒触及 232 个占据体素，在任何训练前停止；扩展边界后重新检查，体积不触边才开始优化。没有使用作者发布的表面点云或留出掩膜初始化。

采用 gsplat 1.3 的 3DGS 密度控制和二阶球谐颜色，完成预定 6,000 次更新，最终 143,412 个高斯。优化中心、尺度、旋转、不透明度和颜色；保持原生像素，裁剪人物外围后同步移动主点。图像损失为 0.8 倍 L1、0.2 倍 DSSIM 与 0.1 倍不透明度误差，随机纯色背景用于前景合成。它是本地首帧诊断，不是官方 4K4D 复现，也没有学习静态墙面和地面。

![留出机位 00 的首帧前景对照](../outputs/dynamic-human-enerf-native-025/first-frame/evaluation/heldout_view-camera00-comparison.png)
> 图源：ENeRF-Outdoor 与本站真实高斯渲染 · 左原始前景，右未参与训练机位 · 无锐化或生成增强。

![保留侧面文字和轮廓残差](../outputs/dynamic-human-enerf-native-025/first-frame/evaluation/heldout_view-camera06-comparison.png)
> 图源：ENeRF-Outdoor 与本站真实高斯渲染 · 留出机位 06、首帧。

3 个留出机位的前景 PSNR 分别为 27.70、25.94、26.59 dB；此处前景指作者掩膜大于 0.9 且重映射有效的像素，包含手持物，不能与此前 AV2 的矩形区域分数直接比较。已查看 3 个机位全部 6 个人物裁图：衣服文字、条纹、裤腿和鞋子主要结构可辨认，侧面文字仍有重影，边缘有残差。灰色是检视背景，不是重建场景。

独立 CPU 复核全部 6 张评测视图（3 个留出、3 个训练）的保存图像，重建参照与原始图像／掩膜重新计算逐像素一致；PSNR 量化前后差异小于 0.03 dB，12 个人物框的 SSIM 与另一实现一致。15 份训练缓存逐值对应原生数据。见[首帧训练条件](../outputs/dynamic-human-enerf-native-025/first-frame/protocol.json)、[首帧逐视图结果](../outputs/dynamic-human-enerf-native-025/first-frame/evaluation/metrics.json)、[独立核验](../outputs/dynamic-human-enerf-native-025/first-frame-independent-audit.json)。

### 连续运动与留出时刻

由首帧模型出发，固定高斯身份、数量、尺度和不透明度，逐时刻优化中心、旋转与球谐颜色。采用 Dynamic 3D Gaussians 的局部刚性、邻域旋转与等距损失形式，每步均匀采样 20,000 个中心计算 20 邻域约束；保留前一状态的独立副本，并按实际帧号间隔初始化运动。由于只有前景，未应用原代码的地面和静态背景项。这是本地适配，不是作者完整实现的等价复现。

其余 53 个训练时刻各完成 600 次更新，共 31,800 次；6 个留出时刻由相邻训练状态插值，不能称为未来预测。162 个留出机位图的前景 PSNR 均值为 25.21 dB，108 个留出时刻图为 26.74 dB；后者包含训练机位，二者不是同一难度的分数。将首帧模型冻结到所有时刻，只作为运动必要性的参照：在相同 3 个留出机位、60 个时刻上，冻结首帧为 14.73 dB，连续模型为 25.19 dB。前者没有进行多时刻静态拟合，不能把这个差值当作等训练预算的方法比较。

![连续模型末帧保留躯干但手和转动物品仍破碎](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera06-frame059-comparison.png)
> 图源：ENeRF-Outdoor 与本站真实渲染 · 留出机位 06、帧 59 · 左原始前景，右连续模型。

实际查看 3 个留出机位在帧 15、30、45、59 的分帧对照，以及首帧全部 6 个人物裁图；另检查训练机位 01、07 的帧 21 和机位 00 末帧的完整画面。躯干文字、条纹与腿部仍能辨认，手部和转动物品的重影、破碎随动作增加。完整画面右上方还出现漂浮伪影，人物紧框会漏掉这一问题，因此独立核验同时计算全画面误差和人物范围外的颜色残差。没有通过最终清晰度验收。

全部 270 张留出渲染的原图合成参照与数据重新读取结果逐像素一致，保存图像 PSNR 独立复算一致；180 张冻结首帧参照也完成复核。54 个保存状态全部有限、形状一致，尺度和不透明度逐值不变。3 段对照视频均为 60 帧，完成逐帧解码；人工视觉范围是上述采样，不宣称每帧均已看图。视频按 30 帧每秒供检查，原采集帧率仍未独立核实。

连续视频：[机位 00](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera00-motion-comparison.mp4)、[机位 06](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera06-motion-comparison.mp4)、[机位 12](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera12-motion-comparison.mp4)。[保留漂浮伪影的完整渲染](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera00-frame059-render.png)、[训练条件](../outputs/dynamic-human-enerf-native-025/motion/protocol.json)、[全部留出分数](../outputs/dynamic-human-enerf-native-025/motion/evaluation/metrics.json)、[图像与视频核验](../outputs/dynamic-human-enerf-native-025/motion-independent-audit.json)、[参数状态核验](../outputs/dynamic-human-enerf-native-025/motion-state-audit.json)、[实际视觉检查](../outputs/dynamic-human-enerf-native-025/motion-visual-review.json)。

### 困难末帧能否独立重建清楚

预先取同一段的末帧 59，用该帧的 15 个训练机位重新构建视觉外壳，并独立完成 6,000 次 3DGS 更新。该对照开放密度控制、尺度与不透明度，不使用前一时刻或时序约束；最终 148,775 个高斯。因为初始化、参数、迭代分配和时间约束同时变化，它是拟合能力诊断，不是单项消融。

![同一区域的原始观测、连续模型与独立末帧重建](../outputs/dynamic-human-enerf-native-025/comparison/camera06-frame059-capacity-comparison.png)
> 图源：ENeRF-Outdoor 与本站真实渲染 · 留出机位 06、帧 59 · 三列同倍率、无增强。

3 个留出机位的前景分数分别由连续模型的 24.48、23.62、22.45 dB 变为独立模型的 23.85、22.65、23.16 dB，没有一致改善。三个机位的第二个人物均已查看：手、物品和袖口依旧有破碎。独立复核 15 份训练输入、6 张评测图像与 12 个人物框，原图参照和指标一致。最初的数据准备虽读取了帧 59，却仍写入硬编码帧号 0；该准备目录未用于训练，保留后在修正的目录重新生成并验证了帧号。

因此，时序累积不足以单独解释当前失败；实际运动区域的跨机位对应、时间一致性、可见性与表面优化仍需分别验证，不能据此直接认定数据不同步。见[困难帧对照与条件](../outputs/dynamic-human-enerf-native-025/comparison/capacity-summary.json)、[独立末帧图像核验](../outputs/dynamic-human-enerf-native-025/last-frame-capacity-control-v2/first-frame-independent-audit.json)。

### 运动区域的跨相机时间对应

在原有 15 个训练机位中，按相机中心的世界横坐标排序，连接相邻三个位置，共 39 对相机。帧 9、29 用于估计相对帧索引，帧 49 用于独立检查。全部观测来自冻结训练划分；候选偏移为 −3～3 帧，未读取机位 00、06、12 或任何留出时刻。

匹配采用前景内的双向 SIFT，对极距离不参与初始筛选。同一目标特征通过前后向光流跟踪到全部候选时刻，各候选共享相同的特征集合。固定原始内外参，只比较 Sampson 距离，避免通过重新选择匹配或重新拟合基础矩阵降低残差。117 份轨迹文件、20,771 个匹配点均独立重算了几何距离；人工查看四对机位的 24 组原生图像块，涵盖手、玩偶、衣物和手臂。该人工范围不代表全部匹配均已确认正确。

将各相机的整数偏移限制在 −2～2，以机位 08 为零，用图上的统一相机偏移保证成对结果的循环一致性。宽度为 3 的有序相机图通过动态规划求得该有限候选空间的最小值；这不代表真实物理时钟已标定。独立帧 49 的各相机对运动点距离中位数，再对 39 对求均值，由 1.307 降到 0.575 像素；19 对使用非零相对偏移，其中 18 对改善、1 对恶化。

证据：[固定匹配与筛选条件](../outputs/dynamic-human-enerf-correspondence-026/graph-neighbors3/protocol.json)、[整数相机偏移与独立检查](../outputs/dynamic-human-enerf-correspondence-026/graph-neighbors3/global-offset-hypothesis.json)、[图像块对应示例](../outputs/dynamic-human-enerf-correspondence-026/identity-09-01-t049.png)。

### 只改变相机取帧编号的训练对照

选择共同索引 49，两组均使用 15 个训练机位、原生 1920×1080 输入、同一份视觉外壳初始点与颜色、种子 73、同样的密度控制和 6,000 次更新。对照组使用全部机位的第 49 帧，另一组依照前述训练数据估计的偏移，分别读取第 48、49 或 50 帧；这些源图均属于原训练划分。调整组复用对照组的初始化，没有额外获得重新构建视觉外壳的优势。

两组分别得到 152,248 和 151,770 个高斯。算法规则与预算相同，密度控制产生的最终数量可以不同。两支全部 30 份训练输入逐像素对回原始解码结果，12 张评测图和 24 个人物框分别复算；保存图像的 8 位量化通过逐像素舍入误差范围核验，未用固定的 dB 容差掩盖高分数下的量化差异。

![留出机位06中手部和玩偶的真实取帧对照](../outputs/dynamic-human-enerf-time-control-027/comparison/camera06-person2-upper-body.png)
> 图源：ENeRF-Outdoor 与本站真实渲染 · 第 49 帧 · 原始前景、同编号输入、按相机调整编号；同一原生像素区域。

固定原有留出照片不变，机位 00、06、12 的前景 PSNR 分别为 25.73→25.07、25.07→25.94、24.85→25.08 dB，平均差仅 +0.15 dB。机位 01 的训练参照始终是原始第 49 帧，其分数由 28.24 提高到 30.36 dB。机位 07 的监督由第 49 改为第 48 帧，不能把两张不同参照的训练分数当成严格配对改善。

![另一个留出机位仍保留衣物和手部误差](../outputs/dynamic-human-enerf-time-control-027/comparison/camera00-person1-upper-body.png)
> 图源：ENeRF-Outdoor 与本站真实渲染 · 机位 00、第 49 帧 · 展示误差恶化的机位，未筛去不利结果。

全部三个留出机位的六组上半身裁图及调整组的三张完整画面均已查看，另查看训练机位 01 的第二个人物。机位 06 的手和玩偶破碎有所减少，但指间细节、衣服文字和侧边漂浮残留仍在；其他机位没有一致改善。本轮不替换连续模型，也没有通过清晰度验收。留出相机自身的相对时间未估计，不能用这三张未调整的参考图证明全部相机时间已经一致。

证据：[相同初始化与来源核验](../outputs/dynamic-human-enerf-time-control-027/control-protocol-and-audit.json)、[逐机位配对结果](../outputs/dynamic-human-enerf-time-control-027/comparison/summary.json)、[对照组独立核验](../outputs/dynamic-human-enerf-time-control-027/same-index/first-frame-independent-audit.json)、[调整组独立核验](../outputs/dynamic-human-enerf-time-control-027/offset-index/first-frame-independent-audit.json)、[机位06完整渲染](../outputs/dynamic-human-enerf-time-control-027/offset-index/first-frame/evaluation/heldout_view-camera06-render.png)、[实际看图范围](../outputs/dynamic-human-enerf-time-control-027/visual-review.json)。

### 时间偏移在其他运动片段中能否成立

另外准备了完全不与第 0～59 帧重合的时间标定数据。18 个相机都可用于这一专门任务，原来的留出机位因此属于“标定可见、外观拟合留出”，不再称为全局未见相机。数据适配器会拒绝把此类标定清单直接交给外观训练。

第一组用帧 109、129 估计、149 检查，各自保留前后 3 帧。第 149 帧只有 3／48 对相机具备至少 12 个可靠运动匹配，无法验证整个相机系统。第二组先仅用相机 08 扫描第 80～1180 帧，按垂直图像运动与最小时间间隔选出 460、640、860，再冻结前两者用于估计、第三者检查。虽然单相机运动充分，同时通过两段跨相机匹配筛选的边只剩 13／48 条，相机图分成 8 个互不连接的分量。

互不连接的分量之间没有时间参照，数值求解仍可能给出一组看似完整、实则任意的相对偏移。已为求解器加入可辨识性检查，拒绝这种全局结果；最初未检查连通性得到的数值候选作为失败记录保留，未应用于图像或模型。两组共 756 张原始照片、对应掩膜及 288 份匹配轨迹完成来源与几何距离的独立复核。

当前结论是：局部取帧调整可改善部分手部外观，但尚无证据支持整段视频采用同一组固定偏移。后续应改进运动部位的跨相机对应，或建立明确划分标定图像与外观测试图像的局部时间协议；不能把失败候选直接用于连续训练。[分离时间标定与拒绝原因](../outputs/dynamic-human-enerf-time-calibration-028/independent-audit-and-decision.json)、[运动片段的选择规则](../outputs/dynamic-human-enerf-time-calibration-028/motion-selection.json)、[不连通相机图的检查](../outputs/dynamic-human-enerf-time-calibration-028/observable-motion/graph-neighbors3/global-offset-hypothesis.json)。

### 局部时间对应与连续人物的同预算对照

重新冻结一组明确区分任务的数据协议：18 个机位在原始帧 6～12、16～22、26～32 中参与时间对应检查，以共同索引 9、29 估计，19 检查；这些照片禁止进入外观训练。相机图连通，独立检查中 47 对有效相机的运动点 Sampson 距离中位数均值从 1.487 降到 0.739 像素，27 对改变相对帧号的相机中 24 对改善。通过事先指定的连通性、至少 20% 残差下降和至少 75% 改变对改善的条件后，才固定局部取帧规则。378 张标定图、对应掩膜与 144 份轨迹共 29,392 个匹配完成独立核验。它仍是局部图像编号对应，不是硬件时钟标定。

外观重建使用共同索引 34～58，共 25 个连续时刻。每组 15 个训练机位提供 375 张照片，机位 00、06、12 提供相同的 75 张外观留出参照；机位 12 的参照读取共同索引加 1 的原始图片。两组动态输入分别读取相同编号或按相机偏移调整的编号，均评估同一批参照。所有机位曾在更早数据上参与时间对应检查，因此不是全局未见相机；本轮不保留此前的时间留出，也不复用此前的时间留出成绩。这是已检查过的开发序列上的新对照，不是未接触过的最终测试集。

两支从完全相同的 42,336 个视觉外壳初始高斯开始，各进行首帧 6,000 次更新，再对后续 24 个时刻各更新 600 次，总计各 20,400 次。静态组复用调整取帧组的首帧检查点，再在全部 375 张相同观测上进行 14,400 次更新，同样总计 20,400 次。静态组允许尺度、不透明度与密度更新；动态组在首帧后固定数量、尺度、不透明度，优化中心、旋转和球谐颜色，保留局部刚性、邻域旋转、等距与颜色连续性约束。因此这是同更新预算的不同表示对照，不等于同计算量或只改变一个参数开关。

训练均保留原生 1920×1080 像素，裁掉人物外围空白并同步移动主点。图像项实际采用 0.8 倍 L1、0.2 倍 $1-\mathrm{SSIM}$ 与 0.1 倍不透明度误差，随机纯色背景；这里明确给出 $1-\mathrm{SSIM}$，避免运行文件中的 DSSIM 简称引起二分之一系数的歧义。没有用作者表面点云、外观留出图或留出掩膜初始化。将全部留出路径设为不可访问，两个训练加载器的 375 张观测都保持逐值一致。

![同预算静态与动态表示在抬手动作中的实际区别](../outputs/dynamic-human-enerf-local-time-029/comparison/camera00-frame046-person1.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站渲染 · 机位 00，共同索引 46 · 原始前景、静态、同编号动态、局部取帧动态；原生像素，无增强。

三组最终高斯数分别为静态 161,300、同编号动态 149,566、调整取帧动态 150,543。每组 75 张图的前景 PSNR 均值分别为 **18.20、24.88、25.24 dB**；调整取帧动态相对静态的 75 张图全部改善，相对同编号动态则 54 张改善、21 张退化。后者在三个机位上的平均差为 +0.44、+0.13、+0.49 dB，整体 +0.36 dB。两个上半身矩形区域的平均 PSNR 为 19.89、26.39、26.78 dB；这个矩形指标含灰色背景，不能称为手部专用分割指标。

![运动后段仍有手部、物品及袖口破碎](../outputs/dynamic-human-enerf-local-time-029/comparison/camera06-frame058-person2.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 机位 06，共同索引 58 · 含义同上。

人工逐一查看三个机位在 34、46、58 的全部 18 组人物上半身对照，以及调整组末帧三张完整画面。动态表示恢复了主要衣物与动作轮廓，局部取帧修正减少了部分手与玩偶破碎，但手指、物品表面、袖口边缘仍未清楚。机位 06 完整画面底部有孤立碎片，不能靠人物紧框隐藏。另按数值选择退化最大点：机位 00、共同索引 50 的前景分数下降 2.26 dB；查看 49～51 的六组人物对照后保留此失败，不用平均改善覆盖它。

![局部取帧修正退化最明显的时刻及原始参照](../outputs/dynamic-human-enerf-local-time-029/comparison/camera00-frame050-person2.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 机位 00，共同索引 50 · 不利结果保留；图像差别需结合原生图检查。

全部 225 张渲染、75 张源图合成参照和 450 个人物区域完成独立 CPU 核验，参照逐像素一致，PSNR、独立 SSIM 及完整画面残差一致。人物掩膜向外扩张 16 像素之外的彩色残差像素均值为 594.17、42.55、30.31；这是阈值残差指标，不是真实不透明度误差。216 组相邻图像变化残差的均值为 0.02264、0.02174、0.02096，范围是归一化 RGB；没有光流补偿，不能直接当作闪烁或三维跟踪精度。三段视频每段 25 帧，均逐帧解码通过，按 12 fps 慢放检查，未声称这是采集帧率。人工视觉范围为上述采样，不宣称已逐帧看完视频。

连续视频：[机位 00](../outputs/dynamic-human-enerf-local-time-029/comparison/camera00-continuous-comparison.mp4)、[机位 06](../outputs/dynamic-human-enerf-local-time-029/comparison/camera06-continuous-comparison.mp4)、[机位 12](../outputs/dynamic-human-enerf-local-time-029/comparison/camera12-continuous-comparison.mp4)。[机位 00 完整图](../outputs/dynamic-human-enerf-local-time-029/evaluation/camera00-frame058-offset-index.png)、[机位 06 完整图](../outputs/dynamic-human-enerf-local-time-029/evaluation/camera06-frame058-offset-index.png)、[机位 12 完整图](../outputs/dynamic-human-enerf-local-time-029/evaluation/camera12-frame058-offset-index.png)。

人物场景节点接口已做实际验证：仅用首帧训练机位的轮廓投票划分两个前景实例，分别包含 71,689 与 78,528 个高斯；不明确的 326 个高斯保留在独立节点，没有删除。每个时刻以节点中心平移和局部形变重新表达全部位置，25 个状态的 float32 世界坐标完全还原，三个重新合成的实际渲染逐像素一致。节点包含手持物，尚非纯人体／SMPL 表面；身份索引连续不代表物理表面跟踪已经正确。该接口没有学习建筑背景，也不是可直接加载的 OmniRe 检查点。

当前仍未通过移动人物清晰度验收。后续固定当前局部观测协议，分别检验表面覆盖、尺度与不透明度在运动阶段的约束；之后才做分离时间／其他人物的验证和联合背景训练。既有失败、旧数据划分和模型均保留。

可核对：[冻结条件](../outputs/dynamic-human-enerf-local-time-029/appearance-control-protocol.json)、[标定来源与几何复算](../outputs/dynamic-human-enerf-local-time-029/independent-calibration-audit.json)、[训练输入隔离](../outputs/dynamic-human-enerf-local-time-029/input-isolation-audit.json)、[逐图结果](../outputs/dynamic-human-enerf-local-time-029/evaluation/metrics.json)、[独立图像核验](../outputs/dynamic-human-enerf-local-time-029/independent-image-audit.json)、[模型状态核验](../outputs/dynamic-human-enerf-local-time-029/independent-state-audit.json)、[配对统计与视频解码](../outputs/dynamic-human-enerf-local-time-029/paired-statistics.json)、[实际看图记录](../outputs/dynamic-human-enerf-local-time-029/visual-review.json)、[人物节点说明](../outputs/dynamic-human-enerf-local-time-029/actor-interface/scene-contract.json)、[节点重组渲染](../outputs/dynamic-human-enerf-local-time-029/actor-interface/render-equivalence.json)。

### 运动阶段的尺度与不透明度对照

沿用局部时间对照的共同索引 34～58 和相同 375 张训练观测，固定 150,543 个高斯及其索引，分别继续训练：尺度／不透明度都固定、仅尺度可变、仅不透明度可变、两者均可变。四组共享已经完成 6,000 次优化的首帧状态，各自再进行 24×600＝14,400 次运动阶段更新；本轮新增有效更新共 57,600 次，不能把共享首帧算成四份新增训练。每组对应的完整历史为 20,400 次更新。

尺度学习率为 0.0005，对数尺度以初始值为中心限制在正负 $\ln 2$，相邻时刻对数尺度差的均方项权重为 0.01。不透明度对数几率学习率为 0.002，以初始值为中心限制在正负 2，实际不透明度相邻差的均方项权重为 0.01。中心、旋转、球谐颜色及其学习率、局部刚性／旋转／等距／颜色连续性项、采样顺序和原生像素均保持原设置。不进行分裂或剪枝。这个干预包括开放参数、边界与正则，不能解释成完全无约束的参数自由度比较。

首帧参数完全一致，但 CUDA 训练重做第一个运动时刻时出现数值差异，因而重新运行完整固定参数基线。该新基线与上一轮固定参数结果在 75 张图上的前景 PSNR 差值均值为 +0.00105 dB，平均绝对差为 0.02902 dB，最大绝对差为 0.13185 dB。这是一次同种子重做的数值敏感性检查，不是多随机种子置信区间；额外用于这项检查的 600 次更新未作为正式模型。

![四种尺度与不透明度条件的实际细节对照](../outputs/dynamic-human-enerf-support-030/comparison/camera00-frame046-person1.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站渲染 · 原始前景、两项固定、仅尺度变化、仅不透明度变化、两者变化；原生裁图，无增强。

四组使用完全相同的 75 张外观留出参照，平均前景 PSNR 依次为 **25.237、25.335、25.299、25.378 dB**。两者均变化相对新基线平均提高 0.14054 dB，63 张改善、9 张退化，3 张共同初始图不变。前景亮度 Sobel 梯度误差从 0.010255 降到 0.009997；它衡量与原图边缘的一致程度，不是越锐越好的锐度分数。轮廓 IoU 从 0.96138 到 0.96168，作者掩膜包含手持物和边缘错误，不能当作精确人体几何真值。三个机位都曾在独立早期图像上参与时间对应检查，因此这里仅留出外观训练；没有时间留出，且属于开发序列。

![同时开放两参数的最差配对退化](../outputs/dynamic-human-enerf-support-030/comparison/regression-camera12-frame044-person1.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 机位 12，索引 44 · 原始前景、两项固定、两者变化。

最差配对在机位 12、索引 44，前景分数下降 0.19905 dB；两个人物及前后相邻帧均已看图。另完成三个训练机位 01、07、14 在 34、46、58 的 36 张诊断渲染，每组 9 张。训练前景 PSNR 均值由固定组的 29.637 到两者变化组的 30.159 dB，但后段训练图中的手和物品仍有混合、拉丝。训练均值与留出均值来自不同机位及数量，不可直接将差值解释成纯泛化差距。

![训练视角中尚未解决的运动细节](../outputs/dynamic-human-enerf-support-030/training-comparison/camera07-frame058-person1-compact.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 训练机位 07，索引 58 · 原始前景、两项固定、两者变化。

人工检查范围为固定索引 34、46、50、58 的 24 张人物对照、最差退化点前后三帧的 6 张对照、末帧 6 张训练人物对照及 3 张完整渲染，共 39 张检查图。主要衣物、脸部和肢体轮廓可辨，但手指、物品和袖口的细节仍不符合要求；初始共同状态已经存在部分问题。机位 06 末帧完整图中仍有右侧人物旁的孤立碎片，没有删除或补绘。结合先前独立困难单帧拟合的失败，当前证据不支持把问题简单归为运动累积，或认定开放尺度与透明度即可解决。

已独立核验 100 个模型状态、336 张 RGB 及相应 alpha 输出、84 张参照、672 个人物区域与 288 组相邻图像变化残差。三个连续视频各 25 帧，全部解码通过；12 fps 是展示速度，未经核验的采集帧率不作声明。相邻残差没有光流补偿，不能直接称为闪烁指标；人工看图范围如上，不宣称逐帧看完了视频。

完整图：[机位 00](../outputs/dynamic-human-enerf-support-030/evaluation/camera00-frame058-both.png)、[机位 06](../outputs/dynamic-human-enerf-support-030/evaluation/camera06-frame058-both.png)、[机位 12](../outputs/dynamic-human-enerf-support-030/evaluation/camera12-frame058-both.png)。连续对照：[机位 00](../outputs/dynamic-human-enerf-support-030/comparison/camera00-continuous-comparison.mp4)、[机位 06](../outputs/dynamic-human-enerf-support-030/comparison/camera06-continuous-comparison.mp4)、[机位 12](../outputs/dynamic-human-enerf-support-030/comparison/camera12-continuous-comparison.mp4)。

核对材料：[训练协议](../outputs/dynamic-human-enerf-support-030/protocol.json)、[模型状态](../outputs/dynamic-human-enerf-support-030/independent-state-audit.json)、[留出图像核验](../outputs/dynamic-human-enerf-support-030/independent-image-audit.json)、[训练诊断核验](../outputs/dynamic-human-enerf-support-030/independent-training-image-audit.json)、[配对统计](../outputs/dynamic-human-enerf-support-030/paired-statistics.json)、[实际看图范围](../outputs/dynamic-human-enerf-support-030/visual-review.json)、[阶段结论](../outputs/dynamic-human-enerf-support-030/final-summary.json)。此轮对照结束，移动人物清晰度和完整场景目标仍未完成。

### 行走遮挡片段与背景几何准备

只使用训练机位 08 筛选另两组人物序列，在 actor2_3 与 actor5_6 中查看了六组、共 18 张缩小展示的源图。actor5_6 的索引 750 附近具有较明显行走及相互遮挡，已选为下一组候选；尚未在该片段训练模型或检查留出相机。作者前景合成在移动人物附近保留了墙面纹理残影，因此不能把这些像素全部作为正确人体表面监督。最初预览的整数乘法溢出已经在检查前修正，原错误产物作为本地记录保留，光流筛选使用原始 RGB，不受此预览错误影响。[筛选与观察范围](../outputs/dynamic-human-enerf-performer-selection-031/visual-review.json)。

在 actor5_6 自身的优化相机坐标中，使用 15 个训练机位提供的静态背景参照，经互相匹配、至少三视图轨迹与已知相机三角化得到 **6,480 个背景候选点**。没有使用来源视角未知的作者点云，也没有使用机位 00、06、12 的背景图。独立重投影复算通过，每条轨迹最大误差限制为 1.5 像素；这验证的是候选点与已知相机的几何一致性，不是已经训练好背景，更不是人物与背景联合重建完成。墙面重复纹理仍可能产生错误匹配，地面覆盖也需补足。不能直接把另一组演员的点拼进这个坐标系。[独立几何核验](../outputs/dynamic-human-enerf-background-032/independent-geometry-audit.json)、[实际源图检查](../outputs/dynamic-human-enerf-background-032/visual-review.json)。

### 背景拟合后的实际纹理检查

actor5_6 的 6,480 个训练视角背景点完成了累计 6,000 次优化。原运行在第 2,401 次更新后超过预设 50 万点数量限制，停止时为 532,803 个高斯；随后保留优化器状态、固定点数继续 3,599 次更新。这是有密度上限恢复的结果，不能称为未经修改的默认密度控制复现。没有发生显存不足，训练记录的 PyTorch 分配峰值约 1.71 GB，不包含其他程序和全部驱动占用。

![背景纹理在训练后仍然存在拉丝](../outputs/dynamic-human-enerf-background-fit-033/density-capped/evaluation/camera06-wall-comparison.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 留出相机 06 · 原图、稀疏初始点、最终背景；原生裁图。

三个留出视角 00、06、12 的有效像素 PSNR 分别由 14.79、15.10、15.24 dB 提高到 22.02、21.36、21.50 dB。完整图和局部图仍显示墙面拉丝、边缘区域重建不足及地面纹理缺失，抽查的训练视角也有同类问题。这个背景没有通过质量检查，尚不能作为可靠的固定背景用于人物联合训练。

独立复算覆盖 15 个训练缓存、12 张 RGB、12 张 alpha 及其指标；实际查看 3 张完整留出图、6 张墙面比较图和 3 张底部比较图。[完整机位 06 输出](../outputs/dynamic-human-enerf-background-fit-033/density-capped/evaluation/camera06-trained.png)、[状态与图像复核](../outputs/dynamic-human-enerf-background-fit-033/density-capped/independent-audit.json)、[检查范围](../outputs/dynamic-human-enerf-background-fit-033/density-capped/visual-review.json)、[阶段结论](../outputs/dynamic-human-enerf-background-fit-033/final-summary.json)。

### 人物透明度中的墙面残留与静态投影检查

在已选的行走片段中，仅使用训练相机 08 的第 740、750、760 帧检查源图。原图包含真实投影阴影，作者前景透明度同时保留了人体以外的部分墙面纹理。这说明背景减除结果不能直接当作语义人体真值，不能把阴影变化全部交给人体几何拟合。

![原始阴影与前景透明度残留的对应](../outputs/dynamic-human-enerf-matte-background-034/camera08-frame000750-wall-matte-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 训练相机 08，第 750 帧；没有清理或补绘原图。

低前景透明度墙面区域的特征对应另按 80% 拟合、20% 检查划分。原空场景参照与当前图像的检查点位移中位数约 2.67～2.88 像素，拟合二维单应性后为 0.21～0.24 像素。这里划分的是同一对图中的对应点，不是新照片测试；墙面单应性也不适用于墙前运动的人体。

再用训练视角三角化的三维点估计 PnP 候选位姿，墙面检查点重投影中位数由 2.42～2.70 像素降到 0.44～0.48 像素。没有参与拟合的底部区域仅找到每帧 10 或 11 个对应点，其中 9 或 10 个改善，中位误差降到 0.36～0.53 像素。点数少且主体几何接近平面，只能支持继续检查标定，不能宣称全序列标定已修复，或把此前另一演员序列的手部模糊都归因于此。

源图、前景透明度及正式相机均未改变；二维配准图仅作诊断，不用作模型效果。实际查看三组原生源图／透明度比较和三组静态配准比较。[二维配准结果](../outputs/dynamic-human-enerf-matte-background-034/static-registration.json)、[PnP 候选](../outputs/dynamic-human-enerf-matte-background-034/static-pose-diagnostic.json)、[独立底部对应检查](../outputs/dynamic-human-enerf-matte-background-034/pose-floor-check.json)、[观察范围](../outputs/dynamic-human-enerf-matte-background-034/visual-review.json)、[阶段结论](../outputs/dynamic-human-enerf-matte-background-034/final-summary.json)。

进一步检查全部 15 个训练相机，固定第 750 帧的部分空间区域拟合、其余区域检查，并在第 740、760 帧验证。第 750 帧各相机检查误差中位数的中位数为 2.17 像素，PnP 候选降到 0.40 像素；仅拟合每个相机的二维像素平移，也能降到 0.49 像素。因此静态纹理错位具有可重复性，但它还不能唯一确定为物理相机位姿变化。[全部训练相机的几何检查](../outputs/dynamic-human-enerf-static-camera-036/independent-audit-and-offset-comparison.json)。

随后使用另一段时间的静态特征拟合更简单的投影偏移：仅在第 700 帧拟合，第 710、720 帧检查。18 个相机的检查误差中位数汇总分别由 1.94、1.89 像素降到约 0.40 像素；将同一组偏移直接用于原有第 740、750、760 帧的静态对应，15 个训练相机的三维重投影误差汇总由约 2.15～2.24 像素降到 0.65～0.67 像素。这次额外读取了外观留出相机 00、06、12 在第 700、710、720 帧的静态标定照片，不能再称为完全没有访问这些相机的 RGB；它们没有进入人体外观训练，也没有用第 750 帧的人物误差拟合偏移。候选只在原图去畸变和裁剪后调整投影主点，15 个真实训练批次的图像、透明度与外参保持逐元素不变。**当前模型仍使用原相机；人物清晰度是否改善，需要下一组配对训练验证。** [时间与数据划分](../outputs/dynamic-human-enerf-static-calibration-037/protocol.json)、[独立复算](../outputs/dynamic-human-enerf-static-calibration-037/independent-calibration-audit.json)、[投影接口检查](../outputs/dynamic-human-enerf-static-calibration-037/projection-adapter-audit.json)。

### 官方 4K4D 的原生单帧人物验证

为检查图像特征能否保留人物细节，接入 4K4D 官方 R4DVSampler，以 actor5_6 第 750 帧完成 1,000 次优化。其表示是每时刻点云、空间与时间特征、投影到源图的颜色融合及球谐残差，并非传统各向异性 3DGS。当前在 WSL 使用官方提供的 PyTorch3D 点渲染路径；每次八张源图中两张保留图像编码器梯度，因此不是论文默认硬件后端的完整复现。

初始化几何仅由 15 个训练机位的轮廓生成，没有使用作者预制表面；外观训练与源图也仅使用这 15 个机位。00、06、12 三个机位留出外观。训练保留原生像素并同步调整裁剪后的主点；额外评测恢复完整 1920×1080 内参和画面，从模型输入中移除目标 RGB 和掩膜，避免把目标轮廓用于最终画面裁剪。另三个训练诊断视角也从纹理来源中排除了自身照片。

![官方模型在原生分辨率下的真实人物对照](../outputs/dynamic-human-4k4d-official-035/native-evaluation/camera00-frame000750-full-person-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站官方模型运行 · 留出机位 00；左为作者透明度合成参照，右为真实渲染，无增强。

三路留出相机在作者前景透明度大于 0.5 的像素上，PSNR 分别为 21.57、18.84、19.44 dB；包含大面积灰底的完整有效画面为 29.10、27.49、27.83 dB。两种口径差异说明，灰色背景会抬高全图分数，人物细节必须单独观察。衣物图案和主要形状可辨，但头发、肩部和鞋边仍有颗粒状黑边，手、玩偶、交叉腿及遮挡区域仍模糊。六组人物对照和三张留出完整图均已查看，尚未通过清晰度要求；这个单帧结果也不能证明时间连续性。

全部 24 张源图／参照／渲染／alpha PNG、12 张原生比较图和各区域指标已复核；模型参数有限，优化器更新次数为 1,000。从相同模型与 Adam 状态继续 9,000 次更新的验证现已完成，累计 10,000 次；图像、相机和来源机位保持不变。三个留出机位的人物前景 PSNR 仅提高 0.16～0.21 dB，完整有效图像 PSNR 反而降低约 0.20～0.41 dB。全部六组人物对照和三张完整留出渲染已查看，手、玩具、交叉腿部仍模糊，轮廓黑色点状伪影仍存在。**增加当前训练预算没有达到清晰度要求。**当前这些视角属于开发验证，不作为从未查看过的最终测试集。参见[完整机位 06](../outputs/dynamic-human-4k4d-official-035/native-evaluation/camera06-frame000750-trained.png)、[实际看图范围](../outputs/dynamic-human-4k4d-official-035/visual-review.json)、[模型状态核验](../outputs/dynamic-human-4k4d-official-035/independent-state-audit.json)、[原生图像核验](../outputs/dynamic-human-4k4d-official-035/native-evaluation/independent-image-audit.json)、[阶段结论](../outputs/dynamic-human-4k4d-official-035/final-summary.json)、[后续固定预算](../outputs/dynamic-human-4k4d-official-035/continuation-10000/protocol.json)。



![相同帧的原透明度参照、一千次与一万次更新的真实渲染](../outputs/dynamic-human-4k4d-official-035/continuation-10000/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06，第 750 帧 · 原生像素，无增强。

这三列中的左列仍是有误差的作者透明度合成参照。不能因为预测更接近一个错误的参照，就认为人体更完整。见[较长训练的结果](../outputs/dynamic-human-4k4d-official-035/continuation-10000/final-summary.json)与[实际看图范围](../outputs/dynamic-human-4k4d-official-035/continuation-10000/visual-review.json)。

### 前景掩码漏掉可见人体后的输入对照

在第 750 帧训练机位 01 的原照片中，黑衣人物的前臂实际可见，但作者透明度在其中一个人工检查的内部像素为零；另外三个内部像素也只保留约 0.11～0.43 的透明度。严格要求全部 15 个训练轮廓通过的视觉外壳初始化，可能因此剔除对应三维候选点；源图乘透明度后也会损伤该处颜色。当前尚未测量被剔除的准确体积，不能把所有模糊归因于这一项。

已用官方 SAM 2.1 Hiera Tiny 在 CPU 上处理 15 个训练视角，分别对两个人物和手持物提供人工框／点提示。整组单框会漏掉另一人，自动多点方案会在衣服和鞋上挖洞，均未采用。单独物体提示仍在遮挡处留下孔洞，因此最终候选只在分割掩码形态学闭运算的支持范围内，补回原透明度大于 0.95 的像素；核的尺寸为原生分辨率下 49×49 像素。曾尝试向外扩展支持范围，但会带回墙面阴影，该版本已保留为失败记录。

15 个最终候选均检查了原图、原透明度和新掩码的完整人体对照，并独立核对保存图中的像素。它们保留了可见前臂、衣物和鞋的主要内部区域，但靠近身体的墙面残留与硬边界仍存在。**这些图片是原照片与分割掩码的输入合成，不能当作模型已经重建清晰的效果。** 它们只进入有界的训练输入对照，不是真值标注；没有使用第 750 帧留出相机 RGB 生成候选。参见[输入检查结论](../outputs/dynamic-human-enerf-mask-quality-038/final-summary.json)、[视觉检查及剩余问题](../outputs/dynamic-human-enerf-mask-quality-038/visual-review.json)、[图像像素核验](../outputs/dynamic-human-enerf-mask-quality-038/independent-panel-audit.json)。

新对照保持原相机、原生像素、相同模型和 1,000 次更新预算，重新从训练机位掩码构建视觉外壳。变化涉及初始化、裁剪、源图乘掩码和训练监督的整个输入流程，不能称为仅修改损失函数的消融。另存的去畸变图像经过官方解码往返核对，原始照片和标定文件不改写；静态投影偏移候选 037 未应用。见[输入对照条件](../outputs/dynamic-human-4k4d-mask-control-039/protocol.json)。


### 输入修复后，真实渲染改善了哪里

相同 1,000 次更新预算的训练与六个原生视角渲染已完成。与原输入流程相比，三个留出视角的人物前景 PSNR 分别变化 −0.27、+0.10、+0.94 dB；完整有效图像则分别提高 0.46、0.76、1.27 dB。前景指标仍使用有误差的作者透明度，因此同时查看原照片与真实渲染。

![同一原照片、原掩码训练和修订掩码训练的真实渲染](../outputs/dynamic-human-4k4d-mask-control-039/paired-comparison/camera12-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 12，第 750 帧 · 相同 1,000 次更新，原生像素，无图像增强。

修订输入后，部分前臂连接与鞋边有所改善，但手指、衣物图案、手持物和交叉肢体仍模糊，头肩轮廓仍有黑色点状伪影。六组完整人物对照、一个前臂对照和三张完整留出渲染均已查看。**这是局部恢复，仍未达到清晰人体重建的要求。** 此处的右列是真实模型渲染，与上一节的分割输入合成有明确区别。

模型、优化器与 1,000 次更新状态，以及全部原生图像和指标，均已单独核对。见[结果结论](../outputs/dynamic-human-4k4d-mask-control-039/final-summary.json)、[看图范围](../outputs/dynamic-human-4k4d-mask-control-039/visual-review.json)、[模型状态核验](../outputs/dynamic-human-4k4d-mask-control-039/independent-state-audit.json)、[图像核验](../outputs/dynamic-human-4k4d-mask-control-039/native-evaluation/independent-image-audit.json)。


### 背景配准改善没有直接转化为人体清晰度

从同一个修订掩码模型与 Adam 状态出发，原投影和静态背景偏移两组各再训练 1,000 次，累计均为 2,000 次。修正组仅在解码、裁剪以后改变投影矩阵中的像平面偏移，原照片、源机位、训练预算与初始化保持一致。全部 12 张原生视角渲染及对应指标已核验。

修正组在三个留出机位的前景 PSNR 分别降低 0.82、0.70、0.66 dB，六个检查视角都下降。六组人物对照与三张完整留出渲染中，手、衣服图案和遮挡关系没有恢复，部分肩部与鞋边伪影仍明显。因此保留原投影继续诊断，不采用这组偏移。这只否定当前父模型与预算下的经验偏移处理，不能推广为“相机标定和时间同步不重要”。见[成对结果](../outputs/dynamic-human-4k4d-projection-control-040/final-summary.json)、[实际看图结论](../outputs/dynamic-human-4k4d-projection-control-040/visual-review.json)、[成对模型状态核验](../outputs/dynamic-human-4k4d-projection-control-040/independent-state-audit.json)。

### 合成公式不同，也要测量实际影响

对当前点渲染实现的核对发现，其透明度通道计算为 $\sum_i T_i a_i o_i$，其中 $o_i$ 是点不透明度，$a_i$ 已包含不透明度与径向权重；标准累计透明度则为 $\sum_i T_i a_i=1-\prod_i(1-a_i)$。这段写法在官方 PyTorch3D 与可微 OpenGL 路径中均存在，不能归因于本机兼容路径。

小例子能清楚地区分两个公式，但实际模型的点不透明度均已高于 0.9978。使用同一批真实光栅化片元重算，六个原生视角的最大透明度变化约为 0.00206，前景 PSNR 变化均不足 0.0007 dB。三个留出视角的人物对照已查看，手、衣物与遮挡仍模糊。**公式差异属实，但它不足以解释当前明显的清晰度问题。** 这是固定模型的合成诊断，尚未声称从头训练时两种公式效果相同。见[公式与实测结论](../outputs/dynamic-human-4k4d-alpha-composition-041/final-summary.json)、[独立片元与像素核验](../outputs/dynamic-human-4k4d-alpha-composition-041/independent-audit.json)。


### 前景支持筛选与源视角选择

固定同一个 2,000 次更新模型、几何、点半径、不透明度与源机位，使用同一批排序片元，仅改变源颜色的融合规则。排除源掩码小于 0.5 的观测并重新归一化后，六个视角的前景 PSNR 均提高，幅度约 0.09～0.36 dB；三个留出机位分别为 +0.36、+0.10、+0.32 dB。六组人物对照均已查看：部分肩部、袖口黑边减轻，但手、衣物图案、玩具和交叉肢体仍模糊。另做了对源颜色解除掩码预乘的控制，没有稳定优于仅筛选支持的方案。见[固定模型结果](../outputs/dynamic-human-4k4d-source-support-042/final-summary.json)与[18 张原生图像核验](../outputs/dynamic-human-4k4d-source-support-042/independent-audit.json)。

继续固定几何，对比仅保留最近两个源机位、将融合权重的四次方重新归一化、每点仅选择最大权重机位。最近两机位只改善了六个视角中的两个，另外两种方法在六个视角都降低前景指标。三个留出机位的三类对照共九张图已查看：最近机位有时让衣服图案更强，但肢体和衣物出现斑块；最大权重选择在白衣和皮肤上产生明显杂色。因此未采用强制源选择。全部 30 张原生图像与 24 张对照图均已核验，见[源选择结果](../outputs/dynamic-human-4k4d-source-selection-043/final-summary.json)及[独立核验](../outputs/dynamic-human-4k4d-source-selection-043/independent-audit.json)。

这些属于不改变模型参数的取色控制，尚不能证明修改训练流程后也有效。下一组从相同模型与优化器状态出发，原融合和前景支持融合各继续 1,000 次更新，累计均为 3,000 次；保持原投影和原透明度计算，以隔离融合规则的影响。见[成对训练条件](../outputs/dynamic-human-4k4d-source-gate-training-044/protocol.json)。


### 行走人物：训练、覆盖范围与表示对照

前景支持筛选的成对训练已经完成。两组从同一个 2,000 次更新模型及 Adam 状态出发，各增加 1,000 次更新；记录的目标与源机位序列一致。筛选组在六个视角均提高前景 PSNR，三个留出机位 00、06、12 分别提高 0.324、0.248、0.234 dB。六张原生人物对照与三张完整留出渲染均已查看：肩部部分黑边减轻，手、衣服图案和遮挡仍不清楚。见[成对训练结论](../outputs/dynamic-human-4k4d-source-gate-training-044/final-summary.json)、[模型状态核验](../outputs/dynamic-human-4k4d-source-gate-training-044/independent-state-audit.json)。

原始半径投影到图像后，参与渲染点的半径中位数约为 2.57～2.97 像素。保持点位置和颜色不变，将半径乘以 0.75 或 0.5，六个视角的前景指标均下降；头发、衣服和皮肤出现更多孔洞，手部没有恢复。见[覆盖范围对照](../outputs/dynamic-human-4k4d-point-footprints-046/final-summary.json)与[图像、半径公式核验](../outputs/dynamic-human-4k4d-point-footprints-046/independent-audit.json)。随后用当前模型的源相机深度排除遮挡取色，也在六个视角都变差；由于深度来自当前不完善的几何，此结果不能否定准确可见性监督的价值。见[可见性检查](../outputs/dynamic-human-4k4d-visibility-047/final-summary.json)。

为检查另一种表示，采用完全相同的 15 个训练机位、原生 RGB、候选掩码和标定，训练直接优化各向异性三维高斯的独立分支，包含自适应分裂与剪枝。固定 6,000 次更新后共有 142,735 个高斯。它改变了初始化、表示、损失与训练预算，是方法替代对照，不是单因素消融，也不是官方 4K4D 复现。

![原照片、4K4D 点式分支与直接优化的 3DGS：留出机位 06](../outputs/dynamic-human-enerf-representation-048/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 保留手臂拉长与遮挡错误。

六个视角均已检查。三个训练诊断视角的前景 PSNR 提高约 1.50～3.59 dB，但留出机位 00、06、12 分别变化 −1.466、−0.066、+0.952 dB。头发、衣服褶皱和大面积表面变清楚，移动手臂、手指、提着的玩具和抬起的鞋仍有拉长或重影。因此该分支仍未通过清晰人物验收。见[训练协议](../outputs/dynamic-human-enerf-representation-048/protocol.json)、[结果与局限](../outputs/dynamic-human-enerf-representation-048/final-summary.json)、[24 张原生图像核验](../outputs/dynamic-human-enerf-representation-048/native-evaluation/independent-image-audit.json)及[六组图像审阅](../outputs/dynamic-human-enerf-representation-048/visual-review.json)。

### 行走片段的跨相机运动一致性

同帧输入检查采用 25 组相邻训练相机，共 2,879 个前景互匹配点，不用对极误差筛选匹配；其中 17 组的 Sampson 误差中位数小于 1 像素。原图检查同时发现少量错误匹配，以及某些移动鞋部的数像素残差。因此不能笼统归咎于全部数据不同步，见[同帧核验](../outputs/dynamic-human-enerf-foreground-geometry-045/final-summary.json)。

进一步固定同一组对应点，跟踪前后 3 帧，以 740、750 帧选候选偏移、760 帧复核。共保存并重算 117 个轨迹数组、5,594 条对应。39 组相机中，22 组偏好零偏移，17 组偏好非零偏移；非零候选中有 14 组在复核帧达到至少 12 个运动点，其中 9 组改善、5 组变差。四组原生三列图的 24 个对应已查看，未据此改写数据时间轴。机位 10 的局部成对训练已经完成，结果见下文。见[轨迹核验](../outputs/dynamic-human-enerf-actor56-time-049/independent-audit.json)与[当前结论](../outputs/dynamic-human-enerf-actor56-time-049/final-summary.json)。


### 单机位时间偏移与联合相机图

只将训练机位 10 的第 750 帧换成第 749 帧，使用相同的 SAM2 提示和掩码处理规则；其余 14 个训练输入、原始标定、留出照片均保持不变。两组从完全相同的高斯参数出发，固定全部 6,000 次目标机位与背景颜色抽样，分别完成 6,000 次更新。三个留出机位 00、06、12 的前景 PSNR 变化为 −0.373、+0.085、+1.003 dB。

![原照片、同帧训练与仅机位 10 前移一帧的留出效果](../outputs/dynamic-human-enerf-camera10-time-050/paired-comparison/camera12-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 12 · 无增强。

六张完整人物对照及预先确定的 17 个人体细节区域均已查看。机位 12 的衣服、部分前臂和提物轮廓改善，但垂下的手臂仍有背景残留与拉伸；机位 00 的手和抬起的鞋变差，机位 06 的交叉手臂未恢复。因此不把单机位改帧作为清晰重建方案。见[结论](../outputs/dynamic-human-enerf-camera10-time-050/final-summary.json)、[成对状态核验](../outputs/dynamic-human-enerf-camera10-time-050/independent-state-audit.json)、[完整与细节指标](../outputs/dynamic-human-enerf-camera10-time-050/paired-comparison/summary.json)及[逐图审阅](../outputs/dynamic-human-enerf-camera10-time-050/visual-review.json)。

进一步只使用 740、750 帧的固定轨迹拟合 15 个训练相机的一致相对帧索引，限制偏移为 −2 至 +2 帧，机位 01 固定为零；29 条有效相机边形成连通图。联合解建议机位 02、05、10 分别取 −2、−1、−1，其余不变。760 帧的 28 条有效复核边平均误差由 0.651 降至 0.579 像素，但新增 02、05 偏移涉及的四条有效边中仅一条改善、三条变差。因此未开展全局改帧训练，也未修改原始数据。这只是图像索引假设，不是硬件时钟标定。见[联合解与复核](../outputs/dynamic-human-enerf-time-graph-051/final-summary.json)和[固定轨迹重算](../outputs/dynamic-human-enerf-time-graph-051/independent-audit.json)。

### 绝对梯度增密与移动肢体

在同帧基线的原生输入、掩码、初始化、6,000 次抽样与更新预算下，切换到 gsplat 1.3 文档提供的 AbsGS 增密选项：启用绝对梯度，并将对应增密阈值设为 0.0008，基线阈值为 0.0002。这是本地增密策略对照，不是 AbsGS 原论文完整基准复现。实际 CUDA 绝对梯度缓冲区已确认存在、有限且非零。

![原照片、标准梯度与绝对梯度增密的完整留出人物](../outputs/dynamic-human-enerf-absgrad-052/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06。

最终高斯数量由 143,073 降为 79,659，减少约 44.3%；三个留出机位的前景 PSNR 变化仅为 −0.036、−0.003、+0.020 dB。六张完整人物对照和相同的 17 个细节区域均已查看，交叉手臂、手指、玩具及抬起的鞋仍有拉丝或模糊。这说明当前设置的模型更小，并未证明人体更清晰。见[结果](../outputs/dynamic-human-enerf-absgrad-052/final-summary.json)、[训练状态核验](../outputs/dynamic-human-enerf-absgrad-052/paired-training-state-audit.json)、[原生图像核验](../outputs/dynamic-human-enerf-absgrad-052/native-evaluation/independent-image-audit.json)、[逐图审阅](../outputs/dynamic-human-enerf-absgrad-052/visual-review.json)及[固定区域指标](../outputs/dynamic-human-enerf-absgrad-052/paired-comparison/summary.json)。

下载包附带的作者表面点云只覆盖文件编号 000000～000199，当前行走对照使用第 750 帧；没有证据支持把点云编号乘以六对应到原图，因此未将它接入当前模型。下一步检查训练相机中的前景残差、表面深度和轮廓约束，再选择几何监督方法。当前 GPU 可完成这些有界对照，没有新增硬件或数据下载阻塞。


### 掩码冲突与稀疏表面检查

以补全掩码为前景、SAM 掩码为非前景的像素作为未知，只关闭该处的损失有效性，不改原始 RGB、目标掩码、相机、初始高斯或 6,000 次采样序列。15 个训练机位总计忽略 46,344 个像素，分别占各自前景的 0.30%～4.86%。最大的 12 个冲突区域和三张完整输入图已查看：这些区域同时包含墙面和真实手臂、衣服，不能统一改成背景。

完成 6,000 次更新后得到 137,951 个高斯。三个留出视角 00、06、12 的前景 PSNR 变化分别为 −0.174、+0.071、+0.037 dB。六张完整人物对照与预先确定的 17 个细节区域均已查看，机位 08 的部分墙面残留减少，但手臂、手指和交叉遮挡仍不合格。见[训练结论](../outputs/dynamic-human-enerf-mask-uncertainty-053/final-summary.json)、[输入与成对状态核验](../outputs/dynamic-human-enerf-mask-uncertainty-053/paired-training-state-audit.json)、[原生图像核验](../outputs/dynamic-human-enerf-mask-uncertainty-053/native-evaluation/independent-image-audit.json)和[图像审阅](../outputs/dynamic-human-enerf-mask-uncertainty-053/visual-review.json)。

![原照片、原监督与忽略冲突区域的训练机位 08 效果](../outputs/dynamic-human-enerf-mask-uncertainty-053/paired-comparison/camera08-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 无增强。

为检验表面位置，从已有训练相机互匹配中建立跨三台以上相机的纹理轨迹；拒绝含重复相机的冲突轨迹，要求多视角三角化后每个观测重投影误差小于 1 像素、最大三角化夹角大于 2°、深度均为正。最后保留 90 条轨迹、375 个观测，未使用留出照片。

这些点与同帧 3DGS 基线的期望深度相比，相对差异中位数为 0.326%，第 90 百分位为 1.001%；使用模型深度反投影后再投到其他观测机位，误差中位数为 1.358 像素。15 张深度图、375 个采样和投影关系已重算，12 组三视角原生照片／渲染局部图及三个机位的覆盖图已查看。匹配明显集中在着地鞋、玩具、面部和少量衣服位置，裸露前臂、手指、抬起的鞋缺少直接支持。这不是独立深度真值，也不能作为人体几何已经恢复的证明。见[几何结论](../outputs/dynamic-human-enerf-surface-tracks-054/final-summary.json)、[独立重算](../outputs/dynamic-human-enerf-surface-tracks-054/independent-audit.json)和[对应与覆盖审阅](../outputs/dynamic-human-enerf-surface-tracks-054/visual-review.json)。


### 二维高斯表面与几何正则项

在独立运行环境中完成 gsplat 1.5.3 的二维高斯 CUDA 渲染、反向传播与两组正式训练。两组各更新 6,000 次，使用完全相同的 52,109 个轮廓壳层初始点、局部 PCA 切平面方向、15 台训练相机的原生图像与掩码，以及全部相机和背景颜色采样。增加约束的一组在有效前景内部加入深度畸变和法线一致性项；已验证约束对几何产生有限、非零梯度。

最终未加约束组为 205,318 个面元，加约束组为 208,340 个。三个未参与训练的开发机位 00、06、12 前景 PSNR 分别提高 0.614、0.628、0.355 dB；训练机位 01、08、17 则分别下降 1.259、0.732、1.325 dB。这里的留出相机已在多轮开发中反复查看，不能当作全新的最终测试集。两组 48 张原生输出、24 张比较图的像素和指标已独立重算，六张完整人物对照与预先确定的 17 个细节区域已逐张查看。

![原照片、无表面约束与有表面约束的完整人物效果](../outputs/dynamic-human-enerf-surface-gs-055/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 无增强。

部分白衣人物的前臂条纹减轻，衣服表面更平顺；黑衣人物的下垂前臂仍有拉伸，交叉手臂、遮挡肩部、衣服图案和抬起的鞋仍有明显失真。结论是局部改善，清晰移动人物仍未验收。与此前三维高斯的整套流程比较还同时涉及渲染器版本、面元表示和初始方向，不能写成单独改变维度的因果结论。

见[结果汇总](../outputs/dynamic-human-enerf-surface-gs-055/final-summary.json)、[事前协议](../outputs/dynamic-human-enerf-surface-gs-055/protocol.json)、[成对训练核验](../outputs/dynamic-human-enerf-surface-gs-055/paired-training-state-audit.json)、[无约束图像核验](../outputs/dynamic-human-enerf-surface-gs-055/surface_only/native-evaluation/independent-image-audit.json)、[有约束图像核验](../outputs/dynamic-human-enerf-surface-gs-055/surface_regularized/native-evaluation/independent-image-audit.json)和[实际图像审阅](../outputs/dynamic-human-enerf-surface-gs-055/visual-review.json)。




### 更长训练与持续增密

在相同原始图像、掩码、相机和初始面元上，重新训练有表面约束的模型至 30,000 次更新；同时把增密延长到第 15,000 次附近，并按更长训练预算衰减位置学习率。这是训练计划整体调整，不能解释为只改变迭代次数。第 14,901 次更新后面元数量达到 800,528，超过预设资源上限 528 个，保存完整检查点后将保护上限调为 805,000 并恢复；参数、优化器、随机状态和损失均未重置，后续不再增密。峰值训练分配显存约 2.3 GB，未发生显存不足。

最终三个训练诊断机位 01、08、17 的前景 PSNR 相对 6,000 次有约束版本提高 2.677、2.234、3.858 dB；但未参与训练的开发机位 00、06、12 分别下降 0.749、0.122、0.450 dB。24 张原生输出和 12 张原生比较图通过独立像素与指标核验；六张完整人物对照与同一组 17 个预定细节区域已查看。

![原照片、六千次表面训练与三万次训练的原生人物对照](../outputs/dynamic-human-enerf-surface-long-056/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 无增强。

训练照片上的衣服、手和玩具更接近观测，但留出视角的前臂、衣服边缘与玩具出现更多细碎条纹和漂浮结构；交叉手臂的位置错误仍在。该结果不作为清晰度提升版本，也不继续把扩大点数和迭代次数作为默认方向。下一步应改善可靠的人体对应与几何约束，而非把训练视角拟合得更好当成完成重建。

见[结果汇总](../outputs/dynamic-human-enerf-surface-long-056/final-summary.json)、[事前计划](../outputs/dynamic-human-enerf-surface-long-056/protocol.json)、[中断与恢复](../outputs/dynamic-human-enerf-surface-long-056/cap-recovery.json)、[最终训练核验](../outputs/dynamic-human-enerf-surface-long-056/training-state-audit.json)、[原生图像重算](../outputs/dynamic-human-enerf-surface-long-056/native-evaluation/independent-image-audit.json)和[实际图像审阅](../outputs/dynamic-human-enerf-surface-long-056/visual-review.json)。

### 训练照片中的人体关节与粗略人体先验

对第 750 帧的 15 台训练相机，使用 Keypoint R-CNN 提取 12 个躯干和四肢关节，以先前的人物实例掩码提出人物对应；未读取机位 00、06、12 的照片。15 张检测叠加图均已查看：被遮挡的黑衣人物存在漏检，机位 13 甚至把露出的头部当作完整人体，其低关节分数被排除。其他遮挡处仍有左右关节和人物混淆，检测分数不是真实可见性的证明。

用成对三角化和鲁棒重投影拟合检查各关节，仅保留至少四台相机支持、重投影残差小于 8 像素且最大观测夹角超过 2° 的候选。两人共 24 个身体关节中保留 23 个，黑衣人物左肘因缺乏支持保持缺失。220 个支持观测的重投影误差中位数为 3.197 像素，已独立重算；五个机位的三角化投影已查看。这些是预测得到的关节假设，不能充当关节真值或人体表面真值。

据此完成 2,000 次 CPU 优化，拟合两个人的 SMPL 身体姿态、10 个体形参数、整体旋转、平移和尺度。关节正向运动学与完整 SMPL 前向输出一致，输出参数与误差已核验。15 张投影图中已查看机位 01、04、07、08、17：躯干和腿部中心大体对应，但黑衣人物头部过直，缺失左肘导致手臂弯曲错误，手掌与脚的朝向仍不准确。该先验没有衣物、头发和玩具，尚未作为硬几何约束加入高斯训练。

![原训练照片与拟合的粗略 SMPL 人体投影；没有进行外观训练](../outputs/dynamic-human-enerf-body-prior-058/camera04-body-prior.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap，用户授权下载的 SMPL 中性模型与本站拟合 · 人体先验投影，不是颜色重建效果。

见[关节观测结论](../outputs/dynamic-human-enerf-body-observations-057/final-summary.json)、[关节几何重算](../outputs/dynamic-human-enerf-body-observations-057/independent-audit.json)、[观测审阅](../outputs/dynamic-human-enerf-body-observations-057/visual-review.json)、[人体先验结论](../outputs/dynamic-human-enerf-body-prior-058/final-summary.json)、[拟合核验](../outputs/dynamic-human-enerf-body-prior-058/independent-audit.json)和[投影审阅](../outputs/dynamic-human-enerf-body-prior-058/visual-review.json)。SMPL 模板、模型参数和网格均留在本地，不作为网站下载资源。


### 全身关键点与人体结构先验的修正

使用 RTMW 的 133 个全身关键点模型，对同一帧 15 台训练相机中的两个人分别推理。人物框沿用此前固定的提示，不使用留出照片。对身体、头部、脚部及指尖的 66 个候选三维点进行鲁棒三角化，64 个获得至少四个视角支持；其中身体、头部和脚部的 46 个候选均有几何支持，补上了此前缺失的黑衣人物左肘。990 个重投影误差已重算，接受点支持观测的中位残差为 2.762 像素。几何一致不等于语义正确：背对相机的白衣人物仍有被模型猜测出来的鼻子和眼睛。

因此，人体拟合排除了白衣人物的鼻子、双眼以及全部指尖，只使用 43 个预测点完成 2,500 次优化。基本 SMPL 没有独立的手指关节，不能用张开的模板手掌强行拟合抓握动作。相对同一组新目标点，原先验平均残差为 0.07770，修正后为 0.01564，单位为数据世界坐标单位，未独立认证为米。保存参数在 CPU 上重新执行完整 SMPL 前向后，顶点最大差异小于 0.000001。

![真实训练照片与修正后的人体结构先验投影](../outputs/dynamic-human-enerf-wholebody-prior-060/camera04-body-prior.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站拟合 · SMPL 模板已获用户下载授权 · 结构投影，不是外观重建。

检查了 15 张投影中的 01、04、07、17 四台相机：头部俯仰、左肘与脚部方向改善，手掌、衣物、鞋和玩具仍不匹配。该先验只用于有限范围的柔性约束试验，不能作为真实衣物表面。见[关键点结论](../outputs/dynamic-human-enerf-wholebody-059/final-summary.json)、[重投影核验](../outputs/dynamic-human-enerf-wholebody-059/independent-audit.json)、[观测检查范围](../outputs/dynamic-human-enerf-wholebody-059/visual-review.json)、[先验结论](../outputs/dynamic-human-enerf-wholebody-prior-060/final-summary.json)、[前向与误差复算](../outputs/dynamic-human-enerf-wholebody-prior-060/independent-audit.json)和[投影检查范围](../outputs/dynamic-human-enerf-wholebody-prior-060/visual-review.json)。模板、网格和拟合参数保留在本地。

### 裸露前臂的柔性表面约束

保持 15 张原生训练图、相机、掩码、初始高斯、采样顺序、优化器、增密计划和表面正则项一致，新增 6,000 次更新的对照。唯一训练改动是从第 1,401 次更新起，对肘至腕中间一段的高斯加入相对人体表面切平面的距离惩罚。容许带宽为前臂长度的 4%，权重为 0.1；手掌、头部和衣物不作为该项的监督目标。空间筛选近似仍可能误含邻近物体，不能视为精确皮肤分割。

![原照片、表面基线与增加前臂约束后的原生人物对照](../outputs/dynamic-human-enerf-forearm-control-061/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站渲染 · 开发留出机位 06 · 无锐化或生成式增强。

最终得到 208,468 个面元，峰值训练分配显存约 1.53 GB。三个开发留出机位 00、06、12 的前景 PSNR 分别变化 −0.014、+0.039、+0.145 dB。24 张原生输出、12 张比较图和同一组 17 个细节区域通过像素核验；六张完整人物对照和全部 17 个细节均已查看。黑衣人物前臂仍拉长，交叉手臂的位置错误和玩具拖影基本保留，不作为达标结果。下一步用连续帧中的人体对应检验跨相机残余时序差异，不能仅根据这组负结果断言模糊由时间误差造成。

见[固定训练条件](../outputs/dynamic-human-enerf-forearm-control-061/protocol.json)、[结果汇总](../outputs/dynamic-human-enerf-forearm-control-061/final-summary.json)、[模型和优化器核验](../outputs/dynamic-human-enerf-forearm-control-061/training-state-audit.json)、[像素与指标重算](../outputs/dynamic-human-enerf-forearm-control-061/native-evaluation/independent-image-audit.json)、[同区域逐项比较](../outputs/dynamic-human-enerf-forearm-control-061/paired-comparison/summary.json)和[看图结论](../outputs/dynamic-human-enerf-forearm-control-061/visual-review.json)。


### 连续人体关键点中的跨相机时间偏移

前臂约束对照未解决拖影后，改用第 744～756 帧的连续人体观测检查时序。15 台训练相机共有 195 张原生照片、390 次分人全身关键点推理；预先选定七个偶数时刻用于拟合，六个奇数时刻用于检查。第 750 帧的全部 15 张输入及关键点结果与上一轮逐项一致。首轮输入因去畸变的浮点精度约定不同触发断言并停止，随后恢复原训练加载方式，通过全部输入一致性检查，未改动原训练数据。

对身体和脚部的 36 条三维点轨迹分别采用三次多项式，联合估计相机时间偏移；相机 01 定义零偏移，其他偏移限制在正负两帧内。对应支持只来自第 750 帧的鲁棒几何检查，遮挡错误仍由鲁棒损失处理。保持相机几何不变，对比零偏移拟合与允许偏移的拟合，后者没有参数触及偏移边界。

在预留的 2,262 个关节点观测中，重投影误差中位数从 3.228 降为 2.636 像素，90% 分位数从 9.398 降为 8.235 像素；均方根误差仅从 18.461 降到 18.176 像素，表明遮挡或错误对应导致的大残差仍然存在，不能只展示中位数。

随后做跨人物检查：只由黑衣人物估计时间偏移，将这些偏移固定，再拟合白衣人物的训练时刻轨迹；也反向进行同样检查。在另一人的预留时刻，黑衣人物中位误差从 3.114 降到 2.542 像素，白衣人物从 3.297 降到 2.902 像素。两人独立估计的偏移差异均方根为 0.361 帧，部分相机差异较大。这支持把残余时序差异作为下一轮图像训练的候选因素，尚不足以认证真实时钟偏移。

这一轮没有按估计偏移重新取图，也没有据此宣称人物清晰度改善。尤其机位 00、06、12 未参与估计，它们相对相机 01 的拍摄时刻仍未知。下一步应在已获得时间关系的训练相机中，比较连续人体模型对原生预留时刻图像的重建，保持原始照片不做锐化或插帧。

见[连续片段与结论](../outputs/dynamic-human-enerf-pose-sequence-062/final-summary.json)、[时序拟合条件](../outputs/dynamic-human-enerf-pose-sequence-062/timing-protocol.json)、[逐相机结果](../outputs/dynamic-human-enerf-pose-sequence-062/timing-summary.json)、[跨人物检查](../outputs/dynamic-human-enerf-pose-sequence-062/person-transfer-summary.json)、[输入和全部误差复算](../outputs/dynamic-human-enerf-pose-sequence-062/independent-audit.json)、[实际看图范围](../outputs/dynamic-human-enerf-pose-sequence-062/visual-review.json)及[输入约定修复](../outputs/dynamic-human-enerf-pose-sequence-062/input-recovery.json)。


### 连续人体运动与拍摄时间的实际重建对照

先把两种关节点轨迹拟合为连续 SMPL 姿态，得到“相同帧编号”与“估计拍摄时间”两组运动。人体形状与尺度共同固定为前一轮结果，局部颈部、头部和手部旋转当时没有可靠运动目标，保持不变。两组各完成 2,500 次姿态更新；每组 195 个相机与时刻组合均由 CPU 重新运行完整 SMPL，验证顶点、仿射矩阵和参照点运动。模型参数和变形数据仅保存在本机。

接着各自训练连续各向异性三维高斯 15,000 次，采用相同 52,109 个初始点、相同 105 张原生 1920×1080 照片、相同抽样次序、随机背景和优化规则。训练时刻是 744～756 中的七个偶数编号；六个奇数编号只用于开发验证。模型通过局部仿射变换同时携带高斯中心和完整协方差，颜色采用规范空间方向的二阶球谐函数。人体姿态在图像训练期间冻结。训练输入采用作者发布的前景透明度，不是人工语义真值；它与此前只修补第 750 帧的掩码实验条件不同。

两组最终分别有 237,066 与 249,560 个高斯。对每组全部 90 张预留时刻图像和 12 张训练诊断图，独立检查了原始照片、去畸变输入、408 张 PNG、102 张双列图及浮点误差指标。两组初始参数、抽样序列和最终优化器更新次数也逐项核对。没有锐化、插帧或将原照片融合进渲染结果。

估计拍摄时间的分支，预留时刻平均前景 PSNR 从 22.6015 提升到 24.1344 dB，90 张中 79 张改善、11 张下降；机位 08 平均下降 0.1926 dB。最差变化发生在机位 01、第 747 帧，下降 2.4975 dB，因此没有把平均改善写成逐帧改善。另有 12 张训练诊断图，平均提升 0.4124 dB。

![预留时刻的真实全人物对照](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 左原照片，中相同帧编号，右估计拍摄时间。

![PSNR 下降最大的机位与时刻](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera01-frame000747-full-person.png)
> 图源：同上 · 机位 01、第 747 帧 · 同一原生区域。

实际看图覆盖预先固定的 12 张全人物图、24 个头部／前臂／鞋部区域，并补看上述最差负例。抬脚边缘与部分衣物图案更集中，但脸部和眼镜重影、手指融合、鞋带模糊、玩偶拉花仍存在；遮挡处仍有厚边。**清晰人物重建仍未通过。**

连续检查另计算了全部 15 台相机的 75 组相邻预留时刻差分残差，平均值从 0.05359 降为 0.04152。该量同时包含运动、遮挡和掩码误差，不是光流补偿后的闪烁指标。四段视频均验证六帧完整解码；4 帧/秒仅用于按顺序观看，不代表经认证的实际拍摄速度。目视结论来自上述离散画面，没有声称逐帧观看全部视频。

视频：[机位 01](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera01-reserved-time-comparison.mp4)、[机位 04](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera04-reserved-time-comparison.mp4)、[机位 08](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera08-reserved-time-comparison.mp4)、[机位 17](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/camera17-reserved-time-comparison.mp4)。

这一结果仅是已知训练相机上的开发时间验证，不能替代新机位测试；机位 00、06、12 的相对拍摄时刻仍未在此实验中确定。估计的时间差只是候选解释，尚未认证真实相机时钟。人体模型本身还把预留身体关节点的投影误差中位数从自由轨迹的 2.636 增大到 5.938 像素，说明仍需改进姿态拟合。

报告：[本轮结论](../outputs/dynamic-human-enerf-articulated-control-064/final-summary.json)、[训练条件](../outputs/dynamic-human-enerf-articulated-control-064/protocol.json)、[真实渲染梯度与协方差检查](../outputs/dynamic-human-enerf-articulated-control-064/runtime-smoke.json)、[成对指标](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/summary.json)、[第一组独立像素复算](../outputs/dynamic-human-enerf-articulated-control-064/fixed_offsets/native-evaluation/independent-image-audit.json)、[第二组独立像素复算](../outputs/dynamic-human-enerf-articulated-control-064/estimated_offsets/native-evaluation/independent-image-audit.json)、[连续图像检查](../outputs/dynamic-human-enerf-articulated-control-064/temporal-comparison/summary.json)、[实际目视范围](../outputs/dynamic-human-enerf-articulated-control-064/visual-review.json)、[连续人体检查](../outputs/dynamic-human-enerf-continuous-body-063/final-summary.json)、[全部仿射复算](../outputs/dynamic-human-enerf-continuous-body-063/independent-audit.json)、[人体投影残差](../outputs/dynamic-human-enerf-continuous-body-063/projection-summary.json)。


### 直接多视角姿态修正与头部运动

在上一轮估计拍摄时间的分支上，固定形状、尺度、相机和时间偏移，直接拟合训练照片中的人体关键点，增加颈部与头部的连续旋转自由度。七个偶数时刻的 3,185 个观测参与 3,000 次优化；白衣人物背向相机时不使用鼻子和眼睛，手指没有作为 SMPL 约束。模型仍没有独立手指和玩偶运动。本次同时改动全身姿态拟合与头颈自由度，属于组合改动，尚不是只改变头部的消融。

为避免重复计算无关网格顶点，实现只计算目标关节及面部／脚部所需顶点；实际前向值与姿态梯度先同完整 SMPL 比对，最大差异分别为 1.79×10⁻⁷ 与 9.54×10⁻⁷。完成后又在 CPU 重新运行全部 195 个完整人体状态，核对仿射变换及全部投影残差。训练使用的观测编号全部为偶数，没有将奇数时刻关键点加入优化。

预留时刻中，身体与脚部的误差中位数从 5.938 降至 4.132 像素；头部从 7.550 降至 3.197 像素。但头部均方根误差仍有 31.341 像素，说明错误关键点和遮挡大残差没有消失；这些都是相对于网络预测的检查，不是真实关节误差。

随后从与上一轮完全相同的高斯初值开始，使用相同 105 张照片、抽样次序、背景、优化规则和 15,000 次更新，得到 238,502 个高斯。全部 102 张原生渲染的来源、PNG 像素和指标均独立复核；同样检查预先固定的 12 张全人物图、24 个细节区域和最差负例。

![头部细节中的明确改善](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 原照片、上一轮姿态、本轮直接拟合。

六个固定头部区域中有五个 PSNR 改善，平均提高 4.2594 dB，目视可见眼镜、脸部和头发的重影减少。与此同时，六个黑衣前臂区域平均下降 0.7443 dB，白衣前臂下降 1.2216 dB，鞋部下降 0.6765 dB。细节矩形中包含背景和遮挡，不能将这些矩形指标当作精确的人体部位分割指标。

![保留头部改善与其他区域问题的全图](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-full-person.png)
> 图源：同上 · 第 755 帧，预留时刻。

全前景平均 PSNR 从 24.1344 降至 23.5000 dB，90 张图中仅 12 张改善；15 台相机的平均变化均为负。最差变化为机位 16、第 745 帧，下降 2.7917 dB，玩偶出现更强的横向条纹。**本轮整体候选不替换上一轮基线，清晰人物重建仍未通过。**

![全前景指标下降最大的真实负例](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera16-frame000745-full-person.png)
> 图源：同上 · 原照片、上一轮姿态、本轮直接拟合。

75 组连续差分残差的平均值从 0.04152 增至 0.04302，四段六帧视频均完成解码检查。该量包含运动、遮挡与掩码误差，不能等同于真实运动或闪烁；目视范围仍为上述固定离散画面。视频：[01](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera01-reserved-time-comparison.mp4)、[04](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera04-reserved-time-comparison.mp4)、[08](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera08-reserved-time-comparison.mp4)、[17](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/camera17-reserved-time-comparison.mp4)。4 帧/秒只表示展示顺序，不认证拍摄速度。

下一步采用更明确的分离对照：保留原有身体运动和世界平移，只优化颈部、头部的连续旋转；若头部改善可保留，再研究手臂、衣物及手持物的残余运动。此处使用的是反复检查过的开发验证时刻，尚未完成新相机与新片段的最终泛化验证。

报告：[本轮结论](../outputs/dynamic-human-enerf-direct-pose-065/final-summary.json)、[姿态优化条件](../outputs/dynamic-human-enerf-direct-pose-065/protocol.json)、[前向与梯度比对](../outputs/dynamic-human-enerf-direct-pose-065/subset-forward-gradient-audit.json)、[全部人体状态复算](../outputs/dynamic-human-enerf-direct-pose-065/independent-audit.json)、[姿态误差摘要](../outputs/dynamic-human-enerf-direct-pose-065/projection-summary.json)、[原生图像复算](../outputs/dynamic-human-enerf-direct-pose-065/direct_reprojection/native-evaluation/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/summary.json)、[连续图像检查](../outputs/dynamic-human-enerf-direct-pose-065/temporal-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-direct-pose-065/visual-review.json)。


### 只优化头颈的隔离对照

保留064的身体姿态、世界平移、体形、尺度、相机与时间偏移，只用七个偶数时刻的546个头部预测观测拟合48个颈部／头部旋转系数，更新3000次。白衣人物的鼻子和眼睛继续排除；预测关键点不等于真实关节。195个相机时刻组合在CPU完整复算，非头部身体关节坐标最大变化为0；SMPL姿态修正项仍导致部分非头部网格顶点移动，最大约0.00734场景单位。

预留时刻468个头部观测的投影误差中位数从7.550降至3.355像素，身体和脚部从5.938降至5.936像素。头部误差均方根仍为31.204像素，未解决错误预测和遮挡大残差。

使用相同105张训练图、高斯初值、抽样顺序、随机背景与15000次更新，得到246435个高斯。102张原生渲染均完成源数据、保存像素与指标复算；实际查看12张固定全图、24个细节区域和机位05第747帧最差退化样例。

![隔离头颈运动后的完整画面](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 原照片、064、066。

90张开发时间验证图的全前景PSNR均值从24.1344增至24.2973dB，62张改善、28张退化。六个固定头部区域平均提高3.9776dB；黑衣前臂为−0.1861dB，白衣前臂为−0.1151dB，鞋部为+0.0618dB。区域矩形包含背景和遮挡，不能等同于精确人体部位分割。脸部改善明显，手指、玩偶和衣物局部仍未清晰恢复。

![保留的最差全前景退化样例](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera05-frame000747-full-person.png)
> 图源：同上 · 机位05、第747帧，PSNR变化−0.3284dB。

75组连续差分残差均值从0.04152降至0.04089；四段六帧对照视频均解码检查。该量包含真实运动、遮挡及掩码误差，不是光流补偿后的闪烁指标；只目视检查每个机位三个分离时刻，没有声称完整播放验收。视频：[01](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera01-reserved-time-comparison.mp4)、[04](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera04-reserved-time-comparison.mp4)、[08](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera08-reserved-time-comparison.mp4)、[17](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/camera17-reserved-time-comparison.mp4)。4帧/秒仅用于按顺序展示。

将066保留为下一轮局部变形的开发对照基础，**清晰人物重建仍未通过**。这些时间与相机已多次用于诊断，最终仍需分离的新时间窗口与相机验证。

报告：[结论](../outputs/dynamic-human-enerf-head-motion-066/final-summary.json)、[头部拟合条件](../outputs/dynamic-human-enerf-head-motion-066/protocol.json)、[完整人体复算](../outputs/dynamic-human-enerf-head-motion-066/independent-audit.json)、[隔离条件](../outputs/dynamic-human-enerf-head-motion-066/head-only-invariance-audit.json)、[投影摘要](../outputs/dynamic-human-enerf-head-motion-066/projection-summary.json)、[保存图像复算](../outputs/dynamic-human-enerf-head-motion-066/head_only/native-evaluation/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/summary.json)、[连续检查](../outputs/dynamic-human-enerf-head-motion-066/temporal-comparison/summary.json)、[看图范围](../outputs/dynamic-human-enerf-head-motion-066/visual-review.json)。


### 骨架之后的局部变形对照

在066的固定身体与头颈运动上，加入384个局部节点，每个人192个；节点由仅训练数据构成的初始视觉外壳确定，每个高斯关联4个同一人物的节点。3072条有向邻接边约束局部刚性。两个三维三次系数张量共6912个可训练标量，参考物理帧索引750处的残余变形严格为恒等。身份关联来自最近人体顶点，是近似绑定，并非实例真值；玩偶尚无独立节点类别。

RGB与透明度损失、105张训练照片、高斯初值、抽样、随机背景及15000次更新均同066一致；增加的归一化局部刚性、旋转幅度、平移幅度正则权重分别为10、0.001、0.1，训练前确定。图参数Adam初始学习率0.003，指数降至初值0.1倍。最终有223,234个高斯；相同更新次数不代表相同计算量或最终高斯数量。模型是本地研究扩展，未冒充作者官方基线。

训练前验证了恒等变换、刚体等变性、有限差分梯度、实际CUDA反向和复制／分裂绑定。训练后在CPU以SciPy独立复算全部384节点、195时刻组合的旋转，并在每个组合抽查512个高斯的加权线性变换；最小奇异值为0.971366。完整102张渲染保存像素和指标均另行复核。

![相同原生画面的全人物比较](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/camera04-frame000745-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站训练 · 原照片、066、067。

90张开发时间验证图的前景PSNR从24.2973变为26.2857dB，变化+1.9884dB，89/90张改善。固定区域平均变化：头部+0.9075dB、黑衣前臂+1.3899dB、白衣前臂+1.9584dB、鞋部+0.1810dB。每类6个区域，包含背景和遮挡，不能当作精确部位分割指标。

四个固定机位的玩偶拖影明显减轻，衣服图案与部分前臂边缘更集中，脸部改善也得以保留；手指、鞋带和交叉遮挡处仍不够清晰。 实际查看了12个固定全图、24个细节区域及全前景退化最大的样例，**清晰人物重建仍未通过**。

![最差变化的完整样例](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/camera02-frame000753-full-person.png)
> 图源：同上 · 机位02、第753帧，前景PSNR变化-0.0051dB。

75组连续差分残差均值从0.04089变为0.03226。它包含运动、遮挡及掩码误差，不等同于真实运动或闪烁。四段六帧视频全部解码检查，目视范围为上述离散画面；4帧/秒仅用于展示：[01](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera01-reserved-time-comparison.mp4)、[04](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera04-reserved-time-comparison.mp4)、[08](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera08-reserved-time-comparison.mp4)、[17](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/camera17-reserved-time-comparison.mp4)。

报告：[结论](../outputs/dynamic-human-enerf-residual-graph-067/final-summary.json)、[节点准备](../outputs/dynamic-human-enerf-residual-graph-067/graph-preparation.json)、[数学检查](../outputs/dynamic-human-enerf-residual-graph-067/math-audit.json)、[实际CUDA检查](../outputs/dynamic-human-enerf-residual-graph-067/runtime-smoke.json)、[训练条件](../outputs/dynamic-human-enerf-residual-graph-067/residual_graph/graph-protocol.json)、[训练后独立复算](../outputs/dynamic-human-enerf-residual-graph-067/trained-graph-independent-audit.json)、[图像复算](../outputs/dynamic-human-enerf-residual-graph-067/residual_graph/native-evaluation/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/summary.json)、[连续检查](../outputs/dynamic-human-enerf-residual-graph-067/temporal-comparison/summary.json)、[看图范围](../outputs/dynamic-human-enerf-residual-graph-067/visual-review.json)。

训练几何的补充诊断：按最近SMPL表面的蒙皮权重近似归类，两个人的手腕／手部区域分别关联28、33个图节点，且分类可能包含手持物。这排除了“整个手部没有节点”的简单解释，但不能证明节点足以描述手指运动。仅据节点数不推断模糊原因，详见[节点覆盖诊断](../outputs/dynamic-human-enerf-residual-graph-067/part-support-diagnostic.json)。


### 留出相机与人物层诊断

068冻结066头颈基线与067局部变形模型，增加00、06、12三个相机的外观验证。这些相机未参与064—067的高斯初始化、训练与姿态拟合，但出现在更早的开发材料中。本轮不是全新最终测试。帧744—756中，21张偶数帧只用于拟合每个相机一个时间偏移，18张奇数帧用于外观评价；全部RGB保持原生1920×1080，按既有float32标定去畸变。新相机内外参沿用数据集，没有优化。

以固定身体和足部三次轨迹为参照，偏移估计为00：−0.488929帧、06：−0.281492帧、12：−0.705588帧。仅采用偶数帧RTMW身体与足部预测，置信度至少0.4、每个关节至少4个支持观测、零偏移中位投影误差不超过40像素；稳健损失尺度3像素，偏移限制±2帧。CPU独立标量最小化与记录结果最大差约3.39×10⁻⁷帧。遮挡导致部分关节预测混入另一人物，尤其机位06，验证预测的RMSE略变差；不能据此宣称完成真实同步标定。

三个分支各18张原生渲染：头颈基线加估计时间、局部变形加估计时间、局部变形加零偏移。模型检查点前后哈希一致，没有外观、身体或变形图更新；原195个身体上下文逐项保留。54张像素结果、216张PNG和54个原生人物对照均独立复算。

![留出机位实际全人物对照](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 原照片、头颈基线、局部变形。

局部变形相对头颈基线，18张前景PSNR均改善，平均22.8839→23.7577dB（+0.8738dB）。每机位平均变化：00为+0.3628dB、06为+0.7647dB、12为+1.4939dB。固定投影前臂框的平均变化为黑衣+1.0987dB（7/9改善）、白衣+1.7789dB（9/9改善）。这些框包括背景与遮挡者；机位06的部分“黑衣前臂框”主要显示白衣衣服，不能把变化当作目标前臂清晰度。

![前景指标改善最小的画面](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera00-frame000753-full-person.png)
> 图源：同上 · 机位00、帧753，前景PSNR变化+0.0062dB；它是最小改善，不是全前景退化。

![时间偏移的冻结模型对照](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-clock-comparison.png)
> 图源：同上 · 原照片、零时间偏移、估计时间偏移。

在同一个冻结局部变形模型上，估计时间相对零偏移的平均前景PSNR提高2.3411dB，但机位06的753、755帧没有改善。时间差分残差均值0.04647→0.03979，共15个相邻奇数帧组合；它包含运动、遮挡与掩码误差，并非运动补偿后的闪烁。三段六帧视频全部解码检查，4fps只用于展示：[00](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/camera00-reserved-time-comparison.mp4)、[06](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/camera06-reserved-time-comparison.mp4)、[12](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/camera12-reserved-time-comparison.mp4)。实际目视范围为9张预定全人物图、18个预定细节框、最小改善图以及3张751帧时间对照，没有声称完整视频播放检查。手指、鞋部和严重遮挡仍未通过清晰度验收。

068记录：[协议](../outputs/dynamic-human-enerf-heldout-cameras-068/protocol.json)、[时间输入独立检查](../outputs/dynamic-human-enerf-heldout-cameras-068/clock-input-independent-audit.json)、[二维观察局限](../outputs/dynamic-human-enerf-heldout-cameras-068/observation-visual-review.json)、[原生像素复算](../outputs/dynamic-human-enerf-heldout-cameras-068/independent-image-audit.json)、[成对指标](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/summary.json)、[连续检查](../outputs/dynamic-human-enerf-heldout-cameras-068/temporal-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-heldout-cameras-068/visual-review.json)、[结论](../outputs/dynamic-human-enerf-heldout-cameras-068/final-summary.json)。

069继续冻结067模型，在同样三个相机的745、751、755帧共9个上下文分别输出联合颜色、每个人物的联合颜色与不透明度贡献、单独人物颜色与不透明度、期望深度。身份沿用最近人体顶点绑定，非实例真值。特征渲染与068原始RGB、alpha逐像素一致；颜色贡献之和与总颜色、不透明度贡献之和与总不透明度的最大绝对差均小于5×10⁻⁶。保存数组、18张对照图及统计均在CPU独立复算。

![冻结模型中的实际人物层](../outputs/dynamic-human-enerf-actor-layers-069/camera06-frame000745-actors.png)
> 图源：同上 · 原照片、联合结果、黑衣人物层、白衣人物层。

按释放前景matte>0.95并腐蚀5×5像素得到的1600009个内部像素中，38294个联合alpha<0.9（2.3934%）；RGB平均绝对误差>0.08的119058个内部像素中，96758个仍有alpha≥0.9（81.2696%）。阈值在分层渲染前固定。这支持“残差并非全部由低alpha造成”，不证明特定几何或人物归属是唯一原因。

机位06的黑衣胸前缺口、白衣层边缘残留，以及00/12机位遮挡手部的肤色散点，是后续可见性与归属约束的检查对象。单独人物层展示了被遮挡的预测区域，但没有这个机位的无遮挡真值；不会据此强行补满透明度或按留出RGB重新指定人物。目视检查5张人物层图、3张机位06贡献图，其余图只做像素和统计检查。

069记录：[固定诊断条件](../outputs/dynamic-human-enerf-actor-layers-069/protocol.json)、[逐图贡献统计](../outputs/dynamic-human-enerf-actor-layers-069/render-summary.json)、[独立像素复算](../outputs/dynamic-human-enerf-actor-layers-069/independent-audit.json)、[看图与下一步](../outputs/dynamic-human-enerf-actor-layers-069/visual-review.json)、[结论](../outputs/dynamic-human-enerf-actor-layers-069/final-summary.json)。保留067为开发基线；下一步仅依据训练数据分析归属与可见支持，再进行同预算训练对照。


### 人物归属监督的同预算训练对照

070以067局部变形模型为基线，从相同初始化重新训练15000次更新。原始105张照片、matte、固定身体与时间模型、384节点图、随机背景、抽样次序及学习率完全一致。额外监督来自第750帧的SAM2.1 Hiera Tiny人物分割：复用旧训练照片的人工提示，在当前精确去畸变RGB上重新推理；15张标签图逐一查看，排除13、17机位的跨人物小区域误标，13个机位共2117574个保守内部像素用于归属损失。不是增加13张照片，而是对已有13张训练照片增加弱标签。手持物和不确定交界不参与该损失。

原始绑定距离检查发现，仅约0.0273%的高斯比自身绑定人物更接近另一人体至少0.05场景单位，因此没有把全体高斯重新分配身份。训练机位的联合贡献检查仍显示重叠机位存在错误归属；三维最近距离不能代替可见像素归属。

实际完成1799次含归属损失的更新，总更新15000次，权重0.05。新增特征通道的RGB与alpha在实际CUDA测试中与原渲染器一致，逐人物贡献之和与总alpha误差小于5×10⁻⁶；位置、透明度与变形参数梯度非零。训练后CPU独立复算全部195个身体上下文的节点旋转、固定样本高斯的线性变换，保持正行列式。最终224,837个高斯，基线为223,234；相同更新次数不等于相同运算量。SAM预测、许可人体派生数组及检查点不作为网站下载资源。

![用于弱监督的实际标签](../outputs/dynamic-human-enerf-actor-supervision-070/masks/camera07-labels-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap、SAM2.1及本站筛选。

90张已知训练机位的留出时间图，前景PSNR均值26.2857→26.3203dB，49/90张改善；18张另外三个机位的奇数帧为23.7577→23.7903dB，12/18张改善。这三个机位保持068的时间估计，没有为新模型重拟合。全部120张原生结果逐项复算。

![固定遮挡机位对照](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 原照片、067基线、070归属监督。

固定全身图和局部图中，两版外观多数近似；手指、抬起的鞋和细衣纹仍偏软。机位06第745帧胸前仍有浅色竖向条纹，第755帧遮挡手部仍有拖影。单独渲染黑衣人物时胸前缺口仍存在，白衣人物层仍带有重叠区域残留。监督改善了部分训练像素的人物归属，但未恢复明显的精细结构。 采用结论：不替换067开发基线；归属损失仅保留为局部证据，清晰人体目标未通过。

![留出机位中变化最差的完整画面](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/camera00-frame000747-full-person.png)
> 图源：同上 · 机位00、帧747，前景变化-0.0561dB。

与训练弱标签不一致的主导人物贡献像素从10440变为8715，不能直接解释为真实实例精度。固定模型的逐人物渲染仍需目视对照；被遮挡表面没有该机位的无遮挡真值。时间检查保留75组已知机位、15组另外机位的图像差分残差，以及7段六帧视频；视频4fps只用于展示，全部机械解码，实际看图范围以记录为准，不宣称完整视频播放审阅。

报告：[输入与标签规则](../outputs/dynamic-human-enerf-actor-supervision-070/mask-protocol.json)、[标签看图筛选](../outputs/dynamic-human-enerf-actor-supervision-070/mask-visual-review.json)、[标签独立检查](../outputs/dynamic-human-enerf-actor-supervision-070/mask-independent-audit.json)、[绑定距离](../outputs/dynamic-human-enerf-actor-supervision-070/anchor-diagnostic.json)、[CUDA检查](../outputs/dynamic-human-enerf-actor-supervision-070/actor-runtime-smoke.json)、[实际训练条件](../outputs/dynamic-human-enerf-actor-supervision-070/training-protocol-audit.json)、[已训练变形复算](../outputs/dynamic-human-enerf-actor-supervision-070/trained-graph-independent-audit.json)、[已知机位比较](../outputs/dynamic-human-enerf-actor-supervision-070/paired-comparison/summary.json)、[另外机位比较](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/summary.json)、[已知机位连续变化](../outputs/dynamic-human-enerf-actor-supervision-070/temporal-comparison/summary.json)、[另外机位连续变化](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/temporal-comparison/summary.json)、[人物层独立核验](../outputs/dynamic-human-enerf-actor-supervision-070/actor-layers/independent-audit.json)、[看图范围](../outputs/dynamic-human-enerf-actor-supervision-070/visual-review.json)、[最终判断](../outputs/dynamic-human-enerf-actor-supervision-070/final-summary.json)。


### 运动控制节点密度的同预算对照

071由067基线重新初始化，只改变运动图的采样密度及相应邻接、绑定距离：每个人物192→768节点，总节点384→1536，运动参数6912→27648。固定105张原生偶数帧、52109个初始高斯、身体运动、时间偏移、随机背景、图像抽样和15000次更新；四节点权重与八邻接数不变，正则项系数不变，但实际空间平滑范围随邻接距离改变。没有采用070的人物归属监督。相同更新预算不是相同计算量。

实际完成15000次更新，最终218579个高斯，训练600.42秒，峰值分配显存1.663GB。所有120张原生渲染完成像素复算，90张已知机位留出时间图平均前景PSNR为26.2857→26.4459dB，77张改善；18张新机位图为23.7577→23.7353dB，7张改善。三个机位00、06、12的平均变化分别为−0.0115、−0.1198、+0.0640dB；最差新机位是06第745帧，变化−0.2220dB。固定手臂区域可能包含遮挡者，不能把区域分数称为隐藏手臂精度。

![固定机位的原生完整对照](../outputs/dynamic-human-enerf-fine-graph-071/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站训练 · 原照片、067、071。

![新机位中变化最差的完整画面](../outputs/dynamic-human-enerf-fine-graph-071/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 机位06、帧745。

机位04的衣物图案和部分前臂边缘稍有改善；手指、抬起的鞋和细衣纹仍偏软。新机位00与12的多数画面差别很小；机位06第745帧胸前的浅色竖向条纹、第755帧遮挡手部的拖影仍在。增加运动节点未带来稳定的新视角清晰度改善。 已查看64个唯一完整或局部对照图，包含预定区域及两个集合各自最差画面。已知机位75组时间差分残差均值0.032258→0.031527，新机位15组为0.039789→0.039708；7段六帧视频机械解码成功，不宣称完整播放审阅。CPU用SciPy复算全部1536节点、195个身体上下文的旋转与固定512高斯样本的线性混合，最小行列式0.9059，最小奇异值0.9522。该检查只排除所查样本的数值退化，不证明几何正确。

结论：不替换067开发基线；已知机位的增益未稳定延伸到新机位。单随机种子、同一人物对、重复使用的开发机位与时段，不能推广为普遍方法排名。下一项从067基线隔离原生训练matte中的局部漏标，保持模型容量不变。

报告：[预先固定协议](../outputs/dynamic-human-enerf-fine-graph-071/ablation-protocol.json)、[实际训练条件](../outputs/dynamic-human-enerf-fine-graph-071/training-protocol-audit.json)、[CPU变形检查](../outputs/dynamic-human-enerf-fine-graph-071/trained-graph-independent-audit.json)、[已知机位](../outputs/dynamic-human-enerf-fine-graph-071/paired-comparison/summary.json)、[另外机位](../outputs/dynamic-human-enerf-fine-graph-071/heldout/paired-comparison/summary.json)、[时间变化](../outputs/dynamic-human-enerf-fine-graph-071/temporal-comparison/summary.json)、[另外机位时间变化](../outputs/dynamic-human-enerf-fine-graph-071/heldout/temporal-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-fine-graph-071/visual-review.json)、[结论](../outputs/dynamic-human-enerf-fine-graph-071/final-summary.json)。


### 三张训练图的局部掩码修正

072从067完全相同的52109高斯和384节点图开始，只改变机位01第750、752、754帧中固定矩形内的训练matte；其他102张训练matte与全部原始RGB保持不变。SAM2.1 Hiera Tiny先在七张原生偶数帧上产生候选，七张全部看图，接受其中三个明显漏掉前臂的候选。规则为SAM前景腐蚀3×3后，在固定区域与有效像素内与原matte取最大值，共改变11561个原生像素，其中5962个原值小于0.5。区域包含少量袖口和衣服边缘；不是纯皮肤真值，也不用于补全隐藏表面。旧039、053试验已有掩码质量的局部或混合结果，此次不是首次发现掩码问题，而是当前动态模型上的隔离验证。

![三帧修正中一个实际输入例子](../outputs/dynamic-human-enerf-forearm-matte-repair-072/candidates/camera01-frame000750-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap、SAM2.1及本站筛选 · 输入诊断。

实际完成15000次更新，其中423次抽到三张修正图，最终224,722个高斯，训练581.07秒。相同初始参数、随机背景、抽样序列、身体与时间模型、学习率和增密日程均已复算。三张matte既改变合成颜色目标也改变alpha目标，不能把效果进一步拆成其中一项的独立作用。

90张已知机位留出时间图沿用原matte指标，平均前景PSNR为26.2857→26.2652dB，37/90张改善。18张另外机位图为23.7577→23.7939dB，15/18张改善。原matte可能漏掉真实皮肤，所以另外固定四张奇数帧的可见皮肤内部多边形，直接对照原照片；只看源图选区，在任何新模型评估之前保存并查看，未用于训练。

这四个内部区域共1604个原生像素，合并MSE后计算的PSNR为28.9135→29.7743dB。该数值只描述同一已知机位的可见皮肤内部，不代表轮廓、手指、新机位或全身清晰度。没有按预测alpha除法还原颜色，比较的是模型实际合成到0.5灰背景的像素。

![留出时刻的前臂原生比较](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/camera01-frame000751-raw-comparison.png)
> 图源：同上 · 原照片、原matte目标、067基线、072修正。

机位01的可见前臂内部颜色更接近原照片，四个预定皮肤内部区域合并PSNR提高0.861dB，三处改善、一处略退步；这些内部像素在基线中已接近完全不透明。完整人物与局部图仍显示手指边界、抬起的鞋和细衣纹偏软，新机位06胸前的浅色条纹及遮挡手部拖影仍在。修正三张训练matte没有带来明显的整体清晰度改善。 采用结论：不替换067开发基线；保留局部肤色收益与掩码漏标证据，整体人物清晰度仍未通过。当前机位与时段反复用于开发，单种子结果不代表重复性或独立最终测试。120张原生结果、90组时间变化均保留；实际看图范围和数值复算范围分别记录。

![固定遮挡机位中的完整对照](../outputs/dynamic-human-enerf-forearm-matte-repair-072/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 原照片、067基线、072局部修正。

报告：[输入修正规则](../outputs/dynamic-human-enerf-forearm-matte-repair-072/protocol.json)、[七张候选看图](../outputs/dynamic-human-enerf-forearm-matte-repair-072/mask-visual-review.json)、[输入独立复算](../outputs/dynamic-human-enerf-forearm-matte-repair-072/mask-independent-audit.json)、[预先固定训练条件](../outputs/dynamic-human-enerf-forearm-matte-repair-072/ablation-protocol.json)、[实际训练核验](../outputs/dynamic-human-enerf-forearm-matte-repair-072/training-protocol-audit.json)、[已训练变形复算](../outputs/dynamic-human-enerf-forearm-matte-repair-072/trained-graph-independent-audit.json)、[已知机位](../outputs/dynamic-human-enerf-forearm-matte-repair-072/paired-comparison/summary.json)、[另外机位](../outputs/dynamic-human-enerf-forearm-matte-repair-072/heldout/paired-comparison/summary.json)、[原照片区域定义](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/protocol.json)、[原照片直接比较](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/summary.json)、[已知机位时间变化](../outputs/dynamic-human-enerf-forearm-matte-repair-072/temporal-comparison/summary.json)、[另外机位时间变化](../outputs/dynamic-human-enerf-forearm-matte-repair-072/heldout/temporal-comparison/summary.json)、[实际看图](../outputs/dynamic-human-enerf-forearm-matte-repair-072/visual-review.json)、[最终判断](../outputs/dynamic-human-enerf-forearm-matte-repair-072/final-summary.json)。


### 手部观测与姿态输入核查

对既有195张原生训练机位图像的780只手预测进行检查。15个机位第750帧的60个原生手部区域已全部查看：遮挡处有落到玩偶、衣服和另一个人上的预测。旧RTMW输出的16380个手部点分数全部大于0.4，最小值约1.115；这个阈值没有筛掉任何点。它是未经概率校准的响应分数，不能按百分比或可见性解释。

旧预测的手部范围在原生图像中的长边中位数约50.4像素，在288×384全身检测输入里约20.5像素。这是预测范围统计，不是人工测量的真实手部尺寸。按旧身体时间偏移插值到参考时刻后，逐相机留出的手指预测重投影误差中位数约10.75像素，1200个检查中472个小于6像素；这些数值描述旧预测的相互一致性，不是真值误差，不能单独证明模糊原因。

![原生照片与局部手部候选](../outputs/dynamic-human-enerf-hand-observability-073/hand-pilot-rgb/camera08-all-points.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap、RTMW与RTMPose Hand · 输入观测比较，非重建结果。

在第750帧先按源图可见性选22只右手，用官方RTMPose Hand模型产生局部候选。14张四列对照全部查看；所有21个点均画出，避免不同模型的分数范围改变显示后造成假改善。候选未加入高斯训练；被玩偶遮住的左手也未用于先导。当前没有手指真值和时序验证，仍须检查对应关系。

核对两个模型的官方压缩包后，发现Normalize配置要求转成RGB，而旧调用链将BGR交给没有颜色转换的包装库。已用原生图像独立重建官方流程：修正后的归一化输入逐元素完全相同；旧调用也能精确复现历史坐标，证实问题发生在输入通道顺序。高斯训练图像此前一直是原始RGB，此问题作用于姿态检测。官方包内的部分裁剪尺寸字段过时，实际采用ONNX输入尺寸与对应解码尺寸，未照抄不一致的默认值。

已重提取195张图像的390个人物预测，并对原图哈希、相机参数、框和训练/留出时间划分复核。三维轨迹、相机时间偏移、2500步身体运动与3000步头部运动已重拟合，195个相机时刻的身体变换通过CPU独立复算。旧相机支持集合和规范人体参考保持固定，以便隔离这次修复；没有把手指候选混入训练。修正后的关键点及拟合一致性不等于重建清晰度，后者由单独的15000步高斯对照判定。

报告：[手部诊断](../outputs/dynamic-human-enerf-hand-observability-073/final-summary.json)、[原照片检查](../outputs/dynamic-human-enerf-hand-observability-073/source-visual-review.json)、[分数独立复算](../outputs/dynamic-human-enerf-hand-observability-073/independent-audit.json)、[局部候选检查](../outputs/dynamic-human-enerf-hand-observability-073/hand-pilot-rgb/visual-review.json)、[通道顺序核验](../outputs/dynamic-human-enerf-rgb-pose-correction-074/preprocessing-audit.json)、[同一目标的轨迹比较](../outputs/dynamic-human-enerf-rgb-pose-correction-074/same-target-trajectory-audit.json)、[运动重拟合条件](../outputs/dynamic-human-enerf-rgb-pose-correction-074/motion-refit-protocol.json)、[CPU变换复算](../outputs/dynamic-human-enerf-rgb-pose-correction-074/head/independent-audit.json)、[高斯对照条件](../outputs/dynamic-human-enerf-rgb-pose-correction-074/gaussian-ablation-protocol.json)。


### 正确颜色输入后的完整重建对照

修正姿态检测的颜色通道后，身体、头部运动与相机时间偏移已重新拟合，再以相同105张原始训练照片、初始化、384节点变形图、抽样顺序和15000次更新训练高斯。掩码、损失权重和增密设置保持一致，没有加入局部手部候选。本轮得到228414个高斯，完成102张原机位、18张另外机位的1920×1080原生渲染；保存的预测与原图裁剪经过数值复算，按预先固定范围及最差退步图实际查看38张对照。

![人物交叉处的原照片、基线和修正姿态后的重建](../outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本地实际渲染 · 机位06、第745帧 · 原生像素，未锐化。

原训练机位的90张留出时间图，平均前景PSNR从26.286变为26.253dB，38张改善；另外三个机位的18张图从23.758变为23.745dB，8张改善。两组像素时间差分残差也略有增大。时间差分包含运动、遮挡和掩码影响，不能单独当作闪烁指标。

上图胸前的浅色条纹仍在。固定完整人物与局部对照显示，两版外观多数相近，手指、头发和抬起的鞋仍偏软；第755帧人物交叉处仍有明显手臂重影。下图保留鞋子的真实细节差距，不用平均分掩盖局部问题。

![原生鞋部细节对照](../outputs/dynamic-human-enerf-rgb-pose-correction-074/paired-comparison/camera01-frame000751-dark_shoe.png)
> 图源：同上 · 机位01、第751帧 · 原照片、基线、修正姿态后的重建。

**正确的RGB预处理继续保留，本轮渲染模型不替换既有开发基线。** 这次修复没有带来稳定的人体清晰度改善。此前的支持集合和规范人体形状为控制变量而保持固定，并非所有上游几何都重建了一遍。以上机位与时间片段已经反复用于开发，且只有一个训练随机种子，不是独立最终测试。清晰手指、鞋部、复杂遮挡及完整大场景仍未完成。

报告：[最终结论](../outputs/dynamic-human-enerf-rgb-pose-correction-074/final-summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-rgb-pose-correction-074/visual-review.json)、[相同训练条件核验](../outputs/dynamic-human-enerf-rgb-pose-correction-074/training-protocol-audit.json)、[训练后变形独立复算](../outputs/dynamic-human-enerf-rgb-pose-correction-074/trained-graph-independent-audit.json)、[原机位数值](../outputs/dynamic-human-enerf-rgb-pose-correction-074/paired-comparison/summary.json)、[另外机位数值](../outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout/paired-comparison/summary.json)、[原机位时间变化](../outputs/dynamic-human-enerf-rgb-pose-correction-074/temporal-comparison/summary.json)、[另外机位时间变化](../outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout/temporal-comparison/summary.json)。


### 局部图像对应的歧义

用13台训练相机的七个偶数时刻、共91张原生照片，检查照片纹理的运动。没有更新高斯或人体参数，也没有使用另外三个机位和奇数帧。参考帧的人物内部来自此前核对过的保守标签；通过往返跟踪、直接与逐帧跟踪一致性、图块相关性和前景范围筛选，5933个特征种子留下7277条跨帧候选观测。它们是局部图像匹配，不是人体关节真值。

![前臂边界上图块相似但位置发生滑动的候选](../outputs/dynamic-human-enerf-native-motion-tracks-075/review/largest-model-disagreement.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 原照片上的特征标注，非重建效果图。

上图的候选沿手臂边界滑动。即使匹配图块的相关峰值达到0.9959，附近仍有几乎同样高的分离峰，峰值间隔只有约0.000089。相关性不是正确对应的概率。加入不依赖模型误差的局部匹配唯一性筛选后，这个候选被剔除，总候选数降为5117条。

两版冻结模型都具有足够源像素不透明度和人物贡献的共同子集为5072条。模型采用可见高斯中心投影位移的加权平均，与图像候选的端点差异中位数分别为0.793和0.801像素。这个比较偏向可跟踪的衣物纹理；高斯中心的加权位移也不等于每个有限高斯内部像素的精确表面运动，不能据此宣布人体运动已正确或模糊原因已经确定。

![位于手部框内但实际落在衣服下摆的特征](../outputs/dynamic-human-enerf-native-motion-tracks-075/unique-patches/review/detail00-camera03-frame000744.png)
> 图源：同上 · 前两列为原照片，后两列加特征标注；颜色标注不参与训练。

筛选后落在“手部框”的10条观测全部查看，其中8条实际属于衣服下摆或相邻裤子，另2条只在手掌附近。没有获得可作为独立指尖真值的标签。8张固定机位时间覆盖图、全部10条手部框候选和最大模型差异图均已查看，另检查了筛选前的两个诊断图。**这些候选没有送入手指训练。** 后续以原照片中手部、鞋部附近的颜色监督为对照，检验小区域的损失权重，而不把匹配不确定性转成强运动约束。

报告：[图像跟踪条件](../outputs/dynamic-human-enerf-native-motion-tracks-075/protocol.json)、[模型位移定义](../outputs/dynamic-human-enerf-native-motion-tracks-075/model-motion-protocol.json)、[唯一性筛选条件](../outputs/dynamic-human-enerf-native-motion-tracks-075/unique-patches/protocol.json)、[筛选结果](../outputs/dynamic-human-enerf-native-motion-tracks-075/unique-patches/summary.json)、[共同子集结果](../outputs/dynamic-human-enerf-native-motion-tracks-075/unique-patches/model-summary.json)、[实际看图](../outputs/dynamic-human-enerf-native-motion-tracks-075/visual-review.json)、[最终判断](../outputs/dynamic-human-enerf-native-motion-tracks-075/final-summary.json)。


### 局部颜色权重与跨视角清晰度

这组对照检验手部、鞋部附近的颜色误差是否被全身平均稀释。保持105张原始训练照片、前景透明度、正确RGB预处理后的冻结人体运动、初始化、384个局部变形节点、抽样顺序和15000次更新一致，只在70张图的局部窗口内增加权重为0.2的RGB绝对误差。另35张图因窗口中的可见部位存在明显错配而保留原损失。窗口不是关节或人体部位真值。

完成15000次更新，得到280237个高斯，训练约586秒，框架统计的显存分配峰值约2.05GB；最终高斯数会随新增损失的梯度而变化，这不是固定点数的消融。102张原机位与18张另机位的1920×1080渲染、原始像素裁图及误差均已复核。预定37张对照、额外一张最差变化图及12张扩大后的手鞋裁图，共50张图片实际查看。

![原照片、原目标与增加局部权重后的完整人物](../outputs/dynamic-human-enerf-detail-weight-076/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 灰色为检查人物使用的合成背景。

原机位90张留出时刻的人物区域平均PSNR为26.253→26.304dB，52张改善；另外三个机位的18张图为23.745→23.628dB，仅2张改善。时间差分残差在原机位略降、另机位略升；这个量包含运动、遮挡和掩码误差，不是独立的闪烁真值。七段视频已完成编码与解码检查，本轮视觉判断来自上述50张PNG。

![扩大到完整手掌与指尖后的原生对照](../outputs/dynamic-human-enerf-detail-weight-076/expanded-detail/camera04-frame000751-dark_forearm.png)
> 图源：同上 · 机位04、帧751 · 每列为相同原始像素范围。

主体、衣物图案和手持物已有可辨外观，但指尖仍粘连，抬起的鞋仍有拖影，胸前条纹和交叉遮挡的重影也没有解决。**这轮不替换保留的开发基线。** 对照只能说明当前窗口与权重组合不足以恢复细节，不能据此断言所有局部加权方法无效，或已经证明模糊只来自运动模型。下一步重点检查失败部位的局部形变与多视角对应。当前仍是同一场景、单一随机种子和反复使用的开发视角，不是最终独立测试；静态大场景与人体的联合重建仍未完成。

报告：[训练条件](../outputs/dynamic-human-enerf-detail-weight-076/protocol.json)、[实际梯度检查](../outputs/dynamic-human-enerf-detail-weight-076/residual_graph/detail-gradient-audit.json)、[共同条件核验](../outputs/dynamic-human-enerf-detail-weight-076/training-protocol-audit.json)、[变形数值核验](../outputs/dynamic-human-enerf-detail-weight-076/trained-graph-independent-audit.json)、[原机位指标](../outputs/dynamic-human-enerf-detail-weight-076/paired-comparison/summary.json)、[另机位指标](../outputs/dynamic-human-enerf-detail-weight-076/heldout/paired-comparison/summary.json)、[看图范围](../outputs/dynamic-human-enerf-detail-weight-076/visual-review.json)、[最终判断](../outputs/dynamic-human-enerf-detail-weight-076/final-summary.json)。


### 局部时间基函数与人物细节

这组对照在原三次多项式运动上增加七个以参考时刻为零的三次B样条项，旋转和平移均可学习。保持105张原始训练照片、冻结身体运动、384个空间控制节点、52109个初始高斯、抽样顺序、颜色损失、正则项、增密日程与15000次更新一致。图运动参数从6912个增加到23040个；这是本地研究扩展，不是4K4D或OmniRe的官方方法复现。

训练约485秒，最终223105个高斯，框架记录的显存分配峰值约1.67GB。完成102张原机位、18张另机位的1920×1080渲染；实际查看37张预定对照、2张最差变化图与12张扩大后的手鞋区域，共51张。新增参数为零时与旧渲染逐像素一致，训练后四组运动参数均经历15000次更新；这些核对说明实现按预期工作，不等于清晰度达标。

原机位90张留出时刻的平均前景PSNR为26.253→26.504dB，72张改善；另外三个机位18张图为23.745→23.818dB，12张改善。其中机位06平均仍下降0.022dB。时间差分残差由0.03254→0.03163、0.04003→0.03945，分别对应两组机位；这个指标包含运动、遮挡与掩码误差，不能单独作为闪烁真值。七段视频已做编码与解码检查，本次视觉判断来自PNG对照。

![原照片、全局三次时间模型与增加局部时间基函数后的鞋部](../outputs/dynamic-human-enerf-temporal-basis-078/expanded-detail/camera04-frame000751-dark_shoe.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 相同原生像素区域。

全身外观和衣服图案已有可辨结构，新增时间项带来小幅改善，但抬起的鞋仍有拖影，手指仍粘连，细衣纹仍偏软。下面的另机位还保留胸前条纹与遮挡重影，不能用更好的平均分数掩盖这些问题。

![另机位中仍未消除的胸前条纹与遮挡错误](../outputs/dynamic-human-enerf-temporal-basis-078/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同上 · 机位06、帧745 · 原照片、原时间模型、新时间模型。

保留该时间模型作为下一轮候选，旧开发基线继续保存。这组结果来自单一随机种子和反复使用的开发片段；下文补充了同一场景另一互不重叠运动片段的重新训练对照，尚不构成独立场景验证。**人物清晰度尚未达到要求，静态大场景与人体联合重建也未完成。**

报告：[训练条件](../outputs/dynamic-human-enerf-temporal-basis-078/protocol.json)、[真实渲染与梯度检查](../outputs/dynamic-human-enerf-temporal-basis-078/runtime-probe.json)、[共同条件](../outputs/dynamic-human-enerf-temporal-basis-078/training-protocol-audit.json)、[变形数值检查](../outputs/dynamic-human-enerf-temporal-basis-078/trained-graph-independent-audit.json)、[原机位指标](../outputs/dynamic-human-enerf-temporal-basis-078/paired-comparison/summary.json)、[另机位指标](../outputs/dynamic-human-enerf-temporal-basis-078/heldout/paired-comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-temporal-basis-078/visual-review.json)、[本轮结论](../outputs/dynamic-human-enerf-temporal-basis-078/final-summary.json)。


### 另一运动片段中的时间表示

在同一 actor5_6 采集中，将帧720至732作为第二段运动，参考时刻为726，与原744至756片段互不重叠。重新估计该片段的人体与头部运动、生成参考几何并训练外观，保留先前的相机时间偏移。两支共享105张偶数帧训练照片、57678个初始高斯、384个空间节点、随机种子、图像抽样、损失与增密日程，各完成15000次更新；唯一表示差别为是否加入局部时间B样条。没有将旧750时刻的身体运动外推到本片段，也没有复用旧外观模型。

原生1920×1080评测共120个相机时刻，每支各渲染一次，共240张。包括90张训练机位的奇数帧、18张另三个机位的奇数帧，以及12张偶数帧诊断图。两支按原始透明度定义的前景PSNR，在前两组分别为24.257→24.637dB、22.491→22.699dB；分别80/90、15/18张改善。机位02的六张留出帧平均下降0.040dB，机位06帧725下降0.224dB。新机位沿用此前另一片段估计的相机时间偏移，这不是完全未知相机条件。

![第二片段的完整人物对照](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、全局三次时间模型、局部时间模型。

全身、衣服图案和腿部已有明确结构，新增时间项的变化仍然有限。已逐一查看24张预定全身对照、24张手持物或鞋部对照，以及4张额外的最差变化图，共52张。张开的手掌有半透明重影，玩偶边缘破碎，抬起鞋子的边缘和鞋带仍软；人物之间及鞋边有墙面、地面残片。下面保留失败区域，不把平均分数改善视为清晰度验收。

![张手与玩偶局部仍不清晰](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：同上 · 机位04、帧731 · 相同原生像素范围，无增强。

![鞋底与鞋带仍偏软](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-both_person_feet.png)
> 图源：同上 · 机位04、帧731。

两组机位的相邻时间差分残差分别由0.03975降至0.03851、0.04541降至0.04434；它包含真实运动、遮挡和掩码误差，不能视作单独的闪烁真值。本轮没有进行视频视觉验收。最终全局模型255371个高斯、局部模型252984个高斯，训练约508秒和543秒，框架记录的显存分配峰值约1.72GB。240张渲染的指标与960张保存PNG均已从实际数组复核。

两个片段的平均改进支持继续保留局部时间表示，但仍是同场景、单一随机种子的开发验证。**张手、鞋部和遮挡尚未达到清晰要求，长序列、独立场景与建筑背景联合重建仍未完成。**

报告：[配对指标](../outputs/dynamic-human-enerf-second-interval-079/comparison/summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-second-interval-079/visual-review.json)、[完整结论](../outputs/dynamic-human-enerf-second-interval-079/final-summary.json)。

### 训练掩码中的墙面阴影与张手细节

检查第二片段机位04、07、13的726和732两帧，共六张实际训练图。原照片、提供的前景透明度与按该透明度生成的训练目标逐像素对应。机位07的玩偶上方墙面阴影被保留；机位13帧732出现大块墙面误前景；机位04的墙面碎片与脚边地面也进入监督。这是在新片段中再次确认此前的掩码问题，不是已经完成修复。

![墙面阴影进入训练目标](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera07-frame000726-training-target.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 机位07、训练帧726 · 原照片、前景透明度、灰底训练目标。

![遮挡机位的大块墙面误前景](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera13-frame000732-training-target.png)
> 图源：同上 · 机位13、训练帧732。

手部不能用同一个原因全部解释。机位04帧732的监督目标仍保留清楚的张开手掌与弯曲手指，而相邻留出帧的重建有明显重影。这里还需要局部姿态、表面覆盖和跨视角对应的约束。

![原始训练目标保留了张手结构](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera04-frame000732-hands-training-target.png)
> 图源：同上 · 机位04、训练帧732 · 原照片、前景透明度、实际训练目标；三列都不是重建结果。

上述工作只完成输入诊断；下节补充实际的训练侧前景支持对照。评测保留原图、原掩码和既定细节范围，避免更换测试口径。[诊断范围与结论](../outputs/dynamic-human-enerf-matte-diagnosis-080/summary.json)。


### 前景支持修正与手部清晰度

相同运动片段720至732中，先对全部105张训练照片生成分割候选，再查看15个机位各7帧。直接使用候选会删掉部分玩偶，以及遮挡机位中的后方人物，因此没有采用这些直接预测。最终仅在9个机位的63张图中保守裁去部分误前景，手、玩偶、头颈和脚部附近保留原监督；另6个机位的42张图完全保留原掩码。去掉的原前景像素累计403794个，不等于获得了完整的人体语义真值。

局部时间B样条模型从相同57678个初始高斯重新训练15000次。身体运动、相机、图像裁剪、随机种子、抽样、随机背景、损失和增密设置均保持一致；唯一输入差别是上述训练侧透明度。约543秒完成训练，最终252373个高斯，框架显存分配峰值约1.72GB。评测使用原来的120个机位时刻，原生1920×1080，未改动测试掩码或进行图像增强。

![保守修正前后的完整人物重建](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、原训练支持、保守修正支持。

原机位90张留出帧的前景PSNR为24.637→24.643dB，48张改善、42张退化；另外3个机位18张为22.699→22.661dB，6张改善、12张退化。既定12个手与玩偶窗口平均变化−0.081dB，12个脚部窗口为+0.025dB。窗口包含周围区域，原掩码也包含部分背景，因此这些不是精确的人体部位语义分数。

![张开的手掌仍有重影](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：同上 · 相同原生像素范围。

![抬脚轮廓仍偏软](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-both_person_feet.png)
> 图源：同上 · 机位04、留出帧731。

逐一查看全部24个预定手物与脚部对照，以及12张全身对照，共36张，覆盖两组机位各自三个最差变化案例。部分墙面残片减弱，但张手、玩偶边缘和抬起的鞋没有出现稳定清晰度改善。机位03帧731下降0.376dB、机位09帧723下降0.297dB；这些退化保留在原记录中。时间差分残差在原机位基本不变，新机位略升，也不能支持时间稳定性提升。

**这轮候选不替换原开发基线。**结论只针对这次不完整的保守前景修正：它不足以解决当前手部模糊，不能据此否定完整且准确的分割。下一步将重点检查原生分辨率手部观测、局部运动自由度和跨视角对应，再引入通过核验的约束。

完整记录：[结论](../outputs/dynamic-human-enerf-support-correction-081/final-summary.json)、[固定口径配对指标](../outputs/dynamic-human-enerf-support-correction-081/comparison/summary.json)、[24个细节窗口](../outputs/dynamic-human-enerf-support-correction-081/comparison/detail-metrics.json)、[实际看图范围](../outputs/dynamic-human-enerf-support-correction-081/render-visual-review.json)。120组数组指标与960张保存PNG均已独立复核，原模型和原始数据完整保留。


### 连续手部观测与掌部运动核验

对象限定为 actor5_6 的浅衣人物右手，帧720至732。先查看15个原机位、三个训练时刻的90个左右手原生裁图，排除抓握玩偶且遮挡严重的左手。右手195张原生照片分别采用三种裁剪范围，完成585次官方 RTMPose-Hand 推理；105张偶数帧用于几何拟合候选，90张奇数帧仅用于检查。原测试机位00、06、12均未进入这项诊断。输入RGB与模型发布包的标准归一化流程逐元素一致。

![同一只手的原照片与裁剪敏感性](../outputs/dynamic-human-enerf-hand-motion-082/candidate-panels/camera04-candidates.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与 RTMPose-Hand 实测 · 从左至右：原照片、全身预测、局部预测、较小裁剪、较大裁剪。

保留已有相机时间偏移，只用偶数帧在五个共同物理时刻进行插值。每个关节每次排除一个目标相机，以其余相机三角化，要求至少四个支持机位、重投影误差小于3像素、正深度及足够的视线夹角。三种观测方式共4725项检查，所有失败项也计入总数。这是检测预测的一致性，不能当作真实关节位置精度。

全身模型与局部模型的手腕误差中位数分别为3.06、2.47像素；指尖则由4.81变为5.94像素。以全部375个指尖目标为分母，低于3像素的比例由24.8%降到18.1%。只保留裁剪变化不超过3像素的支持点后，有效指尖覆盖率仅40%，不能把筛选后的较小误差解释为完整手部恢复。

进一步将04机位整段排除，使用其他机位的偶数帧拟合连续三维轨迹。裁剪稳定方案只有9个关节得到足够支持，不能构成完整的手指运动。原照片对照仍能看见部分曲线离开皮肤、落到玩偶或背景。

![没有参加拟合的04机位中的手部轨迹](../outputs/dynamic-human-enerf-hand-motion-082/motion-panels/camera04-motion.png)
> 图源：同上 · 原照片、局部检测、全身预测的三维拟合、局部预测的三维拟合、裁剪稳定点的三维拟合 · 各行均为未拟合的奇数帧。

再对手腕及四个掌指关节采用固定形状的刚性掌部模型，实际优化21个运动参数，使用324个训练观测。五点间距离保持不变，但04机位65个对应投影的误差中位数由独立曲线的4.47变为5.37像素，未改善独立机位一致性；该模型也没有表达指尖屈伸。

![固定掌部形状前后的原生投影检查](../outputs/dynamic-human-enerf-hand-motion-082/motion-panels/camera04-palm.png)
> 图源：同上 · 原照片、局部检测、独立三维掌部点、固定形状掌部。

**局部手指标签和掌部运动均未加入高斯训练，人物清晰度仍未通过。**本轮没有新的人体RGB重建，以上彩色线条是预测投影。已实际查看38张原生照片对照，并独立复算195个输入文件、4725项跨机位记录、9750项轨迹投影和975项掌部投影。下一项重建干预应优先处理清晰张手区间的局部图像对应与遮挡，不能把这些不可靠指尖直接加成强约束。

记录：[结论与决策](../outputs/dynamic-human-enerf-hand-motion-082/final-summary.json)、[跨机位一致性](../outputs/dynamic-human-enerf-hand-motion-082/geometry-summary.json)、[输入与投影复算](../outputs/dynamic-human-enerf-hand-motion-082/numerical-audit.json)。


### 手部法向运动与真实重建对照

本轮仍使用 actor5_6 帧720至732。先在15个训练机位的728、730、732帧运行官方 RAFT-Large 与 DIS 光流，并检查45个手部区域分割候选。排除玩偶混入的14、16机位后，在13个机位筛出1527个手背、手腕及少量皮肤褶皱对应。源、目标均须位于保守手部内部，往返误差低于0.5像素，两种算法沿图像梯度方向的位移差不超过1像素。只使用这个法向分量；这些观测不是完整光流真值，也不是手指关节标签。04机位正面张手区只有2个点，覆盖不足。

![原生手部照片与保留下来的对应](../outputs/dynamic-human-enerf-open-hand-flow-083/normal-flow/camera13-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站对应筛选 · 机位13 · 每行依次为源图、源图保留点、目标保留点、目标原图。

两支模型从同一个15000步检查点开始，各续训1500步。仅优化27个局部运动节点，其余357个节点，以及252984个高斯的几何、颜色、不透明度与协方差均精确冻结；身体运动、相机和时间偏移保持不变。两支使用相同105张训练照片、随机抽样、背景与正则化。候选额外加入权重0.05、误差尺度2像素的平滑L1法向运动项。训练分别约96秒，框架显存分配峰值约1.86GB；没有图像锐化、上采样或生成补画。

1527个训练观测的平均绝对法向残差由1.245降到0.807像素，下降35.2%。这是对估计运动的拟合程度。另按训练后几何重新计算可见性，残差为1.250→0.824像素，改善仍存在；因此不能仅用固定可见权重解释这个下降。

![相同留出时刻的完整人物重建](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/camera04-frame000731-full-person.png)
> 图源：同上 · 机位04、留出帧731 · 原照片、仅图像续训、加入法向运动约束。

评测保留120个原生1920×1080机位时刻：90张原训练机位的奇数帧、18张另外三个机位的奇数帧，以及12张训练诊断图。原掩码、前景口径与24个细节窗口均不变。两组留出图的平均前景PSNR分别为24.646→24.539dB、22.668→22.616dB；第一组27张改善、63张退化，第二组9张改善、9张退化。12个手与物品窗口平均变化−0.243dB，脚部窗口基本不变。时间差分残差也没有改善。这是同一场景的开发验证，不能宣称通过独立新场景验收。

![留出机位的手与玩偶仍然破碎](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/camera12-frame000731-both_hands_and_objects.png)
> 图源：同上 · 机位12、留出帧731 · 原生像素范围。

已查看24个细节对照与12个全身对照，包含两组留出图各自三个最差变化案例。04机位的张手仍有重影，06、07、12机位的手物交界仍有白黑碎片；大尺度衣服和身体可以辨认，手指没有稳定恢复。最差的原机位案例10/731下降1.254dB；另一组12/731下降0.548dB。**本轮候选不替换原基线，清晰人体与完整场景均未完成。**后续需检验局部几何、外观和运动的共同适配，不能继续把降低这项训练残差当作清晰度提升。

记录：[结论](../outputs/dynamic-human-enerf-open-hand-flow-083/final-summary.json)、[全部配对结果](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/summary.json)、[细节窗口](../outputs/dynamic-human-enerf-open-hand-flow-083/comparison/detail-metrics.json)、[独立数值复核](../outputs/dynamic-human-enerf-open-hand-flow-083/numerical-audit.json)、[实际看图范围](../outputs/dynamic-human-enerf-open-hand-flow-083/render-visual-review.json)、[当前可见性诊断](../outputs/dynamic-human-enerf-open-hand-flow-083/visibility-diagnostic/summary.json)。240张预测的数组、480张PNG与训练抽样一致性均已核验，原始数据与旧模型完整保留。


### 手部局部联合优化与连续画面

这次检验的是：同时允许手部几何、外观和运动调整，是否能修复张手时的碎片。仍从原15000步模型出发，两支各续训3000步，采用完全一致的105张训练照片、抽样顺序、背景、学习率与正则化。候选增加权重0.05、尺度2像素的法向运动项，仍使用前次筛出的1527个估计对应；它们主要覆盖手背与手腕，不能视为手指真值。所有留出划分与原始掩码保持不变。

手部范围由13个训练机位的728、730、732帧确定，需通过保守分割内部、正深度、与原模型渲染深度接近及至少两个机位支持的筛选。选出的6330个高斯可更新位置、尺度、旋转、不透明度和球谐颜色；关联的22个现有节点可更新运动，但更新只作用于这些高斯。其余246654个高斯参数和362个节点精确冻结，身体运动、相机、时间偏移保持不变。本轮没有新增或分裂高斯。该范围仍可能包含手物交界的混合结构，不是准确人体分割真值。

![完整人物的四列对照](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、帧731 · 原照片、原模型、局部图像联合优化、再加入运动约束。

两支实际完成各3000步，耗时约158、163秒，框架显存分配峰值约1.75GB。120个机位时刻均按1920×1080真实渲染，再与原模型配对比较；360张预测及其720张颜色／透明度PNG已独立数值复核。90张原训练机位的留出时间图，前景PSNR为24.637、24.634、24.616dB；另外三个机位的18张图为22.699、22.688、22.679dB。仅图像联合优化与原模型接近，增加运动项后两组均略有退化。

![正面手掌在留出时刻仍然破碎](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera04-frame000731-target_right_hand.png)
> 图源：同上 · 保留160×160原生窗口，无锐化或补画。

12个原有手与物品窗口，相对原模型平均变化为−0.017、−0.074dB；12个脚部窗口数值完全不变。额外固定的6个目标手窗口变化为+0.003、−0.012dB，也不足以支持清晰度提升。这些指标使用预先确定的图像窗口与原掩码交集，并非独立标注的手部真值区域。训练观测的法向运动残差则从1.252降到0.731像素，说明约束拟合改善仍未转化成真实手指细节。

![同一固定窗口中的三个连续片段代表时刻](../outputs/dynamic-human-enerf-hand-joint-084/temporal/camera04-comparison.png)
> 图源：同上 · 从上到下为721、727、731帧 · 每列顺序与前图一致，窗口在整个片段保持不变。

另外完整渲染了04、13两个机位的720至732连续13帧，再裁出固定窗口：共78张模型局部预测，全部PNG与数组一致。已逐帧查看26个机位时刻，从抓握转向张手时，皮肤与玩偶碎片持续重叠；13机位的手背与玩偶接触处也没有稳定改善。04机位相邻帧图像差分误差为0.05922、0.05919、0.06006；13机位为0.04969、0.04972、0.05061，没有一致改善。差分误差比较的是预测与原照片的亮度变化，不等同于运动真值误差。

连续对照视频：[04机位](../outputs/dynamic-human-enerf-hand-joint-084/temporal/camera04-hand-continuity.mp4)、[13机位](../outputs/dynamic-human-enerf-hand-joint-084/temporal/camera13-hand-continuity.mp4)。均为13帧、5帧／秒的慢速检查版本；播放速度不是原始采集帧率。

![另一机位的张手与玩偶边缘](../outputs/dynamic-human-enerf-hand-joint-084/comparison/camera12-frame000731-both_hands_and_objects.png)
> 图源：同上 · 留出机位12、帧731。

实际查看范围共40张对照图／连续帧拼图，包含12个全身案例、12个手物案例、6个固定目标手窗口、8张覆盖全部连续帧的检查图及2张代表时刻拼图。全身案例覆盖两支模型在两组留出数据中的全部前三差案例。**两支候选均不替换原模型；手指清晰度仍未达标。**这是同一场景反复开发的验证，不能视为独立新场景验收。

下一项诊断针对运动控制的空间覆盖。选中高斯的规范坐标包围盒最大边长约0.222场景长度单位，22个原有关联节点均不在这个包围盒内；高斯到最近关联节点的距离中位数为0.106，约为最大边长的48%。从同一训练模型的选中高斯中心提出32个局部节点，可把最近距离中位数降到0.0215。**这仅是未训练的空间覆盖方案，不是手指关节标注，也没有证明图像会改善。**还需核对节点在原照片中的投影，验证零初始化不改变原渲染，再做相同条件的训练对照。高斯数量、控制节点数量和可信观测覆盖是不同问题，不能相互替代。

记录：[本轮结论](../outputs/dynamic-human-enerf-hand-joint-084/final-summary.json)、[配对评测](../outputs/dynamic-human-enerf-hand-joint-084/comparison/summary.json)、[细节窗口](../outputs/dynamic-human-enerf-hand-joint-084/comparison/detail-metrics.json)、[连续画面检查](../outputs/dynamic-human-enerf-hand-joint-084/temporal/summary.json)、[节点空间覆盖](../outputs/dynamic-human-enerf-hand-joint-084/spatial-diagnostic/summary.json)、[数值核验](../outputs/dynamic-human-enerf-hand-joint-084/numerical-audit.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-joint-084/render-visual-review.json)。原始数据与所有旧模型保留；静态背景联合重建和独立场景验证仍未完成。


### 局部运动节点与手部表面

检验的问题是：在上一轮局部几何、颜色与运动联合优化之上，增加局部运动自由度，能否让张开的手指从玩偶中分离。32个新增节点取自原模型选中的6330个高斯中心，随原身体与粗层变形移动，再施加局部残差变形；每个高斯连接4个邻近细节点，节点间建立256条有向邻接边。既有22个节点与6330个高斯仍可更新，其余246654个高斯和362个节点保持冻结。没有新增高斯，也没有新增指尖真值。

新增变形的零初始化与原模型渲染精确一致。两支分别仅用图像、再加入法向运动约束，各训练3000步；105张训练照片、抽样、背景、学习率与上一轮相同。此次也加入细层的刚性和变形幅度正则化，因此对比的是完整的细层方案，不能把差异单独归因于节点数量。投影检查已查看13张原生照片拼图：部分节点仍落在衣服或玩偶边缘，它们是原模型中的候选材料位置，不是准确的皮肤节点。

![细节点方案的完整人物对照](../outputs/dynamic-human-enerf-hand-fine-085/comparison/camera04-frame000731-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 原照片、原模型、细节点图像分支、细节点运动约束分支。

120个机位时刻均按1920×1080渲染，360张预测及720张颜色／透明度PNG独立复核。90张留出时间图的前景PSNR分别为24.637、24.637、24.621dB；另外三个机位的18张图为22.699、22.690、22.679dB。相对上一轮同预算粗节点方案，细层图像分支在两组中的平均变化仅为+0.0023、+0.0021dB，尚不足以说明视觉清晰度改善。

![手指分离仍未恢复](../outputs/dynamic-human-enerf-hand-fine-085/comparison/camera04-frame000731-target_right_hand.png)
> 图源：同上 · 原生160×160窗口，无锐化或补画。

6个固定目标手窗口相对原模型平均变化为+0.019、+0.006dB，但12个手与物品窗口为−0.013、−0.052dB；12个脚部窗口完全不变。1527个估计对应的法向误差从1.236降到0.673像素，也没有恢复手指轮廓。已实际查看40张对照图，包含所有两组留出数据的前三差案例、12个手物窗口、6个目标手窗口及覆盖两个机位全部13连续帧的拼图。**两支候选均不替换原模型。**

![抓握到张手过程中的实际渲染](../outputs/dynamic-human-enerf-hand-fine-085/temporal/camera04-comparison.png)
> 图源：同上 · 721、727、731帧 · 每列顺序同前图。

连续检查使用固定窗口，78张预测PNG与数组逐一一致。04、13机位的相邻帧图像差分误差，在图像分支中仅有小幅下降，运动分支则升高；张手后的皮肤与玩偶碎片仍然相互混杂。[04机位连续视频](../outputs/dynamic-human-enerf-hand-fine-085/temporal/camera04-hand-continuity.mp4)、[13机位连续视频](../outputs/dynamic-human-enerf-hand-fine-085/temporal/camera13-hand-continuity.mp4)为13帧、5帧／秒的检查副本，播放速度不是采集帧率。

另将训练完成的细层临时旁路，保留其余参数不变，在105个原训练机位时刻测量细层的实际作用：图像分支与运动分支的选中高斯中心投影位移中位数分别为0.165、0.412像素，95%分位数为0.686、6.025像素。该统计等权计入遮挡和偏离表面的中心，并非表面误差；它只证明细层已经影响几何，不能证明几何正确。继续加节点或加大运动权重，尚无清晰度证据支持。

补充检查沿用13个训练机位的728、730、732帧及原手部掩码，将选中与被冻结高斯的指示属性一起进行透明度合成，保留原排序和遮挡。各组贡献之和与总透明度数值一致，因而测量的是实际渲染贡献，而非高斯中心是否落在掩码内。

![原照片、原重建、可更新高斯的可见贡献占比与估计手部掩码](../outputs/dynamic-human-enerf-hand-contributions-086/camera04-contribution.png)
> 图源：同上 · 机位04，728、730、732帧 · 第三列黑到白对应占比0至1；第四列为估计掩码。

39个机位时刻等权平均，可更新高斯承担估计手部区域81.5%的贡献；约13.0%的区域像素由冻结组贡献占多数，这些像素承载约12.4%的RGB绝对误差。该结果不支持“多数手部错误因为被冻结而无法修改”的解释。图中白色贡献仍延伸成皮肤之外的碎片，说明可更新范围覆盖手部，不等于其中已有正确的手指表面。04机位728帧的掩码还明显遗漏部分手掌，不能将掩码外的贡献全部判成错误。

13张、39个原生画面的实际查看及数组复核已完成；[侧面机位的同类对照](../outputs/dynamic-human-enerf-hand-contributions-086/camera13-contribution.png)、[贡献诊断](../outputs/dynamic-human-enerf-hand-contributions-086/final-summary.json)、[看图记录](../outputs/dynamic-human-enerf-hand-contributions-086/visual-review.json)保留在这里。此项没有训练新模型。后续梯度分解和局部拟合已完成，见下节；结果不支持手部梯度被其余画面淹没的解释。所有结论仍属于同一场景开发，完整背景与独立新场景验证未完成。

记录：[本轮结论](../outputs/dynamic-human-enerf-hand-fine-085/final-summary.json)、[配对评测](../outputs/dynamic-human-enerf-hand-fine-085/comparison/summary.json)、[同预算粗细节点比较](../outputs/dynamic-human-enerf-hand-fine-085/coarse-versus-fine.json)、[细节窗口](../outputs/dynamic-human-enerf-hand-fine-085/comparison/detail-metrics.json)、[连续画面](../outputs/dynamic-human-enerf-hand-fine-085/temporal/summary.json)、[细层实际作用](../outputs/dynamic-human-enerf-hand-fine-085/fine-effect/summary.json)、[数值核验](../outputs/dynamic-human-enerf-hand-fine-085/numerical-audit.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-fine-085/render-visual-review.json)。


### 局部拟合、时间共享与多视角一致性

这组对照检验三件事：手部图像是否真正驱动局部参数，现有高斯是否能拟合清晰手指，以及局部改善能否同时满足其他视角和其他时刻。输入始终采用原生1920×1080照片、原相机和原前景掩码；局部窗口包含手、玩偶和衣物，不是准确的人体部位真值。模型是本地研究扩展，以下结果不代表论文官方成绩。

**梯度没有被其余画面淹没。**对原模型和细节点图像分支，各检查105张训练照片，按同一全图归一化系数拆分手部窗口与其余区域；SSIM完整邻域按中心归属，正则项另计。窗口平均只占人物裁图4.61%，但选中高斯位置的聚合梯度范数约为其余区域的198倍和131倍，六类高斯参数的局部梯度与总梯度方向均高度一致。这里比较的是同一参数族的梯度，不混加不同单位，也不等同于Adam实际更新。此项只做诊断，[完整分解与限制](../outputs/dynamic-human-enerf-hand-gradient-087/final-summary.json)保留备查。

**固定运动，单时刻与多时刻表面拟合仍不能通过。**两支均从同一模型出发，只更新6330个候选高斯的位置、尺度、旋转、不透明度与球谐系数；其余246654个高斯、全部384个运动节点及绑定保持不变。两支各3000步，机位抽样、背景和学习率一致；一支只见732帧的15个训练机位，另一支见720至732之间7个偶数帧。没有增加高斯或修改掩码。相同总步数意味着每张照片被采样次数不同，因此这是拟合范围的诊断，不能把全部差异归结为时间冲突。

![单时刻拟合与多时刻拟合在留出时间中的实际表现](../outputs/dynamic-human-enerf-hand-time-scope-088/temporal/camera04-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 721、727、731帧 · 原照片、原模型、只拟合732帧、拟合7个时刻。

单时刻分支在732帧的15个手部窗口平均提高0.515dB，但手指仍有碎片；90张其他时刻图像的前景PSNR下降0.392dB，另外三个机位的18张图下降0.232dB。多时刻分支在两组中分别下降0.011和0.017dB，手指也未稳定恢复。134个机位时刻、402张预测的颜色和透明度PNG，以及两个机位各13连续帧，共882次PNG数组核验已完成；实际查看33张拼图或单图，覆盖全部15个同帧手部窗口和两段完整连续序列。两支均不采用。[单／多时刻结论](../outputs/dynamic-human-enerf-hand-time-scope-088/final-summary.json)、[数值核验](../outputs/dynamic-human-enerf-hand-time-scope-088/numerical-audit.json)、[看图记录](../outputs/dynamic-human-enerf-hand-time-scope-088/render-visual-review.json)、[04机位连续视频](../outputs/dynamic-human-enerf-hand-time-scope-088/temporal/camera04-hand-continuity.mp4)、[13机位连续视频](../outputs/dynamic-human-enerf-hand-time-scope-088/temporal/camera13-hand-continuity.mp4)保留在此；视频为5帧／秒慢速检查副本。

**单视角能拟合清楚，其他视角却失真。**再用相同初始模型、6330个高斯、损失、学习率、背景和3000步预算，只拟合04机位的732帧。该机位手部窗口PSNR由22.818提升至25.164dB，手掌与手指轮廓明显恢复；其他14个已知机位的手部窗口却由20.860降至17.523dB，三个原留出机位的全前景均值由22.804降至22.171dB。

![拟合视角变清楚，但其他视角出现几何碎片](../outputs/dynamic-human-enerf-hand-view-scope-089/comparison/single-view-versus-consistency.png)
> 图源：同上 · 上排04、下排13机位 · 原照片、原模型、15视角拟合、单视角拟合。

18个机位的54张预测、108张颜色／透明度PNG独立复核；15个手部窗口及04、00、06、12机位全身图实际查看。该分支只能说明单图拟合能力，不能作为三维或连续模型交付。[单视角诊断](../outputs/dynamic-human-enerf-hand-view-scope-089/final-summary.json)、[完整18视角评测](../outputs/dynamic-human-enerf-hand-view-scope-089/comparison/summary.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-view-scope-089/render-visual-review.json)。

**拍摄时刻有影响，但统一时刻不是已验证的修复。**沿用原估计时间偏移，18个机位的名义732帧实际在模型中对应730.604至733.767的有效帧索引。以04机位的状态为参照，02机位选中高斯的投影位移中位数约24.58像素，17机位则仅约0.098像素；这些是模型预测的几何变化，绝不是观测误差。17机位在单视角拟合后同样明显退化，因此不能把问题全部归因于时间差。[18机位投影与15个可见区域的独立核验](../outputs/dynamic-human-enerf-hand-capture-geometry-090/final-summary.json)。

进一步保持15个训练机位、732帧、同一初始参数和3000步抽样不变，只让这6330个高斯在所有相机中采用04机位的同一个世界坐标状态、协方差与颜色参考方向，其余身体保持原渲染。参考机位的初始RGB／透明度精确复现，其他高斯几何精确不变；候选手部的初始世界几何则因干预而改变。结果中，其他14个训练机位手部窗口相对原捕获时刻拟合由21.383降至20.901dB，三个留出机位全前景由22.813降至22.734dB，仍有指间混合与碎片。所有18视角和108张PNG已复核，12张实际拼图／单图已查看。**此方案也不采用，原标定和时间偏移保持原值。**[统一手部状态对照](../outputs/dynamic-human-enerf-hand-snapshot-091/final-summary.json)、[实际看图记录](../outputs/dynamic-human-enerf-hand-snapshot-091/render-visual-review.json)、[04机位原生窗口](../outputs/dynamic-human-enerf-hand-snapshot-091/comparison/camera04-frame000732-hand.png)。

当前结论是：单个视角有清晰拟合能力，但正确的多视角手部表面尚未建立；单纯加运动自由度、扩大损失权重或强制统一时刻，都没有通过现有画面验证。近同步机位对应与双视角、局部密度对照现已完成，结果见下一节。既有估计指尖和掩码仍不作为真值。上述均是同一场景的开发对照；独立新场景、完整背景联合重建和移动人体清晰度验收仍未完成。


### 双视角拟合与局部密度控制

**几何对应的支持范围。**04／17机位在沿用的时间模型中只差0.00891帧，但相机中心相距约4.063场景长度单位，原图显示不同的手掌表面。原生分辨率SIFT、双向比率0.7匹配没有得到手部窗口对应点，不能据此初始化手指。另检查8组邻近训练机位，窗口内虽有通过极线、正深度、重投影和交会角筛选的点，实际查看的样本大多是玩偶纹理，只有少量手背／前臂候选。匹配筛选先于极线门限；9组基本矩阵与重投影已用另一种矩阵推导复核，但这不等于语义身份或指尖真值得到确认。[几何检查与限制](../outputs/dynamic-human-enerf-hand-pair-geometry-092/final-summary.json)。

**双视角拟合恢复两张图，却未恢复其他方向。**从同一开发对照出发，仅使用04／17机位732帧，更新6330个既有局部高斯3000步，运动、其他246654个高斯及全部绑定冻结。初始模型、损失、学习率、背景采样和预算与前面的单视角对照一致，只改变机位采样。

![双视角恢复与跨视角失败的原生图像对照](../outputs/dynamic-human-enerf-hand-two-view-093/comparison/two-view-versus-consistency.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、17、08机位 · 原照片、原模型、单视角拟合、双视角拟合。

两个拟合机位的手部窗口平均PSNR从21.460提升到23.781dB，手形与手指轮廓明显恢复；其他13个已知机位却从20.918降到17.443dB，出现拉丝、碎片和玩偶变形。三个原留出机位的全前景均值从22.804降到21.788dB。此处窗口包含手、衣物和玩偶，分数并非指尖精度。18机位108张颜色／透明度PNG与指标已复核，全部15个手部窗口及3个留出机位全身图已查看。分支不采用，不宣称跨时间验证完成。[双视角结论](../outputs/dynamic-human-enerf-hand-two-view-093/final-summary.json)、[看图记录](../outputs/dynamic-human-enerf-hand-two-view-093/render-visual-review.json)。

**局部密度控制有实际更新，但清晰度仍不足。**恢复732帧15机位，与固定密度对照使用完全一致的3000步相机和背景采样、学习率、初始参数及损失。400至2000步每200步，对局部高斯按平均归一化投影位置梯度执行克隆／分裂，并剔除低不透明度点。梯度阈值为0.0002，大小分界为场景尺度的0.01倍，分裂尺度除以1.6，局部不透明度剔除阈值0.005，无不透明度重置。这是采用gsplat操作的局部研究改动，不是完整官方3DGS复现。

![局部自适应密度与固定密度的手部对照](../outputs/dynamic-human-enerf-hand-adaptive-094/comparison/adaptive-density-hand.png)
> 图源：同上 · 04、17机位 · 原照片、原模型、固定密度15视角拟合、自适应密度15视角拟合。

实际发生2006次克隆、130次分裂、1384次剔除，局部高斯由6330变为7082；其余246654个高斯完全不变，子高斯继承原绑定，全部384个运动节点冻结。相对固定密度对照，04机位手部窗口提高0.068dB，其他14个训练机位均值提高0.169dB，三个留出机位全前景均值则由22.813变为22.811dB。轮廓有小幅改善，但手指仍不完整。18机位108张PNG与指标复核、全部15个手部窗口及3个留出全身图查看后，该分支仍未通过清晰度验收，不替换开发对照。[密度实验结论](../outputs/dynamic-human-enerf-hand-adaptive-094/final-summary.json)、[密度与冻结边界核验](../outputs/dynamic-human-enerf-hand-adaptive-094/density-audit.json)、[看图记录](../outputs/dynamic-human-enerf-hand-adaptive-094/render-visual-review.json)。

当前保留原连续模型作为开发对照。下一步检查局部表面与连续运动的耦合约束；这些同一场景开发实验没有完成独立新场景验收，也没有完成静态背景联合重建。不会通过增加采样机位数、训练步数或高斯数量来代替真实画面验收。


### 局部连续运动自由度的同预算对照

从保留的079连续模型出发，两组各训练3000步，使用完全一致的105张训练图、相机／帧／背景采样和表面学习率。只更新6330个手部邻域高斯的表面属性；其余246654个高斯、原384个运动节点、绑定、相机与时间偏移均冻结，无增密。新增平移场分别采用32个共享节点或6330条逐高斯轨迹，参数数为672和132930；时间基函数、位移限制和正则化相同。这是局部研究改动，不能称为官方方法复现。

各轨迹用7个紧支撑三次B样条，参考时刻726、归一化时间尺度6帧；系数以双曲正切限制在场景尺度的0.03倍。原始系数学习率0.003、末端衰减到三分之一；归一化位移平方均值权重0.1，相邻整数帧二阶差分平方均值权重10。两组只增加平移，没有新增协方差／球谐坐标旋转或手指关节监督。与更早的细节点实验相比还有学习率与正则化差异，不能混为只改变节点数的消融。

![相同连续时间监督下的手部原生对照](../outputs/dynamic-human-enerf-hand-trajectories-095/comparison/native-motion-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位731帧 · 原照片、原模型、节点平移、逐高斯平移。

90张已知机位留出时刻的平均前景PSNR由24.637变为24.663／24.664dB；18张原留出机位、留出时刻图由22.699变为22.717／22.722dB。15个同一时刻手部窗口均值仅提高0.031／0.033dB；6个目标右手窗口提高0.201／0.147dB。这些窗口包含衣物与手持物，不能解释为指尖误差。逐高斯分支在10机位729帧的全前景下降0.169dB，在12机位729帧下降0.145dB。

实际查看15个同一时刻手部窗口、6个目标手部窗口、12个双手与物品窗口、6组最差／原留出全身图，以及04／13机位各13个连续帧。全身轮廓仍可辨认，但两组手指都破碎或粘连；逐高斯模型在多个掌面视角出现黑斑。张手过程的后段也没有恢复清晰解剖结构。两组均不替换079开发对照。[完整结论](../outputs/dynamic-human-enerf-hand-trajectories-095/final-summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-hand-trajectories-095/visual-review.json)。

连续对照：[04机位](../outputs/dynamic-human-enerf-hand-trajectories-095/temporal/camera04-hand-continuity.mp4)、[13机位](../outputs/dynamic-human-enerf-hand-trajectories-095/temporal/camera13-hand-continuity.mp4)。视频为5fps慢放查看副本，原始PNG逐帧保留；不是拍摄帧率声明。全部104个区域图、26幅对照与24组相邻差分已核验，实际看图覆盖全部26个时刻窗口。[时间评测](../outputs/dynamic-human-enerf-hand-trajectories-095/temporal/summary.json)。

134个原生1920×1080上下文、4个模型的1072张颜色／透明度PNG和指标已独立复核。新增位移在234个相机时刻上下文中以NumPy重建，最大差异小于5×10⁻⁹场景长度单位，参考时刻位移严格为零；这些是实现一致性检查，不是位移真值。[运动检查](../outputs/dynamic-human-enerf-hand-trajectories-095/motion-audit.json)。时间摘要中原先误写的PNG计数78已改为104，实际循环与数据本来就检查104张，图像和指标未改变。

该结果只说明当前平移残差设计与预算不足以恢复手部，不能推断所有局部运动表示都无效。所有机位仍属于反复使用的同场景开发验证，独立新场景、完整长序列和静态背景联合重建尚未验收。手部结构先验接入已开始：用户提供的SMPL+H与MANO模板保留在本地，先核验姿态预测和多视角投影，再决定是否用于训练；模型模板完整不等于重建结果清晰。


### 手部关节先验的接入与可用性

用户提供的SMPL+H与MANO压缩包已接入本地。SMPL+H中性模型的6890个顶点、52个蒙皮关节和手指旋转梯度已检查；HaMeR使用独立MANO右手模板，输出778个顶点和21个关节。旧模板的Chumpy数组仅转换为数值数组，逐项数值完全一致，原文件与许可保留。模型文件、权重和原始三维参数均未加入网站资源。

官方HaMeR权重已完整下载、校验并加载，无缺失或多余键，加载后的全部MANO缓冲与用户模板相同。105张原训练图分别使用0.85、1.0、1.15三个手部上下文裁剪，完成315次推理；不使用00、06、12机位，也不使用奇数留出时刻。RGB顺序、裁剪仿射、归一化、关节重投影与旋转矩阵均已核验。推理阶段约33.7秒、峰值显存2.61GB；该时间不含下载、模型加载与后续CPU网格预览。

![结构先验在可见手掌与遮挡位置的不同表现](../outputs/dynamic-human-hand-prior-096/prior-visible-versus-occluded.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；HaMeR、MANO与本地推理 · 左为原图，中为预测关节，右为预测网格；不是高斯训练结果。

全部105张图的关节裁剪扰动中位数再取中位数为1.37像素，第95百分位为3.40像素，但03机位726帧达到17.23像素。该例抓握与遮挡明显，预测不应直接当作精确标签。04／17机位沿用时间模型的曝光差约0.00894帧，7个偶数时刻的关节Sampson残差中位数为3.27至9.61像素，说明同语义关节的多视角一致性仍需要优化；这些并非人工标注或指尖真值，估计曝光差也不是硬件同步证明。

实际查看23组原生手部网格叠加（15机位732帧，加03／04／13／17机位720及726帧）和2组裁剪变化对照。张开手掌可提供连贯的手指候选；抓握位置仍有姿态偏差与不确定性。网格覆盖照片时没有模拟玩偶／身体遮挡，不能从覆盖区域直接推断隐藏手指正确与否。全批先验未作为硬标签，尚未更新高斯参数。[本轮结论与下一步](../outputs/dynamic-human-hand-prior-096/final-summary.json)、[实际看图记录](../outputs/dynamic-human-hand-prior-096/visual-review.json)。

下一步先在固定相机与现有估计曝光时间下，拟合跨机位共享、随时间连续的手部关节姿态；裁剪不稳定或被遮挡的预测只作为不确定的软观测。重新检查原照片投影后，再考虑关节驱动的局部高斯表面与纹理训练。接入可运行、形状完整的手模板，不等于完成清晰人物或大场景联合重建。

来源：[HaMeR论文](https://arxiv.org/pdf/2312.05251.pdf)、[作者项目页](https://geopavlakos.github.io/hamer/)、[官方实现](https://github.com/geopavlakos/hamer)、[MANO](https://mano.is.tue.mpg.de/)。本地CPU预览兼容修改仅补充OSMesa返回RGB时的深度覆盖通道，没有改变网络输出、顶点或相机；详细运行档案留在本地。


### 连续手部模型的跨视角检验

在固定相机内外参与估计曝光时间下，完成两组连续MANO拟合，各1200次更新。主候选使用15个训练机位的105张偶数帧照片；诊断候选完全排除04、09、13机位，只用12机位84张照片初始化和拟合，其余21张仅作检查。00、06、12机位与所有奇数帧均不参与拟合。模型共享10维手形与尺度，使用9个三次B样条控制点表示手腕运动和16组轴角旋转，共470个可训练参数；没有逐相机形变，也没有采用HaMeR单目深度。

目标是裁剪扰动加权的Cauchy重投影损失，带形状、尺度及轨迹平滑正则。主候选对预测关节的残差中位数从4.56降至2.64像素；诊断候选在排除的3机位上从5.63降至3.91像素，但第90百分位从14.93升至16.05像素。**这些是对神经网络关节预测的一致性指标，并非真实指尖精度或高斯画质指标。** 部分大误差仍未解决，不能只引用中位数宣称准确。

![抓握与张手时的共享手部候选](../outputs/dynamic-human-hand-consensus-097/shared-hand-versus-source.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；HaMeR、MANO与本地拟合 · 从左到右为原照片、单视图预测、全部15机位拟合、排除04／09／13机位拟合 · 原生像素，非高斯重建。

实际检查66组原生窗口：15机位的720、726、732帧共45组，04／09／13机位的6个奇数时刻共18组，以及00／06／12机位732帧共3组。抓握位置的共享候选较单视图预测更收拢，但张手时的指尖和轮廓仍存在偏移，04／09机位729帧仍明显偏屈曲。蓝色手模型没有玩偶／身体深度遮挡，亦未优化皮肤外观，不能作为照片一致的手部表面。

两组参数均通过CPU恢复、独立相机投影、旋转矩阵与连续轨迹检查；66张原图窗口和132个共享网格合成图逐像素核对。相同物理时刻的顶点完全相同，未引入按相机变化的手形。原高斯参数没有更新，当前候选不作为硬几何替换。[结果与适用范围](../outputs/dynamic-human-hand-consensus-097/final-summary.json)、[实际看图记录](../outputs/dynamic-human-hand-consensus-097/visual-review.json)。

后续仅尝试有界的软表面／运动约束，先核对手部高斯与模型的对应关系，再与原RGB损失联合优化；手部上下文中还包含前臂和玩偶，不能整块删除或强制绑定。是否采用以留出机位、奇数时刻及原生手部实际渲染为准。


### 手部先验与关节表面高斯的图像检验

首先检验软先验：在6330个可更新高斯中，按参考时刻距离、手腕范围和既有多相机支持筛选1937个，与连续MANO三角面建立对应；其余上下文高斯保留原有光度优化。使用与无先验逐点运动模型相同的105张训练照片、3000次更新及随机背景。增加鲁棒表面／运动损失后，对候选模型的平均加权跟踪距离从0.02694降至0.00837个场景单位，但这不是相对于真实手部几何的误差。15个同时刻手部窗口相对无先验模型仅改善约0.040 dB，6个固定目标窗口约0.027 dB；原生图中张手仍破碎，局部黑斑与粘连未消除。该候选未替换现有模型。[软先验结论](../outputs/dynamic-human-enerf-hand-prior-control-098/final-summary.json)、[近景与连续帧检查](../outputs/dynamic-human-enerf-hand-prior-control-098/visual-review.json)。

随后比较两种关节表面高斯：每组增加1538个三角面高斯，固定原模型的几何、颜色与运动，只允许6330个原上下文高斯调整不透明度。新高斯初始不透明度0.02，颜色在时间和相机之间共享。两个切向标准差初始化为三条边长均方根的0.55倍，法向为0.0825倍，各轴可乘以1/3至3的有界尺度因子；局部偏移每个坐标限制在场景尺度的0.005倍以内。姿态优化组额外学习9个时间控制点的轴角和手腕位置残差，各坐标分别限制在0.25弧度和场景尺度的0.01倍以内；固定姿态组保留零残差。

两组从同一原模型出发，用相同6000次更新、相同照片和背景顺序；与3000次软先验实验的预算不同，只有两组关节表面高斯构成直接同预算对照。00、06、12机位及全部奇数时刻均不参与拟合。总计每组134张原生1920×1080图像，含90张已知机位的留出时刻、18张留出机位的留出时刻、3张留出机位的732帧；其余为训练范围抽查。完整对比保留原图、全身、手部与玩偶上下文，窗口不是精确的手部语义掩码。

![实际手部渲染对照](../outputs/dynamic-human-enerf-mano-surface-099/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位731帧 · 原照片、原模型、固定姿态、图像优化姿态 · 未缩放的原生窗口。

固定姿态与图像优化姿态两组，在90张已知机位留出时刻的人物区域平均PSNR分别为24.6419和24.6422 dB；在18张留出机位的留出时刻分别为22.6957和22.6956 dB。二者差异很小，21个手部窗口与连续帧抽查均未看到清晰手指恢复。4个固定手部情境、两种模型的只读分解显示：新表面参与场景合成后的累计可见权重，仅为单独渲染时累计不透明度的3.7%至11.6%；分母是模型自身的覆盖，不是手部真值区域。去掉新增高斯后图像变化很小，说明新表面的可见贡献有限；单独新表面仍有斑驳颜色，因此不能把遮挡解除等同于重建完成。两组均未采纳为清晰重建结果。

两组冻结参数与6000步优化器状态核对通过，新增高斯与原高斯共同参与实际渲染；附加标记通道记录新增高斯的透射率加权不透明度，三个固定图像检查确认标记通道不改变RGB结果。另检查04、13机位720至732帧的连续图像及相邻帧残差。该区间反复用于开发，结果不代表独立场景验证，也不代表已完成街景背景联合重建。[结果与参数边界](../outputs/dynamic-human-enerf-mano-surface-099/final-summary.json)、[画面检查](../outputs/dynamic-human-enerf-mano-surface-099/visual-review.json)、[可见贡献](../outputs/dynamic-human-enerf-mano-surface-099/contribution/summary.json)。


### 旧表面遮挡与不透明度初始化

对先前姿态优化模型进行只读分组渲染：1937个几何关联高斯、其余4393个手部邻域高斯、246654个冻结高斯，以及1538个新增表面高斯。04机位731帧、13机位731帧、03机位732帧、04机位732帧的分解显示，仅暂时屏蔽1937个关联点，新增表面的可见权重仍只有其单独渲染权重的6.0%至14.7%；屏蔽全部6330个邻域点也只能达到24.9%至30.8%，且画面出现缺口。这些是同一参数的遮挡归因，并非新重建结果。

进一步检查参考时刻三角面上的欧氏最近点是否落在高斯的三倍标准差协方差椭球内，并结合距离和多视角支持筛选，未增加任何新的候选点。这是保守的最近点启发式，不是椭球与整个三角面的精确相交检验。该负结果被保留，没有把更大的邻域冒充为可靠手部标签。随后做两组可恢复的不透明度初始化：新表面均从0.2开始，分别对1937点或全部6330点把原不透明度上限初始化为0.02；之后6330个旧点的不透明度都可重新学习。两组从同一原模型出发，学习同样的手部表面与连续姿态，使用相同6000次更新、105张原照片及背景顺序；唯一差别是额外4393个旧点的初始不透明度。该邻域可能含前臂和玩偶贡献，需通过实际图像检验而非按名称保证保留。

![不透明度初始化后的留出时刻对照](../outputs/dynamic-human-enerf-hand-visibility-100/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位731帧 · 原照片、原模型、1937点初始化、6330点初始化。

1937点初始化与6330点初始化两组，在90张已知机位留出时刻的人物区域平均PSNR分别为24.6425和24.6426 dB；18张留出机位图像分别为22.6959和22.6957 dB。差异很小，21个手部窗口、6张连续序列抽查和3张全身图未显示清晰手指恢复。扩大初始化范围后，额外4393点的平均不透明度从初始不高于0.02重新增长至0.3466；新增高斯的平均不透明度则从0.2下降至0.0713。参数变化不能直接等同于像素贡献：实际合成的新增可见权重在21个手部上下文中仍很弱，这些窗口包含前臂和玩偶，不是语义手部真值。四个固定遮挡情境的模型期望深度差中位数为0.059至0.066场景单位，新表面位于旧表面之后；这里按可见权重平均高斯中心深度，并非真实第一表面深度。两组均不采纳为清晰重建结果。

沿用134组原生1920×1080输入的评测，包含训练抽查、已知机位奇数时刻、00／06／12留出机位奇数时刻及732帧。另检查04、13机位各13张连续图像、手部和玩偶上下文以及全身图。相机、估计曝光时间与非邻域参数保持固定；此区间仍属于反复使用的开发集，尚不构成独立场景验证。[完整结果](../outputs/dynamic-human-enerf-hand-visibility-100/final-summary.json)、[实际画面检查](../outputs/dynamic-human-enerf-hand-visibility-100/visual-review.json)、[不透明度变化](../outputs/dynamic-human-enerf-hand-visibility-100/opacity-effect.json)、[可见贡献](../outputs/dynamic-human-enerf-hand-visibility-100/contribution/summary.json)。


### 人体与手部坐标、逐像素遮挡核对

在相同的105张训练照片及估计拍摄时刻下，重新计算身体和MANO关节。身体前向结果与原参考几何一致，手模型对应浅色上衣人物，没有两个人之间的身份错配；两套模型的手腕距离中位数为0.0209场景单位，投影距离中位数为8.66像素。这是独立拟合模型之间的差异，不能据此判断哪一套是真实关节位置。三个训练机位的720、726、732帧投影检查仍能看到指尖和源照片轮廓的偏差，因此没有把手模型直接平移到身体手腕上。

随后从03、04、13机位732帧分别选取16个模型判定为新增表面被遮挡的像素，按渲染器实际排序逐项重算不透明度合成，与CUDA输出的最大差异小于0.0000004。原6330点手部邻域的累计可见权重约为0.77至0.80，其他冻结高斯约为0.20至0.22；冻结贡献主要关联同一人物的手、腕和前臂。这些关联来自身体蒙皮锚点，不是物体语义标签；48个选定像素也不代表整个数据集。该诊断说明，新增手部表面的图像贡献同时受到旧邻域、冻结表面和自身姿态误差影响，单次调整初始化不能代替共同几何与可见性优化。[诊断摘要](../outputs/dynamic-human-hand-body-alignment-101/diagnostic-summary.json)。


### 轮廓先验与真实手部渲染

手部轮廓先验采用SAM产生并逐图检查的弱掩码：只保留较可靠的内部正样本和邻域负样本，排除边缘及腕部不确定区。22张原生裁图中18张参与轮廓损失，04、13机位的4张仅用于轮廓诊断；但它们的预测关键点仍参与拟合，因此不能称为独立留出相机。控制组与候选组均使用105组HaMeR预测关键点、相同起点及1500次更新，固定体形和尺度。诊断图的已知区域交并比由0.8847提高至0.9422，实际叠加仍有细指间隙不准。该指标衡量弱掩码覆盖，不是三维手指准确率。

随后用两种先验分别初始化1538个表面高斯，固定关节先验，学习颜色、不透明度、局部偏移与尺度。两组使用同一105张训练照片、同一6000步采样顺序，并持续限制6330个旧邻域高斯的不透明度不超过0.02，其余246654个旧高斯的不透明度及全部旧几何、颜色和运动保持逐项一致。两组之间只改变先验；与此前允许旧不透明度重新增长、允许光度姿态修正的模型相比，则同时改变了其他条件，不能称为单一先验消融。

![轮廓先验与关键点先验的留出时刻原生效果](../outputs/dynamic-human-enerf-contour-surface-103/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 原照片、原模型、关键点先验、增加轮廓约束的先验 · 04、13机位731帧。

两组各完成134张1920×1080原生图像评测。已知机位留出时刻的人物区域平均PSNR分别为24.3942和24.3985 dB，留出机位为22.5255和22.5282 dB。21个手部窗口及连续序列中的6个固定时刻均未显示清晰手指恢复。两组的局部上下文也退化：04机位连续留出时刻的该区域PSNR由原模型的21.22降至轮廓组的20.59 dB，13机位由21.58降至20.85 dB。区域含前臂和玩偶，不是精确手部语义掩码。

只读分解另外保留8组画面，分别显示共同合成、旧表面单独渲染和新表面单独渲染；新表面自身仍有斑块与指缝误差。该反事实渲染不重新训练，不能证明遮挡对训练没有影响，也不是可替代完整场景的新模型。两组均未通过清晰度验收，原模型保留。

134组四模型的指标及1072张PNG经过独立复算；另核验26个连续情境、24组相邻变化及两段5帧/秒慢放视频。视频速度用于检查，不代表采集频率。当前区间仍为开发集，独立场景与完整背景联合训练尚待完成。[完整结果](../outputs/dynamic-human-enerf-contour-surface-103/final-summary.json)、[实际看图范围](../outputs/dynamic-human-enerf-contour-surface-103/visual-review.json)、[可见贡献](../outputs/dynamic-human-enerf-contour-surface-103/contribution/summary.json)、[只读分解](../outputs/dynamic-human-enerf-contour-surface-103/visibility/summary.json)、[04机位分解示例](../outputs/dynamic-human-enerf-contour-surface-103/visibility/contour_prior-camera04-frame000731.png)。


### 固定外观后的手部姿态修正

从上一节轮廓先验模型出发，固定全部新旧高斯的颜色、尺度、偏移和不透明度，以及旧人体运动、相机和时间参数；仅允许连续手部关节与手腕作有界修正。每次优化共同使用全部105张训练照片，完成60次L-BFGS外层更新。图像先按1920×1080完整渲染，再取原生160×160手部上下文计算损失，保持训练与评测的栅格化方式一致。局部窗口也包含前臂、衣物或玩偶，不能当作精确的手部语义掩码。

![固定外观前提下的手部姿态优化对照](../outputs/dynamic-human-hand-photo-pose-105/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 04、13机位，留出时刻731帧 · 原照片、原模型、固定手部姿态、优化手部姿态 · 原生像素。

训练目标从0.094990降至0.094817，约下降0.18%。手部上下文的留出时刻PSNR在04机位由20.591变为20.585 dB，在13机位由20.845变为20.859 dB。实际画面仍有斑块、碎片和指间混合，**本候选不采用，原模型保留**。这只说明该起点和修正范围下的姿态单独优化不足，不证明其他联合优化必然无效。

134个完整原分辨率情境、三模型的804张颜色／透明度PNG、26个连续情境和24组相邻帧变化均已复核；实际查看21个手部上下文、6个连续抽查画面与3组全身图。两段视频完成编码与解码验证，视频本身未作逐帧视觉验收。此区间经过反复开发，不能作为独立场景测试。新演员序列的预处理另存本地，尚未产生训练结果。[完整结果](../outputs/dynamic-human-hand-photo-pose-105/final-summary.json)、[实际看图范围](../outputs/dynamic-human-hand-photo-pose-105/visual-review.json)。


### 独立人物序列的静态与动态对照

ENeRF-Outdoor 的 actor2_3 提供另一组人物，但仍是相同采集场地。15个机位在600至624帧中的13个偶数帧用于训练，共195张1920×1080照片；00、06、12机位在该区间只用于外观评测。静态高斯、同编号动态高斯、带时间偏移的动态高斯，各从相同初始化训练15000次。动态模型使用局部形变图，本次三组对照没有使用MANO或SMPL人体模板，效果不能归因于模板下载。

初始化、照片采样和背景采样相同；三组最终高斯数量不同，因此这里比较相同更新预算，未声称相同参数量或严格相同计算量。其他机位的时间偏移只在582至594帧的独立区间估计，未把校准图像用于高斯初始化或外观训练；这些机位属于外观留出，不属于完全未见过的传感器。偏移单位是图像索引，不是经过标定的硬件时间戳。

已知机位的180张留出时刻中，静态、动态、时间校正动态三组的平均前景PSNR分别为 **15.42、24.77、26.84 dB**。

其他机位的36张奇数帧中，静态、动态、时间校正动态三组的平均前景PSNR分别为 **15.27、22.47、23.86 dB**。

其他机位的39张偶数帧中，静态、动态、时间校正动态三组的平均前景PSNR分别为 **15.06、22.46、23.82 dB**。

时间校正相对同编号动态模型，在180张已知机位留出图像中有170张改善、10张下降；最大下降发生在09机位601帧，约0.92 dB。其他机位的75张验证图像均提高，但平均分提升仍不能替代局部细节验收。

![早期抬脚动作的完整人物对照，保留局部失败](../outputs/dynamic-human-actor23-independent-106/comparison/camera00-frame000601-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 外观留出机位00、帧601 · 原照片、静态、动态、时间校正动态。

动态模型减少了身体和下蹲姿态的重影，但抬起的脚仍有半透明拖影。下图手部上下文中，手指形状仍与原照片不完全一致。**本轮取得主体外观改善，尚未通过完整人物清晰度验收。**

![手部与物体上下文中的真实细节](../outputs/dynamic-human-actor23-independent-106/comparison/camera13-frame000613-crouching_hands_object_context.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位13、留出帧613 · 原生像素。

完整原分辨率评测包含264个情境，其中255张为正式验证照片、9张为明确标注的训练检查；72个细节窗口在查看模型输出前固定。PSNR使用原始前景掩码，也包含玩偶及掩码误差；矩形窗口是上下文，不是精确的手指或腿部语义标注。底部窗口可能截掉抬起的脚，故该失败必须结合完整人物图判断。

[机位00连续回放](../outputs/dynamic-human-actor23-independent-106/continuity/camera00-native-continuity.mp4)与[机位13连续回放](../outputs/dynamic-human-actor23-independent-106/continuity/camera13-native-continuity.mp4)各含25个连续帧，以5帧/秒慢放；这不是采集帧率。前者全部为外观留出，后者包含13张训练帧和12张留出帧。视频是压缩的观看副本，指标来自未增强的原生图像。

[完整结果](../outputs/dynamic-human-actor23-independent-106/final-summary.json)、[实际看图范围](../outputs/dynamic-human-actor23-independent-106/visual-review.json)保留校验与结论边界。该局部前景模型尚未与大场景背景共同优化，也未完成OmniRe接入。

独立复算检查了1584张评测PNG与300张连续序列PNG；实际查看15幅单帧对照和6幅连续帧抽查图。视频完成编码、帧数和解码检查，未作全部帧的视觉验收。

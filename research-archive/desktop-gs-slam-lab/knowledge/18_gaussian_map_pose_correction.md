---
id: "gslab-k18"
slug: "trajectory-correction-gaussian-map-coordinate-consistency"
title: "轨迹更正与高斯地图坐标一致性"
order: 18
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
---

# 轨迹更正与高斯地图坐标一致性

轨迹更正会改变相机在世界坐标中的位置与方向。若高斯地图仍留在原坐标，同一相机看到的高斯中心、深度和椭圆通常也会变化。对明确归属于一个锚点的局部高斯，同时更正相机和高斯可以保持该锚点的相机坐标观测；这是一条刚体坐标恒等式，不能代替跨锚点地图优化或证明回环已经成立。

先从坐标恒等式理解地图更正，再区分统一 gauge 与不同锚点的局部修正。保持某个锚点的相机观测，不代表跨锚点表面自动一致，也不代表已经完成回环优化。

## 相机到世界的位姿更正在世界坐标左侧作用

使用 OpenCV 列向量约定，锚点相机到世界的变换（camera-to-world，C2W）记为 $C$。相机坐标轴为 $x$ 向右、$y$ 向下、$z$ 向前；$R$ 将相机向量旋转到世界，$t$ 是相机原点的世界位置，长度单位为米。令 $\widetilde p=(p^T,1)^T$ 表示齐次坐标，则

$$
C=\begin{bmatrix}
R&t\\
0^T&1
\end{bmatrix},\qquad
\widetilde p_w=C\widetilde p_c.
$$

$C_{\mathrm{old}}$ 和 $C_{\mathrm{new}}$ 必须表示同一个传感器锚点，长度单位均为米，并使用兼容的世界坐标约定。把该锚点的旧世界表达送到更正后的世界表达的变换为

$$
\Delta=C_{\mathrm{new}}C_{\mathrm{old}}^{-1},\qquad
C_{\mathrm{new}}=\Delta C_{\mathrm{old}}.
$$

$\Delta$ 从左侧作用于世界坐标。$C_{\mathrm{old}}^{-1}C_{\mathrm{new}}$ 是另一种局部坐标表达，直接把它乘到原世界高斯上会混用参考系。在固定非交换构造中，这种误用使更正后的世界中心相差 **1.05967 m**。

恒等变换、共轴旋转或重复保存的同一位姿可能掩盖方向错误，因此不能把它们作为有检错力的负例。这里的旋转组合交换差为 0.33210，无量纲 Frobenius 范数；平移组合交换差为 0.58314 m。两种单位分别记录，不把整个 4×4 矩阵的范数解释成长度误差。

更正量必须属于特殊欧氏群 $\operatorname{SE}(3)$：$R^TR=I_3$、$\det R=+1$，齐次矩阵底行为 $(0,0,0,1)$。它不包含尺度更正。单目尺度漂移或相似变换群 $\operatorname{Sim}(3)$ 中的尺度更正需要另行处理尺度和协方差，不能沿用“标准差尺度不变”的结论。

## 均值和协方差必须一起变换

设世界高斯中心为 $\mu$，世界协方差为 $\Sigma$，刚体更正 $\Delta$ 的旋转和平移分别为 $R_\Delta$ 与 $t_\Delta$。同一锚点局部高斯更正后的参数是

$$
\mu_{\mathrm{new}}=R_\Delta\mu_{\mathrm{old}}+t_\Delta,\qquad
\Sigma_{\mathrm{new}}=R_\Delta\Sigma_{\mathrm{old}}R_\Delta^T.
$$

由于 $C_{\mathrm{new}}^{-1}\Delta=C_{\mathrm{old}}^{-1}$，对应锚点的相机空间均值保持。协方差也满足

$$
R_{\mathrm{new}}^T\Sigma_{\mathrm{new}}R_{\mathrm{new}}
=R_{\mathrm{old}}^T\Sigma_{\mathrm{old}}R_{\mathrm{old}}.
$$

中心的单位是米，协方差的单位是平方米，二者的误差应分别记录。只看中心投影会漏掉协方差方向错误：均值更正正确、协方差留在原世界方向时，投影中心可以保持，椭圆却发生变化。

这个恒等式保持的是锚点下固定的局部高斯。它不是在规定真实物体跟着相机运动，也不证明某个估计更正量符合真实场景。实际轨迹更正怎样获得，以及它怎样约束其他锚点，仍是独立的估计问题。

## 标准差保持不变与四元数左乘

若高斯使用

$$
\Sigma=R(q)
\begin{bmatrix}
s_x^2&0&0\\
0&s_y^2&0\\
0&0&s_z^2
\end{bmatrix}
R(q)^T.
$$

其中 $s_x,s_y,s_z>0$ 为局部主轴标准差，单位米；$R(q)$ 将局部主轴转到世界，$q=(w,x,y,z)$ 使用 Hamilton 四元数乘法和主动旋转约定。刚体旋转保留三个标准差及协方差特征值，四元数应更新为

$$
q_{\mathrm{new}}=
\frac{q_\Delta\otimes q_{\mathrm{old}}}
{\|q_\Delta\otimes q_{\mathrm{old}}\|_2}.
$$

这里 $\otimes$ 为 Hamilton 乘法，$q_\Delta$ 表示 $R_\Delta$，更正从左侧复合。交换四元数顺序通常产生不同旋转；本构造中右乘产生的世界协方差 Frobenius 差为 **0.00421256 m²**。$q$ 与 $-q$ 则表示同一旋转，实际检查的旋转矩阵差为零，不应把符号变化当作错误。

写入具体系统前，要核对四元数顺序、主动或被动旋转、局部轴到世界轴的方向，以及矩阵乘法含义。不同库的变量名相同，约定仍可能不同。可分别从旋转矩阵和 Hamilton 四元数重建变换，检查两条参数路线是否一致。

## 相机空间保持后 投影也保持

相机空间均值与协方差不变时，固定内参下的针孔中心投影保持。中心处的一阶投影协方差为

$$
u=f_x\frac{x}{z}+c_x,\qquad v=f_y\frac{y}{z}+c_y,
\qquad z>0,
\qquad
\Sigma_{\mathrm{pixel}}\approx J_\pi\Sigma_cJ_\pi^T,
\qquad
J_\pi=\begin{bmatrix}
\frac{f_x}{z}&0&-\frac{f_xx}{z^2}\\
0&\frac{f_y}{z}&-\frac{f_yy}{z^2}
\end{bmatrix}.
$$

其中 $(x,y,z)^T$ 是相机空间中心，$f_x,f_y$ 为像素焦距，$(c_x,c_y)$ 是主点；$\Sigma_c$ 为相机空间协方差。$J_\pi$ 的单位为 px/m，输入协方差为 m²，输出为 px²。这是局部线性化诊断，没有验证 gsplat 的光栅化、椭圆加权平均滤波（Elliptical Weighted Average，EWA）、裁剪、遮挡或 alpha 合成。构造未指定画幅，也不检查投影是否落在图像内；正深度的针孔坐标恒等式与实际可见性不同。

固定高斯在锚点 A 相机下的均值为 $(0.28,-0.11,2.6)^T$ m，三个尺度为 $(0.08,0.035,0.012)^T$ m。锚点 A、B 使用不同非共轴旋转和平移；每项门限在运行前已经写入 [固定输入](../outputs/gaussian-map-pose-correction-source-preparation-001/fixtures.json)。

| 正例检查量 | 实测误差 | 预定门限 | 单位 |
|---|---:|---:|---|
| 两锚点相机均值 最大分量绝对误差 | 8.88e−16 | 1e−12 | m |
| 两锚点相机协方差 最大元素绝对误差 | 8.67e−19 | 1e−14 | m² |
| 两锚点针孔中心 最大分量绝对误差 | 4.26e−14 | 1e−9 | pixel |
| 两锚点一阶投影协方差 最大元素绝对误差 | 4.26e−14 | 1e−9 | pixel² |
| 左乘四元数与矩阵旋转 最大元素绝对误差 | 1.25e−16 | 1e−12 | 无量纲 |
| 更正前后协方差特征值 最大绝对误差 | 1.73e−18 | 1e−14 | m² |

这些误差只描述小型 float64 合成矩阵的计算结果。它们不是实际地图精度，也不构成 float32 长序列或接近零深度时的误差保证。

## 只改相机与遗漏协方差是不同错误

固定各向异性高斯使用三个不等尺度。只改相机而保持世界高斯不变，会改变相机空间中心和投影；更正均值却不旋转协方差，会保持中心而改变椭圆。

| 固定反例 | 实测差异 | 度量与单位 |
|---|---:|---|
| 只改相机 相机空间中心改变 | 1.54373 | L2，m |
| 只改相机 针孔中心改变 | 204.28988 | L2，pixel |
| 正确更正均值但遗漏协方差旋转 | 0.00377548 | 相机协方差 Frobenius，m² |
| 同一遗漏对应的一阶投影协方差变化 | 42.60872 | Frobenius，pixel² |

遗漏协方差旋转时，中心投影误差仍只有 4.26e−14 pixel。这个例子说明“点投影没变”不足以验证高斯坐标更正。

球形协方差是必要的边界例：旋转不会改变它本身。本控制对球形协方差省略旋转后，仅出现 4.34e−19 m² 的元素误差。因此，球形高斯不能检出该遗漏；各向异性与非交换构造使负例具有实际检错力。

## 不同锚点保持局部观测 仍可能产生接缝

第三种构造把两份原本重合的世界高斯分别归到两个锚点，给它们不同的更正 $\Delta_a$、$\Delta_b$。每份高斯在自己的更正锚点下仍保持相机空间观测，但两份世界中心分离了 **1.28557 m**；用更正后的锚点 A 同时观察两份中心，针孔投影相差 **160.76065 pixel**。

这是潜在接缝的几何反例，没有实际渲染，也没有根据这个距离声称看见了表面质量损害。它说明自身锚点的坐标恒等式并不约束跨锚点重叠关系。

若两份高斯接受同一个世界刚体变换，重合中心差为零。所有相机和所有高斯统一变换属于共同坐标规范自由度（gauge freedom）的改变；逐锚点更正不同则需要额外的重叠约束、融合、优化或变形模型。这些机制没有由本控制实现。

## 来源帧不等于唯一更正归属

[关键帧与历史窗口](../knowledge/14_keyframes_map_window_history.md)解释了冻结教学 mapper 的限制：历史相机在当前反向传播中被视为常量并固定，窗口优化会更新全部高斯参数；来源帧标识记录初始化来源，并不禁止其他视角把点移动或改变形状。当前程序没有历史相机联合束调整（Bundle Adjustment，BA）或回环更正。

因此，不能简单地按照初始化来源帧给最终地图每个高斯套用该帧的 $\Delta$，再把结果称为地图修复。经过跨帧优化或融合的高斯可能同时解释多个锚点，没有唯一更正归属；重复变换、局部变换不一致或忽略重叠约束都会引入新的误差。控制只说明：当归属与刚体假设已经成立时，怎样保持对应锚点的局部坐标一致。

## 常量颜色与球谐表示的视角相关颜色

[冻结 V4 教学程序](../scripts/train_tum_online_gs_v4.py)传入的是常量 RGB，颜色原始参数经逻辑函数映射后交给渲染器，没有球谐函数（spherical harmonics，SH）系数。刚体坐标变换可保留这种常量颜色和不透明度参数。

视角相关颜色另有参考系：若球谐颜色 $c(v)$ 以世界方向 $v$ 表示，世界旋转改变了查询方向；通常需要旋转球谐系数，或满足 $c_{\mathrm{new}}(v^{\prime})=c_{\mathrm{old}}(R_\Delta^Tv^{\prime})$，即在评估外观时把新方向转回原方向，并核对实际基函数约定。本控制不实现或验证 SH 输运，不能从均值、协方差恒等式推出视角相关颜色已经保持。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

本页第一组实测来自一次确定性 float64 CPU 矩阵控制：58 项检查全部通过，未调用渲染器、优化器、真实地图、传感器数据或 GT。正例检查恒等式的数值误差，负例检查错误路径产生的差异；“负例通过”表示检出了预设错误，不表示错误路径可接受。第二组实测使用已有完整地图做两次 GPU 渲染，结果和适用范围单独列在后文。

### 实测：完整地图统一 gauge 的两次渲染

矩阵恒等式之外，还要检查实际光栅化路径。这里使用已完成的 desk V4 prior0 地图，保留全部 **23,838 个高斯**，固定保存的第 548 帧估计 w2c 与 160×120 内参。地图 checkpoint 的 SHA256 为 `64cda836…44d82`；原相机直接使用保存的 float32 字节，没有重新求逆或修正它的轻微非正交误差。

对全部高斯施加同一个世界变换：旋转 `Rz(23°) Ry(−11°) Rx(17°)`，平移 `[0.23,−0.17,0.09] m`。均值与四元数按上文规则更正，相机使用 `W_new = W_old inverse(Delta)`。log scales、颜色 logits、opacity logits 保持原字节；两次实际 GPU 渲染共享同一组激活后的尺度、常量 RGB 与 opacity 缓冲区。

唯一实际运行调用 gsplat 1.5.3 的 `RGB+ED` 两次，`packed=false`、near/far 为 0.1/6 m、黑色背景。没有优化步骤、GT 读取、原始传感器读取、新训练或失败重试。两次输出为：

![固定原地图与相机的模型预测](../outputs/gaussian-global-gauge-renderer-controls-001/original.png)

![共同世界变换之后的模型预测](../outputs/gaussian-global-gauge-renderer-controls-001/global_gauge.png)

这些都是模型预测图，不是传感器 RGB。展示 PNG 经过裁剪和 8 bit 量化；下表由未裁剪 float32 原始数组转 float64 后相减得到，不能靠“图片看起来一样”代替检查。

| 检查量 | 最大绝对差 | RMSE | 比较分母 |
|---|---:|---:|---|
| RGB，无量纲 | 1.06096e−5 | 8.63874e−7 | 57,600 个标量 |
| alpha，无量纲 | 7.62939e−6 | 5.39803e−7 | 19,200 个像素 |
| 共同正深度支持下的 ED，m | 1.29938e−5 | 5.17025e−7 | 19,200 个像素 |
| 共同覆盖下的 ED，m | 1.29938e−5 | 5.17277e−7 | 19,172 个像素 |

支持掩码要求 ED 有限且为正、`alpha>1e−6`；覆盖掩码要求 ED 有限且为正、`alpha>0.5`。有限 ED、正深度支持、覆盖三种掩码的 XOR 都是 **0 像素**；两种共同掩码均非空。深度最大差约为 **0.013 mm**，它是两次预测的差，不能解释为对真实场景的测深精度。

运行前固定的 11 项 GPU 门限全部通过：[原判据与配置](../outputs/gaussian-global-gauge-renderer-cpu-preparation-002/protocol.json)、[完整数值比较](../outputs/gaussian-global-gauge-renderer-controls-001/comparison.json)。RGB/alpha 的最大差门限为 1e−3、RMSE 门限为 1e−5；有限 ED XOR 必须为零，支持/覆盖 XOR 上限各 4；支持 ED 最大差/RMSE 上限为 0.01/1e−4 m，覆盖为 0.005/5e−5 m。没有根据结果放宽门限。

[控制源码](../scripts/check_gaussian_global_gauge_renderer.py)及[实际终态](../outputs/gaussian-global-gauge-renderer-controls-001/metrics.json)记录两次调用、原参数和相机的前后哈希与 CPU 几何检查；[completion](../outputs/gaussian-global-gauge-renderer-controls-001/completion.json)绑定八份归档文件。原始两份 NPZ 留在桌面实验目录，没有复制到网站。

[独立 SOURCE/CPU 审阅](../outputs/gaussian-global-gauge-renderer-independent-source-review-002/metrics.json)使用另一条 SciPy、Hamilton 乘法与通用逆矩阵路线；[其源码](../scripts/review_gaussian_global_gauge_renderer.py)没有执行作者控制。实际渲染后，[独立数组检查源码](../scripts/audit_gaussian_global_gauge_raw.py)读取六个原始数组，以 float64 相减与 `math.fsum` 重算 RMSE；[独立结果](../outputs/gaussian-global-gauge-renderer-independent-raw-audit-001/metrics.json)核对形状、字节哈希、所有分母、掩码和 11 个判据，重算记录与原记录差为 0。[主代理终态复核](../outputs/gaussian-global-gauge-renderer-controls-001/root_source_and_result_review.json)另行记录文件与两图核对。

这支持的是**同一个固定地图、相机、共同 SE3 变换下的 float32 渲染一致性**。只验证一个视角，没有证明逐锚点变形后的地图重叠一致、世界位置准确、SH 输运、回环、联合 BA 或完整 SLAM。全局 gauge 保持观测与修复不同轨迹段的漂移是两个问题。

### CPU 矩阵控制的源码和证据范围

[控制源码](../scripts/check_gaussian_map_pose_correction.py)的 SHA256 为 `67e41dc8…f3b4`，[实际结果](../outputs/gaussian-map-pose-correction-cpu-controls-001/metrics.json)完整保存 58 项数值、门限、单位、状态及实际矩阵，[completion](../outputs/gaussian-map-pose-correction-cpu-controls-001/completion.json)锁定源码、输入和结果字节。源码与输入在运行前后均保持，实际调用计数为 renderer 0、optimizer 0、真实数据读取 0、GPU 0。

[独立 SciPy 审阅源码](../scripts/review_gaussian_pose_transport.py)没有 import 或运行作者函数，另用固定构造与作者输入重建相机、高斯和投影；[审阅结果](../outputs/gaussian-pose-transport-independent-source-review-001/metrics.json)分别保存两种构造的误差。它支持这里的矩阵方向与合成数学范围，没有审计真实地图或渲染性能。

[终态读回记录](../outputs/gaussian-map-pose-correction-source-preparation-001/actual_completion_readback.json)逐条复核保存的 58 个比较判据及 completion 文件哈希。这是保存结果的完整性检查，不能另称第二份独立数学推导。

这些控制不检测回环、不估计更正量，也不验证多锚点融合后的地图准确性。轨迹怎样被更正与地图怎样重新一致，是系统必须分别回答的问题。

[主代理源码与终态核对](../outputs/gaussian-map-pose-correction-cpu-controls-001/root_source_and_result_review.json)记录完整 SOURCE 阅读与实际 58 项判据的复核范围。

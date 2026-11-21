---
id: "gslab-k29"
slug: "3dgs-to-splatam-progression"
title: "从传统 3DGS 到 SplaTAM：哪些假设变了"
order: 29
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://arxiv.org/abs/2308.04079"
  - "https://arxiv.org/abs/2312.02126"
  - "https://github.com/graphdeco-inria/gaussian-splatting/tree/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
---

# 从传统 3DGS 到 SplaTAM：哪些假设变了

[渲染章](27_3dgs_rendering_foundations.md)回答怎样由高斯生成图像，[训练章](28_3dgs_training_density_control.md)回答怎样由照片学习高斯。接下来保留这套渲染数学，把相机位姿也变成未知量，并让地图随观测逐步增长。三维高斯表示由离线场景重建进入同时定位与建图（Simultaneous Localization and Mapping，SLAM）。

## 1. 已知相机参数与未知相机位姿

![SplaTAM原论文的RGB-D输入、跟踪、地图增密与建图流程](../outputs/slam-system-paper-course-001/figures/splatam-fig-02.png)
> 图源：[Keetha 等（2024），Fig. 2](https://arxiv.org/abs/2312.02126) · 原图裁剪 · CC BY-SA 4.0。

传统三维高斯泼溅（3D Gaussian Splatting，3DGS）接收多视图照片以及相机内外参，主要估计地图。记第 $k$ 帧真实彩色图像为 $\mathbf I_k$，地图参数集合为 $\boldsymbol\Theta$，世界到相机的刚体变换为 $\mathbf T_k$：

$$
\mathbf T_k=
\begin{bmatrix}
\mathbf R_{cw,k}&\mathbf t_{cw,k}\\
\mathbf 0^\mathsf T&1
\end{bmatrix}\in SE(3),
\qquad
\mathbf x_{c,k}=\mathbf R_{cw,k}\mathbf x_w+\mathbf t_{cw,k}.
$$

$$
\boldsymbol\Theta^*
=\underset{\boldsymbol\Theta}{\operatorname{argmin}}
\sum_{k\in\mathcal K_{\mathrm{train}}}
\mathcal L_k\bigl(\mathcal R(\boldsymbol\Theta,\mathbf T_k),\mathbf I_k\bigr).
$$

$\mathbf R_{cw,k}\in SO(3)$ 是无量纲旋转矩阵，$\mathbf t_{cw,k}\in\mathbb R^3$ 为平移，$\mathbf T_k\in\mathbb R^{4\times4}$ 是齐次变换矩阵；$SE(3)$ 表示三维刚体变换集合。$\mathcal K_{\mathrm{train}}$ 为训练视图索引集合，$\mathcal R$ 是上一章明确定义的投影与颜色合成关系的简记。$\mathbf T_k$ 是给定条件，$\boldsymbol\Theta$ 包含各高斯的位置、形状、不透明度与外观。多个视角约束同一地图，但相机误差仍可能被地图吸收。

SLAM 接收按时间到来的观测，要估计当前位姿 $\mathbf T_k$ 与地图 $\boldsymbol\Theta_k$。相机位姿有三个平移与三个旋转自由度；深度可提供米制尺度，但世界原点与朝向仍需约定。以首相机建立世界坐标系时，规定：

$$
\mathbf R_{cw,1}=
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1
\end{bmatrix},
\qquad
\mathbf t_{cw,1}=
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}.
$$

这只是选择坐标参考，不是对后续运动准确性的约束。

## 2. 高斯形状、外观与深度合成

![传统3DGS原论文中各向异性与各向同性表示的真实图像对照](../outputs/3dgs-paper-course-001/figures/paper-fig-10.png)
> 图源：[Kerbl 等（2023），Fig. 10](https://arxiv.org/abs/2308.04079) · 原图裁剪 · CC BY 4.0。

该图是传统离线重建的形状消融，说明各向异性可以表达更复杂的局部结构。SplaTAM 改用球形高斯是在线优化中的简化选择，不能把离线消融成绩直接当作其 SLAM 精度结论。

传统 3DGS 用各向异性高斯表达局部结构：

$$
\boldsymbol\Sigma_i=\mathbf R_i
\begin{bmatrix}
s_{i1}^2&0&0\\
0&s_{i2}^2&0\\
0&0&s_{i3}^2
\end{bmatrix}
\mathbf R_i^\mathsf T.
$$

三个 $s_{ij}>0$ 是轴向标准差，$\mathbf R_i\in SO(3)$ 决定主轴方向；$\boldsymbol\Sigma_i\in\mathbb R^{3\times3}$ 为协方差，单位为长度平方。SplaTAM 的默认球形表示使用：

$$
\boldsymbol\Sigma_i=
\begin{bmatrix}
s_i^2&0&0\\
0&s_i^2&0\\
0&0&s_i^2
\end{bmatrix},
\qquad s_i>0.
$$

球形只需要一个形状尺度，自身旋转不改变形状。相机旋转仍然改变中心投影，因此相机姿态依然需要估计。外观则由传统球谐函数（Spherical Harmonics，SH）表达的视角相关颜色，简化为每个高斯的红、绿、蓝三通道颜色（RGB）向量 $\mathbf c_i\in\mathbb R^3$。

中心投影、二维投影分布和前后遮挡合成仍然继承渲染章。高斯按当前相机中的深度由近到远排列，像素 $p\in\mathbb R^2$ 处的有效不透明度、透射率和合成权重为：

$$
\alpha_i(p)=o_i\exp\!\left[
-\frac12(p-\mathbf m_i)^\mathsf T
\boldsymbol\Sigma_{2D,i}^{-1}(p-\mathbf m_i)
\right],
\qquad
T_i^{\mathrm{vis}}(p)=\prod_{j<i}(1-\alpha_j(p)),
\qquad
w_i(p)=T_i^{\mathrm{vis}}(p)\alpha_i(p).
$$

$$
\widehat{\mathbf C}(p)=\sum_{i=1}^{N}w_i(p)\mathbf c_i,
\qquad
S(p)=\sum_{i=1}^{N}w_i(p),
\qquad
\widehat D(p)=\sum_{i=1}^{N}w_i(p)z_i.
$$

$\mathbf m_i\in\mathbb R^2$ 为投影中心，$\boldsymbol\Sigma_{2D,i}\in\mathbb R^{2\times2}$ 为以像素平方计的投影协方差；$o_i$ 为高斯不透明度。$T_i^{\mathrm{vis}}$ 表示透射率，避免与相机变换 $\mathbf T_k$ 混淆；$S$ 为累计不透明度，无量纲。上式颜色使用零背景，深度与覆盖的背景贡献也为零。中心的相机轴向深度明确为：

$$
z_i=
\begin{bmatrix}0&0&1\end{bmatrix}
(\mathbf R_{cw,k}\boldsymbol\mu_i+\mathbf t_{cw,k}).
$$

$z_i$ 是相机坐标系的第三个坐标，单位米，不是相机到高斯中心的欧氏距离。$\widehat D$ 是累计深度；若定义归一化深度：

$$
D_{\mathrm{norm}}(p)=
\frac{\widehat D(p)}{S(p)},
\qquad S(p)>0,
$$

它是不同的量，只有非零覆盖时才有定义。覆盖不足时，累计深度和归一化深度并不相等，不能在损失中直接互换。

## 3. 彩色与深度残差的交替优化

![SplaTAM原论文中的位姿跟踪：用已建地图解释当前RGB-D观测](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-tracking.png)
> 图源：[Keetha 等（2024），Fig. 2 右上分面](https://arxiv.org/abs/2312.02126) · 原图裁剪 · CC BY-SA 4.0。

彩色与深度图像（RGB-D）同时提供外观和几何约束。记颜色残差为 $\mathbf r_{C,k}\in\mathbb R^3$，深度残差为 $r_{D,k}\in\mathbb R$：

$$
\mathbf r_{C,k}(p;\mathbf T,\boldsymbol\Theta)=
\widehat{\mathbf C}(p;\mathbf T,\boldsymbol\Theta)-\mathbf I_k(p),
\qquad
r_{D,k}(p;\mathbf T,\boldsymbol\Theta)=
\widehat D(p;\mathbf T,\boldsymbol\Theta)-D_k(p).
$$

$$
\mathcal L_k(\mathbf T,\boldsymbol\Theta)=
\sum_{p\in\mathcal V_k}
\left[
\lambda_C\sum_{c=1}^{3}|r_{C,k,c}(p;\mathbf T,\boldsymbol\Theta)|
+\lambda_D|r_{D,k}(p;\mathbf T,\boldsymbol\Theta)|
\right].
$$

$D_k$ 是观测轴向深度，单位米；$\mathcal V_k$ 为有效支持像素集合。上式是绝对误差损失（$L_1$ 损失）的统一示意目标，用于说明约束与未知量的关系。$\lambda_C,\lambda_D$ 控制颜色与深度的相对影响；颜色按 $[0,1]$ 表示且总损失无量纲时，$\lambda_C$ 无量纲，$\lambda_D$ 以米的倒数计。支持集合需要考虑传感器有效性与地图覆盖，也可以依赖 $\mathbf T,\boldsymbol\Theta$；跟踪和建图的支持集合、求和或平均方式可以不同。像素数量之外，还需判断纹理与几何是否提供充分运动约束。

跟踪（Tracking）固定已有地图，更新当前相机：

$$
\mathbf T_k^*=
\underset{\mathbf T\in SE(3)}{\operatorname{argmin}}\,
\mathcal L_k(\mathbf T,\boldsymbol\Theta_{k-1}).
$$

建图（Mapping）固定当前与历史窗口的相机，更新高斯：

$$
\boldsymbol\Theta_k^*=
\underset{\boldsymbol\Theta}{\operatorname{argmin}}\,
\sum_{j\in\mathcal W_k}\mathcal L_j(\mathbf T_j,\boldsymbol\Theta).
$$

![SplaTAM原论文中的地图优化：在相机位姿固定后更新高斯地图](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-mapping.png)
> 图源：[Keetha 等（2024），Fig. 2 左下分面](https://arxiv.org/abs/2312.02126) · 原图裁剪 · CC BY-SA 4.0。

$\mathcal W_k$ 是选入建图的观测窗口。Fig. 2 的 Tracking 分面对应位姿更新，Mapping 分面对应地图更新；二者使用相似残差，但改变的是不同未知量。窗口提供历史约束；若需要一起更正历史相机，则还要扩展优化变量。

## 4. 深度反投影与地图点初始化

![SplaTAM原论文依据当前观测向已有地图新增高斯](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-densification.png)
> 图源：[Keetha 等（2024），Fig. 2 右下分面](https://arxiv.org/abs/2312.02126) · 原图裁剪 · CC BY-SA 4.0。

给定无畸变针孔相机的内参矩阵 $\mathbf K$、像素 $(u,v)$ 和轴向深度 $z>0$，相机坐标中的点为：

$$
\mathbf K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix},
\qquad
\mathbf p_c=z\mathbf K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
\begin{bmatrix}
z(u-c_x)/f_x\\
z(v-c_y)/f_y\\
z
\end{bmatrix}.
$$

$$
\mathbf R_{wc}=\mathbf R_{cw}^\mathsf T,
\qquad
\mathbf c_w=-\mathbf R_{cw}^\mathsf T\mathbf t_{cw},
\qquad
\boldsymbol\mu_w=\mathbf R_{wc}\mathbf p_c+\mathbf c_w.
$$

$\mathbf c_w$ 为相机在世界中的中心，$\mathbf R_{wc}$ 把相机向量旋转到世界。深度和位置以米计，焦距 $f_x,f_y$ 与主点坐标 $c_x,c_y$ 以像素计。这里 $z$ 不是相机到点的欧氏距离。与单目运动恢复结构（Structure from Motion，SfM）的任意尺度不同，经过标定的米制深度直接确定反投影点的米制尺度。

低覆盖区域意味着已有地图尚不足以解释观测，有效深度可以提供新高斯中心。用 $S(p)<\tau_S$ 表示一种覆盖不足判据，$0<\tau_S<1$ 为无量纲阈值；实际还需有效深度和其他插点条件。传统高斯复制（clone）与分裂（split）主要重新分配已有表示容量，深度插点则引入当前观测到的新几何位置。

像素间距 $\Delta u$ 对应的相机横向长度间隔为：

$$
\Delta X=\frac{z\,\Delta u}{f_x}.
$$

因此初始化尺度可依据像素对应的空间间距确定；$\Delta u,f_x$ 都以像素计，$\Delta X$ 以米计。该关系不把像素间距视为物体真实厚度。

Fig. 2 右下分面中的新增高斯来自当前观测；位姿决定它们进入世界地图的位置。由反投影公式可推知，位姿偏差会传入新点，新点随后又参与定位。这个反馈关系是公式分析，而不是该图中的受控误差实验。深度同时参与初始化、插点、有效支持与残差；仅减小深度残差权重，不会自动移除其他深度信息。

## 5. 图像质量与轨迹误差的独立评价

![SplaTAM原论文补充材料中的三维地图、估计轨迹与参考轨迹](../outputs/slam-system-paper-course-001/figures/splatam-fig-s01.png)
> 图源：[Keetha 等（2024），Fig. S.1](https://arxiv.org/abs/2312.02126) · 原图裁剪 · CC BY-SA 4.0。

传统重建重点检查独立新视角的图像质量。SLAM 还必须比较运动。设相机中心估计为 $\widehat{\mathbf c}_k$，独立参考为 $\mathbf c_k$，尺度固定的刚体对齐由 $\mathbf R_A\in SO(3)$ 与 $\mathbf t_A\in\mathbb R^3$ 给出，绝对轨迹误差（Absolute Trajectory Error，ATE）的位移均方根形式为：

$$
\mathbf e_k=\mathbf R_A\widehat{\mathbf c}_k+\mathbf t_A-\mathbf c_k,
\qquad
\operatorname{ATE}_{\mathrm{RMSE}}=
\sqrt{\frac1n\sum_{k=1}^n\|\mathbf e_k\|_2^2}.
$$

$n$ 为有效匹配相机中心的数量，RMSE 表示均方根误差（Root Mean Square Error）；该位置误差以米计。旋转 $\mathbf R_A$ 无量纲，平移 $\mathbf t_A$ 以米计，对齐中不包含尺度因子。时间配对、物理参考点和失败覆盖也影响结论，不能只报告一个均值。

若参考和估计都提供完整位姿，以相机到世界变换 $\mathbf U_k=\mathbf T_k^{-1}$ 与 $\widehat{\mathbf U}_k=\widehat{\mathbf T}_k^{-1}$ 定义间隔 $\Delta$ 的相对位姿误差（Relative Pose Error，RPE）：

$$
\mathbf E_{k,\Delta}=
\left(\mathbf U_k^{-1}\mathbf U_{k+\Delta}\right)^{-1}
\left(\widehat{\mathbf U}_k^{-1}\widehat{\mathbf U}_{k+\Delta}\right)
=
\begin{bmatrix}
\mathbf R_{E,k}&\mathbf t_{E,k}\\
\mathbf0^\mathsf T&1
\end{bmatrix},
$$

$$
e_{t,k}=\|\mathbf t_{E,k}\|_2,
\qquad
e_{R,k}=
\arccos\!\left(\frac{\operatorname{tr}(\mathbf R_{E,k})-1}{2}\right).
$$

$e_{t,k}$ 以米计，$e_{R,k}$ 以弧度计；$k$ 与 $k+\Delta$ 必须按声明的帧间隔或时间间隔配对。RPE 用于观察局部相对运动误差，ATE 检查整体轨迹位置偏差。仅有位置参考时，不能把参考中不存在的旋转当作真实旋转计算旋转 RPE。

地图和错误位姿可能共同解释观测，因此图像相似度不能替代轨迹准确度。沿[每帧跟踪与建图](24_splatam_official_rgbd_system.md) → [有效像素与跟踪损失](25_splatam_tracking_loss_support.md) → [关键帧与历史约束](14_keyframes_map_window_history.md)继续阅读，随后进入[轨迹评价](08_trajectory_evaluation.md)、[参考与恢复](17_rejection_reference_recovery.md)、[地图坐标更正](18_gaussian_map_pose_correction.md)。

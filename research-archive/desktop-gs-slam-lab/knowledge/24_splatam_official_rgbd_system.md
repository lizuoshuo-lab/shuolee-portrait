---
id: "gslab-k24"
slug: "splatam-official-rgbd-system"
title: "SplaTAM：从 RGB-D 高斯建图到跟踪、漂移与一致性"
order: 24
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-13"
sources:
  - "https://arxiv.org/abs/2312.02126"
  - "https://spla-tam.github.io/"
  - "https://arxiv.org/abs/2408.10154"
  - "https://github.com/spla-tam/SplaTAM/tree/da6bbcd24c248dc884ac7f49d62e91b841b26ccc"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/datasets/gradslam_datasets/tum.py"
  - "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/tree/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
---

# SplaTAM：从 RGB-D 高斯建图到跟踪、漂移与一致性

传统 3DGS 的两章已经说明：给定相机位姿，高斯怎样生成图像，以及怎样用多视图照片优化地图。SplaTAM 把这套可微渲染用于同步定位与建图（Simultaneous Localization and Mapping，SLAM）：相机位姿不再预先给定，地图也不再等全部照片采完后才建立。系统每收到一帧彩色与深度图像（RGB-D），就估计相机运动、补充新区域、更新已有地图。

本章把 Keetha 等的 SplaTAM 论文方法与公开实验改写为连续讲义。先解释原方法，再用实验理解其设计取舍，最后补充漂移、独立轨迹评测和地图坐标更正。后面的恢复与一致性推导是扩展知识，不是 SplaTAM 原论文已经实现的回环或重定位功能。

## 1. 从离线重建到在线 RGB-D SLAM

![SplaTAM 原论文展示的真实三维高斯地图、相机视锥，以及输入视角和未参与建图的新视角渲染](../outputs/slam-system-paper-course-001/figures/splatam-fig-01.png)
> 图源：Keetha 等（2024），Fig. 1 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图左的三维场景是地图，排列在地图中的视锥表示相机的位置、朝向和视野；图右是从指定相机生成的颜色、深度图。两类输出对应两类未知量：地图参数与相机位姿。只把图像渲染得相似，不能判断相机运动是否正确。

记第 $k$ 帧观测为 $\mathbf I_k,D_k$，分别表示 RGB 图像和轴向深度图；高斯地图为 $\boldsymbol\Theta$。全章采用列向量：世界坐标点 $\mathbf x_w$ 变到相机坐标 $\mathbf x_{c,k}$ 的关系是

$$
\mathbf x_{c,k}=\mathbf R_{cw,k}\mathbf x_w+\mathbf t_{cw,k},
\qquad
\mathbf T_{cw,k}=
\begin{bmatrix}
\mathbf R_{cw,k}&\mathbf t_{cw,k}\\
\mathbf 0^\mathsf T&1
\end{bmatrix}.
$$

$\mathbf R_{cw,k}\in SO(3)$ 把世界向量旋转到相机，$SO(3)$ 是三维旋转矩阵集合；$\mathbf t_{cw,k}$ 以米计。齐次矩阵 $\mathbf T_{cw,k}\in SE(3)$ 表示三维刚体变换，$SE(3)$ 包含旋转和平移，不包含尺度。相机坐标约定为 $X$ 向右、$Y$ 向下、$Z$ 向前。相机到世界变换及相机中心为

$$
\mathbf C_k=\mathbf T_{cw,k}^{-1}
=\begin{bmatrix}
\mathbf R_{cw,k}^\mathsf T&\mathbf c_{w,k}\\
\mathbf 0^\mathsf T&1
\end{bmatrix},
\qquad
\mathbf c_{w,k}=-\mathbf R_{cw,k}^\mathsf T\mathbf t_{cw,k}.
$$

因此，世界到相机矩阵的平移列不是相机在世界中的位置。

设 $\mathcal R$ 表示上一章的可微高斯渲染，传统离线重建主要解

$$
\boldsymbol\Theta^*
=\underset{\boldsymbol\Theta}{\operatorname{argmin}}
\sum_{k\in\mathcal K}
\mathcal L_k\bigl(\mathcal R(\boldsymbol\Theta,\mathbf T_{cw,k}),\mathbf I_k\bigr),
\qquad \mathbf T_{cw,k}\ \text{给定}.
$$

$\mathcal K$ 是训练视图索引集合。SplaTAM 则按时间递推：固定旧地图估计当前位姿，用这个位姿添加新高斯，再固定所选相机更新地图。第一相机规定 $\mathbf C_1=\mathbf I_4$，其中 $\mathbf I_4$ 是四阶单位矩阵。经过标定的深度提供米制尺度；首帧为单位位姿只是在选择世界原点与朝向，不会保证后续没有漂移。

论文 Fig. 1 标出的高速率指图像渲染。完整 SLAM 还要多次反向传播、扩展地图和选择关键帧，其每帧计算时间在第 7 节单独讨论。

## 2. 米制初始化与各向同性高斯

![SplaTAM 流程图的输入区域：当前 RGB-D 观测与已有三维高斯地图](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-input.png)
> 图源：Keetha 等（2024），Fig. 2 输入区域 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图中的 RGB-D 包含两种不同信息。颜色给出外观，深度把像素射线上的位置确定为米制三维点。这里的深度是相机坐标中的 $Z$ 分量，不是从相机中心到物体的欧氏距离。相机光轴是经过投影中心、垂直于像平面的直线，在本章坐标约定下与相机 $Z$ 轴重合。

对已去畸变的针孔图像，内参矩阵与反投影为

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
=\begin{bmatrix}
z(u-c_x)/f_x\\
z(v-c_y)/f_y\\
z
\end{bmatrix}.
$$

$f_x,f_y$ 是以像素计的焦距；$(c_x,c_y)$ 是主点，即光轴与像平面交点的像素坐标；$z>0$ 和 $\mathbf p_c$ 以米计。只有有效深度才能生成新三维点。世界中心由当前相机到世界位姿确定：

$$
\boldsymbol\mu_w=\mathbf R_{cw,k}^\mathsf T\mathbf p_c+\mathbf c_{w,k}.
$$

传统 3DGS 通常用运动恢复结构（Structure from Motion，SfM）得到的稀疏点初始化，并学习各向异性形状及视角相关颜色。SplaTAM 原方法选用各向同性高斯与视角无关 RGB。每个高斯只有八个参数：中心三个分量、RGB 三个分量、尺度 $r_i$ 和不透明度 $o_i$。

$$
\boldsymbol\theta_i=
(\boldsymbol\mu_i,\mathbf c_i,r_i,o_i),
\qquad
\boldsymbol\Sigma_i=
\begin{bmatrix}
r_i^2&0&0\\
0&r_i^2&0\\
0&0&r_i^2
\end{bmatrix},
\qquad
g_i(\mathbf x)=o_i\exp\!\left(
-\frac{\|\mathbf x-\boldsymbol\mu_i\|_2^2}{2r_i^2}
\right).
$$

$\boldsymbol\mu_i,r_i$ 以米计，$\boldsymbol\Sigma_i$ 以平方米计；RGB 与 $o_i\in[0,1]$ 无量纲。论文称 $r_i$ 为半径，它在高斯指数中承担标准差尺度的作用，不表示一个有硬边界的实心球。各向同性协方差经过任意旋转都不改变，因此不需要用高斯自身的旋转描述形状；相机旋转仍然改变投影，必须估计。

首帧跳过跟踪。论文用像素 RGB 初始化颜色、不透明度取 $0.5$，并选取投影约为一像素尺度的初始高斯：

$$
\mathbf c_i^{(0)}=\mathbf I_1(u,v),
\qquad o_i^{(0)}=0.5,
\qquad r_i^{(0)}=\frac{D_1(u,v)}{f}.
$$

$f$ 是论文简化投影使用的像素焦距，$D_1$ 是以米计的输入深度；一像素作为像素坐标的数值单位隐含在上式中。更远的表面需要更大的米制尺度，才能覆盖相近的像素范围。这个初始化控制采样密度，不是在估计表面真实厚度。论文把输入深度记作 $D_{\mathrm{GT}}$；此处它指可用的深度观测，与评测相机轨迹的真值是不同信息。

真实数据还须核对深度单位、内参与 RGB-D 关联。若图像缩放而焦距、主点未相应变换，反投影射线就改变。RGB 与深度时间差 $\Delta t$ 在近似恒定运动下可产生约 $\|\mathbf v\|\,|\Delta t|$ 的平移错配和 $\|\boldsymbol\omega\|\,|\Delta t|$ 的旋转错配；$\mathbf v$ 以米每秒计，$\boldsymbol\omega$ 以弧度每秒计。软件最近邻配对不能证明物理同步。

## 3. 颜色、深度与覆盖的统一渲染

![SplaTAM 在 ScanNet++ S1 新视角的 RGB 与深度配对，比较 Point-SLAM、SplaTAM 与参考图像](../outputs/slam-system-paper-course-001/figures/splatam-fig-03-depth.png)
> 图源：Keetha 等（2024），Fig. 3，S1 新视角面板 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图中左列是 RGB，右列是彩色编码的深度；中间的 Ours 一行由 SplaTAM 同一高斯地图渲染。颜色、深度与覆盖不需要三张独立地图：它们共享高斯的投影和前后遮挡权重，只更换要累积的特征。这是 SplaTAM 把高斯图像拟合用于定位的核心。

设第 $i$ 个高斯在当前相机中的中心为 $(X_i,Y_i,z_i)^\mathsf T$，$z_i>0$。论文的各向同性投影写为

$$
\mathbf m_i=
\begin{bmatrix}
f_xX_i/z_i+c_x\\
f_yY_i/z_i+c_y
\end{bmatrix},
\qquad
\rho_i=\frac{f r_i}{z_i},
\qquad
\alpha_i(\mathbf p)=o_i\exp\!\left(
-\frac{\|\mathbf p-\mathbf m_i\|_2^2}{2\rho_i^2}
\right).
$$

$\mathbf p=(u,v)^\mathsf T$、$\mathbf m_i$ 和投影尺度 $\rho_i$ 以像素计。这里保留论文 Eq. (3) 的球形高斯简化投影；它不等同于上一章对一般协方差使用投影雅可比的全部形式。实际光栅化的滤波与裁剪也不由这条简式完全描述。

沿当前视角由近到远排序。定义像素处的透射率与权重：

$$
\mathcal T_i(\mathbf p)=\prod_{j<i}\bigl(1-\alpha_j(\mathbf p)\bigr),
\qquad
w_i(\mathbf p)=\mathcal T_i(\mathbf p)\alpha_i(\mathbf p).
$$

于是论文 Eq. (2)、(4)、(5) 分别为

$$
\widehat{\mathbf C}(\mathbf p)=\sum_iw_i(\mathbf p)\mathbf c_i,
\qquad
\widehat D(\mathbf p)=\sum_iw_i(\mathbf p)z_i,
\qquad
S(\mathbf p)=\sum_iw_i(\mathbf p).
$$

本式采用颜色、深度和覆盖的零背景。$\widehat{\mathbf C}$ 是渲染 RGB，$\widehat D$ 是累计轴向深度，单位米；$S$ 是累计不透明度（silhouette），无量纲。silhouette 在这里不是二值物体轮廓，也不是表面深度，更不是位姿正确的概率。对有限贡献，覆盖与最终透射率满足

$$
S(\mathbf p)=1-\prod_i\bigl(1-\alpha_i(\mathbf p)\bigr).
$$

因此，$S$ 大表示该像素的射线上已经有充分的高斯不透明度；它无法单独区分地图几何正确与错误高斯叠加。

尤其要区分累计深度与归一化深度：

$$
D_{\mathrm{norm}}(\mathbf p)
=\frac{\widehat D(\mathbf p)}{S(\mathbf p)},
\qquad S(\mathbf p)>0.
$$

当唯一贡献位于 $2\,\mathrm m$、权重为 $0.5$ 时，论文累计深度为 $1\,\mathrm m$，归一化深度则为 $2\,\mathrm m$。把两者互换会同时改变数值与位姿梯度；论文渲染深度的公式没有除以 $S$。这种差别也说明：覆盖不足时，不宜直接把累计深度看作可靠表面测量。

## 4. 位姿跟踪：初值、支持与梯度

![SplaTAM 相机跟踪面板，用高覆盖掩码限制观测与渲染图的比较区域，并优化相机位姿](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-tracking.png)
> 图源：Keetha 等（2024），Fig. 2 相机跟踪区域 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

新帧往往包含之前没见过的区域。图中只把覆盖充分的部分交给位姿优化：相机应该通过已经建立的地图定位，尚未建好的区域不能提供同等可信的图像解释。

论文先用恒速度外推初始化相机中心与四元数，再固定地图优化当前位姿。为了展开其参数空间示意，令 $\mathbf c_{w,k}$ 为相机中心，$\mathbf q_{wc,k}=(w,x,y,z)^\mathsf T$ 为相机到世界主动旋转的单位 Hamilton 四元数。一种与论文描述对应的外推写法为

$$
\mathbf c_{w,k}^{(0)}=2\mathbf c_{w,k-1}-\mathbf c_{w,k-2},
\qquad
\mathbf q_{wc,k}^{(0)}=
\frac{2\mathbf q_{wc,k-1}-\mathbf q_{wc,k-2}}
{\|2\mathbf q_{wc,k-1}-\mathbf q_{wc,k-2}\|_2}.
$$

四元数相邻符号应先保持一致，因为 $\mathbf q$ 与 $-\mathbf q$ 表示同一旋转。这个表达是对论文 Eq. (7) 参数空间外推的展开，不是把两个刚体矩阵逐元素相加后仍当作合法位姿，也不是惯性测量单元（IMU）预积分。不同参数化的外推结果未必相同；急转或前帧误差会削弱初值。

令 $\Omega$ 为图像像素域。排除无效输入后，以论文门限构造跟踪支持集合

$$
\mathcal M_k(\mathbf T)=
\{\mathbf p\in\Omega\mid D_k(\mathbf p)>0,\ S(\mathbf p;\mathbf T)>0.99\}.
$$

对有限有效渲染，论文 Eq. (8) 的绝对误差结构可展开为

$$
\mathcal L_{\mathrm{track}}(\mathbf T)=
\sum_{\mathbf p\in\mathcal M_k(\mathbf T)}
\left[
\lambda_D|\widehat D(\mathbf p;\mathbf T)-D_k(\mathbf p)|
+\lambda_C\sum_{c=1}^{3}
|\widehat C_c(\mathbf p;\mathbf T)-I_{k,c}(\mathbf p)|
\right],
\qquad
\mathbf T_k^*=\underset{\mathbf T\in SE(3)}{\operatorname{argmin}}
\mathcal L_{\mathrm{track}}(\mathbf T).
$$

论文按以米表示的深度数值取深度项权重 $1$、颜色项权重 $0.5$。若把总目标规范为无量纲，可写 $\lambda_D=1\,\mathrm m^{-1}$、$\lambda_C=0.5$；RGB 按 $[0,1]$ 表示。权重调配两种残差的数值尺度，不表示已测得的传感器逆方差。上式颜色项明确采用三通道绝对误差之和；改成通道平均、改成像素平均都要重新说明权重与分母，不能直接沿用原目标数值。

### 为什么梯度能更新相机

固定地图时，位姿改变了高斯中心的相机坐标、像素投影、遮挡和深度；渲染残差随之改变。对投影中心的局部导数为

$$
\mathbf J_\pi=
\begin{bmatrix}
f_x/z&0&-f_xX/z^2\\
0&f_y/z&-f_yY/z^2
\end{bmatrix}.
$$

它把米制的三维位置变化转成像素变化，单位为像素每米。图像纹理越有区别，投影错位越容易产生有方向的颜色梯度；几何和深度变化提供另一类约束。纹理单一、重复结构、局部平面或遮挡可使部分运动方向难以区分。支持像素多，只说明参与残差的数量多，不保证六个运动自由度都充分可观测。

### 总损失下降为什么不等于定位更准

掩码依赖位姿，每次重新渲染后，支持集合可能改变。记 $n=|\mathcal M_k|>0$，每个有效像素的代价为 $\ell_{\mathbf p}$，则

$$
\mathcal L_{\mathrm{sum}}=\sum_{\mathbf p\in\mathcal M_k}\ell_{\mathbf p}
=n\overline\ell,
\qquad
\overline\ell=\frac1n\sum_{\mathbf p\in\mathcal M_k}\ell_{\mathbf p}.
$$

少算困难区域也能让总和下降。比较跟踪质量时，要同时看支持数量、覆盖分布、同一口径的平均残差和独立轨迹误差；空支持不能视作零误差的成功定位。论文的 $0.99$ 是地图覆盖门限，不能解释成“位姿有 99% 置信度”。

要排查支持集合变化的影响，可把初始位姿的支持集合固定为 $\mathcal M_k^0=\mathcal M_k(\mathbf T^{(0)})$，在同一地图上额外计算

$$
\mathcal L_{\mathrm{fixed}}(\mathbf T)
=\sum_{\mathbf p\in\mathcal M_k^0}
\left[\lambda_D\left|\widehat D_k(\mathbf p;\mathbf T)-D_k(\mathbf p)\right|
+\lambda_C\left\|\widehat{\mathbf C}_k(\mathbf p;\mathbf T)-\mathbf I_k(\mathbf p)\right\|_1\right].
$$

这是诊断用的比较，不改变原跟踪目标。比较位姿必须在这些固定像素上都产生有效数值；不能一面剔除无效预测，一面声称支持完全不变。若固定支持下的目标也下降，就不能仅用“少计算了一些像素”解释该下降。仍应拆开颜色和深度项：颜色拟合改善可能抵消深度拟合恶化，而两者都只描述与当前地图的匹配，并不直接约束独立参考中的相机位置。

### 候选位姿必须与其损失对应

跟踪图中的“优化位姿”同时涉及两件事：沿梯度产生新候选，以及从已经评估的候选中选择输出。设第 $j$ 次迭代的世界到相机变换为 $\mathbf T^{(j)}$，固定地图为 $\mathcal G$，则候选与评分应组成同一对量：

$$
\ell_j=\mathcal L_{\mathrm{track}}(\mathbf T^{(j)};\mathcal G),
\qquad
j^*=\underset{0\le j<K}{\operatorname{argmin}}\;\ell_j,
\qquad
\widehat{\mathbf T}=\mathbf T^{(j^*)}.
$$

$K$ 为评估次数，无量纲；损失沿用前文颜色与米制深度的加权定义。若按照 $\ell_j$ 判断优劣，却保存下一次更新后的 $\mathbf T^{(j+1)}$，该评分便不能描述实际输出。有限步长、遮挡切换和支持集合变化，都可能使一次更新后的目标值高于更新前；因此选择输出后应在同一地图上重新计算其目标值。这里是优化过程的一致性条件，不是另一个 SLAM 观测模型。

这一条件仍不保证输出更接近真实相机。设固定对齐变换的旋转和平移为 $\mathbf R_A,\mathbf t_A$，估计相机中心与独立参考为 $\widehat{\mathbf c}_j,\mathbf c_{\mathrm{ref}}$，位置误差为

$$
e_j=\|\mathbf R_A\widehat{\mathbf c}_j+\mathbf t_A-\mathbf c_{\mathrm{ref}}\|_2.
$$

$e_j$ 单位为米。$\ell_a<\ell_b$ 并不推出 $e_a<e_b$：前者衡量当前地图对观测的解释，后者衡量相机位置；地图偏差和观测不足可使二者不同。独立参考仅用于优化完成后的检查，不能进入候选筛选。把这两条证据分开，才能判断修正改善的是目标选择，还是实际定位。

## 5. 地图扩展：用新深度生成高斯

![SplaTAM 地图扩展面板，依据覆盖不足和前方新增几何生成高斯](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-densification.png)
> 图源：Keetha 等（2024），Fig. 2 高斯增密区域 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图中彩色新增区域对应地图尚不能解释的观测。跟踪完成后，系统已有当前位姿；有效深度告诉它新几何应该沿像素射线放在哪里。SplaTAM 的地图扩展不只依赖当前图像误差，也利用显式地图的覆盖边界。

令深度绝对误差及其中位数为

$$
e_D(\mathbf p)=|\widehat D(\mathbf p)-D_k(\mathbf p)|,
\qquad
m_D=\operatorname{median}_{\mathbf p\in\mathcal V_k}e_D(\mathbf p),
\qquad
\mathcal V_k=\{\mathbf p\in\Omega\mid D_k(\mathbf p)>0\}.
$$

$e_D,m_D$ 以米计。这里为说明有效深度条件，在 $\mathcal V_k$ 上定义中位数；论文将中位深度误差记为 MDE，未在紧凑公式中展开其统计集合，具体统计口径需要另行核对。把论文 Eq. (9) 的条件写成集合的并集，可以清楚区分两类插入条件：

$$
\mathcal A_k=\mathcal V_k\cap
\left(
\{\mathbf p\mid S(\mathbf p)<0.5\}
\cup
\{\mathbf p\mid D_k(\mathbf p)<\widehat D(\mathbf p),\ e_D(\mathbf p)>50m_D\}
\right).
$$

第一类是已有地图覆盖不足；第二类是传感器看到的表面比已有渲染更近，而且深度差明显超过典型误差。第二类只关注需要在已有几何前方补充的情况，不能把任意大残差都解释成新表面。$50$ 是论文经验阈值，无量纲；中位数为零或深度异常时，还需要清楚规定实际处理边界。

对 $\mathbf p\in\mathcal A_k$，按第 2 节反投影生成新高斯中心，初始颜色、尺度和不透明度也沿用首帧规则：

$$
\boldsymbol\mu_{i,k}^{(0)}
=\mathbf R_{cw,k}^{\mathsf T}
\left(D_k(\mathbf p)\mathbf K^{-1}
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}\right)
+\mathbf c_{w,k}.
$$

这与传统 3DGS 的复制和分裂承担不同职责：复制、分裂重新分配已有表示容量；深度驱动插点把当前新观测到的三维位置加入地图。高斯数量增加，不能据此判定是哪种机制。

插点的可靠性还依赖位姿。输入深度即使准确，错误相机变换仍会把整片新高斯放到错误世界位置。这些点随后参与下一帧定位，所以增加地图容量不会自动消除相机误差。第 8 节将沿这条反馈解释漂移。

## 6. 地图优化：关键帧与历史约束

![SplaTAM 地图更新面板，在固定已估计相机下优化高斯，使地图解释当前和历史观测](../outputs/slam-system-paper-course-001/figures/splatam-fig-02-mapping.png)
> 图源：Keetha 等（2024），Fig. 2 地图更新区域 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图中相机被视作给定，地图参数沿渲染残差优化。这一步与传统 3DGS 的已知位姿重建有共同结构，但初值是上一时刻已经建立的地图，观测是有限的当前和历史窗口。

关键帧是可再次用于优化的观测及其保存位姿。它需要保持图像、深度、内参、时间和帧身份的对应关系：

$$
\mathcal K_j=(j,\mathbf I_j,D_j,\mathbf K_j,t_j^I,t_j^D,\mathbf T_{cw,j}).
$$

保存位姿是估计值，不是真值。论文每隔若干帧保存关键帧；建图窗口包含当前帧、最近关键帧，以及与当前观测有较高重叠的历史关键帧。对当前深度生成的世界点集合 $\mathcal P_k$，其投到历史视锥的比例可写为

$$
o_{kj}=
\frac{
|\{\mathbf x\in\mathcal P_k\mid z_j(\mathbf x)>0,
\ \pi_j(\mathbf R_{cw,j}\mathbf x+\mathbf t_{cw,j})\in\Omega_j\}|
}{|\mathcal P_k|},
\qquad |\mathcal P_k|>0.
$$

$o_{kj}\in[0,1]$ 无量纲，$\pi_j$ 是历史针孔投影，$\Omega_j$ 是历史图像范围。这个视锥重叠判据表示可能有关联的观测，不等于已完成遮挡一致性或逐点对应检验。位姿偏差也会影响选窗。

### 建图目标怎样利用历史

论文规定建图不沿用跟踪的 silhouette 掩码，并在颜色绝对误差上增加结构相似性（Structural Similarity，SSIM）项，同时去除近零不透明度或过大的高斯。为区分求和与平均，下面给出本章用于解释的规范化目标，而非声称论文逐项给出了这个分母和权重：

$$
\mathcal L_{C,j}=\frac1{3|\Omega_j|}
\sum_{\mathbf p\in\Omega_j}\sum_{c=1}^{3}
|\widehat C_{j,c}(\mathbf p)-I_{j,c}(\mathbf p)|,
\qquad
\mathcal L_{D,j}=\frac1{|\mathcal V_j|}
\sum_{\mathbf p\in\mathcal V_j}
|\widehat D_j(\mathbf p)-D_j(\mathbf p)|.
$$

$$
\boldsymbol\Theta_k^*=
\underset{\boldsymbol\Theta}{\operatorname{argmin}}
\sum_{j\in\mathcal W_k}
\left[
\lambda_D\mathcal L_{D,j}
+\lambda_C\mathcal L_{C,j}
+\lambda_S\bigl(1-\operatorname{SSIM}(\widehat{\mathbf C}_j,\mathbf I_j)\bigr)
\right],
\qquad \{\mathbf T_{cw,j}\}_{j\in\mathcal W_k}\ \text{固定}.
$$

$\mathcal W_k$ 是建图窗口，$\mathcal V_j$ 是有效深度集合，$\lambda_D,\lambda_C,\lambda_S$ 调配几何、颜色与结构误差。颜色和 SSIM 无量纲，平均深度误差以米计；若总目标无量纲，$\lambda_D$ 以米的倒数计。不同归一化会改变图像分辨率和深度缺失对目标尺度的影响，因此跟踪求和损失与建图平均损失不能直接比较。

图像参与多视图地图优化，不意味着它的历史位姿也被更正。联合光束法平差（Bundle Adjustment，BA）需要明确把相机与地图共同列为未知量，并固定共同坐标自由度；SplaTAM 原方法的这一建图步骤固定相机。关键帧窗口本身也不等于回环检测。

退出窗口的观测不再直接约束当前更新。它的影响可以留在地图参数中，但不能因此视为永久监督。随着新视角继续修改颜色、位置和尺度，必须分别检查当前拟合、旧视角重建和整段轨迹一致性。来源帧只记录高斯从哪里初始化，不会使这个高斯永久冻结或只受该帧影响。

## 7. 原论文实验：精度、质量与计算量

![SplaTAM 原论文的 ScanNet++ 输入视角与新视角渲染对照，同时展示 RGB 和深度](../outputs/slam-system-paper-course-001/figures/splatam-fig-03.png)
> 图源：Keetha 等（2024），Fig. 3 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图中的训练视角曾作为 SLAM 输入，新视角来自未参与建图的另一段采集。训练图像好看可以来自地图对已见视图的拟合；新视角更能检查表示是否在空间上具有一致的解释。图中 Point-SLAM 的深度渲染使用参考深度辅助采样，因而比较时还要核对不同方法渲染阶段的输入条件。

### 四种数据条件回答不同问题

原论文用 Replica、TUM-RGBD、原始 ScanNet 和 ScanNet++。Replica 是合成场景，深度完整而准确、相邻运动较小；TUM 与原始 ScanNet 的真实深度缺失、噪声和运动模糊更明显。ScanNet++ 选用两个 DSLR 场景，颜色与深度质量较高，但相邻相机间隔较大，并有独立采集用于新视角评价。

论文对公开基线的结果采用三个随机种子 $0,1,2$ 的平均，消融使用种子 $0$；Replica 输入视角渲染每五帧评一次。Table 1 的数据集平均绝对轨迹误差均方根为：

| 数据条件 | SplaTAM 的 ATE RMSE | 读数应解释什么 |
|---|---:|---|
| Replica，八个合成序列 | $0.36\,\mathrm{cm}$ | 干净深度与较小运动下的定位表现 |
| ScanNet++，两个 DSLR 场景 | $1.2\,\mathrm{cm}$ | 高质量观测与较大相机位移的组合 |
| TUM-RGBD，五个真实序列 | $5.48\,\mathrm{cm}$ | 深度缺失、噪声与模糊仍有明显影响 |
| 原始 ScanNet，六个真实序列 | $11.88\,\mathrm{cm}$ | 不能把合成场景的厘米以下精度推广到所有输入 |

这些是论文公开结果，条件不能合并成一个“通用精度”。TUM 上 ORB-SLAM2 的表中平均误差仍低于 SplaTAM；原始 ScanNet 上不同方法互有优势。论文结论支持特定数据条件下的能力，不能改写成所有场景都领先。

ScanNet++ 的 Table 3 同时给出新视角与输入视角：新视角平均峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）为 $24.41\,\mathrm{dB}$，深度平均绝对误差为 $2.07\,\mathrm{cm}$；输入视角分别为 $27.98\,\mathrm{dB}$ 和 $1.28\,\mathrm{cm}$。新视角使用参考相机位姿对到首帧地图坐标，用于评价渲染；这不同于用参考位姿替代在线相机跟踪。

### 消融怎样检验设计机制

![SplaTAM 在 Replica Room 0 上的颜色、深度、运动外推与覆盖门限消融](../outputs/slam-system-paper-course-001/figures/splatam-table-04-05.png)
> 图源：Keetha 等（2024），Tables 4–5 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

Table 4 在同一 Replica Room 0 上改变跟踪与建图的直接监督：完整目标 ATE 为 $0.27\,\mathrm{cm}$，只用颜色残差为 $1.38\,\mathrm{cm}$，只用深度残差为 $86.03\,\mathrm{cm}$。这与两类约束互补的机制一致：该场景只靠深度不能充分区分图像平面中的运动；只用颜色仍能跟踪，但几何误差增大。这里“只用颜色损失”仍保留深度初始化与插点，不能称为单目系统，也不能称为完全删除深度模块。

Table 5 检查初值与支持选择：去掉运动外推的 ATE 为 $2.95\,\mathrm{cm}$；去掉 silhouette 掩码为 $115.80\,\mathrm{cm}$；把跟踪覆盖门限从 $0.99$ 降到 $0.5$ 为 $1.30\,\mathrm{cm}$。它们说明这些设计在该单场景、单种子条件下影响明显；阈值更严格也会减少可用支持，因此不能据此推出所有观测条件下都应不断增大阈值。

补充材料 Table S.1 比较 ScanNet++ S1 的各向异性与各向同性表示，均不使用视角相关球谐颜色。ATE 分别为 $0.55$ 与 $0.57\,\mathrm{cm}$，新视角 PSNR 分别为 $23.98$ 与 $23.99\,\mathrm{dB}$；各向同性时间为前者的 $83.3\%$，内存为 $57.5\%$。这是简化表示的效率证据，适用范围是这个场景与配置，不否定传统 3DGS 在其他任务中使用各向异性表示的价值。

### 渲染速度为什么不能代表 SLAM 速度

Table 6 在 Replica Room 0、RTX 3080 Ti 上比较完整计算时间。论文对 SplaTAM 描述的每轮渲染分辨率为 $1200\times980$；每帧跟踪 $40$ 轮、建图 $60$ 轮，表中分别约 $1.00\,\mathrm s$ 和 $1.44\,\mathrm s$。较少迭代的 SplaTAM-S 使用 $10$ 与 $15$ 轮，并降低增密分辨率；对应约 $0.19$ 与 $0.33\,\mathrm s$，ATE 从 $0.27$ 变为 $0.39\,\mathrm{cm}$。

渲染一张图、一次优化迭代和完成一帧 SLAM 是三个不同计算单位。高斯光栅化让密集像素监督成为可能，但原论文完整配置的逐帧优化仍有显著开销；这些时间属于论文硬件条件，不能直接预测本机耗时。

## 8. 漂移与独立轨迹评测

![SplaTAM 补充材料的三维重建与参考、估计相机轨迹，展示不同颜色视锥在地图中的空间关系](../outputs/slam-system-paper-course-001/figures/splatam-fig-s01.png)
> 图源：Keetha 等（2024），Fig. S.1 · [原论文](https://arxiv.org/abs/2312.02126) · 裁图 · CC BY-SA 4.0。

图中绿色视锥与红色轨迹表示 SplaTAM 估计，蓝色视锥和轨迹表示参考。这个三维叠加直观展示相机与地图的关系；定量评价仍需要明确坐标、时间关联和误差定义，不能用“看起来重合”代替数值。

### 米制尺度为什么不能防止漂移

深度确定局部几何尺度，不会使每次位姿更新自动正确。论文明确指出运动模糊、较大深度噪声、剧烈旋转会影响方法。低纹理、重复结构和初始化偏差也可能削弱当前残差的辨识力。小位姿误差进入反投影后，产生有偏的新高斯；下一帧再把这些地图点作为定位参考，误差就可能递推。

对新点的世界位置 $\boldsymbol\mu_w=\mathbf R_{wc}\mathbf p_c+\mathbf c_w$，小旋转误差 $\delta\boldsymbol\phi$ 与中心误差 $\delta\mathbf c$ 的一阶传播为

$$
\delta\boldsymbol\mu_w
\approx\delta\mathbf c
-\mathbf R_{wc}[\mathbf p_c]_\times\delta\boldsymbol\phi,
\qquad
[\mathbf p_c]_\times=
\begin{bmatrix}
0&-Z&Y\\
Z&0&-X\\
-Y&X&0
\end{bmatrix}.
$$

$\delta\mathbf c,\delta\boldsymbol\mu_w$ 以米计，$\delta\boldsymbol\phi$ 以弧度计；这里采用旋转右侧的小扰动，是本章补充的一阶推导。它说明同样的角度偏差，对更远的点可能产生更大的位置偏差。后续地图可以调整参数继续拟合图像，所以低渲染损失或高 PSNR 不能代替正确轨迹。

### ATE 检查整体位置偏差

设估计与独立参考的相机中心为 $\widehat{\mathbf c}_k,\mathbf c_k$。先完成时间配对，再用整段统一的无尺度刚体对齐求 $\mathbf R_A,\mathbf t_A$：

$$
(\mathbf R_A,\mathbf t_A)=
\underset{\mathbf R\in SO(3),\,\mathbf t\in\mathbb R^3}{\operatorname{argmin}}
\sum_{k=1}^{n}\|\mathbf R\widehat{\mathbf c}_k+\mathbf t-\mathbf c_k\|_2^2,
\qquad
\operatorname{ATE}_{\mathrm{RMSE}}=
\sqrt{\frac1n\sum_{k=1}^{n}
\|\mathbf R_A\widehat{\mathbf c}_k+\mathbf t_A-\mathbf c_k\|_2^2}.
$$

ATE 即绝对轨迹误差（Absolute Trajectory Error），这里报告相机中心位移的均方根，单位米。它消去共同坐标原点与朝向的差别，不允许逐段任意对齐。对已有米制深度的 RGB-D，加入尺度拟合可能隐藏尺度错误，需要另行声明，不能和无尺度对齐混用。

### RPE 检查相对运动偏差

以相机到世界变换 $\mathbf C_k,\widehat{\mathbf C}_k$ 表示参考与估计，间隔 $\Delta$ 的相对位姿误差（Relative Pose Error，RPE）为

$$
\mathbf E_{k,\Delta}
=\left(\mathbf C_k^{-1}\mathbf C_{k+\Delta}\right)^{-1}
\left(\widehat{\mathbf C}_k^{-1}\widehat{\mathbf C}_{k+\Delta}\right)
=\begin{bmatrix}
\mathbf R_{E,k}&\mathbf t_{E,k}\\
\mathbf0^\mathsf T&1
\end{bmatrix}.
$$

$$
e_{t,k}=\|\mathbf t_{E,k}\|_2,
\qquad
e_{R,k}=\arccos\!\left(
\frac{\operatorname{tr}(\mathbf R_{E,k})-1}{2}
\right),
\qquad
\phi_{R,k}=\frac{180}{\pi}e_{R,k}.
$$

平移误差以米计，$e_{R,k}$ 以弧度计，$\phi_{R,k}$ 以度计；$\operatorname{tr}$ 表示矩阵的迹。反余弦形式要求合法旋转矩阵，数值计算须控制舍入带来的定义域越界。$\Delta$ 必须明确是帧数还是时间间隔；一帧与一秒不是同一口径。只有位置参考时，不能计算参考中不存在的真实旋转 RPE。

ATE 与 RPE 回答不同问题：局部相对误差可以不大，却持续朝一个方向积累；整体对齐又可能吸收共同偏移。因此，应同时保留全序列的误差范围、末端偏差和失败覆盖，而不是只选一个低数值。数据加载是否读取参考轨迹，与在线位姿是否被参考值替代也要分别判断。独立评测应避免把参考位姿用于初始化、插点或在线候选选择。

## 9. 失跟、恢复与回环的边界

![LoopSplat 补充材料的真实三维子地图重叠比较：大、中、小重叠条件下的共同区域](../outputs/slam-system-paper-course-001/figures/loopsplat-fig-g01.png)
> 图源：Zhu 等（2024），Fig. G.1 · [原论文](https://arxiv.org/abs/2408.10154) · 裁图 · CC BY 4.0。

图来自后续的 LoopSplat，显示两张子地图中高斯中心形成的三维点云，着色标出了共同区域。图中重叠率依次为 $0.9$、$0.6$、$0.1$；作者拒绝了重叠极低的回环候选。它直接说明跨历史地图建立约束需要足够的共同观测，识别到相似场景不保证配准成功。这里用图引入系统扩展知识，并不是把 LoopSplat 的回环功能归到 SplaTAM。

SplaTAM 原论文的跟踪、插点与地图更新，没有提供一套全局回环或失跟重定位机制；不能把某帧优化重新下降解释成系统已恢复所有历史误差。

支持过少、几何退化、初始化偏离、残差不一致都可能让局部跟踪失败。判断规则能提示哪个条件不满足，却不能直接证明失败根因，也不能保证所有被接受的位姿都准确。失跟后如果只保存上一估计，保存状态并不意味着相机真的停止运动。

### 相邻恢复为什么可能漏掉一段运动

设最后可信参考为帧 $r$，中间帧 $j$ 跟踪失败，当前帧为 $k$。相对变换 $\mathbf T_{a\leftarrow b}$ 把相机 $b$ 的点送到相机 $a$，则

$$
\mathbf A=\mathbf T_{r\leftarrow j},
\qquad
\mathbf B=\mathbf T_{j\leftarrow k},
\qquad
\mathbf T_{r\leftarrow k}=\mathbf A\mathbf B.
$$

若失败时保留 $\widehat{\mathbf C}_j=\widehat{\mathbf C}_r$，只用下一次相邻配准恢复，会得到

$$
\widehat{\mathbf C}_{k,\mathrm{adjacent}}
=\widehat{\mathbf C}_r\mathbf B,
\qquad
\widehat{\mathbf C}_{k,\mathrm{direct}}
=\widehat{\mathbf C}_r\mathbf A\mathbf B.
$$

第一条遗漏了 $r$ 到 $j$ 的运动。恢复一个相邻候选，与跨越缺失区间恢复累计运动，是不同能力。直接对最后可信帧配准有机会补上这段运动，但参考距离变大时，视野重叠、曝光一致性和初值条件也可能恶化。

参考状态必须同时保存图像身份与位姿：相等的两个位姿矩阵并不表示对应同一张观测。若把当前到旧参考的变换误当作当前到相邻帧，矩阵组合的物理含义就变了。运动序列还通常满足 $\mathbf A\mathbf B\ne\mathbf B\mathbf A$，不能交换乘法次序。

### 重定位与回环各增加什么信息

重定位（Relocalization）通过已有地图或历史观测重新求当前全局位姿；回环检测（Loop Closure Detection）判断当前区域是否与过去区域相同，并生成跨时间约束。随后还需要位姿图优化、联合 BA 或其他全局更正机制，才能把新增约束传播到历史状态。

当前帧重定位成功，不代表此前地图已经同步修复；回环候选被识别，也不表示约束已可靠验证或图优化已完成。评价应保留失败期间的所有采样时刻：保持上一位姿的帧仍有真实误差，恢复产生的位姿跳变也应进入相应 RPE。只评价成功帧会隐藏系统失效区间。

## 10. 位姿更正后的高斯地图一致性

![LoopSplat 的三维子地图配准比较，里程计初始对齐、直接高斯配准与参考对齐采用不同列展示](../outputs/slam-system-paper-course-001/figures/loopsplat-fig-g02.png)
> 图源：Zhu 等（2024），Fig. G.2 · [原论文](https://arxiv.org/abs/2408.10154) · 裁图 · CC BY 4.0。

图将两张子地图的高斯中心以不同颜色显示，比较里程计提供的初始对齐、LoopSplat 高斯配准和参考对齐。椅背、墙面等重叠处的分离提示了两张局部地图之间的相对位置误差。图是子地图配准的定性证据；参考对齐也可能有误差，不能凭某一列外观就宣布真实几何完全正确。

配准或回环优化用于估计更正量；下面的坐标几何推导回答给定更正量之后，怎样一致地变换相机和高斯。它不声称 SplaTAM 原方法已经执行了这种更正。若相机移到更合理的位置，而世界高斯仍留在旧坐标，那么中心投影、深度和遮挡通常也会改变。需要先确定更正作用于哪个参考系，再决定地图如何随之处理。

### 同一锚点的刚体更正

设同一锚点相机到世界位姿由 $\mathbf C_{\mathrm{old}}$ 更正为 $\mathbf C_{\mathrm{new}}$。旧世界坐标到更正世界坐标的刚体变换为

$$
\boldsymbol\Delta=
\mathbf C_{\mathrm{new}}\mathbf C_{\mathrm{old}}^{-1}
=\begin{bmatrix}
\mathbf R_\Delta&\mathbf t_\Delta\\
\mathbf0^\mathsf T&1
\end{bmatrix},
\qquad
\mathbf C_{\mathrm{new}}=\boldsymbol\Delta\mathbf C_{\mathrm{old}}.
$$

对已明确归属这个锚点、可作刚体处理的局部高斯，均值和协方差应一起变换：

$$
\boldsymbol\mu_{\mathrm{new}}
=\mathbf R_\Delta\boldsymbol\mu_{\mathrm{old}}+\mathbf t_\Delta,
\qquad
\boldsymbol\Sigma_{\mathrm{new}}
=\mathbf R_\Delta\boldsymbol\Sigma_{\mathrm{old}}\mathbf R_\Delta^\mathsf T.
$$

均值与平移以米计，协方差以平方米计。因为

$$
\mathbf C_{\mathrm{new}}^{-1}\boldsymbol\Delta
=\mathbf C_{\mathrm{old}}^{-1},
$$

相应锚点下的相机空间均值与协方差保持；固定内参下的投影也保持。只更正相机而不移动地图，不满足这个恒等式。对各向异性高斯，只移动均值而漏掉协方差旋转，投影中心可能保持，但椭圆方向或尺度仍会改变。

若协方差按局部主轴参数表示，$\mathbf q_g=(w,x,y,z)^\mathsf T$ 是将高斯局部轴主动旋转到世界的单位 Hamilton 四元数，则

$$
\boldsymbol\Sigma=
\mathbf R(\mathbf q_g)
\begin{bmatrix}
s_1^2&0&0\\
0&s_2^2&0\\
0&0&s_3^2
\end{bmatrix}
\mathbf R(\mathbf q_g)^\mathsf T,
\qquad
\mathbf q_{g,\mathrm{new}}
=\frac{\mathbf q_\Delta\otimes\mathbf q_{g,\mathrm{old}}}
{\|\mathbf q_\Delta\otimes\mathbf q_{g,\mathrm{old}}\|_2}.
$$

$\otimes$ 是 Hamilton 乘法，更正从左侧复合；刚体变换保留三个米制标准差 $s_1,s_2,s_3$。SplaTAM 的各向同性协方差是特殊情况，旋转不会改变其形状，但中心仍要正确更正。若更正包含尺度，则需要相似变换模型及协方差尺度变换，不能直接沿用这个刚体结论。

### 共同坐标变换与不同局部更正

把所有相机和所有高斯施加同一 $\boldsymbol\Delta$，只是共同坐标规范自由度（gauge freedom）的改变：图像解释保持，没有因此证明真实世界位置更准确。

如果不同锚点得到不同更正 $\boldsymbol\Delta_a,\boldsymbol\Delta_b$，两份原本重合的表面可能分离：

$$
\boldsymbol\mu_a'=\mathbf R_{\Delta_a}\boldsymbol\mu+\mathbf t_{\Delta_a},
\qquad
\boldsymbol\mu_b'=\mathbf R_{\Delta_b}\boldsymbol\mu+\mathbf t_{\Delta_b},
\qquad
\boldsymbol\mu_a'-\boldsymbol\mu_b'\ne\mathbf0\ \text{通常成立}.
$$

每份局部地图在自己的锚点下保持观测，不会自动保证跨锚点表面仍重合。还需要重叠约束、重新融合、地图优化或连续变形模型。一个高斯经过多个关键帧共同优化后，初始化来源帧也未必是它唯一的更正归属，不能简单按来源帧逐点套用更正后就称为地图修复。

SplaTAM 使用视角无关 RGB，所以刚体坐标搬移可以保留其常量颜色和不透明度。若扩展回传统 3DGS 的球谐外观，世界旋转还会改变视角方向的表达，需要旋转外观系数或把查询方向变回原参考系。几何坐标一致与视角相关颜色一致必须分别处理。

### 参考资料与阅读衔接

方法与公开实验依据 [SplaTAM 原论文及补充材料](https://arxiv.org/abs/2312.02126)：§3 与 Eq. (1)–(9) 对应前六节，§4–5 与 Tables 1、3–6 对应实验解释，补充材料 Fig. S.1 与 Tables S.1–S.2 对应三维叠加与表示比较。原论文方法和实验的中文改写以及论文裁图遵循 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；本站增加的推导已在正文标明。第 9–10 节使用 [LoopSplat 预印本及补充材料](https://arxiv.org/abs/2408.10154)的真实三维图，裁图依据 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。独立评测、恢复和坐标更正不能作为 SplaTAM 已实现相关功能的声明。

回看基础可读[传统 3DGS 渲染](27_3dgs_rendering_foundations.md)与[训练及自适应密度控制](28_3dgs_training_density_control.md)。数据与原实现可查 [SplaTAM 项目](https://spla-tam.github.io/)、[固定官方版本](https://github.com/spla-tam/SplaTAM/tree/da6bbcd24c248dc884ac7f49d62e91b841b26ccc)和 [TUM 轨迹评价说明](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools)。具体实现配置的归一化、门控和关键帧访问规则需要另行核对，不能仅凭论文的紧凑公式认定逐项相同。

## 实验附录

### room 局部续跑中的固定支持比较

从第 800、1100 帧检查点分别续跑到第 842、1137 帧，完成 79 帧跟踪、15,800 次跟踪更新与 2,370 次建图迭代。78 帧的原目标下降，其中 38 帧的位置误差反而增加；固定初始像素集合后，74 帧目标下降，其中 36 帧位置误差增加。第 1136 帧的固定支持目标下降约 2.29%，位置误差却从 0.956 米增加到 1.558 米。

旧检查点未保存完整随机状态与辅助量，这两段是新的局部续跑，不能恢复原运行在相同帧号的即时地图，也不能宣称重现了原漂移的因果过程。结论限于：本次续跑中的误差增加不能仅由支持像素数量变化解释。曲线、独立复算与剩余问题见[局部续跑与固定支持](../experiments/34_three_track_mechanism_followup.md#splatam-局部续跑与固定支持)。

### 候选选择一致性的局部回放

在 room 的三个已保存地图之后，分别重新跟踪第 101、801、1101 帧，每帧 200 步，共 600 次优化器更新。保持同一初值、地图、渲染器与优化路径，只比较“按更新前损失保存更新后位姿”和“保存该损失实际对应位姿”两种候选选择。后者的重算损失分别下降 2.152、60.315、52.078，但固定原对齐后的独立位置误差变化分别为 +0.048、−0.066、+0.557 毫米，未显示一致精度改善。

这证明本次三个候选的目标值选择更一致，不能解释 room 的整体漂移，也不构成完整序列修复。原始源码和地图保持不变；下一帧局部回放也不等同于恢复原运行每一次浮点状态。曲线、核验和此前失败记录的边界见[三条主线的机制验证](../experiments/34_three_track_mechanism_followup.md)。

本地已完成完整序列、分辨率对照与重复消融，长序列也呈现了累计漂移。配置、图表与数值保留在[场景与分辨率](../experiments/29_splatam_sequences_resolution.md)、[重复与消融](../experiments/30_splatam_tracking_ablation_repeat.md)和[漂移机制对照](../experiments/32_slam_drift_mechanism.md)，用于检验以上知识的适用条件。


### 固定配置的真值位姿开关

默认 `use_gt_poses=False` 不用逐帧 GT 代替 Tracking；GT 仍参与数据关联、参考系处理与评测。

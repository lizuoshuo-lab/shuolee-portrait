---
id: "gslab-k28"
slug: "3dgs-training-density-control"
title: "传统 3DGS：训练、增密与剪枝"
order: 28
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://arxiv.org/abs/2308.04079"
  - "https://arxiv.org/html/2308.04079v1"
  - "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/train.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/scene/gaussian_model.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/arguments/__init__.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/utils/loss_utils.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/scene/dataset_readers.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/gaussian_renderer/__init__.py"
---

# 传统 3DGS：训练、增密与剪枝

[上一章](27_3dgs_rendering_foundations.md)解释了三维高斯怎样生成图像。本章沿着**优化目标 → 初始化与参数 → 梯度及观测约束 → 密度控制 → 消融作用 → 局限**这条线，综合原论文 §5 与 §7，解释一张高斯地图如何从照片中建立起来。配图全部使用论文原图，公式补充必要的参数定义和单位。

传统 3D Gaussian Splatting（3DGS）面向静态场景的新视角合成，相机参数和稀疏点云通过运动恢复结构（Structure from Motion，SfM）预先估计。训练时固定相机参数，只改变高斯地图。后文将这条条件与未知相机位姿下的 SplaTAM 连接起来。

## 1. 优化目标：一张高斯地图解释多张照片

![原论文方法概览：SfM稀疏点云初始化三维高斯，随后交替优化参数和调整密度](../outputs/3dgs-paper-course-001/figures/paper-fig-02.png)

> 图源：Kerbl 等（2023），Fig. 2 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。


原图中的照片、相机和三维高斯共同规定优化问题：相机参数固定，当前高斯地图经过投影和光栅化生成图像，照片与渲染图像的差异再通过梯度改变地图。密度控制位于同一循环中，用来调整表示的数量和分布。**所有训练视角共同更新同一张地图，这是多视图约束的来源。**

记高斯参数集合为 $\boldsymbol{\Theta}=\{\boldsymbol{\mu}_i,\mathbf{s}_i,\mathbf{q}_i,o_i,\boldsymbol{\beta}_i\}_{i=1}^{N}$，训练视角集合为 $\mathcal{V}_{\mathrm{train}}$，渲染关系为 $\mathcal{R}$。优化目标是：

$$
\widehat{\mathbf{I}}_v=
\mathcal{R}(\boldsymbol{\Theta};\mathbf{K}_v,\mathbf{R}_v,\mathbf{t}_v),
\qquad
\boldsymbol{\Theta}^{*}=
\underset{\boldsymbol{\Theta}}{\operatorname{argmin}}
\frac{1}{|\mathcal{V}_{\mathrm{train}}|}
\sum_{v\in\mathcal{V}_{\mathrm{train}}}\mathcal{L}_v.
$$

其中 $\widehat{\mathbf{I}}_v$ 和 $\mathbf{I}_v$ 分别为渲染与真实 RGB 图像。原论文式（7）采用平均绝对误差与结构不相似性损失的组合：

$$
\mathcal{L}_{1,v}=
\frac{1}{3H_vW_v}
\sum_{p=1}^{H_vW_v}\sum_{c=1}^{3}
\left|\widehat I_v(p,c)-I_v(p,c)\right|,
$$

$$
\mathcal{L}_v=(1-\lambda)\mathcal{L}_{1,v}
+\lambda\bigl(1-\overline{\operatorname{SSIM}}_v\bigr),
\qquad \lambda=0.2.
$$

$H_v,W_v$ 是图像高宽，$p$ 为像素索引，$c$ 为颜色通道索引；分母是图像的颜色数值总数，与高斯数量无关。颜色范围采用 $[0,1]$ 时，平均绝对误差无量纲。论文中的 D-SSIM 项在对应官方实现中为 $1-\operatorname{SSIM}$，若另采用除以 $2$ 的定义，应同时调整权重。

结构相似性（Structural Similarity，SSIM）比较对应局部窗口的亮度、对比度和结构：

$$
\operatorname{SSIM}(\mathbf{x},\mathbf{y})=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
$$

在窗口内使用非负、归一化权重 $\omega_j$，统计量为：

$$
\sum_{j=1}^{M}\omega_j=1,
\quad
\mu_x=\sum_{j=1}^{M}\omega_jx_j,
\quad
\sigma_x^2=\sum_{j=1}^{M}\omega_j(x_j-\mu_x)^2,
\quad
\sigma_{xy}=\sum_{j=1}^{M}\omega_j(x_j-\mu_x)(y_j-\mu_y).
$$

$\mu_y$、$\sigma_y^2$ 同理；$C_1,C_2>0$ 是与颜色动态范围相容的稳定常数。$\overline{\operatorname{SSIM}}_v$ 为窗口和颜色通道上的平均值。$L_1$ 直接约束颜色差，SSIM 补充局部结构差异，两项均由图像计算。

单个视角中，沿观察方向移动高斯、改变不透明度或颜色，可能互相补偿；换到另一个视角后，这些补偿通常不能同时成立。多视图因此增加约束。不过遮挡、有限视角与外观自由度仍会留下歧义，图像拟合目标本身没有提供真实表面距离监督。

## 2. SfM 初始化与受约束的高斯参数

![原论文把训练后的三维高斯缩小以显露其实际各向异性形状](../outputs/3dgs-paper-course-001/figures/paper-fig-03.png)

> 图源：Kerbl 等（2023），Fig. 3 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。


原论文的输入是一组静态场景照片，以及 SfM 求得的相机内外参数和稀疏三维点。点云只在图像特征能够可靠匹配、三角化的位置提供起点。**初始化提供几何先验，后续密度控制补充初始点云未覆盖的区域。**

设世界坐标中的 SfM 点为 $\mathbf{x}_i$，第 $v$ 个相机的世界到相机变换为：

$$
\mathbf{x}_{C_v}=\mathbf{R}_v\mathbf{x}_W+\mathbf{t}_v,
\qquad
\mathbf{K}_v=
\begin{bmatrix}
f_{xv}&0&c_{xv}\\
0&f_{yv}&c_{yv}\\
0&0&1
\end{bmatrix}.
$$

$\mathbf{R}_v\in SO(3)$ 为旋转矩阵，$\mathbf{t}_v\in\mathbb{R}^3$ 为平移，$\mathbf{K}_v$ 为相机内参矩阵；焦距和主点坐标以像素计。点云与平移使用同一场景长度单位。单目 SfM 通常只有统一的相对尺度，除非输入另有尺度标定，不能直接把这个单位当作米。

每个 SfM 点初始化一个三维高斯。论文 §5.1 用最近三个点的平均距离估计初始轴尺度。用 $\mathcal{N}_3(i)$ 表示这三个近邻，则论文描述可写成：

$$
\boldsymbol{\mu}_i=\mathbf{x}_i,
\qquad
\bar d_i=\frac13\sum_{j\in\mathcal{N}_3(i)}
\|\mathbf{x}_i-\mathbf{x}_j\|_2,
\qquad
s_{i1}=s_{i2}=s_{i3}=\bar d_i.
$$

$\boldsymbol{\mu}_i$ 是高斯中心；$\bar d_i$ 和三个轴标准差 $s_{i1},s_{i2},s_{i3}$ 都是长度。初始三个轴相等，因此是各向同性高斯；它不需要由稀疏点云预先估计表面法向。优化后，三个轴的长度和方向可以不同，从而贴合墙面、枝叶和细杆等结构。

邻域距离只决定覆盖范围的起点，不能直接解释为真实表面厚度。初始颜色来自 SfM 点的颜色；较低的初始不透明度使重叠高斯能够在训练中重新分配贡献。相机参数预先给定，也并不表示它们绝无误差：图像重投影偏差仍可能被高斯位置或外观吸收。


图左显示最终合成图像，右侧按作者说明缩小高斯，以显露隐藏在渲染结果中的椭球形状。平坦区域可以用沿表面展开的高斯覆盖；薄结构则需要较窄、方向合适的高斯。图中的椭球是高斯分布的几何可视化，渲染时仍使用连续的高斯权重。

第 $i$ 个高斯的参数包括中心 $\boldsymbol{\mu}_i$、三个轴标准差 $\mathbf{s}_i$、旋转四元数 $\mathbf{q}_i$、不透明度 $o_i$ 和球谐系数 $\boldsymbol{\beta}_i$。为避免优化产生负尺度或非法协方差，使用无约束变量 $\ell_{ij}$、$a_i$ 和非零四元数 $\widetilde{\mathbf{q}}_i$：

$$
s_{ij}=s_{\mathrm{ref}}\exp(\ell_{ij}),
\qquad
o_i=\frac{1}{1+\exp(-a_i)},
\qquad
\mathbf{q}_i=
\frac{\widetilde{\mathbf{q}}_i}{\|\widetilde{\mathbf{q}}_i\|_2}.
$$

$s_{\mathrm{ref}}>0$ 是参考长度；$\ell_{ij}=\ln(s_{ij}/s_{\mathrm{ref}})$ 为无量纲对数尺度；$a_i=\ln(o_i/(1-o_i))$ 是不透明度的对数几率。上述关系保证 $s_{ij}>0$ 和 $0<o_i<1$。原实现按输入场景长度单位直接使用指数尺度，本讲义写出参考长度以明确单位。

四元数采用实部在前的顺序 $\mathbf{q}_i=(w_i,x_i,y_i,z_i)^{\mathsf T}$。它表示高斯局部主轴坐标到世界坐标的旋转：

$$
\mathbf{R}_i=
\begin{bmatrix}
1-2(y_i^2+z_i^2)&2(x_iy_i-w_iz_i)&2(x_iz_i+w_iy_i)\\
2(x_iy_i+w_iz_i)&1-2(x_i^2+z_i^2)&2(y_iz_i-w_ix_i)\\
2(x_iz_i-w_iy_i)&2(y_iz_i+w_ix_i)&1-2(x_i^2+y_i^2)
\end{bmatrix}.
$$

$$
\boldsymbol{\Sigma}_i=
\mathbf{R}_i
\begin{bmatrix}
s_{i1}^2&0&0\\
0&s_{i2}^2&0\\
0&0&s_{i3}^2
\end{bmatrix}
\mathbf{R}_i^{\mathsf T}.
$$

$\boldsymbol{\Sigma}_i$ 为世界坐标中的 $3\times3$ 协方差矩阵，单位是长度平方。正轴尺度和合法旋转保证它对称正定；$\mathbf{R}_i^{\mathsf T}\mathbf{R}_i=\mathbf{I}_3$、$\det\mathbf{R}_i=1$。这里的 $s_{ij}$ 是标准差，不能直接替代协方差对角元素中的 $s_{ij}^2$。

视角相关颜色用球谐函数（Spherical Harmonics，SH）表达。设 $\mathbf{C}_v$ 为相机中心，观察方向为：

$$
\mathbf{d}_{iv}=
\frac{\boldsymbol{\mu}_i-\mathbf{C}_v}
{\|\boldsymbol{\mu}_i-\mathbf{C}_v\|_2},
\qquad
\widetilde c_{ic}(\mathbf{d}_{iv})=
\frac12+
\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}
\beta_{i\ell m,c}Y_{\ell m}(\mathbf{d}_{iv}).
$$

$$
c_{ic}(\mathbf{d}_{iv})=
\begin{cases}
\widetilde c_{ic}(\mathbf{d}_{iv}),&\widetilde c_{ic}(\mathbf{d}_{iv})\geq0,\\
0,&\widetilde c_{ic}(\mathbf{d}_{iv})<0,
\end{cases}
\qquad c\in\{\mathrm R,\mathrm G,\mathrm B\}.
$$

$Y_{\ell m}$ 是实球谐基函数，$\beta_{i\ell m,c}$ 是红、绿、蓝通道的系数，$L$ 为最高阶数。上式给出官方实现所采用的颜色偏置和逐通道下限截断约定；球谐展开共有 $(L+1)^2$ 个基函数。它允许同一个高斯从不同方向呈现不同颜色，但这些系数并不是独立恢复出的材质与光照参数。

## 3. 梯度优化与局部可观测性

![原论文Garden场景限制前10个高斯接收梯度与完整反向传播的对照](../outputs/3dgs-paper-course-001/figures/paper-fig-09.png)

> 图源：Kerbl 等（2023），Fig. 9 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。


原图在 Garden 场景中比较每个像素只让最前方 $10$ 个高斯接收梯度，与完整反向传播的结果。左图退化说明：后方高斯参与图像合成时，也需要相应的优化信号；前向图像计算和反向梯度不能随意使用不同的贡献范围。这个消融检验的是梯度近似的影响，并未证明完整方法中的每一个参数都可由图像唯一确定。

每次选择一个训练视角，渲染、计算损失，再调整高斯参数。在当前可微分支中，链式法则为：

$$
\nabla_{\boldsymbol{\Theta}}\mathcal{L}_v=
\left(
\frac{\partial\operatorname{vec}(\widehat{\mathbf{I}}_v)}
{\partial\boldsymbol{\Theta}}
\right)^{\mathsf T}
\nabla_{\operatorname{vec}(\widehat{\mathbf{I}}_v)}\mathcal{L}_v.
$$

$\operatorname{vec}$ 把图像排列为向量；若共有 $D$ 个标量优化变量，渲染雅可比为 $3H_vW_v\times D$。一个像素依赖多个重叠高斯，每个高斯又影响多个像素，所以误差需要经过投影、形状和透明度合成关系共同传播。参数约束还贡献导数：

$$
\frac{\partial\mathcal{L}}{\partial a_i}
=\frac{\partial\mathcal{L}}{\partial o_i}o_i(1-o_i),
\qquad
\frac{\partial\mathcal{L}}{\partial\ell_{ij}}
=\frac{\partial\mathcal{L}}{\partial s_{ij}}s_{ij}.
$$

原论文附录 B 使用自适应矩估计（Adaptive Moment Estimation，Adam）更新参数。对任一参数组 $\boldsymbol{\theta}$，令 $\mathbf{g}_t=\nabla_{\boldsymbol{\theta}}\mathcal{L}_t$，则：

$$
\begin{aligned}
\mathbf{u}_t&=\beta_1\mathbf{u}_{t-1}+(1-\beta_1)\mathbf{g}_t,\\
\mathbf{v}_t&=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)(\mathbf{g}_t\odot\mathbf{g}_t),\\
\widehat{\mathbf{u}}_t&=\frac{\mathbf{u}_t}{1-\beta_1^t},
\qquad
\widehat{\mathbf{v}}_t=\frac{\mathbf{v}_t}{1-\beta_2^t},\\
\boldsymbol{\theta}_t&=\boldsymbol{\theta}_{t-1}
-\boldsymbol{\eta}_t\odot
\frac{\widehat{\mathbf{u}}_t}{\sqrt{\widehat{\mathbf{v}}_t}+\varepsilon}.
\end{aligned}
$$

$\mathbf{u}_t$、$\mathbf{v}_t$ 分别是一阶、二阶矩估计，$0\leq\beta_1,\beta_2<1$ 控制历史衰减；$\boldsymbol{\eta}_t$ 为各分量学习率，$\varepsilon>0$ 为数值稳定项，乘法、平方根和除法逐分量进行。位置有长度单位，旋转、颜色和不透明度具有不同参数尺度，因此不同组需要不同的步长。原论文仅对位置使用指数衰减的学习率日程。

论文 §7.1 还采用逐步增加问题难度的策略：优化开始使用较低图像分辨率，在第 $250$ 和 $500$ 次迭代上采样；球谐函数先优化零阶项，再每 $1000$ 次迭代加入一阶，直到四个频带全部使用，即 $L=3$。零阶项只表达与方向无关的颜色，较高阶项增加视角相关变化。这样可以减少角度覆盖不足时基础颜色与方向项互相补偿的风险，但不构成几何正确性或全局最优性的保证。

### 图像能够约束哪些参数

梯度说明当前误差要求参数怎样变化；局部可观测性讨论参数变化能否在观测中留下可区分的信号。下面用局部线性化解释这一点，是对图像目标的理论补充，原论文的优化器仍采用上面的 Adam 更新。

固定当前可见性和排序分支，把所有训练视图的颜色数值排列为向量 $\mathbf{y}(\boldsymbol{\theta})$，其雅可比记为 $\mathbf{J}_{I}$：

$$
\Delta\mathbf{y}\approx\mathbf{J}_{I}\Delta\boldsymbol{\theta},
\qquad
\mathbf{J}_{I}=\frac{\partial\mathbf{y}}{\partial\boldsymbol{\theta}}.
$$

若存在非零方向 $\Delta\boldsymbol{\theta}$ 使 $\mathbf{J}_{I}\Delta\boldsymbol{\theta}=\mathbf{0}$，则这个方向在当前一阶近似下不改变观测，存在局部不可观测方向。把第 $r$ 个颜色观测的雅可比行写成 $\mathbf{j}_r$，无量纲正权重写成 $\omega_r>0$，可用下式检查这种约束：

$$
\mathbf{H}_{I}=\sum_{r=1}^{M}\omega_r\mathbf{j}_r^{\mathsf T}\mathbf{j}_r,
\qquad
\Delta\boldsymbol{\theta}^{\mathsf T}\mathbf{H}_{I}\Delta\boldsymbol{\theta}
=\sum_{r=1}^{M}\omega_r(\mathbf{j}_r\Delta\boldsymbol{\theta})^2.
$$

$M$ 是所使用的颜色观测总数；若参数有 $D$ 个分量，则 $\mathbf{J}_{I}\in\mathbb{R}^{M\times D}$、$\mathbf{H}_{I}\in\mathbb{R}^{D\times D}$。各列雅可比的单位随参数变化：中心列为颜色单位除以长度，无量纲外观参数列为颜色单位。比较矩阵特征值时必须先选定各参数的归一化尺度。

纹理均匀区域的位置变化可能只造成很小的图像差；长期被遮挡的高斯缺乏有效观测；少数相近视角下，中心、尺度、不透明度和颜色也可能互相补偿。多视图增加雅可比的行，能够消除部分歧义，但是否消除取决于实际覆盖和外观。**大位置梯度是增密的敏感度信号，不能直接当作充分可观测性的证明。**

## 4. 复制、分裂、剪枝与不透明度重置

![原论文Fig4上排展示覆盖不足后的复制，下排展示一个大高斯分裂后继续优化](../outputs/3dgs-paper-course-001/figures/paper-fig-04.png)

> 图源：Kerbl 等（2023），Fig. 4 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

原论文 Fig. 4 上排是覆盖不足的情况：已有高斯较小，复制增加可独立优化的基元，随后两者继续调整。下排是一个大高斯难以表达细节的情况：用较小子高斯替换它，分别调整位置和形状。黑色轮廓用于解释目标结构，训练并不使用这条轮廓做逐点几何监督。**密度控制分配局部表达能力，连续优化决定新基元最终怎样贡献图像。**


论文依据视图空间位置梯度的平均模长选择候选。用 $\mathbf{m}_{iv}$ 统一记高斯 $i$ 在视图 $v$ 的二维投影位置，$\mathcal{V}_i$ 为统计区间内有效观测索引集合，则：

$$
\bar g_i=
\frac{1}{n_i}\sum_{v\in\mathcal{V}_i}
\left\|\frac{\partial\mathcal{L}_v}{\partial\mathbf{m}_{iv}}\right\|_2,
\qquad n_i=|\mathcal{V}_i|>0.
$$

论文 §5.2 用 $\bar g_i>\tau_g$ 选择增密候选，实验设 $\tau_g=0.0002$。对应官方实现使用大于等于的比较，并累积二维投影位置的内部梯度模长。这个数值属于作者的坐标、损失和统计约定；若 $\mathbf{m}_{iv}$ 改用像素或归一化坐标，阈值需要相应换算。若用像素坐标讲解，梯度单位为损失单位除以像素。没有有效统计观测时，$n_i=0$，不能按上式计算平均值。

这里先取每次梯度的模长，再平均，避免不同视角的方向相消。较大梯度说明图像损失对投影位置敏感，可能来自覆盖不足、错误形状或遮挡；它不是高斯到真实表面的距离测量。

为了写清官方实现的尺度判据，记场景尺度为 $L_{\mathrm{scene}}>0$，相对标准差为 $r_i=\max_j s_{ij}/L_{\mathrm{scene}}$。固定官方版本的复制、分裂候选为：

$$
\mathcal{C}=\{i\mid\bar g_i\geq\tau_g,\ r_i\leq\tau_s\},
\qquad
\mathcal{S}=\{i\mid\bar g_i\geq\tau_g,\ r_i>\tau_s\},
\qquad \tau_s=0.01.
$$

$r_i$ 和 $\tau_s$ 无量纲；$L_{\mathrm{scene}}$ 按该实现的场景范围约定确定，不是物体任意指定的尺寸。 **复制** 保留父高斯并新增一组可独立优化的参数，数量由 $N$ 变成 $N+1$。论文描述复制后沿位置梯度方向移动新高斯；所核对的官方版本则从父高斯的相同中心和属性开始，没有显式加入这一初始位移，之后继续优化。因此不能把论文的示意位移当作该版本必然执行的操作。

 **分裂** 删除父高斯，用其分布采样两个子高斯的位置，并把三个轴标准差都除以 $1.6$：

$$
\boldsymbol{\epsilon}_k\sim\mathcal{N}(\mathbf{0},\mathbf{I}_3),
\qquad
\boldsymbol{\mu}_{ik}=\boldsymbol{\mu}_i+
\mathbf{R}_i
\begin{bmatrix}
s_{i1}&0&0\\
0&s_{i2}&0\\
0&0&s_{i3}
\end{bmatrix}
\boldsymbol{\epsilon}_k,
\qquad k\in\{1,2\},
$$

$$
\mathbf{s}_{ik}=\frac{\mathbf{s}_i}{1.6},
\qquad
\mathbf{I}_3=
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1
\end{bmatrix},
\qquad N'=N-1+2=N+1.
$$

$\boldsymbol{\epsilon}_k$ 无量纲，中心偏移具有场景长度单位；子高斯继承父高斯的初始旋转、颜色和不透明度，再单独更新。这个操作增加局部自由度，不严格保持体积或渲染结果：对于同一马氏距离边界的椭球，子体积之和与父体积之比是 $2/1.6^3\approx0.488$，合成颜色还受重叠及不透明度影响。论文关于维持总体覆盖的描述应理解为几何表达的直观动机。

论文预热后每 $100$ 次迭代进行密度控制；固定官方版本的默认增密区间从 $500$ 到 $15000$ 次迭代，完整训练为 $30000$ 次。新增、删除和阈值选择都是离散操作，所以这是一条交替执行连续优化与表示调整的过程。

### 剪枝与不透明度重置


增密也可能扩大冗余表示。如果不必要的高斯靠近输入相机，图像拟合可能保留这些局部基元并持续增加数量。论文因此把删除弱贡献高斯、限制过大的高斯和周期性降低不透明度结合使用。

 **剪枝（pruning）** 从地图中删除高斯。低不透明度的判断为：

$$
o_i<\tau_o,
\qquad 0<\tau_o<1.
$$

所核对官方版本使用 $\tau_o=0.005$；论文 §5.2 将它记为 $\varepsilon_{\alpha}$，该段没有给出数值。世界坐标中的轴标准差过大和图像投影范围过大，也可触发剪枝。低不透明度只是一种弱贡献近似判据，暂时被单个相机遮挡不能证明高斯对全部视图都无用。

 **不透明度重置** 保留高斯并压低其不透明度。论文每 $3000$ 次迭代执行一次，随后让训练恢复需要的贡献，低贡献基元再由剪枝移除。对应官方版本的关系为：

$$
o_i'=\begin{cases}
o_i,&o_i\leq0.01,\\
0.01,&o_i>0.01.
\end{cases}
$$

重置不改变高斯中心和数量，也不把原本更低的不透明度抬高。它的遮挡作用可以通过前后合成关系理解。若第 $k$ 个高斯之前的透射率为 $T_k(p)$，则：

$$
T_k(p)=\prod_{j<k}(1-\alpha_j(p)),
\qquad
\frac{\partial\mathbf{C}(p)}{\partial\mathbf{c}_k}
=T_k(p)\alpha_k(p)\mathbf{I}_3.
$$

$\alpha_j(p)$ 是不透明度乘以二维高斯权重得到的像素有效不透明度，$\mathbf{C}(p)$ 是最终 RGB 颜色，$\mathbf{c}_k$ 为高斯颜色。在前景遮挡很强时，$T_k$ 很小，后景颜色梯度受到衰减。降低前景遮挡能够重新分配观察信号，但后景不透明度也可能同时变小，因此梯度并不保证立刻增大。重置可能暂时降低图像质量；它是缓和漂浮高斯与密度增长的策略，不保证消除所有错误几何。

## 5. 消融实验说明每个设计的作用

![原论文Garden随机初始化上排与SfM初始化下排的对照，右侧放大背景细节](../outputs/3dgs-paper-course-001/figures/paper-fig-07.png)

> 图源：Kerbl 等（2023），Fig. 7 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

### 消融结论适用的实验范围

原论文真实场景评价包含 Mip-NeRF360 的 $9$ 个场景、Tanks & Temples 的 $2$ 个场景和 Deep Blending 的 $2$ 个场景；每 $8$ 张照片留出 $1$ 张用于测试，3DGS 的测量硬件为 NVIDIA A6000。下面的数值均是原论文结果。消融表 3 则只使用 Truck、Garden、Bicycle，分别比较 $5000$ 与 $30000$ 次迭代，并将高分辨率原图手工缩小到既定渲染分辨率。应在同组场景、相同图像处理和预算下读这些差异。

峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）从图像均方误差计算：

$$
\operatorname{MSE}=\frac{1}{3HW}
\sum_{p=1}^{HW}\sum_{c=1}^{3}
\bigl(\widehat I(p,c)-I(p,c)\bigr)^2,
\qquad
\operatorname{PSNR}=10\log_{10}\frac{A^2}{\operatorname{MSE}}.
$$

$A$ 为颜色动态范围的最大值，$[0,1]$ RGB 取 $A=1$；PSNR 以分贝（dB）计，越高表示平均颜色平方误差越小。论文同时报告 SSIM 和学习感知图像块相似度（Learned Perceptual Image Patch Similarity，LPIPS）；前者比较局部结构，后者比较学习得到的特征差异。 **这组指标评价新视角图像，不直接评价表面距离或相机轨迹。** 

SfM 初始化的作用则由 Fig. 7 的具体背景差异支持。作者用相机包围范围三倍大小的立方体均匀随机初始化后，仍能重建部分场景，但弱观测背景中残留更多漂浮物；表 3 的三场景 $30000$ 次平均 PSNR 从完整方法的 $26.05$ 降至 $20.42\,\mathrm{dB}$。这说明几何先验在该实验中显著帮助优化，不能解释为随机初始化在所有受控条件下都会失败。

受控合成场景提供一个边界条件：论文在相机准确、视角较充分且没有复杂背景的 Blender 场景中，随机初始化仍能得到较好的结果。初始化先验的作用必须与观测条件共同解释。

![原论文5K迭代下完整方法、禁止复制与禁止分裂的图像对照](../outputs/3dgs-paper-course-001/figures/paper-fig-08.png)

> 图源：Kerbl 等（2023），Fig. 8 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。


消融实验固定其余设置，每次去掉一个设计，再检查图像和量化指标。Fig. 8 的完整方法、禁止复制、禁止分裂都在 $5000$ 次迭代比较：复制帮助小基元扩展覆盖，尤其有利于细结构较早出现；分裂使大基元有机会重新表达背景和局部细节。两种操作因此提供不同的自由度。

对密度控制，$30000$ 次迭代三场景平均 PSNR 为：完整方法 $26.05\,\mathrm{dB}$，禁止分裂 $23.90\,\mathrm{dB}$，禁止复制 $25.91\,\mathrm{dB}$。这组结果支持分裂在这些测试中影响较大，复制有助于较早收敛和薄结构；它不支持“每个场景复制后一定提高最终分数”。例如 Truck 在相同表中禁止复制为 $24.82\,\mathrm{dB}$，完整方法为 $24.81\,\mathrm{dB}$，应保留这种场景差异。

![原论文各向异性与各向同性高斯的图像对照，包含叶片和细小结构](../outputs/3dgs-paper-course-001/figures/paper-fig-10.png)

> 图源：Kerbl 等（2023），Fig. 10 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

各向同性消融只允许一个轴标准差，三个方向同时伸缩；各向异性允许不同轴尺度和方向。Fig. 10 中各向异性能够更好地贴合细结构。作者为展示 Ficus 的区别，在两种配置中都把数量限制在最多 $5000$ 个高斯，这个小规模可视化设定不能当作全部实验的点数上限。表 3 三场景 $30000$ 次平均 PSNR 为各向同性 $25.23$、完整方法 $26.05\,\mathrm{dB}$，支持形状自由度的作用，而非单纯增加点数的解释。

完整反向传播的作用已由第 3 节的 Fig. 9 展示。对应的有界量化例子是表 3 的 Truck：$30000$ 次迭代时，限制最前方 $10$ 个高斯接收梯度为 $13.84\,\mathrm{dB}$，完整方法为 $24.81\,\mathrm{dB}$。Garden 图像与 Truck 数值分别来自不同场景，应按各自对象解读。

球谐函数消融去掉方向外观后，表 3 三场景 $30000$ 次平均 PSNR 从 $26.05$ 变成 $25.35\,\mathrm{dB}$。这个结果说明视角相关外观可以改善图像表达；结合第 1 节，它也提醒我们：更好的图像拟合可能包含外观补偿，不能直接转换为更好的几何精度结论。

## 6. 局限：观测不足、可见性近似与后续约束


![原论文Train弱观测区域的失败案例，3DGS出现粗大各向异性高斯和细节不足](../outputs/3dgs-paper-course-001/figures/paper-fig-11.png)

> 图源：Kerbl 等（2023），Fig. 11 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

Fig. 11 的 Train 背景说明：输入视角不能充分约束的区域，可能留下粗大的各向异性高斯，渲染细节不足。前面讨论的连续优化与密度控制仍受已有图像的信息限制；增加参数不会自动产生缺失的观测。

![原论文DrJohnson偏离训练视角后的伪影，显示视角覆盖不足时的新视角泛化限制](../outputs/3dgs-paper-course-001/figures/paper-fig-12.png)

> 图源：Kerbl 等（2023），Fig. 12 · [原论文](https://arxiv.org/abs/2308.04079) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

Fig. 12 在与训练视角重叠较少的位置展示拉长、斑块状伪影。论文还报告运动过程中偶发的突变：较大高斯可能在剔除边界突然消失，简单深度排序也可能让合成顺序突然切换。外观随方向变化的区域更容易受到这些近似影响。

原论文没有在优化中加入额外正则项，作者把未见区域、可见性变化和抗锯齿作为后续改进方向。存储与训练成本也是独立限制：论文原型在大场景训练时峰值 GPU 显存可能超过 $20\,\mathrm{GB}$；这不是每个场景的最低显存要求，也不是本地运行的测量。已训练模型的参数存储与光栅化临时开销应分别估算。

把这些局限连接到后续知识，可以得到三个明确问题：未知相机位姿需要跟踪；外观与几何之间的歧义需要深度等额外约束；连续采集时，局部新视角拟合还需要与历史地图保持一致。下一章[从传统 3DGS 到 SplaTAM](29_3dgs_to_splatam_progression.md)就在同一套渲染关系上，把相机位姿也加入待估变量，讨论 RGB-D 观测下的跟踪与建图。

### 参考资料与原文位置

Kerbl、Kopanas、Leimkühler、Drettakis，*3D Gaussian Splatting for Real-Time Radiance Field Rendering*（2023）：[论文与项目](https://arxiv.org/abs/2308.04079)、[原论文 PDF](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf)。§5.1、§5.2 位于 PDF 第 5–6 页；§7.1 位于第 8 页；§7.2 位于第 8–9 页；§7.3 位于第 9–10 页；§7.4 位于第 10–11 页；表 1 位于第 8 页，表 3 位于第 9 页。正文以中文译解和机制说明重组这些内容，补充公式与官方实现差异均按对应位置明确标注。

官方实现核对采用固定版本：[高斯参数与密度控制](https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/scene/gaussian_model.py)、[训练与剪枝阈值](https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/train.py)、[默认训练参数](https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/arguments/__init__.py)、[损失定义](https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/utils/loss_utils.py)。这些引用用于区分论文描述与所核对版本，不把实现常数当作不依赖坐标和数据的理论定律。

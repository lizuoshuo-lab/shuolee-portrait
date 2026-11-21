---
id: "gslab-k31"
slug: "dynamic-gaussian-simulation-world-models"
title: "动态高斯与物理仿真：对象、时间、编辑与世界模型"
order: 31
status: "theory_with_local_state_dynamics_and_feedback"
updated: "2026-10-13"
sources:
  - "https://arxiv.org/abs/2310.08528v3"
  - "https://github.com/hustvl/4DGaussians"
  - "https://arxiv.org/abs/2311.12198v3"
  - "https://github.com/XPandora/PhysGaussian/tree/8339ed6aa2cd5d50e1001a254a3d95aea678a956"
  - "https://www.nature.com/articles/s41586-025-08744-2"
  - "https://github.com/danijar/dreamerv3"
---

# 动态高斯与物理仿真：对象、时间、编辑与世界模型

本章保留动态表示、物理仿真与控制的扩展知识。当前重建主线已转向[大场景中移动人物的清晰重建](32_dynamic_human_scene_reconstruction.md)，人物重建效果单独验收。

大场景重建解决了相机、几何和渲染规模的问题。接下来，要让场景中的对象随时间变化，并能接受操作。本章沿着三个问题展开：怎样从视频重建运动，怎样用材料与外力生成新运动，以及怎样根据动作预测未来并利用反馈选择行为。

这三项能力对应不同的模型输入。4D Gaussian Splatting 学习带时间戳的外观与形变；PhysGaussian 用连续介质力学推进对象状态；DreamerV3 学习动作条件下的潜在状态变化，并据此学习策略。三维高斯负责表达和显示场景，而动作、动力学、目标与观测反馈决定系统能够怎样交互。

前置知识：[三维高斯表示与渲染](27_3dgs_rendering_foundations.md)、[训练与密度控制](28_3dgs_training_density_control.md)、[大场景重建与层次细节](30_large_scene_gs_sfm_partition_lod.md)。

## 1. 从静态高斯到带时间的动态表示

![4D Gaussian Splatting 原论文完整图3：时空特征平面、形变解码器与动态渲染](../outputs/dynamic-gaussian-001/assets/4dgs-fig03.png)
> 图源：Wu 等（2024），Fig. 3 · [原论文第 3 页](https://arxiv.org/pdf/2310.08528v3#page=3) · 完整原图，仅裁图外内容；arXiv 分发许可，未声明 CC 许可。

先看完整流程：左侧的三维高斯与时间进入六组特征平面；中间的神经网络读取特征；右侧三个输出分支分别预测位置、旋转和尺度变化；最后仍使用高斯光栅化生成图像。新增的关键变量是时间，而相机投影与不透明度合成沿用传统 3DGS 的基本结构。

记规范空间中的第 $p$ 个高斯为

$$
\mathcal G_p^0=
\left\{\boldsymbol\mu_p^0,\boldsymbol\Sigma_p^0,\alpha_p,\mathbf c_p\right\}.
$$

$\boldsymbol\mu_p^0$ 是高斯中心，单位为场景长度；$\boldsymbol\Sigma_p^0$ 是协方差，单位为长度平方；$\alpha_p$ 是无量纲不透明度；$\mathbf c_p$ 是球谐颜色系数。**规范空间**是跨时间共享的参考表示，不必等于视频某一帧的真实形状。

本章将动态渲染统一写成

$$
\widehat{\mathbf I}_{k,t}
=\mathcal R\!\left(\mathbf K_k,\mathbf T_{cw,k,t},\mathcal G(t)\right),
\qquad
\mathcal G(t)=\mathcal D_\theta(\mathcal G^0,t).
$$

$\mathcal R$ 表示渲染，$\mathcal D_\theta$ 表示可学习的形变模型，$\mathbf T_{cw}$ 是世界到相机的变换。摄像机运动由外参描述，对象运动由高斯状态描述；时间戳使同一时刻的多视图可以监督同一组动态高斯。

图中的六组平面对应

$$
\mathcal P=\{(x,y),(x,z),(y,z),(x,t),(y,t),(z,t)\}.
$$

前三组编码空间结构，后三组表达位置与时间的关联。多分辨率特征经过插值与融合，再由多层感知机预测高斯参数的增量。这样，相邻高斯可以共享时空信息，无须为每个时间点单独保存完整场景。网络使用的时间通常经过归一化；时间数值与秒之间的对应由数据集定义。

为明确参数的约束，用 $\boldsymbol\ell$ 表示对数尺度，用 $\mathbf q=(q_w,q_x,q_y,q_z)^{\mathsf T}$ 表示四元数。对应的有效渲染参数可写为

$$
\boldsymbol\mu_p(t)=\boldsymbol\mu_p^0+\Delta\boldsymbol\mu_p(t),
\qquad
\boldsymbol\ell_p(t)=\boldsymbol\ell_p^0+\Delta\boldsymbol\ell_p(t),
$$

$$
\mathbf q_p(t)=
\frac{\mathbf q_p^0+\Delta\mathbf q_p(t)}
{\left\|\mathbf q_p^0+\Delta\mathbf q_p(t)\right\|_2},
\qquad
\mathbf S_p(t)=
\begin{bmatrix}
e^{\ell_{p,1}(t)}&0&0\\
0&e^{\ell_{p,2}(t)}&0\\
0&0&e^{\ell_{p,3}(t)}
\end{bmatrix},
$$

$$
\boldsymbol\Sigma_p(t)=
\mathbf R(\mathbf q_p(t))\mathbf S_p(t)\mathbf S_p(t)^{\mathsf T}
\mathbf R(\mathbf q_p(t))^{\mathsf T}.
$$

这里 $\mathbf R$ 把高斯局部主轴旋转到世界坐标；指数保证尺度为正，归一化保证四元数有效。上述写法显式区分未激活参数与有效参数，是对原论文增量表达的参数约束说明。

## 2. 训练怎样把运动从模糊外观中分离出来

![4D Gaussian Splatting 原论文完整图4：静态初始化与联合形变优化](../outputs/dynamic-gaussian-001/assets/4dgs-fig04.png)
> 图源：Wu 等（2024），Fig. 4 · [原论文第 4 页](https://arxiv.org/pdf/2310.08528v3#page=4) · 完整原图；arXiv 分发许可，未声明 CC 许可。

这张图从左到右展示随机初始化、3,000 次迭代的静态高斯初始化，以及图示的 20,000 次迭代结果。静态阶段先建立可渲染的空间支撑；随后启用时间形变，联合更新高斯与网络，使不同时间的运动部分各自与观测对应。图中的迭代数属于原作者示例，不是本地训练记录。

把原论文的图像损失与网格总变分正则写成显式加权形式：

$$
\min_{\mathcal G^0,\theta}
\frac{1}{|\mathcal O|}\sum_{(k,t)\in\mathcal O}
\left\|\widehat{\mathbf I}_{k,t}-\mathbf I_{k,t}\right\|_1
+\lambda_{\mathrm{TV}}\mathcal L_{\mathrm{TV}}.
$$

$\mathcal O$ 为有效相机与时间观测集合，$\lambda_{\mathrm{TV}}$ 为平衡权重。图像项决定渲染是否贴近观测，网格总变分项抑制特征平面的剧烈局部变化。颜色损失经过渲染器传播到高斯位置、尺度、旋转及形变网络，所以优化同时影响外观和运动表示。

![4D Gaussian Splatting 原论文完整图5：D-NeRF 合成动态场景中的视觉比较](../outputs/dynamic-gaussian-001/assets/4dgs-fig05.png)
> 图源：Wu 等（2024），Fig. 5 · [原论文第 5 页](https://arxiv.org/pdf/2310.08528v3#page=5) · 完整原图；arXiv 分发许可，未声明 CC 许可。

整图保留 Hook 与 Jack 两个场景、参考图像和全部比较方法。观察运动部位的轮廓与局部纹理，可以理解时变表示解决的问题：每个时间点需要不同的几何状态。原作者在 D-NeRF 等数据集上评估，实验硬件为 RTX 3090；图 5 是论文的定性比较，图 4 的单场景迭代标注不能代替所有数据集的训练预算。

## 3. 对象编辑怎样通过物理规律生成新运动

![PhysGaussian 原论文完整图2：重建、可选内部填充、物理推进与多视角渲染](../outputs/dynamic-gaussian-001/assets/physgaussian-fig02.png)
> 图源：Xie 等（2024），Fig. 2 · [原论文第 3 页](https://arxiv.org/pdf/2311.12198v3#page=3) · 完整原图；arXiv 分发许可，未声明 CC 许可。

PhysGaussian 的整图从一个静态重建出发：优化得到高斯，必要时填充物体内部，再用连续介质力学计算运动，最后从多个相机观察同一物理状态。这里可以直接指定对象区域、材料参数、固定边界和外力。物理状态的推进产生时间变化，渲染器负责显示变化后的对象。

选定待编辑对象的高斯集合 $\mathcal P_o$，每个粒子增加质量 $m_p$、参考体积 $V_p^0$、速度 $\mathbf v_p$ 与形变梯度 $\mathbf F_p$。物理密度和不透明度承担不同的角色：

$$
m_p=\rho_p^0 V_p^0,
\qquad
\rho_p^0>0.
$$

采用 SI 单位时，$m_p$ 为千克，$V_p^0$ 为立方米，$\rho_p^0$ 为千克每立方米。透明程度不能直接替代质量密度。实际重建经常只有相对尺度；若先把对象归一化，材料和力必须使用一致的仿真单位，不能把归一化位移直接当成米。

### 中心与协方差共同随形变变化

用 $\mathbf X$ 表示初始材料坐标，用 $\mathbf x$ 表示当前空间位置。连续形变映射和形变梯度分别是

$$
\mathbf x=\boldsymbol\phi(\mathbf X,t),
\qquad
\mathbf F(\mathbf X,t)=\frac{\partial\boldsymbol\phi}{\partial\mathbf X}.
$$

在每个高斯中心附近作一阶局部近似：

$$
\boldsymbol\phi(\mathbf X,t)
\approx\boldsymbol\mu_p(t)+\mathbf F_p(t)(\mathbf X-\boldsymbol\mu_p^0).
$$

因此，高斯中心和协方差更新为

$$
\boldsymbol\mu_p(t)=\boldsymbol\phi(\boldsymbol\mu_p^0,t),
\qquad
\boldsymbol\Sigma_p(t)=
\mathbf F_p(t)\boldsymbol\Sigma_p^0\mathbf F_p(t)^{\mathsf T}.
$$

中心控制对象移到哪里，协方差控制局部体积怎样旋转、拉伸和剪切。$\mathbf F$ 无量纲；当初始协方差正定且 $\mathbf F$ 可逆时，更新后的协方差仍正定。若只有平移，$\mathbf F=\mathbf I$；若只有刚体旋转，更新退化为熟悉的旋转协方差公式。

### 球谐颜色随材料旋转

![PhysGaussian 原论文完整图5：协方差与球谐方向更新的消融比较](../outputs/dynamic-gaussian-001/assets/physgaussian-fig05.png)
> 图源：Xie 等（2024），Fig. 5 · [原论文第 7 页](https://arxiv.org/pdf/2311.12198v3#page=7) · 完整原图；arXiv 分发许可，未声明 CC 许可。

整图比较参考结果、完整方法、固定协方差、仅旋转协方差，以及固定球谐方向。它检验的是形变后的显示质量。完整方法同时更新中心、协方差和方向相关外观，能够保持图中叶片细节；这一消融与前面的运动重建解决不同问题。

对形变梯度作极分解

$$
\mathbf F_p=\mathbf R_p\mathbf U_p,
\qquad
\mathbf R_p^{\mathsf T}\mathbf R_p=\mathbf I,
$$

$\mathbf R_p$ 是从材料参考方向到当前空间方向的旋转，$\mathbf U_p$ 是对称伸长张量。把当前世界坐标中的单位观察方向 $\mathbf d$ 变回材料坐标，再计算原来的球谐颜色：

$$
\mathbf c_p(t,\mathbf d)
=\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}
\mathbf c_{p,\ell m}\,
Y_{\ell m}\!\left(\mathbf R_p^{\mathsf T}\mathbf d\right).
$$

如此处理，相机与材料表面的相对方向保持一致时，方向相关颜色也能随对象转动。它描述外观随材料方向的搬运，并非重新计算任意新光照。

## 4. 材料与时间步怎样决定运动

![PhysGaussian 原论文完整图3：弹性、金属、断裂、颗粒、黏塑性与接触场景](../outputs/dynamic-gaussian-001/assets/physgaussian-fig03.png)
> 图源：Xie 等（2024），Fig. 3 · [原论文第 6 页](https://arxiv.org/pdf/2311.12198v3#page=6) · 完整原图；arXiv 分发许可，未声明 CC 许可。

每一行左侧是静态重建，右侧是随时间变化的结果。不同运动来自不同材料规律和边界条件：弹性对象可以回弹，金属可以保留塑性形变，颗粒材料可以流散。原作者的这些示例采用 RTX 3090 与 Intel i9-10920X；仿真区域被归一化到边长为 2 的立方体。此处展示论文效果，本地实验在下一节单独说明。

### 从材料参数到内部恢复力

以本地 Ficus 使用的固定共旋转弹性模型（fixed corotated elasticity）为例。杨氏模量 $E$ 衡量材料抗变形的刚度，泊松比 $\nu$ 控制拉伸与横向收缩的关系。相应的拉梅参数为

$$
\mu=\frac{E}{2(1+\nu)},
\qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
$$

这里的标量 $\mu$ 是剪切模量，不是表示高斯中心的粗体 $\boldsymbol\mu_p$。$E,\lambda,\mu$ 在 SI 制中均以帕斯卡为单位，$\nu$ 无量纲。对纯弹性材料，令 $J=\det\mathbf F$，基尔霍夫应力为

$$
\boldsymbol\tau
=2\mu(\mathbf F-\mathbf R)\mathbf F^{\mathsf T}
+\lambda(J-1)J\mathbf I.
$$

第一项响应偏离刚体旋转的形变，第二项响应体积变化。固定 $\nu$ 而减小 $E$，两个弹性模量都随之减小。相同外力下，物体通常更容易变形，其回弹时间尺度也会变化。

### 物质点法的一次推进

物质点法（Material Point Method，MPM）用粒子携带材料状态，用背景网格计算相互作用。记 $w_{ip}$ 为粒子 $p$ 对网格节点 $i$ 的插值权重，$\mathbf C_p$ 为仿射速度矩阵。先将粒子的质量和动量传到网格：

$$
m_i=\sum_p w_{ip}m_p,
\qquad
m_i\mathbf v_i=\sum_p w_{ip}m_p
\left[\mathbf v_p+\mathbf C_p(\mathbf x_i-\mathbf x_p)\right].
$$

再由内部应力和外部作用更新非空网格节点的速度：

$$
\mathbf v_i^{n+1}
=\mathbf v_i^n+
\frac{\Delta t}{m_i}
\left[-\sum_p V_p^0\boldsymbol\tau_p^n\nabla w_{ip}^n
+\mathbf f_i^{\mathrm{ext}}\right].
$$

最后把速度传回粒子，并更新位置及形变梯度：

$$
\mathbf v_p^{n+1}=\sum_i w_{ip}^n\mathbf v_i^{n+1},
\qquad
\mathbf x_p^{n+1}=\mathbf x_p^n+\Delta t\mathbf v_p^{n+1},
$$

$$
\mathbf F_p^{n+1}
=\left(\mathbf I+\Delta t\nabla\mathbf v_p^{n+1}\right)\mathbf F_p^n.
$$

这些公式描述弹性情况；塑性材料还需要弹塑性分解与返回映射。固定底部、外力、碰撞等边界条件分别作用于对应粒子或网格节点。仿真得到新的高斯状态后，才进行下一次渲染。

### 仿真步长与视频帧率

一张视频帧之间通常包含多个物理子步：

$$
\Delta t_{\mathrm{frame}}=N_{\mathrm{sub}}\Delta t.
$$

提高视频帧率不能代替减小物理步长。显式积分的步长需与网格间距和材料波速协调；一个常用的量纲参考是

$$
c\sim\sqrt{\frac{\lambda+2\mu}{\rho}},
\qquad
\Delta t\lesssim C\frac{\Delta x}{c+\|\mathbf v\|}.
$$

$C$ 为与求解器有关的稳定性系数，上式是尺度关系，不是本项目已经证明的稳定性界。刚度提高、网格变细，都可能要求更小的步长。比较两个步长时还要保持总冲量和单位时间阻尼一致，才能把变化主要归因于时间离散。

## 5. 本地效果：相同对象的外力与材料编辑

![本地 PhysGaussian 适配实验：无外力、半强度外力、原强度外力及较软材料的同步渲染](../outputs/dynamic-gaussian-001/assets/ficus-interventions.gif)
> 图源：本站本地实测，2026-10-13 · 官方 Ficus 预训练高斯与 MPM 求解器，固定相机，gsplat 渲染适配；动画按仿真时间播放，循环末尾重置。

四个画面使用同一初始对象和相机。左起依次为无外力、半强度外力、原强度外力，以及杨氏模量减半。花盆下部受到固定边界约束，枝叶在短暂受力后运动并回弹。因此可以同时看清“操作哪一个对象”“观察哪一时刻”和“编辑哪一个物理参数”。

本地从官方模型筛选出 171,553 个高斯粒子，每组推进 1.2 个仿真时间单位，输出 31 帧。外力仅作用于开始的一个 $10^{-4}$ 时间步；后续运动由内部弹性作用与阻尼继续推进。原配置把同一力值施加到各选中粒子上，它不是整个对象的总力。模型已归一化，本节不把位移或力标成未经标定的真实米、牛顿。

在同一上部枝叶集合上，半外力组的平均横向位移峰值约为原外力组的 50.01%；杨氏模量减半使峰值增大约 32.03%，负向峰值从 0.32 延后到 0.44 个仿真时间单位。这说明外力强度和材料刚度分别改变响应幅度与时间过程。定量口径与步长复核见下方实验附录。

## 6. 从可编辑仿真连接到世界模型

![DreamerV3 原论文完整图1：世界模型学习与想象轨迹上的策略和价值学习](../outputs/dynamic-gaussian-001/assets/dreamerv3-fig01.png)
> 图源：Hafner 等（2025），Fig. 1 · [Nature 原图](https://www.nature.com/articles/s41586-025-08744-2/figures/1) · 完整原图，未改动 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

左半图用实际观测学习状态表示和动作条件下的转移，右半图在模型预测的轨迹上学习策略与价值。DreamerV3 的递归状态空间模型区分确定性记忆 $\mathbf h_t$ 和随机潜变量 $\mathbf z_t$：

$$
\mathbf h_t=f_\theta(\mathbf h_{t-1},\mathbf z_{t-1},\mathbf a_{t-1}),
\qquad
\mathbf z_t\sim q_\theta(\mathbf z_t\mid\mathbf h_t,\mathbf o_t),
$$

$$
\widehat{\mathbf z}_t\sim p_\theta(\mathbf z_t\mid\mathbf h_t),
\qquad
\mathbf a_t\sim\pi_\psi(\mathbf a_t\mid\mathbf h_t,\mathbf z_t).
$$

$\mathbf o_t$ 是观测，$\mathbf a_t$ 是动作；有观测时使用后验修正状态，想象未来时使用预测先验。策略的目标涉及未来奖励，因此除了预测图像，还需要状态转移、奖励和终止信息。

### 对接高斯仿真时需要什么

对本章的对象，一个可检查的环境状态可以包含粒子位置、速度和形变梯度；动作可以是指定区域的力，图像由固定或已知位姿的相机生成。这是本章对仿真接口的整理，不是 DreamerV3 原论文规定的高斯结构：

$$
\mathbf s_t=\{\mathbf x_p^t,\mathbf v_p^t,\mathbf F_p^t,\mathbf C_p^t\}_{p\in\mathcal P_o},
\qquad
\mathbf s_{t+1}=\mathcal T_{\mathrm{phys}}(\mathbf s_t,\mathbf a_t),
\qquad
\mathbf o_t=\mathcal R(\mathcal G(\mathbf s_t),\mathbf K,\mathbf T_{cw}).
$$

例如，要把枝叶位移稳定到目标 $d^\star$，可以定义一个简单的控制任务：

$$
r_t=-(d_t-d^\star)^2-\eta\|\mathbf a_t\|_2^2,
\qquad \eta\geq 0.
$$

奖励同时考虑跟踪误差与动作开销，$\eta$ 的量纲应使两项可比较。先固定这类可测量目标，便能检验三种不同能力：给定动作是否预测出对应运动；执行后是否接收新观测；遇到扰动后是否重新选择动作。闭环评估应比较相同初始条件、目标和扰动下的误差与控制代价。

视觉重建、物理仿真与世界模型因此形成连续接口：重建提供可观测对象，仿真提供可控转移和动作数据，学习模型压缩并预测这些变化，策略通过反馈完成任务。

### 本地训练：从动作轨迹学习状态转移

![动作条件状态预测：留出回合的真实轨迹、连续预测及多时长误差](../outputs/dynamic-world-model-002/assets/action-prediction.png)
> 图源：本站本地实测，2026-10-13 · 左图固定展示首个留出回合；右图汇总全部 4 个留出回合；长度与时间均采用仿真单位。

图左回答“给定新的动作序列，能否预测枝叶怎样运动”；图右回答“连续预测更久以后，误差怎样累积”。预测期间只提供起始状态与后续动作，不用真实中间状态纠正预测。48 步对应 1.92 个仿真时间单位；动作条件模型的枝叶平均位移终点 RMSE 为 **0.00115**，无动作模型为 **0.03888**。这两个模型分别用相同训练回合拟合、相同验证回合选择正则强度，再接受相同的留出评测。

训练数据来自可重复重置的 Ficus 环境。动作只作用于初始高度大于 1.15 的材料粒子集合 $\mathcal P_a$，其成员不随对象运动改变。每个动作在一个观测间隔内保持不变：

$$
u_t\in[-1,1],\qquad
\mathbf f_{p,t}^{\mathrm{ext}}=
\begin{cases}
2m_pu_t\mathbf e_x,&p\in\mathcal P_a,\\
\mathbf 0,&p\notin\mathcal P_a,
\end{cases}.
$$

$$
\Delta t_{\mathrm{obs}}=400\times10^{-4}=0.04.
$$

$\mathbf e_x$ 为横向单位向量，系数 2 的单位为仿真长度除以仿真时间平方。质量乘以加速度得到各粒子的外力；动作在全部 400 个子步内施加。这与上一节仅在起始子步施加的短暂外力是两种不同的实验条件。

为了先验证动作与动力学的关系，将对象按初始高度分为四个固定区域，每个区域观测横向平均位移与平均速度：

$$
\overline d_{b,t}=\frac{1}{|\mathcal P_b|}\sum_{p\in\mathcal P_b}(x_{p,t}-x_{p,0}),\qquad
\overline v_{b,t}=\frac{1}{|\mathcal P_b|}\sum_{p\in\mathcal P_b}v_{p,x,t},
$$

$$
\mathbf y_t=
\begin{bmatrix}
\overline d_{1,t}&\overline v_{1,t}&\overline d_{2,t}&\overline v_{2,t}&
\overline d_{3,t}&\overline v_{3,t}&\overline d_{4,t}&\overline v_{4,t}
\end{bmatrix}^{\mathsf T}\in\mathbb R^8.
$$

这里的状态由仿真器读取。它既保留对象向哪边弯曲，也保留对象接下来向哪边运动。单独知道位移不足以区分“正在弯曲”和“正在回弹”，速度与历史状态为预测提供这些信息。

本地采用带外生输入的自回归模型（autoregressive model with exogenous input，ARX），其训练目标是下一步状态增量；展开后，预测关系等价于

$$
\widehat{\mathbf y}_{t+1}
=\mathbf A_0\mathbf y_t+\mathbf A_1\mathbf y_{t-1}+\mathbf b u_t+\mathbf c,
\qquad
\mathbf A_0,\mathbf A_1\in\mathbb R^{8\times8},\quad
\mathbf b,\mathbf c\in\mathbb R^8.
$$

状态与增量的标准化统计量只从训练集计算；使用岭回归估计系数，不惩罚截距。16 个训练回合、4 个验证回合和 4 个测试回合使用不同动作种子，每个回合有 48 次状态转移。先在验证集比较完整 48 步连续预测，再冻结模型，最后读取测试集。无动作基线令动作输入缺失，时间基线只输出训练集在对应时刻的平均状态。因此，图中的差异反映给定动作对未来预测提供的信息。

### 本地闭环：用预测选择动作，用观测修正后续动作

![同一正向目标与外部扰动下，无动作、开环预测控制与闭环预测控制的真实高斯渲染](../outputs/dynamic-world-model-002/assets/feedback-comparison.gif)
> 图源：本站本地实测，2026-10-13 · 三组使用相同初态、相机、目标与扰动；每帧间隔 0.04 仿真时间，循环末尾重置。

动画把同一对象置于三种控制方式下。无动作组显示外部扰动造成的运动；开环组按模型预测选择动作；闭环组每隔 0.04 读取实际状态，再重新预测与选择动作。三组均由物理仿真生成新的高斯状态，渲染器显示各自真实执行后的结果。

将上部两个区域的平均位移按粒子数加权，得到被控制的枝叶位移 $d_t=\mathbf w^{\mathsf T}\mathbf y_t$。模型预测控制（Model Predictive Control，MPC）在每个时刻求解有限时域的动作序列：

$$
\begin{aligned}
\min_{u_{t:t+H-1}}\quad&
\sum_{j=1}^{H}\left(\frac{\widehat d_{t+j}-d^\star_{t+j}}{d_s}\right)^2 \\
&+\lambda_u\sum_{j=0}^{H-1}u_{t+j}^2 \\
&+\lambda_{\Delta u}\sum_{j=0}^{H-1}(u_{t+j}-u_{t+j-1})^2, \\
\text{约束：}\quad& |u_{t+j}|\leq1,\quad j=0,\ldots,H-1.
\end{aligned}
$$

$H=15$，对应 0.6 个仿真时间单位；$d_s=0.03$ 将位移误差归一化；$\lambda_u=0.03$ 控制动作强度，$\lambda_{\Delta u}=0.02$ 抑制动作突变。第一项使预测位移接近目标，后两项平衡控制开销与动作平滑程度。每次只执行求得序列的第一个动作，然后重新求解。

开环与闭环使用同一个已冻结模型、相同预测时域、目标预览、动作约束和优化权重。两者的区别是重新求解时的状态来源：开环沿用模型自身预测，闭环用最新观测替换预测状态。这样可以直接检查执行反馈是否有用。

![正负两个目标方向下的实际位移：目标、无动作、开环与闭环的匹配比较](../outputs/dynamic-world-model-002/assets/closed-loop-response.png)
> 图源：本站本地实测，2026-10-13 · 两个方向均在仿真时间 1.12 施加反向速度扰动；所有曲线来自实际执行轨迹。

目标在仿真时间 0.32 变为 $+0.025$ 或 $-0.025$。扰动组在时间 1.12 对动作区域额外施加大小为 0.20、方向与目标相反的速度变化；控制器不提前获得扰动信息。图中虚线为目标，竖线为扰动发生时刻。观察扰动之后的回归过程，可将“模型能够预测”和“系统能够利用反馈纠偏”这两项能力区分开。

在这两个受扰方向上，闭环相对开环的全程位移 RMSE 分别降低 **68.80%** 与 **68.49%**；仅看扰动后的窗口，分别降低 **71.91%** 与 **71.62%**。闭环通过追加纠偏动作获得改善，实际动作平方均值也相应增加；这是相同动作边界和惩罚权重下的结果，不是相同实际耗能下的比较。数值与无扰动对照放在实验附录。

## 7. 适用边界与仍需验证的问题

**动态重建的时间覆盖。** 4DGS 以带时间戳的观测为监督；在训练时间范围内得到清晰渲染，不直接证明对新外力、新动作和长时间未来的预测能力。相机位姿和时间同步误差也可能被形变模型吸收。

**物理参数的可辨识性。** PhysGaussian 的材料、密度和边界需要设定或估计。仅从静态照片无法唯一确定刚度与质量；表面高斯也不一定充分描述内部体积。大形变下的局部仿射近似、内部填充和外观搬运均有适用条件。

**本地结果的范围。** 当前完成单对象弹性仿真、按完整回合划分的数据采集、动作条件的紧凑状态预测，以及基于实际状态反馈的 MPC 对照。学习模型预测 8 维状态，图像仍由物理仿真和高斯渲染生成。控制器读取仿真器状态，尚未从图像估计状态，也未训练图像生成模型、DreamerV3 的策略与价值网络。固定对象与材料下的留出动作结果，不能替代跨对象、跨材料或真实环境的泛化验证。

**实验覆盖与数值条件。** 控制比较覆盖正负目标各自有扰动与无扰动共 4 种预先确定的条件，每种执行 3 种控制方式；这些是匹配对照，尚未构成多种子控制统计。材料没有真实标定，4DGS 视频重建也尚未本地训练。归一化长度、浮点协方差条件与时间离散误差仍需保留各自的解释范围。

## 实验附录

实验条件、原始测量、独立复核及后续验收放在[动态高斯对象的受力与材料对照](../experiments/35_dynamic_gaussian_physics_interventions.md)。五组实验包括无外力、半外力、原外力、半刚度，以及保持冲量和时间阻尼一致的半步长复核。正文只保留解释机制的真实效果图。

动作环境、24 个回合、模型选择与 12 组控制对照见[动作条件预测与闭环控制](../experiments/36_dynamic_world_model_control.md)。下一层验证是从图像估计状态，并用新的独立回合检查视觉反馈控制；动态观测重建方向的 4DGS 视频训练则单独评价新视角与留出时间。

## 参考资料

- Wu 等，*4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*，CVPR 2024：[论文 v3](https://arxiv.org/abs/2310.08528v3)、[官方实现](https://github.com/hustvl/4DGaussians)。本章主要参考方法第 4 节与实验第 5 节；参数约束公式采用本文统一记号。
- Xie 等，*PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics*，CVPR 2024：[论文 v3](https://arxiv.org/abs/2311.12198v3)、[固定源码版本](https://github.com/XPandora/PhysGaussian/tree/8339ed6aa2cd5d50e1001a254a3d95aea678a956)。主要参考第 3 节、MPM 附录 A 与材料附录 B。
- Hafner 等，*Mastering diverse control tasks through world models*，Nature 2025：[论文](https://www.nature.com/articles/s41586-025-08744-2)、[DreamerV3 官方实现](https://github.com/danijar/dreamerv3)。图像原样引用；状态与动作公式使用本章记号。
- [论文配图来源与完整图裁切范围](../outputs/dynamic-gaussian-001/figure-provenance.json)。

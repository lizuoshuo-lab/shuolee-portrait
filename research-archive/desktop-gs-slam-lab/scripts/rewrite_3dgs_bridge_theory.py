"""Preserve source metadata and historical appendix; replace two theory bodies."""
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
ARCHIVE = LAB / 'outputs/3dgs-theory-redesign-001'
ARCHIVE.mkdir(parents=True, exist_ok=True)

BRIDGE = r'''# 从传统 3DGS 到 SplaTAM：哪些假设变了

[渲染章](27_3dgs_rendering_foundations.md)回答怎样由高斯生成图像，[训练章](28_3dgs_training_density_control.md)回答怎样由照片学习高斯。接下来保留这套渲染数学，把相机位姿也变成未知量，并让地图随观测逐步增长。

## 1. 从已知相机重建，到未知相机定位

![离线重建中相机固定、地图未知；在线SLAM中相机运动与地图同时未知](../outputs/3dgs-theory-figures/29-01-known-vs-unknown-cameras.png)
> 图源：本站示意，依据 [Kerbl 等（2023）](https://arxiv.org/abs/2308.04079)与 [Keetha 等（2024）](https://arxiv.org/abs/2312.02126)。

传统 3DGS 接收多视图照片以及相机内外参，主要估计地图。记第 $k$ 帧真实图像为 $I_k$，相机世界到相机变换为 $T_k$，地图参数集合为 $\Theta$，渲染算子为 $\mathcal R$：

$$
\Theta^*=\arg\min_{\Theta}\sum_k\mathcal L\bigl(\mathcal R(\Theta,T_k),I_k\bigr).
$$

$T_k$ 是给定条件，$\Theta$ 包含各高斯的位置、形状、不透明度与外观。训练利用多个视角约束同一地图；相机误差仍可能被地图吸收。

SLAM 接收按时间到来的观测，要估计当前位姿 $T_k$ 与地图 $\Theta_k$。相机位姿有三个平移与三个旋转自由度；深度可提供米制尺度，但世界原点与朝向仍需约定。选择首相机作为参考，只是统一坐标，并不保证后续运动准确。

## 2. 继承投影与合成，简化形状与外观

![各向异性高斯具有不同轴尺度与方向；球形高斯各轴尺度相同](../outputs/3dgs-theory-figures/29-02-anisotropic-vs-isotropic.png)
> 图源：本站二维截面示意，依据 [Kerbl 等（2023）](https://arxiv.org/abs/2308.04079)与 [Keetha 等（2024）](https://arxiv.org/abs/2312.02126)。

传统 3DGS 用各向异性高斯表达局部结构：

$$
\Sigma_i=R_i\operatorname{diag}(s_{i1}^2,s_{i2}^2,s_{i3}^2)R_i^T.
$$

三个 $s_{ij}$ 是轴向标准差，$R_i$ 决定主轴方向；$\Sigma_i$ 为 $3\times3$ 协方差，单位为长度平方。SplaTAM 的默认球形表示使用：

$$
\Sigma_i=s_i^2I_3.
$$

球形只需要一个形状尺度，自身旋转不改变形状。相机旋转仍然改变中心投影，因此相机姿态依然需要估计。外观则由传统方向相关的 SH 颜色，简化为每个高斯的 RGB 向量 $\mathbf c_i$。

中心投影、二维核和前后遮挡合成仍然继承渲染章。以相同的可见性权重 $w_i=T_i^{\mathrm{vis}}\alpha_i$，可以合成颜色、覆盖和深度：

$$
\widehat C(p)=\sum_iw_i(p)\mathbf c_i,\qquad
S(p)=\sum_iw_i(p),\qquad
\widehat D(p)=\sum_iw_i(p)z_i.
$$

$p$ 是像素；$T_i^{\mathrm{vis}}$ 是前方留下的透射率，避免与相机变换 $T_k$ 混淆；$S$ 是累计不透明度，无量纲。$z_i$ 是高斯中心在当前相机中的轴向深度，单位米。这里的 $\widehat D$ 是累计深度；若另定义 $\widehat D/S$，得到的是不同的归一化量。覆盖不足时，两者并不相等。

## 3. 用同一个残差，交替估计相机和地图

![跟踪时地图固定、相机移动；建图时相机固定、高斯位置和形状改变](../outputs/3dgs-theory-figures/29-03-alternating-pose-map.png)
> 图源：本站示意，依据 [Keetha 等，SplaTAM（2024）](https://arxiv.org/abs/2312.02126)。

RGB-D 同时提供外观和几何约束。以有效支持像素集合 $\mathcal V_k$ 定义目标：

$$
\mathcal L_k(T,\Theta)=\sum_{p\in\mathcal V_k}
\left(\lambda_C\|\widehat C(p;T,\Theta)-I_k(p)\|_1
+\lambda_D|\widehat D(p;T,\Theta)-D_k(p)|\right).
$$

$D_k$ 是观测深度，单位米；$\lambda_C,\lambda_D$ 控制颜色与深度的相对影响，也承担不同数值尺度的平衡。支持集合需要同时考虑传感器有效性与地图覆盖；像素数量之外，还要看纹理和几何是否提供充分运动约束。

Tracking 固定已有地图，更新当前相机：

$$
T_k^*=\arg\min_T\mathcal L_k(T,\Theta_{k-1}).
$$

Mapping 固定当前与历史窗口的相机，更新高斯：

$$
\Theta_k^*=\arg\min_\Theta\sum_{j\in\mathcal W_k}\mathcal L_j(T_j,\Theta).
$$

$\mathcal W_k$ 是选入建图的观测窗口。图左相机沿梯度寻找更合理的位置，图右地图调整以解释这些固定视角。二者使用相似残差，但改变的是不同未知量。窗口提供历史约束；若需要一起更正历史相机，则还要扩展优化变量。

## 4. 深度决定初始位置，位姿决定它落在哪里

![有效深度沿像素射线产生三维点；错误相机位置会把整组点放到错误世界位置](../outputs/3dgs-theory-figures/29-04-depth-and-pose-feedback.png)
> 图源：本站示意，依据 [Keetha 等，SplaTAM（2024）](https://arxiv.org/abs/2312.02126)。

给定内参矩阵 $K$、像素 $(u,v)$ 和轴向深度 $z$，相机坐标中的点为：

$$
\mathbf p_c=zK^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix},\qquad
\boldsymbol\mu_w=R_{wc}\mathbf p_c+\mathbf c_w.
$$

$\mathbf c_w$ 为相机在世界中的中心；$R_{wc}$ 把相机向量旋转到世界。深度和位置以米计，焦距与像素坐标用像素单位。这里 $z$ 不是相机到点的欧氏距离。

低覆盖区域意味着已有地图尚不足以解释观测，有效深度可以提供新高斯中心。传统 clone/split 主要重新分配已有表示容量，深度插点则引入当前观测到的新几何位置。

图右显示位姿偏差怎样传给新点。新点又参与下一帧定位，于是地图与相机之间形成反馈。深度同时参与初始化、插点、有效支持与残差；仅减小深度残差权重，不会自动移除其他深度信息。

## 5. 好看的图像与准确的运动，分别评价

![左侧图像相似度评价外观，右侧参考与估计轨迹之间的偏差评价运动](../outputs/3dgs-theory-figures/29-05-image-vs-trajectory-evaluation.png)
> 图源：本站示意，依据 [Kerbl 等（2023）](https://arxiv.org/abs/2308.04079)与 [Keetha 等（2024）](https://arxiv.org/abs/2312.02126)。

传统重建重点检查独立新视角的图像质量。SLAM 还必须比较运动：设相机中心估计为 $\widehat{\mathbf c}_k$，独立参考为 $\mathbf c_k$，声明的坐标对齐为 $A$，位置 ATE 可写成：

$$
\mathrm{ATE}_{\mathrm{RMSE}}=
\sqrt{\frac1n\sum_{k=1}^n\|A\widehat{\mathbf c}_k-\mathbf c_k\|_2^2}.
$$

其单位为米。米制 RGB-D 通常保持尺度固定的刚体对齐；时间配对、物理参考点和失败覆盖也影响结论。相邻或指定间隔的相对运动误差，则用于观察局部漂移。

地图和错误位姿可能共同解释观测，因此图像相似度不能替代轨迹准确度。沿[每帧跟踪与建图](24_splatam_official_rgbd_system.md) → [有效像素与跟踪损失](25_splatam_tracking_loss_support.md) → [关键帧与历史约束](14_keyframes_map_window_history.md)继续阅读，随后进入[轨迹评价](08_trajectory_evaluation.md)、[参考与恢复](17_rejection_reference_recovery.md)、[地图坐标更正](18_gaussian_map_pose_correction.md)。
'''

CAPACITY = r'''# 高斯数量、初始化与密度控制

本页补充一个核心问题：高斯数量、初始位置与训练中的结构变化，分别解决什么？完整基础可沿[渲染](27_3dgs_rendering_foundations.md) → [训练](28_3dgs_training_density_control.md)阅读。

## 1. 数量提供容量，分布决定容量用在哪里

![相同20个高斯，随机分配在空间与沿结构分布，会形成不同覆盖](../outputs/3dgs-theory-figures/05-01-capacity-vs-coverage.png)
> 图源：本站示意，依据 [Kerbl 等，3DGS（2023）](https://arxiv.org/abs/2308.04079)。

地图是 $N$ 个基元的集合：

$$
\Theta=\{\boldsymbol\mu_i,\Sigma_i,o_i,\mathbf f_i\}_{i=1}^N.
$$

$\boldsymbol\mu_i$ 决定位置，$\Sigma_i$ 决定覆盖形状，$o_i$ 决定不透明度，$\mathbf f_i$ 决定外观。增加 $N$ 提供更多独立表达能力，但没有规定这些能力应该分配到哪里。

图中两组都使用20个基元。左侧许多基元偏离目标结构；右侧把相同数量安排到结构附近。高斯总数相同，局部覆盖和可用梯度仍不同。小基元有利于分开细节，大基元容易建立连续覆盖；尺寸和位置必须一起判断。

## 2. 初始化决定从哪一种几何先验开始

![随机空间点、SfM稀疏几何、深度反投影分别提供不同强度和来源的几何先验](../outputs/3dgs-theory-figures/05-02-initialization-priors.png)
> 图源：本站示意，依据 [Kerbl 等（2023）](https://arxiv.org/abs/2308.04079)与 [Keetha 等（2024）](https://arxiv.org/abs/2312.02126)。

随机点只有预设空间范围的先验，训练需要把位置和覆盖逐渐分配到结构上。SfM 点来自多视图对应与三角化，可把中心放到已观察到的局部结构；初始尺度与邻域间距保持同一量级。

RGB-D 则由像素直接反投影：

$$
\mathbf p_c=zK^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix},\qquad
\boldsymbol\mu_w=R_{wc}\mathbf p_c+\mathbf c_w.
$$

$z$ 是相机 Z 轴深度；$K$ 是像素内参；$R_{wc},\mathbf c_w$ 是相机到世界的旋转与相机中心。深度初始化让中心接近观测表面，也把深度噪声、配准和位姿误差一起带入地图。

由针孔几何，一个像素对应的世界长度量级随深度增加：

$$
s_0\propto\frac zf.
$$

$f$ 为焦距，以像素计；$s_0$ 与 $z$ 用相同长度单位。它说明同样的像素覆盖在远处对应更大的空间范围；比例系数和具体足迹仍需由模型选择确定。

## 3. 参数约束改变了可行范围，也改变梯度

![尺度、旋转、位置、不透明度与方向颜色共同决定高斯的形状和像素贡献](../outputs/3dgs-theory-figures/28-02-parameter-constraints.png)
> 图源：本站示意，依据 [Kerbl 等，3DGS（2023）](https://arxiv.org/abs/2308.04079)。

协方差需要合法形状，不透明度需要落在0与1之间。用无约束变量 $\ell,a$ 和非零四元数 $\mathbf q_0$ 可以构造：

$$
s=e^\ell,\qquad o=\frac1{1+e^{-a}},\qquad
\mathbf q=\frac{\mathbf q_0}{\|\mathbf q_0\|_2}.
$$

$s$ 是轴向标准差，$\ell$ 是其对数参数；$o$ 是实际不透明度，$a$ 是对应的 logit。颜色、opacity与logit均按统一数值尺度使用；尺度取对数时，以固定长度单位下的数值表示。归一化四元数生成合法旋转矩阵 $R$。

变换的导数为：

$$
\frac{\partial s}{\partial\ell}=s,\qquad
\frac{\partial o}{\partial a}=o(1-o).
$$

相同对数步长对应相对尺度变化；不透明度靠近0或1时，logit梯度会减弱。很小的足迹可能得不到有效像素约束，过早变得不透明则可能遮住后面的结构。因此参数不仅要合法，还要处于有利于观察和优化的范围。

## 4. 增密重新分配容量，剪枝删除低贡献基元

![高屏幕梯度结合大小选择克隆或拆分，之后继续优化新基元](../outputs/3dgs-theory-figures/28-07-clone-split.png)
> 图源：本站示意，依据 [Kerbl 等，3DGS（2023）](https://arxiv.org/abs/2308.04079)。

密度控制先观察图像目标对屏幕位置的敏感度。对高斯 $i$，在 $n_i$ 次有效观察中取梯度模的平均：

$$
\bar g_i=\frac1{n_i}\sum_{k\in\mathcal V_i}
\left\|\frac{\partial\mathcal L_k}{\partial\mathbf m_i}\right\|_2.
$$

$\mathbf m_i$ 是二维投影中心；$\mathcal V_i$ 是被统计的视角集合。先求模再平均，不同视角的梯度方向不会彼此抵消。该量衡量目标敏感性，并未直接测量点到真实表面的距离。

较小但高梯度的基元适合克隆，增加局部独立容量；较大且高梯度的基元适合拆分，换成多个更小的基元。两者改变表示结构，之后仍要优化位置、形状和外观。子基元并不自动保持父基元的总颜色或体积。

低不透明度基元可以剪枝；暂时压低不透明度则使原先被遮挡的结构重新竞争。新增、删除与阈值选择属于离散结构变化，连续渲染的可微性不等于点数变化也处处可微。

## 5. 用独立视角判断容量是否真正有用

![训练视角用于拟合，独立新视角用于检查图像泛化](../outputs/3dgs-theory-figures/28-09-training-novel-view-evaluation.png)
> 图源：本站示意，依据 [Kerbl 等，3DGS（2023）](https://arxiv.org/abs/2308.04079)。

比较容量时，先统一训练观测、初始化信息、图像范围和优化预算，再检查独立视角。对于归一化到0到1的RGB图像：

$$
\mathrm{MSE}=\frac1{3HW}\sum_{p,c}
\bigl(\widehat I_{p,c}-I_{p,c}\bigr)^2,\qquad
\mathrm{PSNR}=-10\log_{10}(\mathrm{MSE}).
$$

$H,W$ 是图像尺寸，$c$ 遍历RGB三个通道。PSNR反映像素拟合；局部结构相似度和细节观察提供补充。用来选择容量的留出视角属于验证集，最终测试应另外保留。

更多高斯可能改善细节，也可能拟合错误几何或观测噪声。图像泛化和独立几何准确度要分别评价；同时报告最终数量与计算代价，才能判断新增容量带来了什么。
'''

def replace(name, body, keep_appendix=False):
    path = LAB/'knowledge'/name
    raw = path.read_bytes()
    (ARCHIVE/(name[:2]+'-implementation-reference.md')).write_bytes(raw)
    text=raw.decode('utf-8-sig').replace('\r\n','\n')
    end=text.index('\n---\n',4)+5
    metadata=text[:end]
    appendix=''
    if keep_appendix:
        appendix='\n\n## 实验附录\n'+text.split('\n## 实验附录\n',1)[1]
    path.write_text(metadata+'\n'+body.strip()+'\n'+appendix,encoding='utf-8')

replace('29_3dgs_to_splatam_progression.md',BRIDGE)
replace('05_capacity_initialization_density.md',CAPACITY,True)
print('Rewrote theory chapters 29 and 05; source metadata and historical appendix preserved.')

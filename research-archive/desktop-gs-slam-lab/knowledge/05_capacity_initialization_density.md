---
id: "gslab-k05"
slug: "gaussian-capacity-initialization-density"
title: "05 高斯数量、初始化与密度控制"
order: 5
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/examples/image_fitting.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/gaussian_model.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/slam_external.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
---

# 高斯数量、初始化与密度控制

本页补充一个核心问题：高斯数量、初始位置与训练中的结构变化，分别解决什么？完整基础可沿[渲染](27_3dgs_rendering_foundations.md) → [训练](28_3dgs_training_density_control.md)阅读。

## 1. 高斯数量与空间分布

![原论文展示优化后的各向异性高斯如何沿场景结构分布](../outputs/3dgs-paper-course-001/figures/paper-fig-03.png)
> 图源：[Kerbl 等（2023），Fig. 3](https://arxiv.org/abs/2308.04079) · 原图裁剪 · CC BY 4.0。

地图由 $N$ 个高斯基元（每个基元是一个三维高斯）组成：

$$
\Theta=\{\boldsymbol\mu_i,\Sigma_i,o_i,\mathbf f_i\}_{i=1}^N.
$$

$\boldsymbol\mu_i$ 决定位置，$\Sigma_i$ 决定覆盖形状，$o_i$ 决定不透明度，$\mathbf f_i$ 决定外观。增加 $N$ 提供更多独立表达能力，但没有规定这些能力应该分配到哪里。

原论文 Fig. 3 将渲染图像与高斯表示并列，并通过缩小高斯展示其形状与分布。细长或扁平的高斯沿局部结构分配，而不是均匀铺满空间。这个图说明位置、尺度和方向共同决定覆盖；它不是固定数量的受控实验，不能据此推断某个数量一定最优。

## 2. 几何先验与参数初始化

![原论文随机初始化与SfM初始化的真实渲染对照](../outputs/3dgs-paper-course-001/figures/paper-fig-07.png)
> 图源：[Kerbl 等（2023），Fig. 7](https://arxiv.org/abs/2308.04079) · 原图裁剪 · CC BY 4.0。

随机初始化依赖预设的空间采样分布。运动恢复结构（Structure from Motion，SfM）通过多视图特征对应与三角化，估计相机参数和稀疏三维点；这些点提供高斯中心的几何初值，主轴标准差可以按照邻域点间距初始化。

RGB-D 指彩色图像与对应的深度图像。给定无畸变针孔相机内参，可将像素反投影到三维：

$$
K=\begin{bmatrix}f_x&0&c_x\\
0&f_y&c_y\\
0&0&1\end{bmatrix},\qquad
\mathbf p_c=zK^{-1}\begin{bmatrix}u\\
v\\
1\end{bmatrix}
=\begin{bmatrix}z(u-c_x)/f_x\\
z(v-c_y)/f_y\\
z\end{bmatrix}.
$$

$$
\boldsymbol\mu_w=R_{wc}\mathbf p_c+\mathbf c_w.
$$

$z$ 是沿相机 $Z$ 轴测量的深度，单位为长度；$K$ 是相机内参矩阵；$f_x,f_y$ 为焦距，$c_x,c_y$ 为主点坐标，与像素坐标 $u,v$ 一样以像素计。$R_{wc}$ 把相机向量旋转到世界坐标，$\mathbf c_w$ 是世界坐标中的相机中心。主点是光轴与像平面的交点；[光轴的图解与定义](27_3dgs_rendering_foundations.md#第三步-相机坐标变换与透视投影)见渲染章。

对相机前方的有效深度 $z>0$，$z$ 一般不等于相机中心到三维点的欧氏距离 $\rho$：

$$
\rho=\sqrt{\mathbf p_c^\top\mathbf p_c}
=z\sqrt{1+\left(\frac{u-c_x}{f_x}\right)^2+\left(\frac{v-c_y}{f_y}\right)^2}.
$$

光轴上 $u=c_x,v=c_y$，因此 $\rho=z$。深度初始化同时受到深度噪声、图像配准误差和相机位姿误差的影响。

由针孔几何，一个像素对应的世界长度量级随深度增加：

$$
s_0\propto\frac{z\,\Delta u}{f}.
$$

$f$ 为焦距，以像素计；$\Delta u$ 为预期的初始投影宽度，以像素计，例如一个像素宽度。$s_0$ 是初始标准差，与 $z$ 用相同长度单位。该式表示相同投影宽度在远处对应更大的三维范围；比例系数依赖具体初始化方法。

## 3. 参数化、协方差与梯度

![原论文从高斯参数到图像再到图像损失的优化流程](../outputs/3dgs-paper-course-001/figures/paper-fig-02.png)
> 图源：[Kerbl 等（2023），Fig. 2](https://arxiv.org/abs/2308.04079) · 原图裁剪 · CC BY 4.0。

协方差需要合法形状，不透明度需要落在0与1之间。用无约束变量 $\ell,a$ 和非零四元数 $\mathbf q_0$ 可以构造：

$$
s=s_{\mathrm{ref}}e^\ell,\qquad o=\frac1{1+e^{-a}},\qquad
\mathbf q=\frac{\mathbf q_0}{\|\mathbf q_0\|_2}.
$$

$s$ 是主轴标准差，$s_{\mathrm{ref}}$ 为固定参考长度，$\ell=\log(s/s_{\mathrm{ref}})$ 是无量纲的对数参数；$o$ 是不透明度，$a$ 是对应的对数几率（logit）：

$$
a=\log\frac{o}{1-o},\qquad 0<o<1.
$$

颜色、不透明度与对数几率均无量纲，取值范围不同。归一化四元数对应旋转矩阵 $R$；其分量顺序和完整矩阵见[三维高斯与旋转矩阵](27_3dgs_rendering_foundations.md#第二步-各向异性三维高斯与协方差矩阵)。三个主轴标准差 $s_1,s_2,s_3$ 构成协方差：

$$
\Sigma=R\begin{bmatrix}s_1^2&0&0\\
0&s_2^2&0\\
0&0&s_3^2\end{bmatrix}R^\top.
$$

$\Sigma$ 为 $3\times3$ 正定矩阵，单位为长度平方；$R$ 无量纲。

变换的导数为：

$$
\frac{\partial s}{\partial\ell}=s,\qquad
\frac{\partial o}{\partial a}=o(1-o).
$$

相同对数参数变化对应相同的标准差相对变化；不透明度接近 0 或 1 时，对数几率的梯度会减弱。投影覆盖范围过小可能缺少有效像素约束；过早增加不透明度则可能遮挡后方高斯。这些因素影响参数优化的条件。

## 4. 自适应密度控制与剪枝

![原论文上排复制小高斯、下排分裂大高斯，并在之后继续优化](../outputs/3dgs-paper-course-001/figures/paper-fig-04.png)
> 图源：[Kerbl 等（2023），Fig. 4](https://arxiv.org/abs/2308.04079) · 原图裁剪 · CC BY 4.0。

自适应密度控制（adaptive density control）根据图像损失对高斯投影位置的敏感性选择候选。记第 $k$ 个视角下的投影中心为 $\mathbf m_{ik}=(u_{ik},v_{ik})^\top$，对高斯 $i$ 在 $n_i$ 次有效观察中计算梯度欧氏范数的平均：

$$
\bar g_i=\frac1{n_i}\sum_{k\in\mathcal V_i}
\sqrt{\left(\frac{\partial\mathcal L_k}{\partial u_{ik}}\right)^2
+\left(\frac{\partial\mathcal L_k}{\partial v_{ik}}\right)^2}.
$$

$\mathcal V_i$ 是被统计的视角集合，$n_i=|\mathcal V_i|>0$。先计算各次观察的梯度欧氏范数，再求平均，可避免不同视角的梯度向量相互抵消。该量衡量损失敏感性，没有直接测量到真实表面的距离。

高斯复制（cloning）用于较小且高梯度的候选，新增基元随后独立优化；高斯分裂（splitting）用于较大且高梯度的候选，用多个较小子高斯替换父高斯。对一个父高斯，数量变化为：

$$
N_{\mathrm{new}}=\begin{cases}
N+1,&\text{复制并新增一个高斯},\\
N+M-1,&\text{用 }M\text{ 个子高斯替换父高斯}.
\end{cases}
$$

$M\ge2$ 是分裂子高斯数量。两种操作均改变表示结构，后续仍需优化中心、协方差、不透明度与颜色，不保证合成颜色或高斯体积守恒。

高斯剪枝（pruning）删除符合剪枝条件的基元。例如仅使用不透明度阈值 $\tau_o\in(0,1)$ 时，保留集合为：

$$
\mathcal I_{\mathrm{keep}}=\{i:o_i\ge\tau_o\}.
$$

不透明度重置可用上限 $\varepsilon_o\in(0,1)$ 表示：

$$
o_i^{\mathrm{new}}=\begin{cases}o_i,&o_i\le\varepsilon_o,\\
\varepsilon_o,&o_i>\varepsilon_o.\end{cases}
$$

降低前方不透明度可能增加后方高斯的透射权重，但不保证恢复正确几何。新增、删除与阈值选择是离散结构变化；连续渲染可微不等于高斯数量变化处处可微。

## 5. 新视角图像质量评价

![原论文不同场景中的新视角渲染与基准图像对照](../outputs/3dgs-paper-course-001/figures/paper-fig-05.png)
> 图源：[Kerbl 等（2023），Fig. 5](https://arxiv.org/abs/2308.04079) · 原图裁剪 · CC BY 4.0。

比较容量时，先统一训练观测、初始化信息、图像范围和优化预算，再检查独立视角。对于归一化到0到1的RGB图像：

$$
\mathrm{MSE}=\frac1{3HW}\sum_{p,c}
\bigl(\widehat I_{p,c}-I_{p,c}\bigr)^2,\qquad
\mathrm{PSNR}=-10\log_{10}(\mathrm{MSE}).
$$

$H,W$ 是图像尺寸，$c$ 遍历 RGB 三个通道。MSE 为均方误差（mean squared error），PSNR 为峰值信噪比（peak signal-to-noise ratio），单位为分贝。上式的 PSNR 假定图像最大值为 1；一般形式为 $10\log_{10}(I_{\max}^2/\mathrm{MSE})$。结构相似性指标可补充像素误差。用于选择高斯数量的留出视角属于验证集，最终测试应另外保留。

更多高斯可能改善细节，也可能拟合错误几何或观测噪声。图像泛化和独立几何准确度要分别评价；同时报告最终数量与计算代价，才能判断新增容量带来了什么。


## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

当前[多视图教学脚本](../scripts/train_tiny_gs.py)使用**固定数量、随机初始化、已知相机**：优化中心、三轴尺度、旋转、RGB 与不透明度，不执行 densification 或 pruning，也不估计相机位姿。本文的数量对照来自这条管线；官方 3DGS 与 SplaTAM 的密度控制部分属于已核对源码的机制说明，尚未在本实验室完成运行验证。

### 本地实测：4000 与 8000 个固定高斯

本次用同一教学训练器、Tiny NeRF Lego 100×100 图像、100 个训练视角、6 个验证视角、1500 次优化，比较 seeds 0、1、2。每次先计算 6 个验证视角 PSNR 的算术平均，再对三次 seed 结果计算均值和**样本标准差**。

| 固定高斯数 | 验证 PSNR：三次均值 ± 样本标准差 | 结果含义 |
| --- | --- | --- |
| 4000 | 26.844 ± 0.058 dB | 当前配置下的较低容量基线 |
| 8000 | 27.410 ± 0.078 dB | 当前三个 seed 中均取得更高验证 PSNR |

按相同 seed 标签计算，8000 相比 4000 分别增加 0.634、0.498、0.567 dB。此处只能说，在这个小场景和预算下，提高固定数量改善了观测到的验证图像质量。三个 seed 不构成跨场景普适性或统计显著性的证据；同 seed 也不是逐高斯严格配对。

8000 / seed 0 复用了此前已完成的记录，其他五次为本轮运行。没有启用密度控制、深度监督、SH 或位姿优化；本对照不验证 densification 的收益，也不构成 SLAM 测试。耗时很短且有历史记录复用，本表不据此宣称稳定的速度差异。

完整运行名、配置、各次指标、复用标记及限制见[高斯数量对照指标](../outputs/gaussian-count-study-001/metrics.json)。这些指标是实验事实，前述 clone / split / prune 则是源码核对得到的机制解释，两者的证据来源不同。

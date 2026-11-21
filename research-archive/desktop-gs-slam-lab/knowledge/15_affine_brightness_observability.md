---
id: "gslab-k15"
slug: "affine-brightness-pose-observability"
title: "15 仿射亮度与位姿：八维可观测性"
order: 15
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"
  - "https://arxiv.org/abs/1607.02565"
  - "https://raw.githubusercontent.com/JakobEngel/dso/7b0c99f01d238f801c625beaff90240bcb007198/src/util/NumType.h"
---

# 仿射亮度与位姿：新增参数也会新增歧义

光度里程计通常要求同一表面在两张图像中具有近似一致的强度。整体变亮会破坏这个假设：优化器可能尝试移动相机，让错误位置的纹理解释强度变化。给模型增加相对亮度增益（gain）和偏置（bias），可以表达一类整体亮度变化；但新增变量也可能吸收本该由运动解释的信息，使原先可解的位姿问题退化。

本页以[八维教学实现](../scripts/rgbd_affine_photometric_odometry.py)说明位姿与相对亮度增益与偏置 的联合约束。沿用[光度与几何里程计](13_photometric_geometric_odometry.md)的取样和几何定义；它是成对图像模型，不能直接等同于 DSO 的整套辐射、曝光与优化管线。

## 写清亮度变换的方向

设 $\mathbf T$ 把当前相机点变换到参考相机，参考帧通常是上一帧。当前像素强度固定，参考强度在连续投影位置采样：

$$
\mathbf p=\mathbf R_{\mathrm{ref}\leftarrow\mathrm{cur}}\mathbf p_{\mathrm{cur}}
+\mathbf t_{\mathrm{ref}\leftarrow\mathrm{cur}},
\qquad
s=I_{\mathrm{ref}}(\pi(\mathbf p)),\qquad
g=e^a>0,
$$

$$
r_I=g\,s+b-I_{\mathrm{cur}},
\qquad
r_g=\mathbf n^\mathsf T(\mathbf p-\mathbf q),
\qquad
\delta\boldsymbol\theta=
\begin{bmatrix}
\delta\mathbf v\\
\delta\boldsymbol\omega\\
\delta a\\
\delta b
\end{bmatrix}\in\mathbb R^8.
$$

$$
\mathbf T_{\mathrm{new}}=\exp(\delta\boldsymbol\xi^\wedge)\mathbf T,
\qquad
a_{\mathrm{new}}=a+\delta a,\qquad
b_{\mathrm{new}}=b+\delta b,
\qquad
\delta\boldsymbol\xi=
\begin{bmatrix}
\delta\mathbf v\\
\delta\boldsymbol\omega
\end{bmatrix}.
$$

$\mathbf p,\mathbf q$ 均在参考相机坐标系，$\mathbf n$ 为其单位法向；$r_g$ 与平移以米计，旋转以弧度计。$r_I,s,b$ 以归一化强度计，$g,a$ 无量纲。

因此 $g$、$b$ 表示“怎样把参考图像的强度变成当前图像强度”。若当前图像统一加 0.2，正确的偏移方向是 $b=+0.2$。把图像顺序互换后仍沿用相同参数，会改变残差含义。每对配准以 $\mathbf T=\mathbf I_4,\ a=0,\ b=0$ 开始；构造真值只用于生成输入和事后核对，不用于选择候选。

$a$ 是无量纲对数增益，$b$ 的单位与归一化灰度相同。数学上的 $e^a>0$ 保证增益为正，没有限制其大小；它也没有把预测强度限制到 $[0,1]$。本实现不估计曝光时间，不反解相机响应或 sRGB，也不完成暗角标定。它拟合的是两幅图像之间的相对仿射关系，不能把 $g$ 直接当作真实曝光倍率。

## 八维残差雅可比

记 $\mathbf h=\mathbf J_\pi^\mathsf T\nabla I_{\mathrm{ref}}$。采用左扰动时，点的导数为 $\begin{bmatrix}\mathbf I_3&-[\mathbf p]_\times\end{bmatrix}$，$\mathbf p$ 必须是已经变换到参考相机坐标系的点。光度与几何雅可比分别为：

$$
\mathbf h=\mathbf J_\pi^\mathsf T\nabla I_{\mathrm{ref}},
\qquad
\mathbf J_I=
\begin{bmatrix}
g\mathbf h^\mathsf T&
g(\mathbf p\times\mathbf h)^\mathsf T&
g\,s&1
\end{bmatrix},
$$

$$
\mathbf J_g=
\begin{bmatrix}
\mathbf n^\mathsf T&
(\mathbf p\times\mathbf n)^\mathsf T&0&0
\end{bmatrix},
\qquad
\frac{\partial r_I}{\partial a}=e^a s,\qquad
\frac{\partial r_I}{\partial b}=1.
$$

$\nabla I_{\mathrm{ref}}\in\mathbb R^2$ 为连续取样强度的列梯度，$\mathbf J_\pi\in\mathbb R^{2\times3}$ 为投影雅可比；$\mathbf h\in\mathbb R^3$ 是其三维坐标梯度。两个残差雅可比均为 $1\times8$。

位姿对应的前六列也必须乘 $g$，而不是只新增最后两列。几何项不依赖亮度参数，最后两列应严格为零。双线性梯度来自实际取样的同一组四角；更换成 Sobel 近似后，不能再把它叫作同一插值函数的精确导数。

验证时选非单位位姿、非零 $a$/$b$，固定观测和对应，在同一个插值单元内检查八个方向的中心差分。八列具有不同单位，整体最大导数差不是“多少米的轨迹误差”，也不是协方差。局部公式一致与完整算法在真实观测中有效需要分别验证。

## 归一化与鲁棒权重没有消除退化

[固定配置](../outputs/rgbd-affine-photometric-cpu-controls-001/configuration.json)沿用几何尺度 $\sigma_g=0.01\,\mathrm m$、强度尺度 $\sigma_I=0.1$、光度系数 1 和归一化 Huber 分界 $\delta=1$。同一组 $N$ 个联合对应上的目标为：

$$
s_{g,i}=\frac{r_{g,i}}{\sigma_g},\qquad
s_{I,i}=\frac{r_{I,i}}{\sigma_I},\qquad
\mathcal L=
\frac1N\sum_{i=1}^{N}\rho(s_{g,i})
+\frac1N\sum_{i=1}^{N}\rho(s_{I,i}),
$$

$$
\rho(s)=
\begin{cases}
\frac12s^2,&|s|\leq1,\\
|s|-\frac12,&|s|>1,
\end{cases}
\qquad
w(s)=
\begin{cases}
1,&|s|\leq1,\\
1/|s|,&|s|>1.
\end{cases}
$$

$$
A_{i,:}=\frac{\sqrt{w_{g,i}}}{\sigma_g}\mathbf J_{g,i},\qquad
A_{N+i,:}=\frac{\sqrt{w_{I,i}}}{\sigma_I}\mathbf J_{I,i},
\qquad
d_i=-\sqrt{w_{g,i}}s_{g,i},\qquad
d_{N+i}=-\sqrt{w_{I,i}}s_{I,i},
$$

$$
\delta\boldsymbol\theta^*
=\underset{\delta\boldsymbol\theta}{\operatorname{argmin}}
\|\mathbf A\delta\boldsymbol\theta-\mathbf d\|_2^2.
$$

$i=1,\ldots,N$，$\mathbf A\in\mathbb R^{2N\times8}$，$\mathbf d\in\mathbb R^{2N}$；$w_{g,i}=w(s_{g,i})$、$w_{I,i}=w(s_{I,i})$ 在本次线性化中固定。归一化尺度 $\sigma_g=0.01\,\mathrm m$、$\sigma_I=0.1$，对应光度系数为 $1$。

这些尺度是工程选择，没有从观测噪声估计得到。代码对 $\mathbf A$ 的列归一化，用 奇异值分解（Singular Value Decomposition，SVD） 判断数值秩，再用 直接最小二乘求解 求八维增量；条件数记录的是归一化系统最大、最小奇异值比的平方 $\kappa_2(\overline{\mathbf A}^{\mathsf T}\overline{\mathbf A})=(\nu_{\max}/\nu_{\min})^2$，其中 $\nu$ 为 $\overline{\mathbf A}$ 的奇异值，要求 $\nu_{\min}>0$，不是原始 Hessian 的条件数或位姿协方差。

当前实现没有亮度先验、gain/bias 截断或阻尼，要求联合 秩 达到 **8**，不能沿用六维问题的 秩 门槛。原有平移、旋转单步上限会按同一个因子缩放全部八维方向，线搜索也共同缩放八维；这不等于给 gain/bias 设置独立边界。非有限仿射状态会使试探被拒绝。

每个试探固定对应集合与分母，要求投影仍在合法双线性域内，相关四角支持均有效。增加亮度变量没有取消遮挡、深度边缘和支持限制，也没有让跨单元或动态对应变成处处可微的问题。

## 常量与线性纹理的局部退化

理想单平面的 point-to-plane 几何约束只有三个独立运动方向。常量参考强度为 $c$ 时，光度残差只增加 $g\,c+b$ 这一个独立组合：gain 增加可以被 bias 减少抵消，平面内运动也不会改变强度。这种理想模型的联合秩最多为 4，无法确定八个变量。

线性纹理可写作 $I(X,Y)=c+\alpha X+\beta Y$。沿平面移动产生的强度变化包含常量偏移，可能与 $b$ 的变化互相抵消；存在图像梯度仍可能联合退化。八维问题不能沿用六维 秩 门槛；真位姿恰好等于初值时，零位置误差也不能证明解唯一。

新增变量可能使残差更容易降低，同时让运动更难辨认。若只报告较低的光度 loss，不检查秩、谱、亮度参数及位姿，就可能把参数之间的补偿当成正确跟踪。


## 亮度模型的适用范围

相对 gain/bias 能表示一类全图强度变化，不能表达所有阴影、反射、饱和、运动模糊或非线性响应。构造数据中恢复亮度参数，只验证那个模型与实现；真实线搜索拒绝可能来自对应、几何、纹理或初值，不能未经控制归因于曝光。

## 满秩与低残差仍不是定位保证

满秩只说明当前线性化下没有精确零方向，病态谱、误对应和错误局部极小值仍可能存在。新增亮度自由度还能降低光度残差，却吸收运动信息。一起观察归一化谱、亮度参数、接受原因与独立轨迹误差；ATE、RPE 和原参考系误差可能呈不同变化。

[DSO 论文](https://arxiv.org/abs/1607.02565)和[Open3D RGB-D odometry](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html)提供背景；本页的成对相对参数不包含其完整实现。

## 亮度自由度与参考恢复是两个问题

亮度模型解释强度变化，参考是否可用还取决于运动历史、重叠、对应和接受规则。参考长期停留时，即使少拒绝一帧，也不代表恢复了连续定位。见[拒绝与恢复](17_rejection_reference_recovery.md)。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

导数对照见[八维差分记录](../outputs/rgbd-affine-photometric-cpu-controls-001/differential_checks.json)、[独立数学复核](../outputs/rgbd-affine-photometric-cpu-controls-001/root_independent_math_audit.json)及[复核脚本](../scripts/audit_rgbd_affine_math.py)。

本页对应 [八维 CPU 教学实现](../scripts/rgbd_affine_photometric_odometry.py)及[合成控制脚本](../scripts/check_rgbd_affine_photometric_odometry.py)，源码 SHA256 分别以 `15c7f160`、`dee2c5a2` 开头。它复用[知识 13](13_photometric_geometric_odometry.md)中冻结的对应筛选、双线性取样和几何模块。四个合成案例用于隔离数学实现；另有两组已完成的真实 TUM CPU 对照，按各自的指标和拒绝链路解释。仿射模型实验没有调用 GPU。

### 四个实际保存的合成案例

新的双线性纹理为：

```text
I_world(X,Y) = 0.35 + 0.07X + 0.09Y + 0.08XY
```

参考相机正对 `z=2.2 m` 平面，纹理刻意能被参考图像的双线性插值精确表示；三层图像是按构造公式生成的理想输入。这是用来隔离数学实现的有利 toy，没有真实噪声、复杂遮挡、响应非线性或高频纹理失配。

| 合成案例 | 最终联合 rank | 实际行为 |
|---|---:|---|
| 非零已知位姿，gain 1.12、bias 0.045 | 8 | 接受；平移误差 `5.624e−10 m`，旋转误差 `5.298e−9°` |
| identity 位姿，当前强度统一加 0.2 | 8 | 接受；gain 约 1，bias 约 0.2000000000000003 |
| 常量纹理，gain 1.12、bias 0.045 | 4 | 拒绝联合不可观测与病态系统 |
| 线性纹理，gain 1.12、bias 0.045 | 6 | 拒绝联合不可观测与病态系统 |

第一例的 gain、bias 绝对误差分别约为 `2.936e−11`、`3.334e−11`。第二例的真运动本来就是 identity；它证明统一偏移可被正确符号的 bias 解释，没有证明该例恢复了非零运动。极小误差来自精确匹配的构造问题，不能外推为真实相机精度。

[四案例汇总](../outputs/rgbd-affine-photometric-cpu-controls-001/metrics.json)保存了接受状态、失败标记、事后误差和各输入清单的 hash。两个成功案例的[非零位姿结果](../outputs/rgbd-affine-photometric-cpu-controls-001/bilinear-known-pose-gain-bias/result.json)与[统一偏移结果](../outputs/rgbd-affine-photometric-cpu-controls-001/bilinear-brightness-plus-point2/result.json)，以及[常量退化结果](../outputs/rgbd-affine-photometric-cpu-controls-001/constant-texture-rank-rejection/result.json)、[线性耦合结果](../outputs/rgbd-affine-photometric-cpu-controls-001/linear-texture-nuisance-coupling/result.json)，都保留了迭代与试探，没有删除失败控制。

四份输入清单分别记录[非零位姿](../outputs/rgbd-affine-photometric-cpu-controls-001/bilinear-known-pose-gain-bias/input_manifest.json)、[统一偏移](../outputs/rgbd-affine-photometric-cpu-controls-001/bilinear-brightness-plus-point2/input_manifest.json)、[常量纹理](../outputs/rgbd-affine-photometric-cpu-controls-001/constant-texture-rank-rejection/input_manifest.json)和[线性纹理](../outputs/rgbd-affine-photometric-cpu-controls-001/linear-texture-nuisance-coupling/input_manifest.json)的数组形状、类型、hash 与构造真值。输入强度预先检查在 `[0,1]` 内，没有通过截断饱和值人为制造可恢复案例；这同样是 toy 成功的受控条件。

### 不能把亮度 toy 当作真实失败诊断

此前[真实光度与几何对照](../experiments/2026-10-09-photogeometric-real-sequences.md)有四对被最细层线搜索拒绝。已保存诊断说明最终试探没有严格下降，没有证明这些拒绝由曝光变化造成。当前 `+0.2` toy 成功，也没有证明仿射项会修复那四对、降低真实轨迹误差或提高通过率。

相对 gain/bias 只能表达一类全图变化。空间变化的阴影、反射、饱和、运动模糊、错误几何对应或非线性响应不必符合这个模型。更自由的亮度变量还可能吸收运动信息；是否改善真实观测，需要保留失败帧与完整指标的独立对照。

### 真实配准：满秩与较低残差不是定位保证

[固定真实 CPU 对照](../experiments/2026-10-09-affine-photogeometric-real-sequences.md)在 desk、xyz 各保留 100 帧，与旧六维 last-accepted 模型使用相同实际传感器数组和基础权重。每对从 identity、a=0、b=0 重新开始，不累计曝光状态；实际参考仍是最后接受的传感器帧，按对应的 c2w 组合候选。

desk 八维模型的 ATE 从 19.505mm 降到 19.261mm，相邻 RPE 和拒绝数也下降，但 raw 位置 RMSE 与终点误差上升。xyz 的 raw、ATE、平移 RPE 和终点误差上升，旋转 RPE 下降，拒绝数不变。这些结果没有支持全部指标共同改善，也不能把旧拒绝的消失归因为修复光照。

两轮八维候选都记录了联合 rank=8，却仍分别拒绝帧 218/225/268 和 266：最细层严格线搜索没有下降。独立审计复算了原始 GT 轨迹指标、参考组合链和全部保存的八维 trial 更新；真实目标与 rank 属于持久记录一致性检查，没有重新构建所有对应点 SVD。数学 toy 验证、真实满秩、工程接受和最终定位准确，是四种不同证据。

[Open3D 0.19 RGB-D odometry](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html)是光度、几何联合方法的官方背景入口。[DSO 论文](https://arxiv.org/abs/1607.02565)及其固定 commit `7b0c99f0` 的 [`AffLight` 源码](https://raw.githubusercontent.com/JakobEngel/dso/7b0c99f01d238f801c625beaff90240bcb007198/src/util/NumType.h)则提供仿射亮度传递的参考：源码包含每帧参数差和曝光时间比。本文只估计一对图像的相对参数，没有这些完整管线；不能把本页等同于 DSO 的全部方程、辐射模型或 Open3D 安装复现。

### 完整序列揭示的另一层边界

[实验24](../experiments/2026-10-09-full-affine-reference-controls.md)保留549/748个完整关联帧。desk 的六维与八维模型都从134到548连续拒绝415帧，参考停在133；八维少拒绝一帧并未解除停滞。xyz 八维的raw、ATE、平移RPE与末帧误差更高，旋转RPE略低。亮度自由度回答的是模型能解释哪些强度变化，参考是否持续可用还取决于历史、对应和接受规则，见[拒绝与恢复边界](17_rejection_reference_recovery.md)。

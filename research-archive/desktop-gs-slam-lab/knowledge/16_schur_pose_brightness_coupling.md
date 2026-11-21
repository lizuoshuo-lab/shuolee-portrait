---
id: "gslab-k16"
slug: "schur-complement-pose-brightness-coupling"
title: "Schur 补：位姿与亮度怎样互相吸收"
order: 16
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html"
  - "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html"
---

# Schur 补：去掉亮度变量后，位姿还剩多少信息

给光度里程计增加亮度增益（gain）和偏置（bias）后，八列雅可比满不满秩，不能只看位姿六列或亮度两列。**即使这两组列分别满秩，亮度仍可能解释掉某个位姿方向造成的观测变化。** Schur 补把这种耦合变成一个可检查的问题：先允许亮度解释它能解释的部分，再看剩余观测是否足以确定六维位姿。

本页接续[仿射亮度与八维可观测性](15_affine_brightness_observability.md)。用小型线性系统解释参数耦合：允许亮度解释观测后，位姿还剩多少信息。代数推导不自动证明真实图像的可观测性，也不代表某个在线求解器已经采用 Schur 消元。

## 从八维残差写到同一个线性问题

已有八维模型的状态增量是 $\delta=(\delta\mathbf v^\mathsf T,\delta\boldsymbol\omega^\mathsf T,\delta a,\delta b)^\mathsf T\in\mathbb R^8$：平移用 m，旋转用 rad，$a$ 是相对增益的对数，$b$ 是归一化灰度偏置。光度残差为：

$$
\mathbf p=\mathbf R\mathbf p_{\mathrm{cur}}+\mathbf t,\qquad
s=I_{\mathrm{ref}}(\pi(\mathbf p)),\qquad
r_I=e^a s+b-I_{\mathrm{cur}},\qquad
r_g=\mathbf n^\mathsf T(\mathbf p-\mathbf q).
$$

$$
\mathbf J_I=
\begin{bmatrix}
e^a\mathbf h^\mathsf T&
e^a(\mathbf p\times\mathbf h)^\mathsf T&e^a s&1
\end{bmatrix},
\qquad
\mathbf J_g=
\begin{bmatrix}
\mathbf n^\mathsf T&(\mathbf p\times\mathbf n)^\mathsf T&0&0
\end{bmatrix}.
$$

$\mathbf h=\mathbf J_\pi^\mathsf T\nabla I_{\mathrm{ref}}\in\mathbb R^3$，其取样与坐标定义沿用上一章；变换 $\mathbf T$ 从当前相机到参考相机，增量采用左扰动。

光度项前六列也包含增益因子，不能只在旧雅可比后面拼两列就算完成模型。几何项不依赖亮度参数，末两列为零。

接下来**固定对应、有效支持、工程尺度和一次迭代重加权最小二乘（Iteratively Reweighted Least Squares，IRLS）权重**。线性化残差为 $\mathbf r+\mathbf J\delta$，令：

$$
\boldsymbol\Omega=
\begin{bmatrix}
\sqrt{w_1}/\sigma_1&0&\cdots&0\\
0&\sqrt{w_2}/\sigma_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&\sqrt{w_m}/\sigma_m
\end{bmatrix},\qquad
\mathbf A=\boldsymbol\Omega\mathbf J,\qquad
\mathbf y=-\boldsymbol\Omega\mathbf r.
$$

$$
\delta^*=\underset{\delta\in\mathbb R^8}{\operatorname{argmin}}
\|\mathbf A\delta-\mathbf y\|_2^2.
$$

$m$ 为残差总行数，$\mathbf J\in\mathbb R^{m\times8}$、$\mathbf r\in\mathbb R^m$。每行 $\sigma_i>0$ 与该行残差同单位，$w_i\geq0$ 是固定的无量纲鲁棒权重，因而 $\boldsymbol\Omega\mathbf r$ 无量纲。

负号来自残差约定。$\sigma_i$ 是逐行归一化尺度，$w_i$ 是当前残差产生、在本次比较中保持不变的鲁棒权重。[固定配置](../outputs/affine-schur-observability-cpu-controls-003/configuration.json)取几何尺度 0.01 m、灰度尺度 0.1、归一化 Huber 分界 1；16 个残差中前八个是几何行，后八个是灰度行。

这里没有重新匹配、更新 Huber 权重或重新取样图像。因此后面的等价关系只属于这个冻结的加权二次问题，不等于不同非线性优化路径会选择同一候选或收敛到同一解。

## 正交投影：亮度能解释的部分先交给亮度

按位姿与亮度分块：$\mathbf A=\begin{bmatrix}\mathbf A_p&\mathbf A_n\end{bmatrix}$，$\mathbf A_p$ 有六列，$\mathbf A_n$ 有两列。固定一个位姿增量 $\delta_p$ 后，亮度子问题为：

$$
\mathbf A=
\begin{bmatrix}
\mathbf A_p&\mathbf A_n
\end{bmatrix},\qquad
\delta=
\begin{bmatrix}
\delta_p\\
\delta_n
\end{bmatrix},
\qquad
\min_{\delta_n}
\|\mathbf A_n\delta_n-(\mathbf y-\mathbf A_p\delta_p)\|_2^2.
$$

$$
\delta_n=\mathbf A_n^\dagger(\mathbf y-\mathbf A_p\delta_p)+\mathbf z,
\qquad \mathbf z\in\ker(\mathbf A_n),\qquad
\mathbf P=\mathbf I_m-\mathbf A_n\mathbf A_n^\dagger.
$$

$\mathbf A_p\in\mathbb R^{m\times6}$ 对应位姿，$\mathbf A_n\in\mathbb R^{m\times2}$ 对应亮度；$\dagger$ 表示 Moore–Penrose 伪逆，$\ker$ 表示零空间，$\mathbf I_m$ 是 $m$ 阶单位矩阵。

$\mathbf A_n^\dagger$ 是 Moore–Penrose 伪逆。$\mathbf A_n$ 的列空间包含亮度增量能够产生的所有残差变化；$\mathbf P$ 投影到它的正交补，把这些可由亮度解释的方向去掉。消元后的位姿问题是：

$$
\delta_p^*=\underset{\delta_p}{\operatorname{argmin}}
\|\mathbf P\mathbf A_p\delta_p-\mathbf P\mathbf y\|_2^2,\qquad
\mathbf S=\mathbf A_p^\mathsf T\mathbf P\mathbf A_p,
$$

$$
\mathbf P^2=\mathbf P,\qquad
\mathbf P^\mathsf T=\mathbf P,\qquad
\mathbf P\mathbf A_n=\mathbf0.
$$

这里的 $\mathbf P^2=\mathbf P$、$\mathbf P^\mathsf T=\mathbf P$、$\mathbf P\mathbf A_n=\mathbf0$ 是可以直接核对的性质。代码分别通过奇异值分解（Singular Value Decomposition，SVD）的有效左奇异向量，以及带列主元的完整 QR 分解构造这个子空间，并检查投影和拟合是否一致。

“消掉两个亮度参数”不意味着仍然拥有六个独立位姿约束；真正需要看的是 $\operatorname{rank}(\mathbf P\mathbf A_p)$。

## Schur 补：同一消元的块矩阵形式

令 $\mathbf H=\mathbf A^\mathsf T\mathbf A$、$\mathbf g=\mathbf A^\mathsf T\mathbf y$，其中 $\mathbf g$ 是正规方程右端，不是直接把目标梯度命名成 $\mathbf g$。分块后：

$$
\mathbf H=\mathbf A^\mathsf T\mathbf A,\qquad
\mathbf g=\mathbf A^\mathsf T\mathbf y,\qquad
\begin{bmatrix}
\mathbf H_{pp}&\mathbf H_{pn}\\
\mathbf H_{np}&\mathbf H_{nn}
\end{bmatrix}
\begin{bmatrix}
\delta_p\\
\delta_n
\end{bmatrix}
=
\begin{bmatrix}
\mathbf g_p\\
\mathbf g_n
\end{bmatrix}.
$$

$$
\mathbf H_{nn}=\mathbf A_n^\mathsf T\mathbf A_n,\qquad
\mathbf S=\mathbf H_{pp}-\mathbf H_{pn}\mathbf H_{nn}^{-1}\mathbf H_{np},
\qquad
\mathbf b_p=\mathbf g_p-\mathbf H_{pn}\mathbf H_{nn}^{-1}\mathbf g_n,
$$

$$
\mathbf S\delta_p=\mathbf b_p,\qquad
\mathbf H_{nn}\delta_n=\mathbf g_n-\mathbf H_{np}\delta_p.
$$

$\mathbf H_{pp}\in\mathbb R^{6\times6}$，$\mathbf H_{nn}\in\mathbb R^{2\times2}$，交叉块分别为 $6\times2$ 与 $2\times6$；$\mathbf H_{nn}^{-1}$ 的写法要求亮度块满列秩。

当 $\mathbf A_n$ 满列秩时，$\mathbf H_{nn}$ 可逆，且 $\mathbf S=\mathbf A_p^\mathsf T\mathbf P\mathbf A_p$、$\mathbf b_p=\mathbf A_p^\mathsf T\mathbf P\mathbf y$。实现应使用线性求解处理这些块，不必显式形成逆矩阵。解出 $\delta_p$ 后，再恢复亮度增量。

如果 $\mathbf A_n$ 秩亏，普通 $\mathbf H_{nn}$ 的逆就不合法。在当前 Gram 矩阵结构下，伪逆形式仍有：

$$
\mathbf S=
\mathbf H_{pp}-\mathbf H_{pn}\mathbf H_{nn}^\dagger\mathbf H_{np}
=\mathbf A_p^\mathsf T\mathbf P\mathbf A_p.
$$
秩揭示 QR 分解、SVD 与伪逆仍能定义合法的投影消元；数值库偶然返回一个巨大解，不是可逆的证明。即使 $\mathbf H_{nn}$ 可逆，$\mathbf S$ 也可能秩亏，此时不存在唯一的六维位姿解。

下面每组都提供原始 $\mathbf J$、$\mathbf r$、$\sigma_i$、$w_i$，加权 $\mathbf A$、$\mathbf y$，以及投影、块矩阵和结果。合成控制使用 $10^{-11}$ 的相对奇异值阈值检查数值秩；它没有替换真实里程计已有的秩与条件数门槛。

| 构造系统 | $\operatorname{rank}(\mathbf A)$ | $\operatorname{rank}(\mathbf A_n)$ | $\operatorname{rank}(\mathbf P\mathbf A_p)$ | 可复算材料 |
|---|---:|---:|---:|---|
| 完整八维可辨识 | 8 | 2 | 6 | [矩阵](../outputs/affine-schur-observability-cpu-controls-003/full-rank/matrices.json)、[结果](../outputs/affine-schur-observability-cpu-controls-003/full-rank/result.json) |
| 两个亮度列相关 | 7 | 1 | 6 | [矩阵](../outputs/affine-schur-observability-cpu-controls-003/singular-nuisance/matrices.json)、[结果](../outputs/affine-schur-observability-cpu-controls-003/singular-nuisance/result.json) |
| 亮度满秩但吸收一个位姿方向 | 7 | 2 | 5 | [矩阵](../outputs/affine-schur-observability-cpu-controls-003/pose-absorbed-by-full-rank-nuisance/matrices.json)、[结果](../outputs/affine-schur-observability-cpu-controls-003/pose-absorbed-by-full-rank-nuisance/result.json) |
| 近似吸收位姿方向 | 8 | 2 | 6 | [矩阵](../outputs/affine-schur-observability-cpu-controls-003/near-pose-nuisance-coupling/matrices.json)、[结果](../outputs/affine-schur-observability-cpu-controls-003/near-pose-nuisance-coupling/result.json) |

第三组的 $\mathbf A_p$ 本身也是秩 6，但消掉亮度后只剩秩 5。它存在一个同时改变平移和 $a$ 的显式零空间方向，二者作用抵消，预测观测不变。第二组则是亮度内部不可唯一分开，但剩余数据仍能唯一确定六维位姿。两个“联合秩 7”因而具有不同含义。

这些案例还满足 $\operatorname{rank}(\mathbf A)=\operatorname{rank}(\mathbf A_n)+\operatorname{rank}(\mathbf P\mathbf A_p)$。只报联合秩而不看退化发生在哪个子空间，会丢掉这个区别。

## 最小残差相同，最小范数参数未必相同

联合最小二乘求解通常在所有最优解中选择整个八维向量的最小范数解；先求最小范数位姿，再求最小范数亮度，选择规则不同。存在位姿—亮度耦合的秩亏例中，两条路径参数最大元素差约 0.001416，但预测观测差只有约 $3.58\times10^{-15}$，最小加权残差一致。

前一个数字混合了 m、rad、对数亮度增益和灰度强度，不能称为米制位姿误差。秩亏时改变参数单位，也可能改变哪一个解被称为“最小范数”。因此审核先比较拟合值、最小残差、秩和零空间；只有问题确实具有唯一解，才要求恢复后的物理参数也在数值容差内相同。

## 小特征值、列归一化与正规方程

近耦合例仍是秩 8，但原坐标下 $\mathbf S$ 的最小特征值约 $10^{-4}$，与其他方向相差很大。正规块解与联合矩形最小二乘解的预测观测差约 $7.76\times10^{-10}$，参数最大元素差却约 $5.48\times10^{-6}$。拟合接近不代表每个参数同样稳定，也不能把浮点下的近似一致写成字节相同。

该合成线性解甚至包含约 **8.69 m 的一个平移分量与 −217 的 对数亮度增益分量**。这只是冻结线性模型的数学解，没有经过真实非线性重投影与步幅、接受判据，不能直接作为可接受的配准更新；不能声称现有求解器取了这一步，或由此推出重建精度。原里程计的步幅限制、非线性试探和拒绝规则没有在这个小矩阵实验中执行。

设 $\mathbf A$ 每列范数为 $d_j$，列归一化对应的是坐标变换：

$$
d_j=\sqrt{\sum_{i=1}^{m}A_{ij}^2},\qquad
\mathbf D=
\begin{bmatrix}
d_1&0&\cdots&0\\
0&d_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&d_n
\end{bmatrix},
\qquad
\mathbf D^{-1}=
\begin{bmatrix}
1/d_1&0&\cdots&0\\
0&1/d_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&1/d_n
\end{bmatrix}.
$$

$$
\widehat{\mathbf A}=\mathbf A\mathbf D^{-1},\qquad
\mathbf z=\mathbf D\delta,\qquad
\mathbf z^*=\underset{\mathbf z}{\operatorname{argmin}}
\|\widehat{\mathbf A}\mathbf z-\mathbf y\|_2^2,\qquad
\delta=\mathbf D^{-1}\mathbf z.
$$

$n=8$ 为列数，$d_j$ 是第 $j$ 列的欧氏范数，单位与该列相同；上述倒数要求所有 $d_j>0$。零列表示缺少对应方向的观测约束，必须单独处理。$\mathbf z$ 是归一化坐标中的增量，不能直接作为物理增量使用。

直接把 $\mathbf z$ 当成物理增量会改变更新。本材料还把平移从 m 改用 mm，并改变部分参数的符号约定，检查在同时变换雅可比后，满秩系统能恢复同一个物理解。谱与条件数则可以改变：它们依赖参数坐标，不能被当作不带单位约定的绝对质量分数。

精确算术下，满列秩矩阵满足 $\kappa_2(\mathbf A^\mathsf T\mathbf A)=\kappa_2(\mathbf A)^2$。独立的[条件数矩阵](../outputs/affine-schur-observability-cpu-controls-003/normal-equation-conditioning/matrices.json)刻意设置跨度很大的奇异值；[实测结果](../outputs/affine-schur-observability-cpu-controls-003/normal-equation-conditioning/result.json)为 $\kappa_2(\mathbf A)\approx10^6$、$\kappa_2(\mathbf A^\mathsf T\mathbf A)\approx1.000017\times10^{12}$。SVD 解的最大参数误差约 $3.16\times10^{-12}$，正规方程约 $2.25\times10^{-6}$。这是该无量纲数值例的结果，不是所有场景的固定误差倍率。

理论半正定的 $\mathbf S$ 在浮点下也可能出现约 $-2.47\times10^{-12}$ 的特征值。结合矩阵尺度、投影性质和秩阈值，应把本例理解为理论零方向附近的舍入现象，不能据此发明真实负曲率。

## 先验使矩阵可解，是给问题增加假设

设 $\alpha\geq0$ 为归一化坐标中的先验权重。在列归一化坐标给亮度加 $\alpha(z_a^2+z_b^2)$，或给全部状态加 $\alpha\sum_{j=1}^{n}z_j^2$，都定义了新的优化目标。后者若写回原坐标，是 $\alpha\sum_{j=1}^{n}(d_j\delta_j)^2$，不是随意在带混合单位的原矩阵上加同一个常数。

[先验与阻尼控制矩阵](../outputs/affine-schur-observability-cpu-controls-003/prior-and-damping/matrices.json)预先固定 $\alpha=0.25$。[结果](../outputs/affine-schur-observability-cpu-controls-003/prior-and-damping/result.json)显示，原数据秩 7，增加亮度先验后增广系统秩 8，归一化坐标中的 Schur 最小特征值变为 0.2。但这份曲率来自“亮度增量接近零”的假设，不是数据突然提供了新的观测。

该例原数据项最小代价约为 11.84724094，增加亮度先验后的数据项约为 11.84727896，另有先验惩罚；全状态阻尼的数据项约为 11.90347373，也另有惩罚。三者解决的是不同问题，不能因矩阵更好求解就宣称原始数据拟合更好。

阻尼、先验、参数钳制和拒绝更新也不是同一个操作。现有真实八维求解器没有因为本文而增加任何一种。未标定残差噪声、相关性、鲁棒权重和工程尺度时，$\mathbf H$ 或 $\mathbf S$ 的逆同样不能直接称为可信的位姿协方差。


## 从代数回到真实系统

构造矩阵控制验证消元和投影恒等式；真实系统还要从同一观测、权重和线性化状态构建 $\mathbf J$。先验与阻尼能够让矩阵可解，却添加了假设。检查正规化谱、参数单位和真值误差，才能判断剩余信息是否支持可靠运动估计。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

本页接续[仿射亮度与八维可观测性](15_affine_brightness_observability.md)。证据来自一个独立的[CPU 小矩阵生成与检查脚本](../scripts/check_affine_schur_observability.py)，源码 SHA256 为 `74343301…ad27b`，以及[固定合成结果](../outputs/affine-schur-observability-cpu-controls-003/metrics.json)。这些矩阵是刻意构造的线性系统，不是从真实图像或相机采样的雅可比。它没有改动现有里程计，也没有引入新的 Schur 求解器、阻尼或先验。

### 独立审核与真实结果之间的边界

[独立数学审核](../outputs/affine-schur-observability-independent-audit-001/metrics.json)状态为 `passed_independent_synthetic_Schur_audit`，审核源码 SHA256 为 `54650953…a6b3`。[审核脚本](../scripts/audit_affine_schur_observability.py)没有导入生成脚本或其投影函数，而是从保存的 raw J、r、σ、w 重建系统，使用 SciPy 带列主元 QR 的正交补和矩形 `gelsy` 最小二乘复算；还核对了先验增广问题、错号负控和条件数例。

审核验证了 16 份完成产物与 6 份历史失败证据，共重新检查 24 份消费文件；这些计数属于不同范围，不能简单把前两项相加解释成总数。近耦合例的正规方程解存在可见参数差异，审核原样保留，没有要求所有求解路径参数完全一致。

[完成清单](../outputs/affine-schur-observability-cpu-controls-003/completion.json)绑定输入、结果和源码 hash；[失败历史](../outputs/affine-schur-observability-cpu-controls-003/failure_history.json)保存开发期的语法错误、序列化失败和过强负控制断言。早期断言要求错误符号至少增加 0.001 代价，但实际增加为正的 0.000378437；最终检查改为验证解析的 `ΔpᵀSΔp` 与浮点分辨率边界，没有修改构造输入或隐藏失败。

真实观测仍需另外检验。[实验 22](../experiments/2026-10-09-affine-photogeometric-real-sequences.md)中的六维与八维 CPU 对照存在指标取舍，不能由这里的消元公式推导出“加入亮度整体改善”。本页只回答一次固定支持、固定权重线性化中，哪些方向可辨识、怎样合法消元，以及数值坐标和附加假设怎样改变解释。

[完整序列对照](../experiments/2026-10-09-full-affine-reference-controls.md)进一步保留了两种模型的长期拒绝与全部时间点。它使用原求解器，没有因本页数学控制增加先验或阻尼；不能从一次局部Schur系统可解推导出整条参考链稳定。

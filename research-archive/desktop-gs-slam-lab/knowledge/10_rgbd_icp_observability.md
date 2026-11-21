---
id: "gslab-k10"
slug: "rgbd-projective-icp-jacobian-observability"
title: "10 RGB-D ICP：位姿雅可比、几何退化与可观测性"
order: 10
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://www.cs.princeton.edu/~smr/papers/fasticp/"
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/icp_registration.html"
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/robust_kernels.html"
---

# RGB-D ICP：残差小，为什么仍可能跟踪不准

投影式点到平面迭代最近点配准（projective point-to-plane Iterative Closest Point，ICP）用两帧深度的局部几何估计相机运动。它把当前点变换到上一帧，投影寻找对应点，再让点沿目标表面法向的距离减小。这条路径可以帮助拆开高斯 SLAM 中的两个问题：**深度本身能提供多少运动约束，以及地图渲染与位姿优化怎样影响这些约束。**

本页对应已运行的 [CPU ICP 教学脚本](../scripts/track_rgbd_icp.py)。它使用 TUM RGB-D 数据，但优化只使用深度，不使用 RGB 光度损失；不建立地图，不包含回环、重定位或 BA。它也不是 KinectFusion、Open3D ICP 或官方 SplaTAM 的复现。经典 ICP 各阶段的选择可参考 Rusinkiewicz 与 Levoy 的 [Efficient Variants of the ICP Algorithm（2001）](https://www.cs.princeton.edu/~smr/papers/fasticp/)；[Open3D 0.19.0 文档](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/icp_registration.html)提供 point-to-plane 目标函数的公开说明，本实验没有安装或调用 Open3D。

## 先固定变换方向，再讨论梯度

令当前帧为 k，上一帧为 k−1。当前深度反投影得到 $\mathbf q$，目标点 $\mathbf v$ 和目标法向 $\mathbf n$ 都在上一帧相机坐标系中。定义：

$$
\mathbf T_{k-1\leftarrow k}=
\begin{bmatrix}
\mathbf R&\mathbf t\\
\mathbf0^\mathsf T&1
\end{bmatrix},
\qquad
\mathbf p=\mathbf R\mathbf q+\mathbf t,\qquad
r=\mathbf n^\mathsf T(\mathbf p-\mathbf v).
$$

$$
\mathbf C_k=\mathbf C_{k-1}\mathbf T_{k-1\leftarrow k},
\qquad
\mathbf C_0=\mathbf I_4.
$$

$\mathbf R\in SO(3)$ 是旋转，$\mathbf t,\mathbf p,\mathbf q,\mathbf v\in\mathbb R^3$ 以米计；$\mathbf n\in\mathbb R^3$ 是无量纲单位法向，$r$ 是以米计的有符号点到平面残差。$\mathbf C_k$ 是相机到世界的齐次变换，$\mathbf I_4$ 为四阶单位矩阵。

这里的相机到世界变换在代码中记作 c2w。实际 NumPy 点数组按行存放，因此 $\mathbf P_{\mathrm{row}}=\mathbf Q_{\mathrm{row}}\mathbf R^\mathsf T+\mathbf1\,\mathbf t^\mathsf T$；这和上面的列向量公式完全一致。不能因为数组按行存储，就把位姿组合顺序也反过来。

每一对图像的相对变换都从单位变换开始。首帧单位变换定义世界坐标参考，不来自独立真值。相机向右移动时，在同一静态场景中，将当前点送回上一相机的相对变换应包含正的 x 平移；这个方向已用合成场景校验。方向弄反仍可能得到看似合理的矩阵，却会把整条轨迹累计错。

## 投影式对应与法向约束

深度采用光轴方向的 z-depth，而非到相机中心的欧氏距离。有效像素按同一套内参反投影：

$$
\mathbf q(u,v)=
\begin{bmatrix}
(u-c_x)z/f_x\\
(v-c_y)z/f_y\\
z
\end{bmatrix},
\qquad
\begin{bmatrix}
u'\\
v'
\end{bmatrix}
=
\begin{bmatrix}
f_xp_x/p_z+c_x\\
f_yp_y/p_z+c_y
\end{bmatrix},
\qquad z>0,\quad p_z>0.
$$

$f_x,f_y$ 是焦距，$c_x,c_y$ 是主点坐标，与 $u,v$ 一样以像素计；$z$ 为相机轴向深度，单位米。

脚本使用共享 TUM loader 的像素中心和缩放约定，三层均从原始 PNG 读取；深度用 nearest 采样，内参按像素中心关系缩放。这里的金字塔不是对上一层深度反复缩放，也没有双边滤波。对应通过上一帧图像上的最近整数像素查找，不是全局三维最近邻搜索。

depth_geometry() 使用左右、上下中心差分的叉积计算法向，朝相机定向。中心及四个邻点必须全部有效；任意邻点深度跳变超过 $0.04\,\mathrm m+0.02z$ 米，就排除该中心。这样可以减少孔洞边缘、前后景交界处跨越不同表面所形成的假法向。这个规则仍可能在连续相交的墙面处保留混合的差分法向，不是精确的表面分类。

correspondences() 还要求变换后点位于相机前方、投影在图像内、目标法向有效；旋转后的源法向与目标法向夹角不超过 45°，三维点距不超过该层阈值。多个源点投到同一目标像素时，仅保留三维距离最小的一对，避免同一目标像素被重复加权。

这些拒绝规则回答“这对局部几何是否足够可信”，不能保证对应就是同一个物理表面。重复结构、遮挡、动态物体、深度噪声和错误初值仍可能给出小残差的错误对应。

## 左扰动与点到平面残差雅可比

脚本的六维增量顺序是 $\delta\boldsymbol\xi=(\delta v_x,\delta v_y,\delta v_z,\delta\omega_x,\delta\omega_y,\delta\omega_z)^\mathsf T$。平移切向量用米，旋转用弧度，更新在左侧：

$$
\delta\boldsymbol\xi=
\begin{bmatrix}
\delta\mathbf v\\
\delta\boldsymbol\omega
\end{bmatrix}\in\mathbb R^6,
\qquad
[\mathbf p]_\times=
\begin{bmatrix}
0&-p_z&p_y\\
p_z&0&-p_x\\
-p_y&p_x&0
\end{bmatrix}.
$$

$$
\delta\boldsymbol\xi^\wedge=
\begin{bmatrix}
[\delta\boldsymbol\omega]_\times&\delta\mathbf v\\
\mathbf0^\mathsf T&0
\end{bmatrix},
\qquad
\mathbf T_{\mathrm{new}}=\exp(\delta\boldsymbol\xi^\wedge)\mathbf T,
\qquad
\mathbf p_{\mathrm{new}}\approx\mathbf p+\delta\mathbf v+
\delta\boldsymbol\omega\times\mathbf p.
$$

$$
r_{\mathrm{new}}\approx r+
\mathbf n^\mathsf T\delta\mathbf v+
(\mathbf p\times\mathbf n)^\mathsf T\delta\boldsymbol\omega,
\qquad
\mathbf J=
\begin{bmatrix}
\mathbf n^\mathsf T&(\mathbf p\times\mathbf n)^\mathsf T
\end{bmatrix}\in\mathbb R^{1\times6}.
$$

$[\mathbf p]_\times\mathbf a=\mathbf p\times\mathbf a$ 定义反对称矩阵；平移增量以米计，旋转增量以弧度计。线性化时对应点与目标法向固定。

推导时把本轮对应的目标点和目标法向固定。$\mathbf n^\mathsf T(\delta\boldsymbol\omega\times\mathbf p)=(\mathbf p\times\mathbf n)^\mathsf T\delta\boldsymbol\omega$ 决定了叉积顺序；写成 $\mathbf n\times\mathbf p$ 会反转旋转梯度。右扰动、反方向的 $\mathbf T$、相反的残差定义，都需要重新推导配套的雅可比，不能只替换矩阵乘法位置。

最小化线性化后的加权残差，得到：

$$
\min_{\delta\boldsymbol\xi}\,
\frac12\sum_{i=1}^{N}w_i(r_i+\mathbf J_i\delta\boldsymbol\xi)^2,
\qquad
\mathbf H\delta\boldsymbol\xi=-\mathbf g,
\qquad
\mathbf H=\mathbf J^\mathsf T\mathbf W\mathbf J,\quad
\mathbf g=\mathbf J^\mathsf T\mathbf W\mathbf r.
$$

$$
\mathbf W=
\begin{bmatrix}
w_1&0&\cdots&0\\
0&w_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&w_N
\end{bmatrix},
\qquad w_i\geq0.
$$

$\mathbf J\in\mathbb R^{N\times6}$ 汇集各行雅可比，$\mathbf r\in\mathbb R^N$ 汇集残差，$\mathbf H\in\mathbb R^{6\times6}$ 为局部加权信息矩阵。$w_i$ 是本次线性化中固定的无量纲权重。

实现中记录 $\mathbf H$ 的特征值，但用加权、列归一化后的 直接最小二乘求解 求解，没有显式计算 $\mathbf H$ 的逆。直接求最小二乘可避免先形成正规方程再求解所带来的额外条件数平方问题。se3_exp() 使用包含 三维旋转群 SO(3) 的左雅可比的完整 SE(3) 指数映射，有限增量的平移部分并不总等于直接填入 δ$\mathbf v$。

$$
A_{i,:}=\sqrt{w_i}\,\mathbf J_i,\qquad
b_i=-\sqrt{w_i}\,r_i,\qquad
\delta\boldsymbol\xi^*=
\underset{\delta\boldsymbol\xi}{\operatorname{argmin}}
\|\mathbf A\delta\boldsymbol\xi-\mathbf b\|_2^2.
$$

$\mathbf A\in\mathbb R^{N\times6}$ 与 $\mathbf b\in\mathbb R^N$ 是直接最小二乘系统。原数值求解示例保留在末尾实验附录。

## Huber、多尺度与线搜索分别解决什么

Huber 鲁棒损失让小残差保持二次惩罚，让大残差转为线性惩罚。本脚本的 Huber 分界为 $\delta=0.01\,\mathrm m$，迭代重加权最小二乘（Iteratively Reweighted Least Squares，IRLS）权重为 $\min(1,\delta/|r|)$（非零残差）。它降低大残差的影响，不会自动识别所有错误对应；大量几何一致的错误对应仍可能主导估计。[Open3D 的 robust kernels 文档](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/robust_kernels.html)说明了这类加权最小二乘视角。

Huber 损失和迭代重加权最小二乘（Iteratively Reweighted Least Squares，IRLS）权重可明确写为：

$$
\rho_\delta(r)=
\begin{cases}
\frac12r^2,&|r|\leq\delta,\\
\delta(|r|-\frac12\delta),&|r|>\delta,
\end{cases}
\qquad
w(r)=
\begin{cases}
1,&|r|\leq\delta,\\
\delta/|r|,&|r|>\delta.
\end{cases}
$$

$\delta=0.01\,\mathrm m$，因此 $\rho_\delta$ 以平方米计，$w$ 无量纲，零残差时权重为 $1$。

固定的三层设置为：

| 分辨率 | 最大迭代数 | 对应三维距离上限 |
|---|---:|---:|
| 40 × 30 | 15 | 0.10 m |
| 80 × 60 | 12 | 0.07 m |
| 160 × 120 | 10 | 0.05 m |

粗层先估计较大的图像位移，细层继续修正，但 ICP 仍是局部方法。多尺度不会让任意初值都能收敛。每轮重新建立对应后，脚本在该组固定对应和法向上做 Huber 目标的线搜索，接受下降的左增量；下一轮对应发生变化时，整体目标未必严格单调。

单次增量还限制为至多 0.05m / 5°，最终相对位姿阈值为 0.20m / 20°。这些是本教学实现预先指定的保护阈值，不能解释成来自数据集的相机运动定律。两个真实对照使用同一组默认参数，没有根据 GT 再调参。

## 单平面为什么最多只能提供三个独立方向的约束

考虑朝向相机的无限正面平面，目标法向 $\mathbf n=(0,0,-1)^\mathsf T$。雅可比每一行为：

$$
\mathbf J_i=
\begin{bmatrix}
0&0&-1&-p_y&p_x&0
\end{bmatrix}.
$$

足够分散的平面点能够约束沿法向的平移，以及绕平面内两个轴的倾斜；沿平面的两维平移、绕法向的旋转不改变 point-to-plane 距离。理想情况下 秩 为 3；点的分布更差时还会更低。法向一致的数万个点也不能自动补足六维可观测性。

$\mathbf H+\lambda\mathbf I_6$ 可以使线性方程有唯一的数值解，却没有新增传感器观测。缺失方向的结果取决于阻尼、先验或初值，不能写成“几何已经恢复出完整六维位姿”。运动先验或其他传感器可以增加约束，但需要明确它们提供了什么信息。

本脚本对不足 秩 6 或超过条件数阈值的系统直接拒绝该轮求解，不用阻尼掩盖退化。粗层不能求解时，仍会尝试细层；最终是否接受还要查看细层结果。合成单平面测试确实返回 秩 3，拒绝把 5cm 的平面内平移声称为可恢复运动。

## 列归一化条件数不等于位姿协方差

平移列和旋转列具有不同的物理尺度，旋转列还随点到相机的距离变化。脚本同时记录原始加权 $\mathbf H$ 和列归一化后的谱：令 $\mathbf A=\mathbf W^{1/2}\mathbf J$，$\mathbf S$ 的对角项为 $\mathbf A$ 各列的范数，使用 $\overline{\mathbf A}=\mathbf A\mathbf S^{-1}$ 求解，再把增量还原为 $\delta\boldsymbol\xi=\mathbf S^{-1}\boldsymbol\eta$。

设 $s_j=\sqrt{\sum_iA_{ij}^2}>0$ 为第 $j$ 列的欧氏范数，则：

$$
\mathbf S=
\begin{bmatrix}
s_1&0&\cdots&0\\
0&s_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&s_6
\end{bmatrix},
\qquad
\overline{\mathbf A}=\mathbf A\mathbf S^{-1},\qquad
\delta\boldsymbol\xi=\mathbf S^{-1}\boldsymbol\eta.
$$

$s_j=0$ 时相应方向没有该线性系统的观测约束，不能直接取倒数。

列归一化主要改善求解坐标的尺度，并用于预先声明的数值秩与条件数判据。$\overline{\mathbf H}$ 的特征值和条件数依赖这套缩放方法；小条件数不等于每个物理方向的信息都强，也不说明估计无偏。脚本同时保存列范数，避免只看归一化条件数而忽略某个原始方向很弱。

只有在局部线性化、噪声模型、权重和对应关系等假设成立时，经过正确尺度变换的逆信息矩阵才可能用于近似不确定性分析。这里的 depth 噪声、Huber 权重和随迭代变化的对应都需要额外建模，当前输出**没有经过统计标定的位姿协方差**。


## 无失败标记不等于收敛或定位准确

失败规则通常检查对应数量、秩、条件数、残差与运动幅度；零标记只说明未越过这些门限。小更新可能停在局部极小值，用满预算也不一定发散。一起检查轨迹误差、对应分布、谱和停止原因。被拒绝帧要保留，位姿怎样传播必须明确。

## 代码阅读入口

[track_rgbd_icp.py](../scripts/track_rgbd_icp.py) 是教学实现：depth_geometry() 生成法向，correspondences() 筛选对应，residual_and_jacobian() 定义左扰动残差，normal_equations() 求解，register_pair() 运行多尺度和线搜索，integrate_relative() 累计 c2w 并传播失败状态。

结合[深度与覆盖](07_rgbd_depth_coverage.md)、[位姿优化](06_pose_optimization.md)阅读。几何退化是观测缺少方向约束，不能仅靠更多迭代修复。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

### 合成检查分别证明了哪一层

完整记录见 [CPU 检查清单](../manifests/track-rgbd-icp-cpu-checks.json)。其中包含固定配置、Python / NumPy / SciPy 版本及脚本和依赖源文件 SHA256。

| 检查 | 实测结果 | 支持的结论 |
|---|---:|---|
| SE(3) 指数与 SciPy `expm` 对照 | 最大绝对差 6.94e−18 | 指数映射在这组输入上与矩阵指数一致 |
| 左扰动 J 与中心有限差分 | 最大绝对差 1.40e−9 | 本残差、方向、增量顺序下的雅可比符号正确 |
| 精确对应的六维刚体恢复 | 最大矩阵元素误差 1.11e−16 | 求解与左乘更新能恢复该已知刚体运动 |
| 射线与房间平面求交得到的深度，经实际 projective ICP | 0.154210 mm / 0.008548° | 含投影对应、差分法向和多尺度的整条合成路径可恢复这次小运动 |
| 无限单平面与 5cm 横移 | rank 3，拒绝 | 没有把不可观测横移当成六维成功恢复 |
| 孔洞、深度边缘、空深度、失败帧递推 | 检查通过 | 对应的输入筛选和失败保留分支得到覆盖 |

合成房间在墙面相交处存在差分法向与离散匹配误差，因此它和精确对应检查不是同一个精度问题。合成恢复误差也不能替代真实 TUM 的轨迹误差。

本次实测脚本的 SHA256 是：

```text
21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a
```

### 真实 TUM：几何诊断与轨迹评价必须分开

两段均来自同一 TUM `fr1/desk` 数据：关联后帧编号 30–49 和 200–299。编号不是原始 RGB 文件行号。`validation.json` 已逐项核对，与对应 GS V2 对照使用相同 loader SHA、帧编号、RGB / depth / GT 时间戳及传感器文件关联；关联采用严格 20ms 门限、32Hz 上限并禁止重复使用深度帧。

优化从首相机 identity 出发，GT pose 只在轨迹估计完成后进入评价。共享 loader 在帧筛选时使用 GT 时间戳，所以不能把数据处理描述为“完全不读取 GT 信息”；关键边界是 GT pose 没有进入初始化、损失、对应或候选位姿选择。

下表的所有平移量都是 mm。Raw 是首相机规范下直接比较相机中心的 RMSE；ATE 是对整条轨迹做一次无尺度 SE(3) 刚性对齐后的 RMSE；RPE 比较相邻帧相对运动，Δ=1。GS 行是本机教学原型 V2 的结果，`depth0.1` 与 `depth1.0` 表示其深度损失权重。

| 片段 | 实现 / 配置 | Raw RMSE | SE3 ATE | Δ1 RPE 平移 | Δ1 RPE 旋转 |
|---|---|---:|---:|---:|---:|
| 20 帧 30–49 | [CPU ICP](../outputs/tum-icp-short20-001/metrics.json) | 27.019 | 10.225 | 4.306 | 0.503° |
| 20 帧 30–49 | [GS V2 depth0.1](../outputs/tum-online-v2-baseline-001/metrics.json) | 210.795 | 97.715 | 26.276 | 1.144° |
| 20 帧 30–49 | [GS V2 depth1.0](../outputs/tum-online-v2-depth1-001/metrics.json) | 28.497 | 14.323 | 6.695 | 0.604° |
| 100 帧 200–299 | [CPU ICP](../outputs/tum-icp-transfer100-001/metrics.json) | 64.103 | 30.627 | 5.000 | 0.759° |
| 100 帧 200–299 | [GS V2 depth0.1](../outputs/tum-online-v2-transfer-baseline-001/metrics.json) | 481.349 | 208.089 | 16.241 | 0.964° |
| 100 帧 200–299 | [GS V2 depth1.0](../outputs/tum-online-v2-transfer-depth1-001/metrics.json) | 388.756 | 83.753 | 11.321 | 0.862° |

这张表说明，在这两个片段及现有配置下，几何对照得到了表中的较小轨迹误差。它没有单独控制初始化、目标函数、优化器、预算和地图等全部因素，不能把差异唯一归因于高斯表示，也不能推出经典 ICP 在其他场景或成熟 GS-SLAM 系统中的普遍优劣。ICP 没有输出可渲染高斯地图，二者完成的系统工作也不同。

每对真实 ICP 的最终几何诊断如下，均包含全部被处理帧对：

| 片段 | 对应点数量范围 | point-to-plane RMSE 均值 | 最终 rank | 归一化 H 条件数范围 | 拒绝帧对 |
|---|---:|---:|---:|---:|---:|
| 20 帧 | 9,212–10,855 | 5.246 mm | 全部 6 | 20.414–30.530 | 0 / 19 |
| 100 帧 | 8,488–13,828 | 2.394 mm | 全部 6 | 25.745–70.261 | 0 / 99 |

100 帧的局部平面残差均值更小，轨迹 Raw RMSE 却更大。局部配准残差只衡量被选对应上的模型拟合；每步的小偏差仍会在位姿链中累积。良好条件数也只说明当前数值系统的某种局部性质。

### 零 failure flags 为什么不是“每层都收敛”

这两个 ICP 运行没有触发最终失败门限，也没有因 rank / condition 终止的迭代。但 20 帧共 57 个金字塔层中有 35 层用满预算，100 帧共 297 层中有 172 层用满预算；其余层以小更新停止。接受结果不要求每层都满足小更新判据。

脚本将对应点不足、最终 rank 不足、条件数过大、最终残差过大、相对运动超阈值等记录为 failure flags；被拒帧会保留在轨迹中，并传播上一估计。真实两次运行恰好没有走到最终拒绝分支，失败保留逻辑的覆盖来自前述合成测试，不能混称为“真实失败恢复已验证”。

零标记只表示没有越过已定义的保护门限。小更新也可能停在局部极小值，预算用完不一定代表发散。评价时应一起读取轨迹误差、残差、对应数量、谱、停止原因与完整失败记录。

两次 CPU 跟踪循环观测耗时分别为 1.641s 和 8.683s；它们并发启动，且循环计时不包含所有导入和图像加载成本，所以这些数值仅是运行记录，不转换为整套系统 FPS，也不用于硬件速度优劣结论。

### 对应到本地证据和代码

实现入口是 [track_rgbd_icp.py](../scripts/track_rgbd_icp.py)：`depth_geometry()` 生成法向，`correspondences()` 建立并筛选对应，`residual_and_jacobian()` 定义残差和左扰动 J，`normal_equations()` 记录谱并求解，`register_pair()` 运行多尺度和线搜索，`integrate_relative()` 负责 c2w 累计及失败传播。评价调用现有 NumPy 轨迹审计模块；调用另一个指标函数不等于另有独立算法证明 ICP 正确。

| 实测记录 | 指标 | 预先指定配置及源码 SHA | 帧关联与文件完整性检查 |
|---|---|---|---|
| 20 帧 | [metrics](../outputs/tum-icp-short20-001/metrics.json) | [configuration](../outputs/tum-icp-short20-001/configuration.json) | [validation](../outputs/tum-icp-short20-001/validation.json) |
| 100 帧 | [metrics](../outputs/tum-icp-transfer100-001/metrics.json) | [configuration](../outputs/tum-icp-transfer100-001/configuration.json) | [validation](../outputs/tum-icp-transfer100-001/validation.json) |

本地输出还保存每步的 `iterations.jsonl`、`loss.csv`、全部估计 / GT 轨迹和最终评价。深度、覆盖与高斯位姿损失之间的关系见 [07 RGB-D 深度监督](07_rgbd_depth_coverage.md)；渲染输出和梯度语义见 [09 高斯渲染后端替换](09_renderer_backend_semantics.md)。


### 从理论正文保留的实现片段

以下为本轮公式规范前的实现示例，保留原片段供核查，不是新增实验结果。

```python
# 伪代码：省略筛选、退化检查和线搜索。
p = q @ T[:3, :3].T + T[:3, 3]
r = np.sum(n * (p - v), axis=1)
J = np.column_stack((n, np.cross(p, n)))

w = np.minimum(1.0, huber_delta / np.maximum(np.abs(r), 1e-15))
A = J * np.sqrt(w)[:, None]
b = -r * np.sqrt(w)
s = np.linalg.norm(A, axis=0)
# 实际代码先处理零列并检查 rank / condition，不能直接除以零。
eta = np.linalg.lstsq(A / s[None, :], b, rcond=1e-6)[0]
delta = eta / s
T = se3_exp(delta) @ T
```

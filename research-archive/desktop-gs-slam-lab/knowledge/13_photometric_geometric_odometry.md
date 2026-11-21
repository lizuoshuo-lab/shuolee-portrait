---
id: "gslab-k13"
slug: "rgbd-photometric-geometric-odometry"
title: "13 光度与几何里程计：插值导数、单位和可观测性"
order: 13
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
---

# 光度与几何里程计：纹理怎样补充深度约束

单平面上的 point-to-plane ICP 无法仅凭平面距离恢复所有六维运动。平面上的纹理有机会补充缺失的约束：相机沿平面移动时，点到平面的距离可以不变，但投影后的颜色通常会变。能否从这种变化恢复位姿，取决于纹理方向、亮度一致性、投影对应和有效像素范围，不能只看是否添加了一个彩色图像损失。

本页对应[光度与几何教学实现](../scripts/rgbd_photogeometric_odometry.py)，沿取样、残差、雅可比和接受规则理解算法。几何和光度提供互补信息的前提不同，联合目标更低不自动保证每项轨迹指标都改善。

[Open3D 0.19.0 的 RGB-D odometry 教程](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html)提供颜色项与颜色、几何联合项的官方入口，说明了这类方法的背景。本教学代码使用自己的 point-to-plane 几何项、严格双线性导数和对应筛选，没有安装或调用 Open3D，也不等于完整复现其 Hybrid RGB-D Odometry 方法。

## 先写清当前帧到上一帧的方向

设当前帧为 k，上一帧为 k−1。当前有效深度反投影得到 $\mathbf p_{\mathrm{cur}}$；上一帧的对应点 $\mathbf q$、单位法向 $\mathbf n$、图像 $I_{\mathrm{prev}}$ 和内参 $\mathbf K$ 都属于目标相机坐标系。定义：

$$
\mathbf T_{k-1\leftarrow k}=
\begin{bmatrix}
\mathbf R&\mathbf t\\
\mathbf0^\mathsf T&1
\end{bmatrix},
\qquad
\mathbf p=\mathbf R\mathbf p_{\mathrm{cur}}+\mathbf t,\qquad
\pi(\mathbf p)=
\begin{bmatrix}
f_xX/Z+c_x\\
f_yY/Z+c_y
\end{bmatrix},
\qquad Z>0.
$$

$$
r_I(\mathbf T)=I_{\mathrm{prev}}(\pi(\mathbf p))-I_{\mathrm{cur}}(u_{\mathrm{cur}},v_{\mathrm{cur}}),
\qquad
r_g(\mathbf T)=\mathbf n^\mathsf T(\mathbf p-\mathbf q),
\qquad
\mathbf C_k=\mathbf C_{k-1}\mathbf T_{k-1\leftarrow k}.
$$

$\mathbf p_{\mathrm{cur}},\mathbf p,\mathbf q\in\mathbb R^3$ 以米计，$\mathbf n$ 为单位法向；$\pi(\mathbf p)\in\mathbb R^2$ 以像素计。$r_I$ 为归一化强度残差，$r_g$ 为米制几何残差，$\mathbf C_k$ 是相机到世界变换。

源强度固定在当前深度像素上，目标强度在上一帧的连续投影位置取样。不能交换两幅图像后继续沿用原来的变换方向和雅可比。脚本以单位相对变换开始每对配准；合成真值只用于生成受控输入和事后计算误差，不参与候选选择。

六维左扰动的排列为 $(\delta v_x,\delta v_y,\delta v_z,\delta\omega_x,\delta\omega_y,\delta\omega_z)^\mathsf T$，平移切向量单位为米、旋转为弧度：

$$
\delta\boldsymbol\xi=
\begin{bmatrix}
\delta\mathbf v\\
\delta\boldsymbol\omega
\end{bmatrix},
\qquad
\delta\boldsymbol\xi^\wedge=
\begin{bmatrix}
[\delta\boldsymbol\omega]_\times&\delta\mathbf v\\
\mathbf0^\mathsf T&0
\end{bmatrix},
\qquad
\mathbf T_{\mathrm{new}}=\exp(\delta\boldsymbol\xi^\wedge)\mathbf T.
$$

$$
\mathbf p_{\mathrm{new}}\approx\mathbf p+\delta\mathbf v+
\delta\boldsymbol\omega\times\mathbf p,\qquad
\frac{\partial\mathbf p_{\mathrm{new}}}{\partial\delta\boldsymbol\xi}
=
\begin{bmatrix}
\mathbf I_3&-[\mathbf p]_\times
\end{bmatrix}.
$$

$[\mathbf p]_\times\mathbf a=\mathbf p\times\mathbf a$ 定义反对称矩阵，其完整表达见上一章；六维增量按平移在前、旋转在后排列。

这里的 $\mathbf p$ 必须是已经变换到上一相机坐标系的点。有限 SE(3) 增量使用完整指数映射，不能把任意大小的 δv 直接当成指数映射后的平移列。变换方向、左右扰动与叉积符号的更完整讨论见 [ICP 雅可比与可观测性](10_rgbd_icp_observability.md)。

## 双线性插值采样的导数必须来自同一组四角

假设投影 $(u,v)$ 落在一个像素单元内，$a=u-\lfloor u\rfloor$、$b=v-\lfloor v\rfloor$，四角分别是左上 $I_{00}$、右上 $I_{10}$、左下 $I_{01}$、右下 $I_{11}$：

$$
a=u-\lfloor u\rfloor,\qquad b=v-\lfloor v\rfloor,\qquad
I(u,v)=(1-a)(1-b)I_{00}+a(1-b)I_{10}
+(1-a)bI_{01}+abI_{11}.
$$

$$
I_u=\frac{\partial I}{\partial u}
=(1-b)(I_{10}-I_{00})+b(I_{11}-I_{01}),\qquad
I_v=\frac{\partial I}{\partial v}
=(1-a)(I_{01}-I_{00})+a(I_{11}-I_{10}).
$$

$a,b\in[0,1)$ 是插值单元内的相对像素位置，四个 $I_{ij}$ 是同一个单元的固定角点强度。

这两个导数来自实际用于计算残差的插值函数。先用 Sobel 对整幅图像滤波、再插值梯度，是另一种离散近似；它通常不等于上述四角插值的严格导数。数值梯度检查要验证同一个目标函数，不能一边用双线性残差、一边把不同滤波器的梯度称作它的精确导数。

一个可手算的单元是：

$$
\begin{bmatrix}
I_{00}&I_{10}\\
I_{01}&I_{11}
\end{bmatrix}
=
\begin{bmatrix}
0.1&0.4\\
0.2&0.9
\end{bmatrix},
\qquad a=0.3,\quad b=0.6.
$$

$$
\begin{bmatrix}
(1-a)(1-b)&a(1-b)&(1-a)b&ab
\end{bmatrix}
=
\begin{bmatrix}
0.28&0.12&0.42&0.18
\end{bmatrix}.
$$

$$
I=0.028+0.048+0.084+0.162=0.322,\qquad
I_u=0.4\times0.3+0.6\times0.7=0.54,\qquad
I_v=0.7\times0.1+0.3\times0.5=0.22.
$$

该例已写入 [独立补充代数检查](../outputs/rgbd-photogeometric-cpu-controls-003/supplementary_math_controls.json)，不是为示意随意给出的数值。梯度的单位是归一化强度每像素。

## 从图像梯度连接到六维位姿

把采样导数、投影导数与左扰动导数相乘：

$$
\mathbf J_\pi=
\begin{bmatrix}
f_x/Z&0&-f_xX/Z^2\\
0&f_y/Z&-f_yY/Z^2
\end{bmatrix},\qquad
\mathbf h=
\mathbf J_\pi^\mathsf T
\begin{bmatrix}
I_u\\
I_v
\end{bmatrix}
=
\begin{bmatrix}
I_uf_x/Z\\
I_vf_y/Z\\
-(I_uf_xX+I_vf_yY)/Z^2
\end{bmatrix}.
$$

$$
\mathbf J_I=
\begin{bmatrix}
I_u&I_v
\end{bmatrix}\mathbf J_\pi
\begin{bmatrix}
\mathbf I_3&-[\mathbf p]_\times
\end{bmatrix}
=
\begin{bmatrix}
\mathbf h^\mathsf T&(\mathbf p\times\mathbf h)^\mathsf T
\end{bmatrix},
\qquad
\mathbf J_g=
\begin{bmatrix}
\mathbf n^\mathsf T&(\mathbf p\times\mathbf n)^\mathsf T
\end{bmatrix}.
$$

$\mathbf J_\pi\in\mathbb R^{2\times3}$ 以像素每米计，$\mathbf h\in\mathbb R^3$ 以强度每米计；两行位姿雅可比均有六列。

上式将图像梯度传到三维点后的向量记为列向量 $\mathbf h$； $\mathbf p\times\mathbf h$ 返回同样的三个旋转分量。把叉积写成 $\mathbf h\times\mathbf p$ 会反转旋转部分的符号。几何项对 $\mathbf n$、$\mathbf q$ 求导时把本轮对应固定，光度项则让同一个源点随 $\mathbf T$ 改变连续投影位置。

$$
\frac{\partial r_I}{\partial\delta\boldsymbol\xi}
=
\frac{\partial I_{\mathrm{prev}}}{\partial(u,v)}
\frac{\partial(u,v)}{\partial\mathbf p}
\frac{\partial\mathbf p}{\partial\delta\boldsymbol\xi}.
$$

原取样与雅可比构造示例保留在末尾实验附录。

这不是“颜色误差直接除以深度误差”的经验修正，而是一条明确的链式求导路径。近处与远处点的 $f_x/Z$ 不同，纹理方向也改变约束方向；颜色项提供多少信息不能仅用有效点数描述。

## 两种残差先按声明的尺度归一化

几何残差以米为单位，灰度残差则来自 $[0,1]$ 图像。本实现使用 $I=0.299R+0.587G+0.114B$ 得到灰度，没有反解 sRGB、相机响应或曝光变化。

固定配置如下。它们是工程尺度与阈值，没有经噪声统计标定：

| 配置 | 本实现数值 | 含义 |
|---|---:|---|
| $\sigma_g$ | 0.01 m | 几何残差的归一化尺度 |
| $\sigma_I$ | 0.1 | 灰度残差的归一化尺度 |
| $\lambda_I$ | 1 | 归一化光度项的相对系数 |
| Huber $\delta$ | 1 | 归一化残差的二次、线性分界 |
| 最终强度 RMSE 上限 | 0.1 | $\lambda_I$ 大于 0 时启用的拒绝门限 |

在同一组 $N$ 个联合对应上，目标函数为：

$$
s_{g,i}=\frac{r_{g,i}}{\sigma_g},\qquad
s_{I,i}=\frac{r_{I,i}}{\sigma_I},\qquad
\rho_\delta(s)=
\begin{cases}
\frac12s^2,&|s|\leq\delta,\\
\delta(|s|-\frac12\delta),&|s|>\delta.
\end{cases}
$$

$$
\mathcal L=
\frac1N\sum_{i=1}^{N}\rho_\delta(s_{g,i})
+\frac{\lambda_I}{N}\sum_{i=1}^{N}\rho_\delta(s_{I,i}).
$$

$\sigma_g>0$ 以米计，$\sigma_I>0$ 以强度计；$s_{g,i},s_{I,i}$、$\delta$ 与 $\lambda_I\geq0$ 均无量纲。

$\lambda_I=1$ 只表示这两个已经归一化的均值前系数相同，不表示原始米和强度可以直接等量比较，也不表示两种传感信息具有相同可信度。这里 $\sigma$ 的符号不意味着测得的标准差，Huber 权重不等于经过标定的逆噪声协方差。

迭代重加权最小二乘（Iteratively Reweighted Least Squares，IRLS）的单项权重是 $w=\min(1,\delta/|s|)$（非零残差），零残差按权重 1 处理。完整分段定义为：

$$
w(s)=
\begin{cases}
1,&|s|\leq\delta,\\
\delta/|s|,&|s|>\delta.
\end{cases}
$$
组合最小二乘系统的行分别为：

$$
A_{g,i,:}=\frac{\sqrt{w_{g,i}}}{\sigma_g}\mathbf J_{g,i},
\qquad
A_{I,i,:}=\frac{\sqrt{\lambda_Iw_{I,i}}}{\sigma_I}\mathbf J_{I,i},
$$

$$
b_{g,i}=-\sqrt{w_{g,i}}\,s_{g,i},\qquad
b_{I,i}=-\sqrt{\lambda_Iw_{I,i}}\,s_{I,i},\qquad
\mathbf A=
\begin{bmatrix}
\mathbf A_g\\
\mathbf A_I
\end{bmatrix},
\qquad
\mathbf b=
\begin{bmatrix}
\mathbf b_g\\
\mathbf b_I
\end{bmatrix}.
$$

$$
\delta\boldsymbol\xi^*=
\underset{\delta\boldsymbol\xi}{\operatorname{argmin}}
\|\mathbf A\delta\boldsymbol\xi-\mathbf b\|_2^2,
\qquad
(\mathbf A^\mathsf T\mathbf A)\delta\boldsymbol\xi
=\mathbf A^\mathsf T\mathbf b.
$$

$\mathbf A_g,\mathbf A_I\in\mathbb R^{N\times6}$，$\mathbf A\in\mathbb R^{2N\times6}$；各项权重在这次线性化中固定。

因为两项使用相同、固定的 $N$，求这个局部增量时共同省去 $1/\sqrt N$ 不改变解；若两项各自删点、分母不同，就不能无说明地沿用这一简化。实际代码通过列归一化后的 直接最小二乘求解 求解，避免显式求逆。它记录的条件数是列归一化系统最大、最小奇异值之比的平方，既不是原始 $\mathbf H$ 的条件数，也不是位姿协方差。

## 对应范围是目标函数的一部分

pair_correspondences() 先复用冻结 ICP 的深度、法向、法向夹角、三维距离和唯一目标像素筛选。之后才选出满足连续取样要求的联合支持。几何项也使用这个子集，因此添加光度采样门限已经改变了几何项看到的数据。

初次线性化时，脚本要求查询单元周围一圈支持有效：相对单元左上角，横纵偏移都取 $\{-1,0,1,2\}$，共检查 4×4 个目标深度、法向有效像素。这比“查询位置在图像内”严格，也比仅检查本单元四角严格。额外的一圈给局部候选留出移动余地，减少边界像素阻断所有下降方向的情况；它没有保证任意候选都合法。

每次线搜索固定这组源点、目标几何点、目标法向和源强度。候选 $\mathbf T$ 会改变投影位置与目标取样强度，但不能改变本次试探的分母。fixed_correspondence_trial() 要求全部固定源点仍在相机前方、双线性取样域内，且候选投影单元的四角均有有效目标几何支持。任何一点不满足就拒绝整个试探，不能删掉难点后凭更低的平均 loss 接受它。

单元内部的光度雅可比是精确的，整个配准过程仍只有分段可微性。跨单元时插值导数可能跳变；最近整数几何对应、深度边缘、遮挡和支持掩码也会变化。试探可以进入合法的邻接单元，代码会重新计算实际目标值，并没有把所有试探限制在原单元内。下一轮重新选择对应后，目标又可能改变，因此某次固定对应 loss 下降不意味着全程的动态对应 loss 必然单调。

三层配准从 40×30、80×60 到 160×120，默认最多迭代 15、12、10 次，三维对应距离门限为 0.10、0.07、0.05 m。粗到细有助于局部搜索，不能消除错误对应或曝光变化。达到迭代预算、增量很小、最终没有失败标记，也都不是准确性或全局收敛的证明。

## 纹理能补秩，不能保证亮度模型成立

纯 point-to-plane 的理想单平面只能约束沿法向的平移与两维倾斜；平面内的两维平移和绕法向的旋转缺失。若纹理提供足够独立的图像梯度方向，光度项可能补足这些方向。常量灰度的梯度为零，无法补秩；仅有一种边缘方向或重复纹理时，也不能从“有颜色”推出六维可靠观测。

控制 003 的世界平面在 $z=2.2\,\mathrm m$，纹理刻意设为：

$$
I_{\mathrm{world}}(X,Y)=0.4+0.05X+0.07Y+0.05XY.
$$

$X,Y$ 以米计时，一次项系数以米的倒数计，乘积项系数以平方米的倒数计，使灰度值无量纲。

目标相机正对平面，使参考图像中的纹理能由双线性插值精确表示。这个有利构造用来隔离数学实现与方向错误；它不包含真实相机的纹理频谱、非线性响应、噪声或复杂遮挡，不能把极小的恢复误差当作真实 RGB-D 精度预期。

| 合成控制 | 实际记录 | 能说明什么 |
|---|---|---|
| 同一纹理平面的几何项与联合项 | 初始 秩 从 3 变成 6 | 这一组纹理在当前对应上补足了局部方向 |
| 纹理平面恢复 | 接受；平移误差 1.67715×10⁻⁷ m，旋转误差 1.09873×10⁻⁵° | 在刻意匹配插值模型的 toy 上，运动方向与求解链路可恢复构造位姿 |
| 常量强度平面 | 光度雅可比严格为零，秩 3，候选被拒绝 | 没有纹理不能通过新增残差名称补足信息 |
| 当前图像统一增加 0.2 强度 | 最终 秩 6，但强度 RMSE 0.104513，候选被拒绝 | 满秩不能替代亮度一致性假设 |

亮度偏移控制还产生了 1.16658 m 的候选相对平移，超过预先声明的 0.20 m 门限；最终 point-to-plane RMSE 却只有约 4.31849 mm。它展示了一种具体失效：优化器会试图通过运动解释亮度变化，几何残差较小也未必能制止沿平面的错误运动。该候选因强度误差和相对平移两个门限被拒绝，不能将其写成已成功跟踪。

Huber 降低较大残差的权重，并不会自动识别所有系统性曝光偏差。这个实现不估计仿射亮度参数，没有辐射标定、运动补偿或动态物体模型。最终强度门限也是工程拒绝规则，在真实序列上的误拒绝与漏拒绝性质尚不由这些合成控制保证。

## 拒绝帧的位姿传播与零光度权重

register_pair() 保留候选矩阵、每层停止原因、对应数量、谱和全部失败标记。integrate_relative() 只在候选被接受时使用它；拒绝时相对更新取 identity，当前输出 c2w 与上一估计相同：

$$
\mathbf C_k=
\begin{cases}
\mathbf C_{k-1}\mathbf T_{\mathrm{candidate}},&\text{候选被接受},\\
\mathbf C_{k-1},&\text{候选被拒绝}.
\end{cases}
$$

传播上一估计保留了这一帧的位置与失败证据，不代表相机物理上静止，也不代表跟踪已经恢复。错误接受可能积累漂移，持续拒绝也可能让轨迹冻结，所以评测必须保留全部帧和失败情况。相邻矩阵的递推顺序不能反写；非交换控制中，正确世界点组合误差为 4.44×10⁻¹⁶，而错误顺序的矩阵最大差异约 0.00795165。

将新代码的 $\lambda_I$ 设为 0，会把光度项的优化权重置零，并关闭最终强度 RMSE 门限，但联合支持筛选仍然存在。4×4 取样支持、试探域限制、求解与最终检查路径并没有全部回到旧的 track_rgbd_icp.py。因此“新实现 λ0 对比 λ1”与“新实现对比旧 ICP”是不同对照，不能把 λ0 当成旧 ICP 的逐点或逐字节等价复现。


## 完整轨迹与失败帧一起评价

联合目标下降、单帧接受和整体轨迹更准是不同结论。拒绝帧仍保留在轨迹中，其传播规则会影响后续初值和参考。固定实际观测、尺度与接受规则，对比 ATE、RPE、终点误差和拒绝分布；不能删除拒绝帧后只报局部拟合。

## 把公式接到真实 RGB-D 时仍需解释的观测条件

[TUM 官方格式说明](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)约定深度单位、无效值与 RGB、深度的注册关系。这里的反投影要求 z-depth、内参及图像缩放遵循一致约定；空间注册也不意味着 RGB 与深度的时间戳完全相同，更不会自动消除曝光差异。相关的数据关联、深度范围与 coverage 分母见 [RGB-D 深度监督与覆盖率](07_rgbd_depth_coverage.md)。

光度残差依赖同一表面的可见性与近似亮度一致性。遮挡、投影到不同表面、深度边缘、自动曝光、非线性相机响应、滚动快门和运动模糊都可能破坏这些条件。双线性导数写对，只解决了“正在优化的插值函数如何变化”；观测模型是否对应真实采集过程，是另一个需要证据的问题。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

真实对照见[两条序列固定参数比较](../experiments/2026-10-09-photogeometric-real-sequences.md)与[完整序列 ICP](../experiments/2026-10-09-full-icp-sequences.md)，各按自己的输入与评价协议解释。

### 合成检查与源码证据的范围

[控制脚本](../scripts/check_rgbd_photogeometric_odometry.py)在非 identity 的基准位姿、非整数投影点上，对六个左扰动方向分别做中心有限差分。步长为 `1e-7`，投影点到单元边界最近约 0.19 像素，并检查正负扰动仍处于同一单元。控制 003 的光度雅可比最大差为 `7.61305e-10`，几何雅可比最大差为 `2.52197e-9`。这些数字是对应检查量的数值误差，不是轨迹误差、噪声标准差或运行速度。

[补充代数 JSON](../outputs/rgbd-photogeometric-cpu-controls-003/supplementary_math_controls.json)约 2.3 KB，包含四角手算、显式矩阵链式求导，以及覆盖 Huber 二次、线性两种分支的完整目标梯度检查。后者用独立标量插值与 SciPy 矩阵指数计算有限差分，六维目标梯度的最大差为 `8.24465e-8`。[源码与保存结果复核](../outputs/rgbd-photogeometric-cpu-controls-003/root_math_source_review.json)约 1.7 KB，记录了变换方向、归一化分母、支持限制与 λ0 边界。两份小文件可直接下载检查，不需要读取图像或模型权重。

对应的 [源码清单](../outputs/rgbd-photogeometric-cpu-controls-003/source_manifest.json)固定了版本关系：

```text
rgbd_photogeometric_odometry.py
80160ea921a40363b9143d6c2a11c1954063aa5fe610397e4f8dcc336261b125

check_rgbd_photogeometric_odometry.py
e2f71002e0b522f9419fd024f06c165101f04ba2c35da9ee6e23faca3cc22037

控制 003 完整 metrics.json 的 SHA256
c227a5ca78b5b61ab0e87585606b9ac0cb380f542aea7498e8a26f5169237014
```

这些检查支持“保存的实现与声明的局部公式一致，并能拒绝列出的退化与亮度失效控制”。它们没有证明真实场景中对应始终正确、失败门限经过标定，或新目标优于纯几何 ICP、GS tracking 与其他 RGB-D 方法。

### 真实轨迹为什么需要同时看平均误差与拒绝帧

固定四组 CPU 实验使用两条 TUM 序列各 100 个关联帧（200–299），每组 99 对配准；只切换新实现的光度系数 0 与 1。两组共用实际 RGB-D、内参、单位相对初值和初始支持集合，但优化过程中后续对应可能不同，不能假定它们始终优化同一个固定点集。

| 序列 | 光度系数 | SE3 ATE / mm | 相邻平移 RPE / mm | 拒绝帧 / 99 |
|---|---:|---:|---:|---:|
| desk | 0 | 37.428 | 5.286 | 0 |
| desk | 1 | 27.315 | 8.992 | 3 |
| xyz | 0 | 22.678 | 3.990 | 0 |
| xyz | 1 | 17.139 | 6.855 | 1 |

这组观察没有“全指标更好”的结论。ATE 在整个片段上拟合一次无尺度刚体对齐，平移 RPE 衡量逐对局部运动；两者会对不同误差结构作出不同评价。desk 开启光度项后的末帧原始位置误差反而更大，说明较低的片段平均误差也不保证最终漂移更小。

四个拒绝帧原样保留，没有从评测分母中删除。它们触发的是最细层线搜索无可接受下降的标记；这个标记不等于几何 rank 不足，也不能凭它直接诊断曝光问题。拒绝时输出上一位姿，随后如果仍只匹配上一张传感器图像，就可能漏掉被拒绝那一步的运动。区分“这一对被拒绝”“累计轨迹保留下来”和“之后恢复了漏掉的运动”，是理解跟踪失败的重要部分。

完整配置、六条轨迹图（含旧 ICP 只读参考）、实际失败诊断和独立复核见 [真实序列对照](../experiments/2026-10-09-photogeometric-real-sequences.md)。这些是 CPU 两帧里程计结果，没有回环、全局优化或高斯地图，不构成完整 GS-SLAM。


### 从理论正文保留的实现片段

以下为本轮公式规范前的实现示例，保留原片段供核查，不是新增实验结果。

```python
# 伪代码：当前点已用 T 变换为上一相机坐标 p；省略筛选与退化检查。
uv = project(p, K_previous)
sample, grad_uv, inside = bilinear(gray_previous, uv)
r_photo = sample - fixed_current_intensity

g = grad_uv @ J_project_for_this_point
J_photo = np.r_[g, np.cross(p, g)]
J_geo = np.r_[n, np.cross(p, n)]

# 求出左切空间增量后，更新在 T 左侧。
T_trial = se3_exp(delta) @ T
```

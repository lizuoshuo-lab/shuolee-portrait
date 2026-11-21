---
id: "gslab-k06"
slug: "se3-fixed-map-photometric-pose-optimization"
title: "06 固定高斯地图的 SE(3) 位姿优化"
order: 6
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/_wrapper.py"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html"
---

# 06 固定地图时，图像残差如何更新相机

高斯地图优化把已知相机的多视图观测约束送到地图参数；固定地图的相机跟踪则把当前图像残差送到位姿。高斯数量、中心、协方差、颜色与不透明度保持不变，未知量只有三维平移与三维旋转。[固定地图参考实现](../scripts/track_fixed_map.py)和[原始实验记录](../experiments/2026-10-09-fixed-map-pose.md)展示这一链路。估计一张图像的相机，与具有初始化、失败检测和恢复能力的连续跟踪系统，是不同的能力范围。

## 先明确位姿方向和扰动方向

本文采用列向量与世界到相机的变换。世界点、相机点分别为 $\mathbf x_w,\mathbf x_c\in\mathbb R^3$，旋转为 $\mathbf R_{cw}\in SO(3)$，平移为 $\mathbf t_{cw}\in\mathbb R^3$。这里的 $SO(3)$ 是三维旋转群，由正交且行列式为 $1$ 的三阶矩阵组成。点变换和投影中心满足：

$$
\mathbf x_c=\mathbf R_{cw}\mathbf x_w+\mathbf t_{cw},
\qquad
\mathbf c_w=-\mathbf R_{cw}^{\mathsf T}\mathbf t_{cw}.
$$

$\mathbf c_w$ 是相机投影中心在世界坐标系中的位置；$\mathbf t_{cw}$ 是世界原点在相机坐标系中的坐标。点和位置的单位为米，或尚未确定物理尺度的场景长度单位。比较两个世界到相机的平移列，通常不等于比较两个相机中心。

相对于固定初始位姿 $\mathbf T_0$，采用左乘扰动：

$$
\mathbf T(\boldsymbol\xi)=
\exp(\widehat{\boldsymbol\xi})\mathbf T_0,
\qquad
\boldsymbol\xi=
\begin{bmatrix}
\boldsymbol\rho\\
\boldsymbol\omega
\end{bmatrix}\in\mathbb R^6.
$$

$\boldsymbol\rho\in\mathbb R^3$ 是平移切向量，单位为长度；$\boldsymbol\omega\in\mathbb R^3$ 是旋转向量，模长是旋转角，单位为弧度。左乘扰动表达在变换输出端的相机坐标系；右乘扰动表达在输入端，不能只交换乘法顺序而保留原来的变量含义。

这里每轮优化同一个相对固定初值的六维向量。它并不等同于每轮把增量继续左乘到上一轮位姿，解释与保存状态时必须区分。

## 为什么使用 SE(3) 指数映射

三维特殊欧氏群 $SE(3)$ 表示旋转和平移组成的三维刚体变换。直接优化四阶矩阵的全部元素，无法自动保证旋转正交；六维切向量通过指数映射保留刚体结构。令 $\boldsymbol\omega=(\omega_x,\omega_y,\omega_z)^{\mathsf T}$：

$$
\boldsymbol\Omega=\widehat{\boldsymbol\omega}=
\begin{bmatrix}
0&-\omega_z&\omega_y\\
\omega_z&0&-\omega_x\\
-\omega_y&\omega_x&0
\end{bmatrix},
\qquad
\widehat{\boldsymbol\xi}=
\begin{bmatrix}
0&-\omega_z&\omega_y&\rho_x\\
\omega_z&0&-\omega_x&\rho_y\\
-\omega_y&\omega_x&0&\rho_z\\
0&0&0&0
\end{bmatrix}.
$$

$\boldsymbol\Omega$ 是三阶反对称矩阵。令 $\theta=\|\boldsymbol\omega\|_2$，在 $\theta>0$ 时：

$$
\mathbf R_\xi=
\mathbf I_3+
\frac{\sin\theta}{\theta}\boldsymbol\Omega+
\frac{1-\cos\theta}{\theta^2}\boldsymbol\Omega^2,
\qquad
\mathbf V=
\mathbf I_3+
\frac{1-\cos\theta}{\theta^2}\boldsymbol\Omega+
\frac{\theta-\sin\theta}{\theta^3}\boldsymbol\Omega^2.
$$

$$
\exp(\widehat{\boldsymbol\xi})=
\begin{bmatrix}
\mathbf R_\xi&\mathbf V\boldsymbol\rho\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
$$

$\mathbf R_\xi,\mathbf V\in\mathbb R^{3\times3}$ 无量纲，平移列 $\mathbf V\boldsymbol\rho$ 具有长度单位。旋转和平移在指数映射中耦合，$\boldsymbol\rho$ 通常不等于最终平移列。当 $\theta\to0$ 时，系数取连续极限，$\mathbf R_\xi,\mathbf V\to\mathbf I_3$。

参考实现采用矩阵指数，内部旋转统一使用弧度。原记录中，CPU 零点与非零点梯度检查通过，逆变换与正交性误差约 $10^{-15}$，实际单精度旋转正交性误差约 $10^{-7}$；这些数值说明有限精度约束，不能证明位姿准确。

## 固定地图的光度损失

设固定地图为 $\mathcal G$，相机内参为 $\mathbf K$，观测图像为 $\mathbf I\in[0,1]^{H\times W\times3}$，渲染图像为 $\widehat{\mathbf I}$。用 $\mathcal R$ 表示将地图与相机映射为图像的数学渲染算子：

$$
\widehat{\mathbf I}(\boldsymbol\xi)=
\mathcal R\!\left(\mathcal G,\exp(\widehat{\boldsymbol\xi})\mathbf T_0,\mathbf K\right).
$$

$$
\mathcal L_{\mathrm{RGB}}(\boldsymbol\xi)=
\frac{1}{3HW}
\sum_{u=1}^{W}\sum_{v=1}^{H}\sum_{k=1}^{3}
\left(\widehat I_{uvk}(\boldsymbol\xi)-I_{uvk}\right)^2,
\qquad
\boldsymbol\xi^\star=
\underset{\boldsymbol\xi}{\operatorname{argmin}}\,
\mathcal L_{\mathrm{RGB}}(\boldsymbol\xi).
$$

$u,v$ 是像素索引，$k$ 是颜色通道；归一化颜色及该损失均无量纲。梯度通过图像合成、投影高斯、相机变换与指数映射传回六个位姿量，地图参数不参与更新。平移与旋转使用不同单位，不能仅因六个分量组成一个向量，就认为相同数值的学习率给出相同物理步幅。

原真实 RGB 实验没有真值位姿损失。真值仅用于创建明确扰动的初始相机与事后评估，属于受控的局部恢复条件；在线系统仍需用上一帧、恒速预测或其他里程计提供初值，并检验这些条件下的收敛范围。

## 均值与协方差都必须转到相机坐标

高斯中心 $\boldsymbol\mu_w\in\mathbb R^3$ 和世界协方差 $\boldsymbol\Sigma_w\in\mathbb R^{3\times3}$ 同时受相机姿态影响：

$$
\boldsymbol\mu_c=
\mathbf R_{cw}\boldsymbol\mu_w+\mathbf t_{cw},
\qquad
\boldsymbol\Sigma_c=
\mathbf R_{cw}\boldsymbol\Sigma_w\mathbf R_{cw}^{\mathsf T}.
$$

均值具有长度单位，协方差具有长度平方单位。只变换中心而保持各向异性协方差不变，会给出错误的二维投影形状和旋转梯度。

可以由渲染器统一执行世界到相机变换，也可以先将高斯变到相机坐标，再使用单位相机变换；两者不能重复执行。固定版本在原地图与初始位姿下，两条路线的最大像素差约为 $1.70\times10^{-6}$。这是渲染路径的一致性检查，不能解释为地图几何精度。

## 六维梯度怎样独立检查

对第 $i$ 个切向量分量，令 $\mathbf e_i\in\mathbb R^6$ 为标准基，$\varepsilon_i>0$ 为该分量的差分步长：

$$
g_i^{\mathrm{FD}}=
\frac{
\mathcal L(\boldsymbol\xi+\varepsilon_i\mathbf e_i)-
\mathcal L(\boldsymbol\xi-\varepsilon_i\mathbf e_i)
}{2\varepsilon_i}
\approx
\frac{\partial\mathcal L}{\partial\xi_i}.
$$

平移分量的 $\varepsilon_i$ 用米或场景长度单位，旋转分量用弧度；相应梯度单位分别为损失每单位长度、损失每弧度。应检查多档步长，并保持同一地图、观测、内参、参数化和目标定义。

单精度舍入、排序、裁剪及硬掩码会影响比较。零梯度也可能来自对称、无覆盖或缺少纹理；某个状态检查通过，只证明该状态、该目标的局部求导一致。

## 低图像损失为什么仍可能定位错误

跟踪求解的是当前地图 $\mathcal G$ 条件下的相机：

$$
\mathbf T^\star=
\underset{\mathbf T\in SE(3)}{\operatorname{argmin}}\,
\mathcal L(\mathbf T;\mathcal G).
$$

地图中错误的颜色、表面位置、漂浮高斯或覆盖空洞，可能使相机移动来补偿地图误差。因此真实相机未必是该条件目标的最低点。初值偏差、局部极小值、弱可观测性与未充分收敛，也会造成剩余误差，不能只凭一次结果决定原因。

必须分开记录残差变化与独立位姿误差。RGB-D 几何、多尺度、运动预测和失败恢复分别处理不同问题，增加一个目标项不自动保证准确。未经噪声模型和可观测性校验的损失 Hessian，也不能直接称为已标定的位姿协方差。

## 合成控制与真实观测的区别

同一地图在已知相机下的渲染作为目标，可以隔离外观模型不匹配，这是合成控制。真实 RGB 还包含噪声、光照、遮挡和地图误差。

未确定物理尺度的场景只能报告场景长度单位；例如 Lego 的 $0.015$ 个场景长度单位不能直接写成 $1.5$ 厘米。米制数据仍需核对深度单位、内参和时间关联。单帧位置或角度误差不能替代整段轨迹的绝对轨迹误差与相对位姿误差。

## 保存状态必须与打分状态一致

令 $\boldsymbol\xi^{(j)}$ 为第 $j$ 次迭代的相机参数。依据传感器目标选中的最佳候选为：

$$
j^\star=
\underset{j}{\operatorname{argmin}}\,
\mathcal L_{\mathrm{sensor}}\!\left(\boldsymbol\xi^{(j)}\right).
$$

最佳候选可能不是最后一步，也不保证真值误差最小。应分别保存两者的切向量、变换和损失，并从切向量重建矩阵核对。若接受前修正了状态结构，必须对修正后的状态重新计算损失；在线选择候选不能依赖真值误差。

## SE(3) 数值约束与解析刚体逆

合法刚体变换及其逆为：

$$
\mathbf T=
\begin{bmatrix}
\mathbf R&\mathbf t\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix},
\qquad
\mathbf R^{\mathsf T}\mathbf R=\mathbf I_3,
\qquad
\det\mathbf R=1.
$$

$$
\mathbf T^{-1}=
\begin{bmatrix}
\mathbf R^{\mathsf T}&-\mathbf R^{\mathsf T}\mathbf t\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
$$

解析逆要求 $\mathbf R$ 为合法旋转。如果它已明显不正交，转置不能代替一般矩阵逆；应恢复旋转约束，或明确使用一般逆作诊断。齐次矩阵的最后一行为三个零与一个一，不应被当成未知自由度优化。

例如相邻帧间隔相等、左乘相对运动恒定时，恒速预测为：

$$
\mathbf T_t^{\mathrm{pred}}=
\left(\mathbf T_{t-1}\mathbf T_{t-2}^{-1}\right)\mathbf T_{t-1}.
$$

相乘、指数映射及求逆的单精度误差会进入后续状态。检查应包括真正用于几何计算的相机到世界矩阵，而不能只检查保存的世界到相机矩阵。

### 将接受的旋转投影到合法旋转群

设数值旋转矩阵的奇异值分解为：

$$
\mathbf R=
\mathbf U
\begin{bmatrix}
\sigma_1&0&0\\
0&\sigma_2&0\\
0&0&\sigma_3
\end{bmatrix}
\mathbf V^{\mathsf T},
\qquad
\delta=\operatorname{sign}\!\left(\det(\mathbf U\mathbf V^{\mathsf T})\right).
$$

$\mathbf U,\mathbf V$ 为三阶正交矩阵，$\sigma_1,\sigma_2,\sigma_3$ 是按降序排列的非负奇异值，无量纲。最近的合法旋转可写为：

$$
\mathbf R_{\mathrm{proj}}=
\mathbf U
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&\delta
\end{bmatrix}
\mathbf V^{\mathsf T},
\qquad
\mathbf T_{\mathrm{proj}}=
\begin{bmatrix}
\mathbf R_{\mathrm{proj}}&\mathbf t_{\mathrm{keep}}\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
$$

若处理相机到世界变换，$\mathbf t_{\mathrm{keep}}$ 表示保留相机中心；若处理世界到相机变换，它表示保留原平移列。两种选择可能产生不同的数值位置修正，应说明约定。

这一投影可以用于接受或保存状态的结构修正，不依赖真值。若不对投影求导，应将它置于优化路径之外；否则会改变或截断所解释的梯度。投影可能改变渲染，必须重算该状态的传感器损失、平移与旋转步长，并重新应用接受条件。结构修正不自动解决地图偏差、局部极小值或时间误差，收益需单独验证。

## 有限差分比较的是哪个函数

设 $V$ 为有效传感器像素集合，$A(\mathbf p;\mathbf T)$ 为地图在像素 $\mathbf p$ 的累计不透明度，$\tau$ 为覆盖阈值。深度目标可能只在以下集合内计算：

$$
M(\mathbf T)=
\left\{
\mathbf p\in V\mid A(\mathbf p;\mathbf T)>\tau
\right\}.
$$

在当前阈值分支内，自动微分对当前像素集合的连续残差求导；中心差分却可能跨越门槛，比较两个不同集合：

$$
M_+=M(\mathbf T(\boldsymbol\xi+\varepsilon_i\mathbf e_i)),
\qquad
M_-=M(\mathbf T(\boldsymbol\xi-\varepsilon_i\mathbf e_i)).
$$

$$
|M_+|=|M_-|
\quad\not\Rightarrow\quad
M_+=M_-.
$$

$|M|$ 表示像素数量，无量纲；集合对称差 $M_+\triangle M_-$ 才指出哪些像素切换。集合大小相同也可能由不同像素组成，均值分母相同不意味着比较相同残差。

复核需绑定同一地图状态、相机初值、数据、内参、目标定义与门槛。仅有相同源码、种子和最终模型，不足以重建中途梯度；新状态检查通过也不能自动解释旧失败。

## 核心理解自测

- 为什么世界到相机的平移列通常不是相机中心的距离误差？
- 左乘与右乘刚体扰动为何不能不加说明地交换？
- 只旋转高斯中心、不旋转协方差，会影响哪些像素和梯度？
- 为什么图像损失大幅下降，平移误差仍可能较大？
- 真值扰动初始化证明了什么，还缺少哪些在线跟踪证据？
- 为什么 $0.015$ 个场景长度单位不能直接称为 $1.5$ 厘米？
- 为什么代数逆闭环通过，仍需要独立检查旋转约束和齐次矩阵底行？
- 将旋转投影后，为什么必须重算传感器损失与接受条件？
- 当几何初值已经较好时，进一步降低地图渲染残差为何仍可能使相机偏离？
- 为什么正负扰动的覆盖数量相同，不能证明差分使用相同像素集合？

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

### 5. 六维梯度的独立验证

只看到 loss 下降不能证明位姿旋转正确，也可能由错误的自由矩阵变形或地图参数更新造成。必要证据包括：

| 检查 | 检查的问题 | 本地证据 |
|---|---|---|
| SE3 刚性和 gradcheck | 旋转是否合法，零初值是否有梯度 | CPU 零点/非零点通过，逆与正交误差约 1e-15 |
| 参数列表 | 优化器是否偷偷改地图 | 只有 6 个 pose 标量，地图 trainable 参数为 0 |
| 六维解析梯度 | 相机梯度是否有限且覆盖各方向 | 初始 6 维均非零，所有迭代梯度有限 |
| 中心有限差分 | CUDA backward 方向是否符合独立扰动 | 余弦 0.999930，相对 L2 差 0.06944 |
| 两种渲染参数化 | 均值/协方差变换是否一致 | 最大像素差约 1.70e-6 |

中心差分为 `[L(xi+epsilon e_i)-L(xi-epsilon e_i)]/(2epsilon)`，本次 epsilon=1e-4。它仍受 float32、可见性、排序与微小高斯影响；对某个初始化通过，不意味着整个场景的梯度都有相同误差。梯度为零也可能源于对称性、无可见高斯或缺少纹理，必须结合当前场景判断。

### 6. 图像更像，不等于位姿已经准确

首跑用真实目标、300 步优化：RGB MSE 从 0.05405 到 0.002994，旋转误差从 5° 到 1.537°，但平移仅从 0.11572 到 0.10104 scene units。看上去图像恢复很多，平移却仍不准确。

保持相同地图和初始化，第二轮同时增加到 1500 步并提高平移学习率：MSE 到 0.001887，平移误差到 0.014856 scene units，旋转到 0.234°。这个对照说明梯度链路能进一步恢复位姿；由于同时改变两个因素，不能作为“只增加步数”的单因素消融。

固定地图在 GT 相机处也有 MSE 0.001856，建图误差并未消失。错误颜色、漂浮高斯、缺失纹理或遮挡会让相机移动以补偿地图，造成 photometric 最优与几何真值不一致。一次实验的剩余偏差还可能来自收敛不充分，不能未经控制全部归因于地图误差。

单帧损失可能沿某些方向约束很弱，例如纹理匮乏、平面或重复纹理。初始化超出局部收敛范围也会失败。更完整的 tracking 需要 RGB-D 几何约束、有效区域与遮挡处理、多尺度、初值预测、置信度与失败恢复；每项需要独立验证，不能只把更多 loss 加进来就声称解决。

**TUM 已观察到更低 photometric loss 对应非 GT 相机。** 固定已知真值位姿建好的 30000 高斯地图，用未参加地图训练的 frame 62 真实 RGB 恢复位姿，初始平移误差 0.037415 m、旋转 3°。1500 步后按 RGB loss 选中相机的平移误差为 0.009148 m、旋转误差 0.362340°；RGB MSE 降到 0.004756，低于 GT 相机处的 0.005746。相机通过移动降低了地图与观测之间的残差，GT 并非这张图像整图 RGB MSE 的最低点。这一例子直接说明“损失更低”和“位姿更接近真值”不是同一判断。

该 TUM 结果仍采用 GT 扰动初值与固定 GT 地图，仅有一个目标帧。地图训练用了 RGB-D，此次相机优化只用 RGB，不能把结果概括为完整 RGB-D tracking。实际结果、梯度、split 与时间见 [固定地图实测报告](../experiments/2026-10-09-fixed-map-pose.md)。

### 7. 合成 control 的正确含义与单位

把同一地图在 GT 相机处的渲染图设为目标，可以排除真实 RGB 与地图外观不匹配。这里 GT 用于创建目标，必须标成合成 control。当前 control 的 GT loss 为 0，但 1500 步后仍有 6.27e-5 残差及 0.387° 旋转误差，说明当前配置尚未完全收敛；不能拿这个 control 当真实数据 tracking 成绩。

Lego 位移使用未标定的 scene units，不能写“厘米级定位”。TUM 的深度和位姿才采用其公开标定的米制约定，复用时要同时检查 RGB/深度时间关联、内参缩放、坐标轴与 mask。单帧平移误差、旋转角不是 ATE/RPE：后两者需要整段轨迹及相应关联/对齐协议。

### 8. 保存最优与最后状态

按 RGB loss 选出的最佳位姿可能不是最后一轮，也不保证 GT 误差最小。不要用真值误差选 checkpoint，再报告为算法自身最佳状态。

历史 pose-002 选中第 1461 步，但旧 checkpoint 的 `translation_delta/rotation_delta` 对应最后一步，`estimated_w2c` 对应最优 RGB 状态。两者最大矩阵差 6.68e-6，最终指标与 `estimated_w2c` 一致。当前脚本显式分别保存 best 和 last 切向量；旧产物保留并记录差异，读取时使用明确的位姿字段。

新 TUM 运行选中第 973 步，best 与 last 位姿分别保存。CPU 独立检查从 best 切向量重建选中 w2c 的最大差为 5.96e-8，从 last 切向量重建最后 w2c 的最大差为 1.19e-7，确认当前保存实现与评估状态一致。

#### 已验证的 CPU 证据与根因边界

[独立几何诊断](../outputs/independent_se3_numerical_diagnostic.json)使用与当前 TUM 初始化 helper 相同的计算链：先用 c2w 的前 3 行作 affine 点变换，再用完整 `np.linalg.inv(c2w)` 的前 3 行变回相机。

- R 略微非正交、但 bottom row 精确且 R 可逆时，这条代数闭环仍自洽。人为构造 R 正交误差约 2e-4，闭环误差仍约 4.6e-16 米、重投影误差约 1.1e-13 像素。
- 已有 20 帧 checkpoint 的底行全为 identity，虽然旋转正交误差最高约 2.3e-5，类似点闭环仍约 2e-15 米。**R 非正交并不单独解释当前一般 inverse helper 的检查失败**；物理刚性是否正确和这项代数检查是否自洽是两个判断。
- 仅把底行末元素从 1 改为 `1+1e-7`，完整矩阵就不再与“只取前 3 行的 affine 变换”遵循同一约定，示例闭环误差约 7.1e-8 米，超过 helper 阈值。接近零的 z、严重病态矩阵或很大的数值范围也可能放大重投影误差，需要查看失败时的实际输入才能区分。

[固定 seed 8 的 CPU 100 步压力实验](../outputs/independent_se3_numerical_seed8.json)保存了完整切向量和逐步误差：单次 float32 `matrix_exp` 的底行始终正确；第 88 步候选 w2c 的底行仍正确，但通用逆生成的 c2w 首次出现约 5.96e-8 底行误差，后续恒速反馈累计到约 1.07e-6。这说明一般逆与反馈链存在可复现风险，不能把观察直接归因于单次指数映射。

这个 CPU 压力序列与实际 RGB-D 输入、GPU 运算和估计轨迹不同。实际 100 帧旧原型在第 81 帧增图时触发几何检查失败，但没有保存那一帧的候选矩阵；**该 GPU 失败的具体根因尚未确认**。不能把 CPU 第 88 步的模拟风险写成“已经定位 GPU 第 81 帧根因”。

新 V2 原型已实际完成五项运行，并经过 [独立 CPU 审核汇总](../outputs/v2-independent-audit-summary.json)：两项 20 帧对照、一项原片段 100 帧深度权重 1.0，以及非重叠 frame 200–299 的两项 100 帧对照。所有保存的 canonical 位姿底行误差为 0，旋转正交误差最高约 1.05e-7；因果 progress/last-checkpoint 未包含实际 GT pose，最终评价 checkpoint 的 GT 字段单独标明 evaluation-only。

数值结构改善没有自动解决轨迹漂移。基线 20 帧 SE3 ATE 从 V1 的 0.097522 米变为 V2 的 0.097715 米，深度权重 1.0 对照从 0.014353 米变为 0.014323 米，两者与原短片段结果基本相近。非重叠片段的深度权重 1.0 对照虽然对齐 ATE 为 0.083753 米，raw anchored RMSE 仍为 0.388756 米，末帧偏差为 0.532762 米。结构正确、梯度有限与定位准确必须分别验证；旧 GPU 第 81 帧异常的原始矩阵仍缺失，因此原始异常的具体根因没有因新运行成功而自动得到证明。

本地 V3 用独立深度 ICP 累积位姿作为每帧初值，比较“相机直接用于 GS 建图”与“再优化高斯位姿”。100 帧配置仅切换位姿优化模式，锚定位置 RMSE 从 0.06410 m 变为 0.22467 m，SE3 ATE 从 0.03063 m 变为 0.08726 m。后者首个六维检查通过且没有失败标记，但 99 帧的传感器目标均不增加，其中 87 帧候选相机的真值平移误差比同帧 ICP 初值更大。真值只在事后分析使用。

这里没有按真实相机挑选“最好的一步”，也没有把 GS 修正反馈到独立 ICP 累积链；因此能分别观察几何初值和相机 refinement。两条流程后来学到的地图仍不同，不能把整段结果当作同一地图下的单帧损失曲面。20 帧 refinement 另有一个有限差分失败和位姿回退，也不能混入无故障对照。详见[ICP 初始化受控实验](../experiments/2026-10-09-icp-initialization.md)。

真实 TUM 状态的受限[五档步长诊断](../experiments/2026-10-09-pose-gradient-diagnostic.md)显示，固定零位姿覆盖集合后，差分梯度在每档都更接近解析梯度；但剩余差异非零，原 20 帧首 probe 的失败也没有复现。这支持检查覆盖不连续，不构成后端 bug 或历史唯一根因的证明。只读诊断的固定 mask 没有进入训练。

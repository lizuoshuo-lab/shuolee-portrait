---
id: "gslab-k11"
slug: "geometric-pose-prior-units-gauge"
title: "11 位姿先验：相机中心、米与弧度、旋转近似和 gauge"
order: 11
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html"
---

# 几何初值较好，为什么还要限制地图位姿优化

ICP 用相邻深度建立几何初值，GS tracking 则在当前高斯地图上优化图像与深度损失。地图并非无误差：覆盖、颜色、深度、遮挡或早期位姿偏差，都可能让传感器损失的最小点偏离真实相机。更好的初值不保证继续优化后更接近真值。

位姿先验把候选约束在一个固定参考附近，限制地图误差诱导的过大修正。参考本身也可能错误；先验是额外假设，权重越大越接近参考，不能代替新的观测信息或独立精度评价。

## 世界到相机变换与相机中心

采用列向量与世界到相机变换（world-to-camera，w2c），相机坐标系遵循 OpenCV 约定：

$$
\mathbf W=
\begin{bmatrix}
\mathbf R_{cw}&\mathbf t_{cw}\\
\mathbf0^\mathsf T&1
\end{bmatrix},\qquad
\mathbf p_c=\mathbf R_{cw}\mathbf p_w+\mathbf t_{cw}.
$$

相机中心 $\mathbf C$ 是相机原点在世界坐标中的位置，满足 $\mathbf R_{cw}\mathbf C+\mathbf t_{cw}=\mathbf0$。因此：

$$
\mathbf W^{-1}=
\begin{bmatrix}
\mathbf R_{cw}^\mathsf T&-\mathbf R_{cw}^\mathsf T\mathbf t_{cw}\\
\mathbf0^\mathsf T&1
\end{bmatrix},\qquad
\mathbf C=-\mathbf R_{cw}^\mathsf T\mathbf t_{cw},\qquad
\mathbf R=\mathbf R_{cw}^\mathsf T.
$$

**w2c 的平移 $\mathbf t_{cw}$ 不是相机中心。** 两个相机的旋转不同，用 $\|\mathbf t_{cw}-\mathbf t_{cw,\mathrm{ref}}\|_2$ 代替世界坐标中的中心距离，会混入坐标系差异。这个错误在单位旋转的简单例子里可能被隐藏。

CPU 负控制用了非单位旋转与非零平移：正确中心距离平方为 $0.01710835\,\mathrm m^2$，错误的 w2c 平移距离平方为 $0.00719600\,\mathrm m^2$。共同改变世界坐标后，错误的归一化平移项从 2.8784 变成 54.2638，暴露了它不具备所需的共同坐标变换不变性。

解析刚体逆依赖 $\mathbf R^\mathsf T\mathbf R=\mathbf I_3$、$\det(\mathbf R)=1$ 和底行 $\begin{bmatrix}0&0&0&1\end{bmatrix}$。它不是任意 4×4 矩阵的一般逆。helper 提供独立的刚体边界检查，不在损失计算图里用 奇异值分解（Singular Value Decomposition，SVD） 修改候选相机；相机数值结构见[06 位姿优化](06_pose_optimization.md)。

## 先验公式与工程尺度

对 candidate/reference 两个 w2c 取解析刚体逆，得到世界相机中心 $\mathbf C$、$\mathbf C_{\mathrm{ref}}$ 和 c2w 旋转 $\mathbf R$、$\mathbf R_{\mathrm{ref}}$。先验为：

$$
\mathcal L_{\mathrm{prior}}
=\lambda\left[
\frac{\|\mathbf C-\mathbf C_{\mathrm{ref}}\|_2^2}{\sigma_t^2}
+\frac{\|\mathbf R-\mathbf R_{\mathrm{ref}}\|_F^2}{2\sigma_r^2}
\right].
$$

$\mathbf C,\mathbf C_{\mathrm{ref}}\in\mathbb R^3$ 为世界坐标中的相机中心；$\mathbf R,\mathbf R_{\mathrm{ref}}\in SO(3)$ 为相机到世界旋转。$\|\mathbf M\|_F^2=\sum_{i,j}M_{ij}^2$ 定义矩阵 Frobenius 范数的平方。

$\sigma_t>0$ 是平移工程尺度，$\sigma_r>0$ 是旋转工程尺度。$\sigma_t$ 的单位是米，$\sigma_r$ 的单位是弧度。$\lambda$ 为非负、无量纲的工程权重。归一化后的平移与旋转项也无量纲，但它们与传感器损失之间的相对作用仍取决于 $\lambda$ 和传感器目标的定义。

代码接口明确命名 sigma_t_m、sigma_r_rad。5° 必须先换成 $5\pi/180\approx0.08726646\,\mathrm{rad}$，不能直接传入 5。毫米也必须先换成米。这里讨论 TUM 的米制相机；未标定的 TinyNeRF 场景单位不能直接套用以米命名的尺度。

CPU 检查中的 $\sigma_t=0.05\,\mathrm m$、$\sigma_r=5^\circ$ 只是用于验证数学的工程值，没有被选为训练配置。它们不是“允许误差以内完全不罚”的硬门限：该先验随偏移连续增长。0.1m 的纯中心偏移在该 $\sigma_t$ 下贡献 $(0.1/0.05)^2=4$。

### 权重和尺度怎样共同限制修正

保持传感器目标不变时，平移项的系数是 $\lambda/\sigma_t^2$，旋转项的系数是 $\lambda/(2\sigma_r^2)$。因此权重必须连同尺度一起说明：只将 $\lambda$ 乘 4，与保持 $\lambda$ 不变、同时将两种 σ 除以 2，是两种等价的先验增强方式；若想改变参数但保持原目标不变，则可将 $\lambda$ 乘 4、同时将两种 σ 乘 2。

如果初值作为候选保留，初始先验为零，且选中的总目标不高于初值，那么非负传感器目标给出：

$$
\lambda\frac{\|\mathbf C-\mathbf C_{\mathrm{ref}}\|_2^2}{\sigma_t^2}
\leq\mathcal L_{\mathrm{sensor}}(\mathrm{initial}),
\qquad
\|\mathbf C-\mathbf C_{\mathrm{ref}}\|_2
\leq\sigma_t
\sqrt{\frac{\mathcal L_{\mathrm{sensor}}(\mathrm{initial})}{\lambda}},
\qquad \lambda>0.
$$

这是根据当前帧目标值得到的上界，要求初值与候选在同一份固定地图上、采用相同传感器归一化和总目标比较；不能拿 建图 更新后的损失代入原来的候选选择证明。它不是置信区间，也不是配置里的硬距离门限。V4 的 $\lambda=0.01,\ \sigma_t=0.01\,\mathrm m$ 在初始传感器目标约 0.03 时给出的中心偏移上界约为 1.73cm；最终修正还受到旋转项和传感器残差共同影响。[实际上界核对](../outputs/v4-prior-translation-bound-audit.json)独立检查全部 99 个候选均满足该界，最大距离与上界之比约 0.3084。传感器目标包含 RGB 均方误差、加权深度绝对误差和覆盖项，改变它们的归一化会改变该工程约束的实际强度。

### 为什么不能直接称为协方差

有平方残差和 σ² 分母，不足以把目标称为概率模型或经过标定的协方差。这里没有估计 ICP 噪声分布、像素相关性、中心与旋转的交叉项，也没有将六维信息矩阵搬运到一致坐标系。

ICP 的正规方程或列归一化 Hessian 可以反映局部几何可观测性；它受对应、残差权重、坐标单位与线性化点影响。将其条件数当作位姿协方差，或把本页的 σ 当作它的估计标准差，都会超出当前证据。见[10 ICP 雅可比与可观测性](10_rgbd_icp_observability.md)。

## 旋转弦距离与旋转对数映射

令相对旋转角为 $\theta$。对于正确的 SO(3) 旋转：

$$
\|\mathbf R-\mathbf R_{\mathrm{ref}}\|_F^2
=4(1-\cos\theta)=8\sin^2(\theta/2),
$$

$$
\frac{\|\mathbf R-\mathbf R_{\mathrm{ref}}\|_F^2}{2\sigma_r^2}
=\frac{4\sin^2(\theta/2)}{\sigma_r^2}
\approx\frac{\theta^2}{\sigma_r^2},
\qquad |\theta|\ll1.
$$

本 helper 直接计算矩阵元素差，属于旋转弦距离（chordal distance）残差。它没有计算 $\log(\mathbf R_{\mathrm{ref}}^\mathsf T\mathbf R)$，不能称为精确的 SO(3) 对数距离。$\theta$=0.1rad、$\sigma_r$=5° 时，实测 旋转弦距离项为 1.31202864，精确角度平方项为 1.31312254。两者在小角度接近，但大角度时差异增大；$\theta$=π 时，归一化 旋转弦距离项为 $4/\sigma_r^2$，角度平方项则为 $\pi^2/\sigma_r^2$。

矩阵元素的平方差在相等旋转处有有限的零梯度，避开了直接对 acos 在单位旋转附近求导的问题。这个数值方便不意味着它会自动得到正确定位；大范围旋转、ICP 错配和几何退化仍会影响参考的可靠性。

旋转弦距离项对 $\theta$ 的导数正比于 $\sin\theta$，在 $\theta$=π 也为零；它不能保证从任意大旋转恢复正确姿态。这里讨论的是对较好几何初值附近的修正约束。

## 共同世界坐标变换下的不变性

世界坐标的共同参考选择通常称为坐标规范（坐标规范）。给世界坐标共同施加一个刚体变换 $\mathbf G\in SE(3)$，两个 c2w 都左乘 $\mathbf G$，两个 w2c 都右乘 $\mathbf G^{-1}$：

$$
\mathbf G=
\begin{bmatrix}
\mathbf R_g&\mathbf t_g\\
\mathbf0^\mathsf T&1
\end{bmatrix}\in SE(3).
$$

$$
\mathbf C'-\mathbf C_{\mathrm{ref}}'
=\mathbf R_g(\mathbf C-\mathbf C_{\mathrm{ref}}),
\qquad
\mathbf R'-\mathbf R_{\mathrm{ref}}'
=\mathbf R_g(\mathbf R-\mathbf R_{\mathrm{ref}}).
$$

正交旋转不改变向量二范数与矩阵 Frobenius 范数，因此先验数值不依赖选用哪一个共同世界原点或朝向。这里要求两个输入接受**同一个**世界变换；不能给 candidate 和 reference 分别拟合不同对齐，也不涉及 相似变换（Sim(3)）尺度拟合。

CPU float64 的非单位、非零平移控制中，共同变换前后先验为 28.512861210631208 和 28.512861210631193，差约 $1.42\times10^{-14}$。解析刚体逆与一般矩阵逆最大差约 $2.22\times10^{-16}$。这是确定的合成数学检查，不能替代真实轨迹评测。

## 固定参考与跟踪阶段的先验边界

reference 是当帧几何估计，helper 内部 detach，将它作为固定参考。梯度保留到 candidate；不会通过这个 loss 优化 ICP prefix，也不读取真值。即使调用方误把 reference 标为 requires_grad=True，CPU 检查确认它仍不接收梯度。

$\lambda$=0 的语义是关闭先验。helper 返回连接 candidate 图的精确零值与零梯度，不计算 reference 残差；工程尺度仍需合法，避免错误配置被静默掩盖。candidate=reference 时同样为精确零值，六维梯度有限且为零。

这个先验只表达 跟踪时对相机候选偏移的约束。地图优化若固定相机，地图参数不应承担一个“把相机拉回 ICP”的目标；不能把 prior 加到每次 RGB-D 建图 的 data loss，再称为相同建图配置。reference 应保持在当帧 ICP 初值，不能随着候选或每次优化步更新，否则会改变先验含义。


先验也可能束缚在错误 ICP 初值附近；它不是重定位、回环、全局 BA 或几何可观测性的替代物。

### 从短片段扩展到完整轨迹

优化器能降低当前 RGB-D 加先验目标，不代表真实定位误差下降。这里的参考是独立累计的 ICP prefix，先验会约束候选接近该参考；若这个 prefix 已累计偏差，较强约束也可能限制地图提供的修正。单帧目标、当前地图误差、几何参考误差与全程轨迹误差应分别检查，不能从目标值下降直接推出 ATE 改善。


## 数学与代码怎样对应

[geometric_pose_prior.py](../scripts/geometric_pose_prior.py) 实现相机中心与 旋转弦距离先验。阅读时检查：reference 是否冻结、$\lambda$=0 是否零梯度、相等候选是否零残差，以及共同世界坐标参考 是否改变目标。用非单位旋转和平移做有限差分，避免 identity 特例掩盖坐标错误。数学实现一致不决定哪个先验权重能改善真实轨迹。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

先验动机对应的 V3 控制见[公开摘要](../outputs/v3-public-summary-audit.json)与[四轮产物核对](../outputs/v3-independent-audit-summary.json)。这些结果不证明 ICP 普遍优于 GS-SLAM，也不决定先验权重。

V4 已将该 helper 只接入 tracking，并在同一已用于诊断的 100 帧片段上运行 λ=0 与 λ=0.01 两组。采用 σt=0.01m、σr=1° 的工程尺度，锚定位置 RMSE 分别为 22.66cm 与 6.57cm；独立 ICP 为 6.41cm。两轮均没有失败标记。这说明当前先验配置下流程接近几何参考，锚定位置误差仍略高于 ICP。

两轮各自 warmup 后的高斯参数并非逐字节相同：means 最大元素差为 0.03175m。因此它们是同代码、同参数预设的配对观察，不能把全部差值归因为先验。完整指标、数据边界与保存的因果上下文见[位姿先验实验](../experiments/2026-10-09-pose-prior-controls.md)。

[同一实际地图的单帧回放](../experiments/2026-10-09-fixed-context-prior.md)进一步固定了 warmup 参数和当前图像。在关联帧 201，λ=0.01 的候选中心修正为 3.373mm、事后位置误差为 2.268mm；关闭时分别为 8.805mm、6.427mm，ICP 初值误差为 3.215mm。真值只在回放完成后按已有首相机锚定坐标比较，没有新拟合对齐；这些是单帧候选误差，不是轨迹 ATE/RPE。相同输入字节也不消除 GPU 运算的潜在非确定性。这是一个局部案例，不能替代完整流程的 82/99 帧未优于 ICP 观察。

[固定参数换序列](../experiments/2026-10-09-xyz-transfer.md)保持优化配置，在 fr1_xyz 上重新初始化地图。λ=0.01 流程 raw 位置 RMSE 为 41.501mm，独立 ICP 为 41.166mm；ATE 略低，但 raw/RPE 平移略高，仍不能称全面优于几何参考。相对同帧 ICP prefix，λ=0 与 0.01 分别有 83/99、66/99 帧的锚定位置误差变差；两组 warmup 地图也不同。该序列来自相同传感器与相近办公域，不构成跨场景泛化证据。

[固定 seed0 的完整序列实测](../experiments/2026-10-09-full-v4-sequence-controls.md)显示，desk/xyz 的先验组 raw 与 ATE 均高于关闭组，与 100 帧片段的简单排序不同。相邻平移或旋转 RPE 仍有混合变化。这里没有独立隔离哪一帧的 ICP 误差造成哪一段漂移，也没有通过 GT 调整 λ；两组 warmup 参数实际不同。它说明局部控制和短片段证据不能自动推广到完整在线历史，没有证明先验在所有设置下都无效。

### 可复核的数学证据

[helper 源码](../scripts/geometric_pose_prior.py)与[CPU 检查 JSON](../manifests/geometric-pose-prior-cpu-checks.json)保存了公式、单位、源文件哈希、全部六维解析/有限差分数组及负控制。检查采用 torch float64、SE3 左扰动 `exp(ξ^)@base_w2c`，twist 顺序为 `[ρ_m,ω_rad]`；ρ 是李代数平移分量，有限旋转下不应直接当成矩阵 t。

测试覆盖纯平移、纯旋转、非单位 base/reference、零与非零 perturbation、相等输入和 λ=0。中心有限差分步长 `1e-6`，最大解析梯度绝对误差约 `2.44e-9`。共同世界 gauge、固定 reference、错误 w2c 相机中心也单独验证。这些检查回答“公式与梯度是否实现正确”，没有回答“哪个工程权重能改善当前 GS 地图与跟踪”。

[独立数学审查](../outputs/geometric-pose-prior-independent-review.json)另外使用 SciPy 指数映射、NumPy 一般逆与 Torch 可微路径交叉检查。两个非单位姿态的六维有限差分最大绝对差为 8.53e−9；同中心、不同朝向的负控制确认没有把 w2c 的 t 误当世界相机中心。该审查没有运行 GPU 或评测先验训练效果。

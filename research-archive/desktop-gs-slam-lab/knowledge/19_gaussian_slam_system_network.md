---
id: "gslab-k19"
slug: "gaussian-slam-system-evidence-network"
title: "GS-SLAM 知识主线 几何 优化 融合与评测"
order: 19
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/slam_helpers.py"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
---

# GS-SLAM 知识主线 几何 优化 融合与评测

首次阅读，沿[传统 3DGS 渲染](27_3dgs_rendering_foundations.md) → [训练与密度控制](28_3dgs_training_density_control.md) → [从重建进入 SplaTAM](29_3dgs_to_splatam_progression.md)建立基础，再用本页串联系统问题。

学习高斯 SLAM，主线是把“像素与传感器观测”变成“可信的相机轨迹和一致的地图”。高斯表示负责生成可微观测，优化负责调整未知参数，标定与时间处理决定这些观测能否放在一起，评测负责检验结果是否正确。

SplaTAM 适合把 RGB-D 几何、可微渲染和交替优化连起来；FAST-LIVO2 则扩展到 LiDAR、IMU 和视觉状态估计。两者共享坐标、残差、可观测性与漂移问题，地图表示和更新方法各有不同。

学习顺序是相机几何与单位、高斯投影与合成、残差与位姿优化、建图与历史一致性、多传感器融合与时空标定，最后是真值关联与轨迹评价。

按这个顺序理解，每一步都要回答：约束来自哪里、改变了哪些未知量、哪些误差会传到下一步。

## 从像素走到世界坐标

首先分清像素坐标、相机坐标、世界坐标和传感器机体坐标。令 $K$ 为相机内参，$(u,v)$ 为像素坐标。光轴是经过相机投影中心并垂直于像平面的直线，在针孔模型中与相机 $Z$ 轴重合，正 $Z$ 方向朝向可见场景。$z>0$ 为米制轴向深度，即相机坐标 $Z$ 分量；相机点为 $p_c=zK^{-1}(u,v,1)^T$；若传感器给的是径向距离，需要另一种换算。深度尺度、焦距和主点错误都能产生有限数值，却把几何整体算错。

$C=\mathrm{C2W}$ 将相机点变到世界，$W=C^{-1}$ 用于把世界投影回相机。若 $p_c=Rp_w+t$，世界中的相机投影中心是 $-R^Tt$。变换必须带方向，例如 $T_{r\leftarrow k}$ 表示当前相机到参考相机，组合为 $C_k=C_rT_{r\leftarrow k}$；矩阵乘法顺序不能凭变量名猜测。

固定首相机只选择坐标自由度 gauge。它不提供外部绝对位置，也不会消除后续漂移。理解[相机几何](01_camera_geometry.md)和[先验 单位与 gauge](11_pose_prior_units_gauge.md)后，应能解释“世界换一个原点”和“相机估计错了”为何不同。

**核心问题：** 图像缩小一半后哪些内参改变？保存的位姿是在描述相机运动，还是在把世界点变进相机？

## 用高斯把世界变回观测

高斯中心、协方差、颜色和不透明度描述场景；相机位姿与内参决定投影。各向异性高斯的协方差由旋转和主轴标准差组成：

$$
\Sigma=R_g\begin{bmatrix}
s_1^2&0&0\\
0&s_2^2&0\\
0&0&s_3^2
\end{bmatrix}R_g^T.
$$

其中 $R_g$ 将局部主轴转到世界，$s_i>0$ 为场景长度单位的标准差，$\Sigma$ 的单位为长度平方。经投影形成图像上的椭圆覆盖范围，再经遮挡排序和前向不透明度合成生成像素。

按深度排序后，第 $i$ 个高斯的像素不透明度为 $\alpha_i$，相机轴向深度为 $z_i$，透射率为 $\mathcal T_i=\prod_{j<i}(1-\alpha_j)$，合成权重 $w_i=\mathcal T_i\alpha_i$ 同时影响颜色、覆盖和深度。累计深度 $D=\sum_iw_iz_i$ 与归一化深度 $D/S$ 不同，其中 $S=\sum_iw_i$ 为累计不透明度。SplaTAM 的主路径使用累计深度；其他后端可能提供不同语义。只核对形状和设备类型不足以证明两个渲染器等价。

表示容量也影响优化。更多高斯可以解释更多细节，也可能增加计算并吸收位姿误差；球形高斯没有可辨识的自身几何方向。[高斯投影与合成](02_gaussian_projection_compositing.md)、[初始化与容量](05_capacity_initialization_density.md)和[后端语义](09_renderer_backend_semantics.md)共同解释这些选择。

**核心问题：** 深度值偏小是场景更近，还是透明度覆盖较低？外观拟合变好是地图更准确，还是地图自由度增加？

## 用残差约束相机位姿

固定地图时，把渲染结果与当前图像比较，得到光度与深度残差；更新相机，使这些残差下降。设 $r$ 为残差向量，$\delta$ 为局部位姿增量，$J=\partial r/\partial\delta$ 为雅可比矩阵，则局部线性化与加权近似曲率为 $r(\delta)\approx r(0)+J\delta$、$H=J^T\Lambda J$，其中 $\Lambda$ 是残差权重矩阵。

约束是否充分取决于 Jacobian 的方向信息，而不只取决于像素数量。大平面可能缺乏沿平面的几何约束；纹理单一或重复可能缺乏可靠光度约束。深度、颜色、运动先验能互补，但错误先验也会把估计拉向错误参考。讨论秩与条件数时，还要统一平移和旋转单位、考虑噪声尺度与 gauge。

有效像素集合也是目标的一部分。若位姿变化使掩码改变，求和损失下降可能伴随参与像素减少；均值损失也可能因为删去困难像素而下降。因此要一起看支持集合、残差分布、步进和独立位姿误差。

[固定地图位姿优化](06_pose_optimization.md)说明更新过程，[ICP 可观测性](10_rgbd_icp_observability.md)说明几何退化，[光度与几何里程计](13_photometric_geometric_odometry.md)说明互补条件，[SplaTAM Tracking 损失](25_splatam_tracking_loss_support.md)解释动态支持。继续研究亮度变化时，再读[仿射亮度](15_affine_brightness_observability.md)与[Schur 耦合](16_schur_pose_brightness_coupling.md)。

**核心问题：** 低 loss、满秩近似矩阵和低 GT 误差，分别证明了什么？加入一个参数能解释变化，是否也会削弱位姿辨识？

## 把局部估计变成一致地图

Tracking 固定地图优化相机；Mapping 在已有相机下优化场景。前者回答“在哪里看”，后者回答“看到了什么”。SplaTAM 默认交替执行这两个阶段，窗口内历史相机固定，不等于联合 BA。

关键帧让多视角共同约束地图，但有限窗口并不持续约束全部历史。当前位姿有偏时，新高斯也会有偏；后续相机再对有偏地图定位，构成误差反馈。回环约束、历史位姿优化与重定位要分别理解：它们处理的是长时一致性或恢复能力，不能由“窗口变大”自动推出。

位姿更正必须同时考虑地图坐标。令共同刚体世界变换 $G$ 的旋转与平移为 $R_G,t_G$，作用后，应有 $\mu^{\prime}=R_G\mu+t_G$、$\Sigma^{\prime}=R_G\Sigma R_G^T$ 和 $C^{\prime}=GC$，对应 $W^{\prime}=WG^{-1}$。这样相机空间几何保持不变。不同局部锚点接受不同更正时，跨锚点一致性还需另行处理。

[跟踪与建图边界](03_tracking_mapping_scope.md)、[关键帧与历史窗口](14_keyframes_map_window_history.md)、[地图与位姿更正](18_gaussian_map_pose_correction.md)解释一致性。[拒绝 参考与恢复](17_rejection_reference_recovery.md)解释优化失败后怎样保留时间与状态。[SplaTAM 系统原理](24_splatam_official_rgbd_system.md)把这些概念接成一帧循环。

**核心问题：** 一帧被拒绝后，传感器时间是否继续推进？轨迹修正后，高斯位置和协方差为何也需要处理？

## 扩展到融合 标定与时间

多传感器融合增加了不同的约束来源。IMU 支持高频运动传播，LiDAR 提供几何，视觉提供外观与投影残差。融合是否有帮助，取决于观测模型、噪声、场景和误差状态中哪些方向能被约束；增加传感器不保证每段轨迹都更准确。

外参把不同传感器坐标连起来，时间模型把它们放到同一运动时刻。设观测为 $y$，系统状态为 $x$，时间偏移为 $\Delta t$，外参为 $E$，内参为 $\kappa$，观测模型为 $y=h(x(t+\Delta t),E,\kappa)+\varepsilon$，其中 $\varepsilon$ 为测量噪声。运动时，时间偏差会表现为几何误差；错误外参也可能被状态或其他参数部分吸收。时间关联、时间偏移估计、运动去畸变与硬件同步各有作用。

最近邻配对只说明软件选择了哪些样本，不能证明这些样本是同时采集的；沿用公开标定也不能称为重标定完成。视觉阶段留下状态修正说明更新产生了影响，修正方向是否正确仍要结合可观测性和真值评价。

[FAST-LIVO2 融合](20_fast_livo2_fusion.md)、[标定](21_fast_livo2_calibration.md)、[同步](22_fast_livo2_synchronization.md)和[漂移](23_fast_livo2_drift.md)对应这四条链。[视觉更新与可观测性](26_fast_livo2_visual_update_observability.md)进一步解释残差如何影响状态。

**核心问题：** 外参错与时间偏差在静止、匀速和急转运动中怎样表现？测到“状态发生变化”和测到“误差下降”之间还缺哪一步？

## 用评价区分拟合与准确性

RGB-D 与 LiDAR 系统具有米制尺度，轨迹评价通常采用尺度固定的 SE(3) 对齐。ATE 测量对齐后的整体位置误差，RPE 测量指定时间或帧间隔的相对运动误差；必须说明间隔。对齐会去除共同刚体偏移，RPE 小也可能逐步累积成较大漂移。

地图 PSNR、深度误差与轨迹误差测量不同能力。用保存的估计相机渲染，更接近判断地图是否拟合训练观测；用真值相机渲染，还会体现地图与真实参考系的偏差。评价应固定时间关联、覆盖分母、相机约定及真值传感器坐标，保留失败时刻。

[评价协议](04_evaluation_protocol.md)、[轨迹评价](08_trajectory_evaluation.md)和[种子与复现](12_reproducibility_seed_variation.md)说明如何建立可比较结论。提出机制假设后，需要控制变量与重复场景；一条曲线的相关关系不足以确定原因。

**核心问题：** ATE 下降而固定首相机参考系误差上升，如何同时成立？地图图像更清晰，是否代表轨迹更准确？

## 阅读顺序与能力自检

| 顺序 | 先掌握什么 | 应能独立解释什么 |
|---|---|---|
| 1 | 相机几何 高斯渲染 | 从像素反投影到世界 再渲染回图像 |
| 2 | 残差 位姿更新 有效支持 | 区分目标下降与真实位姿改善 |
| 3 | SplaTAM 循环 关键帧 地图一致性 | 解释局部误差怎样进入地图并传播 |
| 4 | 融合 外参 时间 可观测性 | 判断一种新传感器提供了哪些约束 |
| 5 | ATE RPE 地图指标 重复验证 | 给出与结论强度匹配的评价协议 |

能沿这条链讲清一个失败案例，比背诵一组成绩更能体现系统理解：先检查观测与坐标，再检查约束与更新，最后用独立评价判断局部改善有没有转化成整体改善。

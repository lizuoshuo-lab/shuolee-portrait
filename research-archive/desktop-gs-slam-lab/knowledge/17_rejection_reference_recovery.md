---
id: "gslab-k17"
slug: "rejection-reference-recovery"
title: "拒绝之后 参考帧与位姿传播"
order: 17
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"
---

# 拒绝之后 参考帧与位姿传播

配准失败后，系统仍要给当前时刻保存一个状态，并决定下一帧用什么观测作参考。相机估计和参考图像是两类状态：复制上一位姿，不会把当前图像变成上一张图，也不证明相机没有移动。

以下以 CPU RGB-D 教学前端的严格保持上一估计的递推和 ICP 回退为例，解释状态机与误差传播。它们不是官方 SplaTAM 或 FAST-LIVO2 的默认恢复功能。多传感器系统中，某次视觉更新无效也不意味着整体状态冻结，IMU 传播和其他观测可能仍在推进。

## 参考必须包含观测身份

设 $\widehat{\mathbf C}_k$ 是第 $k$ 帧估计的 相机到世界变换（camera-to-world），$\mathbf T_{r\leftarrow k}$ 把当前相机 $k$ 的点变到参考相机 $r$。参考至少包含帧 ID、RGB-D、内参、各自时间戳和保存位姿：

$$
\mathcal O_r=(r,\mathbf I_r,D_r,\mathbf K_r,t_r^I,t_r^D),\qquad
\mathcal H_r=(\mathcal O_r,\widehat{\mathbf C}_r),\qquad
\mathcal O_k=(k,\mathbf I_k,D_k,\mathbf K_k,t_k^I,t_k^D).
$$

$$
\mathbf T_{r\leftarrow k}=
\begin{bmatrix}
\mathbf R_{r\leftarrow k}&\mathbf t_{r\leftarrow k}\\
\mathbf0^\mathsf T&1
\end{bmatrix},\qquad
\mathbf p_r=\mathbf R_{r\leftarrow k}\mathbf p_k+\mathbf t_{r\leftarrow k},\qquad
\widehat{\mathbf C}_k=\widehat{\mathbf C}_r\mathbf T_{r\leftarrow k}.
$$

$r,k$ 为观测帧身份，$\mathbf I,D,\mathbf K$ 为图像、深度和内参，$t^I,t^D$ 为各自时间戳；$\widehat{\mathbf C}$ 是相机到世界变换。参考状态 $\mathcal H_r$ 同时包含观测身份和保存位姿。

求解器输入的图像、深度、法向和掩码，必须与声明的参考身份一致。候选变换的方向还必须与位姿组合匹配。只看两个保存矩阵相等，无法判断观测是否相同；只看字段名含 previous，也无法确定调用时真正传入了哪一帧。

RGB 与深度的软件关联不等于物理同时采样，参考应保留各自时间与原始索引。坐标方向见[相机几何](01_camera_geometry.md)，光度与几何候选见[里程计残差](13_photometric_geometric_odometry.md)。

## 两种参考策略怎样传播

| 策略 | 当前候选对齐到谁 | 接受后的组合 | 拒绝后的观测参考 |
|---|---|---|---|
| previous-frame | 前一传感器帧 $k$−1 | $\widehat{\mathbf C}_k=\widehat{\mathbf C}_{k-1}\mathbf T_{k-1\leftarrow k}$ | 下一次仍使用新的前一传感器帧 |
| last-accepted | 最后接受帧 $r$ | $\widehat{\mathbf C}_k=\widehat{\mathbf C}_r\mathbf T_{r\leftarrow k}$ | 保持 $r$ 的图像与深度 |

前一传感器帧策略（previous-frame）与最后接受帧策略（last-accepted）使用不同的观测身份。本例两种策略拒绝时都复制上一估计。previous-frame 保持短参考间隔，可能较容易建立局部对应，但先前漏掉的运动不会因下一次相邻配准成功而自动补回。last-accepted 直接约束从 $r$ 到 $k$ 的累计运动，有机会跨过短暂失败；代价是参考间隔可能越来越大，重叠与初始化条件变差。

在严格复制上一估计 的 最后接受帧参考状态中，若 $r$ 之后全部拒绝，则 $\widehat{\mathbf C}_{k-1}=\widehat{\mathbf C}_r$，但 $\operatorname{id}_{\mathrm{sensor}}(k-1)$ 与 $\operatorname{id}_{\mathrm{sensor}}(r)$ 仍然不同。此时换一个相等的前缀变量不会改变乘法数值；真正的问题可能是候选对错图像求解，或用错变换方向。状态代数一致是准确定位的必要条件，不是准确性的证明。

## 漏掉的运动不会自动回来

设 $j$ 被拒绝，下一帧 $k$ 恢复。用理想相对变换说明缺失区间：

$$
\mathbf A=\mathbf T_{r\leftarrow j},\qquad
\mathbf B=\mathbf T_{j\leftarrow k},\qquad
\mathbf T_{r\leftarrow k}=\mathbf A\mathbf B.
$$

$$
\widehat{\mathbf C}_j=\widehat{\mathbf C}_r,\qquad
\widehat{\mathbf C}_{k,\mathrm{adjacent}}=\widehat{\mathbf C}_r\mathbf B,\qquad
\widehat{\mathbf C}_{k,\mathrm{direct}}=\widehat{\mathbf C}_r\mathbf A\mathbf B.
$$

第一条恢复路径只重新估计相邻区间，第二条直接估计从参考到当前的累计运动。

前一条路径缺少 $\mathbf A$。错误不在两个相等前缀的变量名，而在运动区间没有被重新观测。旋转和平移通常不交换，$\mathbf A\mathbf B$ 也不能任意改成 $\mathbf B\mathbf A$。

因此“重新接受一个候选”与“恢复此前遗漏的全部运动”是两种能力。更广的参考、重定位或其他传感器可以提供新约束，单次相邻恢复本身不保证补齐历史。

## 主路径、回退与保持的状态递推

教学回退策略先尝试当前帧到 保存的参考帧 的光度—几何配准；失败后才选择当前帧到前一传感器帧的合格 ICP 候选。它使用明确优先级，真值不参与在线选择：

$$
\widehat{\mathbf C}_k=
\begin{cases}
\widehat{\mathbf C}_r\mathbf T_{\mathrm{primary}},&a_p=1,\\
\widehat{\mathbf C}_{k-1}\mathbf T_{\mathrm{fallback}},&a_p=0,\ a_f=1,\\
\widehat{\mathbf C}_{k-1},&a_p=0,\ a_f=0,
\end{cases}
$$

$$
r_{\mathrm{new}}=
\begin{cases}
k,&a_p=1\ \text{或}\ a_f=1,\\
r,&a_p=0,\ a_f=0,
\end{cases}
\qquad
j_{\mathrm{previous\ sensor,new}}=k.
$$

$a_p,a_f\in\{0,1\}$ 分别为主路径与回退候选的接受状态，主路径优先。所有分支都保存当前时刻的状态与时间；前一传感器帧索引始终前移。

两种候选使用不同参考，必须各自保存方向和观测身份。回退 后把 保存的参考帧 提升为 $k$，是这份策略的设计选择，不是所有 SLAM 系统的规则。若保存前缀已经漏记历史运动，它仍可能带着偏差继续传播。

尝试失败与最终保持上一估计 也要区分：主路径 失败、回退 成功时，系统完成了一次状态更新，但 主路径 的失败原因应保留。亮度增益与偏置 不是位姿，不参与 相机到世界矩阵组合；候选状态如何初始化和传给后续分支，必须按模型定义，见[亮度与可观测性](15_affine_brightness_observability.md)。

## 参考停留怎样影响可恢复范围

随着 最后接受帧参考停留，当前与参考之间的运动、曝光差异和遮挡可能增大。原本针对相邻观测设定的对应或运动门限，不一定适合较大的跨帧间隔。

$\Delta k=k-r$ 只表示选中序列的帧 ID 差，不是秒或距离。真正的时间差应由时间戳计算，即 $\Delta t=t_k-t_r$；同样的 gap 在不同帧率和丢帧条件下有不同物理意义。固定相对运动门限也不能与每次迭代的步长限制混用。

连续失败可能涉及缺少对应、局部不可观测、线搜索没有下降或最终运动超限。失败标记描述触发了哪个程序判据，不能单独鉴定曝光、遮挡或漂移根因。扩大门限也不自动增加有效观测，可能只是接受更不可靠的候选。

## 为什么完整路径比较不是逐帧救援计数

一次 回退 改变位姿和参考后，下一次 主路径 面对的历史就改变。两条完整路线后续的候选通常不再一一相同；拒绝数之差不能直接解释成“原来的这些帧逐个获救”。

恢复可能让当前位置接近真值，同时把过去多个间隔的运动集中在一次更新中。因此整体位置误差下降与相邻 RPE 增大可以同时发生。要验证单个候选机制，应固定同一参考、地图和当前观测；要验证恢复策略，应保留各自真实的完整递推历史。两类对照回答不同问题。

## 全帧评价 保留失败时刻

拒绝帧仍对应一个真实采样时刻。复制上一位姿时，如果真实相机继续移动，该帧依然有位置误差；相邻 RPE 也应包含 保持上一估计期间的零估计运动和恢复时的跳变。

评价不能删掉拒绝帧、把结束时间改为最后接受时刻，或对停滞片段另行锚定后报告改善。整体位置、一次 SE(3) 对齐后的 ATE、指定间隔 RPE、末帧误差与失败覆盖应一起解释，见[轨迹评价](08_trajectory_evaluation.md)。

持续输出、候选被接受和位置准确是三个不同结论。多传感器融合时还要区分某观测的更新状态与共享状态的推进，见[FAST-LIVO2 融合](20_fast_livo2_fusion.md)。

## 理解自测

- 当前位姿与 保存的参考帧 位姿相等，为什么图像身份仍重要？同一数值可以由多个拒绝时刻传播，观测内容与候选方向仍不同。
- 回退 接受为何不保证全局恢复？它可能只恢复相邻运动，并继续使用已经有偏的前缀。
- 少了拒绝帧为何仍可能更不准确？接受规则决定能否更新，不直接判断候选离真值多近。
- 比较两个恢复策略时要固定什么？输入与评价口径可统一，策略造成的参考历史分叉必须保留并解释。

## 实验附录

真实成绩与完整状态记录保留在[短段参考策略](../experiments/2026-10-09-last-accepted-reference-recovery.md)、[六维与八维模型对照](../experiments/2026-10-09-affine-photogeometric-real-sequences.md)、[完整参考停留实验](../experiments/2026-10-09-full-affine-reference-controls.md)和[相邻 ICP 回退实验](../experiments/2026-10-09-affine-icp-fallback-controls.md)。[参考历史分叉记录](../outputs/affine-hybrid-primary-prefix-comparison-001/metrics.json)保存候选字段比较与边界帧；这些记录用于检验上述机制，不构成官方系统的恢复能力声明。

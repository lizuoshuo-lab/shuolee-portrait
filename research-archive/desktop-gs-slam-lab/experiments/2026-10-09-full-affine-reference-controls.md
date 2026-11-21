---
id: "gslab-exp-full-affine-reference-controls-20261009"
slug: "full-affine-reference-controls"
title: "24 完整序列：光度模型的拒绝链与参考停滞"
order: 24
status: "completed_four_full_CPU_controls_independently_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"
---

# 完整序列：光度模型的拒绝链与参考停滞

四轮完整 CPU 对照保留全部 2,594 个关联帧。desk 的六维与八维模型都在 134–548 连续拒绝 415 帧，参考停留在 133；八维没有解除停滞。xyz 的八维位置类指标更高，相邻旋转 RPE 略低。短片段成功、局部满秩与进程正常结束都不能代替完整序列稳定性的证据。

原独立结果审计、真实六维字段补充和公开字段保留检查均已通过。拒绝帧仍进入指标；模型与阈值保持冻结，GT 不参与优化或候选选择。本报告接续[参考策略对照](2026-10-09-last-accepted-reference-recovery.md)、[八维短序列对照](2026-10-09-affine-photogeometric-real-sequences.md)与[恢复边界知识](../knowledge/17_rejection_reference_recovery.md)。

## 完整序列保留所有拒绝帧

这次把此前的固定六维/八维模型扩展到 desk 的全部 549 个关联帧和 xyz 的全部 748 个关联帧。它们是冻结 loader 经过 RGB、深度、GT 时间匹配、深度去重和帧率规则得到的序列，不是目录中的全部 raw RGB。四份新轨迹合计 2,594 帧，首相机各固定一次，因此有 2,590 个待配准对。

实际批次按 desk 6D、desk 8D、xyz 6D、xyz 8D 串行执行，四个子进程退出码均为 0，持有的执行会话也返回真实 exit 0。原始模型和参数没有改动，未重试、未挑选片段或根据 GT 改阈值；正常退出只表明进程完成，不等于每帧成功定位。

已终态保留的拒绝计数是 desk 6D/8D 分别 421/420，xyz 6D/8D 分别 15/21。每个拒绝帧仍有对应时间戳、候选、失败标记和传播后的估计，参与全帧指标。一个帧可以同时触发多个标记，不能将 flag 次数相加当作失败帧数。

## “最后接受参考”为什么会长期停留

采用的参考是最后一次成功接受的传感器帧。接受当前候选时，把该参考相机的 c2w 与 current→reference 相对候选相乘，并把当前传感器帧设为新参考。拒绝当前候选时，保留前一个相机估计，参考传感器也不前移。

这次 desk 的两种模型都在关联帧 134–548 连续拒绝，共 415 帧，参考保持在 133，最大 reference gap 为 415。这里传播的是同一个已保存估计，当前传感器时间仍然推进；不是将这 415 帧删除，更不能把末帧误差计算在第 133 帧的时间戳上。

最后接受策略在短暂拒绝后可以提供一个仍可用的参考，但不保证它能处理持续拒绝。一旦参考不前移，当前图像与该参考的观测差异可能继续变化；冻结的对应、共同支持、位移、旋转和强度门槛仍然生效。本次长期拒绝必须作为方法行为保留，不能改写成整轮进程失败，也不能由“连续拒绝”直接推断唯一的物理原因。

原记录中，desk 两组都出现 382 次 `final_insufficient_joint_correspondences`，并保留强度、位移及部分八维可观测性标记。xyz 两组的拒绝标记都是 `finest_level_line_search_no_descent`。这些是程序触发的工程判据，不是对曝光、遮挡、漂移或数值舍入原因的独立鉴定。

## 固定模型与不同接受路径

六维路线复用冻结的 `313d…` last-accepted 递推与原 photo+geometry 求解器；八维路线复用 `bf219…` 管线中已有的仿射模型与递推。几何尺度、强度尺度、Huber、对应筛选、支持边界和原始门槛均保持各自原配置。

两者首相机为 identity，每一对相对位姿都从 identity 开始。八维模型每对重新设置 log-gain=0、bias=0，不把估计的亮度参数累积成全局曝光状态，也没有因为 Schur 数学材料加入先验、阻尼、钳制或新的求解器。

这是两条完整模型与接受路径的比较。即便初始传感器和参数一致，模型维数、可观测性、试探候选、接受决定与后续参考都可能不同；不能把结果包装成逐步固定状态、只增加两个亮度参数的纯因果消融。

## 六条完整轨迹：保留拒绝后的所有时间点

四条新轨迹取自正式汇总 `outputs/rgbd-affine-full-comparison-001/summary.json`，SHA256 为 `d6a5f9e3…635d3`。两条完整 ICP 是只读旧基线，另行绑定旧审核证据。平移列统一为 mm，旋转 RPE 为度；不会把跨序列均值或单个指标作为所有取舍的替代。

| 序列 | 方法 | raw RMSE / mm | SE3 ATE / mm | 相邻 RPE 平移 / mm | 相邻 RPE 旋转 / ° | 末帧位置 / mm | 拒绝帧 / 全帧 |
|---|---|---:|---:|---:|---:|---:|---|
| desk | 旧完整 ICP，只读参考 | 142.881 | 83.043 | 8.913 | 0.868 | 184.811 | 0 / 549 |
| desk | 完整 last-accepted 6D | 935.071 | 825.411 | 17.288 | 1.395 | 1038.993 | 421 / 549 |
| desk | 完整 last-accepted 8D | 933.760 | 825.241 | 17.246 | 1.393 | 1035.527 | 420 / 549 |
| xyz | 旧完整 ICP，只读参考 | 44.151 | 30.163 | 3.540 | 0.504 | 71.448 | 0 / 748 |
| xyz | 完整 last-accepted 6D | 103.241 | 56.944 | 5.624 | 0.414 | 165.885 | 15 / 748 |
| xyz | 完整 last-accepted 8D | 106.978 | 58.622 | 6.134 | 0.408 | 173.651 | 21 / 748 |

desk 中，八维的上述数值略低于六维，但两者共同存在 415 帧的连续拒绝，raw 都约 0.934 m，末帧误差都超过 1 m。少拒绝一帧、局部数值略低，不等于长序列恢复成功。xyz 中，八维比六维多拒绝六帧，raw、ATE、平移 RPE 和末帧误差都更高，只有相邻旋转 RPE 更低。

相对旧 ICP，这四条新轨迹的位置类指标均更高。xyz 的六维/八维相邻旋转 RPE 较低，但不能据此称整体定位更好；desk 的旋转 RPE 也明显更高。全程 raw 旋转 RMSE 另有不同含义：desk 六维/八维为 45.436°/45.402°，xyz 为 4.487°/4.396°；相邻旋转变化较准确，不保证相对起点的全程方向偏差小。

旧 ICP 没有重新运行或重新选段。其数字来自原 `tum-online-v4-full-sequence-recovery-summary-001/summary.json`，并绑定该目录 `independent_summary_audit.json` 的 2,321 文件审核（SHA256 `70f65dfc…01417e`）。本次另行核对两条 baseline 的原 metrics、小 summary、估计轨迹文本与完整传感器元数据，共八份实际文件，与原 input hash 清单逐项一致。其参考策略、残差与支持门槛与新六维/八维路线不同，表格属于固定实现的观察，不是只切换亮度模型的纯因果比较。

raw 使用首个相机定义的共同坐标，SE3 ATE 对整条位置轨迹做一次后验刚体对齐、尺度固定为 1，RPE使用全部相邻对。持续传播的相机也属于估计轨迹：不能在评价前裁掉停滞段、重新起始或换成最后接受时刻。

末相机尤其容易被误解：xyz 六维拒绝了 746、747，两帧保留第 745 帧的估计；表中的末帧误差仍在最终第 747 帧的传感器时刻计算。八维在 747 接受更新，末帧误差却更高。最后一帧是否接受，与它离评价参考多远，是不同问题。

## 局部满秩与短段成功不能替代长序列证据

知识 15、16 的合成控制回答的是局部残差、雅可比、线性化和位姿—亮度耦合是否符合声明。满秩只涉及选定支持与尺度下某个局部系统；它不承诺当前对应正确，不承诺试探越过插值单元后保持同样目标，也不保证当前参考持续可用。

短片段的初始参考、历史和观测范围与整段不同。短段能够完成、局部合成系统可解，都不能推广成完整序列稳定；完整轨迹必须连同拒绝链、原分母和最后相机一起解释。本报告不会把其中一个指标改善称为整体优越，也不会把这两条同域序列称为跨场景普遍结论。

## 来源、GT 与审计的分工

GT 字段、最近时间戳关联和 GT 矩阵的构造确实由冻结 loader 读取；不能写成“从未读取 GT”。但 GT 矩阵不传入对应、相对位姿初始化、亮度估计、目标或接受规则。所有估计与拒绝传播轨迹导出之后，才用首个 GT 构造评价坐标，进行最终指标计算。没有 GT 引导的重定位、重新初始化或候选选择。

原独立审核已实际 exit 0，产物 `outputs/rgbd-affine-full-validation-001/independent_full_affine_result_audit.json` 的 SHA256 为 `a9480a1a…e4afd5`，状态为 `passed_independent_full_input_reference_step_and_trajectory_audit`。它检查 2,594 行、2,590 对、48,697 条保存步骤和 49,086 条试探，重新核对 5,283 份消费文件。它明确复用既有 e807 的原始 GT/SciPy 指标路径与 b549 的八维保存步进核查；六维接受更新和八维试探链按保存证据复算，没有重新运行配准，也没有独立重建所有真实对应的秩和目标值。

后续源码负控发现：原审核器对畸形、缺少本行 proposal 的六维步骤，局部变量可能沿用前一行。原源码、已通过产物和负控均保留。新的[真实字段补充审计](../outputs/rgbd-affine-full-6d-proposal-addendum-001/metrics.json)现已实际 exit 0，状态为 `passed_current_row_6d_proposal_field_addendum`，SHA256 为 `1d75beec…642a00`。它绑定原 `155b…` 审核器、`a948…` 审核产物和 `d6a5…` 汇总，检查两条六维轨迹的全部 1,297 帧、1,295 对、24,464 条步骤；21,723 次接受更新均有本行有限六维 proposal 和 accepted 字段，包含最终被拒绝帧中的已接受内部更新。接受增量与本行 proposal 的最大分量差分别是 **0 m 和 0 rad**，两种单位分别核对。

该补充由[独立字段守卫源码](../scripts/audit_rgbd_full_6d_proposal_addendum.py)、[固定协议](../outputs/rgbd-affine-full-6d-proposal-addendum-cpu-preparation-001/protocol.json)和[执行前源码审阅](../outputs/rgbd-affine-full-6d-proposal-addendum-cpu-preparation-001/root_source_review.json)约束，源码、协议、审阅 SHA256 分别为 `c5087f44…58c8e3`、`76f4c95f…eb6ba1`、`cba08e83…e34ff`。它补上“真实每次更新均来自本行字段”的证据，没有重新消费 GT、重新计算轨迹指标、对应、秩或目标，也没有修改原任何数值。新增守卫发现的是原审核器面对畸形输入的覆盖缺口，实际数据通过并不意味着该缺口可以省略。

JSON 汇总程序严格绑定已通过的原审核精确 hash、原四份 metrics、冻结源码与输入。实际汇总已 exit 0，重新检查 5,252 份自己的消费文件；与独审的 5,283 是不同范围。四份 compact 帧表逐项检查候选矩阵、亮度参数、接受状态、全部 flags、参考 before/after、gap、传感器索引及原误差数组，与原记录完全相同。这个检查只证明子集导出正确，不冒充新的算法审计。

下方材料包含小体积协议、终态批次、原独立审核与实际通过的 addendum、四份完整 compact 帧表、小摘要与汇总源码。当前四份帧表分别为 340,451、347,906、572,108、607,401 B；17 份完成清单文件的实际大小和 hash 已再次核对。原始迭代流、完整候选和大 metrics 留本地；compact 只是可追踪的字段子集，不替代原证据。没有通过删帧或删失败标记满足公开文件大小限制。


## 完整轨迹与参考间隔

左图直接使用已保存c2w的相机中心，叠加原首相机坐标下的GT，仅作XY投影；等比例米坐标不显示Z，也没有对齐、重新锚定或删帧。右图保留全部关联帧的 `gap=k−r`，叉号表示拒绝。重复保存的hold位姿会在轨迹图上重叠，这不是删除记录。

![desk全部549帧的轨迹与415帧拒绝链](../outputs/rgbd-affine-full-trajectory-figures-001/desk-full-trajectory-and-reference-gap.png)

desk两组的参考间隔持续增长到415，而估计相机不再移动。图上的轨迹差距不能当作三维位置误差或ATE重新计算。

![xyz全部748帧的轨迹与参考间隔](../outputs/rgbd-affine-full-trajectory-figures-001/xyz-full-trajectory-and-reference-gap.png)

两张图分别设置显示范围，不能依据画面大小比较两个序列的精度。[绘图源码](../scripts/plot_rgbd_affine_full_trajectories.py)、[原输入和图像hash清单](../outputs/rgbd-affine-full-trajectory-figures-001/figures.json)与[目视核对](../outputs/rgbd-affine-full-trajectory-figures-001/root_visual_review.json)只验证保存结果的呈现，不新增优化或定位指标。

## 可复核材料

- [四轮冻结协议](../outputs/rgbd-affine-full-cpu-preparation-001/protocol.json)
- [root 执行前源码审阅](../outputs/rgbd-affine-full-cpu-preparation-001/root_protocol_review.json)
- [完整输入与模拟控制独立预检](../outputs/rgbd-affine-full-cpu-preparation-001/root-independent-full-preflight-review.json)
- [四个真实进程终态](../outputs/rgbd-affine-full-validation-001/metrics.json)
- [原完整轨迹与保存步进独审](../outputs/rgbd-affine-full-validation-001/independent_full_affine_result_audit.json)
- [完整四组汇总](../outputs/rgbd-affine-full-comparison-001/summary.json)
- [完成文件大小与 hash 清单](../outputs/rgbd-affine-full-comparison-001/completion.json)
- [24列字段独立检查源码](../scripts/check_rgbd_affine_full_public_fields.py)
- [全部62,256个单元格的实际核对](../outputs/rgbd-affine-full-public-field-audit-001/metrics.json)
- [全部公开字段的 root 保留核对](../outputs/rgbd-affine-full-comparison-001/root_public_summary_review.json)
- [5,252 份消费文件索引](../outputs/rgbd-affine-full-comparison-001/input_hashes.json)
- [desk 6D 小摘要](../outputs/rgbd-affine-full-comparison-001/desk-6d-summary.json)
- [desk 6D 完整帧表](../outputs/rgbd-affine-full-comparison-001/desk-6d-frames.json)
- [desk 8D 小摘要](../outputs/rgbd-affine-full-comparison-001/desk-8d-summary.json)
- [desk 8D 完整帧表](../outputs/rgbd-affine-full-comparison-001/desk-8d-frames.json)
- [xyz 6D 小摘要](../outputs/rgbd-affine-full-comparison-001/xyz-6d-summary.json)
- [xyz 6D 完整帧表](../outputs/rgbd-affine-full-comparison-001/xyz-6d-frames.json)
- [xyz 8D 小摘要](../outputs/rgbd-affine-full-comparison-001/xyz-8d-summary.json)
- [xyz 8D 完整帧表](../outputs/rgbd-affine-full-comparison-001/xyz-8d-frames.json)
- [输入校验分页 1](../outputs/rgbd-affine-full-comparison-001/input-hashes-00.json)
- [输入校验分页 2](../outputs/rgbd-affine-full-comparison-001/input-hashes-01.json)
- [输入校验分页 3](../outputs/rgbd-affine-full-comparison-001/input-hashes-02.json)
- [输入校验分页 4](../outputs/rgbd-affine-full-comparison-001/input-hashes-03.json)
- [输入校验分页 5](../outputs/rgbd-affine-full-comparison-001/input-hashes-04.json)
- [输入校验分页 6](../outputs/rgbd-affine-full-comparison-001/input-hashes-05.json)
- [完整序列管线](../scripts/track_rgbd_affine_full.py)
- [固定串行驱动](../scripts/run_rgbd_affine_full.py)
- [原完整结果独审源码](../scripts/audit_rgbd_affine_full_results.py)
- [纯 JSON 汇总源码](../scripts/summarize_rgbd_affine_full_results.py)
- [只读旧完整 ICP 与 GS 汇总](../outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json)
- [旧 ICP 数值审计证据](../outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json)

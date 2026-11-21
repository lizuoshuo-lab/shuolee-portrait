---
id: "gslab-exp-last-accepted-reference-recovery-20261009"
slug: "last-accepted-reference-recovery-controls"
title: "20 最后接受的参考帧：恢复链路与局部误差"
order: 20
status: "two_CPU_recovery_rounds_completed_independently_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
---

# 拒绝之后，参考图像与参考位姿必须指向同一帧

固定光度与几何目标不变，只把拒绝后的配准参考改成“最后一次接受的传感器帧”，已完成 desk、xyz 两轮真实 CPU 对照。**desk 的整体位置统计改善，但相邻运动误差增大、拒绝从 3 帧增至 4 帧；xyz 的原始位置误差与末帧误差下降，ATE 和两项 RPE 略增。** 这项恢复链路修改没有带来所有指标的共同改善，也不能保证更大跨度的图像配准成功。

新运行各保留 100 帧、99 个待配准对，共 **200 帧、198 对**。加上只读复用的两份旧 `previous-frame` 光度 λ1 结果，公开对照包含 **400 帧、396 对**，所有拒绝均留在轨迹和统计分母中。两份新进程正常退出，没有重试或依据 GT 改动参数。它们是 [实验 19 的固定光度与几何里程计](2026-10-09-photogeometric-real-sequences.md)的参考帧策略对照，不建立高斯地图，不包含回环、重定位或完整 SLAM。

## 传感器参考与位姿前缀为何不能混用

用 `C_k` 表示第 k 帧相机到世界的 c2w，`T_reference_from_current` 把当前相机坐标中的点变到所选参考相机坐标。有限变换的方向与组合为：

```text
p_reference = T_reference_from_current @ p_current
C_current = C_reference @ T_reference_from_current
```

旧策略每次都使用时间上的上一帧 RGB-D。若第 k 帧配准被拒绝，保存的 `C_k` 只是传播 `C_(k-1)`；相机真实移动并不会因此停止。下一步若仍选第 k 帧传感器，再把 `k+1 -> k` 的相对变换接到传播的位姿上，就会把“第 k 帧的图像坐标”与“未包含第 k 帧运动的位姿前缀”放在一起。这样保存失败帧是诚实的，但后续链路可能继续遗漏被跳过的运动。

新策略保存一个 **last accepted SENSOR frame** 及其对应 c2w：配准接受时，同时更新参考传感器与位姿；拒绝时，当前轨迹仍传播上一条估计，同时保留原参考传感器。下一帧直接配准到这张参考图像，再按它自己的 `C_reference` 组合。不会把拒绝候选晋升为成功位姿，也不会依据 GT 找一张参考帧。

每对仍从单位相对变换初始化。被拒绝的候选、当前帧 ID、参考帧 ID、参考跨度、更新前后最后接受帧 ID、全部失败标志和估计 c2w 都保存。这里的跨度是 **关联帧 ID 的差**，不是原始 RGB 文件行号，也不是严格等间隔的时间长度。冻结库遗留的 `candidate_previous_from_current` 字段在新运行中表示“当前到选定 target”的候选；公开记录另保存方向明确的 `candidate_reference_from_current`，两者数值完全一致。

## 非交换的 skip 控制只验证链路代数

设玩具相机 1、2、3 的真实相对变换分别为 `T12`（2 → 1）和 `T23`（3 → 2）。在相机 2 被拒绝后，直接匹配相机 3 到相机 1，所需的有限变换是：

```text
T13 = T12 @ T23
C3 = C1 @ T13
```

错误地递推传播后的 `C2=C1` 再乘 `T23` 会漏掉 `T12`；写成 `T23 @ T12` 则改变了坐标变换的顺序。旋转与平移通常不交换。

[合成参考账本检查](../outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/synthetic_reference_ledger_checks.json)使用 SciPy 矩阵指数构造非交换运动：正确 skip 组合的矩阵最大误差为 0；错误上一帧递推为 0.126748，颠倒乘法顺序为 0.008539。这两个负控数值是 **矩阵元素最大差**，混含旋转和平移，不能解释成米或角度误差。检查还保留共同首帧和全拒绝、连续拒绝后恢复、交替拒绝与接受、末帧拒绝、异常时未知候选的完整 ID 账本。

这些玩具变换只验证状态与组合方向。真实 RGB-D 配准估计不必等于玩具的精确 `T13`，通过此检查不意味着 TUM 中的某次恢复准确。

## 两轮只改变参考策略

[预先锁定的执行协议](../outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/protocol.json) SHA 为 `d2adb632c2261e50bc963e3b84c31d44e92233b3f399bd45a23fafd457ce13c1`。它锁定 10 份当前源码与归档副本，并保留旧协议和旧 λ1 指标文件的哈希。新旧运行都使用 desk、xyz 的关联帧 200–299，同一 RGB、深度、有效掩码、灰度、K 和关联元数据。

| 项目 | 新旧 λ1 固定设置 |
|---|---|
| 首相机、每对初值 | 首 c2w 为 identity；每对相对初值为 identity |
| 层级 / 最大迭代 | 40×30、80×60、160×120；15 / 12 / 10 |
| 光度与几何 | λ_photo=1；几何尺度 0.01 m，灰度尺度 0.1 |
| 鲁棒项 / 支持 | 归一化 Huber δ=1；基查询一圈 4×4 支持，试探不删除对应 |
| 距离门限 / 最终强度门限 | 0.10 / 0.07 / 0.05 m；强度 RMSE ≤0.1 |
| 试探步长 | 1、1/2、1/4、1/8、1/16、1/32 |
| 拒绝处理 | 保留候选与全部 flags；当前轨迹保存传播位姿 |
| 唯一策略区别 | target 始终为上一传感器帧，或保持最后接受的传感器帧 |

权重、尺度与门限仍是工程设置。冻结的 `80160e…` 库、旧 source 和旧结果没有修改；运行配置仅增加或改变三个描述参考策略与方向的字段。没有调整光度模型、曝光补偿、权重或收敛门限，也没有重跑旧参考。

[实际参数解析](../outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/cli_checks.json)、[只读输入与归档检查](../outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/preflight_final_checks.json)和 [独立协议审阅](../outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/root_protocol_review.json)在执行前验证了完整配置、旧 λ1 输入、源码副本、两个输出不存在及合成账本。预检中真实配准和线性求解被禁止。

[两轮实际进程账本](../outputs/rgbd-photogeometric-last-accepted-validation-001/metrics.json)记录按 desk → xyz 顺序串行执行、`returncode=[0,0]`、终态 `all_two_predeclared_CPU_rounds_completed`。每轮在 08:00 北京时间截止前启动；进程退出码不能替代 pair 精度。CPU 与其他 GS 工作可能重叠，因此不作耗时或速度比较。

关联 loader 确实读取 GT 字段与时间关联信息。优化器只接收传感器几何和灰度；全部估计轨迹保存后才访问 GT 相机矩阵进行评测。不能把这一隔离边界写成“程序从未读取 GT”。这两段数据已经用于本地诊断，不是未观察过的最终测试集。

## 四组完整指标：累计误差与相邻误差分别呈现

下表来自 [固定两策略对照摘要](../outputs/rgbd-photogeometric-reference-policy-comparison-001/metrics.json)与四份完整紧凑摘要。平移从米换算为 **mm**，旋转为 **度**。raw 和 ATE 包含全部 100 帧；RPE 包含全部 99 个相邻时间区间，拒绝率分母也是 99。ATE 只做 SE(3) 刚体对齐，不拟合尺度；末帧列为帧 299 的原始首相机锚定位置误差。

| 序列 / target 策略 | raw RMSE mm | SE(3) ATE mm | 相邻 RPE 平移 mm | 相邻 RPE 旋转 ° | 末帧 raw mm | 拒绝 / 99 |
|---|---:|---:|---:|---:|---:|---:|
| desk / 上一帧 | 50.417 | 27.315 | 8.992 | 0.576840 | 127.658 | 3 |
| desk / 最后接受帧 | 33.308 | 19.505 | 10.363 | 0.632563 | 100.643 | 4 |
| xyz / 上一帧 | 30.770 | 17.139 | 6.855 | 0.469759 | 36.879 | 1 |
| xyz / 最后接受帧 | 27.791 | 17.218 | 6.978 | 0.488324 | 26.429 | 1 |

desk 的 raw、ATE、末帧误差下降，平移与旋转 RPE 都增大，新增一次拒绝。xyz 的 raw 与末帧误差下降，ATE 与两项 RPE 略增。没有跨序列平均、最佳 seed 选择、单次运行的样本 SD 或显著性结论。

失败帧冻结位姿，之后接受较大跨度候选，会将估计的跨帧相对变换接入当前前缀。相邻时间 RPE 仍按帧 k−1 → k 计算，因而会同时惩罚冻结区间与恢复后的运动跳变。这个时间分配效应说明累计轨迹改善可以与局部 RPE 增大并存；它不能单独证明某一次重新匹配的候选更准或更差。

![全部四条原始首相机锚定轨迹与拒绝点](../outputs/rgbd-photogeometric-reference-policy-comparison-001/trajectories-and-rejections.png)

图中每条轨迹保留全部 100 帧，包括重复的传播位姿。灰色为 GT，估计为彩色，红点为拒绝；同一序列使用相同坐标范围，没有为画图额外拟合 SE(3) 或 Sim(3)。重复红点可能在空间重叠，因此完整失败数量应看标题与账本。

## 实际恢复事件和更大跨度的失败

| 序列 / 当前帧 | 参考传感器帧 | 关联帧跨度 | pair 判定 | 判定后参考帧 |
|---|---:|---:|---|---:|
| desk 216 | 214 | 2 | accepted | 216 |
| desk 266 | 264 | 2 | accepted | 266 |
| desk 281 | 279 | 2 | rejected | 279 |
| desk 282 | 279 | 3 | accepted | 282 |
| xyz 202 | 200 | 2 | accepted | 202 |

desk 215、265、280 被拒绝后，参考分别停在 214、264、279。281 匹配到 279 又被拒绝，于是 282 仍参考 279，跨度增至 3；这一对通过了固定 pair 判据。xyz 201 被拒绝后，202 直接参考 200。

新 desk 全部拒绝 ID 为 **215、265、280、281**，跨度统计为 gap1×95、gap2×3、gap3×1。新 xyz 拒绝 ID 为 **201**，统计为 gap1×98、gap2×1。全部最终失败标志均为 `finest_level_line_search_no_descent`，未把 281 的新失败删除，也未替换旧参考里的失败。

这里的 **accepted 是工程判据接受一个相对候选**，不是 GT 证明“恢复成功”。更大的图像跨度可能包含更大运动、遮挡、亮度差或更少可用对应；本实验没有把这些可能因素当成已证实的单独原因，也没有依据结果放宽门限。

![完整 raw 误差、参考跨度与失败区间](../outputs/rgbd-photogeometric-reference-policy-comparison-001/position-error-reference-gap-and-flags.png)

左列显示全部帧的事后位置误差，中列显示当前 ID 与参考 ID 的差，右列覆盖所有 201–299 待跟踪帧并标出拒绝。gap3 出现在 desk 282，而不是 281；它表示这次尝试选择 279 作为参考，不表示删掉中间两帧。

## 两种独立审计的范围分开记录

[新运行的参考与轨迹独审](../outputs/rgbd-photogeometric-last-accepted-validation-001/independent_reference_trajectory_audit.json)检查两轮 **200 帧、198 对**：参考 ID、gap、传感器元数据、接受时 `C_reference @ T`、拒绝时精确传播、首相机 identity，以及原始 GT 构造与事后轨迹指标。接受或传播前缀的最大差为 0；独立复算的平移标量差在约 `1e−16 m` 范围，逐区间旋转差小于 `2.4e−14°`。审计复用了实验 19 的独立 raw-GT / SciPy 指标实现，不应称为第三套独立评价公式；它不重新执行配准器。

[公开摘要逐字段审计](../outputs/rgbd-photogeometric-reference-policy-comparison-001/public_summary_field_audit.json)检查新旧四方法的 **400 帧**：所有保留字段、候选、估计、参考 ID 与 gap、flags、各层停止与更新计数，以及五份 compact JSON 和两幅图的 SHA。它验证公开内容忠实于原始产物，不重新计算 GT 指标，不能与上一项审计的范围混为一谈。

四份摘要分别为 [desk 上一帧](../outputs/rgbd-photogeometric-reference-policy-comparison-001/desk-previous-frame-summary.json)、[desk 最后接受帧](../outputs/rgbd-photogeometric-reference-policy-comparison-001/desk-last-accepted-summary.json)、[xyz 上一帧](../outputs/rgbd-photogeometric-reference-policy-comparison-001/xyz-previous-frame-summary.json)、[xyz 最后接受帧](../outputs/rgbd-photogeometric-reference-policy-comparison-001/xyz-last-accepted-summary.json)。[完整 400 帧参考与候选账本](../outputs/rgbd-photogeometric-reference-policy-comparison-001/complete-reference-candidate-ledgers.json)保存每一帧的矩阵和拒绝信息；旧策略的参考 ID 按冻结源代码的上一帧政策派生，新策略的 ID 则直接记录并核对。

这些公开 JSON 每份为 498–653 KB，小于 1 MiB。大型原始 `metrics.json`、`iterations.jsonl`、日志和传感器 NPZ 继续作为本地归档，公开摘要记录其出处与哈希，不复制为网页正文。

可审阅实现包括 [冻结光度与几何库](../scripts/rgbd_photogeometric_odometry.py)、[新参考链路与合成控制](../scripts/track_rgbd_photogeometric_last_accepted.py)、[两轮串行驱动](../scripts/run_rgbd_photogeometric_last_accepted.py)、[只读摘要程序](../scripts/summarize_photogeometric_reference_policy.py)、[参考与轨迹独审程序](../scripts/audit_rgbd_last_accepted_results.py)及其 [独立指标实现](../scripts/audit_rgbd_photogeometric_results.py)。源码和产物固定的这一组片段支持链路解释与混合结果，仍不足以证明长序列恢复、重定位或完整 SLAM 的可靠性。

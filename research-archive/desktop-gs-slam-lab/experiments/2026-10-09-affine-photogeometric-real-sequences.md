---
id: "gslab-exp-affine-photogeometric-real-sequences-20261009"
slug: "affine-brightness-real-sequence-controls"
title: "22 仿射亮度真实对照：八维模型的取舍"
order: 22
status: "two_CPU_affine_rounds_completed_independently_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html"
---

# 增加亮度参数，没有带来所有指标共同改善

在 desk、xyz 的同一段关联帧 200..299 中，固定原光度与几何工程参数、最后接受参考帧策略，再联合估计位姿、相对 gain 和 bias，已完成两组真实 CPU 对照。**desk 的 ATE、相邻帧两项 RPE 和拒绝数量改善，但 raw 平移与末帧误差略差；xyz 的 raw、ATE、平移 RPE 和末帧误差略差，旋转 RPE 较好，拒绝数量不变。** 不能将它概括成“加入仿射亮度整体改善”，也没有证明旧拒绝由光照变化造成。

两份新进程均正常退出，未重试，保存 **200 帧、198 个待配准对**；与只读复用的两份旧六维 last-accepted 光度 λ1 结果一起，公开对照保留 **400 帧、396 对**。所有被拒绝的帧及其候选仍在分母中。这是 [实验 20 的最后接受参考帧里程计](2026-10-09-last-accepted-reference-recovery.md)上的模型对照，不建立高斯地图，不包含回环或完整 SLAM；这两段已经被前面的实验观察过，属于同序列诊断，不是独立未见测试集。

## 固定了什么，八维模型又改变了什么

两组仍使用相同的 RGB、深度、有效深度 mask、灰度和 K，分辨率依次为 40×30、80×60、160×120。灰度保持 `0.299R + 0.587G + 0.114B`。深度、法向、距离、唯一目标像素和一格双线性支持边界等筛选与旧库相同；几何尺度 `σ_geo=0.01 m`、灰度尺度 `σ_I=0.1`、光度权重 1、归一化 Huber δ=1、最终灰度 RMSE 门槛 0.1 均未改变，也没有使用真实 GT 调参。

新模型每对同时求解：

```text
state = [vx, vy, vz, wx, wy, wz, a, b]
r_photo = exp(a) * I_reference(project(T_reference_from_current * p_current))
          + b - I_current
```

前三维以米计，旋转扰动以弧度计；`a` 是无量纲的相对 log gain，`b` 使用归一化灰度强度单位。位姿采用左更新 `exp(delta_xi) @ T`，a、b 加性更新。几何雅可比末两列为零；光度位姿雅可比乘 `exp(a)`，新增两列分别为 `exp(a)*I_reference` 与 1。严格双线性梯度和八维有限差分的数学边界见 [仿射亮度与可观测性](../knowledge/15_affine_brightness_observability.md)。

**每对重新初始化 T 为单位矩阵、a=0、b=0，即 gain=1。** gain/bias 只描述当前与所选参考图像的一次相对校正，既不累计为跨帧曝光 gauge，也不作为相机位姿前缀。它不是辐射标定，原 sRGB 灰度也没有因此变成线性光强。`exp(a)` 不设 clamp 或先验；非有限溢出会让整次 trial 无效，不能只忽略坏像素再选择它。

新算法要求列归一化后的联合系统 rank=8，使用相同的条件数门槛，额外检查 a、b 的小更新收敛。原六维模型只要求 rank=6。新增 nuisance 参数、rank、停止条件以及随后变化的候选、支持集合和参考路径，都属于这个八维模型的整体变化。即使输入和第一对的单位变换支持完全相同，也不能称后续每一步只改变了亮度，或把结果差异解释为纯 brightness 因果。

参考策略保持 last accepted SENSOR frame：当前点先变到被选参考相机坐标，再通过对应参考 c2w 组合：

```text
C_current = C_reference @ T_reference_from_current
```

接受时同时更新参考图像和其位姿；拒绝时保存完整候选、gain/bias、全部标志，并传播上一条估计，保留旧参考。参考传感器与位姿前缀始终指向同一 ID。关联 loader 会读取 GT 字段、矩阵和时间以完成匹配，这一点没有隐藏；GT 矩阵进入指标评测是在全部 100 条估计已经保存之后，不进入优化目标、初值、参考选择或参数选择。

## 四份完整结果

raw 是第一相机 gauge 中的相机中心直接误差，不拟合；ATE 对相机中心做一次无尺度 SE3 对齐；RPE 使用相邻关联帧，包含拒绝后传播的位姿。表中三项平移 RMSE 与末帧位置误差由原米值乘 1000，显示为 **mm**；旋转 RPE 为 **度**。每份位置统计含 100 帧，RPE 和拒绝比例的待配准分母为 99 对。

| 序列 | 模型与参考策略 | raw RMSE / mm | SE3 ATE / mm | RPE 平移 / mm | RPE 旋转 / ° | 末帧 raw / mm | 拒绝 / 99 |
|---|---|---:|---:|---:|---:|---:|---:|
| desk | 旧六维，last-accepted | 33.308 | 19.505 | 10.363 | 0.632563 | 100.643 | 4 |
| desk | 新八维 affine，last-accepted | 34.302 | 19.261 | 9.301 | 0.611580 | 101.775 | 3 |
| xyz | 旧六维，last-accepted | 27.791 | 17.218 | 6.978 | 0.488324 | 26.429 | 1 |
| xyz | 新八维 affine，last-accepted | 28.326 | 18.024 | 7.192 | 0.465887 | 29.223 | 1 |

desk 的 raw RMSE 增大约 0.994 mm，ATE 下降约 0.244 mm；相邻平移 RPE 下降约 1.063 mm，末帧 raw 增大约 1.132 mm。对齐后位置统计较好，不能抵消原 gauge 与终点的退化。xyz 的 raw、ATE、平移 RPE 分别增大约 0.535、0.805、0.214 mm，终点增大约 2.795 mm；其旋转 RPE 下降约 0.022436°。

这里说“旋转较好”仅指相邻 RPE。raw 旋转 RMSE 反而都增大：desk 从 2.812691° 到 2.858436°，xyz 从 1.027424° 到 1.075182°。没有跨序列求一个平均排名，没有单种子标准差或速度比较。

![四条估计轨迹与 raw GT](../outputs/rgbd-affine-photogeometric-comparison-002/trajectory-comparison.png)

图为首相机 gauge 中的 XY 投影，没有按 ATE 结果重新摆放。交叉标记表示拒绝后保留的估计位置，并列出实际帧 ID。旧 desk 的 280、281 连续拒绝继承相同位姿，因此在图中位置重合，标签合并为 `280,281`；两条拒绝记录都保留。三维 raw 指标不能只从 XY 投影判断。

## 拒绝换了位置，并未消失

| 序列与模型 | 完整拒绝帧 ID | 最终失败标志 | 下一次较大跨度尝试 |
|---|---|---|---|
| desk，旧六维 | 215、265、280、281 | 四次 `finest_level_line_search_no_descent` | 包含 282→279 的跨度 3 |
| desk，新八维 | 218、225、268 | 三次 `finest_level_line_search_no_descent` | 219→217、226→224、269→267，均跨度 2、pair accepted |
| xyz，旧六维 | 201 | 一次 `finest_level_line_search_no_descent` | 202→200，跨度 2、pair accepted |
| xyz，新八维 | 266 | 一次 `finest_level_line_search_no_descent` | 267→265，跨度 2、pair accepted |

新结果的失败判据均是最细层严格 line search 未找到下降；99 个最终候选在各自八维支持上保存的联合 rank 都为 8，仍有被拒绝的候选。rank=8 只表示当前线性系统在声明门槛下满秩，不能保证严格下降、位姿准确或所有后续配准成功。这里未重新构建真实对应点矩阵来独立证明这些 rank，也没有通过放宽阈值接受失败。

旧拒绝帧在新模型中被接受，并不能证明“修好了这些帧的光照”：新增参数会改变前面已经接受的运动、后续参考、支持与 trial 路径；同时新模型又拒绝了别的帧。`pair accepted` 表示工程判据接受这个有限变换，不能单独证明跳过运动按 GT 正确恢复。

![完整误差、每对 gain/bias 与参考跨度](../outputs/rgbd-affine-photogeometric-comparison-002/errors-brightness-reference-gaps.png)

第一行保留两种模型全部 100 帧的 raw 平移误差，以 mm 显示。中间两行是新模型每对独立估计的 gain 与 bias，含被拒绝候选；锚帧没有虚构一条亮度估计。底行为当前关联帧与 last-accepted 参考的 ID 差。图中不把 gain 曲线解释成连续曝光标定轨迹。

| 新模型候选统计 | desk，99 对 | xyz，99 对 |
|---|---:|---:|
| gain 均值 | 1.008801 | 1.003271 |
| gain 最小～最大 | 0.923751～1.091571 | 0.960050～1.071109 |
| bias 均值 | −0.005572 | −0.002257 |
| bias 最小～最大 | −0.049390～0.025242 | −0.015080～0.022993 |

以上统计包含拒绝候选，分母均为 99；并非只统计通过的 pair 或从 GT 筛选出来的亮度。非零 a、b 表示模型拟合出的相对校正，不能直接归因于曝光，也可能与位姿、遮挡、非朗伯反射、插值和灰度模型误差耦合。

## 独立审计的范围

[实际产物独立审计](../outputs/rgbd-affine-photogeometric-real-validation-001/independent_affine_result_audit.json)通过，检查了两份新终态、200 条帧记录、198 条参考与位姿组合链，保留 4 次拒绝。6 份原始索引、400 张 RGB/深度 PNG、30 份实际三层数组、15 份源码及其归档副本重新核对 SHA；灰度从 RGB 独立复算，原始 GT 的 xyzw 姿态由 SciPy 重建并按第一相机 gauge 锚定。

这里明确复用先前独立实现的 e807 原始 GT/SciPy/generic solve/`align_vectors` 指标 helper，而不是宣称又写了第三套独立指标公式。新加入的真实产物审计使用独立 SciPy `expm` 检查左 SE3 方向、8 维 proposed/accepted delta、a/b 加性 trial、每步状态连接和候选终态。**198 对共有 5,422 条已保存的 step 行、5,835 次 trial、5,074 次接受更新**，均被遍历。独立 SE3 trial 最大矩阵元素差为约 `3.37e−16`；轨迹指标与原记录的最大数值差约 `2.05e−14`，该总括量混合米和度，只用于说明复算一致性，不能当作一种新的误差单位。

trial 目标下降、固定支持分母、最终 rank、条件数和 residual 数值只进行了**持久记录的一致性**检查；未重新采样真实对应点、重建 SVD 或复算每个真实光度/几何目标。因此这部分范围与 toy 中八维雅可比有限差分、真实轨迹的独立 GT 复算不同。真实配准不重跑，不用 GT 重新选候选，不修改原记录。

GT 隔离依赖冻结源码审阅和执行次序，单靠产物中的 `GT_used=false` 或 SHA 不能证明运行时因果。实际候选、每层停止、每次 trial 和异常均留在本地原始记录；如果未来出现数值异常而库没有返回候选，必须保留“candidate unknown”及异常标志，不能猜测一个成功状态。本次两轮没有这种未知候选。

## 可复核的小体积证据

核心证据如下，所有公开 JSON 均小于 1 MiB。四份 compact 各保留 100 条帧记录及完整候选、参考、gain/bias、rank、flags；大型每次 trial 记录留在冻结的本地源产物中。

- [冻结真实 CPU 协议与 15 份源码锁](../outputs/rgbd-affine-photogeometric-real-cpu-preparation-001/protocol.json)
- [协议与数据的 root 独立审阅](../outputs/rgbd-affine-photogeometric-real-cpu-preparation-001/root_protocol_review.json)
- [两组实际 exit0 串行 ledger](../outputs/rgbd-affine-photogeometric-real-validation-001/metrics.json)
- [两组独立参考、八维 step 与轨迹审计](../outputs/rgbd-affine-photogeometric-real-validation-001/independent_affine_result_audit.json)
- [四方法指标、完整分母与小产物 SHA](../outputs/rgbd-affine-photogeometric-comparison-002/metrics.json)
- [desk 旧六维完整摘要](../outputs/rgbd-affine-photogeometric-comparison-002/desk-old6-summary.json)；[desk 新八维完整摘要](../outputs/rgbd-affine-photogeometric-comparison-002/desk-affine8-summary.json)
- [xyz 旧六维完整摘要](../outputs/rgbd-affine-photogeometric-comparison-002/xyz-old6-summary.json)；[xyz 新八维完整摘要](../outputs/rgbd-affine-photogeometric-comparison-002/xyz-affine8-summary.json)
- [全部 400 帧的参考、候选、估计与 gain/bias ledger](../outputs/rgbd-affine-photogeometric-comparison-002/complete-reference-candidate-ledgers.json)
- [全部 400 帧公开字段、历史计数与候选账本复核](../outputs/rgbd-affine-photogeometric-comparison-002/root_public_field_audit.json)，对应[只读字段审计源码](../scripts/audit_affine_public_fields.py)
- [冻结仿射亮度库](../scripts/rgbd_affine_photometric_odometry.py)、[真实配准入口](../scripts/track_rgbd_affine_photogeometric.py)、[两轮串行驱动](../scripts/run_rgbd_affine_photogeometric_validation.py)
- [只读实际产物 auditor](../scripts/audit_rgbd_affine_real_results.py)、[只读 compact 与图生成器](../scripts/summarize_rgbd_affine_real_results.py)
- [数学控制的独立审计](../outputs/rgbd-affine-photometric-cpu-controls-001/root_independent_math_audit.json)

最终摘要 `metrics.json` SHA 为 `673387df3c4e7de9218dfd9dbd68bb1484e570fa072e0b527bc3eb3d7293e5b2`。目录 `comparison-001` 保留第一次图稿；最终 `comparison-002` 将 XY 中完全重合的旧 desk 280、281 标签合并显示，没有修改任何真实结果或数值摘要内容。

大型源指标、完整 iterations、日志和 NPZ 本地归档为 `outputs/tum-desk-affine-photogeometric-lastaccepted-transfer100-001/`、`outputs/tum-xyz-affine-photogeometric-lastaccepted-transfer100-001/` 与 `outputs/rgbd-affine-photogeometric-real-validation-001/`，不通过公开网页搬运它们。对这次固定实验，新增 brightness nuisance 的价值与代价应分别看 raw、ATE、RPE、终点、接受判据和失败链路，现有证据没有支撑总体优胜、光照根因修复或泛化成功。

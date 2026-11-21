---
id: "gslab-exp-photogeometric-real-sequences-20261009"
slug: "fixed-rgbd-photogeometric-real-sequence-controls"
title: "19 光度与几何真实序列：整体误差、局部运动和拒绝帧"
order: 19
status: "four_CPU_rounds_completed_independently_audited"
updated: "2026-10-09"
sources:
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/rgbd_odometry.html"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
---

# 光度项降低整体位置误差后，为什么仍不能说跟踪全面改善

固定 RGB-D 光度与几何程序已完成四轮真实 CPU 对照：TUM fr1/desk、fr1/xyz 各取关联帧 200–299，分别运行 `λ_photo=0` 与 `λ_photo=1`。四轮共保留 **400 条帧记录、396 对相邻帧、4 个被拒绝的相对位姿**；四个进程均正常退出，拒绝帧没有从轨迹或误差分母中删除。

在这两个片段中，加入光度项后的原始位置 RMSE、SE(3) 对齐 ATE 和旋转 RPE 均低于新程序的 λ0 结果；**平移 RPE 却升高，末帧位置误差也更大**。desk 还有三次拒绝，xyz 有一次。它们支持的是这组固定输入、固定参数下不同指标的实测变化，不能概括成所有跟踪性质都改善。

本报告连接 [光度与几何里程计的数学说明](../knowledge/13_photometric_geometric_odometry.md)与真实采集数据。实现只做相邻 RGB-D 帧的里程计，不建立高斯地图，不包含回环、重定位或全局 BA，也不是 Open3D Hybrid RGB-D Odometry 或完整 SLAM 的复现。

## 对照固定了哪些输入和参数

[实际执行协议](../outputs/rgbd-photogeometric-real-cpu-preparation-003/protocol.json)在四轮运行前锁定源码、两段数据、数组和参数。两个片段分别约 3.300028 s、3.300125 s；帧号指共享 loader 筛选后的关联帧，不是原始 RGB 文件行号，也不是完整的 549 / 748 帧序列。

每对配准都从当前相机到上一相机的单位相对变换开始，片段的首个估计相机为 identity。没有用 GT 作为初始位姿、损失或候选选择依据。关联 loader 确实读取 GT 的索引、时间戳和位姿字段；优化只接收预存 RGB、深度、有效掩码、灰度与内参数组。GT 位姿量度在全部 100 个估计保存后执行，不能把“未传入优化器”改写成“程序从未读取 GT”。

| 项目 | 四轮固定设置 |
|---|---|
| 片段 | desk 与 xyz 的关联帧 200–299，各 100 帧 |
| 三层分辨率 | 40×30 → 80×60 → 160×120 |
| 每层最大迭代数 | 15 / 12 / 10 |
| 几何对应距离门限 | 0.10 / 0.07 / 0.05 m |
| 深度、RGB / GT 时间关联 | 最近时间样本，严格小于 20 ms；深度不复用；32 Hz 上限 |
| 归一化尺度 | 几何 0.01 m，灰度 0.1 |
| 鲁棒项 | 归一化 Huber δ=1 |
| 光度权重 | 预先指定 λ0、λ1，各运行一次 |
| 双线性支持 | 初始单元外扩一圈，检查 4×4 深度与法向支持 |
| 试探步长 | 1、1/2、1/4、1/8、1/16、1/32 |
| 最终强度门限 | λ1 时强度 RMSE 不超过 0.1；λ0 关闭这一门限 |
| 拒绝处理 | 保留候选与失败记录，估计 c2w 传播上一帧 |

权重、尺度和门限是工程设置，不是经过噪声标定的协方差。没有依据这四轮的 GT 结果重新选权重、帧段或接受门限。desk、xyz 的这两个片段已经用于本地诊断，也不应称为从未观察过的最终测试集。

同一序列的 λ0、λ1 使用相同的预存传感器数组与单位初始对应检查；运行后位姿不同，后续对应、支持范围和线搜索路径可以不同。因此不能把这项对照描述为“整个优化过程始终使用完全相同的对应”。

## 新 λ0 与旧 ICP 是两个不同的参考

旧 `track_rgbd_icp.py` 的两份 100 帧结果作为只读参考加入表格，没有重新训练或重跑。新 λ0 仍执行双线性支持筛选、试探域约束和新程序的求解、检查流程，只是光度损失权重为零；它不等于旧 ICP 的逐点或逐字节复现。

所以有两个不同的问题：比较新 λ0 与 λ1，观察同一套新程序加入光度项后的变化；比较新程序与旧 ICP，观察整个实现配置的差异。不能将新 λ0 与旧 ICP 的差值归因于“光度权重已经生效”。旧 ICP 的公式、退化和失败定义见 [ICP 可观测性说明](../knowledge/10_rgbd_icp_observability.md)。

## 六组完整结果：位置统计和相邻运动要同时看

下表来自 [六方法比较摘要](../outputs/rgbd-photogeometric-comparison-001/metrics.json)及各方法完整的 100 帧紧凑摘要。所有平移列都从米换算为 **mm**；旋转列为 **度**。raw、ATE 使用全部 100 个位置，RPE 使用全部 99 对相邻帧；拒绝率以 99 个需要跟踪的非首帧为分母。

| 序列 / 方法 | Raw 位置 RMSE (mm) | SE3 ATE (mm) | Δ1 RPE 平移 (mm) | Δ1 RPE 旋转 (°) | 末帧 Raw 位置误差 (mm) | 拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| desk / 旧 ICP | 64.103 | 30.627 | 5.000 | 0.75891 | 92.678 | 0/99 |
| desk / 新 λ0 | 81.276 | 37.428 | 5.286 | 0.76197 | 121.987 | 0/99 |
| desk / 新 λ1 | 50.417 | 27.315 | 8.992 | 0.57684 | 127.658 | 3/99 |
| xyz / 旧 ICP | 41.166 | 20.286 | 3.643 | 0.61778 | 21.560 | 0/99 |
| xyz / 新 λ0 | 41.113 | 22.678 | 3.990 | 0.62792 | 14.409 | 0/99 |
| xyz / 新 λ1 | 30.770 | 17.139 | 6.855 | 0.46976 | 36.879 | 1/99 |

**Raw** 直接比较共同首相机坐标规范下的相机中心，不拟合对齐。GT 先统一变换为 `inverse(GT_200) @ GT_k`，估计轨迹从 identity 累积。**ATE** 再事后拟合一次全局旋转与平移，使整段估计位置对齐 GT；尺度固定为 1。这是指标计算，不会回写估计轨迹，也不会令下面的未对齐轨迹图变好。

**Δ1 RPE** 衡量相邻两帧的相对运动误差。使用 c2w 时，定义是：

```text
ΔGT  = inverse(GT_(k−1)) @ GT_k
ΔEST = inverse(EST_(k−1)) @ EST_k
E_k  = inverse(ΔGT) @ ΔEST

RPE_translation = RMSE(norm(E_k.translation))
RPE_rotation    = RMSE(angle(E_k.rotation))
```

desk 的新 λ1 将 raw 从 81.276 mm 降到 50.417 mm，ATE 从 37.428 mm 降到 27.315 mm，同时平移 RPE 从 5.286 mm 升到 8.992 mm。xyz 出现相同方向的指标变化。整体位置统计与局部相对运动描述不同的性质：局部误差会随轨迹积分相互叠加或部分抵消，事后 ATE 还去除了一个整体刚体偏差，因此两种趋势可以同时存在。

平均误差下降也不意味着终点更准。desk 新 λ1 的第 299 帧 raw 误差为 127.658 mm，高于新 λ0 的 121.987 mm，也高于旧 ICP 的 92.678 mm；xyz 新 λ1 的末帧 36.879 mm 同样高于另外两组。本次没有跨序列求均值，没有重复试验的标准差，也没有用一个指标覆盖这些局限。

## 图中保留共同首相机规范和全部拒绝帧

![两序列六方法未对齐轨迹与拒绝位置](../outputs/rgbd-photogeometric-comparison-001/trajectories-and-rejections.png)

每一行是一段序列，三列依次为旧 ICP、新 λ0、新 λ1。同一行使用相同坐标范围与视角；灰色是首相机归一化 GT，蓝绿色是实际累积估计，红点是被拒绝后保留的估计位置。图没有 SE3 / Sim3 拟合，两个序列的坐标系与绘图尺度独立，不能按上下两行的视觉距离判断方法强弱。

![逐帧原始位置误差与完整拒绝记录](../outputs/rgbd-photogeometric-comparison-001/position-errors-and-rejected-segments.png)

左侧是全部 100 帧的 raw 三维位置误差；右侧灰条对应请求跟踪的帧 201–299，红条标出拒绝帧。desk λ1 在较长一段中位置误差较低，但接近末尾时明显上升；只展示整段 RMSE 会隐藏这个变化。红条表示程序拒绝了一次候选，不表示 GT 宣告那一帧不可跟踪。

## 四个拒绝保留下了什么

新 λ1 在 desk 的 215、265、280 帧，以及 xyz 的 201 帧返回 `finest_level_line_search_no_descent`；新 λ0 两轮没有拒绝。四轮合计的 4/396 是本批次的拒绝计数，不作为跨场景性能评分。拒绝发生后，轨迹递推为：

```text
接受：C_k = C_(k−1) @ T_(k−1←k)_candidate
拒绝：C_k = C_(k−1)
```

失败候选、对应数量、每层停止原因和失败标记仍保留。传播上一估计不代表相机实际静止；下一对仍使用相邻采集帧，程序没有用 GT 重新定位，也没有自动补回被拒绝的那一段运动。独立审计逐帧复算接受组合与拒绝传播，实际矩阵逐项差为 0。

### 实际触发的是严格下降规则，并非支持域失效

[只读拒绝诊断](../outputs/rgbd-photogeometric-comparison-001/rejection-diagnostic.json)重建了四个被拒绝帧对中已保存的 **118 个迭代步骤、178 次线搜索试探**，没有执行新的配准、线性求解或 GT 评估。重算目标与保存目标的最大差为 0，重建候选矩阵与保存矩阵的逐项差也为 0。它复用冻结库建立基准对应和 SE3 变换，再用单独的 NumPy 公式核对试探支持与 Huber 目标，不是另一套独立里程计算法。

178 次保存试探均没有支持域无效的情况。[实际最终门限与终止判据复核](../outputs/rgbd-photogeometric-comparison-001/rejection-diagnostic-source-review.json)再次检查了四个最终拒绝点的 rank、条件数、几何残差、光度残差、运动与对应率门限，均通过；真正触发最终失败标记的是：最细层最后一次迭代的六个有限、有效试探，其目标值全部大于或等于当前值，未满足代码要求的严格 `<`。这四个拒绝不能写成“无有效像素”“单平面退化”或“强度 RMSE 超限”。

| 序列 / 拒绝帧 | 最后六次试探的最小目标增量 | 未接受候选与实际传播位置的距离 (mm) |
|---|---:|---:|
| desk / 215 | +5.20355×10⁻¹¹ | 14.176 |
| desk / 265 | +5.66627×10⁻¹¹ | 19.764 |
| desk / 280 | +5.03445×10⁻¹⁰ | 21.360 |
| xyz / 201 | +8.61978×10⁻¹¹ | 10.725 |

目标增量属于按 0.01 m / 0.1 强度归一化后的联合 Huber 目标，不能当成米。候选距离只描述未被接受的另一份矩阵与实际输出的位置差，没有用 GT 证明候选更好，也没有把它替换进轨迹。

在这些帧对里，较早的迭代已经产生过下降更新和非 identity 候选。最后一次试探没有下降，仍使整个帧对被拒绝；因此“存在已接受的内部迭代”与“最终接受相对位姿”是两层规则。以 desk 265 为例，末次平移增量为 `9.17764e-7 m`，旋转增量为 `1.06840e-6 rad`。代码要求两者同时小于各自的 `1e-6` 阈值才按小增量停止；该记录只满足平移条件，不能直接称作“已按当前规则收敛”。

最后这些试探虽然很小，每次仍有 1–7 个查询点改变了所在的双线性单元。基准点的方向导数为负，不足以保证这六个跨单元的有限步都下降，这也连接了知识页中的分段可微边界。不过，极小的正目标差与跨单元计数都不能单独证明根因是舍入、梯度错误、曝光变化或某个停止阈值。诊断没有修改阈值、追加更小步长、重跑优化或按 GT 选择反事实候选；可访问的 [诊断源码](../scripts/diagnose_photogeometric_rejections.py)和 JSON 保留了这一范围。

## 独立审计确认的层次

[只读结果审计](../outputs/rgbd-photogeometric-comparison-001/independent_result_audit.json)状态为 `passed`；对应 [独立审计源码](../scripts/audit_rgbd_photogeometric_results.py)的 SHA256 为 `e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53`。该程序没有导入里程计 solver、数据 loader 或既有轨迹指标函数，没有重新配准。

输入核查覆盖两段数据的 **6 个索引文件、400 个原始 PNG 文件、30 个实际传感器数组**。30 个数组是两序列、三层分辨率、RGB / depth / valid / gray / K 五类数据，包含批量帧数组与内参矩阵，不是 30 个抽样帧。审计核对文件 hash，以及数组 shape、dtype、字节数、C 顺序原始字节 hash，并重新确认灰度来自固定 RGB 线性加权。

结果核查覆盖新四轮全部 **400 条帧记录、396 对相邻帧、4 条拒绝记录**。它从原始 TUM `xyzw` 重建 GT，用 SciPy 旋转、通用矩阵求解和独立的对齐调用重算指标，检查紧凑摘要保留字段、候选积分与失败传播。六份紧凑摘要的文件 hash 也全部核对；旧 ICP 是冻结参考，不把这一步文件校验说成重新执行了旧 ICP 算法。

与原始记录相比，独立计算的差分按单位分开报告：

| 核查量 | 四轮最大绝对差 | 单位与范围 |
|---|---:|---|
| 平移标量和逐帧 / 逐区间平移误差 | 1.32620×10⁻¹⁵ | m；最大值来自 xyz λ1 的逐帧对齐误差 |
| 旋转标量和逐区间旋转误差 | 2.54241×10⁻¹⁴ | °；最大值来自 desk λ1 的相邻旋转误差 |
| 接受组合 / 拒绝传播矩阵 | 0 | 逐项精确一致；不当作带单一物理单位的误差指标 |

这些差分说明保存产物与独立指标实现一致，不是新的传感器精度，更不是另一个独立跟踪算法证明当前运动估计正确。原比较脚本与旧 ICP 原本共享 `audit_pose_arrays`；这里明确分开新增的独立指标重建，避免把同一函数调用两次当成独立证明。

GT 没有参与优化的证据来自冻结代码的数据流审阅、GT-free 输入包以及已保存的执行顺序；hash 本身只能证明字节关联，不能独自证明运行时的因果隔离。最近 GT 没有插值或运动补偿，RGB 与深度也存在非零时间差，毫米级结果需按这套测量约定理解。

## 可下载的协议、摘要与实现

网站仅提供便于复核的协议、批次记录、紧凑摘要、图和源码。完整原始 metrics、每次迭代日志、NPZ 与原图保留在本地，不作为本页公共附件。

- [固定协议 003](../outputs/rgbd-photogeometric-real-cpu-preparation-003/protocol.json)：约 238 KB；包括四轮配置、输入 hash 和来源关联。
- [四轮终态批次记录](../outputs/rgbd-photogeometric-real-validation-001/metrics.json)：约 17 KB；四轮 `returncode=0`，其中两轮正常完成且保留拒绝。
- [六方法指标摘要](../outputs/rgbd-photogeometric-comparison-001/metrics.json)与 [独立结果审计](../outputs/rgbd-photogeometric-comparison-001/independent_result_audit.json)。
- [四个拒绝帧对的保存试探诊断](../outputs/rgbd-photogeometric-comparison-001/rejection-diagnostic.json)：约 331 KB；保留全部 118 步、178 次试探的重建和分类。
- desk 的 [旧 ICP](../outputs/rgbd-photogeometric-comparison-001/desk-old-ICP-summary.json)、[新 λ0](../outputs/rgbd-photogeometric-comparison-001/desk-photo0-summary.json)、[新 λ1](../outputs/rgbd-photogeometric-comparison-001/desk-photo1-summary.json)完整帧摘要。
- xyz 的 [旧 ICP](../outputs/rgbd-photogeometric-comparison-001/xyz-old-ICP-summary.json)、[新 λ0](../outputs/rgbd-photogeometric-comparison-001/xyz-photo0-summary.json)、[新 λ1](../outputs/rgbd-photogeometric-comparison-001/xyz-photo1-summary.json)完整帧摘要。六份各约 415–499 KB，保留全部 100 帧、失败标记与每层停止信息。
- [数学与配准实现](../scripts/rgbd_photogeometric_odometry.py)、[固定输入跟踪入口](../scripts/track_rgbd_photogeometric.py)、[四轮执行器](../scripts/run_rgbd_photogeometric_validation.py)、[摘要与绘图脚本](../scripts/summarize_rgbd_photogeometric.py)、[独立结果审计脚本](../scripts/audit_rgbd_photogeometric_results.py)。

原始协议 SHA256 为 `81325bceecd336b678e8cadda1af4bf926fd3c7e5788acbb0db128524ccc265b`。网站可移植副本会清理本地路径并可能改变 JSON 格式，因此原文件 hash 与网站返回字节的 hash 各有用途，不能假定二者相同。

这里观察到的是两个 Freiburg1 片段的一次固定参数对照。非恒定曝光、相机响应、模糊、遮挡与分段有效域仍可能破坏光度假设；四个真实拒绝也不能自动归因于合成控制中的 `+0.2` 亮度偏移。本次 CPU 作业可能与其他 GS GPU 作业共享机器资源，记录的耗时没有用于 FPS、方法速度排名或跨序列平均性能结论。

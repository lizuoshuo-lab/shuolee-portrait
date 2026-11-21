---
id: "gslab-exp-fast-livo2-hilti-gt-20261010"
slug: "fast-livo2-hilti-gt"
title: "31 FAST-LIVO2：Hilti GT、输出阶段与时间对照"
order: 31
status: "completed_local_two_mode_dense_reference_comparison"
updated: "2026-10-10"
sources:
  - "https://hilti-challenge.com/dataset-2022"
  - "https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022/tree/e62017f907007fdc5ab8c721842e4ae7359d7f49"
  - "https://github.com/Hilti-Research/hilti-slam-challenge-2022/blob/e4aacf7c111ab832a0bdffe3f069255454068540/README.md"
  - "https://github.com/Hilti-Research/hilti-slam-challenge-2022/blob/e4aacf7c111ab832a0bdffe3f069255454068540/evaluation-2022/evaluation.py"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/src/LIVMapper.cpp"
---

# FAST-LIVO2：Hilti GT、输出阶段与时间对照

带 GT 的融合比较首先要确认三个问题：轨迹描述哪个物理点、时间戳对应哪个观测时刻、保存的是哪一阶段的状态。**本机已实际完成 Hilti2022 `exp14_basement_2` 的 LIO／LIVO 全 bag 半速回放，两次正常退出；冻结评测 child 与 wrapper 均实际退出 0。** 在相同 687 个原始 dense 参考时刻，尺度固定 1 的 SE(3) 位置 ATE RMSE 为 LIO **3.570770 cm**、LIVO **4.132461 cm**；当前结果没有体现 LIVO 的定位增益。

本轮只比较一个地下室序列、每模式一次运行。Dense 轨迹是发布方的开发／调试参考，不继承四个 sparse surveyed 点的毫米级精度。[真实评测指标](../outputs/fast-livo2-hilti-gt-001/evaluation/metrics.json)、[评测执行收据](../outputs/fast-livo2-hilti-gt-001/evaluation/execution.json)和[完成清单](../manifests/fast-livo2-hilti-gt.json)已保存；[349 项独立读回](../outputs/fast-livo2-hilti-gt-001/actual-paired-independent-review.json)及[复核说明](../outputs/fast-livo2-hilti-gt-001/actual-paired-independent-review.md)确认原始轨迹、687 个逐点误差和 676 对 RPE、配置与实际退出。原有无 GT 回放和[SplaTAM 完整重复／room 负结果](30_splatam_tracking_ablation_repeat.md)保留。

## 同时间、同物理点、同输出阶段的实际比较

主对照使用 IMU body 位姿：LiDAR 更新后、本轮 VIO 前，时间戳为 `last_lio_update_time`。LIVO 包含此前视觉更新的历史状态，两模式分别事后 SE(3) 刚体对齐，不拟合尺度或时间偏移。实际参数除模式与输出身份外一致，GT 没有进入估计器。

| 同一共同参考时间支持，主轨迹 | LIO | LIVO |
|---|---:|---:|
| 原始传感器时间位姿行数 | 739 | 738 |
| 各自支持的原始参考查询／689 | 688 | 687 |
| 共同有效参考时刻数 | 687 | 687 |
| SE(3) 位置 ATE RMSE，cm ↓ | 3.570770 | 4.132461 |
| 首共同位姿参考系位置 RMSE，cm ↓ | 10.835310 | 24.764743 |
| 676 对共同参考相邻平移 RPE RMSE，cm ↓ | 0.338871 | 0.350999 |
| 同 676 对旋转 RPE RMSE，° ↓ | 0.145493 | 0.300109 |
| 末共同点 SE(3) 对齐位置误差，cm ↓ | 4.591927 | 5.206272 |
| 末共同点首共同位姿参考系误差，cm ↓ | 19.531173 | 34.778906 |

![FAST-LIVO2 两模式在共同原始 dense 参考时刻的位置误差，两种 gauge 分别展示](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/fast-livo2-hilti-gt-001/evaluation/paired-reference-errors.png)

左图对每条完整轨迹各求一次最佳刚体对齐；右图只用首共同位姿固定坐标约定，保留之后的累计偏离。整体对齐后的厘米级 ATE 没有消除右图中的后段偏离，也不代表系统主动做了回环纠偏。LIVO 在部分区间的左图误差低于 LIO，但整段 ATE、RPE 和首位姿参考系 RMSE 都更高，不能挑某个片段宣布整体更好，也不能由这一次结果推断视觉融合普遍无效。

[回放终端观察](../outputs/fast-livo2-hilti-gt-001/replay-terminal-observations.json)记录实际退出与一次前置失败。首个 LIVO 尝试在私有端口 11321 绑定时遇到 `OSError(98)`，fusion 与 player 尚未启动，没有算法轨迹；失败目录保留。随后以相同冻结 runner、唯一目录和 `--port 11322` 串行执行，实际回放成功。这是运行资源冲突的恢复，未改变算法源码、GT 协议或把失败模型挑出精度统计。

## 输入通过，意味着什么已经确认

数据来自 [Hilti 官方页面](https://hilti-challenge.com/dataset-2022)链接的 Hilti-Research 发布库，固定数据提交 `e62017f907007fdc5ab8c721842e4ae7359d7f49`。完整 bag 为 6,260,771,085 字节，SHA256 与固定发布记录一致；此前离线读取全部 45,082 条记录，消息计数与 bag 索引一致，所需话题类型、头时间戳、GT 区间和标定配置通过。[完整获取与审核收据](../outputs/fast-livo2-gt-input-recovery-003/receipt.json)保留原失败传输、精确 Range 恢复、拼接及前后哈希。该输入审核未启动 ROS，后续实际回放由独立执行收据记录。

| 此序列的实际输入 | 数量／含义 |
|---|---|
| Hesai XT32 `/hesai/pandar` | 740 条扫描消息 |
| `/alphasense/cam0/image_raw` | 2,960 帧，720×540，mono8 |
| `/alphasense/imu` | 29,539 条 IMU 消息 |
| 五路相机合计 | 14,799 帧；本轮 FAST 使用 cam0 |
| dense 6DoF reference | 689 行 IMU 参考位姿，TUM `timestamp xyz qx qy qz qw` |
| surveyed sparse reference | 4 个位置检查点；四元数是占位值 |
| GT 与共同传感器时间区间 | 73.793305 s |

首个点云的字段类型、ring 和逐点时间跨度另有样本核查；完整记录读取不等于逐个点的物理时间或几何已被验证。公开数据的输入校验也不能替代本机融合运行。

两模式的只读预检分别绑定原生源码／明确登记的 CMake 修复、实际编译产物、XT32 handler 符号、完整 bag 与冻结 GT。预检的 `roslaunch` 只解析参数和节点清单，没有启动节点。相机 YAML 保持固定官方字节，cam0 的 mono8 实际离线转换成 540×720×3 uint8 BGR，通过官方 `cv_bridge` 调用，不需要额外图像 republisher。[LIO 预检](../outputs/fast-livo2-hilti-gt-001/preflight-runs/exp14_basement_2_lio_20261010T100100Z_671b04b2/preflight.json)和[LIVO 预检](../outputs/fast-livo2-hilti-gt-001/preflight-runs/exp14_basement_2_livo_20261010T100139Z_01a3ebc1/preflight.json)各有 13 项真实检查；后续两次实际回放进一步确认有限轨迹、点云和输入图像输出。半速功能回放与输出观测不证明每条传感器消息均被算法处理或实时吞吐达标。

Hilti 页面注明 CC BY-NC-SA 3.0，并要求引用 Zhang 等人的 Hilti-Oxford 论文。当前数据保存在本机；知识网站提供收据和来源链接，不将 6.26 GB bag 作为网页资源。

## dense 与 sparse 不能混成同一种真值

发布方明确：GT 在 IMU 参考系；稀疏位置来自测量参考点，毫米级精度描述这类 surveyed positions，选定额外序列的更密集 6DoF 轨迹用于开发和调试。不能把标题中的毫米精度直接推广到全部 689 个 dense pose。四个 sparse 点也不能给出完整漂移曲线或姿态精度，其占位四元数不参与旋转误差评分。

固定官方 evaluator 有一个需要显式处理的文件名分支：它取参考文件最后一个下划线段，只有 `imu.txt` 使用单位变换；其他名字都会给估计轨迹施加 IMU→杆尖平移 `[0.059, -0.00855, 0.1964] m`。因此 `exp14_basement_2_imu.txt` 走 IMU 分支，`exp14_basement_2_imu_3dof.txt` 却会走杆尖分支，尽管当前网页将该文件描述为 IMU 参考系。不能仅因它叫“官方脚本”就直接采用这个 sparse 评分口径。[固定 evaluator 源码](https://github.com/Hilti-Research/hilti-slam-challenge-2022/blob/e4aacf7c111ab832a0bdffe3f069255454068540/evaluation-2022/evaluation.py)已保存并核对 Git blob，仅阅读，未执行或提交结果。

杆臂的物理含义是 `p_tip(t) = p_IMU(t) + R_world_IMU(t) · r_IMU_tip`。随姿态旋转的杆臂不是一个恒定世界平移，必须先确认两个输入表示的物理点与方向约定。NTU-VIRAL 的棱镜外参属于另一套设备，不能搬到 Hilti。

现有[参考系假设诊断](../outputs/fast-livo2-gt-input-recovery-003/gt-frame-hypotheses.json)仅在原始四个 surveyed 时刻插值 dense 位置与姿态，没有拟合 SE(3)、尺度或时间偏移。原始位置距离约 0.1035–0.1183 m；尝试加、减发布杆臂均未建立一致的物理解释。这些是输入之间的数值诊断，不是 FAST 的轨迹误差，也不能用“哪个数更小”决定真值参考系。首轮 dense 对照采用发布方 IMU 命名约定并清楚标为开发参考；sparse 精度结论仍需独立确认语义。

## 一条轨迹可能混淆了输出阶段

FAST 的 `_state.pos_end` 与 `rot_end` 表示 IMU body 状态。相机、LiDAR 和杆尖位置需要各自外参，不能把日志中的 IMU 位置直接叫作相机中心。固定源码的输出位置和调用顺序决定以下区分：

| 输出 | 保存阶段与时间 | 适合的用途 |
|---|---|---|
| `Log/result/<unique_run>.txt` | `handleLIO()` 的 LiDAR 更新后；`last_lio_update_time`；IMU body TUM 位姿 | LIO／LIVO 主轨迹的传感器时间对照 |
| LIVO 主轨迹的含义 | 本轮 VIO 前，仍包含此前视觉更新的状态历史 | 融合历史对后续估计的整体影响，不直接代表当前帧 VIO 修正 |
| `Log/image/image_poses.txt` | `handleVIO()` 后保存；`vio_time`；LIVO 按 `image_save/interval=20` 实际保存 36 行 | 单独报告采样后的 VIO 状态，不称完整 40 Hz post-VIO 轨迹；LIO 未启用 |
| `Log/mat_pre.txt`／`mat_out.txt` | LIO 与 VIO 更新前后状态日志 | 与分阶段运行日志联合识别真实 VIO 状态变化 |
| ROS odometry header | 上游用 `ros::Time::now()` | 传输观察；不能替代精确传感器时间主轨迹 |

LIVO 主轨迹与纯 LIO 都在 LiDAR 更新后保存，但它们之前经历的状态更新不同。只有输出 RGB 图或后 VIO 位姿，能证明相关输出调用发生；要说明视觉约束产生了什么修正，还要核对真实视觉处理日志与对应的前／后状态。`mat_pre/out` 同时记录两种更新阶段，不能不加区分地把相邻两行都当成一次 VIO。

本轮[回放入口](../scripts/fast_livo2_linux/replay_hilti_gt.py)用唯一目录保存实际主轨迹、参数、前后状态和退出证据，评测绑定同一 bag、源码、编译二进制与 runner。主对照有 LIO 739 行和 LIVO 738 行；LIVO 后 VIO 另有 36 个采样位姿，原时间范围为 1649764530.245914–1649764600.238359 s。后者不同阶段且采样稀疏，没有进入上述 687 时刻的主轨迹对照，也不能用来替换一个表现较差的主结果。

模式元数据也必须与输出语义一致：实际 YAML 的 `image_save.interval` 两模式均为 20，但 LIO 不启用后 VIO 输出，所以运行收据的 `post_vio_image_interval` 为 null，LIVO 才为 20。旧汇总器把两模式的收据字段都要求为 20，导致验收失败；新版本仅修正这一模式条件，原 runner、evaluator、轨迹和协议保持原字节。[最小修复收据](../outputs/fast-livo2-hilti-gt-001/postprocessing-mode-fix-receipt.json)保留当时只读检查；随后真实导出[结果记录](../outputs/fast-livo2-hilti-gt-001/progress-record-candidate.json)，[合并收据](../outputs/fast-livo2-hilti-gt-001/progress-merges/20261010T143804Z_fd605ea1/receipt.json)确认新增第 46 条结果且原 45 条保留。[归档独立复核](../outputs/fast-livo2-hilti-gt-001/completion-record-independent-review.json)与[说明](../outputs/fast-livo2-hilti-gt-001/completion-record-independent-review.md)再次读回已提交字节、全部既有结果及绑定关系。后处理修复不产生新精度成绩。

LIVO 日志实际有 737 行 raw-feature marker、736 行非零 retrieve candidate，以及各 737 行 Jacobian／EKF 计时，其中各 736 行计时大于零；LIO 对应视觉 marker 为零。[真实视觉日志证据](../outputs/fast-livo2-hilti-gt-001/runs/exp14_basement_2_livo_20261010T142554Z_ef89e887/visual-log-evidence.json)证明视觉候选和相关更新阶段到达，不是全部 EKF 步被接受的计数，也不是残差下降、全消息无丢弃或融合增益证明。前后状态日志已保存，但当前没有逐 VIO 约束接受／状态修正的独立分析。

## 时间偏移是约定，不能靠 GT 拟合猜符号

发布标定的 `timeshift_cam_imu` 为 **+1.907120424 ms**，固定 FAST 默认 `img_time_offset=0`。源码使用 `corrected_image_stamp = header + img_time_offset`，而 IMU 使用 `corrected_IMU_stamp = header - imu_time_offset`；同为正号参数，不一定表示相同方向。还需区分曝光开始／中点／结束、硬件时钟与驱动填入时间。

几何核对中，相机←IMU 旋转元素最大差 4.41×10⁻⁷、平移元素最大差 3.78×10⁻⁸ m，LiDAR→IMU 平移与固定配置一致。这说明发布配置和本机配置在数值上匹配，不是重新估计外参，也没有验证实际运行中的同步误差。

首轮参数保持 image／IMU／LiDAR offset 为零，记录上述差异，不根据 GT 搜索“最佳”时间偏移。之后若做同步敏感性对照，要先固定偏移符号、范围、步长和评分时间支持，再比较真实轨迹与运行日志。让 GT 帮忙选最小误差参数会把开发参考引入调参，无法再把同一成绩当作独立验证。

## 刚体对齐、插值与覆盖要一起报告

对米制 IMU 位置采用尺度固定 1 的 SE(3) 刚体对齐。用一个旋转 `R` 和平移 `t` 对整段估计位置求解，再计算：

```text
(R*, t*) = argmin Σ ||R · p_est(i) + t − p_ref(i)||²,  R ∈ SO(3)
e(i) = ||R* · p_est(i) + t* − p_ref(i)||
APE_RMSE = sqrt(Σ e(i)² / N)     # 单位 m，尺度始终为 1
```

对齐消除参考坐标原点与朝向差异，不代表地图或轨迹曾被系统主动纠偏。它不纠正局部漂移，也不验证外参或时间戳正确。可另列起始位姿参考系和误差随时间曲线，帮助观察整体对齐掩盖的累计偏离。

本轮共同查询网格直接使用 **689 个原始 dense GT 时刻**，不移动或插值参考时间；对两条估计各自做位置线性插值和四元数 SLERP，最大估计括号间隔为预登记的 **0.2 s**，不外推。LIO 的一个、LIVO 的两个参考查询在各自输出区间之外；没有估计括号超限，二者交集为 **687 / 689 = 99.709724%**，超过登记的 90% 完整性门槛。共同区间为 1649764528.363497–1649764601.956774 s，长度 73.593277 s；这是参考查询覆盖，不是传感器处理率。

GT 中位间隔约 0.099997 s，最大间隔为 0.606181 s。共同网格上只对原始参考索引连续且间隔 ≤0.2 s 的相邻点计算 RPE，不跨越拒绝查询或参考大缺口。687 个共同点之间有 686 个潜在相邻对，排除十对后保留 **676 对**，两模式使用同一集合。固定发布 evaluator 的最近邻阈值 2 s 没有应用到这套本机评分，也不表示传感器同步精度。[回放前登记协议](../outputs/fast-livo2-hilti-gt-001/evaluation-plan.json)、[预登记封存](../outputs/fast-livo2-hilti-gt-001/evaluation/preregistration.json)与[本机评测入口](../scripts/evaluate_fast_livo2_hilti.py)明确这些规则。

LIO 与 LIVO 的输出时间轴可能不同。比较应同时保留各自原生输出的有效区间，并在原始传感器／GT 时间轴定义的共同网格上重新评分；两条估计与参考都满足插值间隔才计入共同集合。报告原始点数、有效匹配数、被大间隔拒绝数、开始／结束时间、共同区间和覆盖率分母。否则删掉失效段或较少输出的一方，可能只剩更容易的点，平均误差看起来反而更好。

后 VIO 轨迹只有采样点，另列其时间支持与覆盖，不混入高频主轨迹平均值。相对运动误差也要固定时间或路程间隔；同样称“相邻帧”，在不同输出频率下并不是同一个比较尺度。

## 这个 GT 对照能建立哪类证据

本轮已建立完整输入、固定编译与参数、两次真实功能回放、同时间支持的主轨迹 GT 对照链。它确认视觉相关路径实际执行，并给出当前单序列的负向对照；尚未测量逐 VIO 状态修正、偏移敏感性、post-VIO 采样精度或语义确认后的 sparse 分数。这几种问题需要各自对应的实验，不能由主轨迹 ATE 代答。

一个平均 APE 不能同时证明融合有效、标定准确、硬件同步正确和场景泛化。尤其不能因为 LIVO 本次误差更高，就把原因直接归给记录中的 +1.907 ms 时间偏移；没有实施该变量对照。应将同时间支持的轨迹误差连接到实际状态更新，再将标定与同步对照连接到对应配置改变。[本轮 GT 报告](../outputs/fast-livo2-hilti-gt-001/report.md)记录这次完整本地对照及实际归档，该对照不把“融合、标定、同步、漂移”整个研究方向宣称完成。

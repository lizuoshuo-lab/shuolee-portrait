---
id: "gslab-exp-full-v4-sequence-controls-20261009"
slug: "full-sequence-online-v4-controls"
title: "18 完整序列 V4：局部先验、累计轨迹与失败恢复"
order: 18
status: "completed_with_disclosed_single_recovery_and_independent_audit"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
---

# 完整序列 V4：局部先验、累计轨迹与失败恢复

在本次完整 desk、xyz 序列中，同版 V4 的位姿先验权重从 0 改为 0.01 后，**raw 位置 RMSE 和无尺度 SE3 ATE 都变差，相邻平移 RPE 却下降**。相对纯几何 ICP，各方法也存在不同取舍。这说明短片段中的观察不能直接推广到全程，单项局部指标改善也不等于累计轨迹整体更好。

结果来自[已完成汇总](../outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json)：两份既有完整 ICP 轨迹，加上四份 V4 完整轨迹。V4 共发生五次运行尝试，其中原第四次因文件写入异常失败，另有一次预先明确的从首帧重启；原失败没有被删除或改成成功。本报告把算法输出、运行完成状态与独立审核分开记录。

## 这里的“完整”是全部关联帧

desk 使用关联索引 0–548，共 549 帧，传感器时间跨度约 19.807 s；xyz 使用 0–747，共 748 帧，跨度约 26.572 s。它们是固定时间匹配、去重与采样规则得到的完整关联序列，不是原目录里的全部 raw RGB：desk 原有 613 张 RGB、595 张深度，xyz 各有 798 张。

[原完整序列协议](../manifests/tum-online-v4-full-sequence-protocol-001.json)固定 seed 0、160×120、有效深度 0.2–5 m，以及同一版[教学程序](../scripts/train_tum_online_gs_v4.py)，SHA256 为 `ee7e2870…506f`。相对对应的 100 帧实验，仅改变起点、帧数和运行名；没有按完整序列 GT 搜索权重。

每个场景的两组只切换先验权重 0 与 0.01。归一化尺度仍为平移 0.01 m、旋转 1°，tracking 每帧 100 步；首帧 warmup 150 步，每隔四个局部帧进行 80 步 mapping，窗口为首帧加最近三个关键帧。高斯初始上限 15,000，总上限 60,000，每次最多添加 3,000 个。

首相机是 identity；几何 ICP 前缀由传感器独立递推，作为 GS 当前相机初始化和先验参考。审核确认四组 V4 保存的 float64 ICP 前缀及每对相对候选，与对应完整 ICP 基线逐项相等。GS 可以偏离这个前缀，但不是把上一帧 GS 修正直接当作下一帧 ICP 前缀递推。地图继续受历史建图更新影响，局部相机优化与地图历史仍会耦合。

GT 位姿用于完成后的评价，不进入 tracking、mapping 或先验参考。loader 仍会读取 GT 字段与时间以建立关联；这里的边界是 GT 位姿值没有用于优化、初始化或候选选择，不能写成完全不读取 GT。先验的相机中心、旋转尺度和工程解释见[知识 11](../knowledge/11_pose_prior_units_gauge.md)。此前的[短片段先验对照](2026-10-09-pose-prior-controls.md)对应另一段输入、起始地图和优化历史，不能与本页视为相同实验的长度外推。

## 四个完成组，来自五次尝试

[原批次终止记录](../outputs/tum-online-v4-full-sequence-001/failure.json)保留退出码 `[0,0,0,1]`，其状态仍为 `batch_stopped_failure_no_retry`。第四次 xyz、权重 0.01 的原运行，在保存进度时出现 Windows `PermissionError / WinError 5`，涉及 `os.replace(progress.json.tmp, progress.json)`。失败记录与 checkpoint 保留了局部索引 520 为止的 521 帧，而先前正式进度文件只保存到索引 516；没有完整 748 帧 metrics 或最终评测地图。

| V4 尝试 | 保存结果 | 纳入完整轨迹比较 | 子进程观测耗时 / s |
|---|---|---|---:|
| desk，权重 0，001 | 549 帧完成，exit 0 | 是 | 843.929 |
| desk，权重 0.01，001 | 549 帧完成，exit 0 | 是 | 798.941 |
| xyz，权重 0，001 | 748 帧完成，exit 0 | 是 | 1188.099 |
| xyz，权重 0.01，001 | 521 帧部分证据，exit 1 | 否，失败独立保留 | 686.137 |
| xyz，权重 0.01，002 | 748 帧完成，exit 0 | 是，明确替代第四组 | 1101.267 |

这些时长包含各进程加载、计算和序列化的实际成本，失败耗时也保留；没有据此计算或比较 SLAM FPS。

恢复依据新的[单轮恢复协议](../manifests/tum-online-v4-full-sequence-recovery-protocol-001.json)，SHA256 为 `3e1a127d…ea3c`，并经过[执行前复核](../outputs/tum-online-v4-full-sequence-recovery-001/root_protocol_review.json)。它指定唯一的新运行 `tum-xyz-online-v4-prior001-full748-002`，从 frame 0 重新初始化，不从失败 checkpoint 续训。36 个配置字段中只允许运行名变化，六个冻结源码（含训练器）版本不变，也没有从多轮恢复中按最佳指标选择。

[恢复终止记录](../outputs/tum-online-v4-full-sequence-recovery-001/metrics.json)确认唯一新进程 exit 0、完整 748 帧。被比较的是“原三份完成结果 + 指定替代结果”，不是把原批次重新标签为四次全部成功。[小型失败证据](../outputs/tum-online-v4-full-sequence-001/fourth_run_failure_evidence.json)与[1,067 份原文件的快照清单](../outputs/tum-online-v4-full-sequence-recovery-001/original_failure_snapshot/manifest.json)保留失败轨迹、临时文件、checkpoint、来源及日志的对应 hash；大文件仍在本地，不作为网页附件。

### 文件锁是可能机制，不是已经确定的历史原因

[执行观察记录](../outputs/tum-online-v4-full-sequence-recovery-001/root_observation.json)披露：主代理曾用 PowerShell `Get-Content` 读取该进度文件，并中断读取会话。粗略记录时间没有独立证明与写入异常准确重叠，因此不能确定历史锁持有者，也不能认定唯一原因。

随后在独立小文件上运行的[Windows 句柄控制](../outputs/windows-progress-file-sharing-controls-002/checks.json)实际显示：没有外部读句柄时替换成功；保持读句柄时，`ShareRead|ShareWrite` 和再加入 `ShareDelete` 的两种设置都出现 WinError 5；关闭句柄后，同一候选文件均可成功替换。[控制源码](../scripts/check_windows_progress_file_sharing.py)没有触碰历史训练进度。这支持一种可能的句柄冲突机制，不能据此宣称单加 `ShareDelete` 就解决问题。先前控制 001 把带删除共享视为必然成功的断言曾失败，该旧控制及源码也保留，没有用 002 抹去这段记录。

## 六条完整轨迹的指标

平移列统一为 mm，旋转 RPE 为度。raw 是首个 GT 相机定义的共同坐标系内位置误差；ATE 对整条估计位置做一次后验 SE3 刚体对齐，尺度固定为 1；RPE 使用全部相邻对，desk 分母为 548，xyz 为 747。末帧列是未经轨迹拟合的最后一个相机中心误差。

| 序列 | 方法 | raw RMSE / mm | SE3 ATE / mm | 相邻 RPE 平移 / mm | 相邻 RPE 旋转 / ° | 末帧位置 / mm |
|---|---|---:|---:|---:|---:|---:|
| desk | 完整 ICP | 142.881 | 83.043 | 8.913 | 0.868 | 184.811 |
| desk | V4，权重 0 | 118.050 | 72.701 | 9.396 | 0.751 | 172.775 |
| desk | V4，权重 0.01 | 143.551 | 83.347 | 8.798 | 0.795 | 185.384 |
| xyz | 完整 ICP | 44.151 | 30.163 | 3.540 | 0.504 | 71.448 |
| xyz | V4，权重 0 | 26.201 | 16.559 | 5.248 | 0.454 | 28.395 |
| xyz | V4，权重 0.01，恢复 002 | 44.141 | 29.914 | 3.700 | 0.440 | 71.368 |

两条序列的权重 0.01 都比权重 0 更贴近 ICP 前缀，也都失去了后者在 raw、ATE 和末帧位置上的大部分改善；相邻平移 RPE 则有所下降。不能只截取 RPE 列就称先验全程更好，也不能只看 ATE 就忽略更大的逐步平移误差。

相对 ICP，权重 0 在两条序列的 raw、ATE、末帧位置及旋转 RPE 都更低，但平移 RPE 都更高。desk 权重 0.01 的 raw、ATE、末帧位置略差于 ICP，两个 RPE 列更低；xyz 权重 0.01 的 raw、ATE、末帧位置略低于 ICP，平移 RPE 更高。xyz 的 raw 仅差约 **0.00968 mm**，不能把这类单次、同域观察写成显著或通用收益。

全程 raw 旋转 RMSE 也与局部旋转 RPE 不同。按 ICP、权重 0、权重 0.01 顺序，desk 分别为 5.880°、4.583°、5.709°；xyz 分别为 3.347°、1.196°、3.086°。xyz 权重 0.01 虽然相邻旋转 RPE 更低，全程方向误差却明显高于权重 0，局部变化量和相对起点的累计方向偏差不是同一个量。

![desk 完整关联序列：未对齐 XZ 投影与全部逐帧指标](../outputs/tum-online-v4-full-sequence-recovery-summary-001/desk_comparison.png)

![xyz 完整关联序列：未对齐 XZ 投影与全部逐帧指标](../outputs/tum-online-v4-full-sequence-recovery-summary-001/xyz_comparison.png)

图中 XZ 轨迹使用共同首相机坐标，没有为画图重新对齐；所有指标按三维位置或三维旋转计算。每张图保留三个方法的全部帧，各方法在同一面板共用尺度；两个序列的坐标范围独立，不能通过跨图曲线大小判断方法优劣。ATE 面板首帧误差可以不为零，因为对齐拟合整条轨迹，而不再强制首帧固定。

## 局部目标下降为什么不保证全局轨迹更准

tracking 在当前地图上选择 sensor objective 与先验项之和最小的候选。GT 不参与候选选择；选中候选也不必是第 100 个原始 Adam 状态。逐帧记录保留初始及选中候选的 sensor、prior、combined 三项，不能把 combined 的下降误读为纯 sensor loss、GT 位置误差或全局 ATE 必然下降。

几何前缀本身会积累误差，而地图可能通过移动或改变高斯外观来解释有偏差的相机。位姿先验限制偏离几何参考，在有些片段能抑制有害修正，在另一些片段也可能限制有益修正。本次结果给出了这些取舍的实际观察，没有独立隔离出每个误差来源的唯一因果机制。

相对于完整 ICP，排除共同首帧后，逐帧 raw 中心误差的事后比较为：

| 序列与配置 | 比 ICP 更低的帧 | 比 ICP 更高的帧 | 相对 ICP 中心修正均值 / mm | 最大中心修正 / mm |
|---|---:|---:|---:|---:|
| desk，权重 0 | 364 / 548 | 184 / 548 | 66.817 | 104.290 |
| desk，权重 0.01 | 193 / 548 | 355 / 548 | 2.170 | 9.448 |
| xyz，权重 0 | 556 / 747 | 191 / 747 | 37.734 | 83.928 |
| xyz，权重 0.01 | 388 / 747 | 359 / 747 | 1.986 | 6.156 |

“修正量”是 GS 中心离 ICP 中心多远，不是它离 GT 多近；误差更低的帧数也不是显著性检验。所有比较都是完成后的描述，没有用它们挑帧、选权重或挑选恢复运行。[desk 全帧表](../outputs/tum-online-v4-full-sequence-recovery-summary-001/desk_frames.json)和 [xyz 全帧表](../outputs/tum-online-v4-full-sequence-recovery-summary-001/xyz_frames.json)保留实际 frame ID、接受状态、所有失败与覆盖警告、位置误差、选中步号、修正量、点数和损失三项，采用列名加完整行数组的紧凑格式。

六条完整轨迹的 failure flags 均为零，四组 GS 的拒绝帧、ICP 失败和覆盖警告也均为零。所有非首帧 GS tracking 都运行了预设 100 步。**零标记表示没有触发已写入的工程拒绝规则，不证明准确、收敛、对应正确或全局一致**。它更不能撤销原第四次运行的系统级失败。

## 配置与输入相同，实际 warmup 地图仍不同

[desk 首次跟踪上下文](../outputs/tum-online-v4-full-sequence-recovery-summary-001/desk_warmup_context_pair_checks.json)和 [xyz 首次跟踪上下文](../outputs/tum-online-v4-full-sequence-recovery-summary-001/xyz_warmup_context_pair_checks.json)检查了首个位姿梯度探针之前实际保存的状态。每个序列的两组共有七项输入——RGB、深度、有效掩码、K、初始 w2c、先验参考 w2c、float64 ICP 前缀——形状、类型及 C 顺序原始字节完全相等。

但两组 warmup 地图的五类参数均不字节相同：means、log_scales、quats、color_logits、opacity_logits 都存在差异。means 的**最大单坐标元素差**在 desk 为 0.0563095 m，在 xyz 为 0.0448102 m；这不是三维中心欧氏距离的最大值。其他参数的差值属于各自的对数尺度、四元数分量或 logit 单位，不能统称为米。

两组来源帧 ID 与点序相同，首帧点数分别为 desk 15,000、xyz 14,329；这些对应关系支持逐元素核查，但不能让不同张量变成相同地图。相同 seed 与相同配置没有强制 CUDA 输出字节确定。因此本页是整条管线的配对观察，不是严格固定同一 warmup 地图、只改变先验权重的因果消融。

## 地图数量、历史来源与梯度探针的范围

| 序列与配置 | 初始高斯 | 最终高斯 | 累计新增 | mapping 事件，含 warmup | 实际有添点的事件 |
|---|---:|---:|---:|---:|---:|
| desk，权重 0 | 15,000 | 23,838 | 8,838 | 138 | 40 |
| desk，权重 0.01 | 15,000 | 23,852 | 8,852 | 138 | 41 |
| xyz，权重 0 | 14,329 | 18,660 | 4,331 | 187 | 22 |
| xyz，权重 0.01 | 14,329 | 18,459 | 4,130 | 187 | 22 |

四轮都没有达到 60,000 点上限。本实验没有测到预算耗尽的效果，也没有因点更多就证明地图更准。所有来源帧统计和 mapping event 已保留；审核检查了没有从未来帧或被拒帧添点、没有使用未来窗口相机。

desk 最后一个窗口是 `[0,540,544,548]`，xyz 是 `[0,736,740,744]`。xyz 的 745–747 帧仍完成跟踪，但没有因序列结束而追加一次 mapping。当前 mapper 固定窗口相机、优化高斯，不包含回环或历史相机 joint BA。首帧之外的旧视角退出窗口后，不再持续提供直接重建残差；具体机制见[关键帧、窗口与历史一致性](../knowledge/14_keyframes_map_window_history.md)。

四个首次六维梯度探针都通过原先声明的工程检查，但并非零差分误差。按 desk 权重 0、desk 权重 0.01、xyz 权重 0、xyz 权重 0.01 的顺序，余弦相似度约为 0.9459、0.9967、0.9998、0.9139，相对 L2 差约为 0.3320、0.0817、0.0462、0.4162。六个方向包含米与弧度参数，整体向量比较依赖参数尺度；这不是位姿协方差，也不是“渲染器全部数值导数精确”的证明。探针通过、loss 下降和轨迹准确必须分别判断。

## 独立审核证明了哪些关系

[独立结果审核](../outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json)通过了原批次与恢复来源链、六条完整轨迹、原始 TUM 索引与 xyzw GT、分块行、失败标记、来源点与实际张量的检查，重新核对 2,321 份消费文件。汇总器自己的[input hash 清单](../outputs/tum-online-v4-full-sequence-recovery-summary-001/input_hashes.json)含 2,307 项；两个数字属于各自消费范围，不是同一个分母。

汇总的轨迹公式复用冻结 `5ce0…` 实现，见本次[汇总源码快照](../outputs/tum-online-v4-full-sequence-recovery-summary-001/collector_source.py)。[独立审核源码](../scripts/audit_v4_full_recovery_summary.py)通过已审核的另一套原始 GT、SciPy、通用逆与刚体对齐路径复算，没有调用训练或渲染器，也不声称再增加了一套全新的第三方指标实现。

两种路径的最大逐对平移 RPE 差约 `3.666e−9 m`，最大逐对旋转 RPE 差约 `2.988e−13°`。保存的 float32 相机不完全满足机器精度的正交关系，使转置式刚体逆和通用矩阵逆产生很小差异。审核把米、度与混合矩阵元素差分开保留，没有把整张 4×4 矩阵的最大差当成位置误差。这些一致性数字衡量计算路径的关系，不表示真实轨迹达到纳米精度。

执行前还有[恢复汇总器 CPU 控制](../outputs/v4-full-sequence-recovery-collector-cpu-check-001/checks.json)和[完整源码复核](../outputs/v4-full-sequence-recovery-collector-cpu-check-001/root_source_review.json)：原汇总器仍拒绝失败批次，新汇总器只允许原三份完成结果加指定替代轮。最终[完成清单](../outputs/tum-online-v4-full-sequence-recovery-summary-001/completion.json)绑定摘要、两图、全帧表、上下文和源码的 hash；摘要 SHA256 为 `97b65b41…10c81d`。本页链接均为不超过 1 MiB 的文本或图片资源，没有公开原始进度大文件、模型权重或日志。

评价仍依赖最近时间戳 GT，没有插值或运动补偿。desk 的最大 RGB–depth、RGB–GT 时间差分别约 18.939 ms、13.142 ms，xyz 约 17.200 ms、7.564 ms；毫米数值是在这一测量约定下得到的，不能当作完全同步的物理真值。两条 Freiburg1 序列、每组单次 seed 0 与一次透明恢复，只支持本次同域完整序列观察；不计算跨序列平均或单种子的标准差，也不构成完整 SLAM、普遍先验收益或方法整体优越性的证明。

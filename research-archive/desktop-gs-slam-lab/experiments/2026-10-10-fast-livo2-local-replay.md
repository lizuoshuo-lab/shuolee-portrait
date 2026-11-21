---
id: "gslab-exp-fast-livo2-local-replay-20261010"
slug: "fast-livo2-local-livo-lio-replay"
title: "27 FAST-LIVO2：本机回放与融合验真"
order: 27
status: "full_local_livo_lio_functional_replay_accepted_accuracy_pending"
updated: "2026-10-10"
sources:
  - "https://github.com/hku-mars/FAST-LIVO2/tree/0d2c0346107b75b59934975adec9a6eeeb913c64"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/launch/mapping_avia.launch"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/src/LIVMapper.cpp"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/src/vio.cpp"
---

# FAST-LIVO2：本机回放与融合验真

Ubuntu 20.04／WSL2、ROS Noetic 和 FAST-LIVO2 已真正编译并通过环境验收；修复本机预编译库兼容问题后，完整 Bright Screen LIVO 与 LIO 均通过半速功能回放，mapper 与图像转发子节点真实退出 0。首轮两次 −11 收尾失败仍保留，另一次未启动融合的端口预检拒绝单列。**当前没有独立 GT，因此没有本机 ATE 或漂移精度，也未完成新 LiDAR／IMU 外参和物理同步。**

## 实际环境与输入

Linux 虚拟磁盘位于 D 盘，源码与编译位于 Linux 文件系统，原 Windows 源码保留作完整性基准。四任务 catkin 构建产生主程序，Sophus 6／6 测试通过；OpenCV 为 4.2.0、PCL 为 1.10.0，`ldd` 无缺库，Avia launch 解析通过。[编译独立读回](../outputs/fast-livo2-linux-bootstrap-001/actual-build-readback-001.json)记录成功日志、二进制哈希和首次失败；[环境验收](../outputs/fast-livo2-linux-bootstrap-001/linux-executions/20261010T035957Z-309/environment-acceptance.json)本身没有启动融合。

两种模式使用相同的 377,748,402 B 完整 bag、官方 Avia 相机与外参参数、模拟时钟及 `--rate=0.5`。原 bag SHA256 在每轮前后相同。只切换视觉使能，以检验共享状态中视觉分支的实际参与；LIVO 另按每十次视觉更新保存 post-VIO 位姿与 PNG。相机偏置 +0.1 s 沿用官方配置，不是此次估计出的物理时差。

每轮使用独立名称、空闲端口上的私有 ROS master，保存实际运行参数以及 `Log/` 前后快照。原始 bag、全部 PNG、完整 ROS 日志与点云数组保留本地；网站引用小型结果与审计资料。

## 修复后两种模式的完整回放

下表属于修复后的两次完整回放。两次使用同一新二进制和相同 bag，保存轨迹、实际参数、原始／转发图像观察以及真实子节点退出证据。[完整汇总](../outputs/fast-livo2-local-replay-002/summary.json)包含四次完整数据回放：首轮两次失败、修复后两次通过；短段 GDB 诊断另有记录，一次端口绑定拒绝的预检没有启动 ROS 或融合，不加入融合次数。[独立报告](../outputs/fast-livo2-local-replay-independent-audit-002/report.md)与[独立数值](../outputs/fast-livo2-local-replay-independent-audit-002/metrics.json)核对了原始 Header、状态事件、轨迹和子节点 PID／退出。

| 观察项 | LIVO | LIO |
|---|---:|---:|
| 独立 observer 的 IMU／LiDAR 消息 | 13,515／667 | 13,515／667 |
| compressed／raw 图像 | 668／668 | 668／668 |
| 完成 LIO handler／保存 pre-VIO 行 | 664／664 | 664／664 |
| 完成 VIO handler／RGB 发布 | 663／663 | 0／0 |
| 新抽样 post-VIO 位姿 | 66 | 0 |
| C++ mapper／图像转发实际收尾退出 | 0／0 | 0／0 |

![本机修复后的 LIO 与 LIVO 轨迹和内部视觉状态变化，无已知 GT](../outputs/fast-livo2-local-replay-002/trajectory-and-evidence.png)

图中 XY 和 Z 使用保存的传感器时间位姿，均处于 LiDAR 更新后、同轮 VIO 前，没有对齐、缩放或原点归一化。右下是有限精度调试日志重建的 VIO 内部打印位置改变量，单位 mm，不是误差。[绘图数据与来源哈希](../outputs/fast-livo2-local-replay-002/chart-metrics.json)绑定新回放与独立审计，不能凭曲线贴合判断哪个模式更准。

独立审计直接读取 bag 消息 bytes，以序号和整数纳秒 Header 对照 observer CSV；三条传感器流的全部键与顺序相同，668 张 compressed／raw Header 一一对应。这证明转发与 observer 接收链路。LIVO 的 `img_cbk` Header 日志只有 667 行，最后一张 raw 图像没有对应打印；而打印还位于部分接收保护之前，因此 667 也不是最终视觉接受计数。**不能把 observer 收齐称作估计器零丢帧。** IMU 两次 Header 倒退在节点 loop-back 日志中仍存在，不能称同步问题已解决。

## 视觉是否真正改变共享状态

源码已有调试文件混写 LIO 与 VIO 事件。独立审计按 handler 的单线程事件顺序配对 19 列 pre 与 20 列 out：无点返回不写 out，完成事件才配对，时间戳和末列点数交叉检查。修复后 LIVO 的 663 个 VIO pre 均与紧邻之前 LIO out 的前 19 值相同，这支持两个分支更新共同状态。

663 次完成 VIO 中，662 次打印位置发生变化，最大单次变化约 0.04382 m；首帧视觉建图没有改变打印位置。这说明视觉处理实际改变打印出的状态，不能把这个数量当成优化接受步数，也不能推出残差下降或精度提升。新、旧构建的输出有所不同，不能把修复前后的数值直接拼成一次运行。

## 为什么位姿输出要标阶段

TUM 的完整轨迹是 IMU body 在 sensor `last_lio_update_time` 的状态，处于 LiDAR 更新后、同轮 VIO 前。`image_poses.txt` 使用 `vio_time`，是 VIO 后的抽样输出。两次修复后回放各有 664 行有限且时间不倒退的 TUM 位姿，四元数范数通过检查；LIVO 的 66 行新 post-VIO 位姿均唯一匹配本次 VIO out。

66 个同时间戳 PNG 在本轮 before／after inventory 中哈希相同，changed-only archive 没有复制 PNG。已有图片的内容与时间吻合，**尚不能证明本轮新写了 66 张图片**；新轨迹与其本轮 VIO 事件配对的证据独立成立。首轮的 PNG 对应结果见[首轮独立报告](../outputs/fast-livo2-local-replay-independent-audit-001/report.md)与[首轮数值](../outputs/fast-livo2-local-replay-independent-audit-001/metrics.json)。这些有限性与一致性检查没有真值精度含义。

ROS odometry Header 使用模拟 `ros::Time::now()`，不能替换保存轨迹的传感器更新时间。LIO 与 LIVO 输出时刻也不完全相同；比较轨迹差异前须处理共同时间区间与输出阶段。末帧注册点云是局部扫描，LIVO 最后一帧可能为空，不能把它当完整累计地图。

## 退出与准确性的完成边界

首轮 rosbag 与 roslaunch 父进程都返回 0，但 mapper 子节点返回 −11。旧 runner 的 passed 状态只检验父进程，独立审查已纠正；原 receipt 与日志保留，[首轮汇总](../outputs/fast-livo2-local-replay-001/summary.json)不把这两次算作通过。父进程正常退出不能替代子节点退出证据，不能用已有有限轨迹掩盖收尾崩溃。故障定位与后续修复回放各自保留源码变化、构建哈希、真实子节点退出和完整运行证据。[当前运行状态](../manifests/fast-livo2-local-runtime.json)独立标记实际运行次数、通过次数与精度范围。

GDB 与系统 PCL 最小对照把共同收尾故障定位到点云数组分配／释放 ABI：`-march=native` 令 Eigen 使用 32 B 对齐与手工对齐释放，focal 预编译 PCL 使用 16 B 约定。Linux CMake 的 x86 分支改为 `-march=x86-64` 后，对照正常结束；算法和析构未改，Windows 官方树保留。[修复报告](../outputs/fast-livo2-shutdown-fix-001/report.md)、[源码覆盖记录](../outputs/fast-livo2-shutdown-fix-001/source-override-receipt.json)、[补丁](../outputs/fast-livo2-shutdown-fix-001/patch.txt)与[新构建核验](../outputs/fast-livo2-shutdown-fix-001/build-after-verification.json)记录修复边界。新完整 LIVO／LIO 的子节点均已验证返回 0，旧失败输出与新构建通过证据分别保留。

半速播放加上观察器与 PNG 写盘，只验证功能与轨迹生成，不是全速实时性能测评。Bright 没有已知 GT；路径长度、首尾距离、两种模式的轨迹差异，都不等于定位误差。EEE03 新 Linux TLS 尝试在 57,376,768 B 提前结束，长度和官方 MD5 不匹配，三份部分文件保留且目前没有活跃下载。[下载失败记录](../outputs/fast-livo2-linux-native-download-001/receipt.json)不能用来支持 EEE03 完整融合或漂移精度。

知识连接：[共享状态与运行验真](../knowledge/20_fast_livo2_fusion.md) · [外参和标定](../knowledge/21_fast_livo2_calibration.md) · [同步和去畸变](../knowledge/22_fast_livo2_synchronization.md) · [漂移评测口径](../knowledge/23_fast_livo2_drift.md)。

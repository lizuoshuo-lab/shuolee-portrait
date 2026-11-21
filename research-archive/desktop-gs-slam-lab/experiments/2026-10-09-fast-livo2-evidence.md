---
id: "gslab-exp-fast-livo2-evidence-20261009"
slug: "fast-livo2-public-data-calibration-timing-evidence"
title: "26 FAST-LIVO2：公开数据、真实标定与时间审计"
order: 26
status: "actual_camera_timing_and_local_fusion_verified_accuracy_pending"
updated: "2026-10-10"
sources:
  - "https://github.com/hku-mars/FAST-LIVO2/tree/0d2c0346107b75b59934975adec9a6eeeb913c64"
  - "https://ntu-aris.github.io/ntu_viral_dataset/sensor_calibration.html"
  - "https://ntu-aris.github.io/ntu_viral_dataset/evaluation_tutorial.html"
  - "https://researchdata.ntu.edu.sg/api/access/datafile/58998"
  - "https://researchdata.ntu.edu.sg/api/access/datafile/68132"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/launch/mapping_avia.launch"
  - "https://discourse.ros.org/t/ros-noetic-end-of-life-may-31-2025/43160"
---

# FAST-LIVO2：公开数据、真实标定与时间审计

该记录对应 FAST-LIVO2 阶段；当前第一优先已调整为 SplaTAM。本地已完成固定源码核验、完整三传感器 bag 的时间审计、真实双目相机重标定和作者保存轨迹重分析。Ubuntu 20.04／WSL2 已实际启动，ROS Noetic／FAST-LIVO2 真正编译，修复库兼容问题后完整 LIVO／LIO 半速回放均通过功能验收。**LiDAR／IMU 外参新标定、物理同步和本机漂移尚未完成。** 较早的 Windows 待重启与离线准备记录是历史阶段证据，不代表当前安装状态。

| 主线 | 已有真实证据 | 未完成的验收 |
|---|---|---|
| 融合 | 官方 75 文件基准；Ubuntu/ROS 编译、库兼容修复及完整 LIO/LIVO 回放，真实子节点退出 0 | 全速性能和有 GT 的定位评测 |
| 标定 | 135 对棋盘图像；左右内参与双目外参求解，留出与读回 | 相机–IMU、LiDAR–Camera、新时差与跨条件验证 |
| 同步 | 完整 66.7 s bag、逐点时钟与 IMU 括号；独立 raw bytes 检查 | 官方缓存回放、时差控制、硬件时钟证据 |
| 漂移 | 位置评测七项控制、作者 EEE03 轨迹重分析 | 本机估计轨迹与 GT 的完整同段比较 |

## 输入与来源

| 数据 | 完整性 | 作用 |
|---|---|---|
| Bright Screen Wall | 完整 ROS1 索引，377,748,402 B，SHA 已核验 | Livox/IMU/相机时间及回放输入；无 GT |
| NTU `calib_stereo.zip` | 51,118,082 B，全 ZIP CRC 通过 | 135 对真实棋盘相机标定 |
| EEE03 `ground_truth.csv` | 完整，2990 行，固定官方提交 | Leica prism 位置评测 |
| EEE03 sensor ZIP | 三次官方传输均截断；保留 317,325,312 B、44,269,568 B 与 Linux 新尝试 57,376,768 B；官方完整长度 1,995,367,404 B | 首个正式有 GT 的融合目标，目前不可用于完整回放 |
| `calib_stereo_imu.bag` | 部分下载已保留，不完整 | 不能据此声称已运行 Kalibr |
| RTP02 sensor ZIP | 部分下载已保留；仅六份前缀 YAML 完整核验 | 配置对照；不视为完整运动输入 |

数据许可与代码许可分别记录。NTU 文档和当前 Dataverse 许可表述存在差别，待具体发布用途核实；Bright bag 没有独立明确的数据许可记录。网站仅同步小型结果和审计文件，原始 bag、ZIP、模型及相机原图不在导出范围。

EEE03 较早两次传输即使 curl 返回 0，实收长度仍与官方不符，所以没有标为下载完成。小流量请求显示 Dataverse 忽略 Range，返回 HTTP 200 与文件开头而非指定区间；官方 OneDrive 入口返回 403。现有部分文件无法按该响应直接续传或拼接。[下载诊断与回放说明](../outputs/fast-livo2-data-recovery-001/recovery-and-replay.md)保留了这些实测边界。

新的 Linux 原生 TLS 单次传输使用独立文件，证书和主机名校验正常，HTTP 200 的 chunked 流却在 57,376,768 B 结束，约 709.5 s 后因长度和 MD5 不匹配返回 1，未到设定的 1800 s 总时限。这不是 socket timeout；当前没有证据把截断归因于代理，也不能称换 Linux 已解决下载。[原生下载失败记录](../outputs/fast-livo2-linux-native-download-001/receipt.json)保留实际流哈希。三份部分文件均保留，当前没有活跃下载；仍未提取 EEE03 传感器包或测得本机漂移。

## Linux 环境与融合输入的验收边界

目标 Ubuntu 20.04／ROS Noetic 复现原仓 ROS1 依赖，Noetic 已于 2025-05-31 结束维护。构建固定非模板 Sophus 与作者 Vikit fork，使用 Linux 文件系统里的独立源码副本和四个并行编译任务；FAST-LIVO2 本身不需 CUDA。原 Windows 官方树的 75 文件保持校验一致；Linux 构建副本的任何平台补丁单独记录。

较早的 [准备报告](../outputs/fast-livo2-linux-bootstrap-001/report.md)、[离线控制](../outputs/fast-livo2-linux-bootstrap-001/offline-checks-002/preparation-checks.json)、[apt 签名验证](../outputs/fast-livo2-linux-bootstrap-001/ros-apt-verification.json)与[准备阶段脚本哈希](../outputs/fast-livo2-linux-bootstrap-001/final-preparation-state.json)只证明准备检查。后续 Linux 真正完成安装与构建：Sophus 6／6 测试通过，OpenCV 4.2.0、PCL 1.10.0，catkin 生成主程序，`ldd` 无缺库，Avia launch 参数解析通过。[编译独立读回](../outputs/fast-livo2-linux-bootstrap-001/actual-build-readback-001.json)绑定真实日志、成功验收和相同二进制哈希；[环境验收 JSON](../outputs/fast-livo2-linux-bootstrap-001/linux-executions/20261010T035957Z-309/environment-acceptance.json)仍明确自身不启动融合。首次 GnuPG 目录与 PCL pkg-config 名称失败已留档，修复的是 bootstrap 脚本，算法源码未改。

首轮输入采用完整 Bright Screen Wall。668 帧 JPEG 已全部解码且统一 1280×1024，Livox 消息结构与官方头文件匹配。官方 `mapping_avia.launch` 已含 compressed → raw 的 `image_transport` 转发节点，所需 compressed transport 插件已安装，沿用现有节点。[兼容性实测](../outputs/fast-livo2-data-recovery-001/bright-screen-compatibility.json)绑定读取前后相同 bag 哈希。实际 raw 图像发布数与估计器接受数需要分别检查；observer 收齐不能证明 C++ 未拒绝消息。Bright 没有已知真值，运行完成也只能建立融合链路证据，不能给出本机漂移精度。

真实首轮两种模式有输出，但 mapper 收尾返回 −11，原 runner 只检验父进程的 passed 结论已被纠正。Linux CMake 的 x86 编译选项修复后，新 LIVO 与 LIO 的 mapper／republish 子节点都返回 0；每种模式 664 行 pre-VIO 位姿，LIVO 另有 66 行抽样 post-VIO。累计四次真实回放为两次失败、两次通过，另一次未启动融合的端口预检拒绝单列。[本机融合验真](2026-10-10-fast-livo2-local-replay.md)解释运行、输出阶段与限制；[完整汇总](../outputs/fast-livo2-local-replay-002/summary.json)保留旧失败和新成功。

## 真实相机标定

9×6 内角点、80 mm 文档方格；预先分离训练与留出图像。留出图像用 27 点估计板位姿，另 27 点评估。左重估／官方留出 RMSE 为 0.187595／0.188581 px；右为 0.180026／0.171907 px，右侧变差已保留。双目重估基线 0.260191 m，右图独立角点 RMSE 0.535388 px。没有覆盖官方配置，也没有据此称作 LiDAR 或 IMU 标定完成。

![真实标定的逐图留出结果](../outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001/heldout-reprojection-errors.png)

[真实标定结果](../outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001/metrics.json)、[划分与检测 CSV](../outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001/image-splits-and-detections.csv)、[逐图误差 CSV](../outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001/per-image-reprojection-errors.csv)、[独立投影读回](../outputs/fast-livo2-config-audit-001/stereo-intrinsics-calibration-001/saved-calibration-readback.json)均保留。完整角点、模型、残差和位姿 NPZ 保存在本地实验目录，不复制到网站。

## 真实时间异常与范围

667 条 LiDAR、668 条图像、13,515 条 IMU；IMU header 两次倒退，幅度 −0.505448、−0.404835 ms，原 record 时间仍正向。独立 raw bytes 核验全部 14,850 行、89,100 单元格相同。不是硬同步或新时差标定。

Livox timebase 相对 header 中位 −99.909773 ms，源码没有使用 timebase。图像 +0.1 s 是 Avia 配置的固定操作，不是本次优化结果。排序后的 IMU 局部覆盖有边界缺口，scan-end 有一个超过 10 ms 的括号间隔；原顺序异常独立报告，不能用排序掩盖。

![三条流的时间增量](../outputs/fast-livo2-bright-timing-audit-002/timestamp-intervals.png)

[第二版主审计](../outputs/fast-livo2-bright-timing-audit-002/metrics.json)明确 IMU 原始 native units；[第一版标签修正](../outputs/fast-livo2-bright-timing-audit-001/unit-label-correction.json)说明错误的 SI 字段名称，原 metrics/CSV 不改写。[独立报告](../outputs/fast-livo2-bright-timing-independent-audit-001/report.md)与[独立 metrics](../outputs/fast-livo2-bright-timing-independent-audit-001/metrics.json)绑定同一未变 bag。

## 漂移评测工具与作者结果的区别

本地位置工具七个合成控制通过。真实重分析读取作者保存 `eee_03_prism.txt`，采用 GT header clock、尺度 1 的一次 SE(3) 对齐。1805 行中匹配 1507（83.49%），298 行在 GT 范围外；位置 APE RMSE 0.0303545 m。**它不是本机融合运行，也不是完整 SE(3) RPE 或姿态误差。**

![明确标注 author_saved 的重分析](../outputs/fast-livo2-author-eee03-position-reanalysis-001/position-reanalysis.png)

证据：[位置评测控制](../outputs/fast-livo2-position-evaluator-controls-002/metrics.json) · [作者轨迹重分析](../outputs/fast-livo2-author-eee03-position-reanalysis-001/metrics.json) · [完整匹配误差 CSV](../outputs/fast-livo2-author-eee03-position-reanalysis-001/matched-position-errors.csv)。

## 可读源码与知识主线

- [融合与共同状态](../knowledge/20_fast_livo2_fusion.md)，[固定官方源码核验](../outputs/fast-livo2-source-audit-001/report.md)。
- [外参、相机留出与标定边界](../knowledge/21_fast_livo2_calibration.md)，[配置核验工具](../scripts/audit_fast_livo2_config.py)，[相机标定工具](../scripts/calibrate_fast_livo2_stereo_intrinsics.py)。
- [同步与去畸变](../knowledge/22_fast_livo2_synchronization.md)，[执行的第二版审计源码](../outputs/fast-livo2-bright-timing-audit-002/audit_source.py)。
- [漂移评测口径](../knowledge/23_fast_livo2_drift.md)，[位置评测源码](../scripts/evaluate_fast_livo2_position.py)。
- [基础实测记录](../manifests/fast-livo2-work.json)和[本机运行状态](../manifests/fast-livo2-local-runtime.json)，区分半速融合功能通过与本机精度未测。

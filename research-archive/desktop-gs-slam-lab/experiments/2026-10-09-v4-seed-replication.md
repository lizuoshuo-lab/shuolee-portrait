---
id: "gslab-exp-v4-seeds-20261009"
slug: "fixed-v4-two-sequence-three-seed-replication"
title: "三个种子复测：位姿先验的稳定性与几何基线"
order: 15
status: "completed_all_predeclared_runs_independently_audited"
updated: "2026-10-09"
sources:
  - "https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/docs/source/notes/randomness.rst"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
---

# 三个种子复测：位姿先验的稳定性与几何基线

在 desk 和 xyz 两条序列上，各用 seed 0、1、2 比较固定 V4 的 λ=0 与 0.01。全部十二份 100 帧轨迹完成；四份 seed 0 复用已有结果，另八份按事先保存的协议运行，没有挑选最好的种子。三种子均重复了原观察：先验显著限制 GS refinement 的偏移，结果靠近 ICP，但平移 raw RMSE 和相邻平移 RPE 仍高于几何参考。

## 协议固定了什么

每次使用关联帧 200–299、160×120、0.2–5m 深度、ICP prefix 初值、150 步首图 warmup、每帧 100 步 tracking、每个建图事件 80 步 mapping。深度权重 1、alpha 权重 0.01，关键帧间隔与窗口均为 4。先验工程尺度固定为 `σt=0.01m、σr=1°`，权重仅比较已有的 0 与 0.01。没有用新增结果调整权重、尺度、阈值或训练步数。

训练器 SHA256 保持 `ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f`。新运行相对同序列、同权重的 seed 0 配置，仅 `run` 和 `seed` 不同。[完整协议](../manifests/tum-online-v4-seed-replication-protocol-001.json)保存实际命令、数据包及源码 hash、四份旧指标 hash、全部八个新 run 名与失败规则；[串行 driver](../scripts/run_v4_seed_replication.py)保留每项进程退出码、日志和完成指标 hash。新八轮均退出 0，没有重试。

GT 时间戳参与关联，loader 会读取 GT 姿态字段；这些姿态不参与初始化、优化、候选选择或权重选择，只在完成后评测。两条序列都来自 Freiburg1，同一传感器与相近域，不能把这次换序列与换种子称为跨传感器或大场景泛化。

## 三种子均值与样本标准差

以下平移单位为 mm，旋转单位为度；`±` 是三份完整运行之间的样本标准差，分母 `n−1`，不是置信区间。raw 在首 GT 相机的锚定坐标中统计，不拟合对齐；ATE 拟合一次 SE3、保持尺度；RPE 使用相邻帧。

| 序列与流程 | raw 位置 RMSE | SE3 对齐 ATE | 平移 RPE | 旋转 RPE |
| --- | ---: | ---: | ---: | ---: |
| desk，独立 ICP | 64.103 | 30.627 | 5.000 | 0.759° |
| desk，λ=0 | 227.479 ± 2.091 | 87.205 ± 1.527 | 10.706 ± 0.213 | 0.795 ± 0.011° |
| desk，λ=0.01 | 65.702 ± 0.059 | 31.419 ± 0.028 | 5.226 ± 0.006 | 0.654 ± 0.001° |
| xyz，独立 ICP | 41.166 | 20.286 | 3.643 | 0.618° |
| xyz，λ=0 | 51.484 ± 0.434 | 21.035 ± 0.647 | 6.870 ± 0.031 | 0.580 ± 0.003° |
| xyz，λ=0.01 | 41.511 ± 0.030 | 20.179 ± 0.024 | 3.927 ± 0.011 | 0.528 ± 0.001° |

![十二份完整运行的种子分布、样本标准差与独立 ICP](../outputs/tum-online-v4-seed-study-summary-001/seed-comparison.png)

图中圆点按 seed 0、1、2 排列，黑色误差条是均值 ± 样本标准差，灰虚线是同输入的独立 ICP。ICP 没有三份不同随机初始化，表中不为它制造种子间标准差。

相比 ICP，先验组 desk 的平均 raw、ATE、平移 RPE 分别增加约 `1.599、0.791、0.227mm`；xyz 分别变化约 `+0.346、−0.107、+0.284mm`。xyz 的 ATE 小幅下降，与 raw/RPE 变差同时存在，不能挑选其中一个数宣称整体定位更好。

raw 朝向 RMSE 也保留在[统计 JSON](../outputs/tum-online-v4-seed-study-summary-001/metrics.json)：desk λ=0 为 `5.444±0.056°`，λ=0.01 为 `4.499±0.005°`；xyz 分别为 `1.424±0.021°` 和 `2.759±0.008°`。xyz 先验组相邻旋转 RPE 较小，整体锚定朝向 RMSE 却比关闭先验更大。全局姿态误差与帧间相对旋转是不同指标。

## 小波动不等于高准确度

desk 先验组的种子间 raw 标准差约 0.059mm，但其位置 RMSE 仍约 65.7mm。前者描述三次运行彼此多接近，后者描述估计与当前 GT 参考的差距；可重复地产生相近偏差，并不会让偏差消失。时间错位、假定标定、深度噪声、地图误差与轨迹漂移也不会因为 seed 标准差小就得到排除。

这组三种子运行同时包含点排序或子集差异、优化历史与 GPU 数值波动。它没有同 seed 重跑的独立重复，因此不能从这份标准差中分离“初始化变化”和“计算非确定性”。n=3 只支持描述性统计，这里没有显著性、IID 或通用性能结论。详细定义见[随机种子与重复实验](../knowledge/12_reproducibility_seed_variation.md)。

## Seed 改变的实际输入

desk 首图有 12108 个候选点，小于 15000 上限；三个 seed 保留相同物理点、颜色与尺度，只改变排列。xyz 有 15037 个候选点，选择 15000，seed 同时改变子集与排列。[实际初始化核对](../outputs/tum-online-v4-seed-replication-001/initialization_seed_semantics.json)保存联合排序后内容 hash 与原顺序 hash。这不是给高斯中心添加额外位置噪声的实验。

[原始张量字节审计](../outputs/seed-replication-warmup-raw-bytes-audit.json)检查六个同 seed 配对：五组 warmup 地图参数的字节 hash 全部不同，七组传感器、K、初值、参考和 prefix 张量字节全部相同，高斯来源 ID 列表相同。审计还用正负零控制区分“数值相等”和“字节相同”。完整流程继承各自地图历史，因此配对差值描述相同配置预设下的观察，不能把全部差异归因于一个损失项。严格共享实际地图与观测的局部机制证据，仍是[固定上下文回放](2026-10-09-fixed-context-prior.md)。

## 全部结果与失败分母

八个新增进程均完整结束，所有首次六维梯度 probe 通过。连同四份旧结果，共 1200 个存储帧、1188 个非首帧，tracking 失败标记为 0；新增部分单独为 800 帧、792 个非首帧。每个固定首相机不计入 tracking 失败率分母。0 标记不意味着 ICP 各层全已收敛，也不意味着定位准确。

| 新增 seed | desk λ=0 / λ=0.01 | xyz λ=0 / λ=0.01 |
| --- | --- | --- |
| 1 | [关闭](../outputs/tum-desk-online-v4-prior0-seed1-transfer100-001/summary.json) / [开启](../outputs/tum-desk-online-v4-prior001-seed1-transfer100-001/summary.json) | [关闭](../outputs/tum-xyz-online-v4-prior0-seed1-transfer100-001/summary.json) / [开启](../outputs/tum-xyz-online-v4-prior001-seed1-transfer100-001/summary.json) |
| 2 | [关闭](../outputs/tum-desk-online-v4-prior0-seed2-transfer100-001/summary.json) / [开启](../outputs/tum-desk-online-v4-prior001-seed2-transfer100-001/summary.json) | [关闭](../outputs/tum-xyz-online-v4-prior0-seed2-transfer100-001/summary.json) / [开启](../outputs/tum-xyz-online-v4-prior001-seed2-transfer100-001/summary.json) |

四份复用的 seed 0 见[desk 记录](2026-10-09-pose-prior-controls.md)和[xyz 记录](2026-10-09-xyz-transfer.md)。[只读统计源码](../scripts/summarize_v4_seed_replication.py)在全部十二份完成且八项进程账本的退出码、指标与日志 hash 相符后才汇总；[精简摘要源码](../scripts/summarize_tum_online_v4.py)保留每帧传感器损失、先验、combined objective、probe 与失败。每项原始 metrics、相机、模型、日志与迭代记录均留在本地，不以均值替代原成果。

[八轮产物审计](../outputs/seed-replication-independent-audit-summary.json)核对实际上下文、相机结构、数据关联、因果来源、先验/combined 与独立轨迹公式。[公开摘要审计](../outputs/seed-replication-public-summary-audit.json)逐项确认八份摘要各 13484 个标量的值与类型不变。每轮独立 ICP prefix 与相应几何基线的矩阵差为零；其标量指标因统计实现舍入可相差约 `1.33×10⁻¹⁵`，不能把所有浮点标量都描述为逐位相同。

[统计独审](../outputs/tum-online-v4-seed-study-summary-001/independent_statistics_audit.json)使用 Python `math.fsum` 和 `statistics.stdev` 从原指标重新计算均值、样本标准差与配对差值，组内最大差 `3.47×10⁻¹⁸`，配对最大差 `2.22×10⁻¹⁶`，并核对 m→mm、全部退出码、日志与原指标 hash。审计源码见[协议约束的产物审计](../scripts/audit_tum_v4_seed_replication.py)与[独立统计及张量字节审计](../scripts/audit_v4_seed_study.py)。

网站下载副本的 JSON 会格式化并替换本机路径，原文件 SHA 与网页副本 SHA 分别记录。协议中的 SHA 指实验室原字节；源码下载副本保留行尾，当前没有路径清洗的 Python 文件与原源码字节一致。

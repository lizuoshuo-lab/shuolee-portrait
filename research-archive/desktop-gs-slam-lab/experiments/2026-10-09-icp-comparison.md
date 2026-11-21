---
id: "gslab-exp-icp-comparison-20261009"
slug: "rgbd-icp-gaussian-tracking-matched-comparison"
title: "几何基线：相同 RGB-D 片段的 ICP 与高斯跟踪对照"
order: 8
status: "completed_with_scoped_accuracy_comparison"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://www.cs.princeton.edu/~smr/papers/fasticp/fasticp_paper.pdf"
  - "https://www.open3d.org/docs/0.19.0/tutorial/pipelines/icp_registration.html"
---

# 先建立几何基线，再解释高斯跟踪误差

在 TUM `fr1_desk` 的相同 200–299 号关联帧上，教学版 RGB-D projective ICP 的第一相机锚定位置 RMSE 为 6.41 cm；高斯在线跟踪的深度权重 1 版本为 38.88 cm。这里的意义是识别当前高斯教学原型的不足，并得到可供初始化的几何参考。一个场景、一个短片段不能证明 ICP 普遍优于 GS-SLAM。

## 哪些输入被锁定

两类方法使用完全相同的 RGB、深度、真值时间关联与相机模型：有效深度 0.2–5 m，TUM 原始深度除以 5000，最高分辨率 160×120，内参随像素中心缩放。RGB 和深度相差不超过 20 ms，真值位姿只在最终评测使用，真值时间戳参与关联；没有真值初始化或把真值误差用于选取位姿。

首相机设为单位矩阵，估计为 `c2w`，真值整体转换到同一第一相机坐标系。ICP 只用相邻两帧深度；高斯跟踪使用已经学习并持续更新的 RGB-D 地图。输入一致，定位目标与地图用法仍有差异。

20 帧为 30–49 号关联帧，持续 0.6323 秒；100 帧为 200–299 号关联帧，持续 3.3000 秒。后一个片段此前用于 V2 验证，查看结果后已经成为诊断数据，不能继续称为未见最终测试集。

## 核心结果

下表位置量均为 cm，RPE 为相邻一帧，无尺度拟合；ATE 只对整条相机中心轨迹做一次 SE3 对齐。

| 片段 | 方法 | 锚定位置 RMSE | SE3 对齐 ATE | 平移 RPE | 转角 RPE |
| --- | --- | ---: | ---: | ---: | ---: |
| 20 帧 | GS V2，深度权重 0.1 | 21.08 | 9.77 | 2.63 | 1.144° |
| 20 帧 | GS V2，深度权重 1 | 2.85 | 1.43 | 0.67 | 0.604° |
| 20 帧 | 深度 projective ICP | 2.70 | 1.02 | 0.43 | 0.503° |
| 100 帧 | GS V2，深度权重 0.1 | 48.13 | 20.81 | 1.62 | 0.964° |
| 100 帧 | GS V2，深度权重 1 | 38.88 | 8.38 | 1.13 | 0.862° |
| 100 帧 | 深度 projective ICP | 6.41 | 3.06 | 0.50 | 0.759° |

100 帧末帧锚定位置误差依次为 75.35、53.28、9.27 cm。短片段上接近的表现，在较长运动中出现明显分离，因此需要同时查看未对齐误差、整段对齐误差和逐步 RPE。

![同输入轨迹、锚定误差、ATE 与 RPE 对照](../outputs/tracking-methods-transfer100-001/trajectory-comparison.png)

图中左上只是 X-Z 投影；右上和下方误差由完整三维位姿计算。绿色轨迹也存在累计漂移，不能因视觉接近黑线就称准确重建完成。

## ICP 真正做了什么

当前帧反投影点 q 变换到上一帧坐标为 p；投影取得目标点 v 与目标法向 n。点到平面残差为 `r=nᵀ(p−v)`。左乘 SE3 小扰动时，平移、旋转排列的雅可比为 `J=[nᵀ,(p×n)ᵀ]`，加权正规方程为 `JᵀWJ δ=−JᵀWr`。

实现包含 40×30、80×60、160×120 三层金字塔、深度边缘和无效法向拒绝、距离/法向夹角门控、目标像素唯一对应、Huber IRLS、列尺度归一化的秩与条件数检查，以及固定对应下的线搜索。当前→上一帧的相对矩阵 `T_prev_from_current` 通过 `c2w_current=c2w_previous @ T_prev_from_current` 累积。

参数在两次真实运行前固定，没有逐片段调参。[CPU 几何与退化校验](../manifests/track-rgbd-icp-cpu-checks.json)包括：

- SE3 指数映射与 SciPy `expm` 的最大差 6.94e−18；解析 J 与中心有限差分最大差 1.40e−9。
- 固定准确对应的已知刚体恢复，矩阵最大差 1.11e−16；此项隔离求解符号，未覆盖投影对应难点。
- 独立射线生成的合成房间，在真实多尺度配准中产生 0.154 mm、0.00855° 位姿误差。
- 单平面信息矩阵秩为 3，明确拒绝；空深度与失败相对位姿保留上一估计，没有删除失败帧。

真实两片段均无拒绝帧；100 帧各层有 35/172 次迭代预算耗尽。零拒绝只表示未触发门控，并不代表优化已收敛，更不等于达到真值。

## 从 3.3 秒扩展到完整关联序列

另用完全相同的 ICP 参数运行 0–548 号全部 549 个关联帧，共 19.8074 秒。锚定位置 RMSE 为 14.29 cm、SE3 ATE 为 8.30 cm、相邻平移 RPE 为 0.891 cm、转角 RPE 为 0.868°；末帧锚定误差为 18.48 cm，最大为 23.43 cm。没有按真值重启或调参。

全序列保留 548 对相邻配准，零拒绝。1,644 个金字塔层中 868 个用满迭代预算，所以仍不能把零 flags 写成全部收敛。较长范围暴露了更多累计漂移；基线可用来定位当前 GS 问题，却不能替代回环和全局优化。

长序列由[独立 wrapper](../scripts/track_rgbd_icp_full.py)复用原 ICP 代码完成，原 ICP/helper 的 SHA 未改。[小型结果摘要](../outputs/tum-icp-full549-001/summary.json)、[配置](../outputs/tum-icp-full549-001/configuration.json)与[验证](../outputs/tum-icp-full549-001/validation.json)记录了范围和原产物哈希。运行前重放短片段，19 对候选和累积位姿与原结果逐项相同；完整运行中与旧短片段重叠的相对配准也保持一致。这里只扩展同一场景的时长，没有跨场景泛化结论。

![完整关联序列的轨迹与累计误差](../outputs/tracking-methods-full549-001/trajectory-comparison.png)

完整序列从首个关联帧重新设定 gauge，未按旧片段的真值重启；其锚定误差与 100 帧结果具有不同的累积起点。图中的整段 SE3 对齐不会消除所有累计误差。

[完整序列产物审计](../outputs/tum-icp-full549-001/independent_full_sequence_audit.json)核对全部 549 帧、时间戳、刚体结构、原指标和统计，以及与旧片段重叠的逐迭代记录。摘要的完整 metrics 哈希由 validation 同时绑定；评测重算仍使用既有 NumPy 公式，不另称第二独立 ICP 公式。

## 为什么 loss 下降还可能漂移

对 GS V2 的 100 帧日志做离线分析，99 次跟踪都降低传感器目标，但深度权重 0.1/1 的锚定位置误差分别在 53/47 帧变大。24 次关键帧地图优化都降低固定相机下的拟合 loss，平均降幅约 56.3%/59.7%，当时关键帧位置已经有误差。

这说明“loss 下降”不足以确认定位改善，也说明地图有能力在偏移相机下继续拟合；它没有单独证明建图导致下一帧漂移。时序检查也没有发现大于 1.5 倍中位数的 RGB 间隔。给匀速平移预测补充时间比例，只得到毫米量级初值敏感性，不能直接归咎于时间戳。

可复核证据见[诊断统计](../outputs/tum-online-drift-diagnostics-001/metrics.json)、[时序与运动误差](../outputs/tum-online-drift-diagnostics-001/timing-motion-drift.png)和[固定相机地图拟合变化](../outputs/tum-online-drift-diagnostics-001/fixed-camera-mapping-loss.png)。下一项受控实验将 ICP 位姿直接用于建图，并与“同一 ICP 初值之后继续优化高斯位姿”比较，以隔离高斯 refinement 的影响。

## 代码、产物与审计边界

- [ICP 源码](../scripts/track_rgbd_icp.py)，SHA256 `21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a`。
- [GS V2 当时源码](../manifests/source-snapshots/09b9f54a99ddd5e9f02b5e2ea6b328a60a09028c5de79661f01560f32386e284/train_tum_online_gs.py)，不是后续改动的活动脚本。
- [20 帧 ICP 指标](../outputs/tum-icp-short20-001/metrics.json)、[配置](../outputs/tum-icp-short20-001/configuration.json)、[验证](../outputs/tum-icp-short20-001/validation.json)。
- [100 帧 ICP 指标](../outputs/tum-icp-transfer100-001/metrics.json)、[配置](../outputs/tum-icp-transfer100-001/configuration.json)、[验证](../outputs/tum-icp-transfer100-001/validation.json)。
- [统一比较源码](../scripts/compare_tracking_methods.py)与[导出轨迹重算结果](../outputs/tracking-methods-transfer100-001/metrics.json)。时间戳逐项相同，关联列表相同，GT gauge 在文本精度内相同，所有帧保留。

ICP 原始评测与这次文本核对共用 NumPy `audit_pose_arrays`。重算能确认文件和内存统计的一致性，不能称为第二套独立 ICP 评测公式。GS V2 另有此前的 NumPy 独立审计。

这里只比较轨迹精度。CPU ICP 与 GPU GS 的计时包含范围、准备阶段、导出和运行并发情况不同，不能据日志得出速度倍数。所有方法都没有回环、全局 BA 或跨场景评测；定位、地图质量与运行速度需要分别论证。

公式细节见[10 ICP 雅可比与几何可观测性](../knowledge/10_rgbd_icp_observability.md)，轨迹口径见[08 ATE/RPE 与 gauge](../knowledge/08_trajectory_evaluation.md)。

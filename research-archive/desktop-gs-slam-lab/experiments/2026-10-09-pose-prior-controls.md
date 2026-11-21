---
id: "gslab-exp-pose-prior-20261009"
slug: "geometric-pose-prior-online-tracking-controls"
title: "几何位姿先验怎样限制高斯跟踪偏移"
order: 12
status: "completed_paired_observation_artifacts_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html"
---

# 几何位姿先验怎样限制高斯跟踪偏移

在 TUM `fr1_desk` 已用于诊断的 100 帧片段中，增加相机中心和旋转先验后，锚定位置 RMSE 为 6.57cm，关闭先验时为 22.66cm；独立深度 ICP 为 6.41cm。先验流程接近几何参考，锚定位置误差仍略高于 ICP。两次独立 warmup 的地图不同，因此这组差值是同代码、同参数预设下的配对观察，不能全部归因于先验。

## 跟踪目标与固定几何参考

每帧从不接收高斯修正的独立 float64 ICP prefix 初始化。跟踪固定当前地图，最小化：

```text
Ltracking = Lsensor + λ · ( ||C−Cicp||² / σt²
                           + ||R−Ricp||F² / (2·σr²) )
Lsensor   = RGB_MSE + 1 · depth_MAE + 0.01 · alpha_loss
```

C 是 `−Rcwᵀ·tcw` 得到的世界相机中心，R 是 c2w 旋转。参考在当帧 100 个优化步中固定并 detach；λ=0 精确关闭先验。工程尺度为 `σt=0.01m、σr=1°`，计算时角度转换为弧度。旋转采用 chordal 矩阵差，并非精确 SO(3) Log 距离。公式、单位、gauge 不变性及目标决定的修正上界见[位姿先验知识页](../knowledge/11_pose_prior_units_gauge.md)。

地图优化继续只用传感器目标，固定被接受的相机。λ 只参与 tracking 的梯度优化与候选选择；不改 ICP、mapping 目标、硬覆盖阈值或梯度检查门槛。最佳候选按 combined objective 选取，刚体规范化后重新计算同一目标，保留初值作为候选，不使用真值选择相机。

## 数据与配置

两轮使用同一版代码和关联帧 200–299，160×120 图像，0.2–5m 深度。首帧地图 warmup 150 步；每帧 tracking 至多 100 步；关键帧间隔 4，mapping 80 步，窗口包含首帧和最近三个关键帧，高斯上限 60000，seed 0。配置除运行名和 λ 外一致。GT 时间戳参与数据关联，位姿仅用于事后轨迹评测；这个片段已用于多轮诊断，不是未查看的最终测试集。

## 轨迹结果

位置误差单位为 cm；RPE 为相邻帧。ATE 仅做一次 SE3 旋转平移对齐，不估计尺度。

| 定位流程 | 锚定位置 RMSE | SE3 对齐 ATE | 平移 RPE | 转角 RPE | 失败帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 独立深度 ICP | 6.410 | 3.063 | 0.500 | 0.759° | 0 |
| GS refinement 关闭先验 | 22.661 | 8.601 | 1.095 | 0.790° | 0 |
| GS refinement λ=0.01 | 6.567 | 3.141 | 0.523 | 0.654° | 0 |

两轮首次六维有限差分 probe 均通过：关闭先验方向余弦 0.99833、相对 L2 差 0.05771；开启先验分别为 0.98816、0.17438。这只覆盖首次初值邻域，不能证明每一帧梯度连续或定位准确。两轮独立 ICP prefix 与 CPU 基线逐帧矩阵差均为零，0 flags 也不能替代精度指标。

![同版先验开关的轨迹、修正与目标观察](../outputs/tum-online-v4-prior-comparison-001/comparison.png)

## 约束修正与改善真值误差分别统计

99 个非首帧的 ICP prefix→GS 相机中心修正，关闭先验平均为 19.10cm、最大 27.40cm；开启先验平均为 3.09mm、最大 7.01mm。两轮的 99/99 个选中 combined objective 均不高于同帧初值。

但对照同一帧独立 ICP prefix 的真值位置误差，关闭先验仍有 87/99 帧变差，开启先验仍有 82/99 帧变差。先验观察到的主要行为是限制离开几何初值的幅度；它没有让多数候选比几何参考更准确。真值比较只在训练完成后进行，未用于权重筛选或选相机。

开启先验的 99 个非首帧，在 mapping 前选中候选的传感器、加权先验与总目标均值分别约为 0.04686、0.00263、0.04949；关闭先验的传感器目标均值约为 0.06173。这里的先验值已经包含 λ。因为两个流程的地图不同，不能把这些均值当作同一地图下的目标值直接比较。[逐帧汇总](../outputs/tum-online-v4-prior-comparison-001/comparison.json)保留全部观察及张量差异；[CPU 分析源码](../scripts/compare_tum_online_v4_prior.py)另外用 NumPy 和 SciPy 计算轨迹指标，平移指标与原始记录最大差约 4e−17m。

## 相同 seed 不保证相同地图

两轮在首次 tracking probe 前保存真实 warmup 参数、当前 RGB-D、K、ICP 初值和固定参考。CPU 比较发现五组高斯参数都不是逐字节相同；means 最大元素差为 0.0317483m、元素 RMS 差为 0.0021204m。这里比较的是同一高斯索引的参数元素，最大元素差不是所有高斯的三维位移范数。

随后两个流程继续使用各自地图历史，因此观察到的轨迹变化混合了先验作用、首次地图差异和后续建图反馈。保存上下文使同一份固定地图的单步优化对照成为可能；完整流程差值本身不提供严格的因果分离。

## 可复核产物

V4 源码 SHA256 为 `ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f`，prior helper 为 `d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b`。旧版源码和失败记录保留。

| 运行 | 精简摘要 | 实际上下文元数据 | 独立产物审计 |
| --- | --- | --- | --- |
| λ=0 | [summary](../outputs/tum-online-v4-prior0-transfer100-001/summary.json) | [context](../outputs/tum-online-v4-prior0-transfer100-001/first_tracking_context.json) | [audit](../outputs/tum-online-v4-prior0-transfer100-001/independent_v4_artifact_audit.json) |
| λ=0.01 | [summary](../outputs/tum-online-v4-prior001-transfer100-001/summary.json) | [context](../outputs/tum-online-v4-prior001-transfer100-001/first_tracking_context.json) | [audit](../outputs/tum-online-v4-prior001-transfer100-001/independent_v4_artifact_audit.json) |

[CPU 接入审查](../outputs/tum-online-v4-independent-cpu-review.json)核对旧传感器与建图函数保持一致、关闭先验的目标和梯度精确不变。真实产物审计实际加载 context，核对文件 hash、图像、深度与 mask、初值、地图只来自帧 200、当前传感器只来自帧 201，以及首次 probe 前优化步数为零。它还用 NumPy 独立重算全部先验项，最大绝对差约为 4.92e−8，总目标加法差最大约为 7.45e−9。轨迹审计沿用共同的轨迹公式，不声称另有第二套独立定位算法。

源码见[训练器](../scripts/train_tum_online_gs_v4.py)、[先验 helper](../scripts/geometric_pose_prior.py)、[产物审计](../scripts/audit_tum_online_v4_artifacts.py)和[精简摘要导出](../scripts/summarize_tum_online_v4.py)。原始大 JSON、上下文张量和地图模型保留在本地，网站只提供可追溯的小型摘要。[两轮产物审计汇总](../outputs/v4-independent-audit-summary.json)与[公开摘要审计](../outputs/v4-public-summary-audit.json)核对每轮 13484 个标量值及类型、全帧、失败记录和原始 metrics hash。

本实验没有回环、全局 BA、动态物体过滤或重定位。工程先验可以限制偏离参考，也可能把候选约束在错误参考附近；单场景的这组诊断结果不能推广为完整 GS-SLAM 性能或跨场景收益。

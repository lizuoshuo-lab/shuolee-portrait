---
id: "gslab-exp-fixed-context-prior-20261009"
slug: "frozen-map-pose-prior-single-frame-replay"
title: "固定同一地图的位姿先验单帧对照"
order: 13
status: "completed_candidate_replay_and_independently_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
---

# 固定同一地图的位姿先验单帧对照

在完全相同的高斯地图、当前 RGB-D、相机初值与 K 下，先验开关都降低了传感器目标；关闭先验的相机中心修正为 8.805mm，开启时为 3.373mm。事后真值检查发现，这一帧关闭先验的位置误差从 3.215mm 增至 6.427mm，开启先验则为 2.268mm。这是一个冻结地图的局部优化案例，不能替代完整轨迹、更多种子或另一条序列评测。

## 保存实际优化前状态

完整流程的[先验配对观察](2026-10-09-pose-prior-controls.md)使用了各自重新 warmup 的地图，参数并非逐字节一致。本对照只读取其中关闭先验运行保存的首次 tracking 上下文：12108 个高斯，仅由关联帧 200 初始化并 warmup；当前图像与深度来自帧 201；ICP 初值和先验参考固定。上下文在首次 probe 前保存，当时位姿优化步数为零，没有未来图像、未来地图或 GT 相机。

同一个 GPU tensor 对象集合供两组串行复用。五组高斯参数与七组输入张量共 12 组，在两组运行前后都有同一逐张量 hash 和聚合 hash；map、图像、深度、K 与参考都不接受梯度。没有重新初始化、warmup、ICP、mapping 或数据集加载。

## 相同优化路径与两个权重

从锁定的训练器中提取原始渲染、传感器损失、先验组合与刚体规范化函数，未执行训练器 main。两组采用相同的 100 步 Adam、平移学习率 0.003、旋转学习率 0.001、每步衰减 0.99、梯度裁剪 1，以及规范化后按 combined objective 选最佳候选并检查最后一个候选。工程尺度固定 `σt=0.01m、σr=1°`，只切换 λ=0 与 0.01。

先验只影响候选梯度与 combined objective，不改变传感器损失函数。固定的是传感器有效深度 mask；渲染覆盖 mask 仍按每个候选的 alpha、有限深度与正深度计算，两组沿用相同规则，不用 GT 位姿构造 mask 或选候选。这里没有上一帧在线接受门限；保存的是优化候选，不能叫接受后的在线轨迹。两组首次六维 probe 均通过，方向余弦约 0.99833，相对 L2 差约 0.05771。原始步长与门槛保持不变。

## 目标与候选结果

两组初始传感器目标精确相同，都是 0.027787473052740097；初始先验均为零。

| 候选 | sensor | 加权 prior | combined | 离开 ICP 初值的中心距离 | 最佳候选步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 关闭先验 | 0.015109 | 0 | 0.015109 | 8.805mm | 77 |
| λ=0.01 | 0.017125 | 0.002119 | 0.019244 | 3.373mm | 23 |

![相同冻结地图下的传感器目标、修正与事后位置误差](../outputs/tum-pose-prior-context-replay-001/replay-comparison.png)

左图比较同一地图下的传感器目标，曲线是第 0–99 步的原始候选，叉号是 100 次更新后最终选中的最佳候选。叉号画在横坐标 100 仅表示回放结束，不是原始第 100 步候选的数值，也不是最佳候选的步索引；两组最佳步分别为 77 和 23。第 100 步候选另作最后检查，保留在原始指标中。中图记录到每步为止选中的规范化候选距离；选相机使用 combined objective，不按最小位移或 GT 排名。右图仅在整个回放完成后计算真值位置误差，没有拟合对齐。

## 单帧真值误差与流程结论的边界

真值采用此前完整运行保存的首 GT 相机锚定坐标，沿用该帧最近 RGB 时间的 GT；没有新拟合 SE3 或 Sim3。帧 201 的 RGB-depth 时间差约为 12.766ms，RGB-GT 时间差约为 0.481ms，当前没有 GT 插值或时间运动补偿。下表是这套关联与测量约定下的参考误差，不是无时间偏差的精确物理真值。位置单位为 mm，旋转单位为度。

| 相机 | 真值中心误差 | 真值旋转误差 |
| --- | ---: | ---: |
| ICP 初值 | 3.215 | 0.8500° |
| 关闭先验的选中候选 | 6.427 | 0.6030° |
| λ=0.01 的选中候选 | 2.268 | 0.5534° |

这两个最终候选中，传感器损失较低者的位置误差却更高；带先验的候选在该案例同时限制了修正并改善位置。完整 100 帧配对观察中，开启先验仍有 82/99 帧的中心误差比独立 ICP 更大，两种观察不能相互替代。局部成功也不能保证参考错误、遮挡、快速运动或弱纹理时有效。

这次固定了实际输入字节，GPU rasterization 和 reduction 仍可能存在非确定性；没有强制改变 CUDA 的确定性模式，也没有把这一次回放称为统计可靠的收益。相同目标的 λ=0 回放候选，与原完整运行首次选中候选的中心仍相差约 0.897μm，因此这里不声称输出逐位复现。单帧误差不称 ATE 或 RPE。

## 失败记录与复现

第一次脚本 CPU 审查发现提前停止后最后候选步数会错误记成 100，以及 renderer 回调异常可能被误当作数值 probe 拒绝。旧版证据保留；修订版记录实际步数，对不完整 probe 或操作异常立即保存失败并停止。完整六维数值拒绝仍保留失败状态，不通过改变阈值放行候选。

实际回放源码 SHA256 为 `c724e7707d6d7a4ca712c4c8025e9ec3292dd1a14bc00087acfde72356834bda`；上下文 SHA256 为 `f1cc7b76a11cd4f4173ad56398bf1310a7dbe70f83b3ac03535e205e9dc6e4eb`。原始每步相机、梯度、优化器更新、loss JSONL、最后候选和上下文张量保留本地。

[回放指标与输入 hash](../outputs/tum-pose-prior-context-replay-001/metrics.json)、[事后分析及全部 200 步摘要](../outputs/tum-pose-prior-context-replay-001/analysis.json)、[修订后独立 CPU 边界检查](../outputs/tum-pose-prior-replay-independent-cpu-review-002.json)与[初版失败边界记录](../outputs/tum-pose-prior-replay-independent-cpu-review-001.json)可复核具体数值。

[实际产物独审](../outputs/tum-pose-prior-context-replay-001/independent_replay_artifact_audit.json)逐项核对输入冻结、每步指数映射与刚体规范化、Adam 动量及偏差修正、学习率与梯度裁剪、先验和最佳候选记录。NumPy 重算 SE3 最大元素差约 8.41e−8，先验最大绝对差约 6.38e−10；最后候选及两组 100 步记录完整。审计的数值一致性不替代真值精度评测，原始检查门槛也未改变。

源码见[固定上下文回放](../scripts/replay_pose_prior_context.py)与[只读事后分析](../scripts/summarize_pose_prior_replay.py)。GT 只在后者读取已完成回放后消费，没有反馈给候选优化或选择。

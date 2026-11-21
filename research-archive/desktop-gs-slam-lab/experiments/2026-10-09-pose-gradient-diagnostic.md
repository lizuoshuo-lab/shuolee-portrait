---
id: "gslab-exp-pose-gradient-diagnostic-20261009"
slug: "pose-gradient-finite-difference-hard-coverage-diagnostic"
title: "真实场景梯度诊断：硬覆盖掩码、有限差分与历史状态"
order: 11
status: "diagnostic_completed_historical_failure_not_reproduced"
updated: "2026-10-09"
sources:
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
---

# 原失败保留，新诊断只回答当前状态的问题

V3 的 20 帧高斯 refinement 在首个位姿有限差分检查失败。本次用同一源码、配置、数据与 ICP 初值重新计算首帧 warmup 地图，再对首个待跟踪相机做受限诊断：五种步长下均通过原门槛，未重现历史失败。原失败没有改写为通过。

本次只到首个 probe；没有相机优化步骤，没有最终真值评测，也没有生成一条虚构的完成轨迹。重算的 warmup 地图在当前诊断中保持固定，当前参数快照与数据输入已经保存；历史运行当时没有保留该 warmup 快照，不能声称两份地图逐位一致。

## 不连续支持域怎样影响有限差分

深度损失只选取 `valid_depth & (alpha > 0.5)` 的像素，并对选中残差取均值。autograd 对当前选出的残差求导；当相机正负扰动把像素的 alpha 推过阈值时，有限差分计算的是两个支持域不同的函数值。

即使选中像素总数相同，也可能有像素退出、另一些进入。因此诊断记录像素集合的 XOR、进入/退出数，以及覆盖比例；只比较 count 不足以判定掩码未变化。

另一项只读控制把零位姿的覆盖集合固定，正负扰动仍重渲染图像与深度，再计算这一固定集合内的深度残差。这个值只用于解释有限差分，没有进入训练、候选筛选或阈值调整。RGB 与 alpha 项仍保持原定义。

## 同一状态的五档步长

六维 twist 顺序为 `[ρ_x,ρ_y,ρ_z,ω_x,ω_y,ω_z]`，步长分别作用于一个分量；前三个量为 m，后三个量为 rad。所有解析及差分梯度在同一地图、同一零扰动相机附近比较。

| 步长 | 原动态掩码 cosine | 原相对 L2 差 | 固定零位姿掩码 cosine | 固定掩码相对 L2 差 |
| --- | ---: | ---: | ---: | ---: |
| 1e−5 | 0.957263 | 0.392666 | 0.992286 | 0.138981 |
| 3e−5 | 0.982205 | 0.307185 | 0.990907 | 0.163652 |
| 1e−4 | 0.978762 | 0.281876 | 0.992491 | 0.149091 |
| 3e−4 | 0.916560 | 0.439016 | 0.996587 | 0.126495 |
| 1e−3 | 0.938015 | 0.361541 | 0.999167 | 0.086277 |

原门槛为 cosine≥0.8、相对 L2 差≤0.5，解析梯度还须有限且非零。本次五档都满足原门槛；没有挑一档更宽松的门槛来替代历史检查。

![动态与固定覆盖集合的有限差分控制](../outputs/tum-pose-gradient-diagnostic-001/gradient-mask-controls.png)

在 1e−4 步长下，六个分量的正负覆盖集合分别有 `[1,2,1,1,5,2]` 个像素不同；在 1e−3 下增至 `[29,28,15,31,49,18]`。固定覆盖集合在五档均更接近解析梯度，表明覆盖切换影响了本次差分。剩余相对误差仍达 8.6%–16.4%；排序、可见性与 float32 差分等因素尚未分别隔离，不能认定所有误差只来自掩码。

这些覆盖计数与 hash 来自 GPU 运行记录，没有为每个扰动导出完整 mask/alpha 张量。CPU 审核可以重算已保存 loss 的差分和计数关系，不能据此独立重建每个 GPU 掩码。方向余弦和相对 L2 是上述六维参数排列下的整体向量指标，不代表每一维分别满足误差上界。

## 为什么不能把新通过结果覆盖旧失败

历史 refine20 在 1e−4 下 cosine 为 0.710901、相对 L2 差为 0.745115。本次为 0.978762、0.281876；相机初值及独立 ICP prefix 矩阵与历史保存值完全相同，但首个 probe 的目标值相差约 0.00010227。缺少历史 warmup 参数，无法确定当时渲染状态逐位一致，也无法用新结果给原失败指定唯一根因。

固定代码版本、种子与超参数是复核的必要条件；关键优化状态和真实输入也需要保留。保存最终地图不能代替保存首次出错的地图。本次保留当前 warmup 参数、相机初值、传感器数据与 K，后续才可能从同一数值状态进一步隔离排序、覆盖与精度问题。

## 执行边界与可复核文件

[独立诊断源码](../scripts/diagnose_online_pose_gradients.py) SHA256 为 `feac0da4f5d6021a17e85b5feb6670642316a7bd7d9691105bf877eddf3acf86`，导入固定 V3 源码 `45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff`。它在首 probe 收集 5×6×2 个扰动后明确中止主流程；callback 任意异常也会中止，防止错误被原跟踪的 RuntimeError 回退逻辑吞掉后继续训练。

- [全部 60 次扰动的原始标量、掩码变化与梯度](../outputs/tum-pose-gradient-diagnostic-001/pose_gradient_diagnostic.json)。
- [诊断结束标记](../outputs/tum-pose-gradient-diagnostic-001/diagnostic_completion.json)：地图和位姿优化参数未变、tracking steps=0、未执行 GT 评测。
- [CPU 分析](../outputs/tum-pose-gradient-diagnostic-001/analysis.json)：相机和 warmup 状态检查、历史目标比较、当前证据边界。
- [图表摘要](../outputs/tum-pose-gradient-diagnostic-001/summary.json)与[只读绘图源码](../scripts/plot_pose_gradient_diagnostic.py)。
- [独立 CPU 产物审核](../outputs/tum-pose-gradient-diagnostic-001/independent_diagnostic_audit.json)：记录完整性、标量重算、checkpoint 和传感器输入一致；没有重跑 GPU。

传感器输入和地图 checkpoint 原文件只保留在本地，网站不复制原始数据或模型。真值时间戳仍参与原数据关联；“不 GT”具体指优化和诊断没有使用真值位姿，也没有执行真值轨迹评测。这是梯度诊断，不是训练性能或 SLAM 精度结果。

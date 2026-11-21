---
id: "gslab-exp-icp-initialization-20261009"
slug: "icp-initialization-gaussian-refinement-controls"
title: "同一 ICP 初值：高斯位姿优化是否进一步改善定位"
order: 10
status: "completed_and_audited_with_refinement_degradation"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://www.cs.princeton.edu/~smr/papers/fasticp/fasticp_paper.pdf"
---

# 较好的几何初值，也可能被高斯损失推偏

100 帧实验中，直接使用深度 ICP 位姿建图，锚定位置 RMSE 为 6.41 cm；每帧从完全相同的独立 ICP 累积位姿出发，继续优化高斯传感器损失后，误差变为 22.47 cm。这轮梯度检查通过、没有失败标记。当前教学原型的损失下降与定位改善之间仍有明显差距。

## 两条流程的配置切换位姿优化

V3 增加两个模式，采用相同深度初始化、高斯参数预设、随机种子、首帧地图 warmup 步数和关键帧建图配置。每对运行的配置除运行名外，仅 `tracking_mode` 不同：

| 流程 | 每帧相机从哪里来 | 是否优化 GS 位姿 | 地图如何更新 |
| --- | --- | --- | --- |
| ICP 位姿 + GS 建图 | 独立 float64 ICP 累积位姿 | 否，六维 probe 标记 `not_applied` | 固定该相机，优化高斯地图 |
| ICP 初值 + GS refinement | 同一独立 ICP 累积位姿 | 是，固定当前地图优化六维相机 | 固定被选中的相机，优化高斯地图 |

ICP 累积链不接收任何 GS 相机修正；下一帧仍从独立 ICP prefix 初始化。因此不会把上一帧的高斯漂移悄悄混入几何基线。ICP 拒绝则保留上一估计、保留失败标记，并跳过该帧的 GS 位姿优化及建图；这四轮没有 ICP 拒绝。

输入沿用此前相同的 TUM `fr1_desk` 20/100 帧关联列表、160×120 相机、0.2–5 m 深度范围。深度权重 1，alpha 权重 0.01；首帧建图 150 步、位姿优化至多 100 步、关键帧间隔 4、地图更新 80 步、窗口包含首帧和最近三关键帧。数据真值只用于时间关联及事后评测，没有真值初始化、优化或候选筛选。

两条流程各自重新训练首帧地图，随后地图历史随选中的相机发生变化。相同随机种子不保证 CUDA 计算与 warmup 参数逐字节一致；这四轮没有保存首次跟踪前的实际地图，不能事后核验首帧参数完全相同。因此这些是相同配置预设下的流程观察，不能替代同一份固定地图下的单帧定位分析，也不能将两轮差值全部归因于一个开关。

## 可核验指标

下表位置误差单位为 cm，RPE 为相邻帧；ATE 只做一次 SE3 旋转和平移对齐，尺度固定。

| 片段 | 流程 | 锚定位置 RMSE | SE3 对齐 ATE | 平移 RPE | 转角 RPE | 失败帧 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20 帧 | ICP 位姿 + GS 建图 | 2.70 | 1.02 | 0.43 | 0.503° | 0 |
| 20 帧 | ICP 初值 + GS refinement | 2.83 | 1.37 | 0.71 | 0.635° | 1 |
| 100 帧 | ICP 位姿 + GS 建图 | 6.41 | 3.06 | 0.50 | 0.759° | 0 |
| 100 帧 | ICP 初值 + GS refinement | 22.47 | 8.73 | 1.10 | 0.788° | 0 |

![相同 ICP prefix 的直接建图与高斯 refinement 对照](../outputs/tracking-methods-hybrid100-001/trajectory-comparison.png)

独立 ICP 与直接建图的轨迹几乎重合，所以蓝线被橙线覆盖。直接建图接受位姿与独立 float64 prefix 的最大平移差仅为 1.36e−7 m，是 float32 相机转换和刚体规范化精度范围。绿色曲线来自额外的 GS 位姿优化，显示三维误差明显增大。

## 降低 loss，与靠近真值分开统计

在 100 帧 refinement 的 99 次跟踪中，选中的传感器 loss 全部不高于 ICP 初值 loss；但与同一帧 ICP 初值比较，87 次被选相机的真值平移误差变坏，只有 12 次改善。分析真值发生在训练后，未参与选相机。

这比单独比较最后 ATE 更直接：当前地图下的残差下降，常常沿着不改善真实位置的方向移动。它支持继续检查地图几何、目标覆盖、深度损失和可信初始化的约束；不能直接推广为所有可微渲染定位方法的缺陷。

100 帧 refinement 的 ICP 初值→GS 候选平移修正平均为 18.84 cm、最大 27.23 cm。当前 `failure_max_step_m` 检查的是“上一接受位姿→当前候选”，并未限制候选偏离独立 ICP 初值的幅度；当上一个 GS 相机已经偏移，邻帧步长仍可能很小。覆盖比例也从直接建图的平均 0.9692 升至 0.9779，却伴随更大的位置误差。因此邻帧连续、覆盖较高、无 flags 三项都不足以证明候选修正合理。此版尚未加入 ICP 先验或初值修正限幅。

[逐帧比较统计](../outputs/tum-online-v3-study-summary-001/comparison.json)保留了所有帧及配置差异。ICP 相邻帧基线本身仍会漂移，也没有回环、全局 BA 或地图驱动重定位。

## 20 帧梯度失败必须单独保留

20 帧 refinement 的首次六维相机检查在关联帧 31 失败：解析与中心有限差分的方向余弦为 0.7109，相对 L2 差为 0.7451；原门槛分别为至少 0.8、最多 0.5。失败帧使用上一相机并跳过建图，后续帧继续记录。

这个结果不是无故障的纯 refinement 对照，首次回退也会影响后续地图。没有降低阈值、删除该帧或选择性重跑。100 帧 refinement 的首次 probe 通过，方向余弦 0.999960、相对 L2 差 0.04309；这也只验证该初始位置和有限差分邻域的梯度，不能保证以后每帧都平滑或定位准确。

小型[渲染语义控制](../experiments/2026-10-09-renderer-semantics.md)已确认连续区域的六维梯度；真实场景里的排序、覆盖阈值、遮挡和可见性仍需独立诊断。后续[有限差分步长与硬覆盖诊断](../experiments/2026-10-09-pose-gradient-diagnostic.md)发现固定 mask 更接近解析梯度，但重算 warmup 后未复现历史失败；没有将其归因为某个后端错误。

## 源码、摘要与独立产物审计

固定 V3 源码 SHA256 为 `45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff`，见[实验时源码快照](../manifests/source-snapshots/45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff/train_tum_online_gs.py)。

[CPU 完整 prefix 校验](../outputs/tum-online-v3-cpu-validation-001/metrics.json)把真值相机替换为 NaN，并故意提供不同的 GS 相机，20/100 帧 prefix 与独立 ICP 基线逐帧差均为零；[独立方向及失败检查](../outputs/hybrid-initializer-independent-cpu-review.json)还验证错误方向负控制。

| 运行 | 精简数值摘要 | 产物审计 |
| --- | --- | --- |
| map20 | [summary](../outputs/tum-online-v3-icpprefix-map20-001/summary.json) | [audit](../outputs/tum-online-v3-icpprefix-map20-001/independent_v3_artifact_audit.json) |
| refine20 | [summary](../outputs/tum-online-v3-icpprefix-refine20-001/summary.json) | [audit](../outputs/tum-online-v3-icpprefix-refine20-001/independent_v3_artifact_audit.json) |
| map100 | [summary](../outputs/tum-online-v3-icpprefix-map100-001/summary.json) | [audit](../outputs/tum-online-v3-icpprefix-map100-001/independent_v3_artifact_audit.json) |
| refine100 | [summary](../outputs/tum-online-v3-icpprefix-refine100-001/summary.json) | [audit](../outputs/tum-online-v3-icpprefix-refine100-001/independent_v3_artifact_audit.json) |

[四轮独立审计汇总](../outputs/v3-independent-audit-summary.json)检查相机刚体结构、ICP prefix、时间戳、所有失败帧、关键帧窗口、检查点和因果地图来源。审计通过表示产物与记录一致，refine20 的梯度失败不会因此改成通过。

完整 `metrics.json` 包含大量逐层 ICP 迭代记录，地图与原始轨迹也保留在本地。网站只复制[摘要生成源码](../scripts/summarize_tum_online_v3.py)生成的小型 `summary.json`，每个摘要保留完整原指标 SHA256、全部帧及失败标记；没有用摘要覆盖完整指标。[统一导出轨迹重算](../outputs/tracking-methods-hybrid100-001/metrics.json)使用同一 NumPy 评测口径；ICP 统计核对不另称第二独立公式。

[公开摘要的逐项审计](../outputs/v3-public-summary-audit.json)独立检查了四份摘要的 22,887 个标量字段，与原始产物保持数值和类型一致；refine20 的失败 probe 及回退没有从精简内容中删除。

本轮没有多种子、跨场景或论文基准复现，100 帧片段此前已用于诊断。CPU 与 GPU 计时没有统一边界，不做速度倍数比较。

---
id: "gslab-exp-online-20261009"
slug: "tum-online-tracking-depth-weight-controls"
title: "20 帧在线跟踪：漂移、深度权重与迭代次数对照"
order: 6
status: "completed_exploratory_controls"
updated: "2026-10-09"
sources:
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
---

# 在线跟踪跑通后，仍需检查轨迹

本地教学原型已经完成逐帧 RGB-D 跟踪与关键帧建图，后续相机位姿不再由数据真值给定。首跑处理 20 帧且没有触发失败标记，但末帧偏离真值 31.7 cm。扩大跟踪迭代预算没有修复漂移；提高深度损失权重明显改善了这个短片段。

这组实验验证了流程、梯度、因果数据使用与误差分析方法。它只有 0.6323 秒、一个片段、一个种子，不能作为完整 SLAM 基准或稳定厘米级定位的证据。

## 数据与坐标

数据为 TUM RGB-D fr1_desk，CC BY 4.0。采用严格关联后的第 30–49 号帧；RGB 和 depth 为唯一配对，允许最多 20 ms 时间差。相机真值按最邻近 RGB 时间关联，未进行插值、运动补偿或去畸变。

分辨率 160×120；RGB 用 Pillow 缩小，K 使用像素中心变换 `u_new=(u_old+0.5)·sx−0.5`；depth 用最近邻缩小、原始值除以 5000 得到米，只接受 0.2–5 m。

第一相机 `c2w=I` 定义地图 gauge，首帧 RGB-D 提供米制尺度，初始化 14,086 个高斯。其余真实位姿不进入损失、位姿初始化或增图；真值时间戳用于关联，真实 pose 只在运行结束后评测。

## 每帧实际做了什么

1. 用前一估计位姿或前两帧的运动预测初始化当前相机。
2. 固定所有高斯，优化六维 SE(3) 左乘增量，选择传感器损失最小的候选位姿。
3. 每四帧，在低 alpha 覆盖且有有效深度的位置反投影；新点通过当前**估计位姿**进入地图。
4. 用首帧和最近三个关键帧约束地图，固定这些估计相机，仅调整高斯参数。
5. 如果触发非有限数、极低覆盖或极大单步运动检查，沿用上一估计并跳过增图；仍把该帧保留在最终轨迹和误差中。

位姿使用 `T(ξ)=exp(ξ̂)·T_initial`。gsplat 渲染 `RGB+ED`，目标函数为：

```text
L = mean((RGB_render − RGB_sensor)²)
  + λ_depth · mean(|D_expected − D_sensor|, M_covered)
  + 0.01 · mean((1 − alpha)², M_sensor)

M_covered = M_sensor and alpha.detach() > 0.5
            and finite(D_expected) and D_expected > 0
```

`λ_depth` 同时作用于首帧 warmup、后续 tracking 和 mapping。因而当前深度权重对照不能单独证明 tracking 阶段贡献。深度项以米计，RGB 项是无量纲 MSE，这个权重依赖尺度、归一化和分辨率。

默认首帧 warmup 150 步、每帧 tracking 100 步、关键帧 mapping 80 步；地图上限 60,000 个高斯。每次 mapping 新建 Adam，动量被重置；这是一项教学实现简化。相机没有联合 BA，地图可能吸收先前位姿误差。

## 同代码、同片段的单因素对照

四组训练脚本 SHA256 完全相同：`b8f3b69c42f8921b07a6b996f45f0c0d2100a183918e23120d773655339f5281`。关联帧、分辨率、seed=0 和地图上限一致。除了输出 run 名，每个对照只修改一个参数。

| 配置 | SE3 无 scale ATE / cm | raw anchored RMSE / cm | Δ1 RPE / cm | Δ1 旋转 RPE / ° | 平均覆盖 | online loop / s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 CV，depth=0.1，100 步 | 9.7522 | 21.0187 | 2.6327 | 1.1601 | 98.057% | 11.510 |
| 位姿初始化改为 previous | 9.1191 | 20.2442 | 2.5078 | 1.0672 | 98.122% | 11.595 |
| 深度权重改为 1.0 | 1.4353 | 2.9057 | 0.6728 | 0.5916 | 95.198% | 11.798 |
| 跟踪步数改为 300 | 10.2546 | 21.7298 | 2.7089 | 1.1928 | 98.017% | 30.285 |

所有实验的检测 failure flags 均为 0。默认方案的末帧 raw error 为 0.31712 m；depth=1 方案仍有 0.05272 m。对齐后 ATE 1.44 cm 不能写成“始终保持 1.44 cm 定位精度”。计时不含首帧 warmup，包含在线循环中的检查与图片保存，也不能当作纯渲染 FPS。

![同片段控制变量：轨迹、锚定误差、深度残差和地图覆盖](../outputs/tum-online-study-001/tracking-controls.png)

左上显示未拟合对齐的相机中心 X–Z 投影，不能代替三维误差；右上是包含三个空间维度的 raw translation error。即使默认方案平均覆盖达到 98%，轨迹仍快速漂移。

![四组实验末帧：左为真实 RGB，右为按估计相机渲染](../outputs/tum-online-study-001/last-frame-comparison.png)

图片从上到下为 default、previous、depth=1、tracking=300。不同方案地图也受各自估计位姿影响，不能把渲染对比解释为同一地图上仅相机的差异。

## 这次排除了什么，尚未证明什么

增加迭代没有消除漂移，previous 初始化的改善也有限。当前最强观察是更大的几何监督权重改善了该片段；它支持继续检查目标函数设计，并不足以证明权重 1 对所有场景最优。

更高的覆盖率反而出现在漂移较大的方案中。alpha 只说明渲染贡献，错误投影同样可以覆盖像素。depth mask 又依赖当前估计与覆盖，位姿、地图和有效集合一起变化，应结合有效像素数、深度残差与轨迹真值评价。

默认方案 GT 路径长约 0.1703 m，而估计路径长约 0.4391 m。短片段的轨迹形状较弱，SE3 对齐旋转约 72.1°；应同时报告 raw anchored error，不能只挑对齐后的较小数字。

没有多种子、多场景或统计显著性检验。此片段已经用于理解参数，因此属于开发数据；后续独立片段的结果必须另列。CUDA 浮点优化的小差异不应解释为确定性改进。

## 独立评测与可复现文件

独立审计器只使用 NumPy 和导出的 TUM 文本轨迹，自己实现 xyzw 四元数、c2w、无 scale Kabsch 对齐和相邻 RPE；不复用训练器的评测函数。时间戳逐行按 Decimal 精确配对，不重新选择更有利的关联。

全部四组独立指标与内部指标差异小于 2.1e−8 m / 1.9e−7°。同时检查了 checkpoint 的 c2w/w2c 互逆、首相机 identity、文本与 JSON/模型轨迹一致、增图的来源帧与关键帧窗口没有未来信息。数值审计证明导出与统计一致，不能证明传感器标定和物理真值完美。

- [对照汇总指标](../outputs/tum-online-study-001/metrics.json)
- [首跑完整配置、逐帧误差与失败标记](../outputs/tum-online-gsplat-001/metrics.json)
- [depth=1 完整结果](../outputs/tum-online-depth1-001/metrics.json)
- [首跑独立轨迹审计](../outputs/tum-online-gsplat-001/independent_trajectory_audit.json)
- [首跑 checkpoint 与增图因果审计](../outputs/tum-online-gsplat-001/independent_checkpoint_provenance_audit.json)
- [depth=1 独立轨迹审计](../outputs/tum-online-depth1-001/independent_trajectory_audit.json)
- [此次对照对应的在线原型源码快照](../manifests/source-snapshots/b8f3b69c42f8921b07a6b996f45f0c0d2100a183918e23120d773655339f5281/train_tum_online_gs.py)
- [独立评测源码](../scripts/audit_tum_trajectory.py)
- [对照整理与绘图源码](../scripts/summarize_online_study.py)
- [ATE、RPE、gauge 与尺度知识](../knowledge/08_trajectory_evaluation.md)

```powershell
.\.venv\Scripts\python.exe scripts\run_local.py scripts\train_tum_online_gs.py --run tum-online-your-run --depth-weight 1.0
.\.venv\Scripts\python.exe scripts\audit_tum_trajectory.py outputs\tum-online-your-run\trajectory_estimated.txt outputs\tum-online-your-run\trajectory_groundtruth_anchored.txt --output outputs\tum-online-your-run\independent_trajectory_audit.json
```

使用新的 run 名保护既有结果。历次本地源码按 SHA256 保存在 `manifests/source-snapshots/`，恢复到 scripts 目录后再执行，快照所在目录不直接作为运行入口。官方 gsplat 和 SplaTAM 固定 commit 保存在仓库清单；本实验是借鉴其表示与优化思路的教学原型，没有声称原始 SplaTAM 全流程复现。

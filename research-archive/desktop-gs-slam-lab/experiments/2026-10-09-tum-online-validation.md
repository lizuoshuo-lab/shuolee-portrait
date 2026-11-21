---
id: "gslab-exp-online-validation-20261009"
slug: "tum-online-se3-numerics-longer-segment-validation"
title: "从 20 帧到 100 帧：SE3 数值约束与非重叠片段验证"
order: 7
status: "completed_and_audited_with_residual_drift"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
---

# 短片段改善，需要经过更长序列检验

强化 SE3 数值约束后，教学原型完成了三组 100 帧运行。30–129 号片段的 ATE 为 2.48 cm；在此前未参与短片段调参的 200–299 号片段上，深度权重 1 的 ATE 为 8.38 cm、末帧锚定误差仍达 53.3 cm。程序运行和数值结构已通过检查，定位漂移仍需单独解释。

## 一个真实的中途失败

V1 的 `tum-online-long-depth1-001` 在第 81 帧准备增图时退出，抛出 `Double-precision initialization coordinate check failed`。已有配置、日志和前 80 帧图片保留，但当时没有保存完整中间位姿，不能计算一个虚构的“100 帧结果”，也不能确定异常那一刻的唯一根因。

独立 CPU 检查发现：单独旋转非正交并不破坏 helper 的代数闭环，因为 helper 使用一般矩阵逆；而齐次底行 `w=1+1e−7` 足以让 affine 反投影和一般四维求逆产生不一致。可复现的 float32 运动预测压力实验中，一般矩阵逆曾引入底行误差并在后续反馈中放大。这证明一种具体风险存在，仍不等于拿到了失踪的第 81 帧 GPU 矩阵证据。

## V2 加强了哪些结构

SE3 位姿具有 `[R,t;0,1]` 的结构。V2 在优化阶段之间对 R 做不带梯度的 float64 SVD 投影，保证 `det(R)>0`，保留 t，显式设置精确底行；运动预测乘积也做结构检查。

相机逆使用解析刚体逆：

```text
T      = [ R     t ]        T_inverse = [ Rᵀ   −Rᵀt ]
         [ 0     1 ]                    [ 0      1   ]
```

解析逆要求旋转已接近 SO3，不能把它作为一般 affine 矩阵的逆。正规化后的候选相机重新计算传感器损失和运动阈值，再参与接受与导出；真值依旧只在最后评测。

每个关键帧和最后一帧保存原子替换的 `progress.json` 和 `last_checkpoint.pt`。它们只包含已观测传感器信息、估计轨迹和地图，不含实际 GT pose；异常另存失败证据。最终 `metrics.json` 仍表示完整执行后的结果，不能把中途状态当作完整实验。

## 五组预先指定的运行

使用相同 V2 代码、160×120、seed=0、100 tracking 步、80 mapping 步，每四帧建图。脚本 SHA256 为 `09b9f54a99ddd5e9f02b5e2ea6b328a60a09028c5de79661f01560f32386e284`。

30–49 是原开发片段；30–129 与它重叠，用于长度与数值稳定性压力测试。200–299 与原开发片段不重叠，比较之前锁定 depth=0.1 和 depth=1 两个配置，没有用 GT 更换配置或挑选成功运行。它仍来自同一个 fr1_desk 序列，不能等同于跨场景泛化。

| 帧范围 / 深度权重 | 时长 / s | SE3 无 scale ATE / cm | raw RMSE / cm | Δ1 RPE / cm | Δ1 旋转 RPE / ° | 末帧 raw / cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30–49 / 0.1 | 0.6323 | 9.7715 | 21.0795 | 2.6276 | 1.1444 | 31.7273 |
| 30–49 / 1.0 | 0.6323 | 1.4323 | 2.8497 | 0.6695 | 0.6036 | 5.1238 |
| 30–129 / 1.0 | 4.0320 | 2.4835 | 4.3093 | 0.9011 | 0.6428 | 1.9125 |
| 200–299 / 0.1 | 3.3000 | 20.8089 | 48.1349 | 1.6241 | 0.9642 | 75.3517 |
| 200–299 / 1.0 | 3.3000 | 8.3753 | 38.8756 | 1.1321 | 0.8623 | 53.2762 |

五组检测 failure flags 均为 0。V2 的 20 帧误差与 V1 相近：数值结构修复没有消除低深度权重的目标偏差。CUDA 浮点非确定性也存在，不能把微小差值全部归因于这项修复。

## 非重叠片段上的改善与残留漂移

![200–299 号片段两配置的轨迹、锚定误差、深度残差与覆盖](../outputs/tum-online-v2-transfer-study-001/tracking-controls.png)

深度权重 1 改善了此片段的 ATE 和相邻平移 RPE，但 raw anchored RMSE 仍为 0.389 m，末帧仍偏 0.533 m。SE3 对齐拟合一个整体刚体变换，不能把对齐后的较小数字当作在线相机位置的绝对精度。

低权重组平均覆盖 98.39%，高权重组 97.78%。覆盖率仍无法检测这种渐进漂移。高权重组的平均 covered depth MAE 从约 0.115 m 降到 0.031 m；更小的传感器残差也没有使姿态完全正确，末帧旋转锚定误差约 16.1°。

![非重叠验证末帧：各行左真实 RGB，右各自估计相机与地图渲染](../outputs/tum-online-v2-transfer-study-001/last-frame-comparison.png)

此验证证据现在已经被查看并用于诊断。后续若据此继续调参，不能再把它宣传为从未触碰的最终测试集；应使用独立片段或其他序列。当前是一场景、单种子的教学实验，未完成整个 TUM 基准、回环、重定位或全局 BA。

## 实际检查范围

NumPy 独立轨迹审计核对 TUM 文本轨迹与内部 ATE/RPE；另检查了首帧 identity、c2w/w2c 一致、增图来源、关键帧窗口没有未来信息、拒绝帧不会从轨迹剔除，以及 progress/checkpoint 与最终状态一致。

五组 canonical 相机底行误差均为 0；旋转正交 Frobenius 误差最大约 1.05e−7，V1 基线约 2.3e−5。所有 100 帧运行完成，严格几何闭环阈值未放宽。结构与导出一致性验证不代表轨迹精度已经达标。

- [非重叠片段两配置汇总](../outputs/tum-online-v2-transfer-study-001/metrics.json)
- [30–129 号完整结果](../outputs/tum-online-v2-long-depth1-001/metrics.json)
- [200–299 号低权重完整结果](../outputs/tum-online-v2-transfer-baseline-001/metrics.json)
- [200–299 号高权重完整结果](../outputs/tum-online-v2-transfer-depth1-001/metrics.json)
- [高权重组独立轨迹审计](../outputs/tum-online-v2-transfer-depth1-001/independent_trajectory_audit.json)
- [高权重组结构与中途存储审计](../outputs/tum-online-v2-transfer-depth1-001/independent_v2_artifact_audit.json)
- [此次 V2 在线源码快照](../manifests/source-snapshots/09b9f54a99ddd5e9f02b5e2ea6b328a60a09028c5de79661f01560f32386e284/train_tum_online_gs.py)
- [SE3 数值约束知识](../knowledge/06_pose_optimization.md)
- [短片段单因素对照](2026-10-09-tum-online-study.md)

```powershell
.\.venv\Scripts\python.exe scripts\run_local.py scripts\train_tum_online_gs.py --run tum-your-validation --frames 100 --start-frame 200 --depth-weight 1.0
```

恢复此次源码版本到 scripts 目录后执行，使用新的 run 名保护既有结果。完整本地模型、中途 checkpoint 和日志留在实验室目录，网站提供经过审阅的知识、图像、指标及源码。

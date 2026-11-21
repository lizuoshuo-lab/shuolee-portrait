---
id: "gslab-exp-full-seen-map-camera-controls-20261009"
slug: "full-sequence-seen-view-camera-controls"
title: "23 同一份地图：参考相机与训练保存相机的渲染对照"
order: 23
status: "completed_twenty_camera_choice_renders_independently_audited"
updated: "2026-10-09"
sources:
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
---

# 同一份地图：参考相机与训练保存相机的渲染对照

四份完整序列终图各自保持不变，只把渲染相机从锚定 GT 换成训练保存的 accepted 估计相机。20 次新渲染全部完成；原始指标显示，四份地图的 RGB PSNR 均提高，自身覆盖、共同覆盖和含孔洞深度 RMSE 均降低。这个对照回答的是地图与其建图相机是否一致，较低的图像误差不额外说明世界几何或定位更准确。

真实终止产物的独立 CPU 原数组审计已通过，固定摘要也通过了全部原字段与失败记录的保留核对。四份地图各自在两种相机下使用相同参数字节；这个控制不改变已完成的地图或轨迹，不调整数字或另选视角。

## 同一地图内只改变相机

沿用[报告 21 的四份地图及固定视角](2026-10-09-full-seen-map-quality.md)：desk 各使用 0、137、274、411、548；xyz 各使用 0、187、374、561、747。全部视角都是训练观测，诊断不作为留出集或未知视角成绩。

原 GT 基线在报告 21 已完成；本轮只读取那 20 份浮点预测 NPZ，不再次渲染 GT。新的 20 次渲染直接读取完成且无 GT 字段的 `last_checkpoint.pt` 中保存的全序列 accepted `estimated_w2c`，按关联 frame ID 取五个矩阵，使用原始 float32 字节。`estimated_c2w` 与 `estimated_w2c` 的方向和逆变换一致性已在 CPU 预检核对，首相机仍为 identity。

```text
参考：c2w_GT_anchored = inverse(c2w_GT_frame0) @ c2w_GT_view
对照：w2c_saved = checkpoint.estimated_w2c[view_frame_id]
render(同一份地图, 同一个 K, w2c_saved)
```

每份地图在两种相机选择下的高斯参数字节完全相同；RGB、深度、有效 mask、K、160×120 分辨率、黑色背景、`RGB+ED`、裁剪范围及指标函数全部沿用原设置。没有拟合 SE3/Sim3，没有变换地图，也没有进行相机或高斯优化。

不同权重组的 warmup 地图字节仍不相同。这里的配对单位是**同一份地图的 GT/保存相机选择**，不把不同地图合称为相同地图的先验因果对照。原 xyz 权重 0.01 `full748-001` 的 521 帧失败证据保持原样；第四份地图明确使用已审核的从 frame0 重启 `full748-002`，详见[报告 18](2026-10-09-full-v4-sequence-controls.md)。

## 为什么建图相机可能更拟合地图

在线流程在 estimated 相机固定时更新高斯位置、尺度、颜色等参数。这些参数因此可以适应当时的估计相机，即使相机与参考位姿存在偏差。随后用同一估计相机渲染已训练图像，可能获得更低的传感器误差；换成参考相机则会暴露地图与参考世界坐标之间的不一致。

本次对照固定了地图字节，能直接观察相机选择引起的渲染变化，但没有把误差拆解到每次建图更新或具体几何偏差，也没有改善轨迹估计。参考相机和保存相机谁的图像误差更低，与谁更接近真实相机是不同的评测问题。定位仍需报告 18 的 raw anchored error、无尺度 SE3 ATE 及邻帧 RPE。

## 三种深度分母都保留

V 为同一帧有效传感器深度像素。每种相机的自身覆盖是 `C = V & (alpha > 0.5) & finite(ED) & (ED > 0)`；共同覆盖为这份地图两种相机的交集 `C_GT & C_saved`。

| 指标 | 像素范围 |
| --- | --- |
| RGB PSNR | 五帧全部 RGB 通道，data range=1；预测数值不裁剪 |
| 自身覆盖深度 RMSE | 该相机自己的 C，分母可随相机改变 |
| 共同覆盖深度 RMSE | 同一地图 GT/保存相机的交集，两边分母相同 |
| 含孔洞深度 RMSE | C 外预测置 0，全部 V 参与 |

五帧先按真实数量汇总 SSE 与像素数，再计算 pooled 指标；没有平均五个 PSNR，没有跨场景池化。底层深度单位为 m，下表转换为 mm。这是 seed0 的固定地图诊断，不报告种子标准差或置信区间。

## 原始指标：GT 相机 → 保存相机

| 地图 | RGB PSNR / dB | 自身覆盖率 / % | 自身覆盖 RMSE / mm | 含孔洞 RMSE / mm |
| --- | ---: | ---: | ---: | ---: |
| desk / 权重 0 | 9.859 → 12.952 | 95.792 → 97.547 | 280.554 → 177.420 | 384.949 → 302.500 |
| desk / 权重 0.01 | 9.609 → 11.576 | 93.999 → 96.878 | 297.231 → 211.376 | 435.170 → 329.155 |
| xyz / 权重 0 | 16.586 → 17.505 | 96.743 → 96.942 | 191.619 → 123.954 | 411.087 → 383.220 |
| xyz / 权重 0.01（恢复 002） | 13.812 → 15.416 | 96.860 → 97.183 | 200.795 → 168.893 | 432.725 → 405.155 |

| 地图 | 同一地图两相机共同覆盖 / % | GT 共同 RMSE / mm | 保存相机共同 RMSE / mm |
| --- | ---: | ---: | ---: |
| desk / 权重 0 | 93.757 | 276.709 | 172.630 |
| desk / 权重 0.01 | 91.777 | 290.908 | 198.287 |
| xyz / 权重 0 | 94.927 | 178.875 | 115.454 |
| xyz / 权重 0.01（恢复 002） | 95.171 | 197.214 | 152.116 |

共同覆盖表显示，即使使用相同的传感器像素分母，保存相机的深度误差仍更低。它排除了“只因 mask 变了”的完整解释，但共同缺失区域仍未参与这一指标，所以含孔洞误差也同时保留。自身覆盖与共同覆盖的数量、SSE、MAE/RMSE 及所有逐视角统计保留在原始 metrics 中，不用共同覆盖替换自身覆盖。

本页的提升只描述四份固定地图在这些训练视角上的相机选择对照。两套预测执行时间不同，CUDA 渲染也可能存在数值非确定性；没有补做相同 GT 相机重渲染控制，因此不把每个末位浮点差都归为确定的相机效应。

## 固定首、中、末视角的直观对照

每图三列按预先固定的首、中、末视角排列；三行依次为传感器 RGB、GT 相机渲染、保存相机渲染。图像采用标准 RGB [0,1] 显示范围，表中指标使用原始浮点预测。所有五个视角仍参与每份地图的汇总。中间视角的模糊和重影也保留，图像拟合提升不等于地图几何已经正确。

![desk，权重 0：相同地图的两种相机渲染](../outputs/tum-v4-estimated-camera-summary-001/public_diagnostics/tum-desk-online-v4-prior0-full549-001.png)

![desk，权重 0.01：相同地图的两种相机渲染](../outputs/tum-v4-estimated-camera-summary-001/public_diagnostics/tum-desk-online-v4-prior001-full549-001.png)

![xyz，权重 0：相同地图的两种相机渲染](../outputs/tum-v4-estimated-camera-summary-001/public_diagnostics/tum-xyz-online-v4-prior0-full748-001.png)

![xyz，权重 0.01，恢复 002：相同地图的两种相机渲染](../outputs/tum-v4-estimated-camera-summary-001/public_diagnostics/tum-xyz-online-v4-prior001-full748-002.png)

[固定例图清单](../outputs/tum-v4-estimated-camera-summary-001/visual_examples.json)绑定四图、已审计预测和[绘图源码](../scripts/plot_same_map_camera_examples.py)；这里只显示已有预测。

## 执行状态与可复核记录

唯一一次新 GPU 作业完成 20/20 个估计相机渲染，0 个失败、0 个 GT 重渲染、0 个优化步、0 次重试。原地图参数、保存相机、观察输入、既有 GT NPZ、原失败快照及来源哈希在执行后保持不变。20 个新 NPZ 的 `rgb`、`depth_ED_m`、`alpha` 浮点数组全部保留。

CPU 预检的初版误要求数值构造的 GT0 相机与 identity bit-exact 相等，触发了断言失败。实际 GT0 构造残差约为 desk `1.88×10⁻¹⁶`、xyz `4.44×10⁻¹⁶`。初版失败与源码已保存；新版只将该数学锚点比较设为 `1×10⁻¹²`，保存估计首相机自身仍要求 exact identity，所有原 proper-rigid 相机阈值 `2×10⁻⁶` 保持不变，没有运行失败的 GPU 轮次。

可复核的[冻结协议](../manifests/tum-online-v4-estimated-seen-map-evaluation-protocol-002.json)、[评测源码](../scripts/evaluate_tum_v4_estimated_seen_maps.py)、[CPU 预检](../outputs/tum-online-v4-estimated-seen-map-evaluation-002/cpu_preflight.json)、[执行前独立复核](../outputs/tum-online-v4-estimated-seen-map-evaluation-002/root_protocol_review.json)和[实际原指标](../outputs/tum-online-v4-estimated-seen-map-evaluation-002/metrics.json)均已存在。协议锁定原 GT20 的 metrics `f2b0c23f…a7159`、独立审核 `834ab16c…ae65`、摘要 `822767cc…af29`，以及全部 20 份原预测和地图失败来源；本轮 actual metrics SHA 为 `daec22d7…1b14f`。

[本轮独立审计](../outputs/tum-online-v4-estimated-seen-map-evaluation-002/independent_estimated_seen_map_audit.json) SHA 为 `c7c5e51f…ba6c3`，由[冻结审核源码](../scripts/audit_tum_v4_estimated_seen_maps.py)（`3eb53470…1aea4`）执行。它明确复用[既有独立数学 helper](../scripts/audit_v4_final_map_quality.py)（`aabd7c51…8f695`），原全序列轨迹和历史的一致性证据也明确复用原独审；这不是新增的第三套数学公式，也没有进行 GPU 重渲染。

实际审核核对了 20 份新预测、20 份原 GT 预测的 120 个数组，20 个同地图共同覆盖视角，全部 2,594 条训练记录，以及保存相机、参考相机、同地图参数字节与失败来源。970 次数值比较通过，最大差约 `3.64×10⁻¹²`，位于原 GT pooled 含孔洞深度 SSE，单位为 m²。2,262 个消耗文件在审核前后哈希不变，原失败运行的 1,067 份原文件及其快照的守卫通过。它验证产物与计算的一致性，不把图像拟合变成定位正确性的证明。

[固定摘要](../outputs/tum-v4-estimated-camera-summary-001/summary.json)为 154,240 字节，SHA `cd276e3c…2cd7`；[原字段与失败保留核对](../outputs/tum-v4-estimated-camera-summary-001/validation.json)实际通过。摘要使用[纯 JSON 源码](../scripts/summarize_tum_v4_estimated_camera_controls.py)（`a483d030…24d15`）和[执行前源码/独审绑定复核](../outputs/tum-v4-estimated-camera-summary-001/root_source_review.json)，复制原始数值，保留全部 20 视角、4 组同地图相机配对、原位姿与字节守卫、限制和失败字段。135 项输入重新哈希未变，摘要没有导入张量或 GPU 库，也没有重算或挑选指标。

[全部原字段与公开相机表的 root 核对](../outputs/tum-v4-estimated-camera-summary-001/root_public_summary_review.json)另外验证了 20 个逐视角记录、4 组相机配对和全部位姿、失败来源及限制的原样保留。

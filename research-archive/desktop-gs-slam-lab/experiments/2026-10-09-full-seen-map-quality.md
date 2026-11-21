---
id: "gslab-exp-full-seen-map-quality-20261009"
slug: "full-sequence-seen-view-map-quality"
title: "21 完整序列终图：已见视角的颜色、深度与地图预算"
order: 21
status: "completed_20_seen_view_renders_independently_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
---

# 完整序列终图：已见视角的颜色、深度与地图预算

四份完整序列最终地图在固定的五个已观测视角下完成了 20 次渲染。desk 的五帧 pooled RGB PSNR 约为 9.6–9.9 dB，覆盖内深度 RMSE 约为 281–297 mm；图像仍有明显模糊、结构错位和缺失。本次两序列的权重 0.01 组，颜色 PSNR 都更低，自身覆盖、共同覆盖和含孔洞深度 RMSE 都更高。这一观察仅适用于本次固定地图与视角。零失败标记和较高覆盖率，不能单独表示重建质量已经合格。

这是 seed 0、既定工程尺度的一次完整流程观察。两组首帧 warmup 地图的参数字节并不相同，不能把全部差异归为先验权重的严格因果效应。本页沿用[完整序列报告](2026-10-09-full-v4-sequence-controls.md)的四份完成组：原 desk 两组、原 xyz 关闭组，以及明确披露的 xyz 权重 0.01 恢复运行 `full748-002`。失败的原 `full748-001` 仍保留 521 帧部分证据，没有被混入最终地图。

## 用固定参考相机检查终图

desk 训练关联索引 0–548，诊断帧在执行前固定为 **0、137、274、411、548**；xyz 训练 0–747，固定为 **0、187、374、561、747**。五帧全部属于各自地图的训练观测。这是 **seen-training-view 地图诊断**，不作为留出集、未知场景或完整 SLAM 基准成绩。

加载的是已完成且无 GT 位姿字段的因果 `last_checkpoint.pt`，其参数与最终地图产物的一致性已独立审核。评测用 GT 直接提供相机，与建图首相机 identity 的世界约定对应：

```text
c2w_eval = inverse(c2w_GT_frame0) @ c2w_GT_view
w2c_eval = inverse(c2w_eval)
```

没有为截图拟合 SE3/Sim3，没有变换地图、优化相机或更新高斯。这里的颜色与深度误差描述“最终地图在给定参考相机下的渲染”，轨迹精度另见报告 18。

仍使用冻结 V4 的原始 `RGB+ED` 渲染函数、160×120、黑色背景、0.1–6 m 渲染裁剪，以及 0.2–5 m 的有效传感器深度。ED 是条件期望深度。GT 取最近 RGB 时刻的样本，没有插值；RGB/深度时间差没有运动补偿，注册 RGB-D 和 Freiburg1 内参仍是读取假设。

## 每个指标使用哪些像素

设 V 为有效传感器深度像素；预测深度为 d，观测深度为 z。每份地图自身的覆盖集合为 `C = V ∩ {alpha > 0.5} ∩ {finite(d), d > 0}`，阈值严格大于 0.5。

| 指标 | 分母与参考 |
| --- | --- |
| RGB PSNR | 五帧所有像素的三个 RGB 通道；data range=1，数值不裁剪 |
| 覆盖率 | 五帧 `Σcount(C) / Σcount(V)` |
| 自身覆盖深度 RMSE | 各地图自己的 C，参考为传感器深度 z |
| 含孔洞深度 RMSE | C 外预测置 0，再在全部 V 上算误差 |
| 共同覆盖深度 RMSE | 同场景两组的 `C_common = C_off ∩ C_on`，两组使用相同像素 |

先汇总每帧 SSE 与实际像素数，再计算 pooled MSE/RMSE/PSNR；没有对五个 PSNR 直接求平均，也没有跨两个场景池化。深度表格用 mm，底层记录仍为 m。只有一个完成种子，未报告种子标准差或置信区间。

## 四份终图的实际结果

| 场景 / 权重 | 最终高斯数 | RGB PSNR / dB | 自身覆盖率 / % | 自身覆盖 RMSE / mm | 含孔洞 RMSE / mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| desk / 0 | 23,838 | 9.859 | 95.792 | 280.554 | 384.949 |
| desk / 0.01 | 23,852 | 9.609 | 93.999 | 297.231 | 435.170 |
| xyz / 0 | 18,660 | 16.586 | 96.743 | 191.619 | 411.087 |
| xyz / 0.01（恢复 002） | 18,459 | 13.812 | 96.860 | 200.795 | 432.725 |

| 场景 | 共同覆盖率 / % | 权重 0 共同 RMSE / mm | 权重 0.01 共同 RMSE / mm |
| --- | ---: | ---: | ---: |
| desk | 91.357 | 267.748 | 291.510 |
| xyz | 94.727 | 179.583 | 195.768 |

xyz 权重 0.01 组的覆盖率略高，但自身覆盖、共同覆盖与含孔洞 RMSE 都更高。覆盖率只计像素数量；预测落在同一个有效像素上，也可能具有明显深度误差。共同覆盖比较使用相同参考像素，有助于区分“mask 变了”和“相同位置的预测变了”，同时仍排除了双方共同缺失的区域。

与[报告 16 的后续十帧诊断](2026-10-09-final-map-quality.md)相比，本页的地图训练长度、参考相机集合和统计样本都不同。两页指标定义一致，表格数值不构成同条件优劣对照。

## 地图预算没有达到总上限

总上限为 60,000 个高斯。四组的逐帧数量都未达到该值，`first_cap_frame` 均为 null；高斯来源直方图与逐帧增图数量、最终参数长度一致。

| 场景 / 权重 | 首帧初始化数 | 后续新增总数 | 最终数 | 是否触及 60,000 |
| --- | ---: | ---: | ---: | --- |
| desk / 0 | 15,000 | 8,838 | 23,838 | 否 |
| desk / 0.01 | 15,000 | 8,852 | 23,852 | 否 |
| xyz / 0 | 14,329 | 4,331 | 18,660 | 否 |
| xyz / 0.01（恢复 002） | 14,329 | 4,130 | 18,459 | 否 |

来源 frame ID 表示初始化或增图时的观测来源，高斯的位置、颜色、尺度等参数后来仍可能被更新；它不表示那些参数一直保持初始值。总数量上限没有触发，不能直接用“容量到顶停止增图”解释本次误差。初始化采样、可见性增图规则、估计位姿与窗口建图历史的影响仍需要分开的控制实验。

四份地图历史逐帧保留全部 2,594 条数量变化、增图、失败与覆盖警告记录，初始化帧不存在的跟踪字段明确记为缺省，而没有填造成功状态。每份文件均小于 1 MiB：[desk 权重 0](../outputs/tum-online-v4-full-seen-map-evaluation-004/map_histories/tum-desk-online-v4-prior0-full549-001.json)、[desk 权重 0.01](../outputs/tum-online-v4-full-seen-map-evaluation-004/map_histories/tum-desk-online-v4-prior001-full549-001.json)、[xyz 权重 0](../outputs/tum-online-v4-full-seen-map-evaluation-004/map_histories/tum-xyz-online-v4-prior0-full748-001.json)、[xyz 权重 0.01 恢复 002](../outputs/tum-online-v4-full-seen-map-evaluation-004/map_histories/tum-xyz-online-v4-prior001-full748-002.json)。

## 固定首帧、中间帧、末帧的图像

每组均展示执行前确定的五帧中的第一、第三、第五帧；没有按清晰度或 GT 误差选截图。五列依次为观察 RGB、预测 RGB、观察深度、预测 ED、alpha。深度共用 0–5 m 色标，alpha 黑色为 0、白色为 1；RGB 的显示裁剪只影响 PNG，不改变原始预测或指标。观察深度中的零值与缺失仍原样显示，不能把它们理解为零残差。

![desk 权重 0：固定 frame0、274、548](../outputs/tum-online-v4-full-seen-map-evaluation-004/public_diagnostics/tum-desk-online-v4-prior0-full549-001.png)

![desk 权重 0.01：固定 frame0、274、548](../outputs/tum-online-v4-full-seen-map-evaluation-004/public_diagnostics/tum-desk-online-v4-prior001-full549-001.png)

![xyz 权重 0：固定 frame0、374、747](../outputs/tum-online-v4-full-seen-map-evaluation-004/public_diagnostics/tum-xyz-online-v4-prior0-full748-001.png)

![xyz 权重 0.01：固定 frame0、374、747，恢复 002](../outputs/tum-online-v4-full-seen-map-evaluation-004/public_diagnostics/tum-xyz-online-v4-prior001-full748-002.png)

首帧较清晰，中间视角尤其存在模糊、重复结构和局部缺失。alpha 接近白色表示有较高累积覆盖，不保证颜色或几何正确。图像与指标共同暴露了当前地图的误差，但本页没有隔离它们究竟来自累计位姿、估计相机下的地图更新、采样/可见性、标定或时间偏差中的哪一项。

## 完成状态、数值审核与可复核记录

本轮为唯一的 20 次固定视角 GPU 诊断，20 次全部成功、0 次失败、0 个优化步；未追加渲染或重试。程序核对模型参数前后原始字节、输入数组与原 checkpoint 文件字节不变。训练失败记录与恢复快照的哈希同样保留。

[独立 CPU 审计](../outputs/tum-online-v4-full-seen-map-evaluation-004/independent_full_seen_map_audit.json)使用[本轮审核源码](../scripts/audit_v4_full_seen_map_quality.py)（`02045155…71abe`），复用[既有独立数学 helper](../scripts/audit_v4_final_map_quality.py)（`aabd7c51…8f695`），没有调用原指标函数或进行 GPU 重渲染。它不是新增的第三套公式实现，而是在本轮全序列产物上复用已审核的独立数学核对。它独立读取观察输入及 20 份 NPZ 的 60 个预测数组，核对相机、每视角/pooled RGB 与深度指标、10 份共同覆盖比较、2594 条训练记录和参数 hash。466 次数值比较通过，最大差约 `3.64×10⁻¹²`，出现在 pooled 含孔洞深度 SSE，单位为 m²；110 个消耗的源文件/输入/产物在审核前后未改变。这是产物一致性证据，不额外保证 SLAM 轨迹或表面重建准确。

[冻结评测协议](../manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json) SHA 为 `7f07d29e…f6e40`，[评测源码](../scripts/evaluate_tum_v4_full_seen_maps.py) SHA 为 `41ee8343…c41d2`，[执行前复核](../outputs/tum-online-v4-full-seen-map-evaluation-004/root_protocol_review.json)保留了独立的 raw GT 相机方向核对。初版 CPU 预检缺省字段异常与两版通过预检的源码均保留在协议引用中，没有改动冻结训练器。

[完整原指标](../outputs/tum-online-v4-full-seen-map-evaluation-004/metrics.json)、[固定摘要](../outputs/tum-v4-full-seen-map-summary-001/summary.json)及[摘要保留核对](../outputs/tum-v4-full-seen-map-summary-001/validation.json)提供全部请求视角、共同覆盖分母、地图历史与恢复来源。摘要由[固定摘要源码](../scripts/summarize_tum_v4_full_seen_maps.py)（`251c8b24…36d60`）生成，执行前的[源码与输入复核](../outputs/tum-v4-full-seen-map-summary-001/root_source_review.json)已保留。实际摘要为 72,738 字节，逐字段原值与失败记录保留核对通过，66 项输入重新哈希未变，GPU 调用为 0。摘要复制已审核的原始数值，不重新定义指标或选择最佳地图；原浮点 RGB、ED、alpha NPZ 保留在本地 `raw_predictions` 目录。

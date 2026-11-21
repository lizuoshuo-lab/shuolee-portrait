---
id: "gslab-exp-xyz-transfer-20261009"
slug: "fixed-configuration-tum-xyz-sequence-transfer"
title: "固定参数迁移到另一条 TUM 序列"
order: 14
status: "completed_sequence_transfer_and_audited"
updated: "2026-10-09"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
---

# 固定参数迁移到另一条 TUM 序列

沿用 `fr1_desk` 的代码、优化参数预设与工程尺度，在 `fr1_xyz` 第 200–299 个关联帧上各自从片段首帧 RGB-D 重新初始化地图，带先验的锚定位置 RMSE 为 41.501mm，独立 ICP 为 41.166mm。ATE 略低于 ICP，平移 raw RMSE 和 RPE 略高，不能用其中一个指标宣称定位全面改善。相同配置下的修正仍被先验限制在几何参考附近。

## 数据和迁移协议

官方将 xyz 描述为办公桌前、朝向基本固定、沿相机主轴移动的调试序列。它与 desk 来自同一 Freiburg1 传感器和相近域；这是换序列的固定参数验证，不能称为跨传感器、跨场景或大世界模型泛化。

公开包实际下载 448204271 字节，本地 SHA256 为 `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`；它是本地文件标识，这里未使用发布方校验和。798 张 RGB、798 张深度、3000 条 GT 的格式、图像完整解码、索引和刚体姿态均检查通过；冻结 loader 保留 748 组时间关联。这些准备检查只验证数据完整性与当前读取约定，不验证定位或重建精度。来源和许可见[数据清单](../manifests/tum-fr1-xyz-preparation.json)，实际关联见[完整列表](../manifests/tum-fr1-xyz-association.json)。

预先固定关联帧 200–299、160×120、0.2–5m 深度、seed 0。V4 代码与 gsplat 版本不变，首图 warmup 150 步、tracking 100 步、mapping 80 步、关键帧间隔与窗口均为 4，深度权重 1、alpha 权重 0.01。先验仍使用 `σt=0.01m、σr=1°`，两组只切换 λ=0 与 0.01，没有根据 xyz GT 另挑权重。首相机为单位矩阵，此后每帧位姿从独立 ICP prefix 初始化，再由 GS tracking 优化和选择；GT 位姿只做事后评测，GT 时间戳仍参与数据关联。

## 轨迹指标

平移单位为 mm，RPE 为相邻帧。ATE 使用一次 SE3 对齐，尺度固定；raw 采用首 GT 相机锚定坐标，不拟合对齐。

| 流程 | raw 位置 RMSE | SE3 对齐 ATE | 平移 RPE | 旋转 RPE | 失败帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 独立 ICP | 41.166 | 20.286 | 3.643 | 0.618° | 0 |
| GS refinement λ=0 | 51.362 | 21.774 | 6.864 | 0.578° | 0 |
| GS refinement λ=0.01 | 41.501 | 20.151 | 3.922 | 0.527° | 0 |

![固定参数在另一条 TUM 序列的观察](../outputs/tum-xyz-online-v4-comparison-001/comparison.png)

相比关闭先验，带先验时相邻旋转 RPE 较小，但首相机锚定后的 raw 朝向 RMSE 更大：2.759° 对 1.447°；独立 ICP 的同项为 3.070°。这项 RMSE 汇总各帧对 GT 的朝向误差，相邻旋转 RPE 则比较帧间相对旋转，两者不能相互替代，不能仅看 RPE 推断整体姿态更好。

独立 ICP 的 297 个金字塔层中，124 层用完迭代预算，173 层以小更新停止，99 个最终局部系统的秩均为 6；0 拒绝不能表述为所有层全部收敛。不同实现的运行时间边界不同，不作 FPS 结论。

## 修正幅度与真值变化

99 个非首帧中，关闭先验离开 ICP prefix 的中心距离平均为 40.81mm、最大 66.60mm；开启时为 2.00mm、最大 4.23mm。两组选中 combined objective 均为 99/99 不高于初值。

关闭时仍有 83/99 帧的真实位置误差高于同帧 ICP prefix，开启时为 66/99。较小修正没有使多数相机超越几何参考；这些真值比较均在训练完成后进行。

两组首次 probe 均通过并保留原门槛。首地图均有 15000 个高斯，最终分别为 17218 与 17287。首次跟踪前的传感器、K、初值和参考逐字节相同，但五组 warmup 地图参数都不同；means 最大单元素差为 84.77mm，同索引三维中心的最大距离为 99.44mm，不能混淆元素差与向量范数。因此整体流程仍属于同代码、同配置预设的配对观察；严格共享地图的机制证据见[单帧回放](2026-10-09-fixed-context-prior.md)。

## 时间戳和可复核证据

TUM 文本采用原 epoch float64 时间的 `.9f` 输出。xyz 的文本时间与原始文件名 Decimal 最大差约 119ns，位于 epoch float64 半 ULP 加末位舍入界内；估计、GT 与 ICP 文本间的时间序列严格相同。这是序列化精度边界，不是帧错配，也没有通过修改时间戳掩盖关联错误。

| 运行 | 指标摘要 | 产物独审 |
| --- | --- | --- |
| ICP | [summary](../outputs/tum-xyz-icp-transfer100-001/summary.json) | [validation](../outputs/tum-xyz-icp-transfer100-001/validation.json) |
| λ=0 | [summary](../outputs/tum-xyz-online-v4-prior0-transfer100-001/summary.json) | [audit](../outputs/tum-xyz-online-v4-prior0-transfer100-001/independent_xyz_v4_artifact_audit.json) |
| λ=0.01 | [summary](../outputs/tum-xyz-online-v4-prior001-transfer100-001/summary.json) | [audit](../outputs/tum-xyz-online-v4-prior001-transfer100-001/independent_xyz_v4_artifact_audit.json) |

[全部逐帧比较](../outputs/tum-xyz-online-v4-comparison-001/comparison.json)、[两轮独审汇总](../outputs/xyz-v4-independent-audit-summary.json)与[摘要审计](../outputs/xyz-v4-public-summary-audit.json)保留所有帧、失败、原始指标 hash、地图差异和固定源码。两组 float64 ICP prefix 与本序列的独立基线逐矩阵差为零；审计实际加载首次上下文，核对只有帧 200 的地图、当前帧 201 观测以及优化前保存；先验由 NumPy 独立重算。每份公开摘要的 13484 个标量值及类型与原始 metrics 一致。

源码见[固定 V4 训练器](../scripts/train_tum_online_gs_v4.py)、[迁移产物审计](../scripts/audit_tum_xyz_v4_artifacts.py)与[CPU 比较](../scripts/compare_tum_online_v4_transfer.py)。训练器 SHA256 保持 `ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f`。它仍是没有回环、全局 BA、动态物体过滤和重定位的教学原型，不是完整论文基准复现。

---
id: "gslab-exp-slam-drift-mechanism-20261011"
slug: "slam-drift-mechanism-support-vio"
title: "32 漂移机制：损失支持与视觉状态修正"
order: 32
status: "mechanism_diagnostics_independently_verified_historical_cause_unresolved"
updated: "2026-10-11"
sources:
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/keyframe_selection.py"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/src/LIVMapper.cpp"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/src/vio.cpp"
---

# 漂移机制：损失支持与视觉状态修正

解释漂移需要连接三个层次：传感器给出了什么观测，估计器实际用哪些观测更新状态，更新后的轨迹与参考运动是否一致。**输入深度很多、损失较低、视觉确实修正了状态，都不能单独证明位姿更准。** 这里用 SplaTAM room 的两处真实误差跃升和 FAST-LIVO2 的真实视觉更新，说明这些证据各自回答什么问题。

实际固定地图测量确认支持像素随位姿变化，但 36 个扰动都没有降低官方加权总损失；FAST 的 737 次 VIO 中，736 次确实留下净状态修正。两组数值均已独立复算，历史漂移原因仍未确定。

[完整重复与 room 结果](30_splatam_tracking_ablation_repeat.md)已经记录 room 的 0.770351 m ATE；[Hilti 两模式 GT 对照](31_fast_livo2_hilti_gt.md)已经记录 LIO 3.570770 cm、LIVO 4.132461 cm 的共同参考 ATE。新增 CPU 诊断读取这些冻结结果，补充逐帧输入、优化支持的源码条件、视觉前后状态和时间对应，不重新拟合 GT 或改变原来的评分。

## 三层证据怎样连接

| 证据层次 | 具体观测 | 能回答的问题 |
|---|---|---|
| 原始观测 | RGB、有效深度、时间关联差、梯度、参考相邻运动 | 输入是否存在明显异常，异常发生在哪些时刻 |
| 估计器内部 | 渲染覆盖、损失支持像素、分项残差、实际状态修正 | 哪些观测真的参与目标函数，状态如何被改变 |
| 物理结果 | 同一时间、同一物理点的 ATE、RPE与固定首位姿参考系误差 | 估计轨迹是否更接近参考运动 |

这三个层次之间需要实测对应。传感器深度有效，仍可能没有落在地图的渲染覆盖内；内部目标下降，仍可能只剩下更少的容易像素；状态发生变化，仍可能朝错误方向改变。不同指标之间出现相反方向，首先要核对它们的观测支持和输出阶段。

## room 两处跃升的输入条件不同

原序列完整保留 1,315 帧，320×240，观测时长 45.371686 s。两处窗口按已有误差曲线事后选择，精确峰值是帧 838 的 28.235792 s 和帧 1133 的 39.339728 s；图上约 40 s 的波动不能替代第二处的精确起点。[完整输入指标](../outputs/splatam-room-input-diagnostic-001/20261010T155109Z_3ba18586/metrics.json)和[逐帧输入 CSV](../outputs/splatam-room-input-diagnostic-001/20261010T155109Z_3ba18586/per-frame-inputs.csv)覆盖全序列。

| 同一预处理后的输入与运动量 | 帧 838 | 帧 1133 |
|---|---:|---:|
| RGB 相邻时间差，ms | 32.003880 | 35.884857 |
| 有效输入深度占比，% | 83.036458 | 74.059896 |
| 有效深度中位数，m | 1.156000 | 1.238400 |
| depth 时间减 RGB 时间，ms | −0.787973 | +6.224155 |
| 归一化灰度 Sobel 梯度均值 | 0.169414 | 0.293535 |
| 归一化灰度 Laplacian 方差 | 0.040461 | 0.087096 |
| 相邻参考位姿平移，mm | 9.097802 | 14.389232 |
| 相邻平移 RPE，mm | 321.926285 | 315.577478 |
| 相邻旋转 RPE，° | 6.003277 | 9.524084 |
| 首相机参考系位置误差，m | 0.743756 | 1.790471 |

![room 第一处跃升附近的真实 RGB 与深度输入，深度零值留白，固定色标 0 至 6 m](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-room-input-diagnostic-001/20261010T155109Z_3ba18586/event-1-rgb-depth.png)

第一处主要看到近距离暖气片与窗边，梯度较低，但帧 838 仍有 83.04% 有效深度，depth／RGB 关联差不到 1 ms。参考相邻平移约 9.10 mm，而相邻平移 RPE 达 321.93 mm；这说明估计的相邻运动明显不一致，不能直接解释成真实相机突然移动了 32 cm。

![room 第二处跃升附近的真实 RGB 与深度输入，包含桌面物体和视角变化](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-room-input-diagnostic-001/20261010T155109Z_3ba18586/event-2-rgb-depth.png)

第二处包含桌面物体，帧 1133 的梯度 0.293535 高于全序列中位数 0.254826；有效深度还有 74.06%。因此两处不能统一归因于低纹理或深度消失。CPU 输入图和纹理统计使用 `uint8 线性 resize→gray/255` 的 320×240 代理，不是官方 loader 的 `float64 resize→float32 tensor/255→CUDA` 位级一致损失输入。此 RGB 区别不改变独立计算的有效深度和时间统计。梯度和 Laplacian 依赖场景，不能单独测量运动模糊或证明位姿可观测；未做运动补偿的相邻灰度差还混合了视角变化。这里的毫秒差来自文件时间关联，不能替代硬件同步误差测量。

### 局部 SE3 误差变小不等于恢复

帧 837→838 的首相机参考系误差从 0.460322 m 增至 0.743756 m，增加 0.283433 m；同两帧的全序列 SE(3) 对齐位置误差却从 0.202472 m 降至 0.159462 m。后者用整段轨迹事后求一个刚体对齐，会重新分配各段偏差，局部数值变小不代表系统主动纠偏。此时 0.321926 m 的相邻 RPE 仍直接揭示运动估计异常。[冻结轨迹诊断](../outputs/splatam-room-mechanism-diagnostic-001/20261010T155434Z_570bc98c/metrics.json)与[全部逐帧诊断](../outputs/splatam-room-mechanism-diagnostic-001/20261010T155434Z_570bc98c/all-frame-diagnostic.csv)保留三种口径。

## SplaTAM 的损失支持会随位姿改变

记输入颜色、深度为 `I, D`，固定高斯地图为 `Θ`，待估计位姿为 `T`，渲染颜色、深度、silhouette 和深度方差为 `Î(T), D̂(T), S(T), V̂(T)`。本轮官方 tracking 配置使用深度 L1、silhouette 门限 0.99，不启用深度离群点过滤。真正参与残差的像素集合是：

```text
M(T) = {D > 0}
       ∩ {D̂(T) 不是 NaN}
       ∩ {V̂(T) 不是 NaN}
       ∩ {S(T) > 0.99}

n(T) = |M(T)|
L(T) = 0.5 × Σ[p∈M(T), c∈RGB] |I[p,c] − Î(T)[p,c]|
       + 1.0 × Σ[p∈M(T)] |D[p] − D̂(T)[p]|
```

`D>0` 只说明传感器有深度；`S(T)>0.99` 取决于当前地图和当前位姿。源码排除的是 NaN，不能改写成排除所有非有限值。两项 tracking 残差都是 **sum**，RGB 使用同一像素集合扩展到三个颜色通道。掩码在求残差前 `detach`，意味着不通过布尔门限传播梯度；每次 forward 仍会按新位姿重算它，并没有把支持像素固定下来。

因此，比较两个位姿时，目标值同时受到残差大小和支持数量影响。例如某项残差从 1,000 个像素、均值 0.02，变成 500 个像素、均值 0.03，sum 从 20 降至 15，但保留像素的拟合已经变差。零支持时求和为零也不能视作成功位姿。这个例子说明一个可测机制，不能据此断言历史帧 838 或 1133 就发生了支持丢失。

要判断实际变化，应同时记录 `n(T)`、RGB／depth 分项 sum、每个支持像素的平均残差，再用基准位姿的固定集合 `M(T₀)` 重算所有条件的残差。固定集合避免不同条件各自删掉难像素；若原集合中出现不可比较的渲染值，应另报数量，不能静默换一套支持。

### 优化次数和建图候选不代表覆盖

全部 1,314 帧 tracking 均完成 200 次迭代，1,315 帧 mapping 均完成 30 次；这是执行证据，不能代替收敛、梯度或最佳候选损失。两处窗口的 mapping 候选数均为 20，也不等于渲染覆盖充分。官方候选选择把 1,600 个采样有效深度点投影到关键帧，保留投影入图比例大于零的帧，随机取最多 18 帧，再加入最新关键帧和当前帧；入图比例没有检验遮挡或深度一致性，候选列表也没有记录 30 次优化实际抽中了哪些帧。

原运行每 100 帧保存 checkpoint，帧 800／1100 的地图分别含 871,051／879,496 个高斯。它们是各自帧 **mapping 后** 的地图，不能还原帧 838／1133 在 tracking 时的地图、损失掩码或优化器状态。checkpoint 中预分配了 1,315 个相机槽位，也不代表当时已经估计完未来帧。

## 固定地图实测支持变化但没有低总损失反例

固定地图对照已在本机 RTX 5080 实际完成，进程退出 0。实验在 GPU 渲染前登记：分别固定帧 800／1100 的 post-mapping 地图、同帧 RGB-D 和相机参数；输入实际由冻结的官方 `TUMDataset` 读取，沿用其缩放、数值类型和归一化。以保存的估计 W2C 为基准，在相机坐标系左乘一个刚体扰动。每份地图包含基准、三个轴的 ±0.05 m 平移，以及三个轴的 ±5°／±15° 旋转，共 **38 个条件、152 次 renderer forward**。没有优化器步骤、地图更新、GT 位姿替换或时偏拟合。每个条件的原官方 `get_loss` 与独立求和核对，**38 项全部通过**。

这里的平移条件改变 W2C 外参，不是沿世界坐标正方向移动相机中心。由 `C=−Rᵀt`，纯左乘平移 `δ` 对应 `ΔC=−R_baseᵀδ`；图中的 `tx+0.05m` 应按这个坐标约定解释。

| 固定地图与预登记条件 | 支持像素 `n(T)` | 官方动态支持总损失 `L(T)` | 固定基准像素总损失 | 动态支持每像素损失 |
|---|---:|---:|---:|---:|
| 800，原估计位姿 | 42,982 | 10,310.946289 | 10,310.946333 | 0.239890 |
| 800，绕 y 轴 +15° | 38,294 | 34,135.750000 | 53,365.609267 | 0.891413 |
| 1100，原估计位姿 | 44,446 | 19,617.769531 | 19,617.768344 | 0.441384 |
| 1100，绕 y 轴 +15° | 36,922 | 24,571.535156 | 26,740.036001 | 0.665498 |

加权损失混合颜色和深度分项，是目标函数数值，不能标成米或物理距离。动态／固定基准求和在基准上的微小差异来自 float32 与独立求和精度，不是换了一份地图。帧 800／1100 基准的支持只占有效输入深度的 **70.313599%／73.402586%**；传感器有效深度与实际损失支持确实是两种数量。

![两份冻结地图全部预登记条件的损失和支持比例，横轴包含基准与全部扰动](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/fixed-map-support-and-loss.png)

**全部 36 个非基准条件的官方加权总损失都高于对应基准，固定基准像素总损失也全部更高，没有零支持条件。** 本次没有复现“减少有效像素，使错误位姿得到更低总损失”的正例。测量证明了支持集合随位姿改变；它没有证明这个机制使历史帧 838／1133 的优化获胜，也没有排除其他地图、初值和位姿范围下的情况。

分项仍能展示尺度问题：帧 1100 的 y 轴 +15° 扰动使 depth sum 从 7,304.938477 降到 6,616.973145，但参与像素从 44,446 降到 36,922，支持像素的 depth MAE 从 **0.164355 m 增至 0.179215 m**。只看深度 sum 会把拟合方向看反；与此同时 RGB 分项升高，最终加权总损失也升高，所以这个现象不能包装成总目标更偏好该扰动。

![帧 800 冻结地图的预登记显示位姿，依次展示真实输入、渲染颜色、silhouette和官方tracking支持掩码](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/checkpoint-800-registered-probes.png)

图只展示渲染前固定的基准、W2C 的 `tx+0.05m`、y 轴 +15°，没有按事后损失挑选；另一份地图的[同样三种显示条件](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/checkpoint-1100-registered-probes.png)也保留。黑白支持来自实际官方联合掩码，而前面的 RGB／depth 输入图仅表示传感器数据。两份 checkpoint 都已经看过各自当前帧并完成建图，测到的是同帧 post-mapping 目标切片，不能把基准较低解释成独立 GT 位姿最优、全局凸性或历史 tracking 已收敛。

[固定地图真实指标](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/metrics.json)、[全部 38 条条件 CSV](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/pose-support.csv)、[渲染前登记协议](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/plan.json)、[独立诊断脚本](../scripts/diagnose_splatam_checkpoint_support.py)和[实际进程退出观察](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/tool-exit-observation.json)绑定本次实测。

[原始数据索引读回](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/auxiliary-native-index-readback.json)确认官方 loader 使用的 native 索引仍与原完整训练输入一致；[实际图像检查](../outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/visual-inspection.json)绑定三张保存图。这些补证确认输入和展示来源，不扩大固定地图实验的结论范围。

## 渲染分数必须附带地图时刻和评测口径

原日志另有 14 个稀疏 post-mapping 渲染报告，来自帧 0、99、199…1299 的当时地图；它们不是逐迭代 tracking loss，也不是最终地图的 66 个离线样本。

| 当时地图的稀疏在线报告 | PSNR，dB | depth L1 MAE，m |
|---|---:|---:|
| 799→899 | 15.25162→10.12175 | 0.08730113→0.18881790 |
| 1099→1199 | 11.46699→11.45374 | 0.15526940→0.08526113 |

第二个粗区间的深度拟合更好，不能抵消轨迹的后段偏离；地图本身也在继续变化。mapping 的 depth 使用 mean，RGB 使用 `0.8×L1 + 0.2×(1−SSIM)`；tracking 的 sum 目标不能与它们直接同尺度比较。上游日志的 `Depth RMSE` 实际计算的是逐像素绝对误差均值；其 `ATE RMSE` 还使用 W2C 平移列对齐后的平均误差，与这里独立 c2w 相机中心 SE(3) RMSE 口径不同。[稀疏在线报告与指标语义](../outputs/splatam-room-mechanism-diagnostic-001/20261010T155434Z_570bc98c/log-observation-supplement-001/metrics.json)保留原标签和明确含义。

## FAST 的视觉阶段确实留下状态修正

FAST-LIVO2 的 `mat_pre`／`mat_out` 混合了 LiDAR 和 VIO 两类阶段，不能直接按行相减。根据阶段输出源码和相同 elapsed 分组，1,479 行 pre、1,475 行 out 中得到 **737 对** VIO 前后状态；每对的 LIO-out 与紧接着的 VIO-pre 前 19 列完全一致。四行初始化 pre-only 和末尾没有后续 VIO 的一条 LIO 保留。

19 列依次为 elapsed、Euler 三维、位置三维、速度三维、陀螺 bias 三维、加速度 bias 三维、inverse exposure、两个零占位；out 的第 20 列是未畸变点云数。零占位不是重力，第 20 列也不是视觉候选或接受数量。Euler 日志乘的是 57.3，应先除回弧度、重建旋转矩阵，再算 SO(3) 相对旋转角。

737 个视觉 stdout 块的 image header 与对应主轨迹传感器时刻完全匹配；36 个 post-VIO 输出匹配每 20 次 VIO 的采样，并与同一 VIO-out 状态在文本量化范围内一致。[列定义与阶段源码证据](../outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/source-schema-evidence.json)和[逐对状态 CSV](../outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/vio-transitions.csv)支持独立复算。

| 737 次 VIO 的净状态修正 | 中位数 | p95 | 最大值 |
|---|---:|---:|---:|
| 位置范数，mm | 1.215936 | 3.873314 | 10.161944 |
| SO(3) 旋转角，° | 0.021812 | 0.126078 | 0.581369 |

**736／737 次在日志精度下留下净状态改变。** 这比“打印了 VIO 阶段”提供更多证据：状态确实受视觉更新影响。但官方光度残差 debug 被注释，内部迭代接受／回退分支没有实际计数；候选点数、Jacobian／EKF 计时和最终净修正都不能替代逐迭代接受率或光度目标下降。

![真实 VIO 净状态修正与冻结共同参考误差，最后一栏展示全部支持区间的描述性关联](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/vio-state-and-paired-errors.png)

主对照保存的是 LiDAR 更新后、本轮 VIO 前的 IMU body 状态；LIVO 包含之前视觉更新的历史影响。36 行本轮 VIO 后的采样属于另一个输出阶段，不能替代原来 738 行主轨迹与 LIO 739 行的比较。

沿用原 687 个共同参考点和 676 对相邻支持，在每个 `[前参考时刻, 当前参考时刻)` 区间累计真实位置修正范数，再与冻结的 `(LIVO位置误差−LIO位置误差)` 变化比较，Pearson 为 **0.017074**，Spearman 为 **0.022760**。[共同支持区间 CSV](../outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/common-reference-signal-windows.csv)包含全部区间，没有挑选时间段或重新对齐。因变量是对齐后两个标量位置误差之差的变化，不是新的 RPE 或本次 VIO 的即时精度贡献。它们是单序列、自相关信号的描述性关联；修正范数还丢失了方向，不能据此判断视觉没有作用或造成了误差增长。

## 能区分机制的最小对照

SplaTAM 的固定地图测量已经完成，其负结果把下一步收窄到历史在线状态。最小补充是原参数短前缀复现，逐迭代记录实际初值、当前／最佳位姿、动态支持、固定基准支持及 RGB／depth 分项；保持随机调用顺序，并先检查基准前缀是否能复现。之后只改变常速度初始化或一个跟踪残差权重，沿同一窗口比较局部 RPE和首相机参考系偏离。固定地图跟踪对照与允许建图演化的在线对照分别报告，避免把地图反馈混入一个位姿优化结论。

FAST 的更小一步是补足同一 VIO 内部迭代的真实光度残差、候选集合、接受／回退状态以及同一 sensor timestamp 的前后位姿。随后保持标定、时偏、回放速度、输出阶段和共同 GT 时间网格固定，单独控制视觉更新，比较整段与预登记区间的误差。仅记录修正大小还不足以分离视觉信息质量、状态历史和地图演化；当前也没有根据 GT 调整官方 1.907 ms 相机时偏。

## 来源和复查

本轮输入、模型、官方源码和原评分保持原字节；CPU 诊断没有训练、GT 参数拟合或新的精度验收声明。room 窗口属于看过误差后的探索性诊断，FAST 属于一次 exp14 成功运行；二者都不能扩展成跨场景的因果结论。

复现入口为[room 输入诊断脚本](../scripts/diagnose_splatam_room_inputs.py)、[冻结 room 轨迹诊断脚本](../outputs/splatam-ablation-evaluations-001/diagnose_room_frozen.py)和[FAST 状态诊断脚本](../scripts/diagnose_fast_livo2_hilti_vio.py)。FAST 的[结构化指标](../outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/metrics.json)、[实际退出观察](../outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/tool-exit-observation.json)与[可复算协议](../outputs/fast-livo2-hilti-vio-diagnostic-001/20261010T160003Z_aa0ce023/protocol.json)将真实状态变化与原评分绑定。

FAST 的[独立原始日志数值复核](../outputs/fast-livo2-hilti-vio-diagnostic-review-001/20261010T161647Z_978a86b2/review.json)与[复核说明](../outputs/fast-livo2-hilti-vio-diagnostic-review-001/20261010T161647Z_978a86b2/review.md)重新读取原始日志和轨迹，复算逐对状态、分位数及全部共同支持区间，并补核 room 选定帧的原输入。复核确认数值来源；状态修正、目标函数值和 GT 精度仍是不同层次的证据。

固定地图的[独立 CPU 读回](../outputs/splatam-checkpoint-support-review-001/20261010T162221Z_da0b8727/review.json)与[复核说明](../outputs/splatam-checkpoint-support-review-001/20261010T162221Z_da0b8727/review.md)核对全部条件的损失代数、支持计数、数据与源码绑定以及负结果，未重复 GPU 渲染。两类诊断的独立复核均实际正常退出，原轨迹成绩保持原口径。

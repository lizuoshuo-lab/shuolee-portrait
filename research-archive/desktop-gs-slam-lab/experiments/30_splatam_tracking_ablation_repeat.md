---
id: "gslab-exp-splatam-tracking-ablation-repeat-20261010"
slug: "splatam-tracking-ablation-repeat"
title: "30 SplaTAM：跟踪损失消融、重复与房间覆盖"
order: 30
status: "full_ablation_repeats_and_room_independently_verified"
updated: "2026-10-10"
sources:
  - "https://github.com/spla-tam/SplaTAM/tree/da6bbcd24c248dc884ac7f49d62e91b841b26ccc"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_room"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
---

# SplaTAM：跟踪损失消融、重复与房间覆盖

本轮已完成 desk 三种 tracking 权重×三个 seed 的九个完整结果，以及 room 的 1,315 帧 RGB-D seed 0 运行。八个新 desk run 加一个新 room run 均实际正常退出并独立验收，旧 RGB-D seed 0 基线原样复用。**desk 的 RGB-D、RGB 残差、深度残差 ATE 均值分别为 2.004277、2.125213、4.057922 cm；room 的 RGB-D ATE 则为 0.770351 m，后段明显偏离。** 完整性验收通过说明执行与证据可信，不意味着定位精度达标。

这轮得到两个值得连接起来的知识点：局部 RPE 改善不必然降低整体 ATE；桌面短轨迹的厘米级成绩也不能保证更大办公室覆盖下的表现。[最终指标](../outputs/splatam-ablation-repeat-001/metrics.json)与[最终独立复核](../outputs/splatam-ablation-repeat-001/final-independent-review.json)绑定十个结果，144 项最终读回全部通过；[复核说明](../outputs/splatam-ablation-repeat-001/final-independent-review.md)保留原始数值、图与执行来源。上一轮的[完整序列与分辨率对照](29_splatam_sequences_resolution.md)及旧基线保留。

## room 的负结果：更大覆盖下误差在哪段增长

room 使用相同 Freiburg1 公共相机参数、320×240、tracking 200／mapping 30 和 RGB-D 基线权重，实际完成 1,314 帧 tracking、1,315 帧 mapping、265 个关键帧。它从桌面扩展到同一办公室的外墙和重访路径，改变覆盖、视角与运动条件；这不是跨建筑或跨传感器泛化实验。

| room 完整 RGB-D、seed 0 | 实际结果 |
|---|---:|
| 保留帧／观测区间 | 1,315 / 45.371686 s |
| SE(3) ATE RMSE，m ↓ | 0.770351 |
| 首相机参考系 RMSE，m ↓ | 0.949985 |
| 相邻帧 RPE translation RMSE，m ↓ | 0.045099 |
| 相邻帧 RPE rotation RMSE，° ↓ | 2.083181 |
| 最终高斯数／关键帧数 | 835,412 / 265 |
| 原始子进程总 wall，min | 76.003726 |
| 66 个已观测估计视角 PSNR，dB ↑ | 14.647660 |
| 同样本有效深度 MAE，m ↓ | 0.137406 |

![room 真实 XY 轨迹和逐帧位置误差：显示 SE(3) 对齐与首相机参考系两种口径](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-ablation-repeat-001/room-trajectory-and-error.png)

图来自冻结轨迹与逐帧误差 CSV，约 28 s 和 40 s 附近可见明显误差跃升。它把平均 ATE 还原成需要检查的时间区间；仅凭曲线不能确定是纹理、有效深度、地图覆盖、优化状态或某个单一模块造成。按**帧数**四等分后的结果如下，区间含两端，不是等时间或等路径长度。

| 帧区间 | SE(3) 对齐位置 RMSE，m | 首相机参考系 RMSE，m |
|---|---:|---:|
| 0–327 | 0.551350 | 0.082462 |
| 328–656 | 0.446056 | 0.222247 |
| 657–985 | 0.683327 | 0.888676 |
| 986–1314 | 1.184482 | 1.661694 |

首相机参考系揭示从起点向后累积的偏离；对整条轨迹作事后 SE(3) 对齐会重新分配早晚误差，所以第一段 0.551350 m 的对齐 RMSE 不能直接解释成起步时已产生同量漂移。SE(3) 尺度固定为 1，评测对齐也不等于算法实施了全局优化。

GT 路径长 15.975221 m，GT 首尾间距本身为 **1.866034 m**，估计首尾间距为 2.745022 m；它们是路径几何量。末帧对照 GT 的真正位置误差分别为 SE(3) 对齐后 **1.374748 m**、首相机参考系 **1.885690 m**。因此不能把估计端距当作漂移，也不能把首尾接近当作回环成功。默认 SplaTAM 没有回环纠偏或全局 BA，输入重访也不证明存在回环模块。

[图与分段数值收据](../outputs/splatam-ablation-repeat-001/plot-receipt.json)绑定原始 CSV／轨迹、图像哈希和单位换算；[room 协议读回](../outputs/splatam-repeat-evaluations-001/freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918/protocol-review.json)确认完整循环、相同 mapping／源码／renderer 和五项跨序列比较检查。[room 运行审核](../outputs/splatam-repeat-evaluations-001/freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918/metrics.json)保留 28 项实际运行读回，22 项独立轨迹核查与来源由根指标绑定。真实渲染另保留[固定末样本 1299](../outputs/splatam-ablation-repeat-001/freiburg1_room-rgbd-s0-last_sample.png)和[最低 PSNR 样本 359](../outputs/splatam-ablation-repeat-001/freiburg1_room-rgbd-s0-lowest_psnr.png)，后者为 8.799132 dB；末采样帧 1299 并不是最终输入帧 1314。这些是已观测估计视角，不是独立新视角或纯地图质量测量。

## 这项消融具体改变什么

| 跟踪配置 | Tracking RGB 权重 | Tracking depth 权重 | Mapping RGB／depth 权重 |
|---|---|---|---|
| RGB-D 基线 `rgbd` | 0.5 | 1.0 | 0.5 / 1.0 |
| RGB 残差 `rgb_only` | 0.5 | 0.0 | 0.5 / 1.0 |
| 深度残差 `depth_only` | 0.0 | 1.0 | 0.5 / 1.0 |

这里只改变相机 tracking 的加权残差。三组仍读取 RGB-D、用深度反投影初始化高斯、使用深度有效性与 silhouette 掩码、按观测插入新点，并沿用 RGB-D mapping。`rgb_only` 不等于单目或无深度 SLAM；`depth_only` 也没有从整套系统删去颜色。程序仍会计算相关渲染和掩码，权重置零不意味着对应计算成本被移除。

Tracking 每帧冻结当时的地图，优化当前相机；Mapping 固定相机优化地图。本轮每条完整运行都继续建图，不是对同一张冻结地图仅做位姿估计。跟踪损失改变后，估计位姿会改变反投影、新点位置与后续地图，误差又可能影响下一帧 tracking。因此完整序列差异是这一项配置改变在整个系统中的传播结果，不能把地图始终解释成相同控制变量。

RGB 项主要约束纹理投影，深度项约束米制几何；少纹理、重复纹理、平面结构、遮挡与运动变化会改变二者可观测性。Tracking 采用掩码内残差求和，图像误差和米制深度误差的单位不同。不同权重下的原始 loss 总值不可直接当作跨配置的质量排名，最终看同口径的独立 ATE／RPE、逐帧异常和保存模型。

## 相同输入与固定协议

三组 desk 均使用完整 592 个官方关联帧、320×240、tracking 200 次、mapping 30 次，关键帧、窗口、优化器、插入与 pruning 沿用固定官方版本。逐帧 GT tracking 关闭，原始 renderer 和算法源码不改。RGB-D seed 0 复用原已验收基线：SE(3) 相机中心 ATE RMSE 0.020140549 m，相邻帧 RPE 0.009632858 m／0.568176353°，223,097 个高斯。[基线指标](../outputs/splatam-project-001/metrics.json)绑定真实运行，而不是本轮的新消融成绩。

输入最近邻 depth／GT 的时间差都小于 0.08 s，再按相邻保留 RGB 大于 1/32 s 节流；允许复用 depth。不更改关联、时间段、缩图、公共内参和深度尺度。完整轨迹用原始 GT 独立重建、C2W 相机中心、尺度固定 1 的 SE(3) 对齐，另保留首相机参考系的误差。相邻帧 RPE 分别报 m 与 °。最终图像指标使用各次估计视角的 30 个已观测样本，深度指标按实际算法标为 MAE。

RGB 残差和深度残差的 20 帧短区间已分别实际完成：各自 19 帧 tracking 为 200 / 200 次、20 帧 mapping 为 30 / 30；独立轨迹验收 22 项、实际运行读回 28 项均通过。与原 20 帧有效配置比较，除唯一 run name 外，分别只改变 `tracking.loss_weights.depth` 或 `tracking.loss_weights.im`，mapping 配置一致。[RGB 残差协议读回](../outputs/splatam-ablation-evaluations-001/freiburg1_desk_20f_320x240_rgb_only_s0_20261010T092759Z_2461d522/protocol-review.json)与[深度残差协议读回](../outputs/splatam-ablation-evaluations-001/freiburg1_desk_20f_320x240_depth_only_s0_20261010T092911Z_e9eaca5a/protocol-review.json)保留实际权重、配置差异和掩码范围。它们验证入口确实执行预期配置，不提供完整序列的消融排名。

计划在任何完整新运行开始前登记，修订版保留此前记录和哈希。[预登记协议](../outputs/splatam-ablation-repeat-001/plan.json)明确权重、种子、场景与验收规则；[本轮证据报告](../outputs/splatam-ablation-repeat-001/report.md)记录真实完成与失败。每次产生唯一目录，实际退出、配置、源码、renderer、输入、模型和逐帧迭代日志共同绑定。

## 三配置 seed 0：定位、RGB 和深度拟合没有同向变化

RGB 残差 `freiburg1_desk_full_320x240_rgb_only_s0_20261010T093555Z_74acaad0` 与深度残差 `freiburg1_desk_full_320x240_depth_only_s0_20261010T100119Z_9ecfbdb0` 均正常退出，各完成 591 帧 tracking、592 帧 mapping，保存有限模型；各自 22 项独立轨迹核查、28 项运行核查和七项根比较检查通过。[RGB 残差完整协议](../outputs/splatam-ablation-evaluations-001/freiburg1_desk_full_320x240_rgb_only_s0_20261010T093555Z_74acaad0/protocol-review.json)与[深度残差完整协议](../outputs/splatam-ablation-evaluations-001/freiburg1_desk_full_320x240_depth_only_s0_20261010T100119Z_9ecfbdb0/protocol-review.json)确认相对旧基线除唯一 run name 外，分别只改变 `tracking.loss_weights.depth=0` 或 `tracking.loss_weights.im=0`。输入、关联时间戳、mapping 配置、算法源码和 renderer 二进制一致。[真实结果索引](../outputs/splatam-ablation-repeat-001/metrics.json)保留两次模型、执行和审核文件的哈希。

| 同一完整 desk、seed 0 | RGB-D 基线 | RGB 残差 | 深度残差 |
|---|---:|---:|---:|
| SE(3) ATE RMSE，m ↓ | 0.020141 | 0.021358 | 0.044898 |
| 首相机参考系 RMSE，m ↓ | 0.043678 | 0.052599 | 0.069562 |
| 相邻帧 RPE translation RMSE，m ↓ | 0.009633 | 0.008424 | 0.011797 |
| 相邻帧 RPE rotation RMSE，° ↓ | 0.568176 | 0.519895 | 1.023363 |
| 最终高斯数 | 223,097 | 212,779 | 236,045 |
| 原始子进程总 wall，min | 27.0064 | 25.2478 | 24.5359 |
| 30 个已观测估计视角 PSNR，dB ↑ | 22.898804 | 23.283889 | 18.078425 |
| 同样本有效深度 MAE，m ↓ | 0.032347 | 0.037492 | 0.021125 |

ATE 衡量整段轨迹在刚体对齐后的误差，首相机参考系保留从起点累计的偏离；相邻帧 RPE 衡量每一步相对运动。RGB 残差局部 RPE 下降而两种整体误差上升，说明“平均单步更好”不能直接推导“整段漂移更小”。持续的方向偏差、误差相关性或少数区间都可能影响累计结果；当前数值本身不能确定是哪一种机制。

RGB 残差 PSNR 上升而深度 MAE 上升；深度残差的 ATE／RPE 与 RGB 重建更差，但深度 MAE 降到 0.021125 m。后者只说明在这一口径下深度残差较小，不能替代定位准确或共同物理视角下的地图几何结论。这些渲染来自各次模型自己的估计相机，地图也随各自轨迹演化；尚未对这组消融执行共同 GT 视角重渲染。不能由这组平均值直接认定纹理不足、几何退化或某个模块是误差原因。

子进程 wall 包括加载、初始化、tracking／mapping、最终采样评测与保存，排除环境搭建、复制及独立审核；这是各一次运行耗时，不是实时速度或移除模块的计算节省证明。

保留真实原图：[RGB 残差固定末样本 579](../outputs/splatam-ablation-repeat-001/freiburg1_desk-rgb_only-s0-last_sample.png)、[RGB 残差最低 PSNR 样本 39](../outputs/splatam-ablation-repeat-001/freiburg1_desk-rgb_only-s0-lowest_psnr.png)、[深度残差固定末样本 579](../outputs/splatam-ablation-repeat-001/freiburg1_desk-depth_only-s0-last_sample.png)及[深度残差最低 PSNR 样本 339](../outputs/splatam-ablation-repeat-001/freiburg1_desk-depth_only-s0-lowest_psnr.png)，复制与原图 SHA256 一致。末样本按固定采样位置选择，低分样本按最低 PSNR 选择；不能把某张图的单次现象当成模块失效归因。这些对照均仅属于 seed 0。

## 三个 seed：均值、样本标准差和配对差值

三组现在都已有 seed 0、1、2 的完整 desk 结果。RGB-D seed 0 复用旧冻结基线，八个新 desk run 独立验收后纳入汇总，没有重跑或覆盖旧模型。[三 seed 数值独立复核](../outputs/splatam-ablation-repeat-001/desk-repeat-summary-review.json)通过 166 项读回，将真实 child 退出、有效配置、源码／renderer／输入、模型、轨迹与独立 seal 逐项绑定，并从原始数值重新计算六种指标。[复核说明](../outputs/splatam-ablation-repeat-001/desk-repeat-summary-review.md)保留统计口径与逐 seed 对照。

![desk 三种 tracking 配置的三个真实 seed 点与均值、样本标准差](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-ablation-repeat-001/tracking-ablation-three-seeds.png)

点为真实 seed 0／1／2，误差条为均值 ± 样本 SD。ATE 和平移 RPE 由 m×100 显示为 cm，旋转保持 °，高斯数按千显示；原始值与换算记录在[图收据](../outputs/splatam-ablation-repeat-001/plot-receipt.json)。

| 完整 desk，n=3，均值 ± 样本 SD | RGB-D | RGB 残差 | 深度残差 |
|---|---:|---:|---:|
| SE(3) ATE RMSE，cm ↓ | 2.004277 ± 0.016138 | 2.125213 ± 0.017541 | 4.057922 ± 0.382036 |
| 首相机参考系 RMSE，cm ↓ | 4.425780 ± 0.050187 | 5.304015 ± 0.061969 | 6.394522 ± 0.496526 |
| 相邻帧 RPE translation RMSE，cm ↓ | 0.939794 ± 0.020582 | 0.841916 ± 0.006156 | 1.137457 ± 0.038648 |
| 相邻帧 RPE rotation RMSE，° ↓ | 0.559609 ± 0.008030 | 0.523953 ± 0.006924 | 1.012327 ± 0.009603 |
| 最终高斯数 | 221,992.7 ± 1,019.6 | 212,674.3 ± 135.0 | 234,898.7 ± 998.1 |
| 实际 child wall，min | 25.917031 ± 1.391971 | 25.150389 ± 1.087652 | 25.437972 ± 1.661441 |

每个表项先得到三次运行各自的指标，再求均值和样本 SD，分母为 `n−1=2`；不是把所有帧合并后重算一个 RMSE。样本 SD 描述这三次运行的变化，不是置信区间。深度残差的 ATE 离散度更大是本轮观测，不据 n=3 宣布普遍不稳定或统计显著性。墙钟包含初始化与最终评测，旧基线和新运行发生在不同时间，不能作为孤立计算性能基准。

同 seed 差值定义为“前一配置的 run 级 ATE RMSE − 后一配置”，单位 cm：

| ATE 差值，cm | seed 0 | seed 1 | seed 2 | 均值 ± 样本 SD |
|---|---:|---:|---:|---:|
| RGB 残差 − RGB-D | +0.121746 | +0.119316 | +0.121746 | +0.120936 ± 0.001403 |
| 深度残差 − RGB-D | +2.475713 | +1.934345 | +1.750876 | +2.053645 ± 0.376857 |
| 深度残差 − RGB 残差 | +2.353967 | +1.815030 | +1.629130 | +1.932709 ± 0.376475 |

RGB 残差相对 RGB-D 的相邻帧平移 RPE 差值在三个 seed 分别为 −0.120863、−0.095641、−0.077130 cm，旋转 RPE 也都更低。ATE 差值却全部为正，所以局部与全局不同方向不是只出现在 seed 0。它支持当前 desk、分辨率和固定预算下的描述性比较，尚不能从平均数确定误差发生在哪一段或给出单一机制因果。

配对 seed 有助于追溯设置，但配置改变会影响状态、地图、关键帧和随机数消费，不证明两种配置逐次抽到了相同随机数。这三次重复没有覆盖新录制的数据采集方差、另一台机器或任意运动条件。

## 相近 ATE 为什么仍要检查实际 seed 与轨迹

官方入口实际执行 `seed_everything(seed=experiment.config['seed'])`，随后将同一 config 交给 SLAM；helper 为 Python、NumPy、Torch 设置 seed。独立复核读回九份真实 stdout、Loaded Config 与 effective-config，seed 0／1／2 均与登记一致。未启用的 wandb name 可保留 `seed0` 外观标签，不能用它代替真正的 `config.seed`。

RGB-D 的 seed 0／2 ATE 相差约 9.28×10⁻⁶ m，但未对齐原始相机中心的分量最大差约 0.029184 m；RGB 残差的对应分量差约 0.031310 m。这里是两个 RMSE 摘要的差和原始轨迹分量差，不是新的物理精度成绩。九个保存模型与原始轨迹哈希各不相同，保存的 camera arrays 也不同；ATE 接近不能推出复制了模型、忽略了 seed 或轨迹一致。官方 seed helper 也没有建立所有 CUDA 操作的逐位确定性，不能把全部运行变化都归因于 seed。

## 续接执行怎样保留真实证据

本轮原 batch 控制器因 Windows 原子替换进度文件的共享锁错误退出 1，当时已启动的真实训练仍在运行。恢复前先保留原失败和 run 身份，再用精确 PID、FILETIME 与命令确认原进程，等待真实 GetExitCodeProcess=0 后核对模型与冻结验收，才串行启动下一条；没有重训那条模型。

[恢复独立审查](../outputs/splatam-ablation-repeat-001/controller-recovery-review.json)的 39 项真实读回与[恢复说明](../outputs/splatam-ablation-repeat-001/controller-recovery-review.md)保留实际句柄、串行时间和当时冻结结果；CIM 时间字符串解析修复没有改变算法。后续 CPU watcher 31019 的退出码和停止原因仍未知，fresh CIM 未发现该进程。同一冻结 v2 的新前台运行只新增验收 room，旧八个 seal、120 个文件及模型／执行收据保持一致，真实返回 accepted 9、failed 0、exit 0。[前台退出观察](../outputs/splatam-ablation-evaluations-001/watchers/room-final-continuation-001/root-terminal-observation.json)与[最终独立复核](../outputs/splatam-ablation-repeat-001/final-independent-review.json)核实这条边界。工程上用真实身份、退出和产物区分控制记录故障、模型失败及评测中断，避免把未知退出补成成功，也避免重复训练已经完成的模型。

## room 与 desk2 的覆盖边界

官方 desk2 描述为与 desk 相似的办公室四张桌子第二次录制。它可检查新增录制／视角条件，不能包装为独立房间或跨环境泛化。官方 room 从这四张桌子开始，沿整个办公室外墙运动并重访；它增加覆盖、视角和时长，仍是同一办公室和 Freiburg1 相机，保留区间的 GT 首尾并不精确重合。

两个官方 HTTPS TGZ 都已完整获取并通过审核，原始数据与 sensor YAML 已复制到本机 WSL。归档完整 gzip CRC／长度、tar 路径／文件类型、全部 PNG chunk CRC 与解码、原始索引／图片 SHA256、官方关联和原生副本逐文件哈希通过。

| 实际公开输入审核 | desk2 | room |
|---|---|---|
| 官方归档字节 | 349,445,005 | 782,381,450 |
| 原始 RGB／depth／GT 行数 | 640 / 639 / 2,428 | 1,362 / 1,360 / 4,887 |
| 0.08 s 关联后／32 Hz 节流后帧数 | 638 / 635 | 1,362 / 1,315 |
| 保留 RGB 观测区间 | 21.305009 s | 45.371686 s |
| 完整解码 PNG／哈希输入文件 | 1,279 / 1,282 | 2,722 / 2,725 |
| RGB-depth 最大时间差 | 0.019403 s | 0.018446 s |
| 原生副本全部归档文件哈希 | 1,283 / 1,283 一致 | 2,726 / 2,726 一致 |

数值来自实际归档和官方关联，不能把网页总时长直接当作保留帧区间。数据使用 CC BY 4.0，注明 TUM／Sturm 等人 IROS 2012；归档 SHA256 为本地获取后计算，没有声称官方发布摘要。[desk2 数据审核](../outputs/splatam-desk2-dataset-audit-001/metrics.json)、[room 数据审核](../outputs/splatam-room-dataset-audit-001/metrics.json)及[room 获取与审核报告](../outputs/splatam-room-dataset-audit-001/report.md)记录来源与检查。room 已完成本文的完整运行；desk2 仅完成输入准备，本轮没有运行成绩。

二者显式复用同一公共 Freiburg1 K 和深度尺度，新建[desk2 sensor YAML](../configs/freiburg1_desk2.yaml)和[room sensor YAML](../configs/freiburg1_room.yaml)，不修改已有标定。参数复用不称为物理重标定，最近邻关联也不证明硬件同步。

room 单独报告一个 RGB-D seed，不计入 desk 的三 seed 统计。它说明当前固定分辨率和迭代预算在更大同办公室覆盖下暴露明显误差，既不能推出跨环境鲁棒，也不能用不同输入直接归因于某一种残差。

## 怎样把结果变成机制证据

如果某种残差配置的轨迹或地图明显变差，先定位误差发生的帧，再查看该段真实纹理、有效深度、地图覆盖和优化日志，验证可观测性解释。单靠平均 ATE 不能确定原因。即使某配置 PSNR 较高，也可能反映地图对其估计视角的拟合；上一轮共同 GT 视角的正控制已经说明，训练视角质量不能替代物理几何和定位精度。

本轮完整重复与 room 验收已完成：desk 支持当前固定条件下的分项比较，room 暴露明显累计偏离。seed 0 的深度 MAE、desk 三 seed 轨迹统计、room 单 seed 覆盖验证是三种证据范围，应分别解释；当前消融没有共同 GT 视角地图评测，不能合成未经测量的多 seed 几何结论。[本轮报告](../outputs/splatam-ablation-repeat-001/report.md)保留协议、实际退出、冻结哈希和负结果，知识页将这些证据连接到可观测性、误差累积与评测口径。

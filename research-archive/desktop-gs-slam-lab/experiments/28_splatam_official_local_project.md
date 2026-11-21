---
id: "gslab-exp-splatam-official-local-project-20261010"
slug: "splatam-official-local-project"
title: "28 SplaTAM：官方本地项目与独立验收"
order: 28
status: "official_smoke_and_full_sequence_independently_verified"
updated: "2026-10-10"
sources:
  - "https://github.com/spla-tam/SplaTAM/tree/da6bbcd24c248dc884ac7f49d62e91b841b26ccc"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/eval_helpers.py"
  - "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/tree/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
---

# SplaTAM：官方本地项目与独立验收

SplaTAM 是当前第一优先。RTX 5080 已在本机 WSL 中运行固定官方 SplaTAM 和原始深度 renderer：**TUM desk 完整 592 / 592 个官方关联帧完成 tracking、mapping、地图保存与独立评测，相机中心 SE(3) ATE RMSE 为 0.020141 m。** 完整运行真实子进程退出 0，22 项独立轨迹验收和 28 项源码／运行读回检查通过，保存 223,097 个高斯和 120 个关键帧。

这里建立的项目主线是公开数据 → 固定源码与环境 → 优化相机位姿 → 优化高斯地图 → 保存配置与真实输出 → 从原始 GT 独立评测。之前的 Windows gsplat 教学实验保留为历史材料；本记录绑定原始 SplaTAM 的实际运行。[系统知识点](../knowledge/24_splatam_official_rgbd_system.md)解释损失、坐标和高斯更新的因果关系。

本页保留首轮 320×240 desk 基线。后续 xyz **759／759 帧**、desk **640×480 的 592／592 帧**及共同 GT 视角重渲染已经完成，最新对照、较差样本与复现入口见[运动序列与分辨率对照](29_splatam_sequences_resolution.md)。首轮模型和数值保持不变。

## 完整序列与短区间结果

| 项目 | 前 20 帧 smoke | 完整 desk 序列 |
|---|---|---|
| 官方关联帧覆盖 | 20 / 592 | 592 / 592 |
| 分辨率 | 320×240 | 320×240 |
| 观测时间跨度 | 0.636 s | 19.843 s |
| SE(3) ATE RMSE，尺度固定 1 | 0.005198 m | 0.020141 m |
| SE(3) ATE 平均／最大误差 | 0.004647 / 0.011114 m | 0.017154 / 0.055275 m |
| 相邻帧平移 RPE RMSE | 0.005233 m，19 对 | 0.009633 m，591 对 |
| 相邻帧旋转 RPE RMSE | 0.419782°，19 对 | 0.568176°，591 对 |
| 保存高斯／关键帧数 | 91,963 / 6 | 223,097 / 120 |
| 官方子进程用时 | 51.336 s | 1,620.381 s，约 27.01 min |
| 独立轨迹验收 | 22 / 22 通过 | 22 / 22 通过 |
| 源码／实际运行读回 | 28 / 28 通过 | 28 / 28 通过 |

完整输入只有约 19.843 s 的观测时间，实际处理与最终评测／保存花费约 27.01 min，这是离线复现，尚未达到实时处理。子进程用时包含数据加载、初始化、tracking／mapping、最终采样图像评测与保存，排除环境安装、Windows 镜像与独立评测，不应当解释为纯 CUDA kernel 性能。

仅按首相机参考系、不给全段额外刚体对齐时，完整序列平移 RMSE 为 0.043678 m，smoke 为 0.023694 m；两种指标允许的坐标调整不同，需要分别标明。本页首轮结果仅覆盖同一室内序列、seed 0 的一次完整运行。20 帧 smoke 与 592 帧 full 的时间跨度和图像采样不同，数值差异不构成模块消融，也不能用于无人机、其他场景或多 seed 鲁棒性结论。

![完整 592 帧真实轨迹与误差](../outputs/splatam-independent-eval-full-001/trajectory-errors.png)

[完整独立数值](../outputs/splatam-independent-eval-full-001/metrics.json)和[完整独立报告](../outputs/splatam-independent-eval-full-001/report.md)绑定 `freiburg1_desk_full_320x240_s0_20261010T062636Z_c516706e`。实际日志逐帧确认：首帧初始化后，591 个 tracking 帧均完成 200 / 200 次迭代，592 个 mapping 帧均完成 30 / 30，最终评测进入并保存模型。预分配 592 个相机参数不能替代这项实际循环证明。[项目汇总](../outputs/splatam-project-001/metrics.json)绑定两次运行、独立评测与实际源码审核哈希；[项目入口清单](../manifests/splatam-local-project.json)集中指向结果、启动与评测工具。

![20 帧真实轨迹与误差](../outputs/splatam-independent-eval-smoke-001/trajectory-errors.png)

[20 帧独立数值](../outputs/splatam-independent-eval-smoke-001/metrics.json)和[20 帧独立报告](../outputs/splatam-independent-eval-smoke-001/report.md)保留 `freiburg1_desk_20f_320x240_s0_20261010T062157Z_6cd94ed5` 的输入哈希、相机内参核对与验收条件。它覆盖约 0.636 s，只作为短区间链路验收。

## 本机环境如何与官方算法对应

| 层次 | 实际本机配置 | 证据与意义 |
|---|---|---|
| 系统与 GPU | WSL2 Ubuntu 20.04；RTX 5080，SM 12.0 | Windows 驱动向 WSL 提供 GPU |
| Python | 3.10.19 | 项目独立虚拟环境 |
| PyTorch | 2.7.1+cu128 | 支持 Blackwell 的实际 CUDA 运行环境 |
| CUDA 编译器 | nvcc 12.8.93 | 官方开发包校验后解包到项目目录 |
| SplaTAM | `da6bbcd24c248dc884ac7f49d62e91b841b26ccc` | 原始 tracking、mapping 和 loader |
| 深度 renderer | `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110` | 原始实现直接编译，未修改 API 或 CUDA 内核 |

环境位于 Linux `/opt/gs-slam-lab/splatam`，虚拟磁盘位于 D 盘。没有安装 Linux 显卡驱动，也没有修改 FAST-LIVO2 的 ROS 环境。CUDA 扩展编译面向 `sm_120`，实际加载的 `.so` 与验收哈希一致。主仓 Linux 文件与固定 Git blob 一致；Windows 源码中的 CRLF 变为官方 LF 的文件逐项留档，不能宣称两端所有原始字节相同。

原始 renderer 的 1,549 个 tracked blob 均与固定官方树匹配。真实 GPU 验收包含 RGB 前向、`[z,1,z²]` 特征渲染、高斯参数梯度和相机旋转／平移梯度，预训练 AlexNet 权重完整校验并运行真实 LPIPS。合成梯度控制使用 anisotropic 高斯；默认 TUM 地图为 isotropic，其自身旋转不改变协方差，不能把控制结果解释为球形高斯方向可观测。第三返回值 depth 只验证前向，官方 backward 没有消费它的梯度。

[环境验收报告](../outputs/splatam-linux-bootstrap-001/report.md)和[环境数值与哈希](../outputs/splatam-linux-bootstrap-001/metrics.json)绑定实际工具链与加载模块。这类兼容验收证明代码能计算，不直接证明 SLAM 精度。

## 一帧 RGB-D 怎样参与优化

首帧用有效深度反投影，建立米制高斯地图并确定初始相机。后续 tracking 固定地图，以渲染颜色与深度优化当前相机，每帧配置 200 次迭代；首帧不执行这段 tracking。Mapping 在选出的重叠关键帧与当前帧中固定相机，优化地图，每帧配置 30 次迭代。每 5 帧保存关键帧，窗口最多 20 帧，沿用固定官方损失、优化器、新点插入与 pruning。默认没有启用联合 BA、回环或重定位。

深度监督来自第二次颜色渲染：把高斯特征设成 `[z,1,z²]`，以相同合成权重得到累计深度 `D`、silhouette `S` 和二阶原始矩。源码用 `D` 计算损失，不能把它替换成归一化 `D/S` 后仍称为相同优化目标。地图低覆盖区域会插入新点，错误相机位姿也可能随反投影写进地图，形成后续漂移的反馈。

本机配置改变了路径、输出保存、图像评测采样、WandB／GUI 开关和 **320×240 分辨率**，保持上述优化规则。论文和上游 TUM 基准使用 640×480，这次运行不能直接与论文表格比较。深度尺度为 5000，公开 Freiburg1 内参随缩图比例缩放；配置虽有 `crop_edge=8`，固定 loader 没有实际应用裁剪或主点偏移。

## 数据、时间关联与 GT 的边界

desk 原始输入有 613 张 RGB、595 张 depth 和 2,335 条 GT。官方规则为每张 RGB 找最近 depth 与 GT，时间差都小于 0.08 s，再保留相邻已选 RGB 间隔大于 1/32 s 的帧，最终是 **592 帧**。592 帧使用 549 张独立 depth，43 次深度复用是官方最近邻关联允许的结果；去重到 549 帧会改变协议。`stride=1` 表示使用关联后的全部帧，不能写成每张原始 RGB 都被使用。

准备阶段对 desk 与 xyz 的 2,810 个原始输入文件完成 SHA256 和 PNG 解码检查；xyz 保留 759 帧，作为已准备的另一条公开序列。desk 最大 RGB-depth 时间差约 50.43 ms，这个值来自软件关联，不是硬件同步精度。[数据审核](../outputs/splatam-dataset-audit-001/report.md)和[审核数值](../outputs/splatam-dataset-audit-001/metrics.json)记录完整输入口径。数据来自 TUM RGB-D，遵循 CC BY 4.0，并注明 Sturm 等人的 IROS 2012 数据集论文。

`use_gt_poses=false` 关闭逐帧 GT 跟踪分支。GT 仍用于 loader 关联、首相机坐标约定、关键帧有限 GT 检查与最终评测；本项目不能声称程序完全不读取 GT。实际 tracking 损失和重叠选择使用估计位姿。

## 轨迹与图像指标各说明什么

保存的相机参数为 `w2c` 和 scalar-first 四元数 `wxyz`。独立评测先归一化四元数、构造矩阵并取逆，得到 `c2w` 相机中心 `−Rᵀt`；再从原始 `groundtruth.txt` 重建关联，核对保存 GT，使用尺度固定为 1 的 SE(3) 刚体对齐。ATE RMSE 按 `sqrt(mean(error²))` 计算，相邻帧 RPE 另报平移米数和旋转角度，不做尺度拟合或 GT 插值。

固定官方 ATE 输出虽然标签写 RMSE，实际返回对齐后的误差均值，而且使用 `w2c` 平移列。它与独立相机中心 ATE RMSE 不是同一指标。原始日志保留，网站采用独立口径。[评测工具的 24 项控制](../outputs/splatam-evaluator-controls-001/controls.json)验证坐标、单位、尺度、无效输入与实际数据绑定；[固定源码审核](../outputs/splatam-official-contract-001/report.md)记录这些差异。控制通过不能代替真实轨迹结果。

完整序列的上游最终图像评测实际采样第 0、19、39……579 帧，共 30 个已观测视角，不能写成 592 帧图像质量或新视角泛化。颜色在无效 GT depth 处置零后仍按整幅图计算上游指标；深度只在有效 GT 像素上评测。上游深度名为 RMSE 的输出实际是平均绝对误差，应标为 MAE。PSNR／MS-SSIM／LPIPS 反映渲染拟合，不能替代 ATE／RPE 的定位结论。

| 完整地图的 30 个已观测采样视角 | 实际结果 |
|---|---|
| 平均 PSNR | 22.898804 dB |
| 最低／最高 PSNR | 18.304008 / 29.001516 dB |
| 平均 MS-SSIM | 0.940193 |
| 平均 LPIPS | 0.126026 |
| 平均深度 MAE | 0.032347 m |

下面同时展示高、低 PSNR 样本。第 579 帧是 30 个固定采样中 PSNR 最高的一帧，29.001516 dB，不能当作平均质量；桌面布局与物体颜色能恢复，纹理细节仍有平滑。第 39 帧按同一组采样的最低 PSNR 自动选出，18.304008 dB，显示器边缘和桌面细节差异更明显。单张图不足以把误差归因于某一个模块，需要按相同协议检查位姿、地图与图像条件。

![第 579 帧：30 个采样视角中 PSNR 最高的实际渲染](../outputs/splatam-project-001/rendered-view-comparison.png)

![第 39 帧：按 30 个采样最低 PSNR 选出的实际渲染](../outputs/splatam-project-001/lowest-psnr-view-comparison.png)

两张图是上游原始输出的字节一致副本，来源与 SHA256 见[项目汇总](../outputs/splatam-project-001/metrics.json)。原图标题的深度 L1 只保留两位小数，显示 `0.00` 不足以证明零误差。Silhouette 栏使用上游未固定 `vmin=0/vmax=1` 的灰度自动归一化；常数全真与全假掩码都可能显示成黑色。因此第 39 帧纯黑不能直接解释为无覆盖，也不能从颜色推断精确覆盖比例。图像比较、全序列轨迹和逐帧误差应共同阅读，避免只靠最好的一张渲染评价系统。

## 复现与结果读取

在 PowerShell 进入桌面实验目录，执行本机入口。每次生成新的唯一目录，原始模型、checkpoint 和日志保留在本地：

```powershell
cd (Join-Path ([Environment]::GetFolderPath('Desktop')) 'GS-SLAM-Lab')
.\.venv\Scripts\python.exe scripts/splatam_local/run_wsl.py --frames 20 --width 320
.\.venv\Scripts\python.exe scripts/splatam_local/run_wsl.py --frames -1 --width 320
```

Windows 入口调用 WSL 的独立环境，把 final 模型、有效配置、实际退出收据、sensor YAML、环境验收和日志复制回 `outputs/splatam-official-runs-001/<运行名称>`。Checkpoint 留在 Linux；它没有保存全部优化器状态，不是任意中断后无损恢复点。

评测已有模型时，指定真实运行目录，而不是创建一个同名空目录：

```powershell
.\.venv\Scripts\python.exe scripts/splatam_local/evaluate_run.py --run-dir outputs/splatam-official-runs-001/<运行名称> --sequence data/tum/rgbd_dataset_freiburg1_desk --output outputs/<新的独立评测目录>
```

[本机入口说明](../configs/splatam/LOCAL_PROJECT.md)、[Windows 启动脚本](../scripts/splatam_local/run_wsl.py)、[Linux 启动脚本](../scripts/splatam_local/run_official.py)、[独立评测源码](../scripts/splatam_local/evaluate_run.py)可直接对照。验收会核对实际训练前后输入字节、固定源码、有效配置、禁用 GT tracking、保存 K、模型哈希、真实子进程退出与全部配置帧数。网站仅同步知识和明确引用的小型结果、图像、脚本；数据集、模型 NPZ 与环境不复制到网站。

这个项目建立了完整室内 RGB-D 序列的可复现基础和累计轨迹误差证据。第二条公开运动序列与 640×480 配置已在后续记录中独立验收；多 seed 和关键模块消融仍需真实对照。默认流程没有覆盖回环纠偏、LiDAR／IMU 融合或无人机户外重建。

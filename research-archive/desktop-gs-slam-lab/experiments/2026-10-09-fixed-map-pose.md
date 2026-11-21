---
id: "gslab-exp-fixed-map-pose-20261008"
slug: "fixed-map-photometric-pose-recovery"
title: "固定高斯地图的六自由度位姿恢复：实测与误差来源"
order: 4
status: "completed_local_pose_recovery_experiments"
updated: "2026-10-09"
sources:
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/_wrapper.py"
  - "https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html"
  - "https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
---

# 固定高斯地图的六自由度位姿恢复

已有地图来自 Tiny NeRF Lego 的 0–99 号视角训练。此次固定全部 8000 个高斯，仅调整 6 个相机位姿参数，用未参与地图训练的 101 号 RGB 图像提供 photometric loss。初始相机为真值 w2c 左乘明确的 SE(3) 扰动；真实位姿只参与初始化和误差评估，实际 RGB 实验的优化目标没有位姿真值项。

这是单帧、局部初始化条件下的固定地图位姿恢复。没有连续轨迹、在线建图、回环或 ATE；不能据此声称完整 GS-SLAM 已完成。实现入口：[track_fixed_map.py](../scripts/track_fixed_map.py)，原理对应 [位姿优化知识页](../knowledge/06_pose_optimization.md)。

## 实际配置与结果

初始化的平移切向量为 `[0.1, 0.05, -0.03]`，旋转切向量为 `[3°, -4°, 0°]`，合成旋转角约 5°。Lego 数据没有校验物理尺度，下表平移均为 **scene units**，不换算成米或厘米。

| 实验 | 目标图像 | 步数与初始学习率 | RGB MSE 初始→最终 | 平移误差初始→最终 | 旋转误差初始→最终 |
|---|---|---|---|---|---|
| 首跑 pose-001 | 真实数据 RGB | 300；平移 0.003、旋转 0.0015 | 0.054050→0.002994 | 0.115723→0.101041 | 5.000°→1.537° |
| 改善 pose-002 | 相同真实数据 RGB | 1500；平移 0.01、旋转 0.0015 | 0.054050→0.001887 | 0.115723→0.014856 | 5.000°→0.234° |
| rendered-control-001 | 同一地图在 GT 相机下渲染的合成 RGB | 1500；平移 0.01、旋转 0.0015 | 0.050804→0.00006273 | 0.115723→0.025139 | 5.000°→0.387° |

所有运行使用 Adam，学习率每步乘 0.995，`packed=False`，100×100 分辨率，旋转以弧度参与优化。第二次同时增加步数和改变平移学习率，所以它证明这套优化能进一步恢复位姿，不能把改善单独归因于“增加步数”。每轮最优位姿只按 RGB loss 选择，未使用 GT 误差选优。

首跑 300 步只是部分恢复：图像残差明显下降，但平移仍接近初始偏差。保存失败程度是必要证据，单看图像变好容易误判 tracking 已准确。第二次把平移误差降到约 0.015 scene units，并把旋转误差降到约 0.234°，仍有剩余误差。

![相同地图、1500 步实际 RGB 位姿优化](../outputs/tiny-fixed-map-pose-002/comparison.png)

优化循环时间分别为约 1.17、5.60、6.23 秒，排除 JIT 编译、数学/梯度检查、最终保存。这个小图实验的循环时间不能换成在线 SLAM 吞吐或大场景性能。地图参数、色彩、尺度、不透明度全部固定，optimizer 中只有 6 个 pose 标量。

## 梯度真的到达了相机吗

固定版本的 `gsplat/cuda/_wrapper.py` 非 packed projection backward 返回 `v_viewmats`。本实验除查看实现外，也做了两类独立数值检查：

1. CPU 上对 SE(3) 指数映射做零切向量和非零切向量的 autograd gradcheck。零变换严格为单位矩阵；刚性、逆变换检查误差约 1e-15。
2. GPU 上对初始相机的 6 个切向量分量做中心有限差分：`[L(+epsilon e_i)-L(-epsilon e_i)]/(2 epsilon)`，epsilon=1e-4。与解析梯度的方向余弦为 **0.999930**，相对 L2 差为 **0.06944**。6 维解析梯度均非零，所有优化迭代梯度均有限。

有限差分存在近似误差：投影、可见性、排序以及 float32 计算会使数值差分受 epsilon 影响。这里只验证这个地图、这个初始化、这个渲染配置；并没有证明所有相机位置的梯度都准确。

另一条路径先把均值与协方差变到相机坐标，再用单位 viewmat 光栅化：

```text
mu_camera = R * mu_world + t
Sigma_camera = R * Sigma_world * transpose(R)
```

两条路径渲染的最大像素差为约 **1.70e-6**。这同时检查了各向异性高斯协方差的旋转没有被遗漏。代码支持 `--render-mode camera-covariance`，当前实际优化采用原生 `viewmat` 路径。

## 地图误差与优化误差要分别看

真实 RGB 在 GT 相机下渲染的 MSE 为 **0.00185643**。即使位姿完全正确，这张教学地图也不能精确还原目标图像；GT 相机只是评估参照，并不是已证明的 photometric 最小点。颜色、几何、遮挡和固定高斯数带来的建图误差可能让相机朝错误方向补偿。

第二次实际 RGB 最终 MSE 为 0.00188710，已接近 GT 相机渲染的残差，但位姿尚有偏差。它既可能包含未完全收敛的优化误差，也可能受到地图误差影响，现有证据不能把两者完全分离。

合成 control 使用同一地图、GT 相机生成目标，因此目标生成已经用到了 GT。它检验一致模型中的位姿优化，不能作为真实图像 tracking 质量报告。其 GT 相机 loss 为 0，而最终仍为 6.27e-5，说明此配置还未完全收敛；不能把该剩余误差归咎于真实 RGB 与地图的域差异。

## 产物与历史 checkpoint 检查

| 运行 | 指标 | 迭代曲线 |
|---|---|---|
| pose-001 | [metrics.json](../outputs/tiny-fixed-map-pose-001/metrics.json) | [loss.csv](../outputs/tiny-fixed-map-pose-001/loss.csv) |
| pose-002 | [metrics.json](../outputs/tiny-fixed-map-pose-002/metrics.json) | [loss.csv](../outputs/tiny-fixed-map-pose-002/loss.csv) |
| rendered-control-001 | [metrics.json](../outputs/tiny-fixed-map-rendered-control-001/metrics.json) | [loss.csv](../outputs/tiny-fixed-map-rendered-control-001/loss.csv) |

每轮还保存 config、目标/初始/最终图片、pose checkpoint 与完整日志；地图文件 SHA256 前后相同。地图来源为已有实验的 `gaussians.pt`，并未重新优化地图。

[checkpoint 独立检查记录](../outputs/fixed-map-checkpoint-audit.json)确认三个历史 checkpoint 的 `estimated_w2c` 与最终平移/旋转指标一致，重算误差差异小于 1e-9。历史字段 `translation_delta`、`rotation_delta` 保存的是最后一步参数，而 `estimated_w2c` 保存最低 RGB loss 选中的位姿。pose-002 的选中步为 1461，最后步为 1500，两者 w2c 最大差为 6.68e-6；历史结果保留，读取评估位姿应使用 `estimated_w2c`。

当前脚本已改为明确保存 `best_translation_delta`、`best_rotation_delta` 与 `last_*`，避免重建时混用。修正后 CPU SE3 checks 再次通过，尚未重写已有三轮产物。合成 control 历史 metrics 的通用 `gt_usage` 文案遗漏了“GT 创建渲染目标”，检查记录与本页已明确更正含义，历史数值不变。

数据署名：Tiny NeRF Lego 来自官方 NeRF 教程；原始 Lego 资产作者 Håvard Dalen / Heinzelnisse，CC BY-NC，许可层级详见实验室数据清单。

## TUM 真实 RGB：单帧局部恢复与 photometric 偏差

复用 `tum-rgbd-gsplat-001` 的已知真值位姿 RGB-D 地图，固定全部 **30000 个高斯**。地图使用 bundle 的 10 个训练视角，当前目标为 **heldout local index 8 / dataset frame ID 62**；它未参与地图优化。输入缩至 160×120，目标来自 TUM fr1_desk 的真实 RGB。深度用于已有地图训练，此次位姿目标仅使用 RGB MSE，没有深度 loss 或 GT pose loss。

| 指标 | 初始 | 按 RGB loss 选中的位姿 |
|---|---|---|
| 相机中心平移误差 | 0.037415 m | **0.009148 m** |
| 旋转误差 | 3.000° | **0.362340°** |
| RGB MSE | 0.097761 | **0.004756** |
| RGB PSNR | 10.10 dB | **23.23 dB** |

左 SE3 初始化切向量为平移 `[0.03,-0.02,0.01]` 米、旋转 `[3°,0°,0°]`，初始平移切向量与变换后的相机中心位移存在指数映射耦合。运行 1500 步，平移/旋转学习率为 0.003/0.0015、每步衰减 0.995。按 RGB loss 选中第 973 步，没有使用真值误差选优；最后一步 MSE 为 0.004757，旋转误差为 0.363148°，与所选状态分别保存。

![TUM heldout frame 62 的固定地图 RGB 位姿恢复](../outputs/tum-fixed-map-pose-001/comparison.png)

**GT 相机渲染的 MSE 为 0.005746，反而高于优化相机的 0.004756。** 在这张固定地图、这张观测图像和整图 RGB MSE 的定义下，真实相机不具有最低观测损失。这个实测说明通过移动相机可以补偿建图或图像模型残差；更低 RGB loss 不能单独证明几何更准确。剩余 9.15 mm 误差是这一帧、这一地图、GT 扰动初始化条件下的结果，不能写成整体系统已实现毫米级或厘米级在线 SLAM。

六维初始相机梯度均非零、全部迭代梯度有限。中心有限差分 epsilon=1e-4：解析/数值梯度余弦 **0.999967**，相对 L2 差 **0.061665**。原生 viewmat 与相机均值/协方差路线最大像素差为 **2.72e-4**，高于 Tiny 示例但小于单个 8-bit 色阶；两条 float32 参数化也可能因排序、可见性或运算次序产生小差异，不能将等价数学公式描述为逐位相同。

优化循环约 **6.16 秒**，排除编译、检查和保存；peak PyTorch allocated 为 24.77 MiB，它不是总进程显存。地图 SHA256 前后相同，仅优化 6 个 pose 标量。当前修正后的 best/last 切向量保存已经实际 GPU 运行：[checkpoint_audit.json](../outputs/tum-fixed-map-pose-001/checkpoint_audit.json)确认 best 切向量重建选中 w2c 的最大差为 5.96e-8，last 重建最后 w2c 的最大差为 1.19e-7；最终平移/旋转指标与保存位姿重算一致。

产物：[指标与配置](../outputs/tum-fixed-map-pose-001/metrics.json)、[迭代曲线](../outputs/tum-fixed-map-pose-001/loss.csv)、[checkpoint 审核](../outputs/tum-fixed-map-pose-001/checkpoint_audit.json)。TUM RGB-D 数据集署名与许可为 CC BY 4.0。


---
id: "gslab-exp-gaussian-count-20261008"
slug: "gaussian-count-three-seeds"
title: "4000 与 8000 高斯：三种子容量对照"
order: 3
status: "completed_exploratory_comparison"
updated: "2026-10-08"
sources:
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
  - "https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb"
  - "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
---

# 高斯数量怎样影响固定相机的地图优化

在同一 Tiny NeRF Lego 数据、相同训练/留出划分与 1500 次优化下，将固定高斯数量从 4000 增至 8000，三种子平均留出 PSNR 从 **26.844 dB 提升至 27.410 dB**。三组种子都观察到提升。这支持当前小场景与训练配置下更大的表示容量改善图像拟合；不直接证明更准确的几何、位姿或大场景性能。

## 配置和控制变量

- 渲染器：固定 gsplat v1.5.3，提交 `937e29912570c372bed6747a5c9bf85fed877bae`，本地 Windows JIT 补丁保持一致。
- 教学训练器：[train_tiny_gs.py](../scripts/train_tiny_gs.py)。随机三维初始化、固定高斯数、RGB 颜色、各向异性尺度；没有 SH、增密、裁剪、深度监督或相机优化。
- 数据：Tiny NeRF Lego，106 个 100×100 视角；训练 0–99，留出 100–105。公开来源和文件校验见数据清单。
- 实验因素：4000/8000 高斯；每组 seed 0、1、2，1500 步。学习率、损失、相机和图像保持一致。
- 同一 seed 保持随机视角抽样序列一致，但不同点数并不产生逐粒一一对应的初始化；不能声称只增加了一组嵌套高斯。
- 8000/seed 0 复用先前完成的实测结果，另外五项新实验顺序运行，避免相互争用 GPU。优化循环计时排除编译和最后评测。

## 实际结果

| 数量 | seed 0 | seed 1 | seed 2 | 均值 ± 样本标准差 |
| --- | --- | --- | --- | --- |
| 4000 | 26.86296 dB | 26.89062 dB | 26.77837 dB | 26.84398 ± 0.05848 dB |
| 8000 | 27.49656 dB | 27.38883 dB | 27.34585 dB | 27.41041 ± 0.07764 dB |

每项值都是该种子的六个留出视角 PSNR 的算术平均，标准差是三个种子均值之间的样本标准差。它不是六个视角的标准差、标准误或置信区间。同种子差值分别为 +0.63361、+0.49820、+0.56748 dB，平均 +0.56643 dB。

| 资源统计 | 4000 | 8000 | 解释 |
| --- | --- | --- | --- |
| 优化循环耗时均值 | 3.3094 s | 3.4607 s | 极小数据、单次短运行，受 GPU 时钟和调度影响；不能作稳定性能结论 |
| PyTorch allocated 显存峰值均值 | 22.0324 MiB | 23.3016 MiB | 包含本脚本张量，不是进程总显存或整卡占用 |

完整数值：[统计 JSON](../outputs/gaussian-count-study-001/metrics.json)。

![高斯数量与三个种子的验证 PSNR](../outputs/gaussian-count-study-001/capacity-comparison.png)

图中圆点是三个种子，方块与误差条是均值 ± 样本标准差。显示真实离散运行，不拟合或暗示样本外趋势。

## 如何解释

高斯数增加让模型有更多中心、形状和颜色参数，可能更细致地覆盖空间和投影细节。当前随机初始化下，更多粒子也可能增加初始覆盖概率，因此结果混合了表示容量和初始化采样效应。

三次运行都提升了图像质量，但仍只是一份小型合成数据、固定相机和固定步数。要判断初始化、优化收敛与容量分别贡献多少，需要引入深度/点云初始化、训练曲线和更多场景，不能只继续扩大粒子数量。

六个留出视角在本次用于比较超参数，准确口径是**探索性验证集**。若后续继续据此选择点数或学习率，应重新划分独立最终测试集，不能把反复查看的六个视角称为未被选择过程使用的最终测试。

## 复现入口

[顺序运行脚本](../scripts/run_gaussian_count_study.py) 检查已有输出的配置与划分，复用已完成结果，遇到不完整目录则停止调查；不会覆盖旧训练。

```powershell
.\.venv\Scripts\python.exe scripts\run_gaussian_count_study.py
.\.venv\Scripts\python.exe scripts\summarize_gaussian_count.py
```

各运行目录保存模型、loss 曲线、六个留出视角 PNG 和原始指标。校验脚本独立检查 checkpoint 中的粒子数量、张量有限性、训练/留出无交集、统计均值及 PNG 渲染与 float 指标的一致性；8-bit PNG 的 PSNR 会受量化影响。

实际校验通过：六项 checkpoint 均为预期数量且张量有限、划分无交集；保存 PNG 独立重算的 PSNR 与原 float 指标最大差为 0.00629 dB，符合量化差异。[校验记录](../outputs/gaussian-count-study-001/validation.json)。

## 与知识点连接

- [高斯表示与投影](../knowledge/02_gaussian_projection_compositing.md)：粒子数量改变了哪些可优化参数，为什么更多粒子不保证几何准确？
- [评测协议](../knowledge/04_evaluation_protocol.md)：同种子对照、种子标准差与验证集反复选择的含义。
- [基线训练](2026-10-08-local-gsplat.md)：8000/seed 0 的原始来源与边界。

数据署名：Tiny NeRF 教程 Lego 数据；原模型 Håvard Dalen / Heinzelnisse，CC BY-NC。这里只展示本地教学实验及结果，不重新分发原始 NPZ。

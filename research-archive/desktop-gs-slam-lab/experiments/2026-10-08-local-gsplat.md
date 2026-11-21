---
id: "gslab-exp-local-gsplat-20261008"
slug: "local-gsplat-first-training"
title: "RTX 5080 上的第一次高斯优化训练"
order: 2
status: "completed_small_scale_experiments"
updated: "2026-10-08"
sources:
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
  - "https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/image_fitting.py"
  - "https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
---

# RTX 5080 上的第一次高斯优化训练

今天完成了两个真实 GPU 优化实验：先运行 gsplat 官方单图拟合示例，再用相同的可微光栅器进行已知位姿多视图教学训练。两者都保存了模型、输入/输出图像和指标。SplaTAM 相机跟踪与建图仍未运行。

## 结果

| 实验 | 输入与配置 | 结果 | 能证明什么 |
| --- | --- | --- | --- |
| 官方单图拟合 | TUM fr1_desk 一张 RGB，128×96，2000 高斯，500 次优化，seed 0 | PSNR 4.82 → 20.17 dB；MSE 0.32925 → 0.00962 | GPU 前向、反向、参数更新能够把高斯投影拟合到目标图像 |
| 已知位姿多视图 | Tiny NeRF Lego，100×100，8000 高斯，1500 次优化，seed 0；训练视角0–99，留出100–105 | 6个留出视角平均 PSNR 27.50 dB；101号视角9.80 → 27.31 dB | 使用已知相机优化共享三维表示，并在未参与优化的图像上评估 |

计时只包含优化循环：单图约2.90秒，多视图约3.50秒，排除了首次编译、数据下载和最后评测。它们是很小的教学数据，不能外推为大场景训练或 SLAM 性能。PyTorch 报告的 allocated 显存峰值分别约1.51 MiB、23.30 MiB；这不包含全部驱动/context/非PyTorch分配，也不是进程总显存。

单图优化后大结构变清楚，但键盘和纸张细节仍模糊。多视图重建恢复了黄色车辆和底座，细小结构、纹理仍有损失。这里只运行一个随机种子，没有完整论文基准或统计显著性结论。

![官方单图拟合](../outputs/official-image-fit-001/comparison.png)

![留出视角的多视图结果](../outputs/tiny-nerf-gsplat-001/comparison.png)

图像数据署名：TUM RGB-D 数据集，CC BY 4.0；Tiny NeRF 教程的 Lego 数据，原始 Lego 模型作者 Håvard Dalen / Heinzelnisse，CC BY-NC。具体来源与许可层级保留在数据清单中；NPZ 本身没有另行核验到独立许可文本。

## 第一层理解：为什么单图拟合还不是重建

官方示例初始化随机三维高斯，然后固定一个相机，对其投影图像计算 MSE。只需要让这个相机看起来正确，深度和遮挡关系可以存在多种解，所以单张拟合变好不意味着恢复了正确三维结构。

本次直接复用了固定版本 `examples/image_fitting.py` 中的 `SimpleTrainer`，没有改其训练逻辑。外层脚本增加了公共图像加载、固定随机种子、初末指标、模型和图片保存。

入口：[单图实验脚本](../scripts/train_official_image.py)。

## 第二层理解：已知位姿多视图增加了哪些约束

多视图实验用所有训练相机观察同一组高斯；同一个高斯参数会同时影响多个角度的投影。每次随机抽取一个训练视角，优化其颜色残差，长期累积来自不同观察方向的约束。

Tiny NeRF 的相机是 OpenGL c2w；先右乘 diag(1,-1,-1,1) 改成 OpenCV 的相机轴约定，再求逆得到光栅器的 w2c。错误地只做求逆会改变前后/上下方向。

为了看清基础链路，本教学训练器使用固定8000个高斯、随机空间初始化、RGB颜色、黑背景，没有SH、densification、pruning、深度监督或相机优化。它调用实际 gsplat CUDA rasterization，不是图片插值；也不是原始3DGS训练管线的完整复现。

入口：[多视图实验脚本](../scripts/train_tiny_gs.py)。同一运行编号禁止覆盖，下一次使用新 run 名。

## 环境问题及解决

本地原有默认 Python 装的是不支持5080架构的旧PyTorch。实验室建立独立`.venv`，只读复用已有PyTorch 2.7.1+cu128运行包，并仅在实验室内增加小依赖。

gsplat v1.5.3的JIT配置把GCC参数`-Wno-attributes`传给MSVC，导致D8021。Windows分支改用`/O2`，CUDA flags不变，补丁保存在`patches/gsplat-windows-jit.patch`。MSVC选择本机14.29工具集；CUDA12.8编译目标为sm_120。

前后向检查验证了均值、尺度、颜色和不透明度梯度，所有检查梯度有限。该检查的两粒高斯是各向同性，因此旋转梯度为0符合对称性，不能据此证明所有旋转场景的梯度正确。最终两个训练实验进一步验证了真实优化链路。

## 今天与知识笔记的对应

- [相机几何](../knowledge/01_camera_geometry.md)：实际处理了OpenGL/OpenCV约定与TUM反投影。
- [投影与合成](../knowledge/02_gaussian_projection_compositing.md)：看到了高斯参数通过可微渲染接受图像损失梯度。
- [跟踪与建图](../knowledge/03_tracking_mapping_scope.md)：当前只优化地图，位姿由数据给定；下一步加入位姿估计才进入tracking。
- [评测协议](../knowledge/04_evaluation_protocol.md)：训练视角与留出视角严格分开，未将纯渲染或小数据训练时间称为SLAM实时性能。

## 下一次3小时

1. 40分钟：对照相机笔记，手写解释OpenGL c2w转换到gsplat w2c，检查一个世界点的投影。
2. 70分钟：只改变高斯数量（例如4000与8000），保持训练划分、迭代数和seed不变，记录训练和留出视角质量。
3. 50分钟：观察细节丢失和漂浮物，结合参数与梯度解释原因；选择一个TUM短序列准备已知位姿RGB-D建图。
4. 20分钟：写实验记录并口述“已知位姿重建和SLAM相差哪一个估计问题”。

先不同时改变分辨率、步数和高斯数。要比较耗时，使用同一机器、同步计时，并把编译与数据加载单列。

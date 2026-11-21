---
id: "gslab-k04"
slug: "evaluation-ate-rpe-ms-ssim-throughput"
title: "04 ATE、RPE、MS-SSIM 与渲染/SLAM 速度的区别"
order: 4
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://github.com/MichaelGrupp/evo/wiki/Metrics"
  - "https://github.com/spla-tam/SplaTAM/blob/main/utils/eval_helpers.py"
  - "https://github.com/spla-tam/SplaTAM/blob/main/utils/slam_external.py"
  - "https://github.com/VainF/pytorch-msssim"
---

# 04 轨迹、图像和速度指标应对应实际问题

单图拟合先报告图像损失、对照、耗时与显存；已知位姿建图区分训练视角与留出视角。使用给定真值位姿时，轨迹指标不能证明算法定位能力；相机跟踪需要独立轨迹评价。

## 绝对轨迹误差：全局对齐后的位置残差

绝对轨迹误差（Absolute Trajectory Error，ATE）常报告平移残差的均方根（Root Mean Square Error，RMSE）。先统一时间、位姿方向、坐标轴、单位和关联阈值，再声明允许的对齐。

设配对后的估计中心与参考中心分别为 $\widehat{\mathbf p}_i,\mathbf p_i$，一次全局刚体对齐为 $R_A,\mathbf t_A$：

$$
e_i=\|\mathbf p_i-(R_A\widehat{\mathbf p}_i+\mathbf t_A)\|_2,
$$

$$
\operatorname{ATE}_{\mathrm{RMSE}}
=\sqrt{\frac1n\sum_{i=1}^n e_i^2},\qquad
\operatorname{ATE}_{\mathrm{mean}}
=\frac1n\sum_{i=1}^n e_i.
$$

$n>0$ 是有效配对数量。位置、平移与两个统计量均使用同一长度单位，旋转无量纲。二者不是同一个指标。所读SplaTAM官方评价取平移误差的均值，见[评价实现](https://github.com/spla-tam/SplaTAM/blob/main/utils/eval_helpers.py)。若另用RMSE，应单列统计量，不把不同聚合方式放进同一排名；具体版本也需固定。

刚体对齐消去共同旋转和平移，但不会证明局部运动稳定。单目无尺度协议可允许相似变换，必须明确声明；不能替米制RGB-D系统静默修正尺度。缺失、丢帧和跟踪失败也不能从统计中静默删除。

## 相对位姿误差：指定间隔的相对运动差异

相对位姿误差（Relative Pose Error，RPE）比较两帧之间的运动。令 $T_i^{\mathrm{gt}},T_i^{\mathrm{est}}$ 均为相机到世界变换，间隔为 $\Delta$：

$$
E_i=
\left((T_i^{\mathrm{gt}})^{-1}T_{i+\Delta}^{\mathrm{gt}}\right)^{-1}
\left((T_i^{\mathrm{est}})^{-1}T_{i+\Delta}^{\mathrm{est}}\right).
$$

$E_i$ 是 $4\times4$ 相对误差变换，其平移部分与旋转角分别统计。$\Delta$ 可以按帧、秒或路程定义，不同选择测量不同问题，必须报告间隔与单位。ATE和RPE互补，不能互相替代。

## 峰值信噪比、结构相似性与多尺度结构相似性

峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）由像素均方误差计算。对同尺寸、同色彩范围的RGB图像：

$$
\operatorname{MSE}=\frac1{3HW}
\sum_{p,c}(\widehat I_{p,c}-I_{p,c})^2,
$$

$$
\operatorname{PSNR}
=10\log_{10}\!\left(\frac{I_{\max}^2}{\operatorname{MSE}}\right).
$$

$H,W$ 是图像高度与宽度，$p$ 遍历像素，$c$ 遍历三个颜色通道，$I_{\max}$ 为声明的峰值。归一化RGB取 $I_{\max}=1$，PSNR单位为分贝；零误差对应无限PSNR。

结构相似性（Structural Similarity，SSIM）比较局部亮度、对比度与结构；多尺度结构相似性（MS-SSIM）在多个降采样尺度组合相关信息，两者不能互换标签。[SplaTAM评价参考](https://github.com/spla-tam/SplaTAM/blob/main/utils/eval_helpers.py)使用范围为1的MS-SSIM，保存文件名虽为ssim.txt，仍应按照实际指标命名。训练中的SSIM损失也不能直接当成评价MS-SSIM。

分辨率、色彩范围、裁剪、有效像素与留出视角都会改变指标；多尺度窗口还对输入尺寸有要求。将无效像素涂黑再评价整图，会改变统计范围。感知距离指标也须说明所用网络和归一化约定。

训练视角质量只说明对已见数据的拟合。已知位姿多视图至少留出部分未参与优化的帧，并单独报告新视角质量；其相机位姿来源仍须说明。

## 渲染帧率与系统速度

| 计时对象 | 应包含的工作 | 合适的报告方式 |
|---|---|---|
| 单次渲染 | 前向投影、排序与合成 | 分辨率、高斯数、毫秒每帧与帧率 |
| 一次优化迭代 | 前向、损失、反向与参数更新 | 毫秒每迭代、迭代数 |
| 相机跟踪 | 每帧全部位姿优化 | 毫秒每输入帧、迭代数、成功率 |
| 建图 | 地图更新与高斯管理 | 更新频率、单次与总耗时 |
| 端到端 | 数据读取、预处理、跟踪与建图 | 吞吐、延迟与总时间 |

处理 $N$ 个输入帧、总墙钟时间为 $\tau$ 秒时，平均吞吐为：

$$
f_{\mathrm{system}}=\frac N\tau.
$$

单位为输入帧每秒。单次渲染耗时 $\tau_{\mathrm{render}}$ 秒对应 $f_{\mathrm{render}}=1/\tau_{\mathrm{render}}$，不能替代系统吞吐。并行阶段的耗时不能简单相加。

GPU执行可能异步，需要同步后的计时或设备事件。先区分预热、首次编译与稳态开销，再声明数据加载、图片保存和可视化是否计入。

## 手算例子与验证方法

对齐后残差为 $0.01,0.02,0.03\,\mathrm m$，则：

$$
\bar e=0.02\,\mathrm m,\qquad
e_{\mathrm{RMSE}}
=\sqrt{\frac{0.0014}{3}}\,\mathrm m
\approx0.0216\,\mathrm m.
$$

真实运动1米、估计1.05米且旋转相同，该段平移RPE为0.05米。归一化图像的MSE为0.01时：

$$
\operatorname{PSNR}=-10\log_{10}(0.01)=20\,\mathrm{dB}.
$$

教学串行系统中，渲染5毫秒、完整跟踪60毫秒、建图40毫秒、读取10毫秒，可以同时得到：

$$
f_{\mathrm{render}}=\frac{1000}{5}=200\,\mathrm{fps},
\qquad
f_{\mathrm{system}}=\frac{1000}{60+40+10}
\approx9.1\,\mathrm{fps}.
$$

跟踪已包含相应渲染时不能重复加一次。这些是教学数值，不是本机实测。验证应固定场景、分辨率、高斯数量、预热和计时边界，分别测前向渲染、完整优化与含输入输出的总耗时。

## 面试自测

1. 所报ATE是均值还是RMSE，允许尺度对齐吗？
2. 为什么共同平移后对齐ATE可以不变，RPE使用多大间隔？
3. 为什么文件名为ssim.txt，数值仍可能是MS-SSIM？
4. 渲染100帧每秒能否推出在线建图100帧每秒？

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

### 实验验证与适用范围

- 资料：已核对 SplaTAM ATE 聚合与 MS-SSIM 调用；轨迹及计时公式为教学推导。
- [本地 gsplat 训练实测](../experiments/2026-10-08-local-gsplat.md)：单图最终 PSNR 为 20.17 dB；Tiny NeRF 训练使用视角 0–99，留出 100–105，6 个留出视角的逐图 PSNR 算术均值为 27.50 dB。后者使用数据提供的位姿，不是定位评测。
- 优化循环实测时间：单图约 2.90 秒，多视图约 3.50 秒；排除首次编译、下载和最终评测。这些是小分辨率教学训练的循环时间，不是渲染 FPS 或 SLAM 吞吐。
- [TUM 单帧几何实测](../experiments/2026-10-08-rgbd-geometry.md)报告数值闭环与重投影误差，不报告轨迹精度。本页引用的早期训练与单帧几何实验未测 ATE、RPE、SSIM、MS-SSIM 或 LPIPS。后续轨迹实验已分别测 raw、尺度固定的 SE3 ATE 与相邻 RPE，见 [轨迹评测与独立复算](../knowledge/08_trajectory_evaluation.md)及[完整序列回退对照](../experiments/2026-10-09-affine-icp-fallback-controls.md)；它们不能反向改变早期实验的验证范围。

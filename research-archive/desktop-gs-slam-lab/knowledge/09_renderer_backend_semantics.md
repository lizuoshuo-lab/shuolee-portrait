---
id: "gslab-k09"
slug: "gaussian-slam-renderer-backend-semantics"
title: "09 高斯渲染后端替换：深度语义、梯度与数值边界"
order: 9
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/slam_helpers.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/recon_helpers.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/eval_helpers.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
  - "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/blob/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110/diff_gaussian_rasterization/__init__.py"
  - "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/blob/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110/cuda_rasterizer/forward.cu"
  - "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/blob/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110/cuda_rasterizer/backward.cu"
  - "https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/blob/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110/cuda_rasterizer/auxiliary.h"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/_wrapper.py"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu"
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu"
  - "https://docs.gsplat.studio/main/apis/rasterization.html"
---

# 高斯渲染后端替换，首先是语义问题

把两个函数的输入输出改成相同形状，只证明接口可以连接。高斯 SLAM 还依赖深度与覆盖的定义、哪些参数接收梯度、可见性与增删点统计，以及渲染数值如何影响跟踪损失。任一层改变，都可能让同一段跟踪（Tracking）与建图（Mapping）流程产生不同轨迹和地图。

代码对应 SplaTAM 原始 rasterizer 与 gsplat 的已核对版本，完整来源在页末。后端语义等价需要分别验证渲染、梯度与系统行为；能够 import 或连接相同形状，不能代替这些检查。

## 颜色、累计深度与累计不透明度

令像素上的前向合成权重为 $w_i=T_i\alpha_i$。SplaTAM 先用 RGB 作为每粒高斯的三个特征，再单独调用一次渲染器，把每粒高斯的特征改成 $(z_i,1,z_i^2)^\mathsf T$。在黑背景下，第二次调用得到：

$$
T_i(p)=\prod_{j<i}(1-\alpha_j(p)),\qquad
w_i(p)=T_i(p)\alpha_i(p),\qquad
\mathbf C(p)=\sum_iw_i(p)\mathbf c_i.
$$

$$
\begin{bmatrix}
D(p)\\
S(p)\\
M_2(p)
\end{bmatrix}
=\sum_iw_i(p)
\begin{bmatrix}
z_i\\
1\\
z_i^2
\end{bmatrix}.
$$

$p$ 是像素坐标，$\alpha_i$ 是像素有效不透明度，$T_i$ 是前方透射率，$w_i$ 是无量纲合成权重。$\mathbf c_i\in\mathbb R^3$ 为颜色向量；$z_i$ 为相机坐标系的轴向深度。$D$ 的单位为长度，$S$ 为累计不透明度，$M_2$ 为长度平方；原实现的第三个特征通道也称为 $M_2$。

[get_depth_and_silhouette()](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/slam_helpers.py#L197) 负责构造这些特征；get_loss() 使用 $D$ 计算深度损失，使用 $S$ 判断已有地图覆盖，并计算 $M_2-D^2$（该统计量不参与求导） 作为 uncertainty。点插入也读取渲染深度与 silhouette，因此这些量的定义直接影响地图拓展。

colors_precomp 在这里是特征容器。RGB 已经是颜色数值，深度特征中的 z、z² 可以大于 1；适配层不能再对它们做 sigmoid、颜色裁剪或球谐转换。构造 z、z² 时也不能 detach，否则会切断位姿到深度特征的直接梯度路径。

### 为什么不能把 $D$ 直接换成 $D_{\mathrm{norm}}$

gsplat 的 $D$ 表示累积 z-depth，$D_{\mathrm{norm}}$ 表示除以累计 alpha 后的期望 z-depth：

$$
D(p)=\sum_iw_i(p)z_i,\qquad
D_{\mathrm{norm}}(p)=\frac{D(p)}{\max(S(p),\varepsilon)},
\qquad \varepsilon>0.
$$

$\varepsilon$ 是无量纲的数值下限；$S>\varepsilon$ 时，归一化深度为 $D/S$，其单位仍为长度。

单个表面位于 $z=2\,\mathrm m$、像素覆盖 $S=0.5$ 时，$D=1\,\mathrm m$，而 $D_{\mathrm{norm}}=2\,\mathrm m$。两者都可以成为某种算法的输出，但配套的损失、mask 和几何解释不同。直接把 SplaTAM 的 $D$ 换成 $D_{\mathrm{norm}}$，会同时改变深度残差、uncertainty 和插点判断。

同一个例子中 $M_2=2$，$M_2-D^2=1$；而按覆盖归一化后的条件深度方差 $M_2/S-(D/S)^2=0$。因此不能把 SplaTAM 的 uncertainty 不加说明地解读为“只对已覆盖表面求出的条件方差”。覆盖本身会影响这个量。深度监督与覆盖的关系见[07 RGB-D 深度监督](07_rgbd_depth_coverage.md)。

### 原渲染器第三个返回值也不是这个 $D$

固定版本 diff_gaussian_rasterization 返回 (color, radii, depth)。其 CUDA 第三个返回值选择透射率跨过 0.5 时的高斯中心深度，属于 median-depth 路径；Python backward 没有传播这个第三输出的梯度。

所核对的 SplaTAM 主循环与评测调用都丢弃第三返回值，实际可微深度来自前面的 $(z,1,z^2)^\mathsf T$ **颜色特征渲染**。仅看函数返回名 depth，很容易接错目标张量。gsplat 返回的则是 (render_colors, render_alphas, meta)，这两个三元组不能按位置直接互换。

## 保持特征含义的合成关系

下面用特征合成公式说明数学关系。rv 来自 SplaTAM 原 helper，$\mathbf V$、$\mathbf K$ 从原 Camera settings 解码；它只展示三个特征的渲染，不构成完整适配器。

$$
\mathbf f_i^{\,C}=\mathbf c_i,\qquad
\mathbf f_i^{\,D}=
\begin{bmatrix}
z_i\\
1\\
z_i^2
\end{bmatrix},
\qquad
\mathbf F(p)=\sum_iw_i(p)\mathbf f_i.
$$

同一个透明度合成关系可作用于颜色特征或深度特征，但每个通道的单位及梯度路径必须保持原定义。原调用示例保留在末尾实验附录。

RGB 与深度特征应保留为两次调用，便于维持原流程中“密度统计只来自 RGB 渲染”的边界。上述代码没有实现 dummy means2D 的梯度桥接、原标量半径、median-depth 返回值或全部公开接口；直接套上原类名并不能补齐这些内容。

## 相机矩阵与位姿梯度

SplaTAM 的 transform_to_frame() 已根据当前估计位姿变换高斯中心；各向异性高斯还变换其旋转。Tracking 时地图几何 detach、相机保留梯度；mapping 默认反过来，并由参数组学习率进一步决定是否更新。

适配器拿到的 rv["means3D"] 已含这条位姿变换路径。随后传给 gsplat 的应是原 Camera settings 中的静态 viewmatrix，不能再把当前优化位姿作为 viewmat 叠加一次。也不能为了方便转换成 NumPy 再建 Tensor，因为这会断开相机梯度。z、z² 特征的计算图同样需要保留。

[setup_camera()](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/recon_helpers.py) 存储的是转置矩阵：

$$
\mathbf A=\mathbf V^\mathsf T,\qquad
\mathbf B=\mathbf V^\mathsf T\mathbf P^\mathsf T,\qquad
\mathbf V=\mathbf A^\mathsf T,\qquad
\mathbf P^\mathsf T=\mathbf A^{-1}\mathbf B.
$$

$$
f_x=\frac{W}{2\tan(\vartheta_x)},\qquad
f_y=\frac{H}{2\tan(\vartheta_y)},\qquad
c_x=\frac{W(P_{02}+1)}2,\qquad
c_y=\frac{H(P_{12}+1)}2.
$$

$\mathbf A,\mathbf B\in\mathbb R^{4\times4}$ 分别表示所存储的视图矩阵与完整投影矩阵；$\mathbf V$ 为视图变换，$\mathbf P$ 为对应投影矩阵。$\vartheta_x,\vartheta_y$ 是视场角的一半，$W,H$ 为像素宽、高，$f_x,f_y,c_x,c_y$ 均以像素计。矩阵转置是存储约定，不能当作另一次物理变换。

不能把 projmatrix 直接当作纯 P，也不能从图像尺寸猜 $c_x=W/2,\ c_y=H/2$。这会丢掉非中心主点和真实相机变换。第一帧常被设为参考系，并不意味着适配器的所有 Camera 都可以无条件替换成单位矩阵。

原 rasterizer 的 ndc2Pix() 让投影中心为 $f_xX/Z+c_x-0.5$，同时在整数像素 j 采样；gsplat 投影为 $f_xX/Z+c_x$，同时在 $j+0.5$ 采样。使用同一 $\mathbf K$ 时，两者的“高斯中心减像素位置”一致。只看到投影中心差半像素就再改一次主点，会引入新的偏移。

## 投影中心梯度与坐标尺度

原接口的 means2D 是形状 $N\times3$ 的 dummy 张量：forward 不使用它改变中心，custom backward 却会把投影中心梯度写给它。SplaTAM 调用 retain_grad()，后续 densification 读取 means2D.grad[seen,:2]。

固定原内核在 backward 中乘上 $W/2$ 与 $H/2$，保存的是 NDC 尺度梯度；gsplat 的 meta["means2d"] 在 packed=False 下为 $1\times N\times2$，梯度以像素坐标计量。对应的尺度关系是：

$$
\frac{\partial\mathcal L}{\partial x_{\mathrm{ndc}}}
=\frac W2\frac{\partial\mathcal L}{\partial u},
\qquad
\frac{\partial\mathcal L}{\partial y_{\mathrm{ndc}}}
=\frac H2\frac{\partial\mathcal L}{\partial v}.
$$

归一化设备坐标（Normalized Device Coordinates，NDC）$x_{\mathrm{ndc}},y_{\mathrm{ndc}}$ 无量纲，像平面坐标 $u,v$ 以像素计。链式法则中的 $W/2,H/2$ 将两种坐标下的梯度联系起来，不是两个梯度数值本来相等。

若只把原 dummy 张量传进一个最终未使用它的 Python 包装，.grad 不会自动出现。若直接把像素梯度拿去与原 densification 阈值比较，增密频率也会改变。gsplat 的 absgrad 又是另一种统计，不能当作原梯度的同义字段。

完整桥接需要同时保持原前向中心、真实投影计算图和 dummy 的自定义反向语义，并把维度、单位与对应的 RGB 调用匹配起来。固定 TUM 配置本来就关闭梯度 densification，因此受限移植可以明确不支持该开关；不能给 dummy 填零后宣称梯度统计正确。

## 投影半径、有效不透明度与裁剪

| 项目 | 固定原 rasterizer | gsplat 1.5.3 | 对替换的影响 |
| --- | --- | --- | --- |
| 半径返回 | $N$ 标量，基于投影协方差最大特征值的约 3σ 球界 | $1\times N\times2$ 两轴矩形界，并考虑 opacity | $\max(r_x,r_y)$ 能补齐形状，但不能恢复原半径定义；seen 与统计可能不同 |
| 像素 alpha 上限 | 0.99 | 0.999 | 高覆盖像素、遮挡、silhouette 与梯度可能改变；全局夹紧 opacity 不等价于逐像素 alpha 夹紧 |
| 有效近裁 | 内核拒绝 $z\leq0.2$ | 接口默认 near=0.01，可显式指定 | 原 settings 构造 near=0.01 的 P，不代表内核实际使用这个阈值；边界不等式也需核对 |
| 支持范围与 tile | 原球界与其 tile 覆盖策略 | opacity-aware 两轴界 | 即使二维协方差接近，可见贡献与排序工作量仍可能不同 |
| 运算与反向内核 | 原 CUDA 实现 | 独立 CUDA 实现 | float32、排序和可见性不连续处不能预设逐位相同 |

这些差异会反馈到 SLAM：$r>0$ 影响 seen，silhouette 影响跟踪 mask 与插点，深度残差影响相机候选，地图变化又影响下一帧。保留同样的 Python 主循环不保证相同数值轨迹。能否接受这种差异，应由同输入下的渲染、梯度和轨迹比较来说明。


## 如何验证后端替换

先用单高斯与固定相机核对特征定义，再用两种相机参数化检查颜色和六维梯度，最后固定数据和配置比较完整 Tracking/Mapping。测试分别检查前向语义、反向路径与系统行为；某一个后端内部两条路径一致，不证明它与另一个后端等价。

由乘积求导可进一步看出：

$$
D=D_{\mathrm{norm}}S,\qquad
\frac{\partial D}{\partial\theta}
=S\frac{\partial D_{\mathrm{norm}}}{\partial\theta}
+D_{\mathrm{norm}}\frac{\partial S}{\partial\theta}.
$$

$\theta$ 为影响渲染的参数；冻结 $S$ 会去掉右侧第二条导数路径，即使前向数值保持不变。

例如透明度权重为 0.5、深度为 2 m 的单高斯，累计深度 $D=1\,\mathrm m$、覆盖 $S=0.5$、归一化深度 $D_{\mathrm{norm}}=$D$/$S$=2\,\mathrm m$。若将 $D_{\mathrm{norm}}$ 再乘 A 重建 $D$，却在乘法中 detach(A)，前向相同而反向丢失一条梯度路径。

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

本页依据固定版本源码：SplaTAM `da6bbcd`、其原始 rasterizer `cb65e4b`、gsplat 1.5.3 的 `937e299`。已有源码评审、CPU 相机代数检查和 gsplat 内部的 GPU 特征与位姿梯度控制实验。2026-10-10 原始 renderer 在本机 RTX 5080 / WSL2 原样编译，真实 RGB、深度特征、地图参数和相机梯度验收通过；官方 20 帧 tracking / mapping 与独立评测也已完成，见[官方项目主线](24_splatam_official_rgbd_system.md)。这些运行仍没有建立两个后端的图像、梯度或轨迹数值等价性；早期 gsplat 教学实验不能作为官方后端替换成功的证据。

### 能 import，也有独立的运行边界

渲染适配并不会自动消除其余依赖。所查主入口无条件 import `wandb`，数据模块需要 `natsort` 与 `kornia`，评测模块需要 `pytorch-msssim` 与 `torchmetrics`；`use_wandb=False` 只控制运行分支，不能跳过已发生的 import。

更隐蔽的是 [eval_helpers.py](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/eval_helpers.py#L19) 在模块级构造 `LearnedPerceptualImagePatchSimilarity(...).cuda()`。导入时就可能读取/获取预训练权重并分配 GPU，而非等到最终评测才发生。因此“只 import 一下看看”也可能启动 GPU。

Open3D 与 FAISS 出现在仓库其他模块中，但不在所核对的离线 TUM 主入口依赖链；默认 projective 初始化以深度/焦距决定尺度，不调用 KNN。它们不应仅凭 requirements 出现就被判定为这条最小路径的首要阻塞。

真正的评测依赖可以安装并验证，或者把评测初始化显式延后并记录这项改动。用空函数替代 LPIPS、返回零指标或静默跳过失败评测，会破坏结果的含义，不能包装成兼容性修复。

### 实测：同一个 gsplat 后端内，特征与深度是否一致

2026-10-09 的合成控制使用同一组几何、零背景和原值特征 `[z,1,z²]`，分别调用 `RGB`、`RGB+D`、`RGB+ED`。32×32 图像、焦距 24、主点 `(15.5,15.5)`，让像素 `(15,15)` 的采样位置恰好位于高斯投影中心。深度是合成场景单位，没有米制标定。这里的 z 是相机轴向深度，不是射线距离。

| 中心像素：实测通道 / 由实测通道计算 | 单高斯：z=2，opacity=0.5 | 重叠双高斯：z=2/4，opacity=0.5/0.7 |
| --- | ---: | ---: |
| 前向权重，手算 | 0.5 | 0.5、0.35 |
| S 特征 / alpha | 0.5 | 0.850000024 |
| D 特征 / 内置 D | 1 | 2.400000095 |
| 内置 ED | 2 | 2.823529482 |
| 二阶矩 M2 | 2 | 7.599999905 |
| M2−D² | 1 | 1.839999447 |
| M2/S−(D/S)² | 0 | 0.968857582 |
| 严格 mask `S>0.5` | false | true |

双高斯第二个权重是 `(1−0.5)×0.7=0.35`；它不是直接使用第二个 opacity。表中的 M2−D² 和条件方差是把实测 float32 通道转为 Python float 后计算，未称为另一个内核的直接输出。两个场景的全图 `S−alpha` 最大绝对误差分别为 `2.98e-8`、`5.96e-8`；D 特征与内置 D、内置 ED 与 `D/max(alpha,1e-10)` 的最大误差均为 0。三种模式保留的前三个特征通道也一致。实际特征值最大为 7.6，证实该调用没有把它们裁剪至 RGB 的 `[0,1]` 范围。

单高斯恰好覆盖 0.5，说明 `>` 和 `>=` 会产生不同 mask。二阶矩的两个方差式也给出不同数值；这些矩描述高斯中心 z 的合成权重，不能直接当作高斯体内的真实深度方差或经校准的不确定度。

小 JSON 证据：[特征实测](../outputs/renderer-feature-semantics-002/feature_semantics.json)、[完整指标与配置](../outputs/renderer-feature-semantics-002/metrics.json)。可审阅的[运行源码快照](../outputs/renderer-feature-semantics-002/source_script.py)与[原始逐像素 JSON](../outputs/renderer-feature-semantics-002/actual_outputs.json)保留了参数和实际张量；NPZ 只作为本地便利格式。

### 实测：相机变换的两条路径与六维梯度

另一个非对称、各向异性的三高斯场景比较两条路径。直接路径把 `W=exp(ξ^)` 传入 viewmat；变换路径先做 `μ_c=Rμ_w+t`、`Σ_c=RΣ_wRᵀ`，再以单位 viewmat 渲染。两条路径都从当前相机中心坐标构造 z、z²，保留其计算图；不能只变换中心而不变换协方差。

同一个三通道标量残差的两路径图像最大差为 `7.15e-7`，loss 差为 `5.82e-11`，解析六维梯度相对 L2 差为 `8.39e-8`。每条路径分别用步长 `1e-4` 的六维中心有限差分检查：

| 路径 | 梯度方向余弦 | 相对 L2 误差 | 有限、非零解析分量 |
| --- | ---: | ---: | ---: |
| 直接 viewmat | 0.999996662 | 0.002631126 | 6/6 |
| 相机系中心与协方差 | 0.999996424 | 0.002635364 | 6/6 |

这是同一后端、同一标量 loss 在**零 twist 及有限差分 `1e-4` 邻域**的实测参数化一致性。非零 target pose 用于构造残差，没有证明两条路径在任意非零被测 pose 处都等价。它没有覆盖排序变化或可见性边界，也没有验证 SplaTAM 的 dummy means2D 反向接口。

#### 负控制为何需要隔离计算图

首轮 `renderer-feature-semantics-001` 用混合三通道图像 loss，要求 detach 深度特征后总梯度变化超过任意设定的 1%。001 留存证据只支持“在 1% 门槛处失败”，没有保存具体梯度变化值。002 复测相同混合 loss，记录变化约 0.16864%；该数值归属于 002。几何投影路径占据了主要梯度，混合 loss 对特征 z 路径的检错力不足，不能由“小变化”推断 detach 安全。首轮[失败记录](../outputs/renderer-feature-semantics-001/failure.json)和[当时源码](../outputs/renderer-feature-semantics-001/source_script.py)保留；不把它列为完整成功实验。

第二轮保存自己的混合 loss 观察，并增加可手算的单高斯中心控制：`ED=D/S=z`，`L=(ED−2.2)²`。在 z=2、零位姿扰动处，`∂L/∂t_z=2(2−2.2)=−0.4`；覆盖权重在比值中消去，隔离出特征 z 的梯度。实际保留计算图时为 `−0.400000095`，中心有限差分为 `−0.399816781`；detach 后解析六维梯度全部为 0，前向 loss 仍相同。这证明相同前向形状和值不足以保证正确反向路径。修正的是控制的辨别能力，没有降低原阈值来强行通过。

### 当前证据支持到哪一层

2026-10-09 的这组实验中，原后端部分限于固定源码评审。CPU NumPy float64 检查构造非中心主点与非单位相机，验证转置/完整投影矩阵恢复：K 与 V 最大绝对误差均为 0，“中心减采样点”位移差约 `2.84e-14` 像素。当时的 GPU 控制只运行 gsplat 1.5.3，实测其特征/D/ED 关系与两种相机参数化的六维梯度；当时没有安装新依赖或运行原 SplaTAM 入口。后续原始后端的实际运行见本页开头引用的官方项目记录。

完整设置、失败控制与原始证据见[渲染特征、深度和梯度控制实验](../experiments/2026-10-09-renderer-semantics.md)。没有完成原后端同输入参考渲染、radii/可见性比较、dummy means2D 桥接、增删点状态对齐或完整适配层轨迹评测。Windows 依赖能否无修改运行也没有由该实验验证。

固定 SplaTAM 主逻辑存在受限 gsplat 移植的可行路径；其中 gsplat 自身的关键特征和位姿梯度语义已获得合成实测支持。在原后端参考对照完成前，仍不能称为可直接替换的官方数值复现。


### 从理论正文保留的实现片段

以下为本轮公式规范前的实现示例，保留原片段供核查，不是新增实验结果。

```python
# rv 可来自 transformed_params2rendervar()，也可来自
# transformed_params2depthplussilhouette()；特征保持原值与计算图。
image, alpha, meta = gsplat.rasterization(
    means=rv["means3D"],
    quats=rv["rotations"],
    scales=rv["scales"] * camera.scale_modifier,
    opacities=rv["opacities"].reshape(-1),
    colors=rv["colors_precomp"],
    viewmats=V[None],
    Ks=K[None],
    width=W,
    height=H,
    sh_degree=None,
    render_mode="RGB",       # 把 RGB 或 [z,1,z²] 当作普通特征合成
    packed=False,
    rasterize_mode="classic",
    eps2d=0.3,
    near_plane=0.2,           # 显式接近原内核有效近裁；边界仍需核对
    far_plane=1e10,
)
channels = image[0].permute(2, 0, 1)  # [1,H,W,3] → [3,H,W]
# 深度特征调用时：D, S, D2 = channels[0], channels[1], channels[2]
```

---
id: "gslab-k25"
slug: "splatam-tracking-loss-support"
title: "25 SplaTAM 跟踪损失：有效像素、梯度与失败条件"
order: 25
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/gs_helpers.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py"
---

# SplaTAM 跟踪损失：有效像素、梯度与失败条件

**SplaTAM 跟踪优化的目标，是当前地图能够解释的一部分 RGB-D 像素。损失变小，既可能是拟合改善，也可能是参与求和的像素变了；它不直接等于位姿更准。** 理解这个目标，需要同时看有效输入、渲染覆盖、残差尺度和位姿可观测性。

SplaTAM 是 RGB-D SLAM：深度提供米制初始化和几何监督，颜色提供外观监督。跟踪（Tracking）固定高斯地图，只更新当前相机；默认建图（Mapping）固定相机、更新地图。完整系统流程见[RGB-D 高斯 SLAM](24_splatam_official_rgbd_system.md)。这里集中解释跟踪损失，不把建图、渲染质量和定位精度混为一项能力。

## 公式与变量

记高斯地图为 $\Theta$，世界到相机变换为 $T$，点变换满足 $p_c=Rp_w+t$。把地图在位姿 $T$ 下渲染，得到颜色 $\widehat I(T)$、累计深度 $\widehat D(T)$、累计不透明度（silhouette）$S(T)$ 和二阶矩表达式 $U(T)$。

| 变量 | 含义与单位 |
|---|---|
| $I_{p,c}$ | 输入像素颜色，RGB 三个通道按 0–1 归一化，无量纲 |
| $\widehat I_{p,c}(T)$ | 渲染颜色，使用相同颜色尺度 |
| $D_p$ | 输入深度，单位 m；零值表示没有有效深度 |
| $\widehat D_p(T)$ | 渲染累计深度 $\sum_iw_iz_i$，单位 m；原实现没有除以 $S$ |
| $S_p(T)$ | 累计不透明度（silhouette）$\sum_iw_i$，无量纲 |
| $U_p(T)$ | 源码的 $\sum_iw_iz_i^2-\widehat D^2$，单位 m²；不能直接当成已标定的测量协方差 |
| $T$ | $\mathrm{W2C}$ 位姿；平移单位 m，相机中心为 $-R^Tt$ |
| $M(T)$、$n(T)$ | 参与残差的像素集合及数量，不是高斯数量 |

前到后不透明度合成中，透射率常记为 $\mathcal T_i=\prod_{j<i}(1-\alpha_j)$，权重为 $w_i=\mathcal T_i\alpha_i$，累计不透明度为 $S=\sum_iw_i$。透射率表示还未被前方高斯遮挡的比例，累计不透明度表示已经累积的遮挡比例；它们不是同一个量。这里的标量透射率 $\mathcal T_i$ 也要与位姿矩阵 $T$ 区分。

其中 $z_i$ 为第 $i$ 个贡献的相机轴向深度，即相机坐标 $Z$ 分量；$\alpha_i$ 是其像素不透明度。$U$ 的数值采用表中二阶矩组合，计算后在当前反向传播中停止梯度。

在官方 TUM 跟踪设置中，启用 silhouette 门限与深度 L1，未启用深度离群点过滤。支持集合与目标为：

令 $\Omega$ 为全部像素，$\overline{\mathbb R}=\mathbb R\cup\{-\infty,+\infty\}$ 为扩展实数。NaN 不属于该集合；以下表达保留了原实现“排除 NaN，但未排除全部无穷值”的边界。

$$
\begin{aligned}
M(T)&=\{p\in\Omega\mid D_p>0,\ \widehat D_p(T)\in\overline{\mathbb R},\ U_p(T)\in\overline{\mathbb R},\ S_p(T)>0.99\},\\
n(T)&=|M(T)|,\\
\mathcal L_{\mathrm{RGB}}(T)&=\sum_{p\in M(T)}\sum_{c\in\{R,G,B\}}|I_{p,c}-\widehat I_{p,c}(T)|,\\
\mathcal L_D(T)&=\sum_{p\in M(T)}|D_p-\widehat D_p(T)|,\\
\mathcal L(T)&=0.5\,\mathcal L_{\mathrm{RGB}}(T)+1.0\,\mathcal L_D(T).
\end{aligned}
$$

输入有深度，不代表地图在该像素有可信覆盖；$D>0$ 和 $S>0.99$ 是不同条件。原实现排除 NaN，没有排除全部非有限值。权重是数值超参数，不是实测噪声的逆协方差；颜色与深度的加权和也不能标成纯粹的米或距离。

如果改变配置，公式也要随之改变。关闭累计不透明度门控，会移除其阈值条件；启用深度离群点过滤，会增加深度误差门控，支持集合更直接依赖当前残差。在两个门控都关闭时，Tracking 的 RGB 分支会对整幅图求和，不再使用上述颜色集合。

## 固定地图时目标怎样影响相机

Tracking（跟踪）固定地图 $\Theta$，只优化当前相机四元数和平移。选中像素的渲染颜色与深度仍然依赖相机参数，残差梯度因此能够更新位姿；冻结地图不表示图像在数学上与地图无关。

默认 Mapping（建图）使用不同目标：深度为有效像素的平均绝对误差，RGB 为 $0.8\,\mathcal L_1+0.2\,[1-\operatorname{SSIM}]$，不沿用 Tracking 的累计不透明度门控。两阶段的目标数值不能直接比较。Tracking 的颜色求和包含三个通道，比较分项平均值时必须说明分母是 $n$ 还是 $3n$。

## 求和目标与支持选择分别意味着什么

设逐像素标量误差为 $r_p(T)$，当 $n(T)>0$ 时，

$$
\mathcal L_{\mathrm{sum}}(T)=\sum_{p\in M(T)}|r_p(T)|=n(T)\overline r(T),\qquad
\overline r(T)=\frac1{n(T)}\sum_{p\in M(T)}|r_p(T)|.
$$
当支持数量变化时，求和同时混合了像素数量和拟合质量。假设从一千个像素、均值 0.02，变成五百个像素、均值 0.03，总和会下降，但保留像素的平均误差更大。这是目标尺度的数学性质，不是已经证实的历史失败原因。

停止梯度（stop-gradient）不会固定之后每一轮的掩码，也不会切断颜色、深度残差对相机的梯度。它表达的是本次反向传播把支持选择当作常量；布尔门限本身没有平滑导数。下一次位姿更新后，渲染器再计算一次 $S,\widehat D,U$，支持集合可能重新变化。因此优化器不能通过门限导数主动“把被排除像素拉回集合”，也不能假定跨迭代始终在比较同一个目标支持。

只把求和改成平均也不充分。平均 消除了数量的直接乘数，却仍可能只保留容易解释的像素。更严格的诊断要同时报告动态支持与固定基准支持：

动态支持为 $M(T)$；固定基准支持为 $M_0=M(T_0)$，所有候选在同一 $M_0$ 上重算残差；共同支持为 $M_\cap(T)=M(T)\cap M_0$。分别报告集合大小与残差，避免把更换像素误当成改善。动态支持上的分项平均为

$$
\operatorname{MAE}_{\mathrm{RGB}}(T)=\frac{\mathcal L_{\mathrm{RGB}}(T)}{3n(T)},\qquad
\operatorname{MAE}_D(T)=\frac{\mathcal L_D(T)}{n(T)},\qquad n(T)>0.
$$

固定支持适合比较候选位姿，但如果某个候选在基准像素上产生 NaN，应单列无效数量，不能静默删掉后继续宣称同支持。空集合的求和可以为零，平均 则可能为 NaN；两者都不说明位姿成功。

## 像素多也可能缺少有效位姿梯度

可以用六维局部位姿增量 $\xi\in\mathbb R^6$ 理解残差向量 $r$ 的雅可比 $J=\partial r/\partial\xi$；这是分析方法，官方实际优化相机四元数参数和平移。设残差权重矩阵为 $\Lambda$，$J^T\Lambda J$ 的弱方向表示某些运动不容易被当前观测区分，约束质量取决于支持内的内容和空间分布。

| 失败条件 | 对位姿约束的影响 |
|---|---|
| 地图覆盖不足或初值偏离太大 | 有效支持减少，渲染对应关系错误，局部优化可能没有进入可用范围 |
| 颜色平坦或重复纹理 | 光度梯度弱，或不同运动产生近似外观，像素数量无法补足区分能力 |
| 深度主要落在单个平面 | 部分沿平面的运动约束较弱，需要不同朝向几何或颜色补充 |
| 光照变化、反光、动态物体 | 当前观测不满足静态地图解释，残差梯度可能推动错误位姿 |
| 先前位姿错误已经写入地图 | 当前跟踪可能在拟合不一致地图；之后新点继续继承错误坐标 |

整幅输入的纹理指标高，不代表实际支持内的渲染梯度高；深度有效率高，不代表各自由度都可观测。需要查看真正进入目标的像素、残差分布和几何方向，不能只数像素或高斯。

## 怎样检测以及怎样评估改法

首先记录每轮的输入有效深度数、silhouette 覆盖数、联合支持数、与基准支持的交集、分项残差求和／均值以及 NaN／Inf 数量。把损失和产生它的位姿一起保存，再比较轨迹误差，才能区分支持改变、局部拟合和真实运动偏离。

诊断时应区分计算目标的位姿与参数更新后保存的候选位姿，不能自动把前者的目标值当成后者重新渲染的结果。具体保存时序见实验附录；这一实现时序本身不能证明发生了漂移。

| 原实现 | 可评估的改法 | 需要防止的问题 |
|---|---|---|
| 在动态支持上求和 | 记录归一化分项，并尝试同支持的候选比较 | 改 RGB 分母为 $3n$ 会改变相对深度权重，必须说明新目标 |
| 使用硬 silhouette 门限 | 检查支持规模、分布和重合；可研究软覆盖权重 | 放宽门限可能引入地图解释不了的像素，软权重也会改变目标和梯度 |
| 空支持仍可得到求和结果 | 将支持不足单列，跳过不可信更新或使用受控回退 | 不能把零损失选成最佳位姿；回退策略需要单独评测 |
| 依赖局部初始化与固定地图跟踪 | 对初值做有界对照，检查分项和可观测方向 | 固定地图结论不能直接推广到有建图反馈的在线运行 |

这些是检测与修复思路，尚不代表原 SplaTAM 已实现或已证明有效。更换深度为 $\widehat D/S$、引入鲁棒核、调整门限或权重，都属于新算法条件，需要原样基准与单变量对照。评估时使用一致的传感器时间和坐标约定，分别看局部 RPE、整段 ATE及首位姿参考系偏离；总损失 或估计视角的高 PSNR不能替代 GT 定位收益。

## 常见面试问题与易错点

| 问题 | 回答要点 |
|---|---|
| 为什么输入深度很多，仍然可能跟丢 | 损失还要求地图覆盖和渲染数值有效；支持的几何方向与纹理决定可观测性 |
| 停止支持选择的梯度，是不是让跟踪没有梯度 | 只固定本次支持选择；选中像素的渲染残差仍对相机求导 |
| loss 下降是否证明相机更准确 | 需要排查支持与地图是否变化，并用同时间、同物理点的 GT 评测 |
| RGB-only tracking 是否就是单目 SLAM | 不是；深度仍参与初始化、支持筛选和建图，残差权重置零不等于移除整个模块 |
| 把求和换成平均是否就解决了失败 | 只能消除数量直接缩放；支持选择偏差、弱约束和错误初值仍在 |
| silhouette 门限降低是否一定更稳 | 覆盖更大和残差更可信不是同一件事，需要控制目标变化并测定位误差 |
| 新增高斯能否修复错误相机 | 新点使用当前估计位姿；错误位姿可能反过来污染地图 |
| 当前深度通道是否是期望深度 | 原路径用累计深度；除以 silhouette 会改变监督含义 |

## 一个有界实测例子

两份保存地图上的固定输入对照确认：一个基准里真正进入损失的支持只占有效深度的 **70.31%**；全部 **36** 个位姿扰动的加权总损失都高于各自基准；某个条件的 depth sum 下降，支持像素的 depth MAE 却上升 **9.04%**。分项求和更低与平均拟合更好并不等价，但这次没有出现靠减少支持取得更低总损失的正例。

![同一输入与固定地图在不同位姿下的渲染和联合支持，右列白色为实际参与tracking损失的像素](C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/splatam-checkpoint-support-diagnostic-001/20261010T160510Z_6e42d904/checkpoint-800-registered-probes.png)

图中输入保持相同，渲染覆盖与联合掩码随位姿改变；输入有效深度图不能替代右列。地图已使用各自当前帧完成建图，这个实验只观察固定地图的目标切片，没有还原历史失败帧的在线地图、掩码或优化过程，因此不能据此确定历史漂移因果。

## 参考资料

- [官方 get_loss 与 tracking 循环](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py)：掩码、求和、梯度范围和候选位姿保存时序。
- [官方渲染变量构造](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/gs_helpers.py)：高斯如何转换成颜色、深度和 silhouette 特征。
- [官方 TUM 配置](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/configs/tum/splatam.py)：本页公式对应的损失和门控设置。
- [高斯 SLAM 系统](24_splatam_official_rgbd_system.md)、[相机几何](01_camera_geometry.md)与[轨迹评测](08_trajectory_evaluation.md)：连接地图反馈、位姿坐标约定和精度口径。

## 实验附录

[漂移机制与固定地图对照](../experiments/32_slam_drift_mechanism.md)保留输入诊断、完整数值、原图和独立复核。正文结论的适用范围是固定地图测量，不是历史漂移的单一原因证明。


### 官方损失实现与梯度范围

## get_loss 怎样实现这个目标

核心代码位于[官方 get_loss](https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py#L214)。按计算顺序可以读成：

```python
# 摘录核心逻辑，省略渲染与变量管理
nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
mask = (curr_data['depth'] > 0) & nan_mask
if tracking and use_sil_for_loss:
    mask = mask & (silhouette > sil_thres)

mask = mask.detach()
loss_depth = abs(curr_data['depth'] - depth)[mask].sum()
color_mask = torch.tile(mask, (3, 1, 1)).detach()
loss_rgb = abs(curr_data['im'] - im)[color_mask].sum()
loss = loss_weights['im'] * loss_rgb + loss_weights['depth'] * loss_depth
```

Tracking 调用 `transform_to_frame` 时设置 `gaussians_grad=False, camera_grad=True`。选中像素的渲染颜色和深度仍然依赖相机参数，残差的梯度因此能更新位姿；地图在这一步固定。

默认 Mapping 使用不同目标：depth 为有效像素的均值，RGB 为 `0.8×L1 + 0.2×(1−SSIM)`，不沿用 tracking 的 silhouette 掩码。两阶段的目标数值不能直接比较。Tracking 的颜色求和还包含三个通道，比较分项平均值时必须说明分母是 `n` 还是 `3n`。



### 固定配置的支持选择开关

如果改变配置，公式也要随之改变。关闭 `use_sil_for_loss` 会移除 silhouette 条件；启用 `ignore_outlier_depth_loss` 会增加深度误差门控，掩码更直接依赖当前残差。在两个门控都关闭时，tracking 的 RGB 分支会对整幅图求和，不再使用上述颜色掩码。

### 候选保存的具体实现时序

原 tracking 循环在 `optimizer.step()` 前计算 loss，之后按这个 loss 条件保存更新后的候选位姿。诊断时应区分计算 loss 的位姿与 step 后位姿，不能自动把记录的 loss 当成保存候选的重新渲染损失；这项代码时序本身也不能证明发生了漂移。
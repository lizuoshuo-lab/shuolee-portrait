---
id: "gslab-k14"
slug: "keyframes-map-window-history-consistency"
title: "关键帧 地图窗口与历史一致性"
order: 14
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://raw.githubusercontent.com/spla-tam/SplaTAM/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://raw.githubusercontent.com/spla-tam/SplaTAM/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/keyframe_selection.py"
---

# 关键帧 地图窗口与历史一致性

关键帧决定哪些观测被长期保存，窗口决定此刻哪些观测参与优化，地图预算决定场景能保留多少表示。三者不能互相代替：地图还留着某帧初始化的高斯，不代表该帧图像仍在监督；窗口保留首帧，也不能保证整段轨迹一致。

理解这条链，应依次问“哪些帧入库、哪些帧被选中、更新了什么、历史还能约束什么、预算耗尽后怎么办”。以下先讲固定官方 SplaTAM 的机制，再区分教学 V4 的简化。

## 入库 保存有用观测

关键帧是一组可被再次使用的观测，通常包含 RGB-D、内参、时间、帧身份和保存的估计位姿。它不是另一份真值。后续优化使用这些数据时，必须保持图像、深度、内参和位姿的对应关系。

记世界到相机的保存位姿为 $\mathbf T_j$，关键帧观测可写为：

$$
\mathcal K_j=(j,\mathbf I_j,D_j,\mathbf K_j,t_j^I,t_j^D,\mathbf T_j).
$$

$j$ 为帧身份，$\mathbf I_j,D_j$ 为彩色与深度图像，$\mathbf K_j$ 为内参矩阵，$t_j^I,t_j^D$ 为各自采样时间；该元组表达需要一致保存的内容，不规定某种存储接口。

固定官方 SplaTAM 把关键帧入库与执行建图（Mapping） 分开控制，按配置间隔入库，并对首帧与倒数第二帧作特殊处理。每帧可以建图，不代表每帧都成为长期保存的关键帧。

固定间隔便于控制开销，却不等于固定信息增量。同样经过几帧，缓慢移动、原地转动和急速前进可能得到完全不同的基线、重叠和新区域。一般关键帧策略需要在跟踪可信度、视角变化、重复观测与新增信息之间取舍。

## 选窗 让相关历史约束当前地图

固定官方流程从当前有效深度采样像素，用当前估计位姿反投影到世界，再投影到历史关键帧。落在历史图像有效范围内且位于相机前方的点比例，是其重叠代理。

令 $\mathcal P_k$ 为当前有效深度采样得到的世界点集合，$\mathbf T_j$ 为历史相机的世界到相机变换，$\Omega_j$ 为历史图像范围，重叠代理为：

$$
\mathcal P_{k\to j}^{+}
=\{\mathbf x\in\mathcal P_k\mid
z_j(\mathbf x)>0,\ \pi_j(\mathbf R_j\mathbf x+\mathbf t_j)\in\Omega_j\},
\qquad
o_{kj}=\frac{|\mathcal P_{k\to j}^{+}|}{|\mathcal P_k|},
\qquad |\mathcal P_k|>0.
$$

$\mathbf R_j,\mathbf t_j$ 是世界到历史相机的旋转与平移；$z_j(\mathbf x)=[0,0,1](\mathbf R_j\mathbf x+\mathbf t_j)$ 是轴向深度，$\pi_j$ 为针孔投影。比例 $o_{kj}\in[0,1]$ 无量纲。

这里的重叠衡量投影范围，不包含完整的历史深度遮挡一致性检验；“投影落在图像里”不能直接证明同一表面可见。采样、位姿误差与视野条件都会影响这个比例。筛选函数参数名 gt_depth 指传入的传感器深度，不代表窗口选择使用轨迹真值。

该版本先保留重叠比例大于零的历史候选，再随机排列并截取所需数量，随后加入最近关键帧和当前帧。虽然前面有排序，它并非严格取重叠率最高的若干帧。需要读到最终选择步骤，才能判断实际窗口规则。

其选择关系可写为：

$$
\mathcal C_k=\{j\in\mathcal H_k\mid o_{kj}>0\},\qquad
\mathcal W_k=
\{j_1,\ldots,j_{\min(m,|\mathcal C_k|)}\}
\cup\{j_{\mathrm{recent}},k\}.
$$

$\mathcal H_k$ 是可用历史关键帧索引集合，$(j_1,\ldots,j_{|\mathcal C_k|})$ 是候选集合的随机排列，$m$ 为所需历史候选数量；最近关键帧与当前帧按该流程另行加入。没有候选时，前一集合为空。

窗口相关性与历史跨度也有取舍：近期帧易产生可靠重叠，较早视角可能提供更大基线或保留旧区域；有限窗口无法同时持续使用所有历史。

## 优化变量 多视角不自动成为 BA

SplaTAM 默认跟踪（Tracking） 固定高斯地图、优化当前相机；Mapping 固定所选相机、优化高斯。历史帧参加了地图监督，不等于它的相机位姿也重新优化。联合光束法平差（Bundle Adjustment，BA）需要明确把选定相机和地图一起作为未知量，并处理坐标自由度。

用残差目标 $\mathcal L_j$ 表示对应视图约束，变量范围为：

$$
\mathbf T_k^*=\underset{\mathbf T_k}{\operatorname{argmin}}
\mathcal L_k(\mathbf T_k,\boldsymbol\Theta_{\mathrm{old}}),\qquad
\boldsymbol\Theta^*=\underset{\boldsymbol\Theta}{\operatorname{argmin}}
\sum_{j\in\mathcal W_k}\mathcal L_j(\mathbf T_j,\boldsymbol\Theta).
$$

第一式固定地图，第二式固定所选相机。若联合优化，未知量变为 $\boldsymbol\Theta$ 与 $\{\mathbf T_j\}_{j\in\mathcal W_k}$，并须固定共同坐标自由度。

地图数组整体进入优化器，也不意味着每个高斯每一步都有有效约束。看不见、被遮挡或像素贡献很小的点，未必得到足够梯度。参数数量描述表示规模，活跃视角描述当前可用信息。

Mapping 的默认实现每次迭代随机选窗口内一个视角；它不保证各视角获得相同次数的更新。教学 V4 则按窗口顺序轮转。两种方式也都不同于每步同时计算所有视角的联合平均：参数及优化器状态在每个更新后改变，访问顺序会影响过程。

还有一个容易忽略的分母：逐帧平均损失再平均，通常不同于汇总全部有效像素后求均值。令视图数为 $M$，第 $j$ 个视图的有效像素集合为 $\mathcal V_j$，$n_j=|\mathcal V_j|>0$，像素代价为 $\ell_{jp}$，则：

$$
\mathcal L_{\mathrm{frame}}=
\frac1M\sum_{j=1}^{M}\frac1{n_j}\sum_{p\in\mathcal V_j}\ell_{jp},
\qquad
\mathcal L_{\mathrm{pixel}}=
\frac{\sum_{j=1}^{M}\sum_{p\in\mathcal V_j}\ell_{jp}}
{\sum_{j=1}^{M}n_j}.
$$

两者分别给视图或像素相同权重，通常不相等。

不同视角的有效深度与覆盖不同，比较时必须注明聚合方式，见[深度与覆盖](07_rgbd_depth_coverage.md)。Tracking 与 Mapping 的变量边界见[任务范围](03_tracking_mapping_scope.md)。

## 历史约束 地图保存不等于观测持续监督

退出窗口的图像不再直接参加当前残差，其影响可能留在地图参数中，但不会因此自动变成永久约束。后续新视角更新同一高斯时，旧区域的颜色、尺度或位置都可能变化。

历史一致性至少包含三种问题：当前窗口能否被同一地图解释，旧视角的重建是否保持，以及整段相机轨迹是否一致。只看当前 loss、终图点数或首帧图像无法同时回答它们。

错误位姿还可能影响新点初始化。新高斯经当前估计相机反投影到世界，有偏位姿会写入有偏地图；后续 Tracking 再把它当成参考。固定历史相机的 Mapping 可以降低拟合误差，却没有同时校正那段相机历史。

回环检测、位姿图优化、重定位和局部窗口优化需要分别说明。共同世界坐标更正还要同步处理高斯均值、协方差和相机，见[地图坐标更正](18_gaussian_map_pose_correction.md)。

## 预算 新区域与旧信息怎样竞争

深度驱动插点补充未覆盖区域，梯度驱动的高斯复制（clone）或分裂（split） 重新分配已有区域的容量，剪枝（pruning）删除部分低价值基元。这些操作承担不同职责，高斯数量增加不说明具体采用了哪种机制。

固定 TUM 配置启用观测驱动插点与不透明度裁剪，关闭基于投影中心梯度的 3DGS 增密（densification）。插点使用估计位姿，因此增加表示能力也会增加对定位质量的依赖。更多高斯不能自动纠正坐标错误。

点数上限仅给出 $N_k\leq N_{\max}$ 的资源约束，其中 $N_k$ 是当前高斯数量。达到上限后，如果没有裁剪、替换、合并或子地图管理，系统可能无法增加新表面；继续拟合还可能要求已有基元改变。是否把预算花在最有价值的区域，需要结合可见性、观测次数和误差来判断，不能只看总点数。

来源帧 记录初始化来源；活跃窗口 记录当前参加残差的图像；可更正归属 决定轨迹调整时如何更新地图。三者不是同一个字段。来源帧 ID 既不会冻结该点，也不证明它永久属于一个独立物理表面。

## 官方流程与教学简化

| 机制 | 固定官方 SplaTAM 默认流程 | 教学 V4 |
|---|---|---|
| 入库与建图 | 两种间隔分别控制 | 跟踪无失败标记且达到固定间隔后建图与入库 |
| 选窗 | 有正投影重叠的历史候选 最近关键帧 当前帧 | 固定首帧加最近若干关键帧 |
| 视角访问 | 每次 Mapping 随机选一帧 | 按窗口顺序轮转 |
| 相机更新 | 默认 Mapping 固定相机 | Mapping 固定相机 |
| 容量管理 | 深度驱动插点与配置裁剪 | 深度驱动插点与数量上限 无裁剪 |

教学简化有助于看清状态与窗口关系，但其参数、输出字段及限制不能当成官方功能。进一步把关键帧接入整帧循环，读[SplaTAM 系统原理](24_splatam_official_rgbd_system.md)。

## 理解自测

- 为什么首帧一直在窗口里，仍可能出现中段漂移？首帧只约束它看见的区域，历史相机也没有自动获得全局修正。
- 为什么没有新增高斯，仍可能发生建图更新？已有参数可以继续优化；插点和优化是不同操作。
- 为什么有多个关键帧不等于 BA？是否 BA 取决于优化变量和约束，而不是图像数量。
- 怎样检查历史区域是否被损害？固定诊断视角与像素规则，对比不同时间的地图，并控制窗口策略。

## 实验附录

教学 V3/V4 的参数、源文件绑定、窗口访问和添点事件保留在[源码阅读记录](../outputs/keyframes-map-window-source-reading-001/source-reading-evidence.json)及[完整教学对照](../experiments/2026-10-09-full-v4-sequence-controls.md)。[历史视角终图诊断](../experiments/2026-10-09-full-seen-map-quality.md)保留真实图像与适用边界；它不能单独确定遗忘的原因。[官方 Tracking 消融与场景验证](../experiments/30_splatam_tracking_ablation_repeat.md)保留另一条独立管线的成绩。

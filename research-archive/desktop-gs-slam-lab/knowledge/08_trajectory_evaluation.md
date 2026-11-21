---
id: "gslab-k08"
slug: "trajectory-ate-rpe-gauge-no-scale"
title: "08 轨迹评测：时间配对、Gauge、ATE 与 RPE"
order: 8
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools"
  - "https://github.com/MichaelGrupp/evo/wiki/Metrics"
  - "https://github.com/MichaelGrupp/evo/wiki/evo_traj"
---

# 08 一段轨迹要先约定坐标、时间和允许消去的误差

SLAM 地图清楚，不表示相机运动准确。轨迹评测须先固定五件事：位姿方向、时间配对、世界坐标参考、对齐自由度及统计方式，再比较任何成绩。

本文将相机到世界的位姿记为 $\mathbf T_{wc}$，位置单位为米、时间单位为秒。绝对轨迹误差（Absolute Trajectory Error，ATE）反映整段位置轨迹，相对位姿误差（Relative Pose Error，RPE）反映相隔指定间隔的运动误差。

## TUM 文件保存相机到世界的位姿

标准 TUM 轨迹的第 $i$ 行包含八个数，依次是时间戳、三维位置和四元数：

$$
\mathbf r_i=
\begin{bmatrix}
t_i&p_{x,i}&p_{y,i}&p_{z,i}&q_{x,i}&q_{y,i}&q_{z,i}&q_{w,i}
\end{bmatrix}.
$$

$t_i$ 单位为秒；$\mathbf p_i=(p_{x,i},p_{y,i},p_{z,i})^{\mathsf T}$ 是相机投影中心在世界坐标系中的位置，单位为米。四元数 $\mathbf q_i=(q_{x,i},q_{y,i},q_{z,i},q_{w,i})^{\mathsf T}$ 无量纲，标量分量在最后，表示相机到世界的旋转：

$$
\mathbf x_w=\mathbf R_{wc,i}\mathbf x_c+\mathbf p_i,
\qquad
\mathbf T_{wc,i}=
\begin{bmatrix}
\mathbf R_{wc,i}&\mathbf p_i\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
$$

gsplat 高斯形状使用标量分量在前的四元数顺序，不能直接复用 TUM 的分量排列。系统内部若保存世界到相机变换，则导出前需取逆：

$$
\mathbf R_{wc}=\mathbf R_{cw}^{\mathsf T},
\qquad
\mathbf p=-\mathbf R_{cw}^{\mathsf T}\mathbf t_{cw},
\qquad
\mathbf T_{wc}\mathbf T_{cw}\approx\mathbf I_4.
$$

把 $\mathbf t_{cw}$ 直接写成 TUM 位置，会得到另一条轨迹。四元数应满足：

$$
\|\mathbf q_i\|_2=1,
\qquad
\mathbf R(\mathbf q_i)=\mathbf R(-\mathbf q_i).
$$

$\mathbf R(\mathbf q)$ 表示单位四元数对应的三阶旋转矩阵。轻微文本舍入可以在预先声明的容差内归一化，并保留归一化前误差；全零、非有限或明显偏离单位范数的四元数应拒绝，不能静默修复。

## 时间匹配必须独立于轨迹误差

估计与参考必须对应同一相机时刻。若用最近邻关联，令估计时刻为 $t_i$、参考时刻为 $\tau_j$：

$$
j^\star(i)=
\underset{j}{\operatorname{argmin}}\,|\tau_j-t_i|,
\qquad
|\tau_{j^\star(i)}-t_i|<\delta_t.
$$

$\delta_t$ 是预先声明的秒制时间门槛。也可以在刚体变换上插值，但应明确平移与旋转的插值方法、最大参考间隔、边界处理和丢弃比例。不能根据几何误差选更有利的参考帧。

若已导出同序、同时间戳的两份文件，逐行精确配对可验证文件一致性；它不证明传感器与真值在物理上同步。较大的纪元秒数在双精度表示中可能失去纳秒精度，严格核对时可使用整数时间单位或精确十进制表示。

## 坐标规范自由度：地图世界原点是人为选择的

世界原点与方向可以任意选择，而相机间相对运动不变；这一自由度称为坐标规范自由度（gauge freedom）。

在线原型将首相机的相机到世界变换设为 $\mathbf I_4$，首帧 RGB-D 地图位于该相机参考系。原始真值可能使用另一世界坐标系，直接作位置差会混入无关的坐标差异。以首个真值相机锚定的参考为：

$$
\widetilde{\mathbf T}_{wc,i}^{\mathrm{gt}}=
\left(\mathbf T_{wc,0}^{\mathrm{gt}}\right)^{-1}
\mathbf T_{wc,i}^{\mathrm{gt}}.
$$

该式把当前相机坐标映射到首个真值相机坐标；第零帧结果为 $\mathbf I_4$。这种变换只在最终评测中使用，不进入在线初始化、相机损失或新增高斯。

若估计首帧也采用同一锚点约定，未再拟合变换的位置误差为：

$$
e_i^{\mathrm{anchor}}=
\left\|\mathbf p_i^{\mathrm{est}}-
\widetilde{\mathbf p}_i^{\mathrm{gt}}\right\|_2.
$$

单位为米，衡量相对首帧锚点积累的位置偏差。未共享首帧参考的两条轨迹，其原始位置差不能直接解释为这种锚定误差。原始世界坐标真值也可以用于另一个已声明的全局对齐协议，但不能混用参考文件。

## ATE：一次全局刚体对齐后的位置残差

设配对相机中心为 $\mathbf p_i^{\mathrm{est}},\mathbf p_i^{\mathrm{gt}}\in\mathbb R^3$，共有 $N$ 帧。允许一个全局旋转和平移：

$$
(\mathbf R_a,\mathbf t_a)=
\underset{\mathbf R\in SO(3),\ \mathbf t\in\mathbb R^3}{\operatorname{argmin}}
\sum_{i=1}^{N}
\left\|\mathbf R\mathbf p_i^{\mathrm{est}}+\mathbf t-\mathbf p_i^{\mathrm{gt}}\right\|_2^2.
$$

$\mathbf R_a$ 无量纲，$\mathbf t_a$ 单位为米。旋转满足 $\mathbf R_a^{\mathsf T}\mathbf R_a=\mathbf I_3$、$\det\mathbf R_a=1$，可用 Kabsch 刚体配准与奇异值分解求解。整段序列只拟合一次；每帧单独拟合会消去原本要测的轨迹误差。

$$
e_i=
\left\|\mathbf R_a\mathbf p_i^{\mathrm{est}}+\mathbf t_a-\mathbf p_i^{\mathrm{gt}}\right\|_2,
\qquad
\mathrm{ATE}_{\mathrm{RMSE}}=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}.
$$

$e_i$ 和均方根误差均为米。残差均值是另一统计量：

$$
\mathrm{ATE}_{\mathrm{mean}}=\frac{1}{N}\sum_{i=1}^{N}e_i.
$$

例如三个位置残差为 $0.01,0.02,0.03\,\mathrm m$，均值为 $0.02\,\mathrm m$，均方根为约 $0.0216\,\mathrm m$。指标必须同时声明统计量、对齐方法和全部有效帧数量，不能仅沿用其他仓库的输出标签。

全局刚体对齐消去共同旋转与平移，保留局部轨迹变形及尺度差异；首相机锚定误差不再拟合旋转和平移，保留相对于起点的累积差异。两者回答不同问题。

## SE(3) 与 Sim(3) 的区别在是否允许尺度

特殊欧氏群 $SE(3)$ 与相似变换群 $Sim(3)$ 分别对应：

$$
\mathbf p'=\mathbf R\mathbf p+\mathbf t,
\qquad
\mathbf p'=s\mathbf R\mathbf p+\mathbf t,\quad s>0.
$$

$\mathbf p,\mathbf t$ 是三维长度向量，$\mathbf R\in SO(3)$，$s$ 为无量纲尺度。刚体对齐固定 $s=1$，相似变换对齐额外拟合尺度。无米制尺度的单目系统可以声明使用相似变换协议；RGB-D 已提供尺度，拟合 $s$ 可能隐藏本应测出的尺度漂移。

有位置跨度的合成轨迹可检查两种误差：把估计位置整体放大两倍，刚体对齐仍应留下尺度残差；将整段估计轨迹左乘一个全局刚体变换，对齐后的 ATE 应不变。这分别检验应保留与应消去的自由度。

## RPE：相隔指定间隔的相对运动误差

对相机到世界的轨迹，令 $\Delta$ 为帧间隔，当前协议采用相邻配对帧，即 $\Delta=1$：

$$
\mathbf Q_i^{\mathrm{gt}}=
\left(\mathbf T_{wc,i}^{\mathrm{gt}}\right)^{-1}
\mathbf T_{wc,i+\Delta}^{\mathrm{gt}},
\qquad
\mathbf Q_i^{\mathrm{est}}=
\left(\mathbf T_{wc,i}^{\mathrm{est}}\right)^{-1}
\mathbf T_{wc,i+\Delta}^{\mathrm{est}}.
$$

$$
\mathbf E_i=
\left(\mathbf Q_i^{\mathrm{gt}}\right)^{-1}\mathbf Q_i^{\mathrm{est}}
=
\begin{bmatrix}
\mathbf R_{E,i}&\mathbf t_{E,i}\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
$$

$\mathbf t_{E,i}$ 是三维相对运动平移残差，单位为米；$\mathbf R_{E,i}$ 是三阶旋转误差矩阵。若共有 $N_\Delta$ 个有效相对运动对：

$$
\mathrm{RPE}_{t,\mathrm{RMSE}}=
\sqrt{\frac{1}{N_\Delta}\sum_i\|\mathbf t_{E,i}\|_2^2},
\qquad
\mathrm{RPE}_{R,\mathrm{RMSE}}=
\sqrt{\frac{1}{N_\Delta}\sum_i\phi_i^2}.
$$

$\phi_i$ 为旋转误差角。本协议将它转换为度后聚合。对合法旋转矩阵 $\mathbf R_{E,i}=(r_{ab})$，可用：

$$
s_i=
\frac{1}{2}
\left\|
\begin{bmatrix}
r_{32}-r_{23}\\
r_{13}-r_{31}\\
r_{21}-r_{12}
\end{bmatrix}
\right\|_2,
\qquad
c_i=\frac{\operatorname{tr}(\mathbf R_{E,i})-1}{2}.
$$

$$
\theta_i=\operatorname{atan2}(s_i,c_i)\in[0,\pi],
\qquad
\phi_i=\frac{180}{\pi}\theta_i.
$$

$s_i,c_i$ 无量纲，$\theta_i$ 以弧度表示，$\phi_i$ 以度表示。小角度下，结合反对称部分与迹比仅从反余弦求角更稳定；前提仍是误差矩阵满足旋转约束。

全局左乘刚体变换 $\mathbf H$ 在相对运动中抵消：

$$
(\mathbf H\mathbf T_{wc,i})^{-1}
(\mathbf H\mathbf T_{wc,j})=
\mathbf T_{wc,i}^{-1}\mathbf T_{wc,j}.
$$

因此 RPE 不需要先作全局刚体对齐。$\Delta=1$ 帧与间隔一秒不同，丢帧或不等采样间隔时尤其不能混用。

已声明的独立控制中，真值每步平移 $1\,\mathrm m$、估计每步 $1.25\,\mathrm m$，平移 RPE 均方根应为 $0.25\,\mathrm m$；真值不旋转、估计每步旋转 $10^\circ$，旋转 RPE 均方根应为 $10^\circ$。这些数值明确的控制检验指标含义，不能替代真实轨迹精度。

## 短片段的退化与失败记录

只有两个位置、几乎共线或近乎静止的轨迹，不能充分约束全局对齐旋转。奇异值分解仍可能给出数值解，但解未必唯一或稳定。应报告奇异值、轨迹跨度、运动类型和样本数；几十帧的小样例不能代替完整数据集评测。

失败帧不能静默删除。当前原型触发规则后，保留该帧、沿用上一估计位姿，同时跳过相应增图：

$$
\mathbf T_{wc,i}^{\mathrm{est}}=
\mathbf T_{wc,i-1}^{\mathrm{est}}
\qquad\text{当第 }i\text{ 帧跟踪被判为失败时}.
$$

这些状态继续进入轨迹误差。成功率、失败率和 ATE、RPE 应一起报告，避免只对幸存帧评价。有限梯度、覆盖及位移门槛只检测各自对应的问题；没有触发规则，不表示接近真值。

## 短片段为什么不能代表完整轨迹

全局对齐消去共同偏移，短片段或退化运动还可能掩盖累积漂移。局部排序会受后续地图历史、失跟和恢复影响。应以固定配置扩展到完整序列与不同场景，保留失败时刻，再判断改善是否稳定。

## 核心理解自测

- 为什么相机到世界的位置与世界到相机的平移列不能视为同一个量？
- 两份文件时间戳相同，为什么仍可能有物理采样时间误差？
- 首相机锚定误差与全局刚体对齐的 ATE，各消去了什么自由度？
- 对 RGB-D 轨迹允许相似变换对齐，会隐藏哪些错误？
- 为什么相邻帧 RPE 不能改名为一秒间隔 RPE？
- 为什么短片段 ATE 低、对齐矩阵满秩或没有失败标记，都不足以证明系统准确？

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

SLAM 的地图图像看起来清楚，不代表相机运动准确。轨迹评测需要分别核对位姿约定、时间配对、世界坐标 gauge、对齐自由度和误差聚合。本页使用本地 20 帧在线教学原型的真实输出，并由 [NumPy 独立审计脚本](../scripts/audit_tum_trajectory.py)重算指标。

### 7. 本地 20 帧实测：独立指标与系统内部吻合

TUM fr1_desk 连续 frame ID 30–49，160×120，GT 只用于最终评估。独立脚本读取导出的文本，不导入在线原型的评价函数：

| 指标 | 独立重算 | 与内部结果的绝对差 |
|---|---|---|
| raw anchored translation RMSE | **0.210186673 m** | 7.55e-11 m |
| SE3 no-scale ATE RMSE | **0.097521820 m** | 6.93e-11 m |
| Δ1 RPE translation RMSE | **0.026326844 m** | 5.79e-9 m |
| Δ1 RPE rotation RMSE | **1.160087486°** | 3.58e-8° |

这是指标实现一致性的证据，结果本身显示明显漂移。片段只有 **0.6323 秒**，参考路程约 **0.1703 米**，估计路程约 **0.4391 米**，末帧 raw 位置偏差约 **0.3171 米**。小分辨率原型能连续执行 tracking/mapping，不等于已经定位准确。

本次拟合的全局旋转约 **72.11°**，位置互协方差的最小/最大奇异值比约 **7.24e-4**。轨迹虽数值 rank=3，短片段及弱方向仍可能让对齐对误差敏感；“满秩”不证明对齐稳健。SE3 对齐后 ATE 降低，也不能让已经观测到的 raw 锚定漂移消失。

同一 20 帧、相同随机种子及其余配置下，又完成三个单因素对照，并对各自文本轨迹独立重算：

| 配置变化 | raw anchored RMSE / m | SE3 ATE RMSE / m | Δ1 RPE / m | Δ1 RPE / ° |
|---|---|---|---|---|
| 基线：constant velocity、depth weight 0.1、100 tracking steps | 0.210187 | 0.097522 | 0.026327 | 1.160087 |
| 仅 initializer 改为 previous estimate | 0.202442 | 0.091191 | 0.025078 | 1.067249 |
| 仅 depth weight 提高到 1.0 | **0.029057** | **0.014353** | **0.006728** | **0.591584** |
| 仅 tracking steps 增加到 300 | 0.217298 | 0.102546 | 0.027089 | 1.192784 |

提高深度权重在这一短片段中减少了实际轨迹偏差；它同时作用于当前原型的 tracking/mapping loss，不能把改善单独归因于某一个阶段。增加优化步数没有改善这次误差，说明“多迭代”无法自动纠正当前目标或地图带来的偏差。这些是单种子、单片段的数值观察，没有多次运行或跨场景统计显著性结论。

depth weight 1.0 对照末帧 raw 偏差仍为 **0.052721 米**。其 0.014353 米对齐 ATE 是整段经过全局拟合的 RMSE，不能改写成“每时每刻都在 1.44 厘米内定位”。独立与内部指标最大差仍小于 2.1e-8 米 / 1.9e-7°，三个对照也通过保存位姿、时间戳、来源点与因果窗口检查。

对照审计：[previous estimate](../outputs/tum-online-previous-001/independent_trajectory_audit.json)、[depth weight 1.0](../outputs/tum-online-depth1-001/independent_trajectory_audit.json)、[300 tracking steps](../outputs/tum-online-steps300-001/independent_trajectory_audit.json)。

### 9. 文件、checkpoint 与新增点来源的交叉检查

[独立 checkpoint/provenance 审核](../outputs/tum-online-gsplat-001/independent_checkpoint_provenance_audit.json)确认：

- checkpoint 的 c2w 与 w2c 双向相乘最大误差约 1.4e-7；首帧严格 identity，PT 与 JSON 保存的估计位姿一致。
- TUM 文本与 PT 的相机中心最大差约 4.82e-10 米；原 float32 旋转累计正交误差约 2.3e-5，导出 normalized quaternion 后旋转元素最大差约 9.86e-6，造成的指标差仍如上表微小。
- 最终高斯来源为 frame 30 的 14086 个首帧点，以及 frame 34/38/42 分别新增 326/124/104 个点，数量与因果增图事件、每帧 point count 一致。
- mapping window 不包含当前时刻以后的帧；静态实现核对确认初始 sensor copy 的 c2w 为 identity，新增点使用已接受的估计 c2w，未使用实际 GT 来放置点。

来源 ID 和日志属于可核对的运行证据，不能单独构成“任意实现都不存在 GT 泄漏”的形式化证明。需要结合源代码数据路径、保存字段与运行协议审计，不应只看某个 `gt_usage` 字符串。

### 10. 可复用的独立审计入口

在实验室根目录运行：

```text
python scripts/audit_tum_trajectory.py \
  outputs/tum-online-gsplat-001/trajectory_estimated.txt \
  outputs/tum-online-gsplat-001/trajectory_groundtruth_anchored.txt \
  --self-test --output outputs/tum-online-gsplat-001/new_audit.json
```

脚本只使用 NumPy 与 Python 标准库，不编译 CUDA、不占 GPU。默认精确配对；行数不一致、重复/非递增时间、非有限值、非法四元数或不匹配的 timestamp 会报错。输出文件要求新名称，避免覆盖历史审计。

产物：[独立轨迹指标](../outputs/tum-online-gsplat-001/independent_trajectory_audit.json)、[checkpoint 与来源检查](../outputs/tum-online-gsplat-001/independent_checkpoint_provenance_audit.json)、[原始系统评价](../outputs/tum-online-gsplat-001/trajectory_evaluation.json)。上述 20 帧首版实验不构成完整序列验证；后续版本的范围和结果应另行报告。

### 11. 短片段结论为何需要完整序列复核

相邻 RPE 衡量局部相对变化，整段 ATE 衡量一次全局刚体对齐后的相机中心偏差。两者不会保证同时下降：每步很小的方向性偏差可以持续积累；整段对齐又可能吸收一部分共同旋转和平移。因此应同时保留 raw、ATE、RPE、终点误差和逐帧曲线，并固定起始 gauge 与评测范围。

[完整序列在线 GS 对照](../experiments/2026-10-09-full-v4-sequence-controls.md)保留 desk 全部 549 帧、xyz 全部 748 帧。同版流程中，先验权重从 0 改为 0.01 后，desk ATE 从 72.701mm 升至 83.347mm，xyz 从 16.559mm 升至 29.914mm；raw 位置误差也均升高。部分相邻 RPE 更低，所以不能只取一个局部指标宣称全局轨迹更好。两组实际 warmup 地图不同，这属于完整流程的配对观察，不能隔离出先验项的唯一因果效应。

六条完整 ICP/GS 轨迹共保留 3,891 个帧行。另有一次 521 帧时中断的 GS 尝试，作为失败记录保存，没有并入完整 748 帧轨迹。它缺少终态指标，不能填零、补齐末段或在成功重启后从尝试账本中消失。

---
id: "gslab-k07"
slug: "rgbd-depth-supervision-coverage-association"
title: "07 RGB-D 深度监督、地图覆盖与时间关联"
order: 7
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/gsplat/rendering.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/da6bbcd24c248dc884ac7f49d62e91b841b26ccc/utils/slam_helpers.py"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
---

# 07 深度把图像拟合变成带尺度的几何约束

RGB 损失约束高斯投影后的颜色，但不同几何、不透明度与相机位姿可能产生相近图像。深度监督比较沿相机光轴的深度，加入米制几何约束。这里的光轴是经过投影中心、垂直于像平面的直线；取相机 $Z$ 轴正方向朝向场景。这里的深度是相机坐标的 $Z$ 分量，即轴向深度。

深度初始化与深度损失是两件事：前者决定几何起点，后者持续限制优化过程。高斯渲染深度是由贡献权重合成的量，也不能不加条件地视为物理表面的精确距离。

## 累积深度和期望深度

对一个像素，按由近到远的顺序编号高斯。令 $\alpha_i\in[0,1]$ 为有效不透明度，$T_i$ 为前方高斯留下的透射率，$w_i$ 为合成权重：

$$
T_i=\prod_{j<i}(1-\alpha_j),
\qquad
w_i=T_i\alpha_i,
\qquad
A=\sum_i w_i=1-\prod_i(1-\alpha_i).
$$

三者均无量纲；$A$ 是累计不透明度，也用来衡量地图提供的渲染覆盖。令 $z_i$ 为第 $i$ 个高斯中心的相机轴向深度，单位为米：

$$
D_{\mathrm{acc}}=\sum_i w_i z_i,
\qquad
D_{\mathrm{exp}}=
\frac{\sum_i w_i z_i}{\max(A,\varepsilon)}.
$$

$D_{\mathrm{acc}}$ 为累积深度，$D_{\mathrm{exp}}$ 为归一化的期望深度，二者均为米；$\varepsilon>0$ 是无量纲的数值下界。仅当 $A>\varepsilon$ 时，后者才是严格按总权重归一化的加权均值。

固定 gsplat 版本支持累积深度及期望深度，两者都可与颜色一起渲染，深度置于最后一个通道。单表面深度 $z=2\,\mathrm m$、累计覆盖 $A=0.5$ 时：

$$
D_{\mathrm{acc}}=0.5\times2=1\,\mathrm m,
\qquad
D_{\mathrm{exp}}=\frac{0.5\times2}{0.5}=2\,\mathrm m.
$$

归一化不能解决空洞。当 $A$ 很小时，少数弱贡献的均值不表示已经存在有效表面。多层表面或深度边界也可能被平均成中间深度，需要同时检查覆盖、有效性与边界。

SplaTAM 渲染深度、累计不透明度和深度平方特征，再根据覆盖和残差选择像素。是否归一化应按其固定版本的目标定义判断，不能仅凭“深度”名称将它等同于另一渲染器的期望深度。

## 有效深度与地图覆盖

以下范围、掩码和权重来自本地教学建图模型，不是官方 SplaTAM 的默认目标。官方跟踪中的非数值过滤、累计不透明度阈值与求和损失，见[跟踪损失](25_splatam_tracking_loss_support.md)。

设像素位置为 $\mathbf p$，观测深度为 $D_{\mathrm{obs}}(\mathbf p)$，预测期望深度为 $D_{\mathrm{exp}}(\mathbf p)$。传感器有效集合、地图覆盖集合及其交集分别为：

$$
M_{\mathrm{sensor}}=
\left\{
\mathbf p\mid
D_{\mathrm{obs}}(\mathbf p)\in\mathbb R,\quad
0.2\,\mathrm m\le D_{\mathrm{obs}}(\mathbf p)\le5\,\mathrm m
\right\}.
$$

$$
M_{\mathrm{map}}=
\left\{
\mathbf p\mid
A(\mathbf p)>0.5,\quad
D_{\mathrm{exp}}(\mathbf p)\in\mathbb R,\quad
D_{\mathrm{exp}}(\mathbf p)>0
\right\},
\qquad
M_{\mathrm{depth}}=
M_{\mathrm{sensor}}\cap M_{\mathrm{map}}.
$$

要求深度为有限实数，排除非数值和无穷值。传感器有效并不表示地图画出了该区域；地图覆盖也不表示预测深度准确。教学模型在 $M_{\mathrm{depth}}$ 内计算深度残差，另在 $M_{\mathrm{sensor}}$ 上惩罚低覆盖。

只报告自身覆盖集合内的误差，遗漏困难区域的模型也可能得到较低分数。额外把未覆盖像素的预测设为零，并以全部传感器有效像素为分母，可以同时反映几何残差与空洞。这是一个已声明的补充指标，不是通用的唯一标准。

## 共同覆盖与含孔洞误差回答不同问题

令固定传感器有效集合为 $V=M_{\mathrm{sensor}}$，两张地图的有效覆盖集合分别为 $C_{\mathrm{off}},C_{\mathrm{on}}\subseteq V$。分别在自身覆盖内计算误差，比较的是预测值和不同像素组成；交集 $C_{\mathrm{off}}\cap C_{\mathrm{on}}$ 才对应同一批覆盖像素。

交集仍排除了空洞。若在全部 $V$ 上评价，则令未覆盖预测为零，观测深度为 $z(\mathbf p)$，覆盖处的残差为：

$$
e_{\mathrm{off}}(\mathbf p)=d_{\mathrm{off}}(\mathbf p)-z(\mathbf p),
\qquad
e_{\mathrm{on}}(\mathbf p)=d_{\mathrm{on}}(\mathbf p)-z(\mathbf p).
$$

$d,z,e$ 的单位均为米。两模型误差平方和的差可以按覆盖变化精确分解：

$$
\Delta\mathrm{SSE}_{\mathrm{common}}=
\sum_{\mathbf p\in C_{\mathrm{off}}\cap C_{\mathrm{on}}}
\left(e_{\mathrm{on}}(\mathbf p)^2-e_{\mathrm{off}}(\mathbf p)^2\right).
$$

$$
\Delta\mathrm{SSE}_{\mathrm{added}}=
\sum_{\mathbf p\in C_{\mathrm{on}}\setminus C_{\mathrm{off}}}
\left(e_{\mathrm{on}}(\mathbf p)^2-z(\mathbf p)^2\right).
$$

$$
\Delta\mathrm{SSE}_{\mathrm{lost}}=
\sum_{\mathbf p\in C_{\mathrm{off}}\setminus C_{\mathrm{on}}}
\left(z(\mathbf p)^2-e_{\mathrm{off}}(\mathbf p)^2\right).
$$

$$
\Delta\mathrm{SSE}=
\Delta\mathrm{SSE}_{\mathrm{common}}+
\Delta\mathrm{SSE}_{\mathrm{added}}+
\Delta\mathrm{SSE}_{\mathrm{lost}},
\qquad
\Delta\mathrm{MSE}=\frac{\Delta\mathrm{SSE}}{|V|}.
$$

始终未覆盖的像素在两边都产生 $z^2$，因此对差值贡献为零。SSE 与 MSE 的单位均为平方米，$|V|$ 为像素数量。各分组 SSE 可以相加，但均方根误差差值不能按组相加：

$$
\Delta\mathrm{RMSE}=
\sqrt{\frac{\mathrm{SSE}_{\mathrm{on}}}{|V|}}-
\sqrt{\frac{\mathrm{SSE}_{\mathrm{off}}}{|V|}}.
$$

应先合并相同分母下的误差平方和，再开平方。新增覆盖不保证误差下降：新预测可能比零深度孔洞离参考更远。帧数乘像素数也不是独立物理点数。

这个分解仅解释传感器参考下的指标组成。传感器深度仍有测量误差，单凭它不能判定偏差来自位姿、建图、标定还是时间差。

## 教学模型的损失与单位

[教学建图参考实现](../scripts/train_tum_gs.py)固定相机，只优化高斯地图。令 $C_{\mathbf p k},\widehat C_{\mathbf p k}\in[0,1]$ 为第 $k$ 个颜色通道，图像大小为 $H\times W$：

$$
\mathcal L_{\mathrm{RGB}}=
\frac{1}{3HW}\sum_{\mathbf p}\sum_{k=1}^{3}
\left(\widehat C_{\mathbf p k}-C_{\mathbf p k}\right)^2.
$$

$$
\mathcal L_D=
\frac{1}{|M_{\mathrm{depth}}|}
\sum_{\mathbf p\in M_{\mathrm{depth}}}
\left|D_{\mathrm{exp}}(\mathbf p)-D_{\mathrm{obs}}(\mathbf p)\right|,
\qquad
\mathcal L_A=
\frac{1}{|M_{\mathrm{sensor}}|}
\sum_{\mathbf p\in M_{\mathrm{sensor}}}(1-A(\mathbf p))^2.
$$

$\mathcal L_{\mathrm{RGB}},\mathcal L_A$ 无量纲，$\mathcal L_D$ 的单位为米；集合为空时，均值没有定义，必须单独处理。教学配置采用：

$$
\mathcal L=
\mathcal L_{\mathrm{RGB}}+\lambda_D\mathcal L_D+\lambda_A\mathcal L_A,
\qquad
\lambda_D=0.1\,\mathrm m^{-1},
\qquad
\lambda_A=0.01.
$$

当深度数值以米存储时，这对应原来的 $0.1$ 和 $0.01$ 数值权重。换成毫米而保持同一个深度数值权重，深度项会放大千倍。分辨率、求和或均值口径、跟踪与建图目标也必须分别说明，不能直接照搬。

阈值分支使用停止梯度的累计不透明度，不对像素是否进入集合求导；连续不透明度本身仍参与覆盖损失。硬阈值会使像素集合离散变化，有限差分与自动梯度比较时，还需检查集合和排序是否改变。

## 为什么关联时间会影响几何

TUM 的颜色、深度与真值各有时间戳。深度已注册到颜色视角，不表示两次观测在物理上同时采样；相机运动或动态物体会将时间差变成几何与颜色错位。

令颜色时刻为 $t_r$，深度时刻为 $t_{d,j}$，参考位姿时刻为 $t_{g,k}$。[本地关联规则](../scripts/tum_rgbd.py)选择最近候选并分别检验：

$$
j^\star=\underset{j}{\operatorname{argmin}}\,|t_{d,j}-t_r|,
\qquad
k^\star=\underset{k}{\operatorname{argmin}}\,|t_{g,k}-t_r|.
$$

$$
|t_{d,j^\star}-t_r|<0.020\,\mathrm s,
\qquad
|t_{g,k^\star}-t_r|<0.020\,\mathrm s.
$$

所有时间单位为秒；深度文件不可重复使用，并限制关联帧率。具体配对与片段最大时间差保存在原配置中。当前这条关联路径没有位姿插值、运动补偿或扫描去畸变，不能将最近邻配对解释成硬件同步。

图像缩放还需同步更新内参。设横纵缩放比例为 $s_x,s_y$，原焦距和主点为 $f_x,f_y,c_x,c_y$。采用像素中心的半像素约定时：

$$
f_x'=s_xf_x,\quad f_y'=s_yf_y,
\qquad
c_x'=s_x(c_x+0.5)-0.5,\quad
c_y'=s_y(c_y+0.5)-0.5.
$$

焦距和主点单位为像素，比例无量纲。深度用最近邻缩放，避免跨表面混出不存在的深度。深度范围、时间配对和内参更新都是优化之前的数据条件。

## 深度消融怎样控制变量

将深度残差权重设为零，只表示删除该目标项：

$$
\lambda_D=0.
$$

若深度仍用于初始化、有效像素选择或新增高斯，这应称为深度损失消融，不能称为纯 RGB 系统。比较需固定观测、相机来源、点数与覆盖口径，同时观察颜色、几何误差和覆盖；相近图像质量不表示相同三维结构。

## 换序列要重读数据契约

核对颜色与深度的注册关系、时钟、尺度、内参与畸变、帧关联和真值坐标。相同传感器的不同序列不等于跨传感器泛化；更换 Freiburg 子集等数据域时，应读取对应参数。能够解码数据且时间索引正确，只验证了输入链路。

## 覆盖充足仍可能深度错误

较大的累计不透明度只表示地图提供了渲染支持，不保证深度正确。自身覆盖、共同覆盖及全部有效深度分别回答不同问题。已知真值相机下的地图检查验证重建，不能替代未知位姿的跟踪评测。

## 常见错误

- 把累积深度当作期望深度，导致低覆盖处的预测深度偏小。
- 对深度作跨表面双线性插值，混出不存在的表面。
- 同时改变初始化、损失和点数，再把全部变化归因于深度监督。
- 只报告覆盖条件下的均方根误差而不报告覆盖率。
- 固定真值相机的建图成功后，直接声称具有定位或完整 SLAM 能力。

## 理解核对

1. 单表面深度 $2\,\mathrm m$、累计不透明度 $0.25$ 时，累积深度与期望深度分别是多少？
2. 为什么期望深度做了归一化，仍需要有效覆盖集合？
3. 同样 $21.5\,\mathrm{dB}$ 的渲染质量，为什么深度误差仍可能差一倍？
4. 当颜色与深度相差 $15\,\mathrm{ms}$ 且相机旋转时，投影与图像残差会如何混合？

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

[十二份最终地图的已知相机评测](../experiments/2026-10-09-final-map-quality.md)给出了实际例子：xyz 的三个种子均值中，自身覆盖 RMSE 从 162.573mm 降到 161.252mm，但共同覆盖 RMSE 从 157.668mm 升到 159.948mm。每个种子先汇总十帧 SSE/像素数再开平方，之后才计算三个种子均值。改变 mask 后的平均误差改善，不等于同一批像素的深度预测改善。

**RMSE 差不可按组相加。** 新覆盖多少个像素只是数量信息，新增或丢失的是哪些深度、原来的残差多大，也会改变 SSE。desk、seed 0 的十帧中，新增覆盖 12254 个、丢失覆盖 11585 个像素观测，净增加 669 个；共同、新增、丢失覆盖分别贡献 `−0.03909381、−0.41298794、+0.51824097m²` 的 ΔMSE，总和仍为 `+0.06615922m²`。这里计数是帧×像素，不是独立物理点；始终未覆盖组的差为零，也不是该区域没有误差。

报告中的[精确组成与 mask 记录](../outputs/tum-v4-coverage-sse-decomposition-001/decomposition.json)保留全部六对地图和60对帧，并核对原含孔洞 SSE；图表使用共同尺度，未按结果调整覆盖阈值。这类分解以 RGB-D 传感器观测为参考，解释指标的组成，不把传感器深度当作绝对真实表面，也不能唯一归因于位姿、建图、标定或时间差。当前结果是已知相机地图检查，不是定位或跟踪精度。

### 深度损失的实际消融

两组实验都使用同一训练帧深度初始化、同一 GT 相机与 alpha 损失；仅将深度损失权重从 0.1 改成 0。它比较的是**深度损失的作用**，不能称为纯 RGB 输入与 RGB-D 输入的比较。

| 五个留出视角 | 无深度损失 | 深度损失权重0.1 |
| --- | --- | --- |
| 平均 RGB PSNR | 21.4795 dB | 21.5193 dB |
| pooled 覆盖条件深度 RMSE | 0.173505 m | 0.089551 m |
| 有效深度像素覆盖 | 99.8898% | 99.6664% |
| 逐视角含洞深度 RMSE 均值 | 0.179943 m | 0.139819 m |

颜色质量非常接近，深度误差差异更明显。这说明图像看起来相似时几何仍可能不同。只做一段约2秒的序列、一个种子，尚不能推广到完整序列或所有场景。[完整建图与消融记录](../experiments/2026-10-09-tum-rgbd-mapping.md)。

### 换序列时先重新核对数据契约

本地另准备了 TUM `fr1_xyz`：798 张 RGB、798 张深度和 3000 条 GT，冻结 loader 在严格 20ms、深度唯一和 32Hz 约束后保留 748 帧，跨度 26.572s。全 PNG 校验与完整解码、索引存在性、时间关联和有限刚体 GT 检查均通过；这只证明数据可读及符合当前约定，不证明定位或重建精度。

官方将该序列描述为朝向基本固定、沿相机主轴平移的办公桌调试序列。它仍来自 Freiburg1 传感器与相近域，可以沿用本 helper 的 Freiburg1 RGB 内参和深度 `/5000` 米制换算；不能把换序列称为跨传感器或大场景泛化。换到 Freiburg2/3 时，应重新读取对应 K，不能把当前硬编码参数直接沿用。

[来源与校验清单](../manifests/tum-fr1-xyz-preparation.json)保存官方许可、相机约定、文件安全检查与本地计算的压缩包 SHA；本地 SHA 是精确文件标识，不是发布方校验和。[完整关联清单](../manifests/tum-fr1-xyz-association.json)保留实际 RGB、depth 与 GT 时间索引，独立最近邻实现与冻结 loader 的 748 组关联完全一致。[数据准备源码](../scripts/prepare_tum_fr1_xyz.py)没有 GPU 训练或模型参数选择。

### 高覆盖率也可能伴随较大几何误差

[完整序列终图的五视角诊断](../experiments/2026-10-09-full-seen-map-quality.md)给出了一个直接反例：xyz 权重 0.01 的有效像素覆盖率约 96.860%，略高于关闭组的 96.743%；但自身覆盖深度 RMSE 从 191.619 mm 增到 200.795 mm，共同覆盖从 179.583 mm 增到 195.768 mm，含孔洞从 411.087 mm 增到 432.725 mm。

像素被高 alpha 覆盖，只代表渲染给出了支持，不能保证深度正确。自身 mask、双方共同 mask 和全部有效传感器像素回答不同问题，应一起报告。此例的相机由 GT 提供，五帧都在训练观测中；它只说明这两份固定终图的质量，不是留出泛化或跟踪成绩。同种子 warmup 地图参数也不同，不能把差异全部当作先验权重的严格因果作用。

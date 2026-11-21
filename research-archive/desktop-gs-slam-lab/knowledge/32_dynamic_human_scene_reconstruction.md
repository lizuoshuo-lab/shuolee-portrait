---
id: "gslab-k32"
slug: "dynamic-human-scene-reconstruction"
title: "大场景中的动态人物重建：清晰外观、运动与遮挡"
order: 32
status: "street_diagnostics_active_clarity_unaccepted"
updated: "2026-10-16"
sources:
  - "https://arxiv.org/abs/2408.16760v2"
  - "https://github.com/ziyc/drivestudio/tree/e59bda4fa681f829dbb1d65f0de582b0f633c450"
  - "https://arxiv.org/abs/2308.09713"
  - "https://github.com/JonathonLuiten/Dynamic3DGaussians/tree/7dbbd4dec404308524ff402756bdb8143a2589b0"
  - "https://github.com/apple-aiml-research/ml-hugs"
---

# 大场景中的动态人物重建：清晰外观、运动与遮挡

目标是在同一个三维场景中，同时重建静态建筑和正在移动的人。人物转身、行走或摆动手臂时，轮廓、肢体位置和衣物纹理应与对应时刻的观测一致；换一个没有参与训练的相机观察，也应保持合理的外观与遮挡关系。

[大场景 SfM、分块与 LOD](30_large_scene_gs_sfm_partition_lod.md)提供坐标与规模处理。本章接着处理动态人物的表示和监督。人物清晰度是主要验收对象，整幅画面的平均分只是辅助。此前的[物理仿真与控制](31_dynamic_gaussian_simulation_world_models.md)保留为扩展专题。

## 当前可见的重建效果

![原照片与静态高斯、动态高斯、校正相机时间偏移后的动态高斯对照](../outputs/dynamic-human-actor23-independent-106/comparison/camera04-frame000613-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧613 · 原生像素，无增强。

同一组静态高斯同时拟合站立、抬脚和下蹲，容易把不同时刻的轮廓平均成重影。让局部三维结构随时间运动后，图中两个人的姿态、衣物和手持物更集中。这组对照使用同一采集地点的另一组人物，三种模型共享训练照片、初始化和更新次数；它说明动态表示怎样减轻跨时刻的监督冲突。

局部细节仍需单独检验：抬起的脚有拖影，张开的手指仍有混合与软化。第三列和第四列的区别是是否使用估计的相机时间偏移，不能仅凭这一张图判断校正对所有视角有效。灰色是检视人物的合成背景，建筑尚未与人物联合训练。[原生细节与连续回放](../experiments/37_dynamic_human_reconstruction.md#独立人物序列的静态与动态对照)保留完整条件、效果和失败案例。[连续运动表示](#时间基函数与局部运动)与[时间对应](#同一帧号不一定对应同一运动状态)解释其中的两个关键问题。

## 1. 静态背景与运动人物怎样共同表示

![OmniRe 完整方法图：场景节点、人体运动表示、统一渲染与监督](../outputs/dynamic-human-reconstruction-003/assets/omnire-figure02.png)
> 图源：Chen 等（2025），OmniRe Fig. 2 · [论文第 5 页](https://arxiv.org/pdf/2408.16760v2#page=5) · 完整原图，仅裁图外内容 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

整图左侧分别存储背景、刚体和非刚体对象，中间将当前时刻的对象变换到共同世界坐标，右侧由实际观测监督。OmniRe 对适合人体模板的行人采用 SMPL 高斯，对其他非刚体对象采用形变表示。它为街景中的背景、车辆与人物联合重建提供了直接相关的开源框架。

采用统一记号，当前时刻的场景可以写为

$$
\mathcal G(t)=\mathcal G_{\mathrm{bg}}
\cup\bigcup_{a=1}^{N_a}\mathcal T_a(t)\mathcal D_a\!\left(\overline{\mathcal G}_a,t\right),
$$

$$
\widehat{\mathbf I}_{k,t}
=\mathcal R\!\left(\mathcal G(t),\mathbf K_k,\mathbf T_{cw,k,t}\right).
$$

$\mathcal G_{\mathrm{bg}}$ 是静态背景，$\overline{\mathcal G}_a$ 是第 $a$ 个对象的参考表示，$\mathcal D_a$ 描述对象内部形变，$\mathcal T_a$ 描述对象整体位姿。人体整体向前走和手臂相对躯干摆动是两种变化，应分别建模。所有对象共同参与深度排序与不透明度合成，才能正确处理人与建筑、人与人之间的遮挡。

$\mathbf K_k$ 为第 $k$ 个相机的内参矩阵，焦距与主点坐标以像素为单位；$\mathbf T_{cw,k,t}$ 将世界坐标变换到相机坐标。$\mathcal R$ 为高斯光栅化与颜色合成，$\widehat{\mathbf I}$ 为预测图像，RGB 值归一化到 $[0,1]$。本地序列用帧索引表示 $t$，没有把未核实的采集频率换算成秒。

若用同一组静态高斯拟合不同时间的人体姿态，优化会同时受到互相冲突的位置和颜色监督。模型可能以膨胀、高透明度、重影或折中的颜色降低平均误差。分块和 LOD 解决规模问题；时间与形变模型解决这些跨时刻冲突。

## 2. 保持同一人物的几何与纹理连续

![Dynamic 3D Gaussians 作者完整展示图：人体外观、几何与持续运动轨迹](../outputs/dynamic-human-reconstruction-003/assets/dynamic3dgaussians-author-teaser.png)
> 图源：Luiten 等，Dynamic 3D Gaussians（3DV 2024）· [作者仓库完整展示图](https://github.com/JonathonLuiten/Dynamic3DGaussians/blob/7dbbd4dec404308524ff402756bdb8143a2589b0/teaser_figure.png) · 原图未改动；图像单独许可未明确。

图中彩色轨迹连接同一高斯在不同时间的位置，旁边保留重建外观与几何。第一组本地基线采用这类持续存在的动态高斯：随时间更新位置与旋转，通过局部形状和旋转约束维持相邻时刻的关系。公开代码设计了允许颜色变化的软正则，与论文中严格固定颜色的设置有区别。

用 $\boldsymbol\mu_{p,t}$ 表示第 $p$ 个高斯在时刻 $t$ 的中心。若 $q$ 是它的邻居，定义局部相对位置

$$
\mathbf e_{pq,t}=\boldsymbol\mu_{q,t}-\boldsymbol\mu_{p,t}.
$$

局部刚性约束希望相邻点的变化能近似由局部旋转解释：

$$
\mathbf e_{pq,t}\approx\mathbf R_{p,t\leftarrow t-1}\mathbf e_{pq,t-1}.
$$

这不是要求整个人像刚体一样移动。手臂、躯干和腿部可以有不同的局部运动，而同一小块衣物或皮肤附近的高斯保持相对连贯。约束过弱可能出现漂浮和撕裂；过强则可能压制真实关节运动，需通过相同数据下的消融检验。

人体模板与骨骼蒙皮进一步提供关节结构先验；无模板动态高斯则直接从多视图学习运动。当前先用同步多视角数据验证外观和运动，再在街景框架中比较人体结构先验、可见像素数量与遮挡的影响。

### 颜色连续性约束需要固定的上一时刻参照

![同一个人物在首中末三个时刻的观测、无有效颜色约束与有效颜色约束对照](../outputs/dynamic-human-color-continuity-004/assets/color-camera0-frames000-015-030.png)
> 图源：本站实测 · 相机 0，帧 0、15、30 · 相同首帧高斯与更新预算，仅改变上一帧颜色参照是否独立固定。

同一块衣物在短时间内的颜色通常连续变化。用上一时刻已求得的颜色作为固定参照，软约束可写为

$$
\mathcal L_{\mathrm{color}}
=\frac{\lambda_c}{N}\sum_{p=1}^{N}
\left\|\mathbf c_{p,t}-\overline{\mathbf c}_{p,t-1}\right\|_1.
$$

$\mathbf c_{p,t}$ 为当前优化的 RGB 颜色向量，$\overline{\mathbf c}_{p,t-1}$ 为本次优化中保持不变的上一时刻颜色，$N$ 为高斯数量，$\lambda_c$ 为约束权重。这项约束允许光照或外观缓慢变化，同时抑制模型用突变的颜色补偿几何误差。如果参照与当前颜色同步改变，两者之差始终为零，约束就失去了作用。

图中首帧两组完全一致。后续仅修正颜色参照后，一些亮斑和彩色噪点减少；全部 124 个视图的人物区域平均 PSNR 为 **28.29 → 28.61 dB**，SSIM 为 **0.885 → 0.898**。按原设置继续训练并验证后续帧 31～60，120 组留出视图的区域 PSNR 均有提升，平均 **28.89 → 29.33 dB**。这表明修正在同一场景的后续区间仍有效，尚不代表跨场景泛化。颜色连续性与几何正确性应分开验证，特别是首帧已有的外观误差不能靠上一帧约束自动消除。

## 3. 怎样判断人物真正重建清楚

![留出相机完整画面的原始观测、静态重建与动态重建对比](../outputs/dynamic-human-reconstruction-003/assets/full-scene-frame030.png)
> 图源：本站在作者公开 Panoptic 篮球数据上的训练与留出渲染 · 相机 10、帧 30 · 每列完整画面 640×360，未锐化或修复。

第一组基准有 27 个训练相机和 4 个留出相机。两组已分别完成连续 31 个时刻的重建，使用相同原始分辨率、训练相机和总更新次数。上图保留全景关系，下面使用事先固定的人物区域，按相同比例展示整个运动段。

![相同人物区域内的连续原始观测、静态重建与动态重建](../outputs/dynamic-human-reconstruction-003/assets/person-detail-comparison.gif)
> 图源：本站实测 · 留出相机 10、帧 0～30 · 同一矩形区域、原生像素；播放速度仅用于展示，非核实的原始时间。

静态表示把多个时刻的人体位置混在一起；动态表示分别恢复了接球时的手臂位置、腿部分离和躯干轮廓。在全部 124 个留出视图上，人物区域平均 PSNR 从 **20.53 提高到 28.33 dB**，SSIM 从 **0.781 提高到 0.885**。两项指标在每个视图上均提高，但这仍不是所有细节都已正确的证明。

全图损失容易被大面积背景主导。对人物区域 $\Omega_h$ 单独计算误差，可避免其贡献随人物面积缩小而被稀释。例如，按区域分别归一化的图像监督为

$$
\begin{aligned}
\mathcal L_{\mathrm{image}}={}&
\frac{\lambda_h}{|\Omega_h|}\sum_{\mathbf u\in\Omega_h}
\left\|\widehat{\mathbf I}(\mathbf u)-\mathbf I(\mathbf u)\right\|_1\\
&+\frac{\lambda_b}{|\Omega_b|}\sum_{\mathbf u\in\Omega_b}
\left\|\widehat{\mathbf I}(\mathbf u)-\mathbf I(\mathbf u)\right\|_1.
\end{aligned}
$$

$\Omega_b$ 为背景区域，$\lambda_h,\lambda_b$ 为监督权重。这是后续人物区域监督消融的表达；当前第一组基线保留作者的图像与前景分割损失，尚未加入该扩展。分割边界质量、人物采样频率和高斯密度应分别检查，不能把多项同时修改后的结果归因于其中一项。

清晰度验收包括人物区域 PSNR、SSIM 和相对于原图的梯度误差，也检查手臂、腿部、衣物边缘是否重影，以及遮挡前后是否跳变。时间差分误差用于辅助检查图像变化，不能替代三维运动对应。指标变好还需通过全部留出相机的视觉检查；只展示最好看的一帧不足以证明完成。

### 原生分辨率和多视角怎样保留衣物细节

![ENeRF 室外人物原始前景与未参与训练机位的高斯重建，保留衣服文字和四肢细节](../outputs/dynamic-human-enerf-native-025/first-frame/evaluation/heldout_view-camera00-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 数据与本站实际渲染 · 左为原图前景，右为留出机位重建 · 原生像素，无锐化。

清晰重建首先需要观测中有细节，并让不同相机约束同一个三维表面。上图来自原生 1920×1080 的室外多视角采集：15 个机位共同拟合首帧，另外 3 个机位不参与初始化或训练。重建保留了衣服文字、短裤轮廓和鞋子的主要结构。这与只让模型复现一张训练照片不同：图中的机位没有提供训练像素。

初始化采用训练机位的前景轮廓。设 $M_v$ 为第 $v$ 个机位的前景掩膜，$\pi_v(\mathbf X)$ 为三维点在该机位的投影，则理想的视觉外壳（visual hull）为

$$
\mathcal V=\bigcap_{v\in\mathcal C_{\mathrm{train}}}
\left\{\mathbf X\mid M_v\!\left(\pi_v(\mathbf X)\right)=1\right\}.
$$

每个轮廓把三维位置约束在一个投影区域内；多个机位的交集缩小可能的体积。高斯先分布在该体积的表面附近，再由图像损失优化中心、协方差、不透明度和球谐颜色。实际掩膜存在边缘误差，因此本地初始化允许少量轮廓不一致；它只是优化起点，不能恢复所有凹陷，也不是真实表面测量。

![另一名人物的条纹衣物与手持物，在留出机位中的原图和真实重建](../outputs/dynamic-human-enerf-native-025/first-frame/evaluation/heldout_view-camera12-comparison.png)
> 图源：ENeRF-Outdoor 数据与本站实际渲染 · 留出机位 12、首帧 · 左原图前景，右高斯重建。

训练保留原生像素，仅裁掉人物外围空白以减少计算；裁剪时同步平移相机主点。裁剪不会把衣物文字缩成更少的像素。上图中的灰色背景用于观察前景，原图前景按作者掩膜合成到灰色上；右图由高斯自身的不透明度合成，没有用原图补上缺失的人物。当前示例只说明人物与手持物的重建，不包含已训练的建筑背景。

首帧的作用是检查几何、颜色和相机约束是否足够。后续运动必须让同一批表面元素随时间变化，并检查未参与训练的中间时刻；首帧清楚不等于移动过程中已经保持清楚。

![同一困难时刻的原始前景、连续模型和独立单帧重建，后两者仍存在手部与物品破碎](../outputs/dynamic-human-enerf-native-025/comparison/camera06-frame059-capacity-comparison.png)
> 图源：ENeRF-Outdoor 与本站真实渲染 · 留出机位 06、帧 59 · 原图、连续模型、独立单帧拟合；同一区域、无增强。

运动中的细节还需要单独检验。上图把同一困难时刻重新作为静态多视角问题训练，手和物品仍没有完整恢复。这说明不能只把失败解释为上一帧误差的累积。应继续检查运动区域的跨相机对应、观测时间、遮挡以及表面优化；该对照尚不能判定唯一原因。连续模型的完整画面还存在人物范围外的漂浮伪影，因此检查范围必须包含全图，不能只保留人物紧框。


### 人物轮廓与背景变化怎样区分

背景减除得到的前景透明度不等于人体分割。人遮住墙面是一种可见性变化；人的影子落在墙上，是同一静态表面的照明变化。两者都可能使当前图像偏离空场景参照，但只有前者需要在墙前放置人体几何。手持物也属于运动前景，不能仅凭是否落在人形范围内就删除。

![原图中的投影阴影与前景透明度中的墙面残留](../outputs/dynamic-human-enerf-matte-background-034/camera08-frame000750-wall-matte-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 原图、空场景参照、作者前景透明度及灰底合成；同一原生像素区域。

从左到右看，原图中人的影子真实地改变了墙面亮度；空场景参照没有这片阴影；作者提供的透明度却还保留了一块墙面纹理，灰底合成后出现人物之外的半透明残片。这类残片不能直接解释成衣物或肢体。

在像素已经几何对齐、颜色以线性 RGB 表示的理想合成模型中，令 $\mathbf C_t(\mathbf u)$ 为运动前景颜色，$\alpha_t(\mathbf u)\in[0,1]$ 为覆盖比例，$\mathbf B_t(\mathbf u)$ 为当前光照下的背景颜色，$\mathbf B_0(\mathbf u)$ 为空场景参照，则

$$
\mathbf I_t-\mathbf B_0
=\alpha_t(\mathbf C_t-\mathbf B_0)
+(1-\alpha_t)(\mathbf B_t-\mathbf B_0).
$$

右侧第二项在没有人体覆盖的位置仍可能非零：阴影、照明变化和曝光差异都能贡献背景残差。若背景图还存在像素位移 $\Delta\mathbf u$，其额外一阶误差近似为 $\nabla\mathbf B_0\,\Delta\mathbf u$；其中图像梯度单位为颜色值每像素，位移单位为像素。墙缝和衣物边缘的梯度较大，几像素错位就会产生明显条纹。这个近似只描述局部小位移，不能用墙面的二维变换代替人体三维几何。

![原始训练照片、作者提供的前景透明度与实际训练目标](../outputs/dynamic-human-enerf-matte-diagnosis-080/camera07-frame000726-training-target.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap · 机位07、训练帧726 · 实际输入对照，不是重建图。

同一问题也出现在另一运动片段中。中列白色代表被当作前景的区域：绿色玩偶上方的墙面阴影、手臂下方的墙面和腿间背景被错误保留，右列训练目标因而包含墙面纹理。这直接证明了监督中的背景污染；下方的相同预算训练对照进一步检验了部分修正能否改善最终重建。按此前景掩码计算的分数也会受到这些像素影响，不能称为精确的人体语义区域分数。[实际训练目标与张手细节](../experiments/37_dynamic_human_reconstruction.md#训练掩码中的墙面阴影与张手细节)给出输入证据。

前景更干净，并不自动意味着手指更清楚。在相同运动模型、初始化和训练预算下，只去掉部分训练图中确认的背景误前景，确实减弱了一些墙面残片，但下图的张手仍然混有半透明重影，玩偶边缘也没有稳定恢复。掩码主要决定哪些像素参与前景颜色与轮廓监督；它不会直接补充手指关节的运动或被遮挡表面的对应关系。这个有限对照支持继续处理局部运动与可见性，不能说明正确掩码不重要。

![原照片与仅改变训练前景支持后的手部重建对照](../outputs/dynamic-human-enerf-support-correction-081/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、原训练支持、保守修正支持 · 原生像素。

这次修正没有替换原开发基线。训练输入中不确定的遮挡人物、手持物与脚部受到保护，仍有背景残留；[相同预算的实际对照](../experiments/37_dynamic_human_reconstruction.md#前景支持修正与手部清晰度)说明了它的有效范围。

因此，人物训练需要同时检查轮廓监督、原图纹理、相机投影和动态表面的对应关系。提高不透明度只能遮住背景残差，不能保证新生成的是正确人体。实际输入中的阴影和错位示例见[背景与轮廓诊断](../experiments/37_dynamic_human_reconstruction.md#人物透明度中的墙面残留与静态投影检查)。

前景估计还会犯相反的错误：把真实的人体像素当作背景删除。若真实表面点 $\mathbf X$ 在某个可见机位的投影满足 $M_v(\pi_v(\mathbf X))=0$，严格轮廓交集就有 $\mathbf X\notin\mathcal V$。其他机位看见这块表面，也不能在“所有机位必须通过”的规则下保住它。这里描述的是初始化的集合约束；后续优化是否能重新恢复这部分几何，仍取决于表示、梯度和剩余观测。

![原照片中可见的前臂、作者透明度造成的缺失与输入分割候选](../outputs/dynamic-human-enerf-mask-quality-038/camera01-full-person-sam2-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap，SAM 2.1 分割候选 · 三列均由原照片与掩码合成，属于输入诊断，不是三维渲染。

先看左侧原照片，黑衣人物的前臂和手实际可见；中间透明度合成却抹去了一部分皮肤；右侧分割候选保留了它。若源图先乘此前景透明度，表面颜色观测也会随之受损。因此，检查清晰度应同时对照原照片、输入掩码和真实渲染。分割网络的高置信度不代表掩码正确，人物相互遮挡、手持物和衣物内部都要逐一检查。相同预算的训练对照显示部分前臂与轮廓有所恢复，但手指、衣物与遮挡仍未清晰；见[输入掩码与训练效果的区别](../experiments/37_dynamic_human_reconstruction.md#前景掩码漏掉可见人体后的输入对照)。

### 多视角颜色融合需要区分前景支持与表面可见性

![原照片、原颜色融合与排除前景外源颜色后的真实渲染](../outputs/dynamic-human-4k4d-source-selection-043/camera06-frame000750-source_foreground_gate-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 同一几何、投影和透明度，机位 06 · 原生像素，无图像增强。

左列保留原始观测；中列使用原颜色融合；右列排除投影落在源图前景掩码外的颜色。右肩部分黑边减轻，但手、衣物和相互遮挡仍未恢复清晰。这个例子区分了“取色有效”与“表面正确”两个条件。

在图像驱动的颜色模型中，设三维点为 $\mathbf X$，源相机编号为 $s$，投影为 $\pi_s$，源图颜色为 $\mathbf C_s$，观察方向为 $\mathbf d$。一类模型用归一化权重融合源颜色，再叠加与观察方向有关的残差：

$$
\widehat{\mathbf c}(\mathbf X,\mathbf d)
=\sum_{s=1}^{S}w_s(\mathbf X)\,\mathbf C_s\!\left(\pi_s(\mathbf X)\right)
+\mathbf r(\mathbf X,\mathbf d),
\qquad w_s\geq0,\quad\sum_{s=1}^{S}w_s=1.
$$

$S$ 是参与融合的源机位数量，$w_s$ 无量纲，颜色采用归一化 RGB；$\mathbf r$ 可以由球谐函数等方向模型表示。这是用于解释图像驱动取色的形式，不能当作传统 3DGS 直接优化球谐系数的同一流程。

若源图已经乘以前景掩码，掩码外的黑色代表无效观测，不能直接当作物体表面很暗。令 $M_s$ 为源图前景支持，定义

$$
h_s(\mathbf X)=\mathbf 1\!\left[M_s\!\left(\pi_s(\mathbf X)\right)\geq\tau\right],
\qquad
\widetilde w_s(\mathbf X)=
\frac{w_s(\mathbf X)h_s(\mathbf X)}{\sum_j w_j(\mathbf X)h_j(\mathbf X)}.
$$

只有分母大于零时才使用右式。本地固定模型对照取 $\tau=0.5$，没有有效源机位时保留原融合，未擅自删除三维点。筛选依据是掩码，而不是颜色亮度，因此真实黑衣服仍可参与取色。

前景支持还不是遮挡可见性。背后的点投影到前方人的衣服上，仍满足 $h_s=1$，但源相机实际看见的是另一个表面。正确取色还依赖几何、深度排序、相机对应与拍摄时刻。简单减少源机位或只选择最大权重，可能让局部花纹更强，也可能把错误表面的颜色直接带进来；强烈的纹理不等于正确的重建。固定模型检查与同起点的成对训练均只改善了部分边缘颜色，尚未恢复手指和衣物细节。完整条件见[颜色融合对照](../experiments/37_dynamic_human_reconstruction.md#前景支持筛选与源视角选择)。

### 纹理清楚与运动表面正确需要分别检验

![原照片、点式图像驱动渲染与直接优化三维高斯的同一机位对照](../outputs/dynamic-human-enerf-representation-048/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 原生像素，无图像增强。

三列依次为原照片、基于 4K4D 点表示的本地单帧分支、直接优化各向异性三维高斯的单帧分支。右列的衣服褶皱、头发与腿部轮廓更清楚，但手臂出现拉长，交叠人物的肩部仍有错误结构。因此，纹理更锐利不能替代对肢体位置与遮挡关系的检查。这是两套完整训练流程的比较，不能把差异单独归因于高斯形状。

设相机 $s$ 的内参和世界到相机变换分别为 $\mathbf K_s$、$\mathbf T_s$，场景参数为 $\Theta$，渲染算子为 $\mathcal R$，观测为 $\mathbf I_s(t_s)$。单帧多视角优化可以写成

$$
\Theta^*=\underset{\Theta}{\operatorname{arg\,min}}\;
\sum_{s\in\mathcal S_{\mathrm{train}}}
\mathcal L\!\left(\mathcal R(\Theta;\mathbf K_s,\mathbf T_s),\mathbf I_s(t_s)\right).
$$

这个目标隐含了共同场景状态的要求。即使图像文件的编号相同，不同观测也可能包含不同运动状态；标定误差、遮挡、分割缺失和错误表面同样可能使观测难以由一套几何共同解释。模型可能用伸长的高斯、厚表面或混合颜色降低训练误差。仅凭渲染中的重影，不能确定其中哪一种原因成立。

因此，需要同时查看原生分辨率下的训练视角与未参与训练的相机：前者检验是否学到纹理，后者检验这种纹理能否依附在合理的空间表面上。对运动一致性的判断还应跟踪同一纹理点，并在另一时刻复核相机之间的对应，不能为每个候选时间偏移重新挑选一组更容易匹配的点。这里的单帧比较尚不代表连续动态人物或大场景联合重建已经完成。


### 表面表示与表面一致性约束

![原照片、仅图像拟合的二维高斯与增加表面约束后的完整人物对照](../outputs/dynamic-human-enerf-surface-gs-055/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 原生像素，无增强。

二维高斯泼溅（2D Gaussian Splatting，2DGS）用位于三维空间的有向面元表示表面。这里的“二维”指每个高斯的局部支撑平面，不是把场景降为一张平面图像。设面元中心为 $\boldsymbol\mu$，正交单位切向量为 $\mathbf t_u,\mathbf t_v$，两个正尺度为 $s_u,s_v$，则面元位置与局部高斯权重写成

$$
\mathbf X(u,v)=\boldsymbol\mu+s_u u\,\mathbf t_u+s_v v\,\mathbf t_v,
\qquad G(u,v)=\exp\!\left[-\frac{u^2+v^2}{2}\right].
$$

$u,v$ 无量纲；中心与尺度采用场景坐标单位，单位法线为 $\mathbf n=\mathbf t_u\times\mathbf t_v$。像素的视线与面元求交后，可以计算局部权重并按深度合成颜色。它使局部表面的方向明确，但仍需要损失函数约束面元之间的关系。

深度畸变约束鼓励同一像素的主要颜色贡献集中在较窄的深度范围；法线一致性约束鼓励面元法线与深度图所导出的表面法线一致。在本站采用的实现变体中，令 $w_i$ 为透明度合成权重，$z_i$ 为按深度排列的面元深度，$\mathcal P$ 为腐蚀后的有效前景像素集，则深度项为

$$
\mathcal L_{\mathrm{dist}}=
\frac{1}{|\mathcal P|}\sum_{p\in\mathcal P}
2\sum_{i>j}w_i(p)w_j(p)\bigl(z_i-z_j\bigr).
$$

排序成立时，它等于双重求和形式的加权绝对深度差；量纲为场景长度单位。这是所用渲染器的 $L_1$ 变体，应与原论文的平方深度差设计区分。令 $\bar{\mathbf N}(p)=\sum_iw_i(p)\mathbf n_i$ 为透明度加权的渲染法线，$A(p)=\sum_iw_i(p)$ 为累计不透明度，$\mathbf N_D(p)$ 为在相同坐标系中由深度图求得的单位法线。本地法线项为

$$
\mathcal L_{\mathrm n}=
\frac{1}{|\mathcal P|}\sum_{p\in\mathcal P}
\left[1-A_{\mathrm{sg}}(p)\,\bar{\mathbf N}(p)^{\mathsf T}\mathbf N_D(p)\right].
$$

下标 $\mathrm{sg}$ 表示在该权重处停止梯度传播。$\bar{\mathbf N}$ 没有重新归一化，因此这一项不能简单等同于两条单位向量的余弦损失。实验采用深度项权重 $0.01$、法线项权重 $0.05$，数值依赖场景尺度和损失归一化方式，不能作为通用参数直接照搬。

上图中，右列的部分前臂条纹和衣服起伏减轻，但交叉手臂的位置、被遮挡的肩部及手指仍不正确。表面正则项提供的是几何偏好，并没有新增人体观测或正确骨架。应把“表面更平滑”和“人体结构恢复正确”分开判断。方法参考 [2DGS](https://surfsplatting.github.io/)；本地成对条件及效果见[表面约束对照](../experiments/37_dynamic_human_reconstruction.md#二维高斯表面与几何正则项)。


训练拟合与跨视角恢复也可能分离。同一批观测上的更长训练与更大面元数量，使训练照片的拟合明显改善，三个开发留出机位却都变差，手臂仍有条纹。这类现象提示当前模型仍在用视角相关的外观和局部几何解释不一致观测；它不能确定唯一误差来源，但已不支持继续单纯增加训练预算。见[更长训练对照](../experiments/37_dynamic_human_reconstruction.md#更长训练与持续增密)。

### 掩码中的未知区域与几何观测覆盖

![原照片、完整掩码监督与忽略掩码冲突区域的重建对照](../outputs/dynamic-human-enerf-mask-uncertainty-053/paired-comparison/camera08-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 训练诊断机位 08 · 原生像素，无增强。

前景掩码漏掉一段前臂时，把该处当成背景会错误地惩罚真实人体；补全掩码若把墙面也包含进来，又会让模型学习背景纹理。上图右列在一部分手臂附近减少了墙面残留，但手臂拉伸并未消失。掩码处理解决的是监督像素的可靠性，不能直接保证三维表面正确。

设 $H(p)$ 是补全后的二值前景掩码，$S(p)$ 是另一分割模型的二值前景结果，$V(p)$ 表示图像去畸变后是否有效。两者冲突的区域可以暂作未知观测，定义损失权重

$$
\omega(p)=V(p)\begin{cases}
0,&H(p)=1\ \land\ S(p)=0,\\
1,&\text{其他情况}.
\end{cases}
$$

未知区域不会被强制标成背景。这种做法也会移除一部分真实皮肤和衣服的监督，因此需要其他相机提供支持。对逐像素颜色项，可用 $\omega(p)$ 加权；SSIM 则依赖局部窗口，直接把无效像素置零还会改变邻域统计，必须说明采用的掩码方式。这里的本地对照沿用既有掩码图像损失，不能解释为所有监督误差均被排除。

几何检查也有覆盖范围。三台以上相机中的纹理匹配可以三角化为空间点，再与渲染深度比较；但鞋带和玩具上的匹配很准确，并不说明缺少纹理的前臂和手指也正确。检查时既要看重投影误差，也要看观测点落在身体哪些位置。对应的掩码与几何检查见[实验记录](../experiments/37_dynamic_human_reconstruction.md#掩码冲突与稀疏表面检查)。


#### 掩码怎样改变颜色监督

![原照片、原训练目标、局部修正目标与改变的像素](../outputs/dynamic-human-enerf-forearm-matte-repair-072/candidates/camera01-frame000750-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap，SAM2.1局部分割及本站核验 · 机位01、帧750 · 输入诊断，非重建结果。

前景掩码既影响轮廓，也可能直接改变颜色监督。令 $M(\mathbf p)\in[0,1]$ 为训练前景透明度，$\mathbf I(\mathbf p)$ 为原照片，$\mathbf b$ 为随机背景色，$\mathbf C_\Theta$ 为已经按可见性和不透明度加权的高斯前景颜色，$A_\Theta$ 为累计不透明度。当前训练的合成图与目标分别为

$$
\widehat{\mathbf I}_{\mathbf b}=\mathbf C_\Theta+(1-A_\Theta)\mathbf b,
\qquad
\mathbf I^*_{\mathbf b}=M\mathbf I+(1-M)\mathbf b.
$$

所有颜色采用归一化RGB，$M$ 和 $A_\Theta$ 无量纲。真实皮肤若被误标为 $M=0$，目标颜色就变成背景色，同时轮廓损失还会要求该处透明。此时仅增加高斯或运动节点，仍在优化错误的目标。修正只应针对原照片确实可见的表面；分割预测不能用来推断被遮挡的手或衣服。

本地对照只修正一个训练机位三张偶数帧中的固定前臂区域，保留其余照片和原有模型。原照片、原掩码、初始化与评估输入均单独保留。候选经逐帧看图接受，仍属于额外弱标签。下图比较实际重新训练后的原生渲染。

![原照片、原掩码目标、基线渲染与局部修正后的真实渲染](../outputs/dynamic-human-enerf-forearm-matte-repair-072/raw-forearm-evaluation/camera01-frame000751-raw-comparison.png)
> 图源：同一公开数据与本站实际训练 · 留出的奇数帧751 · 四列原生像素，无增强。

机位01的可见前臂内部颜色更接近原照片，四个预定皮肤内部区域合并PSNR提高0.861dB，三处改善、一处略退步；这些内部像素在基线中已接近完全不透明。完整人物与局部图仍显示手指边界、抬起的鞋和细衣纹偏软，新机位06胸前的浅色条纹及遮挡手部拖影仍在。修正三张训练matte没有带来明显的整体清晰度改善。

对照原照片的误差只在事先选定、明确可见的皮肤内部计算；它不会把原掩码误删的皮肤排除，也不评价手指、轮廓和隐藏表面。全体机位继续保留原指标，避免更换评估口径后把两次成绩直接混比。[训练条件与完整结果](../experiments/37_dynamic_human_reconstruction.md#三张训练图的局部掩码修正)见实验附录。


颜色监督还取决于局部区域的权重。令 $\Omega_d$ 为手部、鞋部附近的有效像素窗口并集，保留原有目标 $\mathcal L_0$，则额外的局部颜色项可写为

$$
\mathcal L=\mathcal L_0+
\frac{\lambda_d}{3|\Omega_d|}
\sum_{\mathbf p\in\Omega_d}
\left\|\widehat{\mathbf I}_{\mathbf b}(\mathbf p)-\mathbf I^*_{\mathbf b}(\mathbf p)\right\|_1.
$$

$\lambda_d$ 是无量纲权重；分母的 $3$ 对应RGB三个通道。空窗口的额外项为零。窗口可以包含衣物、背景或另一个人物，只是采样范围，并非手部的语义真值。这个目标增加小区域对优化的影响，却没有增加新的视角或正确的表面对应；它也可能改变增密过程。

![原照片、原目标训练与增加手鞋局部颜色权重后的真实重建](../outputs/dynamic-human-enerf-detail-weight-076/expanded-detail/camera04-frame000751-dark_shoe.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 原生像素，无增强。

本地采用 $\lambda_d=0.2$ 的对照，在70张训练图中增加局部项，其余35张保留原损失。上图的抬脚仍有拖影，鞋带没有恢复到原照片的清晰程度；扩大后的手部对照也仍有指尖粘连。原机位未训练时刻的平均误差略降，另外三个机位总体退化，因此不能把增加局部权重等同于细节恢复。[局部颜色权重与跨视角清晰度](../experiments/37_dynamic_human_reconstruction.md#局部颜色权重与跨视角清晰度)保留了完整条件与结果。

### 增密梯度与细节恢复

![原照片、标准梯度增密与绝对梯度增密的完整人物对照](../outputs/dynamic-human-enerf-absgrad-052/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站运行 · 机位 06 · 原生像素，无增强。

三列依次是原照片、标准三维高斯增密、采用 AbsGS 绝对梯度统计的增密。两种训练中的衣服主体都比早期点式结果清楚，但交叉手臂仍然被拉长，手指没有形成正确边界。这里改变了增密决策，未改变相机、图像、掩码与训练预算；不能将增密方式的改变直接等同于人体表面恢复。

增密依赖图像损失对高斯投影中心的梯度。记第 $i$ 个高斯的二维中心为 $\boldsymbol\mu_i^{\mathrm{2D}}$，像素 $p$ 对该中心的反向传播贡献为 $\mathbf g_{ip}$。标准梯度首先求和；若不同像素要求中心向相反方向移动，贡献可能抵消：

$$
\mathbf g_i=\sum_{p\in\mathcal P_i}\mathbf g_{ip},
\qquad s_i=\left\|\mathbf g_i\right\|_2.
$$

其中 $\mathcal P_i$ 是该高斯参与渲染的像素集合。AbsGS 的核心是先对每个贡献的两个坐标分量分别取绝对值，再累积：

$$
\mathbf a_i=
\begin{bmatrix}
\displaystyle\sum_{p\in\mathcal P_i}|g_{ip,x}|\\[3pt]
\displaystyle\sum_{p\in\mathcal P_i}|g_{ip,y}|
\end{bmatrix},
\qquad s_i^{\mathrm{abs}}=\left\|\mathbf a_i\right\|_2.
$$

这种统计能保留相反方向的细节需求。实现中还会按屏幕坐标尺度归一化，并在多次可见观测间累计，超过阈值的高斯再依据空间尺寸复制或分裂。阈值随损失归一化与梯度统计方式变化，不能把两种统计的相同数值视为相同的增密强度。本地采用 gsplat 文档的示例设置，标准统计阈值为 $2\times10^{-4}$，绝对梯度阈值为 $8\times10^{-4}$；这组设置最终产生了更少的高斯，不能笼统理解成“绝对梯度总会增加点数”。

增密解决的是表示分辨率的分配。相机观测之间的运动不一致、掩码中的背景残留、错误深度与遮挡关系，仍需要各自的约束。评价时应将原照片与完整人物、新视角、手臂和提物区域一起查看，不能只凭衣服花纹更锐利作出结论。方法来源见 [AbsGS](https://arxiv.org/abs/2404.10484)，本地条件与真实效果见[增密对照](../experiments/37_dynamic_human_reconstruction.md#绝对梯度增密与移动肢体)。

### 同一帧号不一定对应同一运动状态

![同一运动时刻的原始前景、同编号多视角拟合与按相机调整帧号后的拟合](../outputs/dynamic-human-enerf-time-control-027/comparison/camera06-person2-upper-body.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 机位 06，第 49 帧 · 左原图，中同编号输入，右调整训练机位帧号；原生像素，无锐化。

手指和被握住的物品必须在各个相机中对应同一个运动状态。如果一台相机里的手已经抬高，另一台里的手仍停在较低位置，却让它们监督同一个静态表面，高斯优化就会收到不一致的几何约束。上图右侧的手与玩偶比中间列完整；两组使用完全相同的初始化和更新预算，只改变训练机位选取的图像帧号。

以相机 $r$ 的帧索引 $n$ 为参照，若相机 $v$ 的对应图像索引为 $n+\eta_v$，共同运动状态的监督可写为

$$
\widehat{\mathbf I}_{v,n}
=\mathcal R\!\left(\mathcal G(n),\mathbf K_v,\mathbf T_{cw,v}\right),
\qquad
\widehat{\mathbf I}_{v,n}\ \longleftrightarrow\ \mathbf I_{v,n+\eta_v},
\qquad \eta_r=0.
$$

$\eta_v$ 以帧为单位；这里的静止相机外参不随时间变化。正偏移表示要从该相机取较大的图像编号，才能对应参照相机的当前状态。没有经过核实的采集时间戳时，这只是图像索引对应关系，不能直接解释为以毫秒计的硬件时钟误差。

对于同一表面点，两个相机的齐次像素坐标 $\widetilde{\mathbf u}_i$、$\widetilde{\mathbf u}_j$ 应满足对极约束

$$
\widetilde{\mathbf u}_j(n+\eta_j)^{\mathsf T}
\mathbf F_{ji}\,
\widetilde{\mathbf u}_i(n+\eta_i)\approx0.
$$

$\mathbf F_{ji}$ 是由两个相机的内外参确定的基础矩阵。检查时间偏移时，应沿着同一条图像特征轨迹比较不同候选时刻；如果每个候选都重新挑选一组更容易匹配的点，较小残差可能只是样本更换造成的。

时间偏移是否能被估计，还取决于运动方向。设 $d$ 为点到对极线的有符号距离，$\mathbf n$ 为该线在图像中的单位法向量，$\mathbf v$ 为图像运动速度。小时间变化下有

$$
\Delta d\approx\mathbf n^{\mathsf T}\mathbf v\,\Delta\eta.
$$

$d$ 的单位是像素，$\mathbf v$ 的单位是像素／帧。当点几乎沿对极线运动时，$\mathbf n^{\mathsf T}\mathbf v$ 很小，换几个帧号也难以改变几何残差。因此，人物运动幅度大并不保证时间偏移可观测；需要选择具有足够法向运动的片段，并在另一段运动中验证。

本地训练机位的检查中，独立时刻的平均成对几何残差由约 **1.31 降到 0.58 像素**。但固定同一组留出照片比较时，三个机位的前景 PSNR 有升有降；留出相机自身的相对时间尚未验证。这项对照支持继续检查观测一致性，不能证明一组固定偏移适用于整段视频，也不能替代移动人物的连续清晰度验收。

还需要检查相机之间是否通过有效对应连接起来。若两组相机之间没有可靠的运动点对应，两组各自的内部时间关系可以被估计，但它们之间的相对时间仍无法确定。另一段数据的实际检查就出现了这种情况，因此全序列固定偏移没有被采用。小残差、优化结束和时间标定成功是三个不同的判断。


人物对应还能提供另一种检查：如果同一组相机存在时间偏移，从一个人物估计的偏移应当有助于解释另一个人物的运动。本地连续片段采用一组时刻拟合、另一组时刻检查，跨人物转移的偏移降低了两个人在预留时刻的关节点误差；但大残差仍在，这项关键点检查本身没有衡量原生 RGB 清晰度，后续真实渲染见下节。这比只看同一批拟合残差更有说服力，也仍不能直接等同于时钟标定成功。见[连续人体与跨人物检查](../experiments/37_dynamic_human_reconstruction.md#连续人体关键点中的跨相机时间偏移)。



### 连续人体运动如何带动高斯形状

![原照片、相同帧编号运动模型与考虑拍摄时间偏移后的连续人体重建](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 机位 04、第 755 帧，预留时刻 · 三列均保留原生像素，无增强。

高斯随人体运动时，中心位置与空间形状必须使用一致的变换。设一个规范姿态高斯的中心为 $\boldsymbol\mu_0$，协方差为 $\boldsymbol\Sigma_0$。在局部仿射近似 $\mathbf x_t=\mathbf A_t\mathbf x_0+\mathbf b_t$ 下，有

$$
\boldsymbol\mu_t=\mathbf A_t\boldsymbol\mu_0+\mathbf b_t,
\qquad
\boldsymbol\Sigma_t=\mathbf A_t\boldsymbol\Sigma_0\mathbf A_t^{\mathsf T}.
$$

$\mathbf A_t\in\mathbb R^{3\times3}$ 表示局部线性变换，无量纲；$\mathbf b_t\in\mathbb R^3$ 为平移，单位为场景长度；协方差单位为长度的平方。前式移动中心，后式同时改变椭球的方向与尺度。只移动中心而保持原协方差，会使细长高斯在手臂转动后仍沿旧方向铺开。以上是高斯在仿射变换下的标准关系；用人体表面附近的局部变换近似衣物运动，是额外的建模假设。

本地实现以一个人体表面顶点作为每个高斯的运动参照。设该顶点在规范姿态和当前姿态的位置为 $\mathbf v_0,\mathbf v_t$，蒙皮产生的局部线性变换为 $\mathbf M_0,\mathbf M_t$，则在 $\mathbf M_0$ 可逆时取

$$
\mathbf A_t=\mathbf M_t\mathbf M_0^{-1},
\qquad
\mathbf b_t=\mathbf v_t-\mathbf A_t\mathbf v_0.
$$

因此，参照顶点必定从 $\mathbf v_0$ 运动到 $\mathbf v_t$；高斯允许相对顶点保留可学习的偏移，以覆盖衣物和头发。该关系没有保证宽松衣物、手指或手持物遵循同一运动，距离人体较远的玩偶尤其不能仅靠“绑定最近顶点”获得正确形变。

上图比较两种连续运动假设：相同文件帧编号代表相同时刻，以及每台相机带有估计的时间偏移。两组训练照片、初始高斯和更新预算相同。右列部分衣物纹理与抬脚轮廓更集中，但脸、手和玩偶仍模糊。90 张预留时刻图像的平均前景 PSNR 从 22.60 提高到 24.13 dB，79 张改善，仍有 11 张下降。这支持继续修正运动对应，不能当作清晰人体已经完成。

![同一预留时刻的脸部原生细节，时间偏移后仍有重影](../outputs/dynamic-human-enerf-articulated-control-064/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：同上 · 原照片、相同帧编号模型、考虑拍摄时间的模型 · 固定头部区域。

脸部例子说明，时间模型与姿态模型需要分别检验。自由三维关节点轨迹拟合得好，转换为受骨骼结构限制的人体模型后，不一定保持同样的投影精度。此次模型转换使身体与脚部的预留观测误差中位数从 2.64 增至 5.94 像素；头部局部旋转还曾被固定。这些证据指向直接修正图像中的人体姿态，而不是继续增加高斯数量。完整条件、负例及连续效果见[连续人体重建对照](../experiments/37_dynamic_human_reconstruction.md#连续人体运动与拍摄时间的实际重建对照)。



### 姿态投影更准确之后，还要检查真实纹理

![同一原生头部区域，原照片、旧姿态重建与直接多视角姿态修正后的重建](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 机位 04、第 755 帧，预留时刻。

将人体模型直接投影到各个相机，可以在图像平面优化姿态。设 $\boldsymbol\theta(t)$ 为随时间变化的关节旋转，$\boldsymbol\beta$ 为固定体形，$\mathbf J_j(\boldsymbol\theta(t),\boldsymbol\beta)$ 为经过尺度与全局平移变换后的第 $j$ 个世界坐标关节点，$\mathbf y_{v,t,j}$ 为图像中的预测位置，则一类拟合目标为

$$
\mathcal L_{\mathrm{pose}}=
\frac{1}{\sum_{v,t,j}w_{v,t,j}}
\sum_{v,t,j}w_{v,t,j}\,
\rho_{\delta}\!\left[
\pi_v\!\left(\mathbf J_j(\boldsymbol\theta(t),\boldsymbol\beta)\right)-\mathbf y_{v,t,j}
\right]
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
$$

$\pi_v$ 包含相机内外参；$w_{v,t,j}$ 表示置信度及可用观测权重；残差与鲁棒损失的尺度参数 $\delta$ 以像素计。关节旋转采用弧度，三维位置采用场景长度单位。正则项约束姿态变化与时间曲线，避免少量错误关键点任意扭曲身体。本地目标采用逐坐标的平滑鲁棒损失，尺度为 3 像素；只让训练时刻参与优化，另用中间时刻检查。

上图中的头部重影明显减少，支持检验头部运动自由度的作用。不过这一组合修正同时重新拟合了身体姿态，不能把改善只归因于头部解锁。为区分两者，下面进一步固定身体运动，单独拟合头颈。

![同一次修正中的全人物画面，头部更清楚而手持物仍有明显拉花](../outputs/dynamic-human-enerf-direct-pose-065/paired-comparison/camera04-frame000755-full-person.png)
> 图源：同上 · 原照片、旧姿态重建、直接姿态修正后的重建 · 原生像素，无增强。

全图揭示了另一面：衣服花纹、手指与玩偶没有一同恢复，部分区域反而退化。六个固定头部区域的平均 PSNR 提高 4.26 dB，90 张预留时刻的全前景平均 PSNR 却下降 0.63 dB。因此，关键点拟合更准、局部图更清楚和整体重建更好，是三个需要分别验证的结论。这一组合修正没有替换旧身体运动基线；随后单独检验头颈运动，以保留脸部改善并减少其他区域退化。见[直接姿态修正的完整效果与负例](../experiments/37_dynamic_human_reconstruction.md#直接多视角姿态修正与头部运动)。



### 将头部运动与衣物、手持物运动分开

![只改变头颈运动后的全人物对照](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera04-frame000755-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 原照片、原身体运动、仅头颈修正 · 原生像素。

人体不同部位需要不同的运动自由度。先保持身体关节、世界平移、体形、相机与时间偏移不变，只更新颈部和头部的局部旋转，能够检查原来的脸部重影是否与这两处运动约束有关。本地对照将两个关节的轴角向量写为连续时间三次多项式：

$$
\boldsymbol\omega_j(\tau)=\sum_{k=0}^{3}\mathbf a_{j,k}\tau^k,
\qquad \tau=\frac{t-750}{6},\qquad j\in\{\mathrm{neck},\mathrm{head}\}.
$$

$t$ 表示带相机时间偏移的连续帧索引，不是秒；$\tau$ 无量纲。轴角向量的方向表示旋转轴，长度表示以弧度计的旋转角；系数 $\mathbf a_{j,k}$ 亦以弧度计。两个人、两个关节、四组系数、每项三个分量，共优化 48 个标量。其余身体骨架坐标在核对中保持不变；SMPL 的姿态修正项仍会使部分非头部表面产生小幅变化，所以不能说整个非头部网格完全不动。

![头颈隔离对照中的脸部细节](../outputs/dynamic-human-enerf-head-motion-066/paired-comparison/camera04-frame000755-dark_head.png)
> 图源：同上 · 与上方全图对应的脸部区域。

单独修正头颈后，眼镜、头发和脸部轮廓集中起来。六个固定头部区域的平均 PSNR 提高约 3.98 dB；全前景平均提高约 0.16 dB，90 张开发验证图中有 62 张改善。与上一种全身联合修正相比，这个结果支持保留头颈运动调整，但仍不能宣称手指和衣物细节已经恢复。

上方完整画面中的玩偶仍有横向拖影，因为它的运动并不由人体骨架唯一决定；衣服也可能相对身体滑动和弯曲。接下来的表示需要在骨架提供的大体运动之外加入局部变形，并约束相邻区域的运动一致性。只有在相同原图、训练预算和固定观察区域下真正恢复纹理，才能说明这层自由度有用。[完整对照与保留的负例](../experiments/37_dynamic_human_reconstruction.md#只优化头颈的隔离对照)。



### 骨架之外的局部变形如何进入渲染

![相同原生画面的骨架运动与局部变形重建对照](../outputs/dynamic-human-enerf-residual-graph-067/paired-comparison/camera04-frame000745-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 原照片、头颈修正基线、增加局部变形。

骨架适合描述身体的大体姿态，衣物、手指和手持物还可能相对骨架运动。变形图（deformation graph）用一组邻近控制节点描述这部分变化。本地扩展先应用已有骨架运动，再为每个高斯关联四个控制节点；节点权重随该高斯继承，训练中的复制与分裂也保留这种关联。这是本站的研究对照实现，并非某篇论文官方实现的复现结论。

设骨架变换后的高斯中心与协方差为 $\mathbf x_i$、$\mathbf C_i$，节点位置为 $\mathbf g_k$，节点的残余旋转和平移为 $\mathbf R_k(t)$、$\mathbf d_k(t)$，固定权重满足 $w_{ik}\geq0$、$\sum_k w_{ik}=1$。进一步变形后的中心为

$$
\widetilde{\mathbf x}_i(t)=\sum_{k\in\mathcal N_i}w_{ik}
\left[\mathbf R_k(t)(\mathbf x_i-\mathbf g_k)+\mathbf g_k+\mathbf d_k(t)\right].
$$

中心和位移使用同一场景长度单位；旋转矩阵与权重无量纲。对于权重固定的单个高斯，这个变换的线性部分及其协方差传播为

$$
\mathbf L_i(t)=\sum_{k\in\mathcal N_i}w_{ik}\mathbf R_k(t),
\qquad
\widetilde{\mathbf C}_i(t)=\mathbf L_i(t)\mathbf C_i\mathbf L_i(t)^{\mathsf T}.
$$

因此，局部运动既改变中心，也改变投影前的形状与方向。球谐颜色的观察方向也要逆变换回规范坐标，才能保持外观与表面运动的对应。节点旋转采用有界轴角参数，避免加权线性变换接近奇异。

为使表示在参考时刻保持一致，本地模型的残余旋转向量和平移均使用无常数项的三次多项式：

$$
\boldsymbol\omega_k(\tau)=\sum_{q=1}^{3}\mathbf a_{kq}\tau^q,
\qquad
\mathbf d_k(\tau)=\sum_{q=1}^{3}\mathbf b_{kq}\tau^q,
\qquad \tau=\frac{t-750}{6}.
$$

$t$ 是连续帧索引，$\mathbf a_{kq}$ 以弧度计，$\mathbf b_{kq}$ 使用场景长度单位。在 $t=750$ 时，残余旋转为单位矩阵、平移为零。相邻节点另受局部刚性约束，减少无规律扭曲；这类约束不会自动赋予模型真实手指关节或独立物体结构。

以 $s$ 表示初始高斯包围盒对角线长度的一半，三个坐标分量分别满足 $|a_{kq,r}|\leq 0.1$ 弧度、$|b_{kq,r}|\leq 0.03s$。相邻节点集合记为 $\mathcal E$，边权为 $\eta_{k\ell}>0$，归一化的局部刚性项为

$$
\mathcal L_{\mathrm{rig}}=
\frac{\sum_{(k,\ell)\in\mathcal E}\eta_{k\ell}
\left\|\mathbf R_k(\mathbf g_\ell-\mathbf g_k)+\mathbf g_k+\mathbf d_k-\mathbf g_\ell-\mathbf d_\ell\right\|_2^2}
{s^2\sum_{(k,\ell)\in\mathcal E}\eta_{k\ell}}.
$$

它要求相邻节点的局部旋转和平移尽量相容，而非要求整个人只有一个刚体运动。该项无量纲，权重过大会压制真实形变，过小则可能产生不连续表面；当前权重属于一次固定开发对照，并未完成最优参数搜索。

实际相同更新预算对照中，90张开发时间验证图的前景PSNR均值从 24.30 变为 26.29 dB，89张改善。四个固定机位的玩偶拖影明显减轻，衣服图案与部分前臂边缘更集中，脸部改善也得以保留；手指、鞋带和交叉遮挡处仍不够清晰。 具体条件、细节与负例见[局部变形实验](../experiments/37_dynamic_human_reconstruction.md#骨架之后的局部变形对照)。这些图已用于开发，仍需要新的时间片段和相机检验泛化。




### 时间基函数与局部运动

![原照片、三次多项式时间模型与增加局部时间基函数后的完整人物](../outputs/dynamic-human-enerf-temporal-basis-078/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap与本站实测 · 机位04、帧751 · 原生图像，无增强。

变形图决定空间上哪些高斯共同运动，时间基函数决定这种运动可以怎样变化。一个全局三次多项式用同一组系数描述整个片段；调整某一段运动时，也可能影响其他时刻。具有局部支撑的三次B样条可以增加局部时间变化的自由度。下式是本站的研究扩展，保留原有空间节点、身体运动与颜色损失。

令 $\tau=(t-t_0)/\Delta t$，$t$ 为连续帧索引，$t_0$ 为参考时刻，$\Delta t$ 为时间归一化尺度，二者以帧为单位；本地对照采用 $\Delta t=6$。对每个节点的旋转向量和平移，在原多项式上增加

$$
\begin{aligned}
\boldsymbol\omega_k(\tau)&=\sum_{q=1}^{3}\mathbf a_{kq}\tau^q+\sum_{j=1}^{7}\mathbf u_{kj}\psi_j(\tau),\\
\mathbf d_k(\tau)&=\sum_{q=1}^{3}\mathbf b_{kq}\tau^q+\sum_{j=1}^{7}\mathbf v_{kj}\psi_j(\tau),\\
\psi_j(\tau)&=B_3\!\left(\frac{\tau-c_j}{h}\right)-B_3\!\left(\frac{-c_j}{h}\right).
\end{aligned}
$$

其中 $h=0.5$，$c_j\in\{-1.5,-1,-0.5,0,0.5,1,1.5\}$，均无量纲。减去参考值使 $\psi_j(0)=0$，所以参考时刻的残余变换仍为恒等变换。这里的局部支撑指原始B样条；减去参考常数后，$\psi_j$ 本身不再具有相同的紧支撑。

$$
B_3(x)=
\begin{cases}
\dfrac{2}{3}-x^2+\dfrac{|x|^3}{2},& |x|<1,\\
\dfrac{(2-|x|)^3}{6},&1\leq |x|<2,\\
0,& |x|\geq2.
\end{cases}
$$

该基函数及其前两阶导数连续。新增旋转系数 $\mathbf u_{kj}$ 的各分量限制在 $\pm0.03$ 弧度，平移系数 $\mathbf v_{kj}$ 的各分量限制在 $\pm0.01s$，其中 $s$ 为前文定义的场景尺度。新增系数从零开始，初始渲染与原模型一致；训练后仍需通过相同的旋转矩阵与协方差传播计算图像。

在两个互不重叠的运动片段中，相同训练预算下的局部时间模型均降低了平均图像误差；这支持保留该表示继续研究，但范围仍是同一场景。下面的张手位置说明了平均误差的局限：衣物主体已经成形，手掌和玩偶边缘仍出现混合的半透明结构。

![张开的手掌与手持物仍有重影](../outputs/dynamic-human-enerf-second-interval-079/comparison/camera04-frame000731-both_hands_and_objects.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、全局三次时间模型、局部时间模型。

时间基函数增加的是已有运动场的表达能力，没有直接增加独立手指关节、正确的表面对应或新出现的可见表面。此例中，后续训练帧的原照片与监督目标仍保留了清楚的手掌，因此还需要检查局部几何、姿态和可见性，不能把失败全部归于输入分辨率。[另一运动片段的真实对照](../experiments/37_dynamic_human_reconstruction.md#另一运动片段中的时间表示)保留了条件、退化机位与手鞋细节。

### 运动控制节点的密度与观测支持

![同一画面的原照片、384节点基线与1536节点运动图重建](../outputs/dynamic-human-enerf-fine-graph-071/paired-comparison/camera04-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际训练 · 机位04、帧751 · 原生像素，无增强。

增加变形图节点，可以让相邻小块表面拥有更独立的运动。但控制节点不等于高斯：前者决定局部运动场的自由度，后者决定可渲染几何和颜色的采样。若每个高斯仍由四个邻近节点控制，增加节点会缩短空间支持范围，也会改变上式局部刚性约束中的邻接距离。因此，相同的正则项系数并不代表相同的空间平滑作用。

本地同预算对照将节点从384增至1536，仍使用原生照片、固定身体运动和相同15000次更新。部分衣服图案与前臂边缘略有改善，但手指、抬起的鞋和细衣纹仍偏软。已知机位留出时间的平均前景PSNR提高0.160dB，另外三个机位平均下降0.022dB；更多运动自由度尚未转化为稳定的新视角细节。

![遮挡机位中仍存在的胸前条纹](../outputs/dynamic-human-enerf-fine-graph-071/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同一公开数据与本站训练 · 机位06、帧745 · 原照片、384节点基线、1536节点对照。

遮挡处的浅色条纹并未因增加节点而消失。运动表示只能调整已有监督下的几何与外观，不能自动补回被掩码删除的观测，也不能把遮挡人物的颜色变成后方表面的颜色。节点密度、观测支持和表面可见性需要分别验证。此次保留384节点模型作为开发基线，[完整条件与负例](../experiments/37_dynamic_human_reconstruction.md#运动控制节点密度的同预算对照)保存在实验附录。

### 留出视角中的时间对应与外观验证

![未参与本轮外观训练的机位：原照片、头颈运动基线与局部变形](../outputs/dynamic-human-enerf-heldout-cameras-068/paired-comparison/camera12-frame000751-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 机位12、帧751。

留出相机用于检查学到的三维表示能否从其他方向渲染。动态场景还需要明确时间：不同相机的相同帧编号未必对应完全相同的人体姿态。设相机 $k$ 的连续时间索引为

$$
t_k=n+\delta_k,
$$

其中 $n$ 是图像文件的整数帧索引，$\delta_k$ 是相对参考相机的时间偏移，以帧为单位。这里把图像索引映射到共同时间；前文的 $\eta_k$ 则从共同时间选取图像索引。使用同一参考相机时，二者满足 $\delta_k=-\eta_k$，不能直接混用符号。固定已学到的关节轨迹 $\mathbf X_j(t)$ 后，可以只用校准帧的二维关节预测 $\mathbf u_{k,n,j}$ 估计偏移：

$$
\widehat\delta_k=\underset{|\delta_k|\leq 2}{\arg\min}
\sum_{n\in\mathcal C}\sum_{j\in\mathcal J}\sum_{d\in\{u,v\}}
\rho_s\!\left(\sqrt{q_{k,n,j}}\,[\pi_k(\mathbf X_j(n+\delta_k))-\mathbf u_{k,n,j}]_d\right)
+\rho_s(\beta\delta_k).
$$

$\pi_k$ 是使用给定相机内外参的透视投影，$q_{k,n,j}$ 为截断到1的预测置信度，$d$ 分别取两个图像坐标分量。残差以像素计，稳健损失为 $\rho_s(r)=s^2(\sqrt{1+(r/s)^2}-1)$，本地取 $s=3$ 像素、$\beta=0.1$ 像素/帧。本地设置只使用身体与足部预测，并在拟合前按置信度和与固定轨迹的相容性筛选。随后冻结偏移与所有模型参数，在不属于 $\mathcal C$ 的照片上检查外观。原照片不进行时间插值，只有三维运动在连续时刻求值。

实际对照使用3个未参与本轮高斯训练、初始化和姿态拟合的机位：21张偶数帧只校准时间，18张奇数帧评价外观。局部变形模型的平均前景PSNR为23.76dB，较头颈基线提高0.87dB；图中玩偶斑纹和轮廓更清楚，但手指和鞋部仍偏软。这些相机在更早的开发中使用过，因此属于开发验证，不能作为从未接触的最终测试。二维关节预测也会受遮挡干扰，估计偏移不能替代真实同步标定。

### 不透明度、人物归属与遮挡要分别检查

![固定模型的逐人物渲染：原照片、联合结果、黑衣人物层、白衣人物层](../outputs/dynamic-human-enerf-actor-layers-069/camera06-frame000745-actors.png)
> 图源：同一公开数据与本站实际渲染 · 机位06、帧745 · 单独人物层仅改变参与渲染的高斯集合。

两个人共同渲染时，前方的人会遮住后方人物层的错误。上图将同一个冻结模型分别渲染：黑衣人物层胸前存在缺口和条纹，白衣人物层边缘存在颜色残留。联合画面上的低误差不能证明每个人物已经独立、完整地重建。单独显示的遮挡表面没有这个机位的无遮挡真值，不能据此断言其真实形状。

对某个像素，将全部人物高斯按深度排序。第 $i$ 个高斯的像素不透明度为 $\alpha_i$，到达它之前的透射率为

$$
T_i=\prod_{j<i}(1-\alpha_j).
$$

人物 $a$ 在联合渲染中的颜色与不透明度贡献为

$$
\mathbf C_a=\sum_{i\in\mathcal G_a}T_i\alpha_i\mathbf c_i,
\qquad W_a=\sum_{i\in\mathcal G_a}T_i\alpha_i.
$$

这里的 $T_i$ 包含所有人物的遮挡。若另把人物 $a$ 单独渲染，其他人物被移除，透射率也随之改变，所以单独渲染的不透明度与 $W_a$ 不相同。完整前景满足

$$
\mathbf C=\sum_a\mathbf C_a,\qquad A=\sum_a W_a,
\qquad \mathbf I=\mathbf C+(1-A)\mathbf b,
$$

$\mathbf b$ 是背景颜色。$W_a$ 和 $A$ 无量纲；它们说明当前模型怎样生成图像，不是真实人物分割。

![联合不透明度、逐人物可见贡献与单独人物的期望深度比较](../outputs/dynamic-human-enerf-actor-layers-069/camera06-frame000745-layers.png)
> 图源：本站同一冻结模型的原生分辨率输出 · 橙色为黑衣人物层，蓝色为白衣人物层。

左图白色表示不透明度接近1；中图展示每个人物在联合排序后实际留下的贡献；右图仅在两个单独人物层均有较强不透明度的位置比较期望深度，颜色表示较近的人物层，深度差以场景长度单位计，显示上限为0.25。期望深度是高斯中心深度的加权平均，并非真实表面。

九组画面的前景内部像素中，仅约2.39%的联合不透明度低于0.9；较大颜色误差像素中，约81.27%仍有至少0.9的不透明度。因此，当前残差不能单靠整体提高不透明度解释或修复。接下来需要用训练视角检查人物归属、可见表面支持与局部运动约束，再比较对应训练改动。详细条件和失败画面保存在[留出相机与人物层诊断](../experiments/37_dynamic_human_reconstruction.md#留出相机与人物层诊断)。


### 人物归属监督怎样作用于联合渲染

![训练照片与保守人物内部标签，橙色表示黑衣人物、蓝色表示白衣人物](../outputs/dynamic-human-enerf-actor-supervision-070/masks/camera07-labels-review.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；SAM 2.1分割与本站标签筛选 · 原生训练机位07、帧750。

外观损失只约束合成后的颜色。如果某个人物的高斯替邻近人物解释了部分像素，总颜色仍可能接近照片。人物归属监督直接约束联合渲染中每个人物的可见贡献，而不是仅检查高斯在三维空间中离哪个人体更近。

沿用上节的人物不透明度贡献 $W_a(\mathbf p)$ 与总不透明度 $A(\mathbf p)$，定义归一化贡献

$$
P_a(\mathbf p)=\frac{W_a(\mathbf p)+\varepsilon}{A(\mathbf p)+2\varepsilon},
\qquad \varepsilon=10^{-6}.
$$

两个人物的贡献之和等于总不透明度。$P_a$ 是当前渲染模型归一化后的贡献，不是真实分割概率。设 $\mathcal M_a$ 为保守筛选后属于人物 $a$ 的训练像素集合，按人物分别平均的损失为

$$
\mathcal L_{\mathrm{inst}}
=-\frac12\sum_{a=0}^{1}\frac{1}{|\mathcal M_a|}
\sum_{\mathbf p\in\mathcal M_a}\log P_a(\mathbf p).
$$

分别平均可避免被遮挡人物仅有少量像素时，其监督完全被前景人物覆盖。本地对照在既有颜色、透明度与局部变形损失之外，增加权重0.05的该项；只有经过看图接受的13个训练机位的第750帧使用人物标签，其他照片保持原监督。分割内部腐蚀7×7像素，另一人物及手持物分割膨胀7×7后排除，并与原前景matte大于0.95的有效像素取交集。不确定位置被忽略，不当作背景，也不据此填补被遮挡表面。

相同105张训练照片、初始高斯、视图抽样与15000次更新下，原训练机位的90张留出时间图前景PSNR平均变化+0.035dB，另外三个机位的18张图变化+0.033dB。固定全身图和局部图中，两版外观多数近似；手指、抬起的鞋和细衣纹仍偏软。机位06第745帧胸前仍有浅色竖向条纹，第755帧遮挡手部仍有拖影。单独渲染黑衣人物时胸前缺口仍存在，白衣人物层仍带有重叠区域残留。监督改善了部分训练像素的人物归属，但未恢复明显的精细结构。 这些画面已用于开发验证，本轮没有完成大场景或清晰人体的最终验收。

![同一原生画面的基线与人物归属监督对照](../outputs/dynamic-human-enerf-actor-supervision-070/heldout/paired-comparison/camera06-frame000745-full-person.png)
> 图源：同一公开数据与本站实际训练 · 原照片、局部变形基线、增加人物归属监督。

分割是额外的弱监督，不能当作实例真值；整体误差、人物层归属和肉眼可辨的细节必须分别检查。完整协议、负例与检查范围见[人物归属训练对照](../experiments/37_dynamic_human_reconstruction.md#人物归属监督的同预算训练对照)。

### 连续运动中的同预算对照

![相同时间窗口中原始观测、多时刻静态表示、同编号动态表示与局部取帧修正后的动态表示](../outputs/dynamic-human-enerf-local-time-029/comparison/camera00-frame046-person1.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实际渲染 · 机位 00，共同索引 46 · 四列为原始前景、静态、动态、调整取帧后的动态；原生像素，无增强。

动态表示和时间对应解决不同的约束冲突。上图第二列用同一组静态高斯解释多个时刻，手和物品被拟合成折中的位置；后两列允许高斯随时间移动，因此能跟随抬手动作。第四列进一步让各机位尽量观测同一运动状态，但它不能自动补足表面覆盖或保证手指几何正确。

这组对照保留原生 1920×1080 像素，用 15 个机位训练连续 25 个时刻，另外 3 个机位提供相同的 75 张外观评测照片。所有机位都在更早、分离的图像上参与局部时间对应检查，因此这里是**外观训练留出**，不是全局未见相机；本组也没有留出时间。静态与调整后的动态表示使用相同观测和 20,400 次更新，前景 PSNR 均值为 **18.20 与 25.24 dB**，75 张图均改善。相同更新次数不代表相同计算量，动态模型在首帧后固定了尺度和不透明度。

![运动后段的手与玩偶仍有破碎，保留真实失败外观](../outputs/dynamic-human-enerf-local-time-029/comparison/camera06-frame058-person2.png)
> 图源：ENeRF-Outdoor 与本站实际渲染 · 机位 06，共同索引 58 · 四列含义同上，未删除残差。

仅比较两种动态输入，局部取帧修正把平均前景 PSNR 从 **24.88 提高到 25.24 dB**，其中 54 张改善、21 张退化。上图右侧玩偶的部分孔洞缩小，但手指、袖口和物品表面仍有破碎。由此能确认动态表示改善了运动外观，也能确认单靠整数帧对应不足以解决细节问题。下一步需要分别检验局部表面的覆盖、尺度、不透明度与运动约束；不能把平均分提高当作清晰度验收。

连续对照：[机位 00](../outputs/dynamic-human-enerf-local-time-029/comparison/camera00-continuous-comparison.mp4)、[机位 06](../outputs/dynamic-human-enerf-local-time-029/comparison/camera06-continuous-comparison.mp4)、[机位 12](../outputs/dynamic-human-enerf-local-time-029/comparison/camera12-continuous-comparison.mp4)。视频中的灰色是前景检视背景，尚未联合训练真实场景背景。条件与核验见[实验附录对应记录](../experiments/37_dynamic_human_reconstruction.md#局部时间对应与连续人物的同预算对照)。

### 尺度与不透明度能改善覆盖，但不能替代运动对应

![移动手部在原始观测、固定尺度与不透明度、同时允许两者变化时的真实对照](../outputs/dynamic-human-enerf-support-030/comparison/camera00-frame046-person1-compact.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站渲染 · 机位 00，索引 46 · 原始前景、固定参数、两项均可变；原生裁图，无增强。

高斯的尺度控制它在三维空间中覆盖多大的范围，不透明度控制它参与颜色合成的强弱。设第 $i$ 个高斯在时刻 $t$ 的三个主轴标准差为 $s_{i,a,t}>0$，主轴单位向量为 $\mathbf r_{i,a,t}$，其中 $a\in\{1,2,3\}$。协方差可以写为

$$
\boldsymbol\Sigma_{i,t}
=\sum_{a=1}^{3}s_{i,a,t}^{2}\,
\mathbf r_{i,a,t}\mathbf r_{i,a,t}^{\mathsf T}.
$$

标准差与场景坐标具有相同长度单位；本数据尚未独立核验其物理尺度。即使固定这三个标准差，中心位置与主轴方向变化仍会改变相机中的投影椭圆，因此“尺度固定”不等于“屏幕覆盖固定”。允许尺度再变化，会给表面伸缩和局部覆盖增加自由度，但也可能使不正确位置上的高斯变大而掩盖几何误差。

记高斯固有不透明度为 $o_{i,t}\in(0,1)$，投影中心为 $\mathbf m_{i,t}$，有效二维协方差为 $\boldsymbol\Sigma'_{i,t}$。在忽略实现中的截断与上限时，像素 $\mathbf u$ 处的有效不透明度为

$$
\alpha_{i,t}(\mathbf u)
=o_{i,t}\exp\!\left[
-\frac12(\mathbf u-\mathbf m_{i,t})^{\mathsf T}
\bigl(\boldsymbol\Sigma'_{i,t}\bigr)^{-1}
(\mathbf u-\mathbf m_{i,t})
\right].
$$

$o_{i,t}$ 与 $\alpha_{i,t}$ 都无量纲，但前者是一个高斯的属性，后者还依赖像素位置。降低 $o_{i,t}$ 可能减少错误表面的颜色贡献，也可能产生透明缺口；提高它则可能填补覆盖，也可能遮挡正确表面。因此应同时检查颜色、轮廓、边缘与完整画面中的残留，而不是只看一个不透明度数值。

本站在同一连续片段上做了四组受约束对照：保持两者固定、只开放尺度、只开放不透明度、同时开放。为限制突变，尺度的三个分量分别约束在初始值的 $[0.5,2]$ 倍内，并惩罚相邻状态的对数尺度差；不透明度则在对数几率空间限制初始偏移。令 $\ell_{i,t}=\ln\!\bigl(o_{i,t}/(1-o_{i,t})\bigr)$，本次使用

$$
\begin{aligned}
\frac12s_{i,a,0}&\le s_{i,a,t}\le2s_{i,a,0},
&|\ell_{i,t}-\ell_{i,0}|&\le2,\\
\mathcal L_s&=\frac{0.01}{3N}\sum_{i=1}^{N}\sum_{a=1}^{3}
\left[\ln\!\left(\frac{s_{i,a,t}}{s_{i,a,t-1}}\right)\right]^2,
&\mathcal L_o&=\frac{0.01}{N}\sum_{i=1}^{N}(o_{i,t}-o_{i,t-1})^2.
\end{aligned}
$$

$N$ 是固定的高斯数量，$t=0$ 表示本片段共同起点。两个正则项只在对应参数开放时加入；它们是本站验证的约束设定，不是论文报告的最优参数。四组共享初始状态、输入、迭代数与运动正则，不增删高斯。

这组开放参数使平均前景 PSNR 从 **25.24 提高到 25.38 dB**；75 张外观留出照片中，63 张改善、9 张退化，3 张共同初始图不变。上图的手指与物品仍有混合和拉丝，改善尚不足以满足清晰重建要求。它支持的结论是“这一组受约束参数调整只有有限帮助”，不能推出所有尺度或不透明度建模都无效，也不能把小幅均值变化当作统计显著性。

![训练机位中仍然出现的手指与物品拉丝](../outputs/dynamic-human-enerf-support-030/training-comparison/camera07-frame058-person1-compact.png)
> 图源：ENeRF-Outdoor 与本站渲染 · 训练机位 07，索引 58 · 三列含义同上。

训练机位也存在相同类型的缺陷，说明问题不能只归结为未见视角。此前独立拟合困难单帧也未消除手部破碎，因而需要继续检查运动部位的跨相机对应、初始化与表面覆盖，而不能仅用“时间累积误差”解释。此处仍是灰色背景上的人物与手持物检视，并未完成真实背景和人物的联合场景。完整对照和不利样例见[实验记录](../experiments/37_dynamic_human_reconstruction.md#运动阶段的尺度与不透明度对照)。

### 单帧拟合不等于跨时刻重建

![固定近处训练画面的拟合恢复衣褶，但仅代表该观测的重现](../outputs/dynamic-human-street-local-capacity-020/paired-audit/frame129-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站训练视图诊断 · 帧 129、相机 4 · 同倍率，保留匿名化，未锐化；原始数据 CC BY-NC-SA 4.0。

图中后两列的袖口、衣褶和提袋比共同起点清楚。这说明当前高斯表示能够表达这些已观测细节，却不能证明它恢复了真实衣物表面。单张图像同时受到几何、遮挡、不透明度和颜色影响，多种不同的三维参数都可能产生相似的投影。

设静态背景为 $\mathcal B$，人物参考姿态中的高斯集合为 $\mathcal G$，人体运动参数为 $\boldsymbol\theta_t$。形变算子 $\mathcal D$ 将参考人物变换到时刻 $t$，相机 $k$ 的渲染为

$$
\widehat{\mathbf I}_{t,k}
=\mathcal R\!\left(
\mathcal B\cup\mathcal D(\mathcal G,\boldsymbol\theta_t);
\mathbf K_k,\mathbf T_{cw,k}(t)
\right).
$$

$\mathbf K_k$ 是相机内参，$\mathbf T_{cw,k}(t)$ 将世界坐标变换到该时刻的相机坐标，$\mathcal R$ 包含投影、遮挡排序与颜色合成。跨时刻重建要求同一组 $\mathcal G$ 在不同运动参数和相机下共同解释观测。例如，固定姿态后，表面拟合可写为

$$
\mathcal G^{\star}
=\underset{\mathcal G}{\arg\min}\;
\frac{1}{|\mathcal S|}
\sum_{(t,k)\in\mathcal S}
\frac{1}{|\Omega_{t,k}|}
\sum_{\mathbf u\in\Omega_{t,k}}
\rho\!\left(
\widehat{\mathbf I}_{t,k}(\mathbf u)-\mathbf I_{t,k}(\mathbf u)
\right)
+\lambda\mathcal P(\mathcal G).
$$

$\mathcal S$ 是参与优化的时刻与相机集合，$\Omega_{t,k}$ 是人物监督区域，$\rho$ 为图像残差函数，$\mathcal P$ 为表面变化先验，$\lambda$ 为其权重。上式用于说明共享约束；具体实现还可采用结构相似性等图像损失。当 $\mathcal S$ 只有一张图时，拟合可以利用该视角下的颜色与遮挡关系补偿不正确几何。扩大到两个时刻也未必消除这些多解。

![两张相邻训练画面拟合之后，中间留出时刻仍有破碎衣物和提袋伪影](../outputs/dynamic-human-street-local-capacity-020/paired-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站留出渲染 · 帧 130、相机 4 · 本帧未参与两组局部拟合，未锐化。

上图中的第 130 帧位于两个被拟合时刻之间，仍出现破碎外观。对运动参数做插值，不会自动使表面位置、可见性和颜色贡献正确地随时间变化。诊断时应依次检查：固定观测是否能拟合清楚、同一表面是否能解释多个时刻、未参与优化的中间时刻是否仍然正确。局部拟合成功而跨时刻失败时，不能只归因于高斯数量不足；也不能在没有进一步分解的情况下，将全部错误归因于姿态插值。

### 不透明度覆盖与清晰外观是两项不同的检查

![第一帧的原始观测、未修改渲染、颜色范围诊断与前景不透明度贡献](../outputs/dynamic-human-first-frame-diagnosis-007/assets/first-frame-appearance-camera15-grid.png)
> 图源：本站实测 · 留出相机 15、第一帧 · 几何与不透明度保持相同；左下仅作颜色范围诊断，右下为前景贡献的灰度显示。

上图右下的人体内部接近白色，但右上的衣物仍有明显斑块。这两种现象可以同时出现。按从近到远的顺序排列像素处的高斯，令 $s_p\in\{0,1\}$ 表示该高斯是否属于前景，有

$$
\begin{aligned}
T_p(\mathbf u)&=\prod_{q<p}\left(1-\alpha_q(\mathbf u)\right),\\
\widehat{\mathbf I}(\mathbf u)&=\sum_p T_p(\mathbf u)\alpha_p(\mathbf u)\mathbf c_p,\\
S_h(\mathbf u)&=\sum_p T_p(\mathbf u)\alpha_p(\mathbf u)s_p.
\end{aligned}
$$

$\alpha_p$ 为该高斯在像素处的有效不透明度，$T_p$ 为其前方累积透射率，两者均无量纲。$\mathbf c_p$ 是颜色，$S_h$ 则是前景对最终不透明度的贡献。这里的前景沿用数据提供的人物与被操控物体标签；例如篮球也可能属于前景。黑色背景下，完整颜色合成由上式给出。

$S_h$ 接近 1 只能说明该像素主要被前景解释，并不能证明贡献来自正确深度处的衣物表面，也不能保证这些高斯在不同相机中组成一致纹理。几何位置、排序或颜色不一致，都可能在覆盖完整时留下斑块。因此，轮廓监督、深度与表面一致性、颜色拟合需要分别检查。

在街景中，还要区分人物对最终画面的贡献与“把人物删掉以后”的画面。记第 $a$ 个人物的高斯集合为 $\mathcal G_a$，保留所有对象的深度排序和不透明度，其线性颜色贡献为

$$
\mathbf C_a(\mathbf u)=\sum_{p\in\mathcal G_a}T_p(\mathbf u)\alpha_p(\mathbf u)\mathbf c_p.
$$

这里的 $T_p$ 仍包含其他人物和建筑的遮挡。若真的移除该人物，后方高斯的透射率则变为

$$
\widetilde T_p(\mathbf u)=\prod_{\substack{q<p\\ q\notin\mathcal G_a}}\left(1-\alpha_q(\mathbf u)\right).
$$

因此，移除人物后的图像通常不等于原图减去 $\mathbf C_a$：原先被挡住的表面会显露出来。前者用于检查场景中是否还残留人物形状，后者用于定位当前像素由哪些高斯解释。它们都是模型诊断，不能代替真实人体分割。颜色贡献的相加关系成立于线性合成阶段；如果显示前有颜色截断，须先相加，再按同一顺序截断，不能直接相加已经截断的展示图片。

这个例子中，依据原图选定的衣物内部区域，$S_h$ 最低约为 0.98；保持所有几何参数不变，仅限制每个高斯的颜色范围，外观仍几乎不变。由此可排除该区域“大量透明孔洞”的简单解释，也说明直接截断颜色不足以解决现有斑块。该检查帮助确定下一步诊断方向，不是一次训练改善。

### 空间颜色一致性不能代替表面一致性

![固定同一组几何时，原始观测、原模型、仅图像拟合与空间颜色约束的真实对照](../outputs/dynamic-human-appearance-refinement-008/assets/camera15-person.png)
> 图源：本站训练对照 · 相机 15、第一帧 · 两支各更新 2,000 次，仅优化颜色；四幅均保留原生像素，未锐化或修复。

图中左下表明，在固定几何上继续降低训练图像误差，仍可能让新视角出现更多彩色碎片。不同高斯的颜色可以互相补偿，而这种补偿未必在另一个视角成立。右下加入相邻高斯颜色一致性后，新增的彩色噪点受到抑制，但原有衣物的大块斑驳仍在。

为了单独检验颜色拟合，可以固定位置、旋转、尺度、不透明度与相机颜色校正，只对前景高斯的 RGB 参数增加空间正则。下式是本站诊断用的扩展，并非 Dynamic 3D Gaussians 原论文损失：

$$
\begin{aligned}
\mathcal L_{\mathrm{spatial}}
={}&\frac{\lambda_s}{3N_h}
\sum_{p\in\mathcal H}\sum_{q\in\mathcal N_p}w_{pq}\\
&\quad\times\sum_{k=1}^{3}\left[\sqrt{(c_{p,k}-c_{q,k})^2+\varepsilon^2}-\varepsilon\right],\\
&\sum_{q\in\mathcal N_p}w_{pq}=1.
\end{aligned}
$$

$\mathcal H$ 是前景高斯集合，$N_h$ 是其数量；$\mathcal N_p$ 为同属前景的 8 个空间近邻。$w_{pq}$ 由固定中心之间的距离确定，距离较近的邻居获得较大权重，并逐点归一化。$c_{p,k}$ 为第 $k$ 个颜色通道，$\varepsilon=0.001$ 平滑绝对值在零点的不可导处，$\lambda_s=0.01$ 为本次固定权重。它约束三维表示中的颜色差异，不是对输出图片做模糊滤波；但真实纹理边界也可能受到平滑影响。

比较时必须同时保留“额外训练”的对照和原模型。空间正则相对同等更新次数的纯图像拟合改善了平均指标，但相对原模型，四个留出视角平均 PSNR 仅增加 0.02 dB，SSIM 下降约 0.003，衣物仍不清楚。合理结论是它能抑制这次额外颜色拟合的恶化，尚未修复原始重建。接下来仍需检查表面位置、可见性与人体结构先验，不能把正则项变小视为几何正确。

## 4. 从人物基准进入街景

![OmniRe 完整人体姿态优化图，展示初始姿态与优化后姿态的对齐](../outputs/dynamic-human-reconstruction-003/assets/omnire-figure06.png)
> 图源：Chen 等（2025），OmniRe Fig. 6 · [论文第 10 页](https://arxiv.org/pdf/2408.16760v2#page=10) · 完整原图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

每一对图中的 Noisy 和 Refined 分别是初始人体姿态与联合优化后的姿态。彩色人体轮廓应与真实行人对齐。若姿态偏移，图像残差可能驱使高斯拉长或膨胀以补偿错误的位置；同时优化关节姿态和高斯属性，可以让运动与外观共同解释观测。这是 OmniRe 的论文结果，尚非本地街景实测。

SMPL 高斯用线性混合蒙皮（linear blend skinning）把参考姿态中的点带到当前人体姿态。对高斯中心，可用齐次坐标写出这一关系：

$$
\begin{bmatrix}\boldsymbol\mu_{p,t}\\
1\end{bmatrix}
=\mathbf T_a(t)\sum_{j=1}^{J}w_{pj}\mathbf B_j(\boldsymbol\theta_t)
\begin{bmatrix}\overline{\boldsymbol\mu}_p\\
1\end{bmatrix},
\qquad \sum_{j=1}^{J}w_{pj}=1.
$$

$\overline{\boldsymbol\mu}_p$ 为参考姿态下的高斯中心；$\boldsymbol\theta_t$ 为关节姿态参数；$\mathbf B_j$ 为包含参考姿态校正的第 $j$ 个关节变换；$w_{pj}$ 为该关节对高斯的蒙皮权重；$J$ 为关节数。$\mathbf T_a$ 再将人体局部坐标变换到世界坐标。该式解释中心位置的变化，高斯方向也随人体变形更新，不能只移动中心而保持整个椭球朝向不变。

人体表面的自由度也需要受到约束。参考姿态下的高斯中心可写成

$$
\overline{\boldsymbol\mu}_p=\mathbf v_p(\boldsymbol\beta)+\Delta\boldsymbol\mu_p.
$$

$\mathbf v_p(\boldsymbol\beta)$ 是体形参数 $\boldsymbol\beta$ 决定的参考人体表面点，$\Delta\boldsymbol\mu_p$ 是高斯相对该表面的位移，单位为米。关节姿态和这个位移都能改变投影位置，因此相同的图像残差可能通过不同参数得到补偿。若姿态仍错位，允许高斯自由移动并不保证形成正确衣物表面。检查时应分别比较只随关节运动的参考表面和完整高斯表示，再验证姿态校正、表面自由度与外观细节的作用。

跨时刻共享的高斯表面和逐时刻的人体姿态，需要共同解释同一组训练观测。用前文的渲染关系表示这一优化问题：

$$
\begin{aligned}
\left(\mathcal G^\star,\{\boldsymbol\theta_t^\star\}\right)
={}&\underset{\mathcal G,\{\boldsymbol\theta_t\}}{\arg\min}\;
\frac{1}{|\mathcal S|}\sum_{(t,k)\in\mathcal S}
\mathcal L_{\mathrm{image}}\!\left(
\widehat{\mathbf I}_{t,k},\mathbf I_{t,k};\Omega_{t,k}
\right)\\
&+\lambda_G\mathcal P_G(\mathcal G)
+\lambda_\theta\mathcal P_\theta(\{\boldsymbol\theta_t\}).
\end{aligned}
$$

$\mathcal G$ 包含参考姿态下的高斯位置、尺度、旋转、不透明度和颜色；$\boldsymbol\theta_t$ 在此包含人物整体平移、整体旋转及关节旋转。$\mathcal S$ 是训练时刻与相机的配对集合，$\Omega_{t,k}$ 是对应人物区域。$\mathcal P_G$ 限制表面偏离起点的幅度，$\mathcal P_\theta$ 限制姿态修正及其随时间的变化，$\lambda_G,\lambda_\theta$ 为权重。这是共同优化的概括表达，具体正则与参数范围应随实验说明，不能视为原论文唯一的损失形式。

增加可优化参数并不会自动消除多解。同一时刻来自不同方向的观测，能约束仅在一个投影中成立的表面与姿态补偿。本地目标的 90 个有效训练时刻中，按当前投影框与尺寸筛选规则，79 个只有一个训练相机区域，11 个有两个；这描述的是实际进入损失的观测，不能据此断言人物在其他相机完全不可见。因此，需要把联合优化与同步多视角监督分别检验，最终仍以未参与训练的时刻和相机判断几何与外观是否一致。

![相同训练观测下，表面拟合和姿态表面联合拟合的真实留出对照](../outputs/dynamic-human-street-joint-pose-surface-024/paired-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 与本站留出渲染 · 帧 130、相机 4 · 后两组各 100 次全区域更新，同倍率、未锐化；原始数据 CC BY-NC-SA 4.0。

本地对照中，额外允许姿态修正后，13 个人物留出区域的平均 PSNR 从仅优化表面的 **20.30 dB 提高到 20.43 dB**，但其中 4 个区域的 PSNR 下降。上图的袖口出现局部变化，提袋、衣物和双腿仍然模糊；训练画面中的衣物也没有恢复清楚。全部 13 组留出裁图与 4 组训练裁图的检查表明，这组联合优化改善了部分拟合，尚未解决清晰重建。这个结果不能证明姿态已经正确，也不能把剩余误差只归因于观测数量。[训练条件与真实对照](../experiments/37_dynamic_human_reconstruction.md#共同优化姿态表面与颜色)。


![OmniRe 完整人体建模消融图，比较形变对象和 SMPL 人体表示](../outputs/dynamic-human-reconstruction-003/assets/omnire-figure07.png)
> 图源：Chen 等（2025），OmniRe Fig. 7 · [论文第 10 页](https://arxiv.org/pdf/2408.16760v2#page=10) · 完整原图，保留 A、B 两组 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

A 组比较是否纳入可形变对象，B 组比较是否使用 SMPL 人体表示；腿部运动是 B 组最直接的观察位置。论文表 2 的人体区域新视图 PSNR 在完整模型下为 24.36 dB，去除 SMPL 人体节点后为 23.18 dB。这支持显式人体运动表示在该设置中的作用，并不意味着仅扩大背景高斯数量就能解决人物模糊。

以上为作者在 Waymo 上的实验：使用三个前向相机、图像尺寸 960×640、激光深度与实例框监督，每第 10 帧留作测试；每场景联合训练 30,000 次，硬件为单张 RTX 4090。这里的测试是留出时刻，与本地篮球基准的留出相机测试不同，数值不能直接横向比较。

进入更大场景后，背景沿用统一坐标、分块和细节层级，人物需要独立的身份、运动与形变表示。人物跨越背景分块时仍应保持同一对象身份，并在统一坐标系中渲染。训练资源优先保证人物原始像素和局部几何，不能为扩大覆盖范围而把人物不断降采样。

街景阶段采用 OmniRe／DriveStudio 作为重点框架，先选具有足够人物像素、完整运动段和多相机可见性的公开片段。HUGS Human Gaussian Splats 可作为单目人物与场景联合重建的另一路参考。是否完成大场景任务，要由街景中的独立人物评测决定。



### 人体关键点约束的是结构，不是真实衣物表面

人体关节或脚部标志点的多视角预测，可以先通过三角化得到三维假设，再用于拟合身体姿态。设 $\widehat{\mathbf X}_j$ 为第 $j$ 个预测三维点，$\mathbf J_j(\boldsymbol\beta,\boldsymbol\theta)$ 为人体模型对应点，基本拟合目标可写为

$$
\underset{\boldsymbol\beta,\boldsymbol\theta,s,\mathbf t}{\operatorname{minimize}}
\quad\sum_j w_j\,\rho\!\left(
\frac{\left\|s\mathbf J_j(\boldsymbol\beta,\boldsymbol\theta)+\mathbf t-\widehat{\mathbf X}_j\right\|_2}{\ell}
\right)+\lambda\mathcal P(\boldsymbol\beta,\boldsymbol\theta,s).
$$

$s$ 为尺度，$\mathbf t$ 为平移，$\ell$ 是用于归一化残差的身体长度；$w_j$ 表示观测可信度，$\rho$ 为鲁棒损失，$\mathcal P$ 为体形与姿态先验。这是结构拟合的概括表达。多视角预测相互接近，并不能证明被遮挡的关节真实可见；背面人物的面部猜测尤其不应自动加入约束。基础 SMPL 的手指自由度也不足以解释所有抓握动作。

![真实照片与人体结构先验的投影，展示身体与衣物之间的差别](../outputs/dynamic-human-enerf-wholebody-prior-060/camera04-body-prior.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站拟合 · SMPL 结构检查，不是外观重建。

上图中，头部、肘部和脚部可以对齐，而宽松衣服、头发、鞋和玩具仍位于模型表面之外。因此，更小的关节点误差不能替代清晰外观的验证；表面约束也需要允许真实衣物偏离人体模板。本地限定在裸露前臂的柔性约束仅产生小幅、混合变化，未解决交叉手臂和物体拖影。仍应分别检查身体对应、可见表面与跨时刻运动，而不是把整个人强制压到模板上。[实际拟合与同预算对照](../experiments/37_dynamic_human_reconstruction.md#全身关键点与人体结构先验的修正)。


#### 手部观测必须区分可见性与预测分数

![原生手掌与全身及局部手部模型的预测](../outputs/dynamic-human-enerf-hand-motion-082/candidate-panels/camera04-candidates.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与 RTMW、RTMPose-Hand 实测 · 机位04，帧720、726、732 · 原生像素；彩色点为预测。

每行从左至右是原照片、全身模型、局部手部模型，以及缩小和扩大裁剪范围后的局部预测。上两行的手正在抓握玩偶，部分预测落到玩偶表面；仅改变裁剪范围，某些关节就明显移动。最下行张开的手掌较清楚，局部预测也较稳定，但稳定并不等于每根手指都对应正确。

网络的响应分数不能直接解释成“这个指尖真实可见的概率”。不可见区域也可能得到完整的21点结构；多人交互时，还必须确认预测属于哪一个人。相同模型在不同裁剪下保持一致，只检验了对裁剪的敏感性，不能代替跨机位的独立核验。

对人物 $a$ 的关节 $j$，二维观测约束可概括为

$$
\mathcal L_{\mathrm{keypoint}}
=\sum_{c,t,a,j}v_{ctaj}\,u_{ctaj}\,w_{ctaj}\,
\rho\!\left(\left\|\pi_c\!\left(\mathbf X_{aj}(t)\right)-\mathbf x_{ctaj}\right\|_2\right).
$$

$\mathbf x_{ctaj}$ 是机位 $c$、时刻 $t$ 的二维预测，单位为像素；$\mathbf X_{aj}(t)$ 为三维关节，$\pi_c$ 为相机投影。$v_{ctaj}$ 表示可见性筛选，$u_{ctaj}$ 表示人物归属筛选，均可取0或1；$w_{ctaj}$ 是保留下来的观测权重。不可见或归属不确定时，不应仅因响应较高而令其成为强约束。鲁棒损失 $\rho$ 可以减弱离群值，不能替代这两项筛选。

细节还受到检测输入分辨率限制：全身检测器缩放整个人后，手指只占其中很小一块。局部检测能让网络使用更多输入像素描述这只手，却不能创造被遮挡的指尖。跨机位核验时，应把目标相机从三角化与拟合中完全排除，再检查投影是否与该相机的照片一致；没有形成有效三维假设的点也必须计入覆盖率，不能只报告成功点的平均误差。

在这一运动片段中，局部检测改善了手腕的跨机位一致性，指尖却反而变差。裁剪稳定性筛选还会失去大量指尖观测。由这些预测得到的平滑三维曲线仍可能偏离真实皮肤，因而没有作为高斯训练的手指约束。

掌部结构先验可以进一步限制不合理形变。例如，短时间内将手腕和四个掌指关节的参考位置记为 $\overline{\mathbf X}_j$，刚性掌部近似为

$$
\mathbf X_j(t)=\mathbf R(t)\overline{\mathbf X}_j+\mathbf t(t),\qquad
\|\mathbf X_i(t)-\mathbf X_j(t)\|_2
=\|\overline{\mathbf X}_i-\overline{\mathbf X}_j\|_2.
$$

$\mathbf R(t)\in\mathrm{SO}(3)$ 为掌部旋转，$\mathbf t(t)\in\mathbb R^3$ 为平移；各点间距离固定。它约束的是掌部的局部形状，手指屈伸还需要独立的关节运动。形状不再拉伸，也不保证位置正确：本地五点掌部拟合在未参与拟合的相机中没有改善投影，因此同样没有用于驱动高斯。完整条件与原生照片见[连续手部观测与掌部运动核验](../experiments/37_dynamic_human_reconstruction.md#连续手部观测与掌部运动核验)。

#### 图像运动约束与重建清晰度

沿一条近似直线的边缘移动时，多个位置可能具有相似图块。这是光流中的孔径问题：局部亮度通常只能约束垂直于等亮度线的运动，不能唯一确定沿线方向的运动。往返跟踪一致、图块相关性很高，也未必代表完整的二维对应正确。[原生图块中的反例](../experiments/37_dynamic_human_reconstruction.md#局部图像对应的歧义)说明了这一点。

在图像梯度非零处，定义单位方向与有限帧间位移的法向分量

$$
\mathbf n(\mathbf p)=\frac{\nabla I(\mathbf p)}{\|\nabla I(\mathbf p)\|_2},\qquad
d_n(\mathbf p)=\mathbf n(\mathbf p)^{\mathsf T}\mathbf v(\mathbf p).
$$

$I$ 为灰度图像，$\mathbf p$ 为像素位置，$\mathbf v\in\mathbb R^2$ 为两张照片之间的位移，单位为像素。这里的“法向”是**图像等亮度线的法向**，不是三维表面法线。只约束 $d_n$ 可以保留沿边缘方向的不确定性；它仍须经过可见性、人物归属和遮挡筛选，不能把玩偶表面的运动当成手的运动。

对动态高斯，可先将同一高斯中心在两个时刻的投影相减，再按源图中的可见贡献加权：

$$
\widehat{\mathbf v}(\mathbf p)=
\frac{\sum_i\overline w_i(\mathbf p)
\left[\pi_c\!\left(\boldsymbol\mu_i(t')\right)-\pi_c\!\left(\boldsymbol\mu_i(t)\right)\right]}
{\sum_i\overline w_i(\mathbf p)}.
$$

$\boldsymbol\mu_i(t)$ 是第 $i$ 个高斯的世界坐标中心，$\pi_c$ 为相机投影；$\overline w_i$ 是由源图透明度与前方遮挡确定的贡献。此式是高斯**中心位移的图像近似**，没有精确描述一个高斯覆盖范围内所有表面点的运动。若为稳定优化而固定源图权重，还须在优化后按当前几何重新计算可见性，检查近似是否失效。

在原有图像与正则化损失 $\mathcal L_0$ 上加入一项运动约束，可写为

$$
\mathcal L=\mathcal L_0+
\frac{\lambda_n}{|\Omega|}\sum_{\mathbf p\in\Omega}
\rho\!\left(\frac{\mathbf n(\mathbf p)^{\mathsf T}\widehat{\mathbf v}(\mathbf p)-d_n(\mathbf p)}{\delta}\right).
$$

$\Omega$ 为通过筛选的可见手部像素，$\delta$ 是像素误差尺度，$\lambda_n$ 为无量纲权重，$\rho$ 为鲁棒损失。图像估计的 $d_n$ 仍是候选观测，不能视为真实三维运动。尤其是只覆盖手背和手腕的约束，并未提供每根手指的完整形状与屈伸信息。

![张手时原照片与增加局部运动控制后的原生重建](../outputs/dynamic-human-enerf-hand-fine-085/comparison/camera04-frame000731-target_right_hand.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位04、留出帧731 · 原照片、原模型、细节点与图像联合优化、再加入运动约束。

原照片中的手指已经与玩偶分离，三种重建仍出现皮肤与玩偶边缘混合的碎片。此次同时优化局部高斯的几何、颜色、不透明度和运动，并增加32个局部控制节点。加入运动项后，训练观测的法向残差从约1.24降到0.67像素，留出图像中的指间分离仍未恢复。因此，运动约束拟合更好，不能直接解释为表面重建更准确。

还要区分**表面表示的密度**与**运动控制的空间分辨率**。前文[变形图](#骨架之外的局部变形如何进入渲染)中，高斯运动由邻近节点及权重共同决定。增加节点，可以使邻近区域获得更多不同运动；它并不提供新的皮肤表面，也不保证手和玩偶已正确分离。本地新增节点沿用原模型的候选材料位置，一些投影仍落在衣服或玩偶边缘，不能把这些节点当成手部关节。

新增变形已经实际生效，但不足以恢复手指。还需看高斯对手部像素的**实际可见贡献**：中心不在手部掩码内的高斯，其投影覆盖范围仍可能跨入手部。当前检查中，可更新高斯平均承担估计手部区域约81.5%的贡献；手部区域对这些参数的图像梯度也没有被其余像素淹没。因此，简单扩大更新范围或提高局部损失权重，并没有得到当前证据支持。估计掩码会漏掉手指，也不能充当表面真值。原生全身图与连续对照见[局部运动节点与手部表面](../experiments/37_dynamic_human_reconstruction.md#局部运动节点与手部表面)。

#### 为什么单视角拟合不等于三维重建

![同一手部在拟合视角与其他视角中的实际重建](../outputs/dynamic-human-enerf-hand-view-scope-089/comparison/single-view-versus-consistency.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 原生窗口 · 原照片、原模型、15视角拟合、单视角拟合。

上排是参与单视角拟合的04机位，下排是没有参与该次拟合的13机位。只用04机位的一张照片优化后，手掌和手指轮廓明显恢复；同一模型从13机位看，却出现拉长的皮肤色碎片。这些都是模型的实际渲染，未经过锐化或补画。**图像变清楚只能说明该视角的拟合能力，不能证明三维表面正确。**

对于相机中心 $\mathbf o_c$ 和像素 $\mathbf p$，针孔模型对应一条空间射线：

$$
\mathbf X(\lambda)=\mathbf o_c+\lambda\mathbf d_c(\mathbf p),\qquad
\lambda>0,\qquad
\pi_c\!\left(\mathbf X(\lambda)\right)=\mathbf p.
$$

$\mathbf d_c(\mathbf p)$ 为世界坐标系下的单位射线方向，$\lambda$ 是沿射线的距离，与场景坐标使用相同长度单位；$\pi_c$ 为相机投影。同一条射线上的许多三维位置具有相同投影，单张照片不能唯一确定深度。高斯还可以联合调整位置、协方差、不透明度与视角相关颜色，使一张图像看起来正确，却在其他方向暴露错误形状。

多视角训练要求同一组材料参数同时解释不同相机的观测；动态场景还要使用各张照片真实拍摄时刻对应的形变。这里的两个条件必须分别核验：**共享的局部表面是否合理，以及这个表面在各次曝光时刻的位置是否正确**。单个名义帧编号并不保证各相机完全同时曝光。

本地对照固定其他身体部分，分别进行单视角、双视角、同一名义帧的多视角及多个时刻的多视角拟合。两个参与拟合的机位可以同时出现清楚的手形，其他方向却仍暴露碎片，说明**两张图都清楚也不足以验证三维表面**。核验对应点时，还要确认它们属于同一个可见表面：相近的曝光时刻不代表相近的观察方向，玩偶上的纹理匹配也不能证明手指几何正确。

![两个拟合机位的清晰手形与其他机位的失真](../outputs/dynamic-human-enerf-hand-two-view-093/comparison/two-view-versus-consistency.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 从上到下为04、17、08机位 · 原照片、原模型、单视角拟合、双视角拟合 · 原生像素。

图中04和17机位参与双视角拟合，08机位没有参与该次拟合。它们沿用各自估计拍摄时刻的运动状态，没有把两张图当作完全同步的几何真值。前两排手形恢复，第三排却出现皮肤色碎片和玩偶变形。相邻机位的原图匹配主要覆盖玩偶纹理，仅有少量手背或前臂候选点，尚不能据此建立可靠的手指表面。

另一项条件独立的实验恢复全部15个训练机位，并让局部高斯按投影位置梯度克隆或分裂。高斯数量和分布实际发生变化，一些轮廓有所改善，但指间粘连与局部破碎仍然存在。**密度控制提供更多表面表示能力，无法单独保证观测对应、局部运动与几何一致性**。这并不证明密度控制无用，而是说明当前条件下它还不足以解决问题。强制统一手部时刻也未通过画面验证，因此现有证据不支持直接清零相机时间偏移。

![当前开发对照的全身轮廓与衣物表现](../outputs/dynamic-human-enerf-hand-adaptive-094/comparison/current-baseline-full-person.png)
> 图源：同上 · 06机位、732帧 · 左为原照片，右为保留的开发对照模型；灰色区域为人物单独渲染的背景，未表示完成大场景背景联合重建。

全身图能看清躯干、衣物和步态轮廓，局部原生窗口则显示尚未恢复的手指。这两个尺度应同时展示：既不能只用全身缩略图掩盖局部模糊，也不能把单个拟合窗口变清楚当作人物整体、跨视角和连续运动都已通过。[双视角与密度对照](../experiments/37_dynamic_human_reconstruction.md#双视角拟合与局部密度控制)保留完整实验条件。

增加运动自由度也不能自动恢复手指。在相同多视角观测下，为每个高斯增加连续平移残差，仍然可能得到连续移动的碎片。手部关节结构提供了更强的形状约束，但它首先要与照片中的姿态和遮挡相符。[连续运动对照](../experiments/37_dynamic_human_reconstruction.md#局部连续运动自由度的同预算对照)保留具体条件。

手部参数模型用共同的形状和关节连接，约束不同视角中的候选手形。HaMeR逐图预测MANO参数；要形成同一个动态三维对象，还需在世界坐标系中共享形状，并让姿态只随拍摄时刻变化。以本地连续拟合为例，第$i$个手部顶点为

$$
\mathbf v_i(t)=s\mathbf R(t)
\left[\mathcal M_i\!\left(\boldsymbol\beta,\boldsymbol\theta(t)\right)
-\mathbf J_0\!\left(\boldsymbol\beta,\boldsymbol\theta(t)\right)\right]
+\mathbf p(t).
$$

$\mathcal M_i$为MANO生成的局部顶点，$\mathbf J_0$为手腕关节；两者相减使局部坐标以手腕为原点。$\boldsymbol\beta$是整个区间共享的10维形状参数，$\boldsymbol\theta(t)$包含15个局部关节的轴角旋转，$s$将MANO长度换算为场景长度，$\mathbf R(t)$将手部方向变换到世界坐标，$\mathbf p(t)$为世界坐标中的手腕位置。关节角以弧度计，位置与场景坐标使用相同长度单位。

本地模型用三次B样条表示连续轨迹：$\mathbf p(t)=\sum_k B_k(t)\mathbf p_k$，各关节的轴角也由同一时间基函数展开。同一拍摄时刻，所有相机看到的是**同一套三维顶点**，而不是各自独立预测的手。这里的$t$以帧为单位并包含现有相机时间偏移；时间偏移仍是估计值。网络给出的单目深度没有直接当作场景深度。

![单视图预测与共享手部模型在原照片中的投影](../outputs/dynamic-human-hand-consensus-097/shared-hand-versus-source.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；HaMeR、MANO与本地拟合 · 上为03机位726帧，下为04机位732帧 · 原照片、单视图预测、15机位共享模型、排除04／09／13机位的共享模型 · 蓝色为结构先验。

上排共享模型把单视图较伸展的预测收拢为抓握候选，下排显示张手时的连续手指结构。但部分指尖与原照片轮廓仍不重合；蓝色材质也没有恢复皮肤纹理。手被玩偶遮挡时，当前叠加没有使用玩偶深度，不能凭蓝色覆盖关系确认隐藏手指的位置。

共享模型可通过鲁棒重投影误差拟合：

$$
\mathcal L_{\mathrm{proj}}=
\frac{\sum_{c,f,j} w_{cfj}\,
\rho\!\left(\left\|\pi_c\!\left(\mathbf J_j(t_{cf})\right)
-\widehat{\mathbf u}_{cfj}\right\|_2\right)}
{\sum_{c,f,j}w_{cfj}},\qquad
\rho(r)=a^2\ln\!\left(1+\frac{r^2}{a^2}\right).
$$

$c,f,j$分别索引相机、照片帧和关节；$\widehat{\mathbf u}_{cfj}$是HaMeR预测的二维关节，$\pi_c$使用原标定相机，$t_{cf}$为估计拍摄时刻。权重$w_{cfj}=\sigma_0^2/(\sigma_0^2+\delta_{cfj}^2)$随裁剪扰动$\delta_{cfj}$增大而减小；本地诊断取$\sigma_0=2$像素、$a=4$像素，另加形状与运动平滑正则。这是本地检验目标，并非HaMeR原论文的训练损失；裁剪稳定也不能证明关节可见或预测正确。

**重投影一致性与照片重建质量需要分别验证。** 共享模型使多视角关节预测更一致，但仍有指尖偏移。把已有高斯的位置和运动软约束到这个候选表面后，高斯确实更接近模型，实际渲染却仍有破碎的手指。几何先验的损失下降，不能代替照片重建质量的检验。[共享模型检查](../experiments/37_dynamic_human_reconstruction.md#连续手部模型的跨视角检验)与[先验和表面高斯对照](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)保留实验条件。

#### 手部表面高斯与关节运动

让高斯直接附着在手部三角面上，可以把关节运动同时传递给中心位置和协方差方向。以下是本地验证采用的参数化。对三角面$i$，以三个顶点的平均位置为中心，以两条正交切向量与单位法向量组成旋转矩阵$\mathbf F_i(t)$：

$$
\begin{aligned}
\mathbf c_i(t)&=\frac{\mathbf v_{i1}(t)+\mathbf v_{i2}(t)+\mathbf v_{i3}(t)}{3},\\
\mathbf F_i(t)&=\begin{bmatrix}\mathbf e_{i1}(t)&\mathbf e_{i2}(t)&\mathbf n_i(t)\end{bmatrix},\\
\boldsymbol\mu_i(t)&=\mathbf c_i(t)+\mathbf F_i(t)\boldsymbol\delta_i,\\
\boldsymbol\Sigma_i(t)&=\mathbf F_i(t)
\begin{bmatrix}
\sigma_{i1}^{2}(t)&0&0\\
0&\sigma_{i2}^{2}(t)&0\\
0&0&\sigma_{in}^{2}(t)
\end{bmatrix}
\mathbf F_i(t)^{\mathsf T}.
\end{aligned}
$$

$\boldsymbol\delta_i$是可训练的局部位置偏移；$\sigma_{i1},\sigma_{i2}$控制沿表面的覆盖范围，$\sigma_{in}$控制法向厚度，均采用场景长度单位。手指弯曲时，三角面的位置与方向随关节变化，高斯也随之平移、旋转。局部偏移和尺度保留有限调整能力，以容纳模板与真实表面的差异。

本地对照在每个手部三角面放置一个高斯，共1538个。两组分别采用仅由预测关键点拟合的手部姿态，以及额外利用图像轮廓约束拟合的姿态；颜色、不透明度、局部偏移与尺度使用同样的105张照片优化6000次。两组均固定关节先验，并在训练全程把6330个旧邻域高斯的不透明度限制在不高于0.02，以检验改善先验轮廓能否改善真实图像。

![两种手部先验在相同训练条件下的原生渲染对照](../outputs/dynamic-human-enerf-contour-surface-103/native-articulated-hand-comparison.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 上为04机位，下为13机位，均为留出时刻731帧 · 原照片、原模型、关键点先验、增加轮廓约束的先验 · 原生像素。

**轮廓拟合更好，并不保证手指渲染更清楚。** 右侧两组仍有明显斑块和错误指缝，限制旧表面后还损伤了部分手部与玩偶边界。新增表面参与合成的程度提高，但其单独渲染也没有恢复准确的手指外观。因此，轮廓覆盖、表面可见性和最终重建清晰度必须分别检查，不能用前两者代替真实图像效果。

判断新表面是否真正参与图像，应检查它在最终像素中的合成权重：

$$
w_{\mathrm{hand}}(\mathbf p,t)=
\sum_{i\in\mathcal H}T_i(\mathbf p,t)\,\alpha_i(\mathbf p,t),
\qquad
T_i=\prod_{j<i}\left(1-\alpha_j\right).
$$

$\mathcal H$表示新增手部高斯，$\alpha_i$为投影后的有效不透明度，$T_i$为到达该高斯前的透射率。三角面存在、参数得到梯度，并不保证它在相机中有足够的可见贡献；已有表面的深度与不透明度也参与合成。贡献大同样不代表表面正确，仍需同时对照原照片、未训练机位和连续时刻。对于共享颜色$\mathbf c_i$，高斯颜色的光度梯度也包含同样的可见权重：

$$
\frac{\partial\mathcal L}{\partial\mathbf c_i}
=\sum_{\mathbf p,t}
T_i(\mathbf p,t)\alpha_i(\mathbf p,t)
\frac{\partial\mathcal L}{\partial\mathbf C(\mathbf p,t)}.
$$

因此，前方旧表面接近不透明时，新表面即使位于合适的投影位置，也可能只能获得很小的颜色梯度。对每个旧高斯设置较低的不透明度上限，并不意味着整层旧表面变得透明。若某条光线经过的100个旧高斯在该像素处均满足$\alpha_j=0.02$，其剩余透射率为

$$
T_{\mathrm{old}}=(1-0.02)^{100}\approx0.133.
$$

这个合成示例说明，多个低不透明度高斯仍可共同遮挡后方表面。降低旧表面的贡献时，还要检查其是否同时承载前臂、衣物或道具；几何邻域不能直接当作语义手部标签。另一方面，即使可见性改善，表面位置、手指对应或共享颜色拟合不准确，仍会形成斑块。本例的两组候选均未替换原模型。随后固定全部高斯属性，仅用原照片优化手部关节和手腕，仍未恢复清晰指缝。这说明本例中有限的姿态修正不足以修复现有表面外观，不能由训练损失下降推断细节恢复；也不能据此认定姿态已经准确。[姿态修正的原生图像检验](../experiments/37_dynamic_human_reconstruction.md#固定外观后的手部姿态修正)保留对应条件与画面。[先验与表面表示](../experiments/37_dynamic_human_reconstruction.md#手部先验与关节表面高斯的图像检验)、[不透明度初始化对照](../experiments/37_dynamic_human_reconstruction.md#旧表面遮挡与不透明度初始化)和[轮廓先验的真实图像检验](../experiments/37_dynamic_human_reconstruction.md#轮廓先验与真实手部渲染)保留实验条件及原生图片。


### 相机与人物必须对应同一个拍摄时刻

![真实街景原图与两种时间处理下的人物三维标注框投影](../outputs/dynamic-human-street-005/assets/street-camera-time-review.png)
> 图源：Argoverse 2 原始观测与本站标注投影核查 · 相机 4、帧 134 · 原图保留匿名化 · © 2021 Argo AI, LLC，CC BY-NC-SA 4.0。

三列依次是原图、采用激光时刻自车位姿的投影、采用照片拍摄时刻自车位姿的投影。彩色线框来自同一个人物三维标注框，不是人体重建结果。人物与车辆都在运动时，“最近的一帧”并不意味着完全同时拍摄；靠近相机的衣物边缘和肢体对时间差尤其敏感。

令 $t_c$ 为照片拍摄时间，$t_l$ 为激光参考时间。人物参考坐标中的点 $\overline{\mathbf p}$ 在照片中的正确投影为

$$
\begin{aligned}
\begin{bmatrix}\mathbf p_c(t_c)\\
1\end{bmatrix}
&=\mathbf T_{cw}(t_c)\mathbf T_{wa}(t_c)
\begin{bmatrix}\overline{\mathbf p}\\
1\end{bmatrix},\\
\mathbf u(t_c)&=\pi_{\mathbf K}\!\left(\mathbf p_c(t_c)\right).
\end{aligned}
$$

$\mathbf T_{wa}$ 将人物局部坐标变换到世界坐标；$\mathbf T_{cw}$ 将世界坐标变换到相机坐标；$\pi_{\mathbf K}$ 是包含相机内参的透视投影。若点随关节形变，还应先应用前文的蒙皮变换。两种变换均应在 $t_c$ 求值，不能只补偿相机运动而把人物冻结在另一个时刻。

当点来自激光参考时刻的自车坐标系时，自车运动补偿关系为

$$
\begin{bmatrix}\mathbf p_{e(t_c)}\\
1\end{bmatrix}
=\mathbf T_{e(t_c)w}\mathbf T_{we(t_l)}
\begin{bmatrix}\mathbf p_{e(t_l)}\\
1\end{bmatrix}.
$$

下标 $e(t)$ 表示时刻 $t$ 的自车坐标系，$w$ 表示世界坐标系。该关系把同一世界位置重新表达在照片时刻的自车坐标中。运动人物在两个时刻的真实位置还需要其自身轨迹和关节运动来描述。

本地街景中，相机与最近激光帧的时间差最大约 24 毫秒；相应相机位置差最大约 0.22 米。因而输入必须保留原始时间戳和拍摄时刻位姿，而不能在配对之后只留下统一帧号。上图展示的只是自车运动补偿。接入人体模板后的实际投影仍发现头部、躯干和肢体偏移，说明相机位姿时间正确并不意味着人体姿态已经与观测对齐。

### 颜色和不透明度为何会共同补偿表面误差

![同一训练视图分别替换颜色、几何与不透明度，展示参数之间的相互依赖](../outputs/dynamic-human-street-capacity-components-022/paired-audit/frame129-camera4-geometry-color.png)
> 图源：Argoverse 2 与本站参数替换诊断 · 帧 129、相机 4 · 原图匿名化保留，无锐化；数据 CC BY-NC-SA 4.0。

完整拟合的衣纹并不是只存在于颜色参数中。上图分别把局部拟合得到的颜色、几何与不透明度放回共同起点：只替换颜色产生色块，只替换几何与不透明度能显示部分褶皱结构，但仍有破碎外观。两组参数共同替换时，才恢复被监督画面的主要细节。这说明参数之间存在相互补偿；它不是各参数分别训练的消融，也不能据此确定唯一误差来源。

对于像素位置 $\mathbf u$，由近到远排序的高斯贡献为

$$
\begin{aligned}
\widehat{\mathbf C}(\mathbf u)
&=\sum_{i=1}^{N}T_i(\mathbf u)\alpha_i(\mathbf u)\mathbf c_i
+T_{N+1}(\mathbf u)\mathbf c_{\mathrm{bg}},\\
T_i(\mathbf u)&=\prod_{j<i}\bigl(1-\alpha_j(\mathbf u)\bigr),\\
\alpha_i(\mathbf u)&=o_i\exp\!\left[-\frac12
(\mathbf u-\boldsymbol\mu_i')^{\mathsf T}
(\boldsymbol\Sigma_i')^{-1}
(\mathbf u-\boldsymbol\mu_i')\right].
\end{aligned}
$$

$o_i$ 为不透明度参数，$\boldsymbol\mu_i'$ 为投影中心，$\boldsymbol\Sigma_i'$ 为像平面协方差，$T_i$ 为前方高斯之后剩余的透射率。$\mathbf c_{\mathrm{bg}}$ 是剩余透射率下的背景颜色。移动中心、改变尺度或不透明度，都会改变哪些颜色能到达同一像素；因此，把已经共同拟合的颜色放到另一组几何上，一般不会保留原效果。

![恢复原来的方向相关颜色、零阶颜色或不透明度，均未自动解决另一时刻的破碎外观](../outputs/dynamic-human-street-capacity-components-022/paired-audit/frame140-camera4-color-opacity.png)
> 图源：Argoverse 2 与本站只读渲染 · 帧 140、相机 4 · 未进入局部双帧拟合；相同区域、无图像增强。

本例人体使用一阶球谐颜色，包括一个零阶基函数和三个方向相关基函数。恢复源模型的方向相关系数后，其他时刻仍然存在明显伪影，恢复零阶系数或不透明度也没有完成修复。因此，不能将跨时刻失败简单归结为球谐阶数太高。更完整的训练必须同时约束参考表面、人体运动与可见性，并在多个时刻保留清晰观测的监督。

## 适用边界

![人物区域误差最大的动态重建视图，展示仍然存在的衣物斑点和破碎外观](../outputs/dynamic-human-reconstruction-003/assets/person-detail-hard-view.png)
> 图源：本站实测 · 留出相机 15、帧 21 · 按全部视图中动态组人物区域 PSNR 最低选取，非只展示较好视角。

侧后视角已经恢复人体的大体姿态，但近处衣物仍有彩色斑点和破碎外观，尚不能称为各视角均清晰。本轮对照同时改变了动态表示与时序约束，且两组高斯数量不同，因此不能把全部改善单独归因于某个损失项。

室内人物运动基准、街景可变形模型与人体关节模型均已完成训练及留出图像评测，但街景近处衣物和肢体仍未达到清晰重建要求。室内原始图像只有 640×360；训练不能可靠恢复原图未记录的面部或服装微细节。该发布数据没有留出相机的前景分割，因此本次人物局部指标采用事先固定的矩形区域，不能称作精确人体掩码指标。

**公开原始图像不等于未经匿名化的图像。** Argoverse 2 在发布前已模糊人脸与车牌；本地保留了这些处理，没有额外修复人脸。因而该数据可以检验衣物、肢体、遮挡和轨迹，却不能提供被覆盖面部纹理的真实监督。模型拟合出原有马赛克，也不能视为恢复了面部细节。当前街景训练还按原图宽、高各约三分之一加载，缩放带来的信息损失应与匿名化、模型拟合误差分别说明。

验证精细人物时，应选择原观测中确实存在所需细节、同一时刻有足够视角覆盖的采集。ActorsHQ 的论文展示了高分辨率观测中的头发和皮肤细节；DNA-Rendering 提供多种服装和动作的同步多视角记录；ENeRF-Outdoor 提供室外背景中的人物运动。前两者主要用于人物精细重建，后者可衔接室外动态场景，但都不能单独证明城市尺度联合重建已经完成。数据选择还应检查人物占据的实际像素数，不能只比较整幅图像的分辨率。

作者提供的初始化点云被保留，其生成时使用了哪些相机未在发布包中说明；本次可确认留出 RGB 未进入优化损失，但不宣称初始化几何已经证明只来自训练相机。单目遮挡区域、跨人物泛化和街景尺度还需独立验证。

![修正颜色参照后侧面衣物仍存在亮斑与破碎外观](../outputs/dynamic-human-color-continuity-004/assets/color-camera15-frame021.png)
> 图源：本站实测 · 相机 15、帧 21 · 两组仅改变颜色参照，右列仍未达到清晰衣物重建要求。

颜色参照修正改善了整体外观连续性，但近处衣物仍有明显残留。首帧也存在相似误差，因此后续诊断需要返回首帧几何、颜色和可见性，而不是把时序正则当成所有模糊的解法。

![真实街景的原始观测、可变形模型和人体关节模型，同为三万次更新](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站留出渲染 · 帧 130、相机 4、人物 79 · 同分辨率、同区域放大，保留原图匿名化，未锐化或修复 · 原始数据 CC BY-NC-SA 4.0。

这组街景对照说明需要区分“原图没有记录细节”和“模型没有拟合出已有细节”。左图在当前训练分辨率下仍能看见衣物褶皱、手提袋边缘和分开的双腿；中、右两种表示均完成 30,000 次更新，人体关节模型仍有肢体错位和破碎外观。主目标全部 13 个留出区域的平均 PSNR 从 19.07 提高到 19.70 dB，但本图的近处侧面反而下降，衣物清晰度也未通过。人体先验为运动提供了结构，姿态、表面和可见性仍需与观测一致，不能用平均误差改善替代这三项检查。

## 实验附录

ENeRF-Outdoor 的三组官方压缩包已下载到本地。第一组 `actor1_4` 已完整校验和解压，包含 18 个机位、每机位 1,200 帧；首段 60 帧已核验原生图像、作者前景掩膜及标定。首帧高斯模型完成 6,000 次更新，初始化只使用 15 个训练机位；3 个留出机位的 6 个人物对照均已查看，主要衣物结构恢复，侧面文字与轮廓仍有残差。它采用 gsplat 的密度控制与本地数据适配，不是 4K4D 或 OmniRe 官方完整复现。见[首帧条件](../outputs/dynamic-human-enerf-native-025/first-frame/protocol.json)、[全部首帧结果](../outputs/dynamic-human-enerf-native-025/first-frame/evaluation/metrics.json)、[原图与指标独立核验](../outputs/dynamic-human-enerf-native-025/first-frame-independent-audit.json)。

首段 60 帧已完成连续重建：54 个时刻参与拟合，6 个中间时刻留出并由相邻状态插值；另外 3 个机位全程不参与训练。全部 270 个留出视图已渲染；人物抬手和转动物品时仍有明显模糊，未通过清晰度验收。连续对照：[机位 00](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera00-motion-comparison.mp4)、[机位 06](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera06-motion-comparison.mp4)、[机位 12](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera12-motion-comparison.mp4)。视频以 30 帧每秒供检查，非已核实的采集帧率。参见[连续模型与困难帧对照](../outputs/dynamic-human-enerf-native-025/comparison/capacity-summary.json)、[全图渲染示例](../outputs/dynamic-human-enerf-native-025/motion/evaluation/camera00-frame059-render.png)和[实际视觉检查范围](../outputs/dynamic-human-enerf-native-025/motion-visual-review.json)。

本轮静态与动态两组各完成 70,000 次梯度更新，按冻结协议评测 4 个相机 × 31 帧。已核对全部相机的首、中、末帧及误差最大视图；所有帧均完成数值评测并保存渲染。保存图片的 PSNR 独立复算与浮点结果最大相差 0.0077 dB，属于 8 位图片量化差异。

颜色参照检查确认：当前固定代码版本的“上一帧颜色”与正在优化的参数共享存储，因而该颜色正则项和对应梯度始终为零。新的两组对照从同一个首帧模型开始、重置为相同优化器状态，只改变是否独立保存上一帧颜色。两组各完成 60,000 次后续更新和全部 124 个视图评测；这是对已观察问题的机制诊断。后续帧 31～60 也已完成训练、120 组留出视图评测和 240 张保存图片的独立复算，最大 PSNR 量化差异为 0.00875 dB。新区间扩大了事先固定的人物区域，不能与前一区间裁剪不同的均值直接横比。[完整验证与视觉检查范围](../outputs/dynamic-human-temporal-validation-006/independent-audit.json)。

固定几何的首帧颜色对照已完成，四个留出相机均检查原模型与两支再训练结果。该扩展未恢复近处衣物细节，两支诊断模型均未替代原动态基线。[训练协议](../outputs/dynamic-human-appearance-refinement-008/protocol.json)、[逐视角指标](../outputs/dynamic-human-appearance-refinement-008/evaluation.json)与[独立复核](../outputs/dynamic-human-appearance-refinement-008/independent-audit.json)保留负结果。

街景方向已获取约 1.26 GB 的 Argoverse 2 原始日志，并完成 156 个有效时刻、7 路相机共 1,092 张图片的标定、激光、人物轨迹和粗边界框掩膜导出。额外保存每张照片拍摄时刻的相机位姿；原图逐张复制并核对哈希，未重复 JPEG 压缩。原作者提供的 27 个人物姿态序列通过了帧数、人物编号、有效相机索引与旋转矩阵检查。官方 SMPL 中性模板已接入，主目标 79 的 100 个有效姿态均能生成有限值网格；七组原图叠加检查发现明显初始偏移，对齐尚未验收。

逐像素人物与天空掩膜采用 SegFormer B5 预测。1,092 张图片的全部 6,552 张输出已完成并通过独立核验；四个固定视图的抽查仍发现人物遮挡、手提物和匿名化头部分割误差，预测掩膜不能作为独立评测真值。Linux 环境已完成核心算子编译、完整训练模块导入和 RTX 5080 上的实际前向／反向检查；拍摄时刻位姿适配器已进入完整相机加载器，14 个真实图像案例的图片、掩膜和射线核对通过。可变形模型与人体关节模型均已完成 30,000 次更新和全部 105 张留出图评测，近处人物仍未达到清晰重建要求；后续重点是姿态对齐和衣物表面。详见[街景输入核查](../outputs/dynamic-human-street-005/input-audit.json)、[掩膜输出核验](../outputs/dynamic-human-street-005/semantic-output-audit.json)、[实际 CUDA 检查](../outputs/dynamic-human-street-005/linux-cuda-smoke.json)、[完整相机加载核验](../outputs/dynamic-human-street-005/full-camera-loader-audit.json)及[街景对照训练协议](../outputs/dynamic-human-street-joint-009/baseline-bounded/protocol.json)。

整段 156 帧的完整数据构造也已通过，包含 1,092 张照片、激光与 2,366 个有效人体姿态的坐标转换。此过程发现，原流程会用测试照片为初始化点云赋色。已增加仅使用训练时刻照片与激光点的初始化：在真实 21 帧片段中替换全部 14 张测试照片，训练图像、点云、背景种子及 7 个刚体／41 个可变形实例的种子逐项保持一致，原实现则会改变点云颜色。稀疏深度改用同一像素最近的有效激光点，并通过独立计算核对。相机与对象轨迹先验仍然已知，这种留出评测不代表未来运动预测。见[整段数据核验与边界](../outputs/dynamic-human-street-005/full-scene-dataset-audit.json)、[测试照片干预与初始化隔离核验](../outputs/dynamic-human-street-005/training-only-initialization-audit.json)。

见[动态人物重建基准与评测约定](../experiments/37_dynamic_human_reconstruction.md)、[当前方向与完成范围](../manifests/dynamic-human-work.json)、[配图来源](../outputs/dynamic-human-reconstruction-003/figure-provenance.json)。

街景基线在显存分配失败后，从 10,000 次更新的完整状态恢复模型、优化器、随机数与加密统计；三个恢复渲染与原检查点逐像素一致，下一张训练样本也一致。现已完成原定 30,000 次预算，全部 105 张最终图与 435 个人物框通过独立误差复算；连同早期检查点，共核对 114 张图和 471 个区域。保存目标图与原始照片按作者流程缩放的结果逐像素一致。主目标的全部 13 个留出区域已逐一看图，近处衣物、肢体和遮挡仍有明显误差。见[恢复协议](../outputs/dynamic-human-street-joint-009/resume-10000-expandable/protocol.json)、[状态恢复核验](../outputs/dynamic-human-street-joint-009/resume-10000-expandable/recovery-audit.json)、[最终图像独立核验](../outputs/dynamic-human-street-joint-009/final-image-audit/independent-audit.json)、[视觉检查范围与结论](../outputs/dynamic-human-street-joint-009/final-image-audit/visual-review.json)。

SMPL 下载包通过 ZIP 校验，中性模板安装到 Windows 与 WSL；Python 2.7 到 Python 3 的兼容转换保留全部数值，包括文件中提供的 300 个体形基，当前作者模型使用前 10 个。模板包含 6,890 个顶点、13,776 个三角面与 24 个关节。几何叠加仅检查初始姿态，不是训练后的颜色重建；仍需校正可见偏移。[模板安装核查](../outputs/dynamic-human-smpl-010/template-installation.json)、[数值保真的兼容转换](../outputs/dynamic-human-smpl-010/template-conversion.json)、[100 帧网格检查](../outputs/dynamic-human-smpl-010/mesh-projection-audit.json)、[七组实际投影视觉检查](../outputs/dynamic-human-smpl-010/visual-review.json)。模板文件仅保存在本地，不作为网站下载资源。

完整 156 帧、7 路相机的 SMPL 联合模型已完成正式 30,000 次更新；两支最终 210 张完整图和 870 个人物框均通过独立误差复算。主目标的 139 帧标注由 100 帧人体关节表示和 39 帧缺姿态时的可变形表示覆盖，二者不在同一时刻重复出现；这只证明表示没有漏帧，尚未证明切换连续或人体清晰。作者姿态先验包含留出时刻，当前训练 RGB 与点云初始化仍只使用训练时刻；该设置不是未知姿态或未来运动预测。见[完整人体分支运行检查](../outputs/dynamic-human-street-smpl-joint-011/preflight/completion.json)、[初始化与时间覆盖](../outputs/dynamic-human-street-smpl-joint-011/preflight/initialization.json)、[正式对照协议](../outputs/dynamic-human-street-smpl-joint-011/articulated/protocol.json)。

人体关节模型的中途结果再次说明：表面颜色接近观测，不代表关节和轮廓已经对齐。同为 10,000 次更新的三组预定视图仍未恢复清晰衣物；独立恢复的实际人体几何在采样训练帧中也存在偏移，因而不能把所有问题归为留出时刻插值。训练帧中的二维关节检测还会在多人重叠时混入旁人的手臂，预测置信度不能直接代替正确对应关系。参见[同预算真实对照](../outputs/dynamic-human-street-smpl-joint-011/paired-updates-10000-audit/frame130-camera4-person79-comparison.png)、[实际人体几何检查](../outputs/dynamic-human-street-smpl-joint-011/geometry-03000-cpu/visual-review.json)及[遮挡下的关节对应检查](../outputs/dynamic-human-training-alignment-013/keypoint-visual-review.json)。


街景人体对照的主目标全部 13 组最终裁图均已查看，最差全图也单独检查；两支均未通过清晰度验收。[最终配对核验](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/independent-audit.json)、[视觉结论](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/visual-review.json)。四段时间窗口另保存两支共 160 张渲染、152 组相邻变化及 8 段对照视频。视频按原拍摄约 10 Hz 播放，明确区分训练与留出时刻；矩形区域含背景和遮挡者，相邻差分残差不是纯闪烁或光流误差。[侧面连续对照](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/side-occlusion-fixed-union.mp4)、[时间窗口独立核验](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/independent-audit.json)。

训练帧的进一步几何分解显示：移除高斯位置偏移及学习到的蒙皮修正后，保留已优化关节的参考人体仍有头部和腿部偏移；完整高斯中心则出现额外锯齿状边界。该投影仅是诊断，不是衣物网格，也没有场景遮挡合成。下一步应优先检验姿态拟合，再评估表面位移的约束。[训练帧几何分解](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/frame129-camera4-decomposition.png)、[检查范围](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/visual-review.json)。

固定同一个街景和全部高斯属性后，仅优化主目标的训练时刻姿态，可以进一步区分“姿态更新发生”和“姿态拟合有效”。单图随机更新的诊断同时恶化了训练和留出结果，已拒绝采用；改用全部 101 个人物训练区域的平均目标和线搜索，训练误差稳定下降，留出人物框 PSNR 从 19.70 提高到 19.88 dB，SSIM 从 0.580 提高到 0.596。独立模型检查确认其他人物、背景、表面、颜色和留出姿态条目均未变化；但四组近处裁图仍未恢复清晰衣物，尚不能据此证明位姿准确或时间连续。参见[姿态对照与实际图片](../experiments/37_dynamic_human_reconstruction.md#从姿态更新到有效拟合)、[逐项模型核验](../outputs/dynamic-human-street-pose-batch-015/independent-state-audit.json)。

相同目标的长预算验证完成 120 次迭代后，训练人物区域 PSNR 提高至 20.40 dB，留出目标区域提高至 19.94 dB；全部 13 组留出裁图仍未恢复清晰衣物和肢体。部分试探点触及关节变化限制，这不是已经收敛或姿态准确的证据。后续需要检查高斯表面尺度与可见贡献，不能仅靠延长姿态优化解决模糊。[长预算真实结果与检查范围](../experiments/37_dynamic_human_reconstruction.md#延长姿态优化后衣物模糊仍然存在)。

六个街景视图的只读分解进一步确认，目标人物自身的颜色贡献仍有模糊边界；直接把全部目标高斯尺度缩小到一半或四分之一会暴露孔洞与斑点，六个视图误差均恶化。尺度缩小改变了覆盖，不能等同于恢复了表面细节；后续应以颜色拟合为对照，检验受约束的表面与颜色联合拟合。[实际遮挡贡献、尺度干预与核验](../experiments/37_dynamic_human_reconstruction.md#高斯覆盖范围与遮挡合成的实际分解)。

固定姿态后的两组街景对照均已完成：主目标留出区域平均 PSNR 从 19.94 dB 分别提高到仅颜色拟合的 20.10 dB、受约束表面与颜色联合拟合的 20.30 dB。全部 13 组真实裁图逐一检查后，近处衣褶、手提袋和双腿仍不清晰。三种候选另完成 240 张连续渲染、228 组相邻差分的独立复核与 8 段对照视频；抽看的 16 个时刻仍有同类模糊。因此，放开表面属性有助于当前拟合，但尚未完成清晰重建。[共同条件、真实效果与检查范围](../experiments/37_dynamic_human_reconstruction.md#固定姿态后的表面与颜色对照)、[连续运动对照](../experiments/37_dynamic_human_reconstruction.md#表面拟合后的连续运动对照)。

局部表面容量诊断已完成：仅拟合一个近处训练视图时，衣褶与提袋明显恢复；同时拟合两张相邻训练视图也能改善这两张画面，但中间留出帧仍有伪影，其他时刻显著退化。两组均被拒绝作为场景模型替代。全部 210 张留出图、870 个区域和冻结模型状态通过独立核验，实际看图范围为 2 组训练视图及 4 组留出视图。[局部拟合与跨时刻失败的完整对照](../experiments/37_dynamic_human_reconstruction.md#单帧与相邻双帧的表面拟合能力)。

运动区时间对应检查保留 117 份固定匹配轨迹，使用训练机位帧 9、29 估计、帧 49 检查；同初始化的两支取帧对照各完成 6,000 次更新。三个留出机位有升有降，六组人物上半身和三张调整后全图均已查看，未通过连续人物清晰度验收。见[几何对应与取帧训练对照](../experiments/37_dynamic_human_reconstruction.md#运动区域的跨相机时间对应)、[实际训练效果](../outputs/dynamic-human-enerf-time-control-027/comparison/summary.json)。

连续人物的三组同预算对照另完成 225 张原生渲染、450 个人物区域和三段 25 帧视频；固定首、中、末帧的 18 组人物对照及最差退化点前后三帧均已看图。人物节点可无损重组，三个实际渲染逐像素一致，但仍是前景接口，尚未完成 OmniRe 背景联合训练。[连续对照与实际效果](../experiments/37_dynamic_human_reconstruction.md#局部时间对应与连续人物的同预算对照)、[全画面残差](../outputs/dynamic-human-enerf-local-time-029/evaluation/camera06-frame058-offset-index.png)、[人物节点接口说明](../outputs/dynamic-human-enerf-local-time-029/actor-interface/scene-contract.json)。

## 参考资料


- Chen 等，OmniRe，ICLR 2025：[论文 v2](https://arxiv.org/abs/2408.16760v2)、[官方 DriveStudio](https://github.com/ziyc/drivestudio)。
- Luiten 等，Dynamic 3D Gaussians，3DV 2024：[论文](https://arxiv.org/abs/2308.09713)、[官方实现与公开数据入口](https://github.com/JonathonLuiten/Dynamic3DGaussians)。
- Kocabas 等，HUGS Human Gaussian Splats，CVPR 2024：[官方实现](https://github.com/apple-aiml-research/ml-hugs)。
- Argoverse 2：[官方隐私与图像规格说明](https://argoverse.github.io/user-guide/datasets/sensor.html#privacy)。
- Işık 等，HumanRF / ActorsHQ，SIGGRAPH 2023：[论文第 4 节与 Fig. 4](https://arxiv.org/html/2305.06356v1#S4)、[官方分段下载说明](https://github.com/synthesiaresearch/humanrf#data-access-and-download)。
- Cheng 等，DNA-Rendering，ICCV 2023：[官方数据规格与申请条件](https://dna-rendering.github.io/inner-download.html)。
- Lin 等，ENeRF，SIGGRAPH Asia 2022：[室外数据与访问条件](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md)、[作者提供的小型示例](https://github.com/zju3dv/EasyVolcap#examples)。
- Xu 等，4K4D，CVPR 2024：[项目与论文](https://zju3dv.github.io/4k4d/)。ENeRF-Outdoor 数据提供方要求同时引用 [EasyVolcap](https://github.com/zju3dv/EasyVolcap)。

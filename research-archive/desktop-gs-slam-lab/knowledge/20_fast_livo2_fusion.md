---
id: "gslab-k20"
slug: "fast-livo2-fusion-state-estimation"
title: "FAST-LIVO2：融合、标定、同步与漂移机制"
order: 20
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-13"
sources:
  - "https://arxiv.org/pdf/2408.14035v2"
  - "https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf"
  - "https://github.com/hku-mars/FAST-LIVO2/tree/0d2c0346107b75b59934975adec9a6eeeb913c64"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/include/common_lib.h"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/src/LIVMapper.cpp"
  - "https://github.com/hku-mars/FAST-LIVO2/blob/0d2c0346107b75b59934975adec9a6eeeb913c64/launch/mapping_avia.launch"
  - "https://discourse.ros.org/t/ros-noetic-end-of-life-may-31-2025/43160"
---

# FAST-LIVO2：融合、标定、同步与漂移机制

传统三维高斯泼溅（3D Gaussian Splatting，3DGS）假设相机位姿已经给定；进入移动机器人场景后，首先要解决的是：设备此刻在哪里，不同传感器是否在描述同一时刻、同一坐标系中的场景。FAST-LIVO2 采用激光雷达（LiDAR）、惯性测量单元（Inertial Measurement Unit，IMU）和相机共同估计运动，并维护几何与图像块共享的地图。它提供的是定位与地图观测的基础，其原生地图并不是高斯表示。

本文将 Zheng 等人 2024 年论文的方法、公开实验与补充材料改写为连续讲义。论文的状态与测量模型是主线；坐标变换、去畸变、可观测性和评测公式中的补充推导会明确说明约定。公开论文结果与本机实验记录分开，后者保留在文末实验附录。

## 1. 传感器输入与共同状态

![FAST-LIVO2 原论文系统总览：IMU传播、顺序滤波和共享体素地图](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-02.png)
> 图源：Zheng 等（2024），Fig. 2 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

读这幅原图时，先沿三条观测路径看它们各自提供什么信息。IMU 高频测量角速度与比力，用来预测短时间运动；LiDAR 直接测量距离，为地图提供米制三维几何；图像中的亮度分布对像素位置变化敏感，提供几何之外的纹理约束。三者更新同一个状态，而不是各自生成一条轨迹后求平均。

以世界坐标系 W、IMU 坐标系 I 表示运动。论文的状态包括

$$
x=\left(R_{WI},p_{WI},v_W,b_g,b_a,g_W,e\right),\qquad
\delta x=\begin{bmatrix}
\delta\theta^T&\delta p^T&\delta v^T&\delta b_g^T&\delta b_a^T&\delta g^T&\delta e
\end{bmatrix}^T\in\mathbb R^{19}.
$$

旋转属于特殊正交群 $\operatorname{SO}(3)$；$R_{WI}$ 把 IMU 坐标中的向量变到世界坐标。$p_{WI}$ 是 IMU 原点的世界位置，单位 m；$v_W$ 为世界速度，m/s；$b_g$ 为陀螺仪偏置，rad/s；$b_a$ 为加速度计偏置，m/s²；$g_W$ 为重力向量，m/s²。$e$ 是相对于第一帧的逆曝光倍率，无量纲，首帧固定为 $e_0=1$。这十九维是误差状态自由度，旋转虽由矩阵表示，其局部增量只有三维。

本文采用 IMU 旋转的右乘增量和世界位置的加法增量：

$$
R_{WI}^{+}=R_{WI}\exp([\delta\theta]_\times),\qquad
p_{WI}^{+}=p_{WI}+\delta p.
$$

$\delta\theta$ 在 IMU 局部坐标中表达，单位 rad；$\delta p$ 在世界坐标中表达，单位 m。对向量 $a=(a_1,a_2,a_3)^T$，叉乘矩阵定义为

$$
[a]_\times=\begin{bmatrix}
0&-a_3&a_2\\
a_3&0&-a_1\\
-a_2&a_1&0
\end{bmatrix},\qquad [a]_\times b=a\times b.
$$

上面的增量顺序对应论文将曝光放在最后的表达；本站固定版本的官方状态把曝光增量放在位置之后。两者维数相同，组装雅可比或协方差时必须按实际顺序排列。论文称 $e$ 为相对逆曝光时间，固定源码也用它作为乘法亮度参数；没有第一帧绝对曝光和相机响应标定，不能把它直接标成 s⁻¹。传感器外参与时间偏移是预先提供的参数，不在这里的十九维在线状态内。

## 2. IMU 传播：预测运动，同时传播不确定性

![FAST-LIVO2 原论文相机时刻、IMU前向传播和LiDAR逐点反向补偿](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-03.png)
> 图源：Zheng 等（2024），Fig. 3 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

图中细密的 IMU 测量连接相邻图像时刻。外部观测尚未到来时，设备仍在运动，滤波器根据这一段惯性测量向前预测状态。忽略噪声的连续运动模型为

$$
\begin{aligned}
\dot R_{WI}&=R_{WI}[\omega_m-b_g]_\times,\\
\dot p_{WI}&=v_W,\\
\dot v_W&=R_{WI}(a_m-b_a)+g_W.
\end{aligned}
$$

$\omega_m$ 是角速度测量，rad/s；$a_m$ 是比力测量，m/s²。加速度计测到的比力不等于世界系运动加速度：必须减去偏置，旋转到世界，再加上重力。以时间间隔 $\Delta t$（s）内运动近似不变为条件，论文离散模型中的位置项包含

$$
a_W=R_{WI}(a_m-b_a)+g_W,\qquad
p_{WI,k+1}=p_{WI,k}+v_{W,k}\Delta t+\frac12a_W\Delta t^2,\qquad
v_{W,k+1}=v_{W,k}+a_W\Delta t.
$$

这解释了偏置为何能累积为漂移：一个持续的加速度误差先进入速度，再通过积分进入位置。陀螺仪偏置还会影响姿态，使重力补偿方向错误，进一步造成平移误差。固定实现使用相邻 IMU 测量的均值，并结合初始化尺度和区间边界传播；上述关系说明物理机制，不替代完整的离散积分器。

状态预测必须同时传播误差协方差：

$$
P_{k+1}^{-}=F_kP_k^{+}F_k^T+Q_{d,k}.
$$

$P_k^{+}$ 为上次观测更新后的协方差，$P_{k+1}^{-}$ 为当前观测前的预测协方差；$F_k$ 为误差状态传播雅可比，$Q_{d,k}$ 为按采样间隔离散化的过程噪声协方差。不同协方差块具有各自状态单位的乘积，不能直接比较旋转块与位置块的数值大小。传播会建立位置、速度和偏置之间的相关性；因此后续只直接观测位姿的 LiDAR 或图像，也可能间接修正速度和偏置。

## 3. 时间同步、扫描重组与逐点去畸变

![FAST-LIVO2 原论文采集设备和STM32同步触发方案](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-09.png)
> 图源：Zheng 等（2024），Fig. 9 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

论文明确假设传感器时间偏移已知，并通过预先标定或同步获得。图中的实际平台用同步定时器产生触发信号，给相机和 LiDAR 提供共同时间依据。这一层解决采样时间的一致性；算法中的扫描重组则把已经具有时间语义的 LiDAR 点组织到相机更新时刻。两者作用不同。

不同设备时钟的简化关系为

$$
t_{\mathrm{ref}}=a\,t_{\mathrm{sensor}}+b.
$$

两种时间都以 s 表示；$a$ 是无量纲钟速比例，$b$ 是秒制偏移。仅加固定偏移相当于假设 $a=1$，不能自动修复钟速漂移或未知接收延迟。相机时间转换到 IMU 时间轴时，可以定义

$$
t_{C\to I}=t_C+\Delta_{CI}.
$$

$\Delta_{CI}$ 的单位是 s，符号由这条变换方向定义。消息记录时间、主机收到消息的时间和曝光采样时间可能不同；两个 header 数值接近，也不保证硬件采样同时。实际输入路径的修正符号还可能不同，需要依据使用的驱动与固定版本确认。

扫描式 LiDAR 的一帧点云包含许多采样时刻：

$$
t_j=t_{\mathrm{scan}}+\delta t_j.
$$

$t_j$ 是点 $j$ 的采样时刻，$t_{\mathrm{scan}}$ 是扫描参考时刻，$\delta t_j$ 是逐点偏移，均以 s 表示。原字段可能以 ns 或 ms 存储，必须先换算单位并确认扫描参考是起点还是终点。FAST-LIVO2 按相邻相机时刻重组扫描，保留后续时刻的点；IMU 缓存要覆盖这一段运动，才能同时预测状态并补偿点云形变。

令 $T_{AB}$ 将 B 中的齐次点变到 A，LiDAR 到世界的变换为 $T_{WL}(t)=T_{WI}(t)T_{IL}$。对静态场景点，补充推导给出的去畸变关系是

$$
\widetilde p_{L,\mathrm{ref}}=
T_{WL}(t_{\mathrm{ref}})^{-1}T_{WL}(t_j)\widetilde p_{L,j}.
$$

$\widetilde p$ 的前三维为米，末项为 1。该式先把采样时刻的点放入世界，再转到统一参考时刻的 LiDAR 坐标；论文通过前向传播得到运动先验，再对点做反向运动补偿。它依赖正确的逐点时间、外参和运动估计，且假设物体静止。动态物体自身的运动不能由机体去畸变消除。

小时间误差 $\delta\tau$ 会引起近似运动和像素误差：

$$
\delta p\approx v\,\delta\tau,\qquad
\delta\theta\approx\omega\,\delta\tau,\qquad
\delta u\approx\dot u\,\delta\tau.
$$

$v$ 为 m/s，$\omega$ 为 rad/s，$\dot u$ 为 px/s，对应结果分别为 m、rad 和 px。这是局部一阶近似：同样的时差在快速运动中更严重，在静止片段却可能不易发现。曝光倍率属于光度模型，不能用于修正采样时差。

## 4. 坐标链与标定：参数要能解释独立观测

![FAST-LIVO2 原论文刚性安装的LiDAR、相机和IMU采集平台](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-09.png)
> 图源：Zheng 等（2024），Fig. 9 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

图中（a）的三个传感器刚性连接，却不具有同一个坐标原点和坐标轴；（b）说明采样时间的一致性，还需要独立的触发与时钟关系。空间标定要确定这些固定关系；时间标定要确定前一节的观测时刻。外参合法、程序能运行和真实标定可靠，是三个不同层次。

列向量变换约定完整写为

$$
p_A=R_{AB}p_B+t_{AB},\qquad
T_{AB}=\begin{bmatrix}
R_{AB}&t_{AB}\\
0^T&1
\end{bmatrix},\qquad
T_{AC}=T_{AB}T_{BC}.
$$

$R_{AB}$ 无量纲，把 B 向量转到 A；$t_{AB}$ 是 B 原点在 A 中的位置，单位 m。下标方向为 B → A。逆变换必须同时处理旋转和平移：

$$
R_{BA}=R_{AB}^T,\qquad
t_{BA}=-R_{AB}^Tt_{AB}.
$$

记 L 为 LiDAR、C 为相机、I 为主 IMU，则固定实现中的完整空间关系为

$$
T_{CI}=T_{CL}T_{IL}^{-1},\qquad
T_{CW}=T_{CI}T_{WI}^{-1},\qquad
p_W=R_{WI}(R_{IL}p_L+t_{IL})+p_{WI}.
$$

第一式由 LiDAR–相机与 LiDAR–IMU 外参得到 IMU–相机关系；第二式把世界点投影前转换到当前相机；第三式把激光点放入世界。矩阵乘法顺序不可互换，只把平移取负也不是正确的刚体求逆。

相机内参决定相机坐标如何变成像素。采用相机 $X$ 向右、$Y$ 向下、$Z$ 向前的无畸变针孔模型：

$$
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix},\qquad
\pi(X,Y,Z)=\begin{bmatrix}
f_xX/Z+c_x\\
f_yY/Z+c_y
\end{bmatrix},\quad Z>0.
$$

$f_x,f_y$ 是像素焦距，$c_x,c_y$ 是主点，单位均为 px。光轴是通过相机投影中心、垂直于像平面的直线，在这个模型中与相机 $Z$ 轴重合；主点是其与像平面交点对应的像素。$Z$ 是轴向深度，单位 m，并不等于点到相机中心的欧氏距离。真实镜头还可能需要畸变模型，鱼眼投影也不能直接用针孔公式替代。

如果图像经过缩放和裁剪，内参要按实际像素变换更新：

$$
\begin{aligned}
u'&=s_xu+d_x,&f_x'&=s_xf_x,&c_x'&=s_xc_x+d_x,\\
v'&=s_yv+d_y,&f_y'&=s_yf_y,&c_y'&=s_yc_y+d_y.
\end{aligned}
$$

$s_x,s_y$ 无量纲，$d_x,d_y$ 为 px；像素中心和重采样约定也应一致。否则正确的镜头参数会因图像处理而失配。

标定能否可靠，取决于数据是否区分参数。靶标应覆盖图像中心与边缘，并改变姿态和距离；相机–IMU 外参需要足够的多轴转动和平移激励；时间偏移需要随时间变化且可以对应的运动信号。长期静止或单一运动，容易让焦距、外参、时差与位姿互相补偿。应在独立观测或预先划定的留出数据上检查重投影和空间对齐，而不是只看参与拟合的数据残差。原论文将空间外参和时间偏移视作前置条件；读取它们不表示本机已经重新完成全部物理标定。

## 5. LiDAR 残差：平面方向与测量不确定性

![FAST-LIVO2 原论文激光束发散三维模型及真实扫描点不确定性](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-06.png)
> 图源：Zheng 等（2024），Fig. 6 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

图中红色激光束在远处形成具有面积的光斑，斜入射时覆盖的表面范围更大。下方真实扫描表明，考虑束发散后，地面等大入射角区域的测量不确定性增加。因此，几何更新不仅要问点距平面多远，还要问这一个距离有多可靠。

FAST-LIVO2 的自适应体素地图保存局部平面中心 $q_W$、单位法向 $n_W$ 及其不确定性。去畸变后的点进入世界后，点到平面残差为

$$
p_I=R_{IL}p_L+t_{IL},\qquad
p_W=R_{WI}p_I+p_{WI},\qquad
r_l=n_W^T(p_W-q_W).
$$

点和中心单位为 m，$r_l$ 单位为 m。系统直接利用原始点建立这些约束，仍会降采样与筛选，并非不加选择地使用所有返回点。在本文右旋转增量与世界平移增量下，固定匹配平面的局部位姿雅可比为

$$
H_{l,\mathrm{pose}}=
\begin{bmatrix}
-n_W^TR_{WI}[p_I]_\times&n_W^T
\end{bmatrix}.
$$

旋转列单位为 m/rad，平移列无量纲。一个残差只有一个标量约束；同一面墙上的点再多，沿墙平移仍可能几乎不改变距离。这是几何方向退化，不能仅靠点数解决。

为说明权重来源，在固定状态附近对点与平面误差做一阶传播，残差方差可近似写为

$$
\begin{aligned}
J_{nq}&=\begin{bmatrix}(p_W-q_W)^T&-n_W^T\end{bmatrix},\\
\sigma_{r_l}^2&\approx
n_W^TR_{WL}\Sigma_{p_L}R_{WL}^Tn_W+
J_{nq}\Sigma_{nq}J_{nq}^T,\qquad R_{WL}=R_{WI}R_{IL}.
\end{aligned}
$$

$\Sigma_{p_L}$ 是激光点的坐标测量协方差，单位 m²；$\Sigma_{nq}$ 联合描述法向与平面中心误差，结果 $\sigma_{r_l}^2$ 为 m²。单位法向的扰动受切平面约束，该式采用三维嵌入表示。这里忽略点误差与平面误差的交叉相关，仅用于解释论文的不确定性加权思想；地图平面不确定性还包含历史点与历史位姿的不确定性，不能把当前状态先验重复算成一份独立测量噪声。

论文同时考虑飞行时间测距误差、方向误差与激光束发散。默认根体素边长为 $0.5\,\mathrm m$，八叉树最大层级为 3；LiDAR 测距与角度参数随传感器配置改变。根体素只是局部地图组织尺度，不代表点云精度为 0.5 m。斜入射点权重降低也不意味着它们全部无用，而是让滤波器避免把不可靠距离当成强约束。

## 6. 视觉残差：图像块、平面先验与曝光

![FAST-LIVO2 原论文参考图像块与当前图像块的平面几何关系](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-05.png)
> 图源：Zheng 等（2024），Fig. 5 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

图像块（image patch）是地图点周围的一小片像素。相机移动后，同一表面会产生缩放、旋转和形变，不能总把参考方块原样搬到当前图像。图中的三维平面和两个观察位置说明：LiDAR 不只提供图像块中心的深度，还能提供局部表面的朝向。

对参考相机 R 中的局部平面，设单位法向为 $n_R$，平面方程为 $n_R^Tp_R=d_R$，$d_R$ 以 m 表示。当前相机 C 与参考相机的变换为 $p_C=R_{CR}p_R+t_{CR}$。在针孔模型下，平面诱导的单应关系为

$$
\widetilde u_C\sim
K_C\left(R_{CR}+\frac{t_{CR}n_R^T}{d_R}\right)K_R^{-1}\widetilde u_R.
$$

$\widetilde u$ 是齐次像素坐标，$\sim$ 表示齐次比例等价；正号由前述平面方程和变换方向决定。该式是用于解释论文平面先验的标准几何推导。FAST-LIVO2 在小图像块附近采用局部仿射变换，并支持实际相机模型；并非把任意鱼眼投影也写成一个固定内参矩阵。

![FAST-LIVO2 补充材料中常量深度与平面先验的实际图像块投影对比](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-s01.png)
> 图源：Zheng 等（2024），Fig. S1 · [原论文补充材料](https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf) · 裁图

这组真实投影把机制表现出来：倾斜表面上的不同像素具有不同深度，常量深度假设容易产生块间错位；平面先验能使文字、线条等结构更连贯。它同时受位姿和图像块几何影响，因此一幅更清晰的投影不能独立证明全部轨迹误差都更小。

地图点先经过完整坐标链进入当前相机：

$$
p_C=R_{CI}R_{WI}^T(p_W-p_{WI})+t_{CI},\qquad u_C=\pi(p_C).
$$

对图像块像素偏移 $\Delta u_j$，光度残差为

$$
r_{c,ij}=e_C I_C(u_{C,i}+\Delta u_j)
-e_R I_R(u_{R,i}+A_{R\leftarrow C,i}\Delta u_j).
$$

$I_C,I_R$ 是当前和参考图像的灰度采样；$A_{R\leftarrow C,i}$ 把当前图像块偏移变到参考图像块偏移；$e_C,e_R$ 是各帧相对逆曝光倍率。像素坐标为 px，残差是灰度值，不能与米制点到平面残差直接比较大小。实际加权需要各自的噪声模型。

曝光倍率的作用是解释整体乘法亮度变化。理想线性相机且增益固定时，可理解为 $e_k=T_0/T_k$，其中 $T_k$ 为第 k 帧的曝光时间、单位 s；若相机增益也变化，倍率会受到共同影响。未知响应曲线、饱和、渐晕、局部阴影和滚动快门，都超出了一个全局倍率的表达能力。

![FAST-LIVO2 补充材料中估计曝光与参考曝光的逐时间对照](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-s05.png)
> 图源：Zheng 等（2024），Fig. S5 · [原论文补充材料](https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf) · 裁图

图中的 ms 曲线来自作者使用相机 API 记录的曝光，并利用第一帧曝光把相对估计换回绝对时间；不是仅凭一个倍率就得到绝对毫秒。补充实验还包含对固定曝光图像施加正弦亮度因子的受控验证。曝光能减轻光照变化带来的假运动，不能替代空间标定或时间同步。

在本文 IMU 右旋转增量约定下，固定世界点的相机坐标导数为

$$
p_I=R_{WI}^T(p_W-p_{WI}),\qquad
\frac{\partial p_C}{\partial(\delta\theta,\delta p)}=
\begin{bmatrix}R_{CI}[p_I]_\times&-R_{CI}R_{WI}^T\end{bmatrix}.
$$

于是固定参考采样与局部支持时，当前图像梯度分支的链式关系是

$$
J_{c,\mathrm{pose}}=
e_C\nabla I_C(u_C)D\pi(p_C)
\begin{bmatrix}R_{CI}[p_I]_\times&-R_{CI}R_{WI}^T\end{bmatrix},\qquad
J_e=I_C(u_C).
$$

图像梯度的单位为灰度/px；位姿雅可比分别为灰度/rad 与灰度/m。原论文采用逆向组合（inverse compositional）来复用参考图像块梯度；固定源码还提供当前图像梯度分支。两种分支的线性化不能混写。逆向组合、逆向采样和逆深度是不同概念；这里的三维点来自 LiDAR 地图，未把逆深度增加为十九维在线状态。

## 7. 视觉支持：可见性与参考图像块质量

![FAST-LIVO2 原论文遮挡和深度边界外点，以及真实图像中的保留与拒绝点](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-08.png)
> 图源：Zheng 等（2024），Fig. 8 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

一块纹理只有来自同一可见表面，才适合形成上一节的残差。图中绿色点被保留，红色点被拒绝；遮挡点看不到原来那块表面，深度边界附近的方块可能同时包含前后两个物体。即使这些区域梯度很强，也可能产生强而错误的约束。

FAST-LIVO2 先查询当前 LiDAR 扫描和上一帧可见点对应的地图体素。相机视场未被 LiDAR 覆盖，或者 LiDAR 进入近距离盲区时，再按需沿空缺图像网格的视线查询历史地图。它召回已有视觉地图点，并不凭空测得新的深度。

![FAST-LIVO2 补充材料中狭窄走廊的真实地图和射线查询召回的图像点](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-s04.png)
> 图源：Zheng 等（2024），Fig. S4 · [原论文补充材料](https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf) · 裁图

图中走廊约 1.9 m 宽，转向近墙后当前 LiDAR 返回点有限。蓝色点是射线查询召回的历史地图点，黄色点来自体素查询，红色点来自当前扫描。这解释了为什么“当前点少”不等于“完全没有可用图像约束”，也说明这一机制依赖此前已经建立且仍然可信的地图。

参考图像块也需要更新。视角过斜会降低表面纹理的有效分辨率，动态遮挡会污染外观，仅挑选与当前视角最接近的一帧还可能使约束过弱。论文综合图像块间的归一化互相关（Normalized Cross-Correlation，NCC）与观察方向，偏向选择外观一致、正对表面且保留细节的参考图像块。

![FAST-LIVO2 补充材料真实点云中同一表面的多视角图像块与参考选择](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-s03.png)
> 图源：Zheng 等（2024），Fig. S3 · [原论文补充材料](https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf) · 裁图

右侧红框标出所选参考图像块；为显示细节，图中图像块放大到 $40\times40$ px，并非默认对齐窗口大小。论文默认图像对齐使用 $8\times8$ px 图像块和三层金字塔，法向细化使用 $11\times11$ px。法向细化利用不同图像块的光度一致性修正 LiDAR 平面先验；默认基准配置关闭这一可选模块，并非所有结果都来自细化后的法向。

## 8. 顺序融合：后验、线性化与互补方向

![FAST-LIVO2 补充材料中三种融合更新策略的实际大尺度航空地图对比](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-s06.png)
> 图源：Zheng 等（2024），Fig. S6 · [原论文补充材料](https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf) · 裁图

IMU 传播先给出状态先验；同一目标时刻，LiDAR 先修正几何位姿，图像再从修正后的状态与协方差继续。若给定状态后两类测量噪声条件独立，精确贝叶斯后验满足

$$
p(x\mid y_l,y_c)\propto
p(y_c\mid x)\,p(y_l\mid x)\,p(x),\qquad
p(x\mid y_l)\propto p(y_l\mid x)p(x).
$$

这说明顺序更新可以表达联合融合。但非线性系统使用有限迭代、重新匹配、图像金字塔和线性化近似，实际顺序会影响收敛与计算量；共享地图误差和像素相关性也需要考虑。不能把这个理论分解理解为所有近似实现必然数值相同。

对同一线性化坐标中的增量 $\delta x$，一次局部估计可理解为

$$
\delta x^*=\arg\min_{\delta x}
\left[
\|\delta x-\delta x_{\mathrm{prior}}\|_{(P^-)^{-1}}^2+
\|r+H\delta x\|_{R_n^{-1}}^2
\right],\qquad \|a\|_B^2=a^TBa.
$$

$r$ 为当前残差，$H$ 为观测雅可比，$R_n$ 为相应残差噪声协方差。先验和残差经过各自协方差加权后才可进入同一目标；旋转不能当矩阵逐元素欧氏相减。反复线性化改善同一批观测的求解，不应把每次内部迭代当作新的独立观测，重复累计信息。

互补性要看方向，而非只看观测数量。固定地图和支持集合的局部信息近似为

$$
\Lambda_{\mathrm{local}}=(P^-)^{-1}
+H_l^TR_l^{-1}H_l+H_c^TR_c^{-1}H_c.
$$

当某个方向 $d$ 满足 $H_ld\approx0$ 时，LiDAR 对它较弱；若 $H_cd$ 在可信图像支持上足够大，视觉才可能补上这个方向。一面墙缺少沿墙平移的几何约束，但墙面纹理可能产生有用像移；若图像也缺纹理、只有平行边缘或被遮挡，两类传感器可能同时缺信息。信息矩阵可逆还可能来自强先验，不能据此宣布当前测量本身完全可观测。

曝光会与位姿耦合。允许曝光变量解释亮度变化时，边缘化曝光后的局部位姿信息为

$$
\Lambda_{\mathrm{pose,marginal}}
=\Lambda_{pp}-\Lambda_{pe}\Lambda_{ee}^{-1}\Lambda_{ep}.
$$

该式要求曝光信息块 $\Lambda_{ee}$ 可逆，是解释耦合的舒尔补推导，并非宣称官方另有一个舒尔补模块。比较不同单位状态的弱方向，应先白化噪声并声明工程尺度，不能直接把原始特征值当作通用分数。

上方原图来自 MARS-LVIG 的 AMvalley03：约 600 s、飞行速度约 12 m/s 的大尺度航空序列。补充材料报告，异步标准更新、同步标准更新、同步顺序更新的位置 APE RMSE 分别为 3.12 m、2.45 m、0.68 m，平均处理时间约为 27.6 ms、49.9 ms、23.1 ms，使用论文的 Intel i7-10700K 平台。LiDAR 最多 5 次迭代；顺序视觉在三层金字塔每层最多 5 次迭代；同步标准更新每层最多 3 次，并额外调节异质残差尺度。图中道路分层体现不同融合策略的运行差异；因为更新预算和具体线性化策略也不同，不能把性能差异全部归因于一个抽象的“同步”变量。

### 多尺度残差与实际输出状态

图像金字塔各层对应不同采样尺度，因而应在同一层内比较光度目标。令 $\ell$ 为层编号，$j$ 为迭代编号，$N_\ell$ 为有效像素残差数，则灰度残差的均方值为

$$
E_\ell(\mathbf x^{(j)})
=\frac{1}{N_\ell}\sum_{n=1}^{N_\ell}
r_{\ell,n}(\mathbf x^{(j)})^2,
\qquad N_\ell>0.
$$

$E_\ell$ 的单位是灰度值的平方。不同层的模糊程度、采样位置和参考图像块不同，跨层数值不能直接当作同一目标的下降曲线。它也不等于包含状态先验的完整后验目标，更不等于米制位置误差。

如果先评价 $E_\ell(\mathbf x^{(j)})$，再计算增量 $\delta\mathbf x^{(j)}$ 并返回

$$
\mathbf x^{(j+1)}=\mathbf x^{(j)}\boxplus\delta\mathbf x^{(j)},
$$

则已记录的评分属于更新前状态。$\boxplus$ 表示按状态定义施加增量，其中旋转需要在旋转群上更新。因增量很小或迭代次数耗尽而退出，不会自动获得返回状态的重新评分。分析“更新被接受”时，应明确指当前状态通过残差比较、一次增量被施加，还是最终输出经过重新评价；三者不能互相替代。

## 9. 从公开实验理解模块作用与适用边界

![FAST-LIVO2 原论文单墙几何退化场景中的真实彩色点云对比](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-10.png)
> 图源：Zheng 等（2024），Fig. 10 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

这些实验让传感器长期朝向一面墙，沿墙运动。点到平面残差缺少沿墙方向约束；墙上的文字、图案和屏幕纹理则可能让视觉识别沿墙运动。原图比较的是作者的真实重建结果，既应看墙面几何是否变形，也应看同一结构的纹理是否清晰。彩色纹理清楚支持像素对齐质量，但不等于已经获得整段真值轨迹。

论文在 NTU-VIRAL、Hilti’22 和 Hilti’23 的 25 条序列上报告绝对平移 RMSE。下面选取 Table II 中能解释模块机制的平均值，单位 m；这些是作者公开论文结果，不是本机显卡实测。

| 论文配置 | 25 序列平均平移 RMSE | 可支持的机制解释 |
|---|---:|---|
| 仅使用本文的 LIO 子系统 | 0.097 m | 几何和惯性提供基础，但部分方向需要其他观测 |
| 不估计曝光 | 0.051 m | 亮度变化会影响直接法，收益随场景而变 |
| 不更新参考图像块 | 0.089 m | 参考外观与观察方向质量对光度约束很重要 |
| 默认完整 FAST-LIVO2 | 0.045 m | 启用曝光和参考更新，关闭法向细化 |
| 完整方法并启用法向细化 | 0.044 m | 此组平均值略改善，不能证明所有场景均受益 |

论文 Table II 的默认完整方法平均值为 0.045 m；正文还描述了启用法向细化后的 0.044 m 结果。引用时应保留配置差别，不将两者混成同一次默认运行。逐序列结果同样重要：Hilti’22 的 Outside Building 上，FAST-LIO2 为 0.028 m、默认 FAST-LIVO2 为 0.035 m；Hilti’23 的 Large room (dark) 上，两者为 0.040 m 和 0.046 m。图像质量不足时，增加视觉不保证优于几何惯性系统。

作者采用 Intel i7-10700K 与 32 GB RAM 进行桌面评测；另测 Qualcomm Kryo585 与 8 GB RAM 的 ARM 平台。论文全组平均桌面处理时间为 30.03 ms，其中 LiDAR 17.13 ms、图像 12.90 ms；ARM 平均为 78.44 ms。这说明论文配置在对应输入下支持 10 Hz 实时处理，不可直接换算为本机吞吐，也不代表任意图像分辨率、点云规模和参数都满足实时要求。

理解消融要保留观测边界。参考更新的效果同时涉及纹理、视角和异常外观；平面先验实验同时影响图像块形变与位姿收敛；法向细化在非平面区域或低质量图像上也可能受到模型失配影响。状态确实被更新、光度目标降低和定位误差改善，需要分别验证，不能互相替代。

## 10. 漂移评测：时间、物理点与 ATE、RPE

![FAST-LIVO2 原论文复杂退化场景的真实地图、轨迹及局部扫描](../outputs/slam-system-paper-course-001/figures/fast-livo2-fig-11.png)
> 图源：Zheng 等（2024），Fig. 11 · [原论文](https://arxiv.org/pdf/2408.14035v2) · 裁图

图中复杂走廊、暗矿道与弱结构区域展示了作者的鲁棒性实验。返回起点的偏差可以描述末端一致性，但它只检查首尾，不提供途中每一时刻的误差，也不证明系统实现了回环优化。FAST-LIVO2 的这一方法是局部里程计与地图更新，不能将小首尾误差称为全局漂移闭合。

评测首先需要把估计和参考变成同一个物理点。算法状态描述 IMU 原点；全站仪可能跟踪设备上的棱镜。若 M 为被测标记，$\ell_{IM}$ 为 IMU 原点指向 M、在 IMU 坐标表示的固定杆臂，则

$$
\widehat p_{WM}(t)=\widehat p_{WI}(t)+\widehat R_{WI}(t)\ell_{IM}.
$$

位置与杆臂单位为 m。杆臂要随姿态旋转；直接在世界系加固定向量，会在设备转动时产生假误差。采用前述右旋转增量，其一阶误差为

$$
\delta p_{WM}\approx\delta p_{WI}
-R_{WI}[\ell_{IM}]_\times\delta\theta.
$$

因此标记点位置误差还混合了姿态误差。杆臂属于具体设备，不能直接跨数据集复用；已经转换到标记点的轨迹也不能再补偿一次。

随后在共同采样时刻关联估计与参考。位置可在可靠两侧样本之间线性插值，旋转应在 $\operatorname{SO}(3)$ 上插值；禁止外推或跨越预定大缺口。比较 LIO 与 LIVO 时，应使用同一查询网格和支持交集，同时保留各自覆盖与失败区间。仅有位置参考，或姿态列属于格式占位时，只能评估有真实测量语义的量。

世界原点与朝向的固定差异，可用尺度固定的刚体对齐消除：

$$
(R^*,t^*)=\arg\min_{R\in\operatorname{SO}(3),\,t\in\mathbb R^3}
\sum_i\|R\widehat p_i+t-p_i\|_2^2,\qquad
\widehat p_i^{\mathrm{aligned}}=R^*\widehat p_i+t^*.
$$

旋转无量纲，位置和平移为 m，尺度保持 1。该对齐不修复局部漂移，也不等于算法做了回环。每个短片段分别对齐会反复消除累计误差；拟合自由尺度还会改变米制系统的评测问题。

位置绝对轨迹误差（Absolute Trajectory Error，ATE）的 RMSE 定义为

$$
\operatorname{ATE}_{\mathrm{position,RMSE}}=
\sqrt{\frac1N\sum_{i=1}^N
\|\widehat p_i^{\mathrm{aligned}}-p_i\|_2^2}.
$$

$N$ 为有效关联位置数量，结果为 m。它描述整段位置一致性；全序列最佳对齐可能把偏差重新分配到前后阶段，所以还应观察随时间的误差和预先确定间隔的局部运动误差。

对具有真实六自由度参考的同一机体，固定时间或路程间隔的一对位姿 $i,j$ 的相对位姿误差（Relative Pose Error，RPE）为

$$
\begin{aligned}
\Delta T_{\mathrm{ref}}&=T_{\mathrm{ref},i}^{-1}T_{\mathrm{ref},j},\\
\Delta T_{\mathrm{est}}&=\widehat T_{\mathrm{est},i}^{-1}\widehat T_{\mathrm{est},j},\\
E_{ij}&=\Delta T_{\mathrm{ref}}^{-1}\Delta T_{\mathrm{est}}
=\begin{bmatrix}
R_E&t_E\\
0^T&1
\end{bmatrix},\\
e_t&=\|t_E\|_2,\qquad
e_R=\|\operatorname{Log}(R_E)^\vee\|_2.
\end{aligned}
$$

$e_t$ 单位为 m，$e_R$ 单位为 rad，可显式换算为度。$\operatorname{Log}(R_E)^\vee$ 是旋转对数对应的三维旋转向量。这里要求参考与估计的机体轴一致；只有位置真值时，不能用占位姿态计算完整 RPE。输出频率不同的“相邻帧”也不是同一评测间隔。

固定实现还需区分输出阶段：主轨迹描述 IMU 机体，在当前 LiDAR 更新后、当前视觉更新前写出；图像位姿在视觉更新后保存。主轨迹仍继承此前视觉修正的历史影响。ROS 发布时间也不必等于传感器采样时间。比较两个独立运行的轨迹差，不能将其当成某一次视觉更新的即时贡献。

### 同一时刻视觉更新前后的误差

评估一次视觉修正时，先固定同一机体、时间戳和参考坐标。记同一运行在视觉更新前后的 IMU 位置分别为 $\widehat p_i^-$ 与 $\widehat p_i^+$，共同参考为 $p_i^{\mathrm{ref}}$。保持一组既定刚体对齐 $R_A,t_A$，定义

$$
e_i^-=\|R_A\widehat p_i^-+t_A-p_i^{\mathrm{ref}}\|_2,
\qquad
e_i^+=\|R_A\widehat p_i^++t_A-p_i^{\mathrm{ref}}\|_2,
\qquad
\Delta e_i=e_i^+-e_i^-.
$$

位置及误差单位为米，$\Delta e_i<0$ 表示该参考时刻的更新后位置更接近参考。两组状态分别重新对齐，会把输出阶段差异和坐标拟合差异混在一起；固定对齐使这个诊断的比较条件一致。参考时刻落在两个状态样本之间时，两阶段必须使用相同插值区间与权重，只比较二者都有支持的时间范围。

状态修正量 $\|\widehat p_i^+-\widehat p_i^-\|_2$ 与 $\Delta e_i$ 不是同一个量：前者说明更新有多大，后者说明相对给定参考的误差怎样改变。两阶段都继承了此前视觉修正的历史，所以这项同一运行内的比较也不等同于关闭视觉后的独立运行。光度残差下降、状态发生变化、参考误差下降，应分别建立证据。

### 与高斯建图的连接

连接相机重建后端时，先得到相机到世界变换 $T_{WC}=T_{WI}T_{IC}$，再按后端要求提供相机到世界或世界到相机的位姿、内参、采样时间和米制尺度。LiDAR 几何可以作为深度或初始化依据，但把彩色点云转换成高斯地图仍需要新的表示、渲染与优化设计。定位误差会移动地图点并影响重投影，错误的地图又会反过来影响直接跟踪；传感器融合与高斯表示解决相互连接的不同问题。

### 参考与阅读约定

- [FAST-LIVO2 原论文](https://arxiv.org/pdf/2408.14035v2)：§IV–VII 为共同状态、顺序更新、地图和测量模型；§VIII–IX 为数据集、配置与公开实验。
- [官方补充材料](https://raw.githubusercontent.com/hku-mars/FAST-LIVO2/main/Supplementary/LIVO2_supplementary.pdf)：§I-A–I-E 对应平面图像块、参考更新、射线查询、曝光与更新策略实验。
- [本站固定官方版本](https://github.com/hku-mars/FAST-LIVO2/tree/0d2c0346107b75b59934975adec9a6eeeb913c64)：用于确认状态顺序、当前梯度分支、时间组织与输出阶段；实现接口保留在实验附录。
- [NTU-VIRAL 传感器说明](https://ntu-aris.github.io/ntu_viral_dataset/sensors_and_usage.html)与[评价说明](https://ntu-aris.github.io/ntu_viral_dataset/evaluation_tutorial.html)：用于确认棱镜位置、杆臂与参考时间。论文的官方 Hilti 评测结果不能自动赋予其他公开开发参考同样的精度。

## 实验附录

### 独立回放中的光度残差与回退

隔离编译仅增加观测输出的版本，按原参数完成 Hilti exp14 全包回放；原环境与算法公式保留。737 个视觉帧中，首帧初始化视觉地图，736 帧产生视觉约束，记录到 9,833 次迭代：8,301 次进入接受分支，1,532 次进入拒绝并回退分支。每次回退的位置均与保存的已接受位置一致；736 个有约束帧的起止位置均与独立状态日志匹配。

最细层有 596 帧的最后一个已评价且接受的状态，光度均方误差低于该层初值。但 216 帧因小增量、141 帧因预算耗尽退出，返回新状态未在该层再次评分。因此这些残差不能全部当成最终输出位姿的残差，也不能证明定位改善。该本地配置使用四层金字塔、每层最多五次迭代、八像素边长图像块以及当前图像梯度分支；它与上文论文航空实验的三层配置分别记录。见[视觉迭代的残差与回退](../experiments/34_three_track_mechanism_followup.md#fast-livo2-视觉迭代的残差与回退)。

### 更新后状态的共同参考对照

从已有 Hilti exp14 日志恢复 737 组视觉更新前后状态，固定原 LIVO 对齐，在两阶段都有插值支持的 686 个参考时刻比较。更新前后位置误差 RMSE 分别为 4.1307、4.1463 厘米；误差变化均值为 +0.222 毫米。原始主轨迹的误差数组同时被精确复算，最后一个缺少视觉更新后支持的参考点明确排除。

仅考虑状态文本坐标的六位有效数字舍入，误差变化均值仍落在 +0.164 至 +0.280 毫米内。因此在这段记录中，改用真正的视觉更新后状态没有扭转精度未改善的结果。该区间不是 GT 精度或统计置信区间，也不能用于断言视觉模块普遍有害；外参、时差、观测退化的因果区分仍未完成。详见[三条主线的机制验证](../experiments/34_three_track_mechanism_followup.md)。

本机已完成无 GT 的 Bright Screen 两模式功能回放，以及 Hilti exp14 的两模式开发参考轨迹对照；后者具有定位误差结果，但不能证明全部标定或物理同步正确。环境兼容、回放记录及详细成绩见[本机融合回放](../experiments/2026-10-10-fast-livo2-local-replay.md)、[Hilti 定位对照](../experiments/31_fast_livo2_hilti_gt.md)和[视觉净状态修正诊断](../experiments/32_slam_drift_mechanism.md)。


### 官方状态字段对照

| 变量 | 物理含义与约定 | 主要信息来源 |
|---|---|---|
| `rot_end`、`δθ` | IMU 到世界的旋转；局部旋转增量用 rad | IMU 传播，几何和光度观测修正 |
| `pos_end`、`δp` | IMU 原点在世界中的位置，m | 速度积分，几何和光度观测修正 |
| `inv_expo_time`、`δe` | 乘法亮度参数，无绝对曝光时间尺度 | 启用曝光估计时的光度观测 |
| `vel_end`、`δv` | 世界系速度，m/s | IMU 传播和状态交叉协方差 |
| `bias_g`、`δbg` | 陀螺 bias，rad/s | 运动历史与外部观测的联合约束 |
| `bias_a`、`δba` | 加速度 bias，m/s² | 运动历史与外部观测的联合约束 |
| `gravity`、`δg` | 世界系重力向量，m/s² | 初始化与启用时的后续估计 |

### 固定实现的模块与输出接口

## 三个模块如何在代码中连接

```text
时间缓存组织同一目标时刻的测量
  → IMU 状态与协方差传播、点云去畸变
  → LiDAR 几何更新与局部地图维护
  → 对应图像的视觉更新
  → 修正后的共同状态进入后续传播
```

| 代码位置 | 应读懂的连接 |
|---|---|
| `include/common_lib.h`：`StatesGroup` | 状态顺序、单位、旋转增量与协方差 |
| `src/LIVMapper.cpp`：`sync_packages()` | LIVO 按图像目标时刻切分点云、保留未来点，并组织 LIO／VIO 阶段 |
| `src/LIVMapper.cpp`：`processImu()` | 调用 `Process2()`，保存传播先验，并将状态交给几何更新 |
| `src/IMU_Processing.cpp`：`Process2()`、`UndistortPcl()` | 初始化、IMU 传播、逐点运动补偿 |
| `src/voxel_map.cpp`：`StateEstimation()` | 点到面残差、几何权重、迭代状态与协方差更新 |
| `src/LIVMapper.cpp`：`handleLIO()` | 将几何更新结果写回 `_state`，维护地图并保存主轨迹 |
| `src/LIVMapper.cpp`：`handleVIO()` 与 `src/vio.cpp` | `vio_manager->state=&_state`，图像直接更新同一状态 |

`run()` 不会无条件按三行调用顺序处理任意消息；时间缓存和 `lio_vio_flg` 决定当前阶段。理解这个状态机，比只看三个模块名称更能解释一条轨迹到底在什么时刻、什么阶段写出。

## 输出位姿的阶段也是算法语义

`Log/result/<seq>.txt` 保存 IMU body 位姿，位于当前 LiDAR 更新后、当前 VIO 前；该位姿仍包含此前视觉更新的历史影响。`Log/image/image_poses.txt` 保存 VIO 后的位姿，启用图像保存时还受采样间隔影响。两种文件不能混成同一频率和同一阶段的轨迹。

ROS odometry/path 的 header 在这份源码中使用 `ros::Time::now()`。传输发布时间与测量时间不同，离线误差评测需要明确传感器时间和输出阶段。[漂移与位置评测](23_fast_livo2_drift.md)解释如何建立共同的比较时间支持。

连接重建后端时，还要把 IMU 位姿通过外参转换到相机中心，声明使用 C2W 还是 W2C，保留米制尺度和传感器时间。FAST-LIVO2 维护的是几何与视觉地图；它本身没有把地图表示为高斯。[GS-SLAM 系统连接](19_gaussian_slam_system_network.md)讨论定位前端与后续地图表示的接口。


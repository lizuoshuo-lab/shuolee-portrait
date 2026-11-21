---
id: "gslab-k01"
slug: "camera-geometry-depth-units"
title: "01 相机内外参、w2c/c2w、深度单位与图像缩放"
order: 1
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats"
  - "https://colmap.github.io/format.html"
  - "https://github.com/spla-tam/SplaTAM/blob/main/scripts/splatam.py"
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
---

# 01 相机内参与外参：像素、深度和位姿的坐标约定

相机几何贯穿单张图像拟合、已知位姿多视图建图与同步定位建图（Simultaneous Localization and Mapping，SLAM）。已知位姿建图使用给定相机位姿，不估计轨迹，不能据此声称完成相机跟踪。

## 原理

采用列向量，相机坐标轴为 $X$ 向右、$Y$ 向下、$Z$ 向前。**投影中心**是针孔模型中所有投影射线经过的点，也是相机坐标原点。**光轴**是经过投影中心、垂直像平面的直线；本页取其正方向为相机 $Z$ 轴。光轴与像平面的交点称为**主点**，其像素坐标不一定等于图像中心。

设相机点为 $\mathbf p_c=(X,Y,Z)^\top$，且 $Z>0$，无畸变针孔投影与内参矩阵为：

$$
\begin{aligned}
u&=f_xX/Z+c_x,\\
v&=f_yY/Z+c_y,
\end{aligned}
\qquad
K=
\begin{pmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{pmatrix}.
$$

$X,Y,Z$ 使用统一长度单位；焦距 $f_x,f_y$、主点 $c_x,c_y$ 与像素坐标 $u,v$ 的单位均为像素。图像高度、宽度分别为 $H,W$。畸变不包含在这个简单的 $K$ 中；输入未去畸变时，不能把投影误差全部归因于高斯模型。令 $X=Y=0$，即可得到光轴上的点投影到 $(c_x,c_y)$。

世界到相机变换记为 $T_{cw}$，相机到世界变换记为 $T_{wc}$。下标明确变换方向，二者互逆：

$$
T_{cw}=
\begin{pmatrix}
R_{cw}&\mathbf t_{cw}\\
\mathbf 0^\top&1
\end{pmatrix},
\qquad
\mathbf p_c=R_{cw}\mathbf p_w+\mathbf t_{cw},
\qquad
T_{wc}=T_{cw}^{-1}.
$$

$$
R_{wc}=R_{cw}^\top,\qquad
\mathbf c_w=\mathbf t_{wc}=-R_{cw}^\top\mathbf t_{cw}.
$$

$R$ 是无量纲的 $3\times3$ 旋转矩阵；$\mathbf t$、点坐标及世界中的投影中心 $\mathbf c_w$ 是三维长度向量。**世界到相机变换的平移列，通常不是相机的世界位置。** 若以首相机为参考，原始相机到世界轨迹 $T_{wc}(t)$ 应整体转换为：

$$
T_{0c}(t)=T_{wc}(0)^{-1}T_{wc}(t).
$$

之后再求逆得到相应的世界到相机变换。只给轨迹平移减去首帧位置而忽略旋转，不能完成坐标转换。

深度若表示沿光轴方向的 $Z$，反投影为：

$$
\mathbf p_c=
\begin{pmatrix}
(u-c_x)Z/f_x\\
(v-c_y)Z/f_y\\
Z
\end{pmatrix},
\qquad
\mathbf p_w=R_{wc}\mathbf p_c+\mathbf c_w.
$$

光轴深度和沿视线的欧氏距离 $\rho$ 不同：

$$
\rho=\|\mathbf p_c\|_2
=Z\sqrt{1+\left(\frac{u-c_x}{f_x}\right)^2+
\left(\frac{v-c_y}{f_y}\right)^2}.
$$

若数据给出 $\rho$，应先将反投影射线归一化，再乘该距离。TUM RGB-D 的16位PNG深度按其官方格式换算为：

$$
Z=\frac{d_{\mathrm{PNG}}}{5000}\,\mathrm m.
$$

$d_{\mathrm{PNG}}$ 是存储的整数深度，零值无效。除数5000只适用于该文件格式，不能套给浮点深度、其他数据集或已换算数据。TUM 官方深度已应用其公布的校正，不再重复乘校正系数。

图像从 $W\times H$ 变为 $W'\times H'$，定义无量纲缩放系数：

$$
s_x=\frac{W'}{W},\qquad s_y=\frac{H'}{H},\qquad
f_x'=s_xf_x,\qquad f_y'=s_yf_y.
$$

主点必须遵循与像素相同的变换。简单缩放约定与半像素采样约定分别给出：

$$
\begin{aligned}
u'&=s_xu,&c_x'&=s_xc_x,\\
v'&=s_yv,&c_y'&=s_yc_y,
\end{aligned}
$$

$$
\begin{aligned}
u'&=s_x(u+\tfrac12)-\tfrac12,
&c_x'&=s_x(c_x+\tfrac12)-\tfrac12,\\
v'&=s_y(v+\tfrac12)-\tfrac12,
&c_y'&=s_y(c_y+\tfrac12)-\tfrac12.
\end{aligned}
$$

裁剪还需减去裁剪起点的像素坐标。缩图不改变深度的米单位。深度边界宜使用最近邻或明确的有效性处理，避免双线性插值把前景和背景混成不存在的表面。

## 资料与模型边界

- [SplaTAM 反投影参考](https://github.com/spla-tam/SplaTAM/blob/main/scripts/splatam.py)使用内参与光轴深度反投影，再经相机到世界变换放置三维点。
- [COLMAP 格式说明](https://colmap.github.io/format.html)定义世界到相机的外参、坐标轴与四元数顺序。TUM 轨迹采用 $(q_x,q_y,q_z,q_w)$ 顺序，转换时不能直接照搬不同约定的数组。
- [gsplat v1.5.3](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3)使用世界到相机变换。切换接口时，应重新核对矩阵方向。
- TUM 与 NeRF 风格数据可能采用不同相机坐标约定；切换数据集前应检查相机朝向与正深度，不能直接复用矩阵。

## 手算例子与验证方法

设 $f_x=f_y=500\,\mathrm{px}$、$(c_x,c_y)=(320,240)\,\mathrm{px}$，像素 $(420,290)\,\mathrm{px}$ 的PNG深度为10000，则：

$$
Z=2\,\mathrm m,\qquad
\mathbf p_c=
\begin{pmatrix}0.4\\
0.2\\
2\end{pmatrix}\mathrm m.
$$

若相机到世界旋转为 $I_3$、相机中心为 $(1,0,0)^\top\,\mathrm m$，则：

$$
\mathbf p_w=
\begin{pmatrix}1.4\\
0.2\\
2\end{pmatrix}\mathrm m,\qquad
\mathbf t_{cw}=
\begin{pmatrix}-1\\
0\\
0\end{pmatrix}\mathrm m.
$$

按 $u'=0.5u,\ v'=0.5v$ 缩图，焦距变为250像素、主点变为 $(160,120)$ 像素，对应像素 $(210,145)$ 仍反投影到同一个 $\mathbf p_c$。

验证时从有效深度执行反投影、相机到世界变换、逆变换与重投影，先过滤无效或超范围深度。同步缩放图像与内参后，对应射线和三维点应保持一致。代数闭环验证约定自洽，真实标定精度仍需独立观测。

## 面试自测

1. 为什么 $\mathbf t_{cw}$ 不等于投影中心的世界位置？写出求解式。
2. 深度少除一个5000，会怎样影响尺度、裁剪阈值和优化？
3. 只缩小图像而不修改 $K$，会发生什么？
4. 全局坐标变换与单目尺度歧义有什么不同？

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

### 实验验证与适用范围

- 资料：已阅读并核对上述官方格式与 SplaTAM 反投影函数；本文使用独立推导。
- [TUM RGB-D 单帧几何实测](../experiments/2026-10-08-rgbd-geometry.md)：有效深度比例 83.0697%，导出 16,008 个彩色点；相机→世界→相机最大误差 9.5505e-16 m，重投影最大误差 3.5951e-13 pixel。
- [多视图高斯训练实测](../experiments/2026-10-08-local-gsplat.md)：实际采用 `c2w_cv = c2w_gl @ diag(1,-1,-1,1)`，随后取逆得到 gsplat 的 w2c。
- 适用范围：闭环误差验证数值自洽，不证明物理标定或深度尺度正确；本次未补偿 RGB-depth 时间差、未去畸变，也未验证缩图前后三维点一致性。

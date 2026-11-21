---
id: "gslab-k02"
slug: "gaussian-parameters-projection-compositing"
title: "02 标准 3DGS 参数、投影合成与 SplaTAM 简化"
order: 2
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py"
  - "https://github.com/spla-tam/SplaTAM/blob/main/scripts/splatam.py"
  - "https://github.com/spla-tam/SplaTAM/blob/main/utils/slam_helpers.py"
  - "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3"
---

# 02 高斯表示、透视投影与不透明度合成

本页保留不同教学模型与 SplaTAM 简化的对照。系统学习原版 3DGS，请先读[传统 3DGS 渲染基础](27_3dgs_rendering_foundations.md)，再读[训练与密度控制](28_3dgs_training_density_control.md)；两章按原论文图与公式展开。

单张图像拟合可以展示参数、渲染、图像损失、梯度与更新的连续关系。多相机和深度增加跨视角与几何约束。单图拟合即使使用三维光栅化器，也不能单独证明真实三维几何或同步定位建图能力。

## 原理

标准三维高斯泼溅（3D Gaussian Splatting，3DGS）为每个高斯维护均值 $\boldsymbol\mu\in\mathbb R^3$、三轴标准差 $\mathbf s\in\mathbb R_{>0}^3$、旋转四元数 $\mathbf q\in\mathbb R^4$、不透明度 $o$ 与球谐函数系数。球谐函数（Spherical Harmonics，SH）表示视角相关颜色，不是完整的材质与光照分解。

均值与标准差采用长度单位 $L$。非零四元数按 $(q_w,q_x,q_y,q_z)$ 顺序归一化，令 $\widehat{\mathbf q}=(w,x,y,z)^\top$：

$$
\widehat{\mathbf q}=\frac{\mathbf q}{\|\mathbf q\|_2},\qquad
R=
\begin{pmatrix}
1-2(y^2+z^2)&2(xy-wz)&2(xz+wy)\\
2(xy+wz)&1-2(x^2+z^2)&2(yz-wx)\\
2(xz-wy)&2(yz+wx)&1-2(x^2+y^2)
\end{pmatrix}.
$$

$R$ 是无量纲的 $3\times3$ 旋转矩阵，将高斯局部主轴转到世界坐标。世界协方差为：

$$
\Sigma_w=
R\begin{pmatrix}
s_x^2&0&0\\
0&s_y^2&0\\
0&0&s_z^2
\end{pmatrix}R^\top.
$$

$\Sigma_w$ 的单位为 $L^2$，三个标准差平方才是主轴方差。高斯核的等值面是椭球，不是具有硬边界的实体，也不自动等于物体真实表面。

采用世界到相机的旋转和平移 $R_{cw},\mathbf t_{cw}$，中心与协方差应一起转换：

$$
\boldsymbol\mu_c=R_{cw}\boldsymbol\mu_w+\mathbf t_{cw},\qquad
\Sigma_c=R_{cw}\Sigma_wR_{cw}^\top.
$$

令高斯中心在相机坐标系中的坐标为 $\boldsymbol\mu_c=(X,Y,Z)^\top$、$Z>0$。针孔投影的局部雅可比与二维投影协方差为：

$$
J=
\begin{pmatrix}
f_x/Z&0&-f_xX/Z^2\\
0&f_y/Z&-f_yY/Z^2
\end{pmatrix},\qquad
\Sigma_{2D}\approx J\Sigma_cJ^\top.
$$

焦距 $f_x,f_y$ 以像素计；$J$ 为 $2\times3$、单位 $\mathrm{px}/L$，$\Sigma_{2D}$ 为 $2\times2$、单位 $\mathrm{px}^2$。中心经过透视除法，协方差用一阶近似形成屏幕椭圆；实现还可加入屏幕低通项与数值约束，不能把该近似当成精确透视后的分布。

记第 $i$ 个高斯的二维投影中心为 $\mathbf m_i$、二维投影协方差为 $\overline\Sigma_{2Di}$。像素 $\mathbf p$ 上的有效不透明度、前方透射率与最终颜色为：

$$
\alpha_i(\mathbf p)=o_i\exp\!\left[
-\frac12(\mathbf p-\mathbf m_i)^\top
\overline\Sigma_{2Di}^{-1}(\mathbf p-\mathbf m_i)\right],
$$

$$
T_i(\mathbf p)=\prod_{j<i}\bigl(1-\alpha_j(\mathbf p)\bigr),
$$

$$
\mathbf C(\mathbf p)=
\sum_iT_i(\mathbf p)\alpha_i(\mathbf p)\mathbf c_i
+T_{\mathrm{end}}(\mathbf p)\mathbf c_{\mathrm{bg}},\qquad
T_{\mathrm{end}}(\mathbf p)=\prod_i(1-\alpha_i(\mathbf p)).
$$

$\mathbf p,\mathbf m_i$ 的单位是像素；$o_i,\alpha_i,T_i$ 无量纲。高斯按深度从前向后参与合成；$o_i$ 是高斯参数，$\alpha_i$ 是该像素上的有效不透明度。直接相加所有颜色会丢失遮挡关系。排序与投影含有近似，不能等同于精确体积积分或光线追踪。

正标准差与不透明度可由无约束参数构造：

$$
s_j=s_{\mathrm{ref}}e^{\ell_j},\qquad
o=\frac{1}{1+e^{-a}},\qquad
\widehat{\mathbf q}=\frac{\mathbf q}{\|\mathbf q\|_2}.
$$

$s_{\mathrm{ref}}$ 是固定参考长度，$\ell_j,a$ 无量纲；归一化要求 $\mathbf q\ne0$。不同参数的单位与敏感性不同，学习率也可不同。图像损失降低可能来自尺度、不透明度或外观变化，未必来自几何更准确。

## SplaTAM 的简化与模型边界

- [SplaTAM 参数初始化](https://github.com/spla-tam/SplaTAM/blob/main/scripts/splatam.py)直接使用RGB颜色，而非原版高阶球谐函数。
- 默认各向同性高斯使用一个标准差 $s$，其协方差为：

$$
\Sigma=s^2I_3=
\begin{pmatrix}
s^2&0&0\\
0&s^2&0\\
0&0&s^2
\end{pmatrix}.
$$

此时均值3维、RGB颜色3维、标准差1维、不透明度1维，共8个有效场景自由度。自身旋转不改变形状，但参考实现仍可保留旋转参数以复用渲染接口。
- 当前官方实现也有各向异性配置分支；不能把论文简化说成永远只支持球形。
- [渲染参数转换参考](https://github.com/spla-tam/SplaTAM/blob/main/utils/slam_helpers.py)将对数尺度、不透明度的对数几率与四元数转换成渲染参数；各向同性尺度在三个轴上相同。
- SplaTAM 还渲染深度与累计不透明度，处理彩色和深度约束以及地图覆盖；高覆盖不能证明几何正确。
- [gsplat v1.5.3 示例与源码](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3)用于本地入门；具体入口与参数需对应实际版本。

## 手算例子与验证方法

黑色背景上，同一像素前方红色、后方蓝色，两者的有效不透明度均为0.5：

$$
\mathbf C=
0.5\begin{pmatrix}1\\
0\\
0\end{pmatrix}
+(1-0.5)0.5\begin{pmatrix}0\\
0\\
1\end{pmatrix}
=\begin{pmatrix}0.5\\
0\\
0.25\end{pmatrix},
$$

$$
A=1-(1-0.5)(1-0.5)=0.75.
$$

$\mathbf C$ 是归一化RGB颜色，$A$ 是累计不透明度。交换前后顺序后，颜色变为 $(0.25,0,0.5)^\top$，总不透明度不变。这说明排序影响颜色，而不仅是总覆盖。

验证时固定观测和其他参数，单独改变高斯数量或尺度，观察覆盖、细节与收敛。单视图有多解；多视图和深度约束提供额外观测信息，不能用单图拟合证明三维几何正确。

## 面试自测

1. 为什么用旋转与正标准差构造协方差，而不是优化任意 $3\times3$ 矩阵？
2. 各向同性三维高斯投影到图像上一定为圆形吗？
3. 高斯不透明度、像素有效不透明度与累计不透明度有什么区别？
4. 为什么训练视角正确，换视角仍可能失败？

## 实验附录

以下为对应教学实现的历史实测与核验，供查证具体数据及适用范围。

### 实验验证与适用范围

- 资料：已核对 SplaTAM 参数初始化、渲染参数转换与配置分支。
- [本地 gsplat 训练实测](../experiments/2026-10-08-local-gsplat.md)：官方单图示例以 2000 个高斯、500 次优化使 MSE 从 0.32925 降至 0.00962，PSNR 从 4.82 提升至 20.17 dB；初始与最终渲染已保存。
- 同一报告的多视图实验优化 8000 个高斯的中心、三轴尺度、四元数、RGB 与 opacity，采用固定数据位姿，未启用 SH、高斯增密或裁剪。6 个留出视角平均 PSNR 为 27.50 dB。
- 单图与多视图的 PyTorch allocated 显存峰值分别约 1.51 MiB、23.30 MiB，不等于进程总显存。这份早期基线验证优化链路，未验证论文基准或 SLAM 能力；后来完成的多视图数量对照见 [容量与初始化](../knowledge/05_capacity_initialization_density.md)，没有运行增密或裁剪。

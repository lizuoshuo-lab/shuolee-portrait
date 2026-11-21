---
id: "gslab-k27"
slug: "3dgs-rendering-foundations"
title: "传统 3DGS：从三维高斯到像素颜色"
order: 27
status: "knowledge_synthesis_from_verified_sources"
updated: "2026-10-11"
sources:
  - "https://arxiv.org/abs/2308.04079"
  - "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/scene/gaussian_model.py"
  - "https://github.com/graphdeco-inria/gaussian-splatting/blob/b2ada78a779ba0455dfdc2b718bdf1726b05a1b6/gaussian_renderer/__init__.py"
  - "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/8064f52ca233942bdec2d1a1451c026deedd320b/cuda_rasterizer/forward.cu"
  - "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/8064f52ca233942bdec2d1a1451c026deedd320b/cuda_rasterizer/rasterizer_impl.cu"
  - "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/8064f52ca233942bdec2d1a1451c026deedd320b/cuda_rasterizer/backward.cu"
---

# 传统 3DGS：从三维高斯到像素颜色

传统三维高斯泼溅（3D Gaussian Splatting，3DGS）把一个静态场景表示为一组可优化的三维高斯。每个高斯描述局部空间分布、颜色和不透明度；给定相机后，渲染器将它们投到图像，再按前后关系合成像素。图像误差反过来约束这些参数，形成重建过程。

理解这套方法，需要连续回答六个问题：输入给定了什么，三维形状怎样表示，空间分布怎样投到图像，颜色怎样随观察方向变化，遮挡怎样进入像素颜色，以及图像损失怎样传回场景参数。下面综合原论文的方法、消融与相关研究解释这些关系。后续论文的图片只用于说明共同几何基础或传统模型的局限，方法差别在正文中注明。

## 第一步：重建问题与场景参数

![原论文总览：从SfM稀疏点建立高斯，投影并渲染图像，以图像误差优化场景参数与密度](../outputs/3dgs-paper-course-001/figures/paper-fig-02.png)
> 图源：[Kerbl 等，3DGS（2023），Fig. 2，第 5 页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf#page=5) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

原始 3DGS 的输入是同一静态场景的多幅照片，以及运动恢复结构（Structure from Motion，SfM）得到的相机参数和稀疏三维点。相机是给定条件，待优化的是场景表示；它没有同时完成同步定位与建图（Simultaneous Localization and Mapping，SLAM）中的相机位姿估计。

SfM 点提供与照片关联的空间起点，但不一定能可靠估计表面法向量。作者因此选择完整三维高斯，而不是必须指定法向量的二维圆盘。高斯中心从稀疏点开始，随后位置、形状、颜色和不透明度都可以调整，高斯数量也可以增加或减少。

固定当前高斯集合，场景参数为：

$$
\Theta=\left\{\boldsymbol\mu_i,\mathbf s_i,\mathbf q_i,o_i,\boldsymbol\beta_i\right\}_{i=1}^{N}.
$$

$N$ 是高斯数；$\boldsymbol\mu_i\in\mathbb R^3$ 是世界坐标中的中心；$\mathbf s_i\in\mathbb R_{>0}^3$ 是主轴标准差；$\mathbf q_i\in\mathbb R^4$ 表示主轴旋转；$o_i\in(0,1)$ 为不透明度；$\boldsymbol\beta_i$ 为球谐颜色系数。位置和标准差采用同一长度单位 $L$；其余参数无量纲。单目 SfM 没有尺度参照时，$L$ 不能自动解释为米。

Fig. 2 中，黑色箭头表示参数怎样形成图像，蓝色箭头表示图像误差怎样约束参数。下面先建立这一前向关系；高斯的增加、分裂与剪枝在[训练与密度控制](28_3dgs_training_density_control.md)中继续展开。

## 第二步：各向异性三维高斯与协方差矩阵

![原论文将优化后的高斯缩小展示，显示适应百叶窗等局部结构的各向异性形状](../outputs/3dgs-paper-course-001/figures/paper-fig-03.png)
> 图源：[Kerbl 等，3DGS（2023），Fig. 3，第 5 页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf#page=5) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

各向异性（anisotropy）表示不同方向具有不同尺度。高斯可以近似球形，也可以扁平或细长：平滑区域可以由较大的扁平分布表达，细结构可以由沿结构方向延伸的分布表达。Fig. 3 将优化后的高斯缩小 60%，使相互重叠的形状可见；间隙来自展示方式，不是训练目标。

中心为 $\boldsymbol\mu_i$、世界坐标协方差为 $\Sigma_{wi}$ 时，三维高斯核为：

$$
G_i(\mathbf x)=\exp\!\left[-\frac12(\mathbf x-\boldsymbol\mu_i)^\top\Sigma_{wi}^{-1}(\mathbf x-\boldsymbol\mu_i)\right].
$$

$\mathbf x$ 的单位为 $L$，$\Sigma_{wi}$ 的单位为 $L^2$，指数无量纲。这个核在中心取 1，没有概率密度的归一化系数；高斯不透明度另外进入图像合成。可见椭球是核的等值面，不能直接等同于物体表面。

协方差必须半正定，直接独立更新各矩阵元素可能破坏这一条件。原论文将形状拆成标准差与旋转：

$$
S_i=\begin{bmatrix}
s_{ix}&0&0\\
0&s_{iy}&0\\
0&0&s_{iz}
\end{bmatrix},\qquad
S_i^2=\begin{bmatrix}
s_{ix}^2&0&0\\
0&s_{iy}^2&0\\
0&0&s_{iz}^2
\end{bmatrix},\qquad
\Sigma_{wi}=R_iS_i^2R_i^\top.
$$

$S_i$ 的单位为 $L$，$R_i$ 无量纲，并将高斯局部主轴坐标转到世界坐标。三个标准差平方后才是主轴方差。对任意非零向量 $\mathbf a$，正标准差保证：

$$
\mathbf a^\top\Sigma_{wi}\mathbf a
=\|S_iR_i^\top\mathbf a\|^2>0.
$$

因此协方差正定，其逆矩阵存在。非对角元素表达主轴与世界坐标轴不重合时，坐标分量之间的相关性。

四元数采用标量在前的顺序。对无约束变量 $\widetilde{\mathbf q}_i\ne\mathbf0$ 归一化后：

$$
\mathbf q_i=\frac{\widetilde{\mathbf q}_i}{\sqrt{\widetilde{\mathbf q}_i^\top\widetilde{\mathbf q}_i}}
=\begin{bmatrix}w_i\\
x_i\\
y_i\\
z_i\end{bmatrix},\qquad
w_i^2+x_i^2+y_i^2+z_i^2=1.
$$

对应的局部到世界旋转为：

$$
R_i=\begin{bmatrix}
1-2(y_i^2+z_i^2)&2(x_iy_i-w_iz_i)&2(x_iz_i+w_iy_i)\\
2(x_iy_i+w_iz_i)&1-2(x_i^2+z_i^2)&2(y_iz_i-w_ix_i)\\
2(x_iz_i-w_iy_i)&2(y_iz_i+w_ix_i)&1-2(x_i^2+y_i^2)
\end{bmatrix}.
$$

$$
R_i^\top R_i=I_3,\qquad \det R_i=1,\qquad
I_3=\begin{bmatrix}1&0&0\\
0&1&0\\
0&0&1\end{bmatrix}.
$$

单位四元数具有三个旋转自由度，$\mathbf q$ 与 $-\mathbf q$ 表示同一旋转。当三个标准差相等时，高斯球对称，改变旋转不改变形状。

等值面可以严格写为：

$$
(\mathbf x-\boldsymbol\mu_i)^\top\Sigma_{wi}^{-1}(\mathbf x-\boldsymbol\mu_i)=r^2,
\qquad G_i(\mathbf x)=e^{-r^2/2},\qquad r>0.
$$

$r$ 是无量纲马氏距离。$r=1$ 时，主轴半轴长分别为三个标准差；高斯在这个椭球之外仍然非零。空间形状决定覆盖范围，但仅凭图像损失优化的椭球并不保证已经恢复真实表面。

## 第三步：相机坐标变换与透视投影

![综述完整Fig3：三维高斯投影、图块划分、候选复制与排序、逐像素合成](../outputs/3dgs-paper-only-002/figures/survey-fig-03-forward-rendering.png)
> 图源：[Chen 与 Wang，A Survey on 3D Gaussian Splatting（2024），Fig. 3，第 4 页，v1](https://arxiv.org/pdf/2401.03890v1#page=4) · 完整原图 · [原文发布许可](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)。

完整 Fig. 3 将投影与后续像素计算放在同一流程中。本节先看左上方 (a)：三维高斯投到图像空间，形成二维核；(b)–(d) 的图块组织、排序和像素合成将在第六步继续解释。下面将位置变换与协方差投影分别展开。图中的可见椭球和椭圆帮助理解分布形状，不意味着高斯具有精确有限边界。

三维中心先转到相机坐标，三维形状也必须同步变换，然后才能在图像上计算贡献。本文采用列向量、理想针孔相机，相机 $X$ 轴向右、$Y$ 轴向下、$Z$ 轴正方向朝向场景；像平面等效放在投影中心前方，采用正焦距。

**光轴（optical axis）**是经过投影中心、垂直于像平面的直线，与本约定下的相机 $Z$ 轴重合。它与像平面的交点是**主点（principal point）**，不一定处在图像中心。前向光轴射线满足：

$$
\mathbf p_{\mathrm{axis}}(t)=\begin{bmatrix}0\\
0\\
t\end{bmatrix},\qquad t>0,\qquad
\pi\bigl(\mathbf p_{\mathrm{axis}}(t)\bigr)
=\begin{bmatrix}c_x\\
c_y\end{bmatrix}.
$$

$t$ 的单位为 $L$。世界到相机的中心变换为：

$$
\boldsymbol\mu_{ci}=R_{cw}\boldsymbol\mu_i+\mathbf t_{cw}
=R_{cw}(\boldsymbol\mu_i-\mathbf C_v),\qquad
\mathbf t_{cw}=-R_{cw}\mathbf C_v.
$$

$\mathbf C_v$ 是相机投影中心的世界坐标；$R_{cw}$ 把世界向量转到相机坐标；$\mathbf t_{cw}$ 是对应平移。位置和平移的单位为 $L$，旋转无量纲。高斯自身旋转 $R_i$ 决定主轴，相机旋转 $R_{cw}$ 决定观察坐标，不能相互替换。

令 $\boldsymbol\mu_{ci}=(X,Y,Z)^\top$ 且 $Z>0$，无畸变、像素坐标轴正交时：

$$
K=\begin{bmatrix}f_x&0&c_x\\
0&f_y&c_y\\
0&0&1\end{bmatrix},\qquad
\mathbf m_i=\pi(\boldsymbol\mu_{ci})=
\begin{bmatrix}f_xX/Z+c_x\\
f_yY/Z+c_y\end{bmatrix}.
$$

$X,Y,Z$ 的单位为 $L$，$Z$ 是轴向深度；焦距 $f_x,f_y$、主点 $c_x,c_y$ 与投影中心 $\mathbf m_i$ 都用像素 $\mathrm{px}$ 表示。除以深度产生透视缩小。单个像素只确定一条射线，不能单独确定点的深度。基本坐标关系可回看[相机几何](01_camera_geometry.md)。

视锥体（view frustum）决定有限渲染区域。近裁剪平面与远裁剪平面位于轴向深度 $n$、$f_{\mathrm{far}}$；它们不是像平面或焦距。对一个点，图像宽、高为 $W,H$ 时，可以用连续图像边界写为：

$$
n\le Z\le f_{\mathrm{far}},\qquad
0\le f_xX/Z+c_x\le W,\qquad
0\le f_yY/Z+c_y\le H.
$$

$0<n<f_{\mathrm{far}}$，$n,f_{\mathrm{far}}$ 的单位为 $L$，$W,H$ 的单位为 $\mathrm{px}$。这些是点的区域条件；无限支持的高斯不能仅凭中心是否在视锥体内，就判断整段分布可见与否。

形状转到相机坐标时：

$$
\Sigma_{ci}=R_{cw}\Sigma_{wi}R_{cw}^\top.
$$

平移改变中心，不改变相对中心的分布；旋转改变相机看到的主轴方向。$\Sigma_{ci}$ 仍为 $3\times3$、单位 $L^2$。

透视投影不是线性的，精确变换后的分布一般不再是二维高斯。原始 3DGS 使用中心附近的一阶近似：

$$
\pi(\boldsymbol\mu_{ci}+\delta\mathbf x)
\approx\pi(\boldsymbol\mu_{ci})+J_i\delta\mathbf x,
$$

$$
J_i=\left.\frac{\partial\pi}{\partial\mathbf x}\right|_{\boldsymbol\mu_{ci}}
=\begin{bmatrix}
f_x/Z&0&-f_xX/Z^2\\
0&f_y/Z&-f_yY/Z^2
\end{bmatrix},\qquad
\Sigma_{2Di}\approx J_i\Sigma_{ci}J_i^\top.
$$

$\delta\mathbf x$ 为单位 $L$ 的小位移；$J_i$ 是 $2\times3$ 雅可比，单位 $\mathrm{px}/L$；$\Sigma_{2Di}$ 是 $2\times2$ 投影协方差，单位 $\mathrm{px}^2$。其特征向量给出图像椭圆主轴方向，特征值的平方根给出主轴标准差。这个椭圆描述近似二维核，不是三维椭球精确透视轮廓或物体边界。

在光轴上，中心对深度的投影导数为零：

$$
\left.\frac{\partial\pi}{\partial Z}\right|_{X=Y=0}
=\begin{bmatrix}0\\
0\end{bmatrix}.
$$

这只表示沿光轴移动不改变投影中心。投影形状仍随深度改变：对光轴上的各向同性高斯，三维标准差为 $s$ 且 $f_x=f_y=f$ 时：

$$
\Sigma_{2Di}=\frac{f^2s^2}{Z^2}I_2,\qquad
s_{\mathrm{image}}=\frac{fs}{Z},\qquad
I_2=\begin{bmatrix}1&0\\
0&1\end{bmatrix}.
$$

$s_{\mathrm{image}}$ 的单位为像素。相机距离、尺度和焦距共同决定覆盖范围；场景位置、相机平移与高斯标准差同时乘同一比例，投影可以保持不变。这是纯单目表示无法自行确定绝对尺度的原因之一。

为离散像素采样加入最小平滑范围时，可以写为：

$$
\overline\Sigma_{2Di}=\Sigma_{2Di}+\varepsilon I_2,\qquad \varepsilon>0.
$$

$\varepsilon$ 的单位为 $\mathrm{px}^2$。固定屏幕平滑不是任意相机距离与分辨率下都准确的抗锯齿方案。中心附近线性化也要求高斯相对深度较小、远离 $Z=0$；大高斯、近相机或极斜视角会增加近似误差。

对像素中心 $\mathbf p=(u,v)^\top$，投影分布给出二维核及像素有效不透明度：

$$
\mathbf d_i=\mathbf p-\mathbf m_i,\qquad A_i=\overline\Sigma_{2Di}^{-1},
$$

$$
g_i(\mathbf p)=\exp\!\left(-\frac12\mathbf d_i^\top A_i\mathbf d_i\right),\qquad
\alpha_i(\mathbf p)=o_i\,g_i(\mathbf p).
$$

$\mathbf d_i$ 的单位为 $\mathrm{px}$，$A_i$ 的单位为 $\mathrm{px}^{-2}$；$g_i,o_i,\alpha_i$ 无量纲。$o_i$ 是一个高斯的参数，$\alpha_i(\mathbf p)$ 随像素位置改变。理想核满足：

$$
0<g_i(\mathbf p)\le1,\qquad 0<\alpha_i(\mathbf p)\le o_i<1.
$$

离投影中心越远，马氏距离越大，贡献越小；像素靠近哪个主轴方向，影响范围由协方差决定。扩大标准差、提高不透明度或移动中心，都可能降低局部图像误差，却对应不同的空间解释。核变宽不自动保持积分或颜色贡献守恒。传统 3DGS 直接在像平面评估这个核，没有对每条射线上的三维密度进行精确积分。

![Mip-Splatting原图诊断传统3DGS：三维分布投影与固定屏幕膨胀在不同采样尺度下产生不同失真](../outputs/3dgs-paper-only-002/figures/mip-fig-01-projection-dilation.png)
> 图源：[Yu 等，Mip-Splatting（2024），Fig. 1，第 1 页](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf#page=1) · 原图 · [原文发布许可](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)。

Mip-Splatting 的 Fig. 1 进一步诊断传统模型：黑色投影核与红色屏幕膨胀结合，在训练尺度上可能掩盖过薄的三维高斯；改变焦距或相机距离后，出现过粗、过细或高频伪影。该后续方法加入三维频率约束，并改用带幅值补偿的二维 Mip 滤波。它与传统固定屏幕膨胀不同，不能把这些改进反写成原始 3DGS 已有的步骤。

## 第四步：球谐函数与视角相关颜色

![Plenoxels完整Fig2：稀疏体素网格、球谐颜色与三线性插值、射线体渲染和参数优化](../outputs/3dgs-paper-only-002/figures/plenoxels-fig-02-sh-model.png)
> 图源：[Fridovich-Keil 等，Plenoxels（2022），Fig. 2，第 3 页](https://openaccess.thecvf.com/content/CVPR2022/papers/Fridovich-Keil_Plenoxels_Radiance_Fields_Without_Neural_Networks_CVPR_2022_paper.pdf#page=3) · 完整原图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

几何投影给出了像素覆盖，却还没有给出当前视角的颜色。反光表面的外观会随观察方向改变，只给每个高斯一个固定 RGB 向量会损失这部分表达能力。原始 3DGS 使用球谐函数（Spherical Harmonics，SH），在单位球面上展开颜色。

先沿完整 Fig. 2 阅读 Plenoxels 的关系：(a) 稀疏体素网格的角点保存密度与球谐系数，相机射线穿过网格；(b) 用相邻八个角点的参数做三线性插值，得到射线采样点的密度和球谐系数，再由观察方向求颜色；上方的球谐基按系数加权，彩色小球表示方向相关颜色函数在单位球面上的分布，不是三维高斯或球形物体；(c) 沿射线进行可微体渲染，得到预测像素颜色，图中 $\sigma$ 表示体密度；(d) 将预测结果与训练照片比较，通过均方误差重建损失和总变差（Total Variation，TV）正则化更新网格参数。中间的放大连接和黄色箭头将体素邻域、射线采样与插值连在一起。

本节借用的是 (b) 上方的球谐颜色表达。Plenoxels 的体素网格、三线性插值、射线体渲染与总变差正则属于它自身的方法；传统 3DGS 使用三维高斯、投影核和 Alpha 合成。保留完整图是为了说明球谐系数在空间表示和渲染中的位置，不能把整张 Plenoxels 流程当作传统 3DGS 的步骤。图片可点击打开完整分辨率，下面将两者共享的球谐关系写成公式。

对视角 $v$，将方向定义为世界坐标中“相机指向高斯中心”的单位向量：

$$
\mathbf d_{iv}=\frac{\boldsymbol\mu_i-\mathbf C_v}{\|\boldsymbol\mu_i-\mathbf C_v\|}.
$$

$\mathbf C_v$ 是该视角的相机中心，$\mathbf d_{iv}$ 无量纲。反向定义会改变奇数阶项的符号，不能保持同一组系数却随意改换方向约定。对颜色通道 $c\in\{R,G,B\}$，包含零阶的展开为：

$$
\widetilde c_{i,c}(\mathbf d_{iv})=\frac12+
\sum_{\ell=0}^{D}\sum_{m=-\ell}^{\ell}
\beta_{i\ell m,c}\,Y_{\ell m}(\mathbf d_{iv}).
$$

$D$ 是最大阶数；$Y_{\ell m}$ 是实球谐基函数；$\beta_{i\ell m,c}$ 是该通道对应的系数；$1/2$ 是原始模型的颜色数值偏置。基函数、系数和颜色无量纲。零至 $D$ 阶有 $(D+1)^2$ 个基函数，每个通道保存一组系数。

令 $\mathbf d=(d_x,d_y,d_z)^\top$ 且 $\|\mathbf d\|=1$，本文的实球谐基符号约定下，零阶与一阶为：

$$
\begin{aligned}
Y_{00}(\mathbf d)&=\frac1{\sqrt{4\pi}},&
Y_{1,-1}(\mathbf d)&=-\sqrt{\frac3{4\pi}}d_y,\\
Y_{10}(\mathbf d)&=\sqrt{\frac3{4\pi}}d_z,&
Y_{11}(\mathbf d)&=-\sqrt{\frac3{4\pi}}d_x.
\end{aligned}
$$

零阶是方向无关的常量，一阶允许沿坐标方向改变颜色，更高阶表达更细的角度变化。图中的 Plenoxels 使用零至二阶，即每通道 9 个系数、RGB 共 27 个；传统 3DGS 可用零至三阶，即每通道 16 个、RGB 共 48 个。这是方向相关外观的参数化，不独立恢复物理材质、入射光照和真实表面法向量。

球谐展开本身不限制颜色范围。原始模型对每个通道施加非负截断：

$$
 c_{i,c}(\mathbf d)=\begin{cases}
\widetilde c_{i,c}(\mathbf d),&\widetilde c_{i,c}(\mathbf d)>0,\\
0,&\widetilde c_{i,c}(\mathbf d)\le0.
\end{cases}
$$

这是下界截断，并未同时把上限限制为 1。外观自由度也需要足够多的观察方向来约束。原论文第 7.1 节指出，拍摄角度缺失时，高阶与基础颜色可能相互补偿，使零阶项不合理；作者因此先优化零阶，再每 1,000 次迭代引入新的阶次，直到包含零至三阶。照片拟合更好，并不自动意味着几何更准确。

## 第五步：Alpha 合成与透射率

![StopThePop原图诊断中心深度排序：同一像素射线的贡献顺序与相机朝向投影顺序可以不同](../outputs/3dgs-paper-only-002/figures/stopthepop-fig-02-depth-order.png)
> 图源：[Radl 等，StopThePop（2024），Fig. 2，第 2 页](https://r4dl.github.io/StopThePop/static/paper/StopThePop.pdf#page=2) · 原图 · [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。

一个像素通常由多个高斯覆盖。Alpha 合成（alpha compositing）把它们按前后顺序组合：前方先贡献颜色，后方只获得穿过前方后剩余的透射率。因此遮挡不是额外加在最后的效果，而是每一层颜色权重的一部分。

将参与当前像素的高斯按相机深度从近到远编号，前方透射率和当前高斯权重为：

$$
T_i(\mathbf p)=\prod_{j<i}\bigl(1-\alpha_j(\mathbf p)\bigr),\qquad
w_i(\mathbf p)=T_i(\mathbf p)\alpha_i(\mathbf p).
$$

最终颜色为：

$$
\mathbf C(\mathbf p)=\sum_{i=1}^{M_{\mathbf p}}w_i(\mathbf p)\mathbf c_i
+T_{\mathrm{bg}}(\mathbf p)\mathbf c_{\mathrm{bg}},\qquad
T_{\mathrm{bg}}(\mathbf p)=\prod_{i=1}^{M_{\mathbf p}}\bigl(1-\alpha_i(\mathbf p)\bigr).
$$

$M_{\mathbf p}$ 是参与当前像素的高斯数；$\mathbf c_i$ 是当前观察方向下的颜色；$T_i,w_i,T_{\mathrm{bg}}$ 无量纲，所有颜色采用一致 RGB 尺度。累计不透明度与权重满足：

$$
A_{\mathrm{total}}(\mathbf p)=1-T_{\mathrm{bg}}(\mathbf p),\qquad
\sum_{i=1}^{M_{\mathbf p}}w_i(\mathbf p)+T_{\mathrm{bg}}(\mathbf p)=1.
$$

顺序之所以重要，可以从两个高斯直接推导。固定像素、各自的颜色与有效不透明度，交换前后关系后：

$$
\mathbf C_{a\rightarrow b}-\mathbf C_{b\rightarrow a}
=\alpha_a\alpha_b(\mathbf c_a-\mathbf c_b).
$$

两者都非透明且颜色不同，合成顺序就会改变结果。前景越不透明，后景权重和颜色梯度越小，被遮挡结构需要其他可见视角提供约束。

传统 3DGS 为效率采用中心深度的图块排序，避免每个像素重新排序。对大而倾斜、相互穿插的高斯，中心深度顺序未必对应每条射线上最准确的贡献顺序；相机移动时发生顺序交换，还可能导致外观突变。这是传统可见性模型的近似条件，而不是球谐颜色阶数可以独自解决的问题。

StopThePop 的 Fig. 2 具体展示了这一局限：图 (a) 沿像素射线观察两个空间分布，图 (b) 用中心轴向深度近似其贡献位置，图 (c) 的相机旋转使近似顺序发生交换。图中的 $\mathbf r$ 是像素射线、$\mathbf v$ 是相机朝向，不能混用为上一节每个高斯的球谐方向 $\mathbf d_{iv}$。这是后续论文对传统模型的诊断；StopThePop 使用射线上的贡献深度与分层重新排序，传统 3DGS 原始方法没有这些步骤。

## 第六步：可微光栅化与参数梯度

![综述原图的前向渲染流程：高斯投影、划分图块、复制候选、按图块与深度排序、逐像素合成](../outputs/3dgs-paper-only-002/figures/survey-fig-03-forward-rendering.png)
> 图源：[Chen 与 Wang，A Survey on 3D Gaussian Splatting（2024），Fig. 3，第 4 页，v1](https://arxiv.org/pdf/2401.03890v1#page=4) · 原图 · [原文发布许可](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)。

综述 Fig. 3 将渲染过程连接起来：(a) 是已经推导的三维到二维投影；(b) 将图像划为图块，并确定高斯与图块的重叠；(c) 左侧为各个重叠图块复制候选条目，右侧先按图块分组，再按中心深度排序；(d) 在图块内的每个像素处，并行计算核值和 Alpha 合成。共享候选和顺序减少了重复计算，但像素颜色仍由各自的二维位置决定。

光栅化（rasterization）把连续投影表示转换为像素颜色。原论文将图像划为 $16\times16$ 像素图块，每个图块只处理可能影响它的高斯；一个高斯可进入多个图块。记图块为 $\tau$、投影高斯用于有限计算的包围区域为 $\mathcal B_i$，则：

$$
\mathcal G_\tau=\left\{i\in\{1,\ldots,N\}\mid\mathcal B_i\cap\tau\ne\varnothing\right\}.
$$

高斯没有精确有限支持，包围区域会舍去很小的贡献。图块内共享候选和排序，但每个像素分别计算核值与合成权重。透射率足够小时可以停止，计算成本因而还依赖覆盖范围、重叠数量和遮挡程度，不能只由高斯总数判断。

渲染图像与目标图像之间的损失记为无量纲标量 $\mathcal L$。颜色改变自身 RGB 贡献，不透明度同时改变自身权重与后方遮挡；中心影响投影位置、协方差投影及球谐方向；标准差与旋转影响形状。下面的局部导数直接揭示这些作用。

固定当前候选集合和前后顺序，对有贡献的高斯：

$$
\frac{\partial\mathbf C}{\partial\mathbf c_i}
=T_i\alpha_iI_3,\qquad
\frac{\partial\mathbf C}{\partial\alpha_i}
=T_i(\mathbf c_i-\mathbf B_i).
$$

$\mathbf B_i$ 是从该高斯后方开始、包括背景的合成颜色，尺度与 $\mathbf c_i$ 相同。两个导数无量纲。颜色导数随遮挡减小；不透明度导数取决于自身与后方的颜色差，因为它改变的是两者之间的权重分配。

再固定二维投影协方差，高斯核对投影中心的导数为：

$$
\nabla_{\mathbf m_i}g_i=g_iA_i(\mathbf p-\mathbf m_i).
$$

该向量的单位为 $\mathrm{px}^{-1}$，可经投影雅可比传到三维中心。完整梯度还包含中心对 $J_i$、协方差和观察方向的影响，不能仅保留移动二维核中心这一路径。

![原论文梯度数量消融：左侧仅最前面10个高斯接收梯度，右侧为完整方法](../outputs/3dgs-paper-course-001/figures/paper-fig-09.png)
> 图源：[Kerbl 等，3DGS（2023），Fig. 9，第 10 页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf#page=10) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

Fig. 9 检验了前向模型与反向梯度是否一致：限制的是每个像素中接收梯度的混合基元数量，而不是把前向图像改为只有十个高斯。后方基元曾参与颜色形成，却得不到相应更新，所用梯度就偏离当前前向模型。原论文报告这种近似使优化不稳定，强调应覆盖所有实际贡献者。

可微关系有适用范围。正深度、正定协方差、非零四元数及固定可见性分支内，上述关系平滑；候选进出、排序交换、贡献或颜色截断会改变分支。可微渲染并不意味着处处光滑，也不保证损失具有唯一正确的三维解。对无量纲损失，中心和标准差梯度的单位为 $L^{-1}$，协方差梯度为 $L^{-2}$；梯度大表示当前目标对参数敏感，不能直接解释为几何误差大。

## 论文实验如何验证这些设计

![原论文各向异性启用与禁用对照，展示完整协方差对细结构和渲染外观的影响](../outputs/3dgs-paper-course-001/figures/paper-fig-10.png)
> 图源：[Kerbl 等，3DGS（2023），Fig. 10，第 11 页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf#page=11) · 裁图 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

原论文第 7.3 节分别移除各项设计。图像指标采用峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）。Table 3 中，Truck、Garden、Bicycle 三场景在 30,000 次迭代后的平均 PSNR，完整方法为 $26.05\,\mathrm{dB}$，各向同性为 $25.23\,\mathrm{dB}$，移除球谐为 $25.35\,\mathrm{dB}$，限制反向梯度为 $19.19\,\mathrm{dB}$。该消融专门从高分辨率输入降采样，结果只在其数据与设置下成立，不是本站成绩或任意场景上的固定增益。

各向异性消融支持的是形状表达能力：单一半径不能同时沿表面展开、沿法向收缩。Fig. 10 的 Ficus 对照将两种配置均限制为最多 5,000 个高斯，使有限容量下的区别可见。球谐消融支持的是方向相关外观对照片拟合的价值，不能单独证明物理材质或光照恢复正确。梯度数量消融则支持前向贡献与反向约束需要一致；Truck 的 30K PSNR 从 $24.81$ 降到 $13.84\,\mathrm{dB}$，不是只有后方局部颜色受到影响。

原论文第 7.4 节还报告了覆盖不足区域的伪影、细长或斑块状高斯，以及大高斯剔除和排序交换引起的画面突变。PSNR、结构相似性（Structural Similarity，SSIM）等首先评价图像重建，不能自动等同于准确的三维表面。

这条渲染关系现在闭合：空间参数给出三维分布，局部投影给出图像核，球谐给出方向相关颜色，透射率决定遮挡权重，损失梯度再约束这些参数。下一章继续解释[参数训练、增密与剪枝](28_3dgs_training_density_control.md)，再进入相机位姿也未知的 SplaTAM。

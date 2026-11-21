"""Canonical content for the two living reports. No experiment-by-experiment decks."""
from pathlib import Path
import json, hashlib, re
import matplotlib
matplotlib.use('Agg')
from matplotlib import mathtext, font_manager
import pymupdf as fitz

LAB=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
HOME=ROOT/'shuolee'
BUILD=HOME/'current/presentation-build'
AS=BUILD/'assets'; AS.mkdir(parents=True,exist_ok=True)
SRC=HOME/'code/presentation-sources'
MS=LAB/'outputs/mini-splatting2-route-001/assets'
PAPER=LAB/'outputs/3dgs-paper-only-002/figures'
BASE=LAB/'outputs/3dgs-foundations-figures'
NATIVE=ROOT/'native-resolution-cache-full-18000-s0/resolution-evaluation-v2/full'
MS_URL='https://arxiv.org/abs/2411.12788v1'
MS_CODE='https://github.com/fatPeter/mini-splatting2/tree/5ec9202c2db7d900728bfc09733e85467cff885e'
MIP_URL='https://niujinshuchong.github.io/mip-splatting/'
PG_URL='https://zju3dv.github.io/pgsr/'
PG_CODE='https://github.com/zju3dv/PGSR/tree/de24f1a38b350387e8d8fe381b2cd70c1ae946e7'
DA_URL='https://github.com/ByteDance-Seed/Depth-Anything-3'
VX_URL='https://kunhao-liu.github.io/ViewExtrapolator/'
DF_URL='https://research.nvidia.com/labs/toronto-ai/difix3d/'
GO_URL='https://gaussianobject.github.io/'

# Complete paper figures, preserving internal panels and annotations.
doc=fitz.open(SRC/'pgsr.pdf')
for key,page,rect in [('pgsr-depth',1,(48,53,563,197)),('pgsr-overview',3,(48,54,563,258))]:
    doc[page].get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(rect),alpha=False).save(AS/(key+'.png'))

EQ={
'axisratio':r's_{ia}=\exp(\ell_{ia}),\quad q_i=\frac{\max_a s_{ia}}{\min_a s_{ia}},\quad v_i=\mathbf1[\rho_i^{2D}>0]',
'axisregularization':r'\mathcal L(t)=\mathcal L_{\rm RGB}(t)+\mathbf1[\,t\ \mathrm{mod}\ 10=0\,]\;\frac{0.1}{N}\sum_{i=1}^{N}v_i\,[\max(q_i,10)-10]',
'groupcontribution':r'G_k(\mathbf u)=\sum_{i\in\mathcal G_k}\alpha_i(\mathbf u)T_i(\mathbf u),\qquad \sum_{k=1}^{4}G_k(\mathbf u)=A(\mathbf u)',
'diagnosticblend':r'\mathbf C(\mathbf u)=\sum_i w_i(\mathbf u)\mathbf c_i+(1-A(\mathbf u))\mathbf b,\qquad A(\mathbf u)=\sum_i w_i(\mathbf u)',
'cov':r'\Sigma_i=\sum_{k=1}^{3}s_{ik}^{2}\mathbf r_{ik}\mathbf r_{ik}^{\mathsf T},\qquad s_{ik}=\exp(\ell_{ik})>0',
'gaussian':r'G_i(\mathbf X)=\exp\!\left[-\frac{1}{2}(\mathbf X-\boldsymbol\mu_i)^{\mathsf T}\Sigma_i^{-1}(\mathbf X-\boldsymbol\mu_i)\right]',
'projection':r'\mathbf x_c=R_{cw}\mathbf X+t_{cw},\quad u=f_xX_c/Z_c+c_x,\quad v=f_yY_c/Z_c+c_y',
'cov2':r'\Sigma_i^{2D}=J_iR_{cw}\Sigma_iR_{cw}^{\mathsf T}J_i^{\mathsf T}',
'blend':r'C(\mathbf u)=\sum_i T_i(\mathbf u)a_i(\mathbf u)c_i+T_{N+1}(\mathbf u)c_{\rm bg}',
'weight':r'a_i=o_iG_i^{2D}(\mathbf u),\qquad T_i=\prod_{j<i}(1-a_j),\qquad w_i=T_i a_i',
'sh':r'c_i(\mathbf d)=\frac12+\sum_{l=0}^{L}\sum_{m=-l}^{l}b_{i,lm}Y_l^m(\mathbf d),\qquad (L+1)^2=16\ (L=3)',
'loss':r'\mathcal L_{\rm RGB}=0.8\,\mathcal L_1+0.2\,[1-\operatorname{SSIM}(C,I)]',
'adam':r'm_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2',
'adamstep':r'\theta_{t+1}=\theta_t-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}',
'lr':r'\eta_{\mu}(t)=r_{\rm scene}\exp[(1-a)\log\eta_0+a\log\eta_1],\quad a=\min(t/30000,1)',
'critical':r'i^*(\mathbf u)=\arg\max_i w_i(\mathbf u),\qquad \mathcal C=\bigcup_{k,\mathbf u}\{i_k^*(\mathbf u)\}',
'mass':r'\frac{\sum_{i\in\mathcal S}I_i}{\sum_i I_i}\simeq0.99,\qquad I_i=\sum_{k,\mathbf u}w_i^k(\mathbf u)',
'clonealpha':r'o_{\rm new}=1-\sqrt{1-o_{\rm old}},\qquad 1-(1-o_{\rm new})^2=o_{\rm old}',
'clonecov':r'\Sigma_{\rm new}=\left[\frac{o_{\rm old}}{2o_{\rm new}-o_{\rm new}^{2}/\sqrt{2}}\right]^2\Sigma_{\rm old}',
'depth':r'd(\mathbf u)=d_{i^*(\mathbf u)}^{\rm mid}(\mathbf u),\qquad \mathbf X_w=R_{wc}\big[dK^{-1}\widetilde{\mathbf u}\big]+t_{wc}',
'depthsample':r'p(\mathbf u)\propto1-A(\mathbf u),\quad A=\sum_iw_i,\quad N_{\rm depth}=1.0\,N_{\rm current}',
'simp':r'p_i=I_i/\sum_jI_j,\qquad N_{\rm sample}=\left\lfloor0.6N\frac{\#\{i:p_i>0\}}{N}\right\rfloor',
'visible':r'I_i^k=\sum_{\mathbf u}w_i^k(\mathbf u),\qquad V_i^k=\mathbf1[I_i^k>\tau_k]',
'importance':r'I_i^{\rm indoor}=\sum_k I_i^k,\qquad I_i^{\rm outdoor}=\sum_{k:\,A_{i,\max}^{k}>0}\frac{I_i^k}{A_{i,\rm proj}^{k}}',
'grad':r'\overline g_i=\frac1{n_i}\sum_{t\in\mathcal V_i}\left\|\nabla_{\boldsymbol\mu_i^{2D}}\mathcal L_t\right\|_2>2\times10^{-4}',
'dilate':r'\Sigma^{2D}_{\rm out}=\Sigma^{2D}+sI_2,\qquad \sigma^{\rm pixel}\propto f\sigma^{\rm world}/z',
'mip3':r'\widetilde\Sigma_i=\Sigma_i+\sigma_{f,i}^2I_3,\quad \widetilde o_i=o_i\sqrt{\frac{\det\Sigma_i}{\det\widetilde\Sigma_i}}',
'mip2':r'\widehat\Sigma_i^{2D}=\Sigma_i^{2D}+sI_2,\quad \widehat o_i=\widetilde o_i\sqrt{\frac{\det\Sigma_i^{2D}}{\det\widehat\Sigma_i^{2D}}}',
'depthdef':r'\widehat D_{\rm exp}(\mathbf u)=\frac{\sum_iw_i z_i}{\sum_iw_i+\varepsilon},\qquad r=Z\|K^{-1}\widetilde{\mathbf u}\|_2',
'align':r'(a^*,b^*)=\arg\min_{a,b}\sum_j q_j\rho(aD_j^{\rm prior}+b-Z_j^{\rm SfM})',
'depthloss':r'\mathcal L_D=\frac{\sum_{\mathbf u\in\Omega}q(\mathbf u)\rho[\widehat D(\mathbf u)-D^*(\mathbf u)]}{\sum_{\mathbf u\in\Omega}q(\mathbf u)+\varepsilon}',
'plane':r'\mathbf n^{\mathsf T}\mathbf x+d=0,\qquad \mathbf x=ZK^{-1}\widetilde{\mathbf u},\qquad Z=-\frac d{\mathbf n^{\mathsf T}K^{-1}\widetilde{\mathbf u}}',
'pgdepth':r'\mathbf N=\sum_iw_i\mathbf n_i,\quad D_{\rm plane}=\sum_iw_id_i,\quad Z=-D_{\rm plane}/(\mathbf N^{\mathsf T}\mathbf r)',
'normal':r'\mathbf n_D\propto[\mathbf P(u+1,v)-\mathbf P(u-1,v)]\times[\mathbf P(u,v+1)-\mathbf P(u,v-1)]',
'scale':r'\mathcal L_s=\frac1N\sum_i\min(s_{i1},s_{i2},s_{i3}),\quad\mathcal L_N=\frac1{|\Omega|}\sum_{\mathbf u}q_I\|\mathbf n_D-\widehat{\mathbf n}\|_1',
'reproj':r'\mathbf u\longrightarrow\mathbf X(D_k)\longrightarrow\mathbf u_l\longrightarrow\mathbf X(D_l)\longrightarrow\mathbf u_k^\prime,\quad e=\|\mathbf u_k^\prime-\mathbf u\|_2',
'homography':r'H_{lk}=K_l\left(R_{lk}-\frac{t_{lk}\mathbf n_k^{\mathsf T}}{d_k}\right)K_k^{-1}',
'ncc':r'\operatorname{NCC}(P,Q)=\frac{\sum_j(P_j-\bar P)(Q_j-\bar Q)}{\sqrt{\sum_j(P_j-\bar P)^2\sum_j(Q_j-\bar Q)^2}+\varepsilon}',
'pgsum':r'\mathcal L=\mathcal L_{\rm RGB}+100\mathcal L_s+0.015\mathcal L_N+0.03\mathcal L_G+0.15\mathcal L_P',
'hull':r'\mathcal H=\bigcap_{k\in\mathcal T}\{\mathbf X:\pi_k(\mathbf X)\in M_k\}',
'diffusion':r'\mathbf z_t=\alpha_t\mathbf z_0+\sigma_t\boldsymbol\epsilon,\qquad \boldsymbol\epsilon_\theta(\mathbf z_t,t,\mathcal C)',
'feedback':r'\mathcal L_{\rm total}=\mathcal L_{\rm obs}+\lambda_D\mathcal L_D+\lambda_{\rm mv}\mathcal L_{\rm mv}+\lambda_{\rm gen}\mathcal L_{\rm gen}',
'genmask':r'\mathcal L_{\rm gen}=\frac{\sum_{k,\mathbf u}q_k(\mathbf u)\rho[C_\Theta(k,\mathbf u)-I_k^{\rm gen}(\mathbf u)]}{\sum_{k,\mathbf u}q_k(\mathbf u)+\varepsilon}',
'psnr':r'\operatorname{PSNR}=10\log_{10}\frac1{\operatorname{MSE}},\qquad\operatorname{MSE}=\frac1{3|\Omega|}\sum_{\mathbf u\in\Omega}\|C-I\|_2^2',
'smpl':r'\mathbf x_t=\sum_{j=1}^{J}w_j(\mathbf x)T_j(\boldsymbol\theta_t,\boldsymbol\beta)\widetilde{\mathbf x},\qquad\sum_jw_j=1',
'coverage':r'\operatorname{coverage}=|\Omega_{\rm valid}|/|\Omega_{\rm requested}|',
'headcontribution':r'I_i^H=\sum_{k\in\mathcal T}\;\sum_{\mathbf u\in H_k}\;T_i^k(\mathbf u)\alpha_i^k(\mathbf u)',
'headrigid':r'\boldsymbol\mu_i(t)=R_h(t)\boldsymbol\mu_i^0+\mathbf t_h(t),\qquad\Sigma_i(t)=R_h(t)\Sigma_i^0R_h(t)^{\mathsf T}',
'blendalpha':r'\mathbf C(\mathbf u)=\sum_i T_i(\mathbf u)\alpha_i(\mathbf u)\mathbf c_i+(1-A(\mathbf u))\mathbf b,\qquad A(\mathbf u)=\sum_i T_i(\mathbf u)\alpha_i(\mathbf u)',
'dropgaussian':'m_{i,t}\\sim\\operatorname{Bernoulli}(1-p_t),\\qquad \\widetilde o_{i,t}=\\frac{m_{i,t}o_i}{1-p_t},\\qquad p_t=0.2\\frac{t}{18000}',
'backgroundalpha':r'\mathcal L=0.8\mathcal L_1+0.2(1-\operatorname{SSIM})+\frac{\lambda_{\rm bg}}{|\mathcal B|}\sum_{\mathbf u\in\mathcal B}A(\mathbf u),\qquad \lambda_{\rm bg}\in\{0,0.05\}',
}
for name,eq in EQ.items():
    eq=re.sub(r'\\(mathbf|mathsf|mathcal|mathbb|boldsymbol|widehat|widetilde|overline|bar|hat)\s*(\\[A-Za-z]+|[A-Za-z0-9])',r'\\\1{\2}',eq)
    eq=eq.replace(r'\frac12',r'\frac{1}{2}').replace(r'\frac d{',r'\frac{d}{')
    eq=re.sub(r'\\big(?![a-zA-Z])','',eq)
    eq=re.sub(r'\\frac([A-Za-z0-9])',r'\\frac{\1}',eq)
    eq=re.sub(r'(\\frac\{[^{}]+\})([A-Za-z0-9])',r'\1{\2}',eq)
    mathtext.math_to_image('$'+eq+'$',AS/('eq-'+name+'.png'),prop=font_manager.FontProperties(size=26),dpi=300,color='#183342')

decks=[]
def new_deck(key,title,filename):
    d={'key':key,'title':title,'filename':filename,'slides':[]}; decks.append(d); return d['slides']
def add(slides,key,title,body=(),eq=None,images=(),caption='',source=(),notes='',layout=None,chart=None,subtitle=''):
    if not source: source=[MS_URL,MS_CODE] if slides is A else []
    slides.append({'key':key,'title':title,'body':list(body),'eq':([eq] if isinstance(eq,str) else eq or []),
        'images':[str(x) for x in images],'caption':caption,'sources':list(source),'notes':notes,
        'layout':layout or ('figure' if images else 'formula' if eq else 'text'),'chart':chart,'subtitle':subtitle})

A=new_deck('mini-splatting2','Mini-Splatting2 训练原理与复现','01_Mini-Splatting2_训练原理与复现.pptx')
add(A,'cover','Mini-Splatting2\n训练原理与复现',['从稀疏点到高效场景表示','完整训练流程、公式、参数与实验依据','shuolee · 持续维护版本'],images=[MS/'fig1.png'],layout='cover',caption='Fang & Wang, Mini-Splatting2 (2024), Fig. 1 完整原图。')
add(A,'scope','论文、实现与本地工程的对应关系',[
'方法依据：Mini-Splatting2 原始论文 v1（2024）及发布代码。未指定版本的 arXiv 页面已有后续扩展，本文固定原版语义。',
'官方核心：激进增密、深度重初始化、两次简化、按视图可见性裁剪。训练主体仍是可微高斯渲染。',
'本地派生：运动区域筛选、Mip 滤波、深度监督与全分辨率训练。它们属于独立改动，分别说明和验证。'],notes='官方代码固定提交 5ec9202c2db7d900728bfc09733e85467cff885e。官方论文结果与本机 RTX 5080 结果分开。后续原理不将本地 Mip 或深度约束归为 Mini-Splatting2 原设计。')
add(A,'motivation','提前建立几何容量，再压缩与精修',[
'图中上排显示高斯中心，下排显示从当前表示提取的深度点。训练早期的深度点已显现物体结构。',
'核心设计：利用这些早期几何信息快速增加有效高斯，再通过贡献筛选控制最终规模。'],images=[MS/'fig2.png'],caption='Fang & Wang (2024), Fig. 2，保留全部迭代阶段与上下两排。')
add(A,'inputs','训练输入建立统一的几何坐标',[
'输入是同一近似静态场景的多视图 RGB、相机内外参，以及 SfM 稀疏点。图像去畸变后使用一致的针孔相机模型。',
'先固定训练与留出视图，再提取、初始化、选择邻居和拟合先验。测试照片不进入训练监督或先验拟合。',
'图像缩放时同步缩放焦距与主点。世界长度沿用 SfM 单位，单目 SfM 默认不提供绝对米制尺度。'],eq='projection',notes='相机坐标采用 x 向右、y 向下、z 向前。R_cw,t_cw 把世界点变到相机坐标。f_x,f_y,c_x,c_y 的单位均为像素。列向量公式和代码中转置矩阵须对应。')
add(A,'parameters','每个高斯同时保存形状与外观',[
'位置 μ 为三维中心。三个正尺度 s 与单位四元数 q 决定各向异性椭球。rₖ 是旋转矩阵的第 k 列。',
'不透明度 o 由 sigmoid 参数化，范围为 0 到 1。尺度使用指数参数化，保证协方差正定。',
'外观使用 RGB 球谐系数。位置、尺度、不透明度、旋转、外观进入不同学习率的优化组。'],eq='cov',notes='原代码四元数顺序为 (w,x,y,z)，归一化后构造旋转矩阵。s 的单位是 SfM 世界长度，Σ 为长度平方。不透明度无单位。协方差使用展开的特征方向求和，未使用代码化 diag。')
add(A,'projection','协方差随相机一起投影到像平面',[
'J 是在高斯中心处计算的透视投影雅可比。变换后的二维协方差决定屏幕椭圆的方向与大小。',
'高斯中心决定投影均值，协方差决定像素覆盖范围。相同三维尺度在近距离通常覆盖更多像素。',
'训练梯度通过二维投影回传到三维位置和形状，这是图像损失能够更新几何的通路。'],eq='cov2',source=['https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/',MS_CODE],notes='J 第一行：(f_x/Z,0,-f_x X/Z²)；第二行：(0,f_y/Z,-f_y Y/Z²)。本页为 3DGS 共有基础，仅解释 Mini-Splatting2 训练所需部分。')
add(A,'composite','遮挡与颜色共同决定像素贡献',[
'按深度顺序累积贡献。前方高斯降低后方高斯的透过率，最终余量乘以背景色。',
'wᵢ 同时包含投影覆盖、不透明度与前方遮挡。它成为后续关键高斯识别、简化和可见性计算的共同依据。'],eq=['blend','weight'],notes='o_i 表示高斯基准不透明度，a_i(u) 表示该像素处实际 alpha，避免把两者混为一谈。实际光栅器含 alpha 截断、贡献阈值和提前终止，公式说明连续模型。')
add(A,'appearance','球谐阶数随训练逐步增加',[
'每个颜色通道在三阶时含 16 个系数，共 48 个外观参数。常数项先学习主体颜色，高阶项表达视角相关变化。',
'官方稀疏版本在前 3,000 步保持零阶。第一次简化后扩展系数，4,000／5,000／6,000 步逐步升至三阶。',
'这样把早期显存与优化容量优先用于几何，再加入更丰富的方向外观。'],eq='sh',notes='代码 c 的球谐计算加 0.5，并对负值做截断；通用数学定义与实际颜色域要对应。默认 sh_degree=3，开始实例化 sh_degree=0；3k extend_features_rest，随后每 1k oneupSHdegree。')
add(A,'initialization','稀疏点初始化为可优化高斯',[
'中心取 SfM 点，常数颜色取稀疏点 RGB。尺度根据近邻距离建立初始空间覆盖，初始旋转设为单位四元数。',
'默认初始不透明度为 0.1。可见高斯参与渲染与梯度更新，模型逐步从点云颜色转为图像监督外观。',
'在 2,000 步深度重初始化时，这套初始化再次执行，并重建优化器状态。'],eq='gaussian',notes='scene/gaussian_model.py:create_from_pcd 与 reinitial_pts。distCUDA2 计算近邻距离量，clamp_min=1e-7，scale 参数取 log(sqrt(dist2))。位置可导，高阶 SH 初始为零。原始 SfM 文件保持不变。')
add(A,'loss','每一步以真实照片计算重建损失',[
'L₁ 约束逐像素颜色误差，结构相似性 SSIM 约束局部亮度、对比度与结构。官方权重 λ = 0.2。',
'每次迭代抽取一个训练相机，完成前向渲染、损失、反向传播，再更新可见高斯。',
'训练相机列表无放回抽取，遍历完后重新生成列表。照片曝光和颜色处理需要保持一致。'],eq='loss',notes='官方使用 fused_ssim，loss=0.8*Ll1+0.2*(1-ssim_value)。这是图像域监督；训练监控 SSIM 不等于三维几何误差。代码末次迭代不再执行 optimizer.step，应按实际保存流程核对复现。')
add(A,'optimizer','分组 Adam 控制不同参数的更新速度',[
'梯度先形成一阶、二阶动量，再按参数组学习率更新。稀疏 Adam 只处理本轮可见项，减少无效工作。',
'颜色和形状处在不同数值尺度，使用同一个学习率容易使某一类参数主导优化。',
'增点、删点与重初始化时同步调整参数数组和优化状态，保证参数与动量一一对应。'],eq=['adam','adamstep'],notes='SparseGaussianAdam 源自 diff_gaussian_rasterization_ms。eps=1e-15，参数组 xyz/f_dc/f_rest/opacity/scaling/rotation。标准 Adam 公式解释机制，具体稀疏内核的状态更新以发布代码为准。')
add(A,'rates','官方 fast 配置的学习率',[
'位置：初始 1.6×10⁻⁴，终值参数 1.6×10⁻⁶，再乘场景尺度。调度长度仍为 30,000，fast 总步数为 18,000。',
'颜色：常数项 0.0025，高阶项 0.000125。不透明度：0.025。',
'尺度：0.005。旋转：0.001。复现时保存最终解析后的参数，不能只抄默认总步数。'],eq='lr',notes='config/fast 覆盖总迭代与增密节点，但不覆盖 position_lr_max_steps。18k 时尚未到达该调度器 30k 的终值。scene/gaussian_model.py 使用 spatial_lr_scale。get_expon_lr_func 默认 delay_steps=0，因此本式表示实际无延迟主体。')
add(A,'schedule','18,000 步中的四段训练职责',[
'1–3,000：半宽、半高预热，常规增密与激进克隆并行；2,000 步执行深度重初始化。',
'3,000–8,000：第一次重要性采样后继续优化，转回目标分辨率，逐步提升球谐阶数。',
'8,000–13,000：第二次贡献保留简化后精修。13,000–18,000：关闭预计算可见性屏蔽，完成最终优化。'],notes='getTrainCameras_warn_up 使用 iteration<=3000 对应 resolution_scale=2。视图栈在刷新时选择分辨率，实际切换可能延后到下一轮栈刷新。两次简化在3k和8k，13k=(8k+18k)//2。不把预热分辨率写成固定像素数。')
add(A,'density-original','常规增密提供局部误差驱动',[
'累计屏幕空间位置梯度。超过阈值的区域需要更多表示能力，小高斯克隆、大高斯分裂。',
'默认每 100 步检查一次，起始条件是步数大于 500。fast 配置将增密截止提前至 3,000。',
'官方还记录覆盖面积过大的模糊高斯。模糊区域分裂与不透明度裁剪配合，重新分配容量。'],eq='grad',notes='阈值 0.0002，percent_dense=0.01；min opacity=0.005。mask_blur 使用 area_max>H*W/5000。深度重初始化步跳过通常增密。opacity_reset_interval 是参数名，不能因此断言本训练循环每3k主动重置不透明度。')
add(A,'critical','关键高斯来自像素最大混合贡献',[
'对每个训练像素找到混合贡献最大的高斯，再跨视图汇总，优先扩充已解释主要表面的高斯。',
'累计贡献衡量影响大小，是否曾成为最大贡献者限制候选集合。这个依据直接联系渲染中的遮挡关系。',
'每 250 步执行一次关键识别与克隆，从第 500 步开始，跳过第 2,000 步。'],eq='critical',notes='论文Eq.3使用最大贡献近似说明关键集合，实际渲染仍使用完整alpha混合。代码先累计accum_weights和area_max，area_max总和为0的候选贡献清零，再以CDF贡献阈值筛选。不是直接把渲染改为最近点或最大权重点。')
add(A,'cdf','99% 是贡献累计阈值',[
'将高斯按重要性排序，剔除累计贡献很小的尾部。保留数量由场景的贡献分布决定。',
'关键克隆和逐视图可见性使用约 0.99 的贡献质量。前置全局弱贡献筛选还使用 0.999。',
'代码另外剔除有效可见视图数不超过 1 的高斯，形成多视图支持约束。'],eq='mass',notes='init_cdf_mask 调用 get_important_ids，升序累加后按累计贡献与总量的比例确定阈值。论文将quantile用于阈值描述，实现不是简单torch.quantile保留固定数量。严格边界和同值处理由源码决定，故式中用约等于。')
add(A,'clone-opacity','激进克隆保持中心处的不透明度',[
'原高斯与新增副本使用同一个新不透明度，中心位置保持一致。两者组合后中心 alpha 与原来相同。',
'例如原不透明度为 0.75，新值为 0.5。两层累积为 1−(1−0.5)²=0.75。',
'减少一次大量克隆引起的遮挡突变，让新增容量继续接受梯度分化。'],eq='clonealpha',notes='原高斯和新高斯都改 opacity。代码将新opacity夹在[0.0051,1-eps]后转回logit，极小值附近因截断不再严格满足等式。本页先解释中心约束。')
add(A,'clone-shape','尺度修正配合克隆的透明度变化',[
'新协方差按共同系数缩放，原高斯与副本沿用位置、旋转与颜色。尺度修正调节克隆后空间覆盖。',
'源码存储标准差，所以对 s 乘一次系数；协方差 Σ 的公式需要平方该系数。',
'克隆完成后继续优化，两个高斯可以在后续梯度下分化。'],eq='clonecov',notes='对应论文Sec.4和gaussian_model.py:clone。公式源于状态转移近似，中心alpha等式精确，但不能宣称整个椭圆内所有像素颜色/梯度完全不变。该适用边界在最后不足页总结。')
add(A,'depth-reinit','由最大贡献高斯计算深度并反投影',[
'对一条像素射线，选取混合贡献最大的高斯，用射线与该高斯椭球交点的中点深度重建世界坐标点。',
'这些点利用当前高斯表示形成更密集的表面覆盖。点颜色来自对应训练照片。',
'这一步是表示自身提取的几何，不依赖额外的单目深度网络。'],eq='depth',notes='这里d应按render_depth/out_pts的实际坐标约定解释。它不是常见归一化期望中心深度，也不是深度传感器GT。外参式中R_wc=R_cw^T,t_wc=-R_cw^T t_cw。')
add(A,'depth-budget','深度重初始化控制采样数和覆盖',[
'第 2,000 步按当前高斯数设置重建预算，默认数量因子为 1。各训练视图分配近似相同的采样数量。',
'采样概率与剩余透过率相关，使低累积不透明度区域得到更多机会。采样采用无放回方式。',
'重建中心、尺度、旋转、不透明度与颜色参数后，重新初始化优化器和逐视图屏蔽矩阵。'],eq='depthsample',notes='num_depth在intersection_preserving前从当前数量计算。每视图num_sampled取整，合计可能略少于请求预算。存在低置信度或背景深度时需检查out_pts有效性。本地修改不能混同为官方额外深度监督。')
add(A,'density-effect','早期增密显著改变收敛过程',[
'原论文同时给出高斯数量与渲染质量的迭代变化。激进策略在早期提供更大表示容量，随后通过简化回收。',
'阅读时把增密节点、简化节点和曲线一起看，理解计算量如何在阶段间分配。'],images=[MS/'fig3.png'],caption='Fang & Wang (2024), Fig. 3 完整原图；bicycle 示例。')
add(A,'simplify-one','第一次简化按重要性无放回采样',[
'3,000 步后，将已有容量压缩为更紧凑的高斯集合。每个高斯的采样概率与其累计渲染贡献相关。',
'默认采样因子为 0.6，并按非零贡献高斯比例修正采样数量。弱多视图支持项同时剔除。',
'简化后扩展球谐特征并重建优化器，进入目标分辨率的外观精修。'],eq='simp',notes='对应culling_with_interesction_sampling。prob=imp_score/sum；numpy.random.choice(...replace=False)。不是按0.6固定保留最终所有高斯，也不是随机均匀删点。indoor/outdoor重要性定义见下一页。')
add(A,'importance-scene','室内与室外采用不同重要性聚合',[
'室内直接累计渲染贡献。室外对有效最大贡献区域使用投影面积归一化，控制大覆盖高斯在统计中的影响。',
'两种度量都服务于简化与保留集合，场景类别是原作者的经验选项。',
'复现时固定该选择，改变度量要作为独立对照，观察远景、背景和细结构的变化。'],eq='importance',notes='代码在outdoor情况下只对area_max非零视图更新，accum_weights/area_proj。最终总area_max为零的贡献置零。不是用世界体积归一化。')
add(A,'simplify-two','第二次简化保留主要交点贡献',[
'8,000 步再次统计所有训练视图的贡献，按累计贡献阈值保留关键集合。',
'与第一次概率采样不同，这一步主要使用贡献阈值保留，加上多视图支持筛选。',
'后续继续优化已有高斯，完成质量和模型规模之间的折中。'],eq='mass',notes='culling_with_interesction_preserving默认thres=.99，count_vis<=1剔除。此时fast增密已在3k结束。稀疏MSv2与密集MSv2-D是不同配置/训练分支，报告中不能混用高斯数和质量。')
add(A,'visibility','可见性裁剪进一步缩小每视图的工作集',[
'视锥裁剪先确定屏幕可能覆盖的高斯，贡献可见性再排除对该训练视图几乎无影响的高斯。',
'图中给出两类模型在同一视图的保留范围与数量。按视图计算，可将运算集中到实际贡献项。'],images=[MS/'fig4.png'],caption='Fang & Wang (2024), Fig. 4，Frustum Culling 与 Visibility Culling 的完整对照。')
add(A,'visibility-training','可见性掩码随关键阶段刷新',[
'每个训练视图保存一个高斯可见集合。克隆与简化阶段一起重算贡献，并同步参数与掩码索引。',
'仅对当前视图的保留集合参与光栅化。被屏蔽的项不参与该步渲染贡献，减少前后向成本。',
'第 13,000 步将屏蔽清零，在最终精修阶段恢复普通视锥规则。'],eq='visible',notes='visibility_mask=True为保留；内部_culling保存其反值，True表示剔除。使用count_vis/(count_rad+0.1)校正被屏蔽期间累计的增密梯度，防止可见次数变化直接扭曲阈值。此布尔含义容易读反。')
add(A,'resource','早期外观降阶让几何容量更经济',[
'零阶 RGB 只需 3 个外观系数，三阶需要 48 个。高斯数量快速增加时，延后高阶外观能够降低参数与优化状态开销。',
'每视图可见性减少活跃计算量，稀疏优化与融合 SSIM 降低实现开销。',
'显存验收同时记录高斯数、光栅化临时缓冲与峰值分配，最终 PLY 文件大小仅反映存储的一部分。'],notes='论文Sec.6.2讨论SH属性48/59的占比。官方继承Taming-3DGS的加速实现，不能把全部加速都归因于激进增密。本页解释工程代价，未给出本机与论文硬件的跨机速度结论。')
add(A,'paper-effects','原论文效果显示质量与规模的折中',[
'对照包含 3DGS 加速版、Mini-Splatting 系列、MSv2-D 与 MSv2。完整图保留原始列与场景。',
'MSv2 面向紧凑模型，MSv2-D 保留更密集表示。两者按实际用途分别选择。'],images=[MS/'fig5.png'],caption='Fang & Wang (2024), Fig. 5 完整原图，图中数值属于论文实验。')
add(A,'ablation-time','组件消融说明训练加速来自哪里',[
'相同论文实验中，激进增密把 Mini-Splatting* 的训练时间从 479 s 降到 244 s，可见性裁剪再降到 214 s。',
'对应 PSNR 为 27.23、27.43、27.33 dB。收益集中在训练效率，同时保留接近的图像质量。'],chart={'categories':['Mini-Splatting*','+ 激进增密','+ 可见性裁剪'],'values':[479,244,214],'name':'训练时间 / s'},layout='chart',notes='数据来自原论文Table3，Mip-NeRF360平均，RTX3090，IntelXeonPlatinum8383C。Mini-Splatting*是Mini-Splatting与3DGS-accel结合。不是本机测试，不能把这些时间外推到工作视频。')
add(A,'ablation-density','完整增密策略比单纯缩短训练更有效',[
'同一消融中，仅将常规增密截止改为 3K，PSNR 为 26.49 dB；降低梯度阈值为 26.58 dB；完整激进策略为 27.33 dB。',
'对应关键变化包括贡献选择、克隆状态修正、深度重初始化和后续简化，不能以“多加点”概括。'],chart={'categories':['仅缩短增密','降低梯度阈值','完整激进增密'],'values':[26.49,26.58,27.33],'name':'PSNR / dB'},layout='chart',notes='原论文Table4，Mip-NeRF360，其他预算按论文设置。本页只比较论文结果。不是单一参数对所有场景的最优值证明。')
add(A,'local-effects','本地工作素材的真实视角渲染',[
'固定工作视频、去畸变相机和 51 个训练／8 个开发视图。当前全分辨率参考模型为 18,000 步、316,044 个高斯。',
'照片与渲染使用同一相机，主体细节和完整场景同时检查。该参考已包含 Mip 与原深度配置，属于派生版本。'],images=[NATIVE/'000016-gt.png',NATIVE/'000016-render.png'],caption='左：真实开发照片；右：本地全分辨率参考渲染。1514 × 2694，工作视频 HOVER_VERSA_0102。',source=[str(NATIVE),str(ROOT/'native-resolution-cache-full-18000-s0/model')],notes='8个开发视图曾反复用于开发，不作为独立测试。该图用于展示当前输出，不声明解决大幅俯仰。原模型SHA256 0bf77062ffc33b3b364ae98a2c2853a2605d874b50f477c60cd32a3213100111。')
add(A,'local-detail','同一帧的人物细节单独验收',[
'头部裁剪使用同一像素区域，保留源分辨率。重点观察面部结构、眼镜、发际线和衣领连续性。',
'与完整场景并看，防止只优化人物局部而牺牲背景，也防止只凭整体指标掩盖主体模糊。'],images=[NATIVE/'000016-head-gt.png',NATIVE/'000016-head.png'],caption='同一开发帧：左为照片局部，右为本地参考渲染局部；无生成增强。',source=[str(NATIVE)],notes='局部框包含颈部和少量背景，后续所有候选应复用同一裁剪。此图不是新的独立样本，PPT中保留完整场景图供上下文对应。')
add(A,'acceptance','复现验收同时覆盖图像与空间行为',[
'真实留出视图：主体、头部、背景分别报告 PSNR、SSIM、LPIPS，并保留同相机效果图。',
'连续路径：物理距离、焦距与输出分辨率分开变化。俯仰路径检查轮廓、遮挡关系、闪烁和漂浮层。',
'可复现工程：固定代码、配置、图像拆分与随机种子，保存必要基线、当前模型和通过验收的最佳模型。'],eq='psnr',notes='像素强度按[0,1]定义PSNR，缩放、掩码和有效覆盖固定。若没有该路径真实照片，可做连续观感与几何一致性评价，但不能填写伪造GT误差。')
add(A,'solves','Mini-Splatting2 解决的核心问题',[
'缩短单场景高斯优化时间：早期高效扩充有贡献的表示，再简化和精修。',
'降低最终模型冗余：贡献引导的两次简化形成紧凑高斯集合，方便后续查看与部署。',
'提供可扩展基线：贡献统计、深度重初始化和可见性规则可与采样、几何和监督模块分开研究。'],notes='这里是方法设计目标及论文证据支持的范围。本项目以其作为基础表示和效率基线，不把其论文360场景结果视为有限水平人物视频的验收。')
add(A,'limits','当前方案的不足与下一轮接口',[
'观察范围有限时，未观测的头顶、下颌和背面结构仍缺少真实约束。高图像分数不能唯一证明三维位置正确。',
'克隆状态修正保持中心透明度并近似调节覆盖，并非整个投影处处严格等价；激进早期增密也会形成较高峰值显存。',
'本地仍需验证尺度变化、表面几何和俯仰外推。静态通过后再引入时间与人体形变，优化细节统一更新第二份报告。'],notes='论文Sec.7讨论峰值显存与几何质量；附录指出稀疏重叠背景MVS更稳健。有限视角人物问题是本项目扩展范围。当前不存在已通过大幅俯仰验收的结论。')

B=new_deck('reconstruction-optimization','有限视角重建优化路线','02_有限视角重建优化路线.pptx')
add(B,'cover','有限视角重建\n优化路线',['从尺度采样、表面几何到连续新视角补全','以 Mini-Splatting2 为参考，按问题逐项验证','shuolee · 持续维护版本'],images=[HOME/'code/pgsr/assets/teaser.jpg'],layout='cover',caption='Chen 等，PGSR 官方项目图。场景示例属于论文，不是本地训练结果。',source=[PG_URL])
add(B,'objective','目标是可直接查看的三维重建质量',[
'首先处理近似静态人物：远近观察清晰、上下观察结构稳定、漂浮物可控，并保持背景质量。',
'训练后的同一组三维参数应能沿连续相机路径直接渲染。主体外观、遮挡关系和空间位置一起验收。',
'动态人物放在静态验收之后，通过时间与形变表示接入。最终效果由真实照片、连续观察和几何证据共同判断。'],source=[str(HOME/'long-term-plan.md')])
add(B,'input-audit','相机与图像先建立一致关系',[
'逐项核对去畸变图、内参缩放、世界到相机变换和查看器投影。相同相机应投影到相同像素。',
'裁剪、缩放或改变主点时同步变更相机内参。深度或法线图沿用同一坐标域。',
'局部检查人物特征轨迹、视差和遮挡，再决定更换相机估计或增加几何监督。'],eq='projection',source=['https://colmap.github.io/tutorial.html',str(HOME/'long-term-plan.md')],notes='本地数值投影与COLMAP差约2.45e-12px只排除了该数值变换，不等于完成浏览器像素QA或证明所有外参真实正确。')
add(B,'scale-types','三类尺度变化分别控制',[
'移动相机改变相机到物体的距离，同时改变遮挡和所见表面。改变焦距主要改变视场与像素采样密度。',
'改变输出分辨率也改变像素采样尺度。每次实验只改变一项，并同步内参缩放。'],images=[PAPER/'mip-fig-03-camera-sampling.png'],caption='Yu 等，Mip-Splatting (CVPR 2024), Fig. 3 完整原图。',source=[MIP_URL])
add(B,'mip-why','Mip 滤波使采样尺度与高斯覆盖一致',[
'普通二维扩张扩大屏幕椭圆。缩小时，如果不校正积分能量，像素可能过亮或轮廓膨胀。',
'Mip-Splatting 同时限制三维高频并校正二维采样足迹，面向不同分辨率、焦距和距离的渲染。'],images=[PAPER/'mip-fig-01-projection-dilation.png'],caption='Yu 等，Mip-Splatting (2024), Fig. 1 完整原图。',source=[MIP_URL])
add(B,'mip-3d','三维平滑限制训练视图可支持的频率',[
'每个高斯从训练相机的采样密度估计滤波尺度 σf。原协方差加上各向同性滤波项。',
'行列式比例调整不透明度，使扩大覆盖后的积分贡献保持相应尺度。',
'密集观察能够支持更细结构，低采样区域使用更宽的平滑。滤波参数随增密和位置变化按实现更新。'],eq='mip3',source=[MIP_URL,'https://github.com/autonomousvision/mip-splatting'],notes='等式为高斯与归一化高斯卷积的闭式形式。σf的代码实现通常按训练视图可见深度/焦距与常数得到，不能把局部项目参数无条件视为论文通用最优。3D滤波并非改变真实观测覆盖。')
add(B,'mip-2d','二维 Mip 滤波校正像素积分',[
'投影后协方差增加像素滤波核，s 的单位为像素平方。渲染时按当前分辨率与焦距计算屏幕覆盖。',
'二维行列式比补偿幅值，防止只扩张椭圆而不补偿透明度。',
'训练、离线渲染和通用查看器必须使用相同滤波语义，加载普通 PLY 时还要保存相关参数。'],eq='mip2',source=[MIP_URL],notes='二维s与世界空间σf不能混用。需区别原始协方差、融合3D滤波后的协方差，以及每次渲染的2D滤波。避免同时预融合与运行时重复加滤波。')
add(B,'mip-test','尺度实验固定三维模型与相机路径',[
'分辨率组：同相机分别以半分辨率、基准和双倍采样渲染。焦距组：位置固定，改变焦距。',
'距离组：焦距固定，沿预先确定的路径移动相机。人物和背景一起保留，检查细节、边缘与闪烁。',
'对照包括未滤波与正确滤波，并核对透明度、SH、颜色域和背景。只替换滤波时保持其他模块相同。'],source=[MIP_URL,str(HOME/'camera-paths.json')],notes='现有比较路径各60帧，576×1024。下一次实际受控Mip消融需从同源配置和预算启动，旧历史模型多项修改不能直接作单模块因果结论。')
add(B,'depth-definition','深度监督先确定量的定义',[
'轴向深度 Z、相机到点的距离 r、逆深度 1/Z 具有不同数值和误差尺度。先验与渲染必须对齐同一物理量。',
'透明度加权的高斯中心期望深度可以处于多个半透明层之间。表面射线交点深度需要不同计算。',
'记录深度有效像素与累积不透明度，同步报告覆盖率。'],eq='depthdef',source=[PG_URL,str(HOME/'long-term-plan.md')],notes='r=Z*||K^-1 u||使用未归一化射线z分量为1。当前PGSR对照比较plane_depth与MiniGS期望中心深度，定义不同必须显式标注。')
add(B,'depth-alignment','训练几何为先验提供尺度与偏移',[
'从训练 SfM 对应点取得轴向深度，用置信度与鲁棒误差拟合先验尺度。不同先验分别判断是否需要偏移项。',
'对齐点与诊断点按固定规则分开，避免用同一批点拟合后只报告拟合残差。',
'对齐在模型实际输出域进行。深度域线性变换与逆深度域线性变换不是同一个映射。'],eq='align',source=[DA_URL,str(HOME/'long-term-plan.md')],notes='若先验是米制深度或已知相机联合预测，先验证其尺度定义，再决定是否允许自由偏移。SfM尺度本身不是绝对真值。当前SfM点参与过模型初始化，残差仍是开发诊断。')
add(B,'depth-confidence','可靠深度通过带权损失进入训练',[
'q 综合有效性、跨视图一致性、遮挡、深度边缘以及局部 SfM 冲突。错误区域降低权重或不监督。',
'损失更新位置、尺度和透明度时分组观察梯度，防止仅通过透明度调整降低深度误差。',
'λD 从小规模受控对照确定，同时保留真实 RGB 锚点。几何改善与外观改善分别检查。'],eq='depthloss',source=[DA_URL,str(HOME/'long-term-plan.md')],notes='Huber/Charbonnier等ρ属于可选鲁棒损失，本页是本项目统一监督模板，不声称等同DA3网络训练目标。q不是经过校准的概率，筛选覆盖不能代替整体质量。')
add(B,'da3-role','多视图深度先验提供跨帧联合信息',[
'DA3 可输入已知相机，联合预测多个视图。当前实验固定原相机，比较先验质量而不同时替换位姿。',
'本地已完成 51 个训练视图、504×280 的 DA3-BASE 推理，保留相机、深度、置信度与筛选规则。',
'先用于粗几何与前后关系。高分辨率外观仍由原始照片监督，面部和手指由专门区域检查。'],source=[DA_URL,str(HOME/'current/da3-refinement')],notes='同一高置信度头部SfM保留点上DA3相对误差中位数1.3129%，p90 4.6778%。先验指标与后续3D改进分开，本轮短程3D负结果在末尾汇总。')
add(B,'pgsr-route','PGSR 将表面表示和几何约束连在一起',[
'先将高斯压平，再渲染平面距离与法线，得到射线与表面的深度。',
'单视图法线一致性和多视图几何／光度约束联合优化表面。曝光补偿是可选分支。'],images=[AS/'pgsr-overview.png'],caption='Chen 等，PGSR，Fig. 4 完整原图（arXiv:2406.06521v2）。',source=[PG_URL,'https://arxiv.org/abs/2406.06521v2',PG_CODE])
add(B,'pgsr-plane','平面深度取决于射线与表面的交点',[
'平面法线 n 指定方向，d 是带符号平面偏移。对像素反投影得到射线 r，再计算射线和平面的交点。',
'同一个倾斜平面在不同像素有不同深度。该关系同时依赖位置、旋转与相机。',
'代码采用自己的法线方向约定，读入与可视化时保持符号一致。'],eq='plane',source=[PG_URL,PG_CODE],notes='采用平面方程n^Tx+d=0，x在相机坐标，z深度为Z，r=K^-1u，故Z=-d/(n^Tr)。代码可通过法线面向相机、abs距离和abs分母形成等价正深度表达，不能机械拼接两种符号。')
add(B,'pgsr-depth-effect','原论文对比了两种深度表示',[
'左侧保留原论文的几何说明，右侧给出同一深度监督下的重建点云示例。',
'设计重点是把深度与平面法线、距离关联起来，使几何监督作用于表示中的表面。'],images=[AS/'pgsr-depth.png'],caption='Chen 等，PGSR，Fig. 2 完整原图。效果来自论文实验。',source=[PG_URL,'https://arxiv.org/abs/2406.06521v2'])
add(B,'pgsr-aggregation','平面参数先混合，再转换为深度',[
'按同一可见性权重累积法线与平面偏移，再由当前像素射线计算深度。',
'采用共同权重时，公共归一化因子在深度比值中抵消。遮挡边界和低透明度区域单独检查有效性。',
'深度不再简单等于多个高斯中心 z 坐标的均值。'],eq='pgdepth',source=[PG_URL,PG_CODE],notes='本页使用统一带符号平面约定。实现的rendered_distance、rendered_normal和plane_depth见gaussian_renderer/__init__.py及CUDA。N不必单位长度，但法线比较前需要使用一致规范。')
add(B,'pgsr-flat','尺度与单视图法线约束建立局部表面',[
'最小尺度损失鼓励高斯变平，最短轴成为表面法线方向。',
'从相邻像素反投影点的叉积构建深度法线，再与渲染法线比较。图像边缘使用较低权重。',
'训练早期先拟合图像和粗表面，单视图一致性在第 7,000 步之后启用。'],eq=['normal','scale'],source=[PG_CODE],notes='官方代码min_scale_weight=100；single_view_weight=.015。图像权重为(1-get_img_grad_weight)^2并detach。法线差使用绝对值通道和。公式中L_N为未乘全局权重的项。')
add(B,'pgsr-geometry','多视图往返投影检验几何一致性',[
'从参考深度反投影到三维，投到邻居图，再取邻居深度反投影回参考图。闭环像素误差推动相邻表面一致。',
'默认有效阈值为 1 像素，使用 exp(−e) 权重并停止权重梯度。遮挡、无效深度和运动区域参与筛选。',
'邻居必须有真实重叠。当前工程在固定相机中选择邻居，避免同时估计位姿。'],eq='reproj',source=[PG_CODE,str(HOME/'current/pgsr-surface/adaptation.json')],notes='本地对源/目标静态mask和NCC patch做门控。几何目标是通过有效门控后的权重像素往返误差，不应称为独立GT深度损失。当前每视图5–8邻居。')
add(B,'pgsr-photometric','局部平面单应性对齐邻居图像块',[
'参考像素附近假设为局部平面，用该平面的法线与距离建立两相机之间的单应矩阵。',
'将参考图像块变换到邻居，在灰度域计算归一化互相关，减少局部亮度尺度的影响。',
'默认块半径 3，即 7×7 像素。每步最多采样 102,400 个有效中心，先做几何与遮罩筛选。'],eq=['homography','ncc'],source=[PG_CODE],notes='H_lk适用于X_l=R_lkX_k+t_lk与n_k^TX_k+d_k=0。源码NCC scale可能使用缩小图像域，半径按该域解释。真实邻居才计算NCC，当前不启用虚拟相机。')
add(B,'pgsr-objective','当前 PGSR 对照的完整训练目标',[
'RGB 仍为 0.8 L₁ + 0.2(1−SSIM)。最小尺度项从早期启用，法线、多视图几何、局部光度在 7,000 步后启用。',
'本轮 30,000 步，增密截止 15,000，最多 100 万高斯，不透明度裁剪 0.05，关闭额外绝对梯度分裂。',
'使用原完整分辨率 1514×2694、51 训练／8 开发视图。曝光补偿、DA3、Mip、diffusion 未同时加入。'],eq='pgsum',source=[PG_CODE,str(HOME/'current/pgsr-surface/adaptation.json')],notes='本轮受16GB显存预算约束。默认PGSR max_all_points=6M,opacity=.005,abs_split=50k；本地1M/.05/0是资源适配，应显式保留，不能称官方原样。式L_N,L_G,L_P为未乘全局系数项。')
add(B,'pgsr-evaluation','方法对照使用一致的输入和观察路径',[
'同一组 8 张真实开发照片，完整图和人物／头部局部一起比较。两条 60 帧路径分别检查距离和俯仰。',
'几何诊断使用共同支持的 SfM 点、遮挡条件和透明度覆盖。两种深度定义分开标注。',
'PGSR 与 Mini-Splatting2 的表示、光栅器和训练预算不同。本轮先判断路线价值，后续才拆分单模块消融。'],source=[str(HOME/'current/pgsr-surface/evaluation/protocol.json')],notes='协议冻结8真实全分辨率视图+120路径相机+9训练SfM锚点。3528观测、157头部；SfM点参与过原初始化，不作为独立三维GT。CPU协议相机校验差为0。')
add(B,'initialization','低纹理前景可由轮廓建立初始体积',[
'在已知相机下，把每张训练轮廓反投影成视锥区域，交集给出物体的外部粗形体。',
'轮廓约束补充稀疏纹理匹配，先建立连续前景体积，再优化颜色、表面与细节。',
'裁掉的图像区域视作未知，避免把被裁掉的身体误当作背景挖掉。'],eq='hull',source=['https://doi.org/10.1109/34.273735',GO_URL],notes='这是经典Visual Hull公式。本地实现为允许未知观测、轮廓阈值与轻微膨胀的保守离散体素交集；并非完整GaussianObject复现。')
add(B,'gaussianobject','GaussianObject 展示了轮廓与生成修复的衔接',[
'完整流程从轮廓初始化建立粗高斯，再进行视图修复与细化。它提供另一种初始化和监督组织思路。',
'本项目目前只落地了 CPU 轮廓初始化。生成修复需要通过独立视图一致性与真实观察验收后接入。'],images=[SRC/'gaussianobject-pipeline.png'],caption='GaussianObject (2024)，官方项目完整方法图。',source=[GO_URL,'https://github.com/chensjtu/GaussianObject'])
add(B,'public-elevation','公开样例提供真实的多高度相机',[
'选用作者公开 Diffuman4D 小样例的 0023_06、第 75 帧。48 台相机覆盖三层高度，图像为 1024×1024。',
'预先冻结：低层 16 台训练，中层 16 台开发，上层 16 台最终保留。训练和初始化只读低层图像与遮罩。',
'低层固定相机三角化得到 101 个点，轮廓初始化提供 30,000 个前景点，用于后续相同预算的初始化对照。'],source=['https://huggingface.co/datasets/krahets/diffuman4d_example',str(HOME/'data/public-validation/diffuman4d-example/elevation-benchmark')],notes='数据修订d776fa8311f49b2ee2223d3b3570fbe6024d33b7。CC-BY-NC-SA4.0，仅研究验证，非公司可商用授权证明。作者预处理WebP样例，不声称原始无损或全部DNA-Rendering。此处项目保留测试不保证预训练模型未见过。高度角约36.54°跨度相对估计相机up，不是绝对重力角。')
add(B,'hull-result','已知相机与轮廓初始化支撑原生训练',[
'固定已知标定，仅用 16 个水平视图的轮廓构建 30,000 点视觉壳。原生 Mini-Splatting2 保留深度重初始化与简化，完成 18,000 步。',
'嘴眼、轮廓与服装拟合随继续训练改善，最终 37,977 点。未加入 Mip、深度网络或扩散；图中结果仅说明训练相机的拟合。'],images=[HOME/'current/public-validation/current/review/final-head-38.png'],caption='Diffuman4D 公开样例 0023_06／000075，训练相机 38；从左到右：源照片、4,200 步、18,000 步。',source=['https://huggingface.co/datasets/krahets/diffuman4d_example',str(HOME/'current/public-validation/current/assessment.json'),str(HOME/'current/public-validation/current/final-validation.json'),str(HOME/'current/public-validation/current/review/crop-protocol.json')],notes='公开验证采用作者预处理样例0023_06第000075帧，48台已知标定相机分成水平训练16、中层开发16、高层封存16。训练固定相机与1024×1024去畸变黑背景合成图，只以训练轮廓构建30000点保守视觉壳；未使用作者全视图点云，也未用开发/封存轮廓调初始化。视觉壳是各训练轮廓反投影视锥的交集，仅作粗体积，不等于真实几何。原生msv2 fast与原生CUDA扩展，未使用本地Mip扩展、深度网络或扩散；保留原生深度重初始化和两次简化。损失为原生全图0.8 L1+0.2(1-SSIM)，黑背景；512分辨率预热后本轮第3009步首次真实1024训练，该时刻与工作视频的3010不同，源于各自保存相机栈。4200暂停后完整恢复参数、优化器、culling/factor、SH、相机栈及Python/NumPy/CPU/CUDA随机状态，13800次后续相机选择均与保存状态独立预测一致；原4200重渲染三相机逐像素相同。最终18000遵循原生最后一次loss/backward、最后实际optimizer更新17999的策略，未压缩调度。最终37977点、SH三阶；PLY59个字段与完整状态逐值相同，61项源码/输入哈希未变。16训练前景PSNR均值36.60399dB、有效全图42.40574dB；这是同一基线继续训练的拟合结果，不是新增模块的增益，不代表未见角度通过或新最佳。8000简化从54416到37977点，3个固定训练相机即时前景PSNR均值34.11降到28.63dB；后续原生训练恢复，不能由瞬时下降断言永久破坏。3相机节点均值与16相机终点均值不直接对比。图为原始像素裁片，按训练轮廓上部12%加固定padding选择，无结果导向裁剪、无锐化或二维增强，完整保留既有三列对照；原图嘴眼仍柔化，手指边界有差异。仅将训练端18k状态记为完整；中层开发与高层封存评估另列，不在此页用训练PSNR作泛化验收。')
add(B,'generative-scope','连续视频先验补充未覆盖观察',[
'给定相机路径，将当前三维模型的连续渲染与真实照片作为条件，生成沿路径一致的候选画面。',
'潜变量扩散从噪声状态逐步恢复图像或视频，条件信息决定相机、外观和结构的约束方式。',
'当前优先验证连续视角身份、姿态、轮廓和遮挡，随后才把可靠区域用于三维训练。'],eq='diffusion',source=[VX_URL,DF_URL],notes='式为通用扩散前向噪声模型；不同候选网络可能预测epsilon/v/x0，采样器各异，不能据此称相同实现。当前SVD输入已准备不等于网络已运行。')
add(B,'video-diffusion','ViewExtrapolator 用视频先验处理连续外推',[
'原图展示以已渲染轨迹为输入的外推流程。连续帧条件有助于把修复关联到同一段相机运动。',
'本项目已准备上／下俯仰各 25 帧输入，先验证帧间一致性和真实视图连接处，再决定是否蒸馏回三维。'],images=[SRC/'viewextrapolator-overview.jpg'],caption='ViewExtrapolator 官方项目完整方法图；本地候选流程尚待运行。',source=[VX_URL,'https://github.com/kunhao-liu/ViewExtrapolator'],notes='候选依赖StableVideoDiffusion，具体权重与采样配置需在实际执行时冻结。不要编造denoise步数、cfg、seed为已跑参数。原方法输出图像/视频效果，与原高斯参数是否改变分开。')
add(B,'difix-loop','Difix3D 提供修复视图回写三维的参考',[
'以当前重建渲染作为待修复输入，生成更可靠的监督，再继续优化场景表示。',
'左图把修复结果回写三维；右图是渲染后的二维增强。最终三维验收关闭右侧增强，真实观测保留为锚点。'],images=[SRC/'difix-pipeline.jpg'],caption='Difix3D (2025)，官方项目完整流程图。',source=[DF_URL,'https://github.com/nv-tlabs/Difix3D'],notes='当前本地曾做逐图修复与弱蒸馏小试但未通过。此页解释官方可借鉴机制，不能写作本项目完成官方复现。')
add(B,'feedback-loss','生成监督以置信度参与实际三维更新',[
'先冻结相机和拓扑，仅检查外观更新。真实照片继续约束已观测区域，生成结果使用较低且可撤销的权重。',
'外观阶段通过后，才独立比较小幅几何更新。每次只解冻一组变量，记录梯度与深度／轮廓变化。',
'启用前 λgen = 0。启用后的权重通过开发集与路径审阅确定，再冻结测试协议。'],eq=['feedback','genmask'],source=[DF_URL,str(HOME/'long-term-plan.md')],notes='这是本项目拟验证的训练接口，尚未实现完整通过。生成图像作为软监督而非GT。ρ可用鲁棒颜色误差加结构损失，最终形式随受控试验冻结。')
add(B,'generative-filter','可靠区域由真实锚点和跨视图共同筛选',[
'照片锚点：面部、衣服图案、颜色与轮廓保持一致。连续路径：检查部位跳变、闪烁与不合理遮挡。',
'跨视图：利用当前几何投影检查对应区域。冲突区域降权，同时保留冲突率，防止只保留容易区域。',
'三维回写后关闭所有二维增强，重新渲染未参与生成监督的路径，直接审阅模型。'],source=[DF_URL,VX_URL,str(HOME/'long-term-plan.md')],notes='可选CLIP/无参考图像分数只能辅助筛选，不能替代身份与几何检查；人像感知模型也不构成真实空间GT。拒绝把单帧锐化效果当作三维成功。')
add(B,'alternative','不同误差需要不同的替代思路',['表面表示：2DGS 采用定向二维高斯面元与射线交点，是与 PGSR 不同的几何路线。优先做独立方法对照。', '稀疏视角正则：DropGaussian 用训练时的随机丢弃缓解视角过拟合，CoR-GS 用多模型分歧施加一致性约束。先独立检验一种机制。', '人体与背景分离：HumanSplat／StudioRecon 提供结构先验思路。只有人体形体错误持续出现，才接入更复杂人体分支。'],source=['https://github.com/hbb1/2d-gaussian-splatting', 'https://github.com/jiaw-z/CoR-GS', 'https://github.com/humansplat/humansplat', 'https://github.com/sisyphm/StudioRecon', 'https://arxiv.org/html/2504.00773v1', 'https://github.com/DCVL-3D/DropGaussian_release/tree/c13da4aaddfed432fb556fdcf5d1d028b967ada8'],notes='备选项目不全部安装。StudioRecon默认渲染可能带DiFiX增强，验收须关掉。第三方权重、数据和公司使用许可逐项核对；这里只说明研究启动条件，不声称全部工程可直接商用。\n\nDropGaussian适配已单独完成匹配18k训练，方法与当前结论见35、47、48页。此处按原理区分随机丢弃与多模型一致性，不把另一方法作为本轮共同改动。')
add(B,'dynamic-extension','静态验收后再引入人体时间变化',[
'人体高斯可以放在规范姿态空间，通过骨骼变换与蒙皮权重映射到每个时刻，外观和非刚性残差另行建模。',
'先固定身体身份与静态背景，再验证运动姿态、手部和衣服变化，避免同时把相机误差与形变混在一起。',
'动态评测增加同步、多时刻遮挡和时间连续性，仍保留独立相机视角。'],eq='smpl',source=['https://smpl.is.tue.mpg.de/','https://github.com/humansplat/humansplat','https://github.com/zju3dv/Diffuman4D'],notes='式为LBS示意，x使用齐次坐标，T_j包含层级关节变换。Diffuman4D官方提供静态splatfacto入口，论文LongVolcap部分尚未开源，不称完整4D复现已就绪。SMPL/MANO模板在用户许可下本地保存，不公开分发。')
add(B,'quality','验收同时观察照片、主体与连续空间',[
'真实照片：全图、人物、头部分开报告 PSNR／SSIM／LPIPS，固定裁剪、分辨率、色彩和掩码。',
'主观观察：同相机同帧对照，不显示方法名，记录五官、发丝、衣纹、边缘与身份一致性。',
'连续空间：查看远近与俯仰视频、深度／法线、遮挡和漂浮层。覆盖率与错误同时报告。'],eq='coverage',source=[str(HOME/'current/pgsr-surface/evaluation/protocol.json')],notes='无真实对应照片的外推视角不计算虚假PSNR；允许主观清晰度、连续性与可见几何一致性证据。无参考锐度指标对伪细节敏感，仅辅助。')
add(B,'controlled-order','每轮实验只回答一个主要问题',[
'初始化与几何：比较固定输入下的表面路线或可靠深度，不同时引入生成视图。',
'尺度与外观：几何验收后分别改变滤波和采样率。控制训练步数、初始模型与评价相机。',
'生成与回写：先检查候选连续图像，再检查关闭生成器后的三维输出，最后进行重复与第二场景验证。'],source=[str(HOME/'long-term-plan.md')],notes='当前PGSR是方法筛选而非单模块因果消融。计划第二场景和多种子属于待执行，不用计划数量冒充完成数量。优先保持单一GPU大任务，不与评测争抢显存。')
add(B,'local-status','当前有效证据与待验收项',['公开样例：原生基线与 DropGaussian 适配均完成 18,000 步。两组从相同视觉壳初态出发，使用固定的 16 个水平训练相机。', '工作视频：公司服务器两组 4,200 步可见性刷新、真实视图评测及路径渲染完成。固定帧审阅未见稳定收益，本轮候选不晋级。', '中层 16 视图的预设数值筛选通过。两条路径已渲染并检查固定帧，整体人物质量仍未验收；重复与跨场景验证待完成，上层视图继续封存。'],source=['https://arxiv.org/html/2504.00773v1', 'https://github.com/DCVL-3D/DropGaussian_release/tree/c13da4aaddfed432fb556fdcf5d1d028b967ada8', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\preflight.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\company-server\\resolution-culling-review\\shared-prefix\\server-results\\evidence\\runs\\shuolee-culling-quality-001\\comparison.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\paths-root-review.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\diagnostic-protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\diagnostic\\result.json'],notes='本页只区分当前验证阶段。公开Diffuman4D人物帧由固定16水平视图训练，两臂从同一冻结30000点视觉壳出发，完整18000步分别结束，本轮native_control37580点、DropGaussian33089点。原生重初始化与简化规则保留，随机dropout只作用于候选训练RGB，完整机制见35页。训练16与中层开发16均使用关闭dropout/culling的原生完整模型渲染；upper16保持封存。开发五项筛选通过，仍是阶段候选，训练头部和前景指标退化且固定开发图有黑缺损，详见47页。\n\n公司视频的PGSR、5%ROI保留和4200步culling刷新对照均未晋级，固定帧检查不等于用户连续视频验收。它们与公开静态人物样例的数据、表示或预算不同，不能合并成受控排名。原alpha背景分支在一次位置梯度隔离配对后已关闭，保留最小机制与拒绝证据，不再追加权重扫描。\n\n当前两条预固定连续路径均已完成原生渲染、全部60帧解码及8个固定帧审阅。仍见胸前/近侧手部黑缺损、头顶条纹和手部模糊，没有连续人物质量验收；程序完整解码及固定帧模型审阅不等于用户连续观看验收。随后固定训练38与开发37的只读SH3/SH0/A及中心深度矩诊断已完成，模型参数和随机状态未变；合成观测及其边界见新增48页。后续还需同协议重复与第二场景。尚无新最佳模型或未见俯仰完整重建结论，旧18k极端协方差的原生梯度语义边界也不因当前前向评测通过而消失。实际证据以dropgaussian/protocol.json、preflight.json、两臂completion及evaluation/metrics.json为准，较早文件的pending字段保留历史阶段语义。')
add(B,'negative-evidence','更准确的先验尚未转化为明显外观收益',[
'同一组开发几何点上，多视图先验比旧单目先验更一致。但 510 步固定拓扑的配对训练未形成明显人物清晰度提升。',
'本轮头部 PSNR 差约 +0.0012 dB、SSIM 差约 +0.000029。它是短程开发试验，未晋级为最佳模型。'],images=[HOME/'current/da3-refinement/review/key-comparison.png'],caption='本地 DA3 小规模配对结果。开发视图与外推路径同条件审阅，无明显收益。',source=[str(HOME/'current/da3-refinement/review/summary.json')],notes='只说明本次固定拓扑、单seed、510步试验。不能据此否定可靠深度参与完整训练。原先验误差中位数23.18%，DA3为1.3129%；诊断SfM参与模型初始化，不能宣称独立三维精度。')
add(B,'solves','各模块承担的目标与已知作用',[
'Mip：改善不同采样尺度下的覆盖和能量一致性。深度与表面约束：增加前后关系、法线和跨视图几何约束。',
'轮廓初始化：补充低纹理前景的粗体积。人体先验：约束身份与关节形体。',
'连续扩散：探索缺失观察的合理外观。只有可靠监督写回模型并通过直接渲染验收，才记作本项目的重建收益。'],source=[MIP_URL,PG_URL,GO_URL,VX_URL,DF_URL])
add(B,'limitations','DropGaussian 的开发视角对照与当前不足',['两臂均训练 18,000 步。16 个中层开发视图的头部 MAE 降低 14.76%，14／16 视图改善。图中面部碎裂减少，上身噪纹更平滑。', '头顶条纹和胸前／手部黑缺损仍在，候选上胸有局部暗区。训练头部 MAE 增加 43.92%，16／16 视图回退，当前仅为阶段候选。'],images=['C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\source\\development\\37.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\native_control\\development\\37.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\dropgaussian\\development\\37.png'],caption='Diffuman4D 开发相机 37 的完整图像。左：源图，中：原生 Mini-Splatting2，右：DropGaussian 适配。',layout='figure',source=['https://arxiv.org/html/2504.00773v1', 'https://github.com/DCVL-3D/DropGaussian_release/tree/c13da4aaddfed432fb556fdcf5d1d028b967ada8', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\preflight.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\paths-root-review.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\diagnostic-protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\diagnostic\\result.json'],notes='本页三列是同一预固定development相机37的1024×1024完整原图。左为Diffuman4D作者分割合成源图，中为本轮重新训练的native_control，右为本轮DropGaussian适配。二者相同初始化、相同16水平训练相机、各18000步，不把旧37977点的历史基线冒充本轮37580点对照。候选33089点。渲染保留三阶SH和黑背景，关闭dropout与训练culling，使用全部高斯；没有裁剪脸部、锐化、超分、色彩增强或其他二维修补。\n\n开发数值使用中层16个已知标定视图，等权统计各视图区域指标。头部RGB MAE从0.0451071435降至0.0384472504，相对−14.764608%，14/16改善、2/16变差。头部SSIM+0.0587975，Sobel梯度RGB MAE−16.918518%。前景RGB MAE−4.649128%，前景PSNR+0.702520dB。这里+0.7025dB属于前景，头部PSNR差为+0.814602dB，不能混用区域。全图黑色面积较大，不能以全图分数代替人物区域。\n\n五项预冻结筛选均通过：头部平均MAE至少下降2%、至少9/16头部改善、头部梯度MAE不增加、头部SSIM下降不超过0.002、前景MAE增加不超过1%。这是允许进入复验和路径审阅的筛选门，不是最终重建验收。相关相机不是16个独立训练试验；只有一组匹配执行和一个人物帧，未做统计显著性推断。\n\nroot和本报告维护者分别查看固定37源图与两臂完整RGB：候选面部碎裂减少、上身噪纹较平滑，但头顶长条仍在，胸前和近侧手部大黑缺损仍在，候选上胸局部暗区明显。该图只展示当前实际变化，不能声称人体已清晰完整重建。训练相机的头部MAE却相对+43.921330%，16/16退化，头PSNR−3.138534dB，头SSIM−0.00831414；前景MAE+39.612312%，16/16退化。因此开发改善伴随训练拟合代价，不能只展示有利一侧，也不能把较平滑图像自动等同更准确几何。\n\n论文不透明度补偿不保证最终合成RGB无偏，因为透过率含不透明度乘积，且原生渲染有alpha截断与提前终止。当前后向规则、稀疏Adam和Mini拓扑调度与作者原始实现存在差异。已使用过的中层开发集是开发证据，不是未污染独立测试；高层16视图继续封存。本轮未读取该保留层，也未根据开发分数回调概率。两条连续路径已完成原生渲染、完整解码和预定固定帧审阅，胸手暗缺损与条纹仍在，未通过整体质量验收，也未经过用户连续播放验收。重复与跨场景验证尚未完成，没有新最佳模型。后续只读诊断显示开发37暗区在SH0下仍在且A较高，不能把RGB暗区直接解释为几何空洞或错误遮挡；详见48页。\n\n此前早期背景alpha及位置梯度隔离分支仍关闭。它们曾降低分割人物层外的投影不透明支持，但人物与背景联合要求没有通过，不再追加权重搜索或18k训练。DropGaussian本轮没有沿用其辅助A损失，不能把两种干预的指标差当成单变量直接比较。')
add(B,'maintenance','尺度轴比正则化约束高斯形状',['轴比描述最长与最短主轴标准差的比值。超过 10 的部分，每 10 步加入一次软惩罚。', '对参与本次 RGB 渲染的高斯施加正则，以总数 N 归一化。它直接作用于对数尺度，RGB 更新全参数。', '两组均完成 18,000 步。全体高斯中轴比超过 10 的比例，由对照的 70.38% 降至候选的 0.45%。'],source=['https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\mechanism-summary.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\training-evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\development-evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\root-review.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\continuous-paths\\result.json'],notes='本页对应同参数、同原生调度的最终18000步配对。两臂均由同一30000点视觉壳初态出发，使用16个水平训练相机、相同DropGaussian、RGB目标、深度重初始化与原生简化；候选额外添加尺度轴比软正则。4200状态完整恢复后续训到18000，最终均为三阶SH。第50页只使用这两臂最终18000状态的图片和指标；第47、48页是前一轮独立DropGaussian对照与其贡献诊断，不能混为本轮同一模型。\n\n公式中a属于{1,2,3}，s_ia=exp(ell_ia)为第a主轴的正标准差，q_i=max(s_i)/min(s_i)为无量纲三维轴比；rho_i^{2D}为同次RGB渲染的二维投影支持半径，单位像素。v_i=1[rho_i^{2D}>0]不反传；随机丢弃和持久culling后未参与本次RGB渲染的高斯不进入分子。每10步添加0.1/N乘以参与渲染的max(q_i,10)-10之和，N为当前全体高斯总数。对照没有此项。前页暗色0.1与投影20px的分类阈值不参与正则，也不是剪枝条件。\n\n参考nerfstudio Splatfacto的PhysGaussian尺度正则，源git blob 475cc5c388cf5a05f46462636d423e684ecd5453，文件SHA 8dfddd34dfcf2860a9d12fe7c057d72d0060edc1db430354528a93faec1b65da。Mini的SparseAdam只更新可见行，因此本适配把分子限制为同次RGB可见行而保留总N归一化。在共同状态与相同可见行上，各相应行梯度相同；标量不等同官方全点损失，不称完整原样官方复现。amax/amin并列极值均分梯度，maximum在精确阈值处取半梯度，沿用原语义。\n\n当q_i>10且极值轴唯一时，正则对最长轴的对数尺度梯度为正、最短轴为负；梯度下降能够缩短最长轴或放大最短轴。它直接作用于尺度，不直接给位置、旋转、opacity、SH和屏幕均值梯度。RGB仍优化全部参数，后续可见性和拓扑可随训练共同变化；不能称除尺度外所有参数保持冻结。也不能据轴比降低证明中心位置、深度或表面正确；均匀放大各向同性高斯不受此项惩罚，细发与薄纱仍需保护。\n\n最终全体存储高斯每点一票：对照33491点，轴比大于10的23570点，占70.37711624%；候选34990点，大于10的158点，占0.45155759%。轴比P99分别548.98387451与9.95412881，最大值18560.78125与10.31310558；全部最终尺度为有限正数。正文70.38%到0.45%为两个点群的无权分布比较，不是逐点配对、投影覆盖或不透明度加权比例。点数与拓扑不同，较少极端各向异性仅是形状机制证据；候选最大值仍超过10，符合软约束而非硬裁剪。\n\n执行细节留于讲稿：首4200阶段420次调度，327次正损失，302次正则梯度抵达实际优化步骤，25次同一步参数替换使梯度未抵达。续训4201至18000共1380次调度，1378次抵达优化步骤；8000一次结构替换、18000原生终步无optimizer更新。总计1680次抵达，不等于1680次画质收益。最终实际optimizer更新至17999，18000仍执行原生loss/backward。8000原生交集简化分别54595→33491与55102→34990，13000仅按原生规则处理culling与factor，不增减高斯。不能把26次结构替换加1次终步跳过统称为27次结构丢弃。\n\nmechanism-summary.json为保存状态和日志的CPU只读审计，没有额外CUDA前向。其证明当前记录的一致性，不声称与一次不间断长训逐位等价。最终外观改善与训练代价分别见第50、51页；上层保留图未读，没有新最佳模型。',eq=['axisratio', 'axisregularization'],layout='formula')


# Renderer audit is a topic in the existing optimization deck, not a new report.
add(B,'remy-renderer-evidence','分享页的清晰度先拆解为可验证因素',[
'Remy3D 的 SOG 元数据：727,737 个高斯，三阶球谐；输出相机文件含 200 个相机，输入记录含 252 帧、1440×1920 图像。',
'实测显示：2560×1440 画布对应 1280×720 窗口；自适应画质开启，当前 TAAU 关闭、SH 上限为 3、渲染比例为 1。',
'已检查的查看器加载几何、尺度、旋转与 SH 压缩数据，未见独立人脸增强分支。离线是否做过超分，网页资料不能确定。'],
source=['https://www.remy3d.cn/_nuxt/9qrz95w_.js','https://www.remy3d.cn/_nuxt/D7EXdWoG.js','https://www.remy3d.cn/_nuxt/BksRqccV.js',str(HOME/'current/remy3d-analysis/sog-meta.json')],notes='2026-10-19 浏览器开发日志、可见 stats 和 DOM 画布尺寸核验。SOG 通过约 80 KiB 范围请求读取目录及元数据，没有额外下载整包。当前 TAAU 关闭只描述本次观察；其他设备可能自动降级。高斯数/相机数与本地数据不同，不能据此作因果结论。输入 pcd 路径 HEAD 返回404。')
B.insert(30,B.pop())
add(B,'face-detail-route','人物局部容量与时序一致性的验证顺序',[
'局部分配：先比较重初始化与简化的即时前后，再比较相同步数的重新拟合结果，区分短暂覆盖损失与持续重影。',
'时序变化：固定场景相机，先估计头部整体刚性运动，再检查嘴、眼等区域剩余的非刚性误差。',
'独立验证：固定总点数检验局部保留，再单独更新可见性剔除掩码。每轮保持其余训练变量一致，用真实照片与连续路径共同检查。'],
source=[str(HOME/'current/company-problem-audit/analysis.json'),str(HOME/'current/company-problem-audit/node-audit/assessment.json'),str(HOME/'current/company-problem-audit/retention-pilot/assessment.json')],notes='本地fast配置预算18k步，增密只持续到3000步，2000步重初始化，3000/8000步简化。节点诊断已完成至8200，5%局部保留受控对照已完成至4200，候选未晋级。配置warm-up结束为3000，但2999恢复状态的相机栈仍有10个半分辨率条目，3000至3009实际继续使用半分辨率，3010首次真实全分辨率训练。3001节点图只是1514×2694诊断探针，不代表训练相机已切换。源码已有大投影面积分裂及激进克隆；头部RGB加权和后期增密已经试过。后续culling对照固定保留集合、全局点数、factor、损失与训练顺序，只刷新逐点逐相机掩码，不能直接调用同时prune点的原函数。头部刚性/非刚性运动原因尚未分离；超分另作后续外观候选，不能代替几何与时序检查。')
B.insert(31,B.pop())

# Desktop questions refine the existing optimization aspect. Full original method figures.
P=HOME/'current/company-problem-audit'
def before(key): B.insert(next(i for i,s in enumerate(B) if s['key']==('quality' if key=='validation' else key)),B.pop())
add(B,'head-capacity','DropGaussian 以随机丢弃约束稀疏视角训练',['训练时临时移除部分遮挡，增加后方高斯参与优化的机会。保留点按概率补偿不透明度。结构调整沿用原生流程；推理关闭随机丢弃，渲染全部高斯。'],eq='dropgaussian',images=['C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\presentation-build\\sources\\dropgaussian\\dropgaussian-fig2-full.png'],caption='Park、Ryu 与 Kim，DropGaussian（2025），Fig. 2 完整原图，arXiv:2504.00773v1。',layout='formula-figure',source=['https://arxiv.org/html/2504.00773v1', 'https://github.com/DCVL-3D/DropGaussian_release/tree/c13da4aaddfed432fb556fdcf5d1d028b967ada8', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\preflight.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\metrics.json'],notes='原论文为Hyunwoo Park、Gun Ryu、Wonjun Kim，DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting，arXiv:2504.00773v1（2025）。图为arXiv HTML公开的完整Figure 2，1890×900，未裁剪子图、未重绘或替换标注。图的左、中、右分别保留初始化、训练时渐进随机丢弃、推理时完整场景。图中SfM是论文流程；本地人物样例使用已知相机和训练轮廓视觉壳初始化，不能把论文初始化图当作本地SfM结果。\n\n每个高斯每步独立采样保留变量m_i,t，服从Bernoulli(1−p_t)。保留点临时以o_i/(1−p_t)参与当前RGB渲染，被丢弃点当前贡献为零。公式是论文Eq.(4)中M(i)=1/(1−r)或0的等价写法，p_t是丢弃概率，o_i是sigmoid后的基准不透明度。E[m_i,t o_i/(1−p_t)]=o_i描述单个不透明度的期望补偿。随机移除前方贡献改变透过率，使原本被遮挡的点有机会参与颜色拟合。此处“移除”是单步采样，不是永久剪枝。\n\n本地固定p_t=0.2t/18000，t为实际训练步数，末步概率0.2，未搜索权重或概率。0.2为本轮协议选择，不是普适最优值。RGB目标保持0.8L1+0.2(1−fused SSIM)，训练RGB未经截断；相机、视觉壳初态、原生参数、持久可见性掩码和结构调度相同。两臂都采用18k完整预算，候选只引入上述训练随机丢弃，不叠加alpha背景损失、Mip、外部深度网络或扩散。\n\n实现使用独立CUDA随机数发生器seed87219，避免额外采样消耗相机和原生拓扑的随机流。当前临时可见性掩码为持久culling与drop集合的并集，不写回模型。只有训练RGB前向/反向启用dropout，2000步深度重初始化、3000/8000步简化、最终评测均关闭dropout；结构调整沿用Mini-Splatting2原生流程及原生可见性规则，不叠加随机丢弃。最终评测也关闭训练culling，不以训练屏蔽掩盖外部支持。\n\n这是DropGaussian核心机制在Mini-Splatting2中的适配。原生稀疏Adam按radius>0更新，丢弃点半径为0，因此该步参数和Adam矩均冻结。增密统计也排除被丢弃点，不向计数器填零或重新映射。Mini增密在3000步结束，此时p约0.03333，随后还有8000简化；因此不能称为官方dense-Adam与10k增密调度的逐项完整复现。保持原生位置学习率30000步时域、3009首次全分辨率和18k末步不再optimizer.step的规则。初态零丢弃等价及被丢弃点零导数检查通过，未改变模型、优化器或随机状态；这不是任意状态的有限差分导数证明。\n\n实际两臂18k均完成，终点点数37580与33089，差异允许来自训练变化引起的结构演化。固定相机流相同不代表拓扑随机抽样逐位相同。当前效果与局限见47、48页，所有推理都关闭dropout和二维增强。')
before('validation')
add(B,'absgs','AbsGS 的像素梯度累积与细节分裂',[
'图中相反方向的像素梯度会抵消。AbsGS 先逐像素、逐坐标分量取绝对值，再累积，增强大高斯的分裂信号。',
'接入时保留真实像素梯度统计和原训练控制，单独比较分裂准则。已求和梯度的绝对值无法替代这一机制。'],images=[P/'sources/absgs-method.png'],caption='Ye 等，AbsGS（2024），官方完整方法图。',source=['https://ty424.github.io/AbsGS.github.io/','https://github.com/TY424/AbsGS'],notes='图(a)投影及覆盖像素，(b)普通梯度相消，(c)逐像素分量绝对值后求和。官方模型用常规梯度克隆、绝对梯度分裂，需要自定义rasterizer输出。本地MS2已含大面积mask，故只修改分裂信号并保留其他流程，才可判断新增收益。官方源码还存在额外pruning，不应无说明一起搬入。')
before('validation')
add(B,'pixelgs','Pixel-GS 的像素覆盖加权',[
'用每个视图实际参与计算的像素数加权梯度，让充分覆盖高斯的观察对增密判断贡献更多。',
'图右按相机深度缩放梯度，以抑制近相机漂浮物。首轮独立于 AbsGS 验证，避免同时改变两种统计。'],images=[P/'sources/pixelgs-method.png'],caption='Zhang 等，Pixel-GS（ECCV 2024），官方完整方法图。',source=['https://pixelgs.github.io/','https://github.com/zhengzhang01/Pixel-GS'],notes='p_i表示该视图参与计算的覆盖像素数，不是椭圆包围盒面积。官方add_densification_stats中累积norm(gradient)*pixels，分母累积pixels。MiniGS原area_max不是这一统计；接入需核对光栅器可见性及culling后坐标梯度归一化。点数预算、原深度/Mip和训练相机不同时变更。')
before('validation')
add(B,'head-canonical','头部规范空间与每帧刚体变换',[
'共享一个规范姿态的头部高斯。每帧用头部旋转与平移变换中心和协方差，背景仍保持静态。',
'场景相机与头部运动分开估计。先拟合刚性变化，再用独立跟踪点检查嘴、眼、脸颊的剩余误差。',
'静态展示选择一个参考姿态与表情。存在表情残差时，再评估表面绑定或时间条件形变。'],eq='headrigid',source=['https://github.com/ShenhanQian/GaussianAvatars','https://github.com/ingra14m/Deformable-3D-Gaussians',str(P/'next-experiment.json')],notes='此页是拟验证的本项目分层方案，不能称为已集成GaussianAvatars。先固定场景相机消除相机/头部坐标混淆，使用时间平滑、参考姿态和留出跟踪点约束自由度。静止表情子集必须匹配角度覆盖与样本预算，避免将视角减少误归因于去运动。')
before('validation')
add(B,'gaussianavatars','GaussianAvatars 的表面绑定与表情驱动',[
'头部跟踪得到形状、姿态和表情，三角面局部坐标承载高斯。面片随表情变化时，高斯同步变换。',
'分裂后的高斯继承面片绑定。VHAP 提供单目跟踪入口，场景背景仍可沿用独立的静态表示。'],images=[P/'sources/gaussianavatars-method.jpg'],caption='Qian 等，GaussianAvatars（CVPR 2024），官方完整方法图。',source=['https://github.com/ShenhanQian/GaussianAvatars','https://github.com/ShenhanQian/VHAP'],notes='完整方法图保留输入、FLAME参数、局部坐标、可微渲染和损失。官方代码需要FLAME2023含jaw rotation及FLAME_masks，SMPL/MANO不可替代。本项目的背景/头部组合需要坐标与遮挡接口实现，尚未完成。跟踪误差、头发和眼镜边界及许可条件在最后不足页说明。')
before('validation')
add(B,'problem-evidence','局部保留改善暂态，持续重影仍在',[
'两组均为 459,873 点。5% ROI 保留交换后，三个训练头部框的即时 PSNR 提高 0.81–1.76 dB；4,200 步后优势基本消失。',
'8 个开发视图的头部平均 PSNR 差为 −0.0043 dB，固定路径未见稳定清晰度改善。该规则不晋级；运动原因仍待独立分离。'],images=[P/'retention-pilot/baseline/evaluation/000024-head.png',P/'retention-pilot/roi_swap05/evaluation/000024-head.png'],caption='本地 4,200 步直接渲染，同一开发相机 000024：左为基线，右为 5% ROI 保留交换。',source=[str(P/'retention-pilot/assessment.json'),str(P/'retention-pilot/comparison.json'),str(P/'node-audit/assessment.json'),str(P/'motion-diagnosis/assessment.json')],notes='本页代表图为已审阅全部8个开发视图中的同一正脸相机；未做锐化或二维增强，也未进一步裁剪保存的头肩图。受控干预固定原版有效池和459873全局保留点，将原版保留点中最低ROI可见贡献者与遗漏点中最高贡献者配对；仅严格增加贡献时交换，最多floor(0.05K)，实际交换22993点。51个训练头部框包含头发、颈部及部分背景，排除已有训练运动mask，不是精确人脸分割。两组继承相同culling和factor，只按保留索引切片；未刷新culling、未改变Mip、深度、损失或增密。共用冻结原版keep/culling/factor/ROI分数与采样后的Python、NumPy、CPU/CUDA随机状态，属于受控重放，不能称独立原生执行逐位复现。相机抽样轨迹、最终随机状态及剩余栈一致，六项观察器检查通过，双进程均核对实际原生ms扩展。视图1/23/47即时头部PSNR差为+0.8059/+1.5833/+1.7639dB；4200时为+0.0071/+0.1846/−0.0244dB。99%贡献所需点数由10081/10613/11193增加到14385/14603/16473，但外观收益没有相应保持，不能将更多局部点直接等同更好重建。8个dev头部PSNR均差−0.0042853dB，归一化梯度误差差−0.0042810；主体PSNR+0.0022027dB、背景+0.0150428dB、全图未掩码+0.0005973dB。指标轻微升降不构成清晰度验收；全部8个头肩/主体/背景对照及两条60帧路径的0/15/30/45关键帧已检查，仍有相似的纹理混合与未覆盖区域。完整连续视频人工验收尚未完成。5%候选不晋级，不继续18k，不利用反复使用的dev微调保留比例；本结果只否定本次短程规则晋级，不能否定所有局部保留方式。简化即时覆盖损伤与持久跨视角重影是不同的待解释现象。后续已完成3010首次真实全分辨率时的culling刷新对照：固定保留集合/点数，不与ROI交换叠加，两臂均到4200。头部PSNR均差−0.316461dB，固定路径仍未见稳定清晰度收益，本轮刷新策略不晋级，见42与48页。原始微动诊断：t−3/t+3三角化预测锚帧t，头部59点残差中位0.442px；框外992点0.125px但880点集中帧48，纹理、深度和时间分布不同，不作因果比值。嘴眼覆盖不足，t±6仅帧48有13点。所有帧参与SfM，非完全独立相机估计；运动、反射、跟踪误差、位姿偏差与滚动快门尚未分离。需独立稳定特征验证头部SE(3)，再判断非刚性残差。')
before('negative-evidence')
add(B,'pgsr-result','PGSR 的几何对照与人物效果',[
'已完成 30,000 步。相同 8 张开发照片中，头部 PSNR 差 −0.194 dB，LPIPS 差 +0.00677。关键新视角仍有重影。',
'部分深度统计改善，尚未转化为人物清晰度收益。本候选不晋级，继续检查局部分配和头部时序一致性。'],images=[HOME/'current/pgsr-surface/evaluation/review/novel-view-comparison.png'],caption='本地同相机比较：Mini-Splatting2 参考与 PGSR，固定俯仰及距离样本。',source=[str(HOME/'current/pgsr-surface/evaluation/review/preliminary-decision.json')],notes='PGSR567187点30k，参考316044点18k，表示/渲染器/预算不同，是方法筛选非数量消融。SfM共同头部148观测中位误差.01997降到.00856，但均值.25381升到.26924，框外覆盖率下降；SfM参与初始化非独立GT。初审只看固定关键图和真实头部拼图，完整视频主观验收未完成。缺失开发运动mask修复有来源图独立标注附录，两模型冻结渲染同处理并同时报告全图不遮罩指标。')
before('solves')

next(s for s in B if s['key']=='gaussianavatars')['layout']='figure-page'
# Same frozen representative elevation sample, enlarged as an A/B pair.
pgslide=next(s for s in B if s['key']=='pgsr-result')
pgslide['images']=[str(HOME/'current/pgsr-surface/evaluation'/arm/'elevation-045.png') for arm in ['baseline','pgsr']]
pgslide['caption']='同一仰视相机：左为 Mini-Splatting2 参考，右为 PGSR。均为直接渲染。'

add(B,'opacity-color-diagnostic','胸手暗区的高斯合成贡献',['大暗组：支持半径超过 20 px，且零阶 SH 三通道均值低于 0.1；小暗组半径不超过 20 px。阈值只描述分类，不是剪枝条件。', '候选胸部 331 个事后暗像素：大／小暗组占 47.08%／44.82%；近手 2,303 个：72.63%／23.38%。仅为该候选内支持份额。'],images=['C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\source\\development\\37.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\evaluation\\dropgaussian\\development\\37.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\contribution-attribution\\results\\dropgaussian-37-dark_large.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\contribution-attribution\\results\\dropgaussian-37-dark_small.png'],eq=['groupcontribution'],caption='本地原生测量：开发相机 37，18k 候选，完整原图；右两幅统一使用 0–1 线性灰度。',layout='diagnostic-four',source=['C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\contribution-attribution\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\contribution-attribution\\results\\result.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\contribution-attribution\\independent-review.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\diagnostic\\result.json'],notes='四幅均为固定开发相机37的1024×1024完整图，按原比例放置，没有裁剪、增强、锐化或超分。左为Diffuman4D作者分割人物黑背景合成源图；第二幅为18k DropGaussian适配候选的原生三阶SH渲染；右两幅分别是该候选大暗组与小暗组经过所有高斯遮挡后的真实累积合成权重。两张支持图使用相同线性灰度0至1，不按单图重新拉伸对比度。它们不是独立渲染所选组的图，也不是去掉该组后的反事实图。\n\n公式中的G_k(u)=sum_{i in class k} alpha_i(u)T_i(u)，alpha_i是该原生普通光栅器实际使用的截断后像素不透明度，T_i是所有位于前方且被实际遍历的高斯累积透过率。先保持完整高斯集合、相同位置/协方差/opacity/排序及相机，再只把输出颜色替换为互斥组标签，得到G_k。四组穷尽所有点，因此sum_k G_k=A。支持图测量的是当前模型的可见合成贡献；点数比例不能代替它。实际删除高斯会改变后续T_i，不能通过从RGB或A中减去本图预测删除结果。\n\n分类使用原生SH0的每通道下截零颜色，取三通道算术均值。均值低于0.10为暗组，不是物理亮度或真实黑色材质标签；该颜色不做上截一再分类。较大定义为当前相机的原生投影支持半径严格大于20px，小组为不超过20px；该半径依赖opacity、投影协方差和原生+0.3二维滤波，是支持范围界而非物体真实尺寸。图中只展示大暗/小暗两组，另外两组为其他大/其他小。0.10和20px是在测量前冻结的描述阈值，不是已验证剪枝参数，也不能据它们判定某点正确或错误。\n\n当前候选相机37，胸部事后框[610,365,705,470]内沿用既有源图亮而渲染暗子集331像素；近手框[590,465,705,545]的同类子集2303像素。区域份额严格按sum_pixels G_k / sum_pixels A计算，不是逐像素比例的均值，更不是高斯数量占比。胸部大暗组47.08169249%、小暗组44.82299029%；近手分别72.62510188%、23.38116784%，其余份额由另外两组贡献。胸部/近手平均A分别0.99981672与0.99952182，与此前A诊断一致。胸部小暗组支持很大，单靠大尺寸条件不能覆盖全部已观察暗区。\n\n上述像素集合为先前事后探索沿用，不是新冻结的独立验证集。两臂各自使用自己的暗像素子集，因此不同臂的子集份额不能当同像素配对改善；只有固定框全部像素共享空间位置。此页仅给同一候选内的比例，不声称GT精度、训练收益、几何正确、错误遮挡或唯一训练成因。头部矩形包含背景，不用其组份额来宣称正常黑发事实或点的合法性。\n\n本轮12次只读前向、0次反向、0次训练更新；源SH3复现、四组和与A、SH0强度、参数和RNG保持等契约通过。独立CPU审查重新计算保存数组的组标签、固定框与事后子集统计以及24张PNG像素。最大组和误差6.281770765781403e−7，旧A复用误差0，SH0强度误差8.344650268554688e−7；没有读取原始DATA或封存upper，没有新模型晋级。原始result的pending_interpretation是执行完成时历史状态，随后independent-review.json已记录复核通过。root已逐张查看37源图/SH3、大暗/小暗组和固定38大暗组。\n\n既有深度诊断仍只表示合成中心深度矩，不能当真实表面或厚度；本页不再使用深度伪彩色。证据result SHA256 a64f2caaceae08ce72de0545de4e759f117747c86b2c381c373b147bde8c4a1d。')
before('maintenance')


add(B,'scale-effect','18,000 步的开发视角效果',['16 个开发视图：头部 MAE 降低 6.03%，头部 SSIM 增加 0.0108，前景 MAE 降低 3.20%。', '完整训练至 18,000 步后，第 37 视图额头与胸前长条杂斑减轻。三幅均为同相机完整原图。'],images=['C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\development-evaluation\\source\\37.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\development-evaluation\\drop_control\\37.png', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\development-evaluation\\drop_scale\\37.png'],caption='图源：Diffuman4D 开发相机 37。左为源图，中为 DropGaussian 对照，右为尺度正则候选；两组均为 18,000 步。',layout='figure',source=['https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\mechanism-summary.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\training-evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\development-evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\root-review.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\continuous-paths\\result.json'],notes='三幅为Diffuman4D 0023_06第000075帧、中层开发相机37的完整1024×1024图。左为作者分割人物黑背景源图，中为同DropGaussian的drop_control，右为加入标准尺度轴比正则的drop_scale。两臂均完成18000步且SH3，按全部高斯渲染，推理关闭随机丢弃与训练culling。图片不裁剪、不增强、不锐化、不超分；采用continuation18k/development-evaluation内的最终图，与本页18000指标绑定，未混入4200图。\n\n本页数字为16个已使用中层开发视图逐视图区域指标的等权均值，不是37单图分数、独立训练重复或最终保留集。头部RGB MAE 0.03926345950→0.03689592029，相对下降6.02987929%，11/16较好、5/16较差；SSIM 0.79943395779→0.81020594016，绝对增加0.01077198237；头部梯度RGB MAE下降3.94440798%。前景RGB MAE 0.04453420360→0.04311093595，相对下降3.19589783%，14/16较好；前景梯度MAE下降0.88502881%。全图黑背景会稀释人物错误，因此不用全图分数替代人物区域。\n\nroot实际审阅37完整三图、16头部原像素拼图以及两条路径各8固定帧，胸前及额头长条明显减轻。报告维护者也逐张检查了37完整源图和两臂最终图。只读开发评测32次原生前向、0更新，参数和随机状态未变；上层未读。连续路径程序完成两条各60帧并全部解码，不等同用户连续播放验收。人脸和手的剩余问题及训练代价集中下一页，不把区域平均误差下降称为人物清晰重建完成。\n\n最终development metrics SHA256 cd4d8637f77fae4d027cd8346af6a5f8d366fe1a51bb8371ff200854cd5336a7；root-review SHA256 e819b53867ee912e1eb1c1cc1bc5bf8d62ba02968537e26b37156be4faf197bf。metrics里pending_root_review为生成时状态，后续root-review.json已记录固定图和路径采样审阅。源37 SHA71168fd20348e40b451f3092f0bb0d7ce2ccf4dda2ee30e8a1c3dfc25a4f1fcc；对照37 SHAa045a5e2ada81e90b48f2063f79b213ce772db3ddc2617420d5ffcac6988c9bf；候选37 SHA4088d0278f7077a134cfa712f15f7f09528de230c977e3911bc4e8374c9b60b5。')
add(B,'scale-conclusion','当前不足与下一步验证',['近手大暗块、柔化和失真的面部仍在。训练视图头部与前景 MAE 分别增加 3.28% 和 2.85%，尚无整体清晰度突破。', '当前仅有一个人物帧、单种子的 18,000 步配对。连续播放验收、独立重复和第二场景待完成；上层仍封存，未晋级最佳。', '下一步先做固定相机的多视图立体支持诊断，区分可靠几何与未知区域，再决定深度约束；继续保护手、头发和薄纱。'],source=['https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\protocol.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\mechanism-summary.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\training-evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\development-evaluation\\metrics.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\root-review.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\scale-regularization\\continuation18k\\continuous-paths\\result.json', 'C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\public-validation\\current\\dropgaussian\\fixed-pose-mvs\\cpu-validation.json'],notes='本轮完成单人物帧、单随机种子、相同原生18000调度的匹配配对，并完成训练和开发评测、固定图与路径采样审阅。相同参数续训已实际完成，不再写为下一步。中层16视图用于开发；高层保留图未读取。没有独立重复、第二场景或用户连续视频验收，不晋级最佳模型；当前仍未满足人物清晰重建要求。\n\nroot图审发现候选额头和胸前长条明显减轻，但37近侧手边仍有大黑块，脸部细节软且有失真，衣料和手指尚不能稳定分辨。两条路径各8固定帧中，远近变化没有清晰度突破；最近端固定路径裁到了头，不能作为细节改善证据。视频均完成60帧程序解码；本次是完整解码加离散固定帧审阅，不是人类连续播放验收。\n\n反向证据：最终训练视图头部RGB MAE 0.00774061430→0.00799442924，相对增加3.27900257%，13/16恶化；前景RGB MAE 0.00779929131→0.00802134376，相对增加2.84708493%，15/16恶化。正文所谓训练代价是拟合误差，不是耗时或显存。训练头部与前景梯度MAE分别增加约1.33%与2.15%。开发平均改善和训练拟合代价需同时保留，不能说所有视角、所有区域都提升。轴比尾部收缩也不证明正确位置、遮挡或真实三维表面。\n\n下一步先做固定相机多视图立体的独立支持诊断。CPU仅导出冻结6个训练RGB、相机、mask和3组隔离workspace及小型公司Linux包，18PNG与输入清单已哈希核对；未读取开发或高层图用于MVS，尚未得到MVS深度或几何改善。现有公司COLMAP3.9构建不带CUDA，存在RTX3090不能据此推断dense stereo可运行；只读server-preflight及隔离CUDA构建决策由主任务另行处理。不在报告中宣称MVS执行完成，不自动将未知或无支持区域写成GT深度。\n\n后续先区分真实多视图一致支持、遮挡和未知区域，再判断是否值得引入带置信度的几何监督。手指、头发和薄纱继续作为保护对象；颜色与点大小不能单独作为删点依据。已经关闭的alpha优化分支不重启权重扫描。当前两臂保留供机制分析，不代表新最佳或工作视频已修复。')

closing = next(s for s in B if s['key'] == 'scale-conclusion')
closing['body'] = [p.replace('近手大暗块', '手部附近的大面积黑色伪影').replace('上层仍封存', '高俯仰留出视角尚未评测').replace('多视图立体支持诊断', '多视图立体匹配与几何一致性检查') for p in closing['body']]

out={'created':'2026-10-19','owner':'shuolee','source_version':'living-1','decks':decks,'equations':EQ}
(BUILD/'content.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'decks':[(d['filename'],len(d['slides'])) for d in decks],'formula_count':len(EQ)},ensure_ascii=False))

"""Deterministic, original theory illustrations; not paper figures or measured results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/3dgs-theory-figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 21,
                     'axes.unicode_minus': False, 'mathtext.fontset': 'stix'})
BG, INK, GREEN, ORANGE, BLUE = '#fbfaf5', '#263d34', '#548f6d', '#c97843', '#498caa'
saved = []

def canvas(n=2, height=5.3):
    fig, axes = plt.subplots(1, n, figsize=(12, height), dpi=150, facecolor=BG)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.3, 1.5)
        ax.set_aspect('equal'); ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, top=.90, bottom=.08, wspace=.15)
    return fig, axes

def label(ax, x, y, text, color=INK, size=21, ha='center'):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va='center')

def arrow(ax, a, b, color=ORANGE, style='-', width=2):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle':'->','color':color,'lw':width,'linestyle':style})

def camera(ax, p, color=ORANGE, scale=.12, orient=0):
    pts=np.array([[-1,-.8],[1,0],[-1,.8]])*scale
    rot=np.array([[np.cos(orient),-np.sin(orient)],[np.sin(orient),np.cos(orient)]])
    ax.add_patch(Polygon(pts@rot.T+np.array(p), closed=True, color=color, zorder=5))

def cloud(ax, shift=(0,0), color=GREEN):
    pts=np.array([[-.45,.15],[-.2,.4],[.15,.25],[.4,.5],[.58,.1],[.1,-.15]])+shift
    for i,p in enumerate(pts):
        ax.add_patch(Ellipse(p,.34,.15,angle=i*23,facecolor=color,edgecolor=color,alpha=.35))
        ax.plot(*p,'o',color=color,ms=4)
    return pts

def matrix(ax, entries, center=(0, -1.45)):
    """Draw a conventional three-row matrix; mathtext has no bmatrix support."""
    cx, cy = center
    for row, values in enumerate(entries):
        for col, value in enumerate(values):
            label(ax, cx+(col-1)*.49, cy+(1-row)*.27, value, size=21)
    left, right, bottom, top = cx-.78, cx+.78, cy-.43, cy+.43
    for side, inward in [(left,.10),(right,-.10)]:
        ax.plot([side+inward,side,side,side+inward],[top,top,bottom,bottom],color=INK,lw=1.8)

def save(fig, name):
    fig.savefig(OUT/name, facecolor=BG, dpi=150)
    plt.close(fig); saved.append(name)

fig, axes=canvas()
for ax, title in zip(axes,['离线重建：相机位姿给定','在线 SLAM：相机位姿待估']):
    ax.set_title(title,color=INK,fontsize=24,pad=18); cloud(ax,shift=(.2,.15))
cams=[(-1.0,-.8),(-.2,-1.0),(.75,-.8)]
for p in cams:
    camera(axes[0],p); arrow(axes[0],p,(.35,.4),color='#b8c6b5',style='--',width=1.3)
label(axes[0],0,1.12,r'$\Theta$ 待估，$T_1,T_2,T_3$ 给定',size=20)
path=np.array([[-1.2,-.9],[-.7,-.7],[-.35,-.8],[.1,-.6],[.55,-.75]])
axes[1].plot(path[:,0],path[:,1],color=ORANGE,lw=2,ls='--')
for p in [path[0],path[2],path[4]]: camera(axes[1],p)
arrow(axes[1],path[1],path[3])
label(axes[1],0,1.12,r'$\Theta_k$ 与 $T_k$ 都待估',size=20)
save(fig,'29-01-known-vs-unknown-cameras.png')

fig, axes=canvas(height=6.5)
for ax in axes: ax.set_ylim(-2.1,1.5)
axes[0].set_title('各向异性高斯',fontsize=23,color=INK,pad=18)
axes[1].set_title('各向同性高斯',fontsize=23,color=INK,pad=18)
for ax in axes: ax.plot(0,.15,'o',color=INK,ms=7)
axes[0].add_patch(Ellipse((0,.15),2.0,.65,angle=32,facecolor=GREEN,alpha=.3,edgecolor=GREEN,lw=2))
arrow(axes[0],(0,.15),(.85,.68),GREEN); arrow(axes[0],(0,.15),(-.2,.47),BLUE)
label(axes[0],.9,.88,r'$s_1$',GREEN);label(axes[0],-.45,.58,r'$s_2$',BLUE)
axes[1].add_patch(Ellipse((0,.15),1.55,1.55,facecolor=GREEN,alpha=.3,edgecolor=GREEN,lw=2))
arrow(axes[1],(0,.15),(.77,.15),GREEN);label(axes[1],.35,-.05,r'$s$',GREEN)
for ax in axes: label(ax,0,-.80,'高斯核等值面的二维截面',size=18)
label(axes[0],-1.18,-1.45,r'$\Sigma=R$',size=22)
matrix(axes[0], [[r'$s_1^2$',r'$0$',r'$0$'],[r'$0$',r'$s_2^2$',r'$0$'],[r'$0$',r'$0$',r'$s_3^2$']])
label(axes[0],1.03,-1.45,r'$R^T$',size=22)
label(axes[1],-1.18,-1.45,r'$\Sigma=s^2$',size=22)
matrix(axes[1], [[r'$1$',r'$0$',r'$0$'],[r'$0$',r'$1$',r'$0$'],[r'$0$',r'$0$',r'$1$']])
save(fig,'29-02-anisotropic-vs-isotropic.png')

fig, axes=canvas()
axes[0].set_title('跟踪：固定地图，优化位姿',fontsize=23,color=INK,pad=18)
axes[1].set_title('建图：固定位姿，优化地图',fontsize=23,color=INK,pad=18)
cloud(axes[0],(.2,.15)); camera(axes[0],(-1,-.6),color='#dabba5');camera(axes[0],(-.45,-.6));arrow(axes[0],(-.9,-.6),(-.55,-.6))
label(axes[0],0,1.15,r'$\Theta_{k-1}$ 固定',GREEN,20);label(axes[0],-.7,-1.03,r'$T_k^{(0)}\ \longrightarrow\ T_k^*$',size=22)
cloud(axes[1],(-.12,.08),color='#b5cbb1'); cloud(axes[1],(.2,.15)); camera(axes[1],(-.65,-.6))
arrow(axes[1],(.02,.5),(.34,.58),GREEN)
label(axes[1],0,1.15,r'$T_k^*$ 固定',ORANGE,20);label(axes[1],0,-1.03,r'$\Theta^{(0)}\ \longrightarrow\ \Theta_k^*$',size=22)
save(fig,'29-03-alternating-pose-map.png')

fig, axes=canvas()
axes[0].set_title('深度给出射线上的位置',fontsize=24,color=INK,pad=18)
axes[1].set_title('位姿误差会写入地图',fontsize=24,color=INK,pad=18)
origin=np.array([-1.2,-.6]);camera(axes[0],origin,orient=.55)
pts=np.array([[.3,-.2],[.8,.45],[.05,.85]])
for p in pts:
    axes[0].plot([origin[0],p[0]],[origin[1],p[1]],color=BLUE,lw=1.6)
    axes[0].plot(*p,'o',color=GREEN,ms=12)
direction=np.array([np.cos(.55),np.sin(.55)])
normal=np.array([-direction[1],direction[0]])
plane_center=origin+.85*direction
plane_ends=np.vstack([plane_center-.52*normal,plane_center+.52*normal])
axes[0].plot(plane_ends[:,0],plane_ends[:,1],color=ORANGE,lw=5)
axis_end=origin+2.35*direction
axes[0].plot([origin[0],axis_end[0]],[origin[1],axis_end[1]],color=INK,lw=1.5,ls='--')
axes[0].annotate('光轴（Z轴）',xy=axis_end,xytext=(.80,.90),color=INK,
                 fontsize=17,ha='center',va='center',
                 arrowprops={'arrowstyle':'-','color':INK,'lw':1.2})
label(axes[0],-.95,.25,'像平面',ORANGE,17)
label(axes[0],-.5,1.1,'像素与深度 $z$',BLUE,21)
label(axes[0],0,-1.05,r'$p_c=zK^{-1}[u,v,1]^T$',size=22)
camera(axes[1],origin);camera(axes[1],origin+np.array([.3,0]),color='#dabba5')
for p in pts:
    axes[1].plot(*p,'o',color=GREEN,ms=12);axes[1].plot(*(p+np.array([.3,0])),'o',color=ORANGE,ms=8)
    arrow(axes[1],p,p+np.array([.28,0]),ORANGE,width=1.5)
label(axes[1],0,1.17,'参考点与位姿扰动后的点',size=18)
label(axes[1],0,-1.05,r'$\mu_w=R_{wc}p_c+c_w$',size=22)
save(fig,'29-04-depth-and-pose-feedback.png')

fig, axes=canvas()
axes[0].set_title('图像评价：观察外观',fontsize=24,color=INK,pad=18)
axes[1].set_title('轨迹评价：观察运动',fontsize=24,color=INK,pad=18)
xy=np.linspace(-1,1,64);xx,yy=np.meshgrid(xy,xy)
im=np.stack([.45+.2*np.cos(xx*4)*np.cos(yy*3),.6+.18*np.sin(xx*3),.35+.16*np.cos(yy*4)],axis=-1)
axes[0].imshow(np.clip(im,0,1),extent=[-1,1,-.7,1]);label(axes[0],0,-1.07,'PSNR / SSIM：图像相似',size=20)
t=np.linspace(0,1,60);gt=np.c_[1.8*t-.9,.6*np.sin(1.2*np.pi*t)+.05];est=gt+np.c_[.3*t*t,-.5*t*t]
axes[1].plot(gt[:,0],gt[:,1],color=GREEN,lw=3);axes[1].plot(est[:,0],est[:,1],color=ORANGE,lw=3,ls='--')
for i in [20,35,50]:arrow(axes[1],gt[i],est[i],BLUE,width=1.4)
label(axes[1],-.8,1.15,'参考',GREEN,20);label(axes[1],.75,1.15,'估计',ORANGE,20)
label(axes[1],0,-1.07,'ATE / RPE：位姿误差',size=20)
save(fig,'29-05-image-vs-trajectory-evaluation.png')

fig, axes=canvas()
for ax,title in zip(axes,['随机空间分布','沿参考轮廓分布']):
    ax.set_title(title,fontsize=24,color=INK,pad=18)
    x=np.linspace(-1.2,1.2,80);ax.plot(x,.6*np.sin(2*x),color=INK,lw=2)
rng=np.random.default_rng(31);random_pts=rng.uniform([-1.2,-.8],[1.2,.8],(20,2))
surface_x=np.linspace(-1.2,1.2,20);surface_pts=np.c_[surface_x,.6*np.sin(2*surface_x)]
for ax,pts in zip(axes,[random_pts,surface_pts]):
    for p in pts: ax.add_patch(Ellipse(p,.26,.20,facecolor=GREEN,edgecolor=GREEN,alpha=.24))
    ax.scatter(pts[:,0],pts[:,1],c=GREEN,s=17,zorder=5)
    label(ax,0,-1.12,r'$N=20$',size=23)
save(fig,'05-01-capacity-vs-coverage.png')

fig, axes=canvas(3,height=4.6)
for ax,title in zip(axes,['随机空间点','SfM 稀疏点','深度反投影']):
    ax.set_title(title,fontsize=24,color=INK,pad=14)
    x=np.linspace(-1.1,1.1,60);ax.plot(x,.55*np.sin(2*x),color=INK,lw=2)
pts=rng.uniform([-1.1,-.9],[1.1,.9],(18,2));axes[0].scatter(pts[:,0],pts[:,1],c=GREEN,s=40)
x=np.linspace(-1.0,1.0,9); axes[1].scatter(x,.55*np.sin(2*x),c=GREEN,s=60)
camera(axes[1],(-1,-.9));camera(axes[1],(.8,-.9),orient=2.3)
x=np.linspace(-.9,.9,13);y=.55*np.sin(2*x);camera(axes[2],(-1,-.95),orient=.55)
for p in np.c_[x,y][::3]: axes[2].plot([-1,p[0]],[-.95,p[1]],color=BLUE,lw=1,alpha=.6)
axes[2].scatter(x,y,c=GREEN,s=45)
for ax,txt in zip(axes,['预设初始化分布','多视图几何先验','观测深度先验']):label(ax,0,-1.18,txt,size=19)
save(fig,'05-02-initialization-priors.png')

(LAB/'outputs/3dgs-theory-redesign-001').mkdir(parents=True,exist_ok=True)
(LAB/'outputs/3dgs-theory-redesign-001/root-figure-manifest.json').write_text(json.dumps({
    'kind':'original_scientific_illustrations', 'basis':['https://arxiv.org/abs/2308.04079','https://arxiv.org/abs/2312.02126'],
    'not_paper_figures':True, 'not_experimental_results':True, 'files':saved,
    'generator':'scripts/draw_3dgs_bridge_figures.py', 'seed':31
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'generated':len(saved),'files':saved},ensure_ascii=False))

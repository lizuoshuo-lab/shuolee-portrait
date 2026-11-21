"""Deterministic 3DGS teaching figures. No dataset, GPU, TeX, or AI imagery.

Run: .venv/Scripts/python.exe -X utf8 scripts/draw_3dgs_theory_figures.py
The examples illustrate mathematics; they are not measured project results.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs' / '3dgs-theory-figures'
OUT.mkdir(parents=True, exist_ok=True)
BG, INK = '#FBFAF6', '#273C3B'
GREEN, CYAN, ORANGE = '#248367', '#239CAE', '#DA8641'
GREY, LIGHT = '#92A5A4', '#E5ECE8'
RED, BLUE = '#E74C4C', '#427BDC'
CMAP = LinearSegmentedColormap.from_list('warm_green', [BG, '#D2E8DD', GREEN])
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 26,
                     'axes.titlesize': 28, 'axes.labelsize': 25,
                     'xtick.labelsize': 20, 'ytick.labelsize': 20,
                     'mathtext.fontset': 'dejavusans', 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'savefig.facecolor': BG,
                     'axes.edgecolor': GREY, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'lines.linewidth': 3})
MANIFEST = []

def figure(n=1, ratios=None, height=6.6):
    fig, axs = plt.subplots(1, n, figsize=(16, height), dpi=100,
                            gridspec_kw={'width_ratios': ratios} if ratios else None)
    fig.subplots_adjust(left=.045, right=.97, top=.86, bottom=.17, wspace=.30)
    return fig, np.atleast_1d(axs)

def clean(ax, xlim=None, ylim=None, equal=False):
    ax.set_axis_off()
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if equal: ax.set_aspect('equal')

def note(ax, xy, s, color=INK, size=26, **kw):
    ax.text(*xy, s, fontsize=size, color=color, ha=kw.pop('ha', 'center'),
            va=kw.pop('va', 'center'), **kw)

def arrow(ax, a, b, color=INK, text=None, rad=0, lw=3, **kw):
    arr = FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=24, linewidth=lw,
                          color=color, connectionstyle=f'arc3,rad={rad}', **kw)
    ax.add_patch(arr)
    if text:
        mid = (np.array(a)+np.array(b))/2
        note(ax, (mid[0], mid[1]+.17), text, color, 24)

def ellipse(ax, center, cov, k=1, color=GREEN, fill=True, alpha=.23, lw=3, **kw):
    val, vec = np.linalg.eigh(np.asarray(cov))
    major = vec[:, -1]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    e = Ellipse(center, 2*k*np.sqrt(val[-1]), 2*k*np.sqrt(val[0]), angle=angle,
                facecolor=color if fill else 'none', edgecolor=color,
                alpha=alpha if fill else 1, linewidth=lw, **kw)
    ax.add_patch(e)
    return e

def rot(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s],[s,c]])

def ellipsoid(ax, center=(0,0), scales=(1.8,.9,.5), color=GREEN):
    """Axonometric projection of a true 3D ellipsoid and its principal axes."""
    a,b,c=np.radians([28,-22,16])
    rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    ry=np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    rx=np.array([[1,0,0],[0,np.cos(c),-np.sin(c)],[0,np.sin(c),np.cos(c)]])
    R=rz@ry@rx
    P=np.array([[1,0,-.65],[0,1,.38]])
    mu=np.array(center)
    t=np.linspace(0,2*np.pi,180)
    for phi in np.linspace(-np.pi/2,np.pi/2,11):
        xyz=np.array([np.cos(t)*np.cos(phi),np.sin(t)*np.cos(phi),np.full_like(t,np.sin(phi))])
        uv=(P@R@(np.array(scales)[:,None]*xyz)).T+mu
        ax.plot(uv[:,0],uv[:,1],color=color,alpha=.35,linewidth=1.4)
    for theta in np.linspace(0,np.pi,9):
        xyz=np.array([np.cos(theta)*np.cos(t),np.sin(theta)*np.cos(t),np.sin(t)])
        uv=(P@R@(np.array(scales)[:,None]*xyz)).T+mu
        ax.plot(uv[:,0],uv[:,1],color=color,alpha=.3,linewidth=1.4)
    ax.scatter(*mu,s=65,c=INK,zorder=8)
    axes=P@R@np.diag(scales)
    for i,col in enumerate([ORANGE,CYAN,GREEN]):
        end=mu+axes[:,i]
        arrow(ax,mu,end,col,lw=3)
        label=mu+1.23*axes[:,i]
        note(ax,label,rf'$s_{i+1}$',col,28)
    return axes

def kernel(xx,yy,mu,cov):
    delta=np.stack([xx-mu[0], yy-mu[1]],axis=-1)
    power=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(cov),delta)
    return np.exp(-.5*power)

def camera(ax, center, target=(0,0), color=CYAN, size=.35):
    center=np.array(center,dtype=float); target=np.array(target,dtype=float)
    d=target-center; d/=np.linalg.norm(d)
    side=np.array([-d[1],d[0]])
    tri=np.array([center+size*d,center-size*d+.72*size*side,center-size*d-.72*size*side])
    ax.add_patch(Polygon(tri,closed=True,facecolor=color,edgecolor=color,alpha=.9))
    return center+size*d

def save(fig, name, use, calculation=''):
    path=OUT/(name+'.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    with Image.open(path) as im: size=list(im.size)
    MANIFEST.append({'file':path.name, 'assetPath':'outputs/3dgs-theory-figures/'+path.name,
                     'kind':'self_drawn_scientific_illustration', 'readingUse':use,
                     'calculation':calculation, 'pixelSize':size,
                     'attribution':'本站示意，依据 Kerbl 等（2023）；自绘图，不是原论文图。'})

def covariance_matrix_formula(fig, box, fontsize=26):
    """Typeset a real diagonal matrix with vector brackets, not a code operator."""
    ax = fig.add_axes(box)
    clean(ax, (0,12), (0,3))
    note(ax, (1.65,1.5), r'$\Sigma=R(q)$', size=fontsize)
    note(ax, (10.1,1.5), r'$R(q)^T$', size=fontsize)
    for x, side in [(3.7,1),(8.3,-1)]:
        ax.plot([x+side*.18,x,x,x+side*.18],[2.75,2.75,.25,.25],color=INK,lw=2)
    for row,y in enumerate([2.3,1.5,.7]):
        for col,x in enumerate([4.5,6,7.5]):
            value=rf'$s_{row+1}^2$' if row==col else r'$0$'
            note(ax,(x,y),value,size=fontsize)

def copy_paper():
    source=LAB/'outputs'/'3dgs-foundations-figures'
    manifest=json.loads((source/'figure-manifest.json').read_text(encoding='utf-8'))
    for original,new in [('figure-02-3dgs-pipeline.png','01-paper-pipeline.png'),
                         ('figure-03-anisotropic-gaussians.png','02-paper-anisotropic.png')]:
        shutil.copy2(source/original,OUT/new)
        old=next(v for v in manifest['figures'] if Path(v['assetPath']).name==original)
        MANIFEST.append({'file':new,'assetPath':'outputs/3dgs-theory-figures/'+new,
                         'kind':'original_paper_crop_copied_unchanged','figNo':old['figNo'],
                         'pdfPage1Based':old['pdfPage1Based'],'sourceURL':old['sourceURL'],
                         'pixelSize':old['assetPixelSize'],'license':'CC-BY-4.0',
                         'licenseURL':'https://creativecommons.org/licenses/by/4.0/',
                         'attribution':old['attribution'],'captionOriginal':old['captionOriginal']})

def shape_parameters():
    fig,axs=figure(2,[1.15,1])
    ax=axs[0]; clean(ax,(-3.0,3.0),(-2.0,2.1),True)
    ax.set_title('中心、三轴尺度与旋转',pad=26)
    ellipsoid(ax)
    note(ax,(.1,-.35),r'$\mu$',INK,30)
    covariance_matrix_formula(fig,[.035,.012,.48,.235],fontsize=24)
    ax=axs[1]; clean(ax,(-3.0,3.0),(-2.0,2.1),True)
    ax.set_title('同中心，不同形状',pad=26)
    ellipse(ax,(-1.4,.55),np.eye(2)*.7**2,color=CYAN)
    ellipse(ax,(1.35,.55),rot(.4)@np.diag([1.05**2,.34**2])@rot(.4).T,color=GREEN)
    ax.scatter([-1.4,1.35],[.55,.55],s=45,c=INK)
    note(ax,(-1.4,-.65),'各向同性',CYAN)
    note(ax,(1.35,-.65),'各向异性',GREEN)
    note(ax,(0,-1.7),'尺度 '+r'$s$'+' 是标准差，方差为 '+r'$s^2$',size=26)
    save(fig,'02-gaussian-shape-parameters','高斯中心、三条标准差主轴与旋转共同决定协方差。','3D ellipsoid wireframes projected axonometrically; positive principal scales.')

def coordinate_perspective():
    fig,axs=figure(3,[1,1,1.05],height=6.7)
    ax=axs[0];clean(ax,(-1.5,4.5),(-1.3,4.7),True);ax.set_title('世界坐标 W',pad=22)
    cw=np.array([.1,.15]);Rwc=rot(-.33);pc=np.array([.8,4]);pw=cw+Rwc@pc
    arrow(ax,(-1,-.8),(1,-.8),GREY);arrow(ax,(-1,-.8),(-1,1.15),GREY)
    note(ax,(1.05,-1.08),r'$X_w$',GREY,23);note(ax,(-1.05,1.45),r'$Z_w$',GREY,23)
    ax.scatter(*cw,s=80,c=CYAN);ax.scatter(*pw,s=100,c=GREEN)
    arrow(ax,cw,pw,GREEN)
    for vec,label in [(Rwc[:,0]*1.0,r'$X_c$'),(Rwc[:,1]*1.05,r'$Z_c$')]:
        arrow(ax,cw,cw+vec,CYAN,lw=2)
        note(ax,cw+vec*1.3,label,CYAN,23)
    note(ax,(-.35,.25),r'$c_w$',CYAN,28);note(ax,pw+np.array([.25,.35]),r'$p_w$',GREEN,30)
    ax=axs[1];clean(ax,(-1.4,2.5),(-.7,5.4),True);ax.set_title('相机坐标 C',pad=22)
    arrow(ax,(0,0),(1.95,0),GREY);arrow(ax,(0,0),(0,4.9),GREY)
    note(ax,(2.15,0),r'$X$',GREY);note(ax,(0,5.14),r'$Z$',GREY)
    note(ax,(-.55,2.9),'光轴（相机 Z 轴）',GREY,22,rotation=90)
    ax.scatter([0],[0],s=60,c=CYAN,zorder=7)
    note(ax,(-.30,-.42),'相机中心',CYAN,22)
    ax.plot([0,.8],[0,4],color=GREEN);ax.scatter([.8],[4],s=100,c=GREEN)
    ax.plot([-1,1.7],[1,1],color=ORANGE,linewidth=4)
    ax.scatter([.2],[1],s=110,c=ORANGE,zorder=7)
    note(ax,(1.1,4.3),r'$p_c$',GREEN,29);note(ax,(1.28,1.45),'虚拟像平面',ORANGE,22)
    ax.plot([0,.18,.18],[1.18,1.18,1],color=ORANGE,lw=2)
    note(ax,(1.28,.66),'垂直于光轴',ORANGE,22)
    note(ax,(.65,2.4),r'$Z>0$',GREEN,26)
    ax=axs[2];ax.set_title('像素坐标',pad=22)
    ax.set_xlim(0,640);ax.set_ylim(480,0);ax.set_aspect('equal')
    ax.add_patch(Rectangle((0,0),640,480,facecolor=LIGHT,edgecolor=GREY,linewidth=2))
    ax.scatter([320],[240],s=75,c=GREY);ax.scatter([440],[300],s=110,c=GREEN)
    arrow(ax,(320,240),(440,300),GREEN)
    note(ax,(278,203),r'$(c_x,c_y)$',GREY,23);note(ax,(490,337),r'$m$',GREEN,29)
    ax.set_xticks([0,320,640]);ax.set_yticks([0,240,480]);ax.set_xlabel('u / px');ax.set_ylabel('v / px')
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.5,.055,r'$p_c=R_{cw}(p_w-c_w)\qquad u=f_x\frac{X}{Z}+c_x,\quad v=f_y\frac{Y}{Z}+c_y$',ha='center',fontsize=27)
    save(fig,'03-coordinate-perspective','同一个点依次变换到相机坐标，再由正轴向深度投到像素。','pc=(0.8,0.4,4), fx=fy=600, cx=320, cy=240 gives m=(440,300). Left panel shows the XZ cross-section.')

def covariance_projection():
    fig,axs=figure(2,[1,1.1]);ax=axs[0]
    clean(ax,(-2.6,2.6),(-1.7,1.7),True);ax.set_title('相机中的三维高斯',pad=26)
    ellipsoid(ax,scales=(1.45,.8,.45))
    note(ax,(0,-.3),r'$\mu_c$',size=29)
    note(ax,(0,-1.5),r'$\Sigma_c\quad [L^2]$',size=30)
    mu=np.array([.8,.4,4.]);R=np.eye(3);R[:2,:2]=rot(.6)
    cov=R@np.diag([.30**2,.18**2,.12**2])@R.T
    X,Y,Z=mu;J=np.array([[600/Z,0,-600*X/Z**2],[0,600/Z,-600*Y/Z**2]])
    cov2=J@cov@J.T
    ax=axs[1];ax.set_title('中心处局部线性近似',pad=26)
    ellipse(ax,(0,0),cov2,k=2,color=CYAN)
    ax.scatter([0],[0],s=85,c=INK)
    ax.set_xlim(-105,105);ax.set_ylim(-85,85);ax.set_aspect('equal')
    ax.set_xlabel('Δu / px');ax.set_ylabel('Δv / px')
    ax.set_xticks([-80,0,80]);ax.set_yticks([-60,0,60]);ax.grid(alpha=.15)
    note(ax,(0,-73),r'$\Sigma_{2D}\quad [px^2]$',CYAN,28)
    fig.text(.5,.057,r'$\delta m\approx J\delta x\qquad\Sigma_{2D}\approx J\Sigma_cJ^T$',ha='center',fontsize=30)
    save(fig,'04-covariance-projection','中心处投影雅可比把三维局部协方差传播成二维协方差。','Right ellipse computed from J Sigma_c J.T, two-standard-deviation contour, before adding the pixel-space low-pass. Left ellipsoid is an axonometric conceptual shape.')

def footprint_alpha():
    fig,axs=figure(3,[1,1,1.3],height=6.7)
    v=np.linspace(-2.4,2.4,201);x,y=np.meshgrid(v,v);g=kernel(x,y,[0,0],np.diag([1,.6**2]))
    for ax,data,title in zip(axs[:2],[g,.6*g],['高斯核 '+r'$g$','像素 '+r'$\alpha=0.6g$']):
        ax.imshow(data,extent=[-2.4,2.4,-2.4,2.4],origin='lower',cmap=CMAP,vmin=0,vmax=1)
        ellipse(ax,(0,0),np.diag([1,.6**2]),fill=False,color=ORANGE,lw=2)
        ax.set_title(title,pad=20);ax.set_xticks([-1,0,1]);ax.set_yticks([]);ax.set_xlabel('横向偏移 / '+r'$\sigma_x$')
        ax.scatter([0],[0],s=60,c=INK);note(ax,(0,1.65),r'$1$' if data is g else r'$0.6$',size=28)
    ax=axs[2];ax.set_title('沿主轴剖面',pad=20)
    xx=np.linspace(-2.4,2.4,300);gg=np.exp(-.5*xx**2)
    ax.plot(xx,gg,color=GREEN,label=r'$g$');ax.plot(xx,.6*gg,color=CYAN,label=r'$\alpha$')
    ax.axvline(1,color=ORANGE,ls='--',lw=2);ax.scatter([1,1],[np.exp(-.5),.6*np.exp(-.5)],color=[GREEN,CYAN],s=65)
    note(ax,(1.08,.81),r'$\sigma_x$',ORANGE,25);ax.set_xlim(-2.4,2.4);ax.set_ylim(0,1.15)
    ax.set_xticks([-2,0,1,2]);ax.set_yticks([0,.6,1]);ax.grid(alpha=.14);ax.legend(fontsize=25,loc='upper left',frameon=False)
    fig.text(.5,.05,'中心：1 / 0.6     一标准差：0.607 / 0.364',ha='center',fontsize=27)
    save(fig,'05-footprint-alpha','同一投影分布乘不同不透明度，中心与一标准差处的像素不透明度随之改变。','g=exp(-0.5(x²+y²/0.6²)); alpha=0.6g, with shared heatmap normalization [0,1].')

def sh_direction():
    fig,axs=figure(2,[1.25,1]);ax=axs[0];clean(ax,(-3.8,3.8),(-3.1,3.1),True)
    ax.set_title('几何固定，观察方向改变',pad=24)
    ellipse(ax,(0,0),rot(.3)@np.diag([.7**2,.35**2])@rot(.3).T)
    vs=np.array([[1,0],[-.5,np.sqrt(3)/2],[-.5,-np.sqrt(3)/2]])
    colors=[]
    for k,v in enumerate(vs):
        col=(.5+.2*v[0],.5+.2*v[1],.5);colors.append(col)
        cw=-2.7*v;start=camera(ax,cw,color=CYAN,size=.35)
        arrow(ax,start,-.75*v,GREEN,lw=3)
        note(ax,cw*1.19,str(k+1),CYAN,27)
        note(ax,-1.6*v+np.array([.2,.1]),rf'$v_{k+1}$',GREEN,26)
    note(ax,(0,.73),r'$\mu$',size=30)
    ax=axs[1];clean(ax,(0,4),(0,4.8));ax.set_title('同一高斯的视角相关颜色',pad=24)
    for k,c in enumerate(colors):
        yy=3.5-k*1.28
        ax.add_patch(Rectangle((.2,yy-.35),1.0,.7,facecolor=c,edgecolor=GREY,linewidth=2))
        note(ax,(2.7,yy),rf'$c_{k+1}=({c[0]:.2f},{c[1]:.2f},0.50)$',size=24)
    fig.text(.5,.052,r'$v=\frac{\mu-c_w}{\|\mu-c_w\|}\quad c(v)=(0.5+0.2v_x,\ 0.5+0.2v_y,\ 0.5)$',ha='center',fontsize=27)
    save(fig,'06-sh-directional-color','箭头从相机指向高斯；固定几何的颜色可依观察方向变化。','Illustrative degree-1 direction function c(v)=(.5+.2vx,.5+.2vy,.5), not trained colors or a physical BRDF.')

def front_to_back():
    fig,axs=figure(2,height=6.7)
    for ax,reverse in zip(axs,[False,True]):
        clean(ax,(-.6,6.2),(-.3,4.25));ax.set_title('红在前，蓝在后' if not reverse else '蓝在前，红在后',pad=23)
        front=RED if not reverse else BLUE;back=BLUE if not reverse else RED
        af=.6 if not reverse else .5;ab=.5 if not reverse else .6
        arrow(ax,(.15,2.75),(5.65,2.75),GREY,lw=2)
        for xx,col,aa in [(1.5,front,af),(3.3,back,ab)]:
            ax.add_patch(Ellipse((xx,2.75),.7,1.25,facecolor=col,edgecolor=col,alpha=.65))
            note(ax,(xx,3.85),rf'$\alpha={aa}$',col,27)
        weights=[af,(1-af)*ab,(1-af)*(1-ab)]
        for xx,ww,col in zip([.2,2.1,4.0],weights,[front,back,'#333333']):
            ax.add_patch(Rectangle((xx,.8),1.45,.65,facecolor=col,edgecolor='none'))
            note(ax,(xx+.725,.40),rf'$w={ww:g}$',size=27)
        rgb=(.6,0,.2) if not reverse else (.3,0,.5)
        ax.add_patch(Rectangle((4.9,2.15),1.1,1.05,facecolor=rgb,edgecolor=INK,linewidth=2))
        note(ax,(5.45,1.83),'RGB',size=23)
    fig.text(.5,.046,r'$C_{\rm red\ first}=(0.6,0,0.2)\qquad C_{\rm blue\ first}=(0.3,0,0.5)$',ha='center',fontsize=28)
    save(fig,'07-front-to-back-color','后方颜色先乘前方透射率；交换前后顺序改变最终RGB。','Exact alpha compositing: red alpha .6, blue alpha .5, black background. Weights .6/.2/.2 vs .5/.3/.2; RGB patches show computed linear values.')

def tile_overlap():
    fig,axs=figure(2,[1.2,1],height=6.8);ax=axs[0]
    clean(ax,(0,7),(0,5),True);ax.set_title('投影覆盖范围与图像分块',pad=23)
    specs=[(np.array([2.5,2.5]),rot(.4)@np.diag([.8**2,.35**2])@rot(.4).T,GREEN,'A'),
           (np.array([4.,2.2]),rot(-.45)@np.diag([.55**2,.32**2])@rot(-.45).T,ORANGE,'B')]
    for mu,cov,col,label in specs:
        radius=3*np.sqrt(np.linalg.eigvalsh(cov)[-1])
        lo=np.maximum(np.floor(mu-radius).astype(int),0);hi=np.minimum(np.ceil(mu+radius).astype(int),[7,5])
        ax.add_patch(Rectangle(lo,*(hi-lo),facecolor=col,alpha=.09,edgecolor='none'))
        ellipse(ax,mu,cov,k=2,color=col)
        note(ax,mu+np.array([0,.4]),label,col,30)
    for x in range(8):ax.axvline(x,color=GREY,lw=1.2)
    for y in range(6):ax.axhline(y,color=GREY,lw=1.2)
    ax.add_patch(Rectangle((3,2),1,1,facecolor='none',edgecolor=INK,linewidth=4))
    ax=axs[1];ax.set_title('候选 A、B：逐像素 '+r'$\alpha$',pad=23)
    x,y=np.meshgrid(np.linspace(3,4,120),np.linspace(2,3,120));g=kernel(x,y,specs[0][0],specs[0][1])
    ax.imshow(.6*g,origin='lower',extent=[3,4,2,3],cmap=CMAP,vmin=0,vmax=.6)
    ax.set_xticks([]);ax.set_yticks([])
    for x in np.linspace(3,4,7):ax.axvline(x,color=BG,lw=1.1)
    for y in np.linspace(2,3,7):ax.axhline(y,color=BG,lw=1.1)
    note(ax,(3.5,2.08),'A 的像素不透明度',GREEN,24,
         bbox={'facecolor':BG,'alpha':.90,'edgecolor':'none','pad':4})
    fig.text(.5,.045,'候选范围允许冗余；像素核再判断贡献',ha='center',fontsize=28)
    save(fig,'08-tile-overlap','屏幕范围决定tile候选，而同一tile内的各像素仍有自己的核值与alpha。','Two covariance ellipses with enclosing 3-max-axis radius tile ranges. The zoom is alpha_A=.6g_A in tile [3,4]×[2,3], not a combined opacity image.')

def gradient_chain():
    fig,axs=figure(height=6.5);ax=axs[0];clean(ax,(0,16),(0,5.5))
    def node(x,y,s,col=GREEN,w=1.6):
        ax.add_patch(Rectangle((x-w/2,y-.39),w,.78,facecolor=col,alpha=.12,edgecolor=col,lw=2))
        note(ax,(x,y),s,col,28)
    node(.9,2.7,r'$\mathcal{L}$',INK,1.3);node(3,2.7,r'$C$',INK,1.2)
    node(5,4.2,r'$c_i$',CYAN);node(5,1.8,r'$\alpha_i$',GREEN)
    node(8,4.8,r'$f_{lm}$',CYAN);node(8,.65,r'$o_i$',ORANGE)
    node(8,2.4,r'$m,\Sigma_{2D}$',GREEN,2.3);node(11.5,2.4,r'$\mu,\Sigma_w$',GREEN,2.3)
    node(14.7,2.4,r'$s,q$',GREEN,1.6)
    arrow(ax,(1.6,2.7),(2.35,2.7));arrow(ax,(3.6,2.85),(4.25,3.85),CYAN)
    arrow(ax,(3.6,2.55),(4.25,2.05),GREEN);arrow(ax,(5.8,4.35),(7.15,4.7),CYAN)
    arrow(ax,(5.8,1.9),(6.8,2.3),GREEN);arrow(ax,(5.75,1.55),(7.2,.8),ORANGE)
    arrow(ax,(9.2,2.4),(10.3,2.4),GREEN);arrow(ax,(12.7,2.4),(13.8,2.4),GREEN)
    note(ax,(13.25,3.06),'经 '+r'$\Sigma_w$',GREEN,24)
    arrow(ax,(5.8,4.1),(11,2.83),CYAN,rad=-.12)
    note(ax,(9.6,3.88),'经视线方向 '+r'$v$',CYAN,24)
    ax.plot([.7,15.5],[.02,.02],ls='--',color=GREY,lw=2)
    fig.text(.5,.055,'连续分支反传梯度；排序 / 候选变化有离散边界',ha='center',fontsize=28)
    save(fig,'09-gradient-chain','颜色、alpha与协方差分支把图像误差传回几何、opacity和SH；均值也通过方向影响颜色。','Directed edges show gradient dependency, with fixed local sorting/candidate branches; no derivative is assigned to discrete sorting.')

def sfm_initialization():
    fig,axs=figure(2,[1,1]);rng=np.random.default_rng(13)
    pts=np.column_stack([np.linspace(-1.7,1.7,11),.28*np.sin(np.linspace(-2,2,11))+rng.normal(0,.08,11)])
    for k,ax in enumerate(axs):
        clean(ax,(-3.4,3.4),(-2.0,2.7),True)
        ax.set_title('已知相机与 SfM 稀疏点' if k==0 else '同坐标系与尺度的高斯',pad=23)
        for c in [(-2.6,1.7),(0,2.25),(2.6,1.7)]:
            camera(ax,c,color=CYAN,size=.25)
            ax.plot([c[0],0],[c[1],.1],ls='--',color=CYAN,alpha=.28,lw=1.8)
        if k==0:ax.scatter(pts[:,0],pts[:,1],s=70,c=GREEN)
        else:
            for p in pts:ellipse(ax,p,np.eye(2)*.16**2,color=GREEN,alpha=.25)
            ax.scatter(pts[:,0],pts[:,1],s=20,c=GREEN)
        note(ax,(0,-1.1),r'$\mu\leftarrow p_{SfM}$' if k else '相机固定',size=28)
    fig.text(.5,.054,'每个稀疏点初始化一个不透明度较低的高斯',ha='center',fontsize=28)
    save(fig,'28-01-sfm-initialization','同一坐标系的稀疏点初始化中心、局部尺度与低不透明度高斯。','Deterministic point coordinates; same fixed cameras and unchanged point centers in both panels.')

def parameter_constraints():
    fig,axs=figure(3,[1.2,1,1],height=6.8)
    fig.subplots_adjust(bottom=.38)
    ax=axs[0];clean(ax,(-2.5,2.5),(-2.4,2.4),True);ax.set_title('几何与外观参数',pad=22)
    ellipsoid(ax,scales=(1.45,.75,.42));note(ax,(0,-.3),r'$\mu$',size=29)
    note(ax,(0,1.9),r'$q=\frac{q_{\mathrm{raw}}}{\|q_{\mathrm{raw}}\|}$',CYAN,25)
    note(ax,(0,-1.55),'不透明度 '+r'$o$',size=24)
    note(ax,(0,-2.04),'视角相关颜色',size=24)
    ax=axs[1];ax.set_title('指数映射：正尺度',pad=22)
    x=np.linspace(-2,1.1,200);ax.plot(x,np.exp(x),color=GREEN)
    ax.set_xlim(-2,1.1);ax.set_ylim(0,3.3);ax.set_xticks([-2,0,1]);ax.set_yticks([0,1,3]);ax.grid(alpha=.16)
    ax.set_xlabel(r'$\ell$');ax.set_ylabel(r'$\frac{s}{s_{\mathrm{ref}}}=e^{\ell}$')
    ax=axs[2];ax.set_title('不透明度约束',pad=22)
    x=np.linspace(-6,6,200);o=1/(1+np.exp(-x));ax.plot(x,o,color=ORANGE)
    ax.set_xlim(-6,6);ax.set_ylim(0,1);ax.set_xticks([-6,0,6]);ax.set_yticks([0,.5,1]);ax.grid(alpha=.16)
    ax.set_xlabel(r'$a$');ax.set_ylabel(r'$o=\frac{1}{1+e^{-a}}$')
    covariance_matrix_formula(fig,[.115,.005,.77,.24],fontsize=26)
    save(fig,'28-02-parameter-constraints','原始参数先经过激活或归一化，得到合法尺度、旋转和opacity。','Exponent and logistic activation curves; ellipsoid axes and normalized quaternion covariance relation.')

def toy_image(shift=0, brightness=0):
    x,y=np.meshgrid(np.linspace(-1,1,64),np.linspace(-1,1,64))
    im=np.zeros((64,64,3));im[:]=[.89,.89,.84]
    rect=(np.abs(x-shift)<.58)&(np.abs(y)<.7)
    im[rect]=[.22,.55,.46]
    stripe=rect&(np.sin(17*y)>0)
    im[stripe]=[.12,.28,.25]
    return np.clip(im+brightness,0,1)

def multiview_loop():
    fig,axs=figure(3,[1.05,1,1],height=6.7)
    ax=axs[0];clean(ax,(-2.8,2.8),(-1.8,2.8),True);ax.set_title('① 选择训练视图',pad=22)
    for pos,label,col in [((-2,1.5),'A',GREY),((0,2.3),'B',CYAN),((2,1.5),'C',GREY)]:
        camera(ax,pos,color=col,size=.3);note(ax,(pos[0],pos[1]+.4),label,col,27)
    for xx in [-.8,0,.8]:ellipse(ax,(xx,0),np.diag([.38**2,.25**2]),color=GREEN)
    arrow(ax,(0,1.9),(0,.45),CYAN);note(ax,(0,-1.05),'当前地图',GREEN,28)
    for ax,img,title in [(axs[1],toy_image(.08),'② 当前地图渲染'),(axs[2],toy_image(),'③ 对应真实图像')]:
        ax.imshow(img);ax.set_title(title,pad=22);ax.set_xticks([]);ax.set_yticks([])
    fig.text(.5,.062,'损失函数 → 反向传播 → 参数更新 → 下一训练视图',ha='center',fontsize=28)
    save(fig,'28-03-multiview-training-loop','已知相机中随机抽取一帧，渲染当前地图并与该帧对应RGB比较，更新后换视角。','Toy image pair with a horizontal structure offset, not real training images or measurements. Cameras are fixed inputs.')

def ssim_value(a,b):
    # Independent teaching metric, valid 7x7 Gaussian windows, per RGB channel.
    v=np.arange(7)-3;w=np.exp(-v*v/(2*1.5**2));w/=w.sum();win=w[:,None]*w[None,:]
    def conv(im):
        views=np.lib.stride_tricks.sliding_window_view(im,(7,7),axis=(0,1))
        return np.einsum('hwcij,ij->hwc',views,win)
    ma,mb=conv(a),conv(b);va=conv(a*a)-ma*ma;vb=conv(b*b)-mb*mb;ab=conv(a*b)-ma*mb
    return float(np.mean(((2*ma*mb+.01**2)*(2*ab+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))))

def loss_fig():
    fig,axs=figure(3,height=6.7);target=toy_image()
    fig.subplots_adjust(bottom=.30)
    images=[target,toy_image(brightness=.10),toy_image(shift=.22)]
    for ax,im,title in zip(axs,images,['目标 RGB','整体颜色偏移','局部结构偏移']):
        ax.imshow(im);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,pad=22)
        l1=float(np.abs(im-target).mean());ss=ssim_value(im,target)
        value=rf'$L_1={l1:.3f}$'+'\n'+rf'$\mathrm{{SSIM}}={ss:.3f}$'
        ax.text(.5,-.10,value,transform=ax.transAxes,ha='center',va='top',fontsize=26)
    fig.text(.5,.043,r'$\mathcal{L}=0.8\,L_1+0.2\,(1-\mathrm{SSIM})$',ha='center',fontsize=30)
    save(fig,'28-04-l1-ssim-loss','逐像素颜色偏差与局部结构统计提供互补的图像约束。','Synthetic RGB arrays. L1 averaged over all pixels/channels; displayed SSIM uses a declared 7×7 valid Gaussian window (sigma 1.5), illustrating the concept rather than exact official boundary/window settings.')

def gradient_adam():
    fig,axs=figure(2,[1.1,1],height=6.8);ax=axs[0];ax.set_title('二维二次目标：更新示意',pad=22)
    fig.subplots_adjust(bottom=.25)
    x,y=np.meshgrid(np.linspace(-.4,2.4,180),np.linspace(-.4,2.4,180))
    loss=.5*(x*x+12*y*y);ax.contour(x,y,loss,levels=[.2,.6,1.5,3,6,12,20,30],colors=GREY,alpha=.55,linewidths=1.5)
    th=np.array([2.,2.]);m=np.zeros(2);v=np.zeros(2);path=[th.copy()]
    for t in range(1,19):
        g=th*np.array([1.,12.]);m=.9*m+.1*g;v=.999*v+.001*g*g
        mh=m/(1-.9**t);vh=v/(1-.999**t);th=th-.12*mh/(np.sqrt(vh)+1e-8);path.append(th.copy())
    path=np.array(path);ax.plot(path[:,0],path[:,1],'-o',color=GREEN,markersize=5)
    ax.scatter([0],[0],s=90,c=ORANGE,marker='*');note(ax,(1.78,2.22),'起点',GREEN,25)
    ax.set_xlim(-.35,2.45);ax.set_ylim(-.35,2.45);ax.set_xlabel(r'$\theta_1$');ax.set_ylabel(r'$\theta_2$');ax.set_xticks([0,1,2]);ax.set_yticks([0,1,2])
    ax=axs[1];clean(ax,(0,6),(0,5));ax.set_title('一阶 / 二阶矩调节步长',pad=22)
    xs=np.array([1.1,3.4]);g0=np.array([2.,24.]);scaled=g0/24
    ax.bar(xs,scaled,width=.7,color=[GREEN,ORANGE],alpha=.8)
    note(ax,(1.1,1.4),r'$g_1$',GREEN,28);note(ax,(3.4,1.4),r'$g_2$',ORANGE,28)
    note(ax,(2.6,2.2),'梯度量级不同',size=27)
    note(ax,(2.7,3.1),r'$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$',size=24)
    note(ax,(2.7,4.1),r'$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$',size=24)
    fig.text(.5,.080,r'$\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$',ha='center',va='center',fontsize=28)
    save(fig,'28-05-gradient-adam','梯度经一阶与二阶矩归一化形成参数步长，方向和单位影响更新。','Computed Adam path on L=.5(theta1²+12theta2²), beta1=.9, beta2=.999, lr=.12, epsilon=1e-8, bias correction enabled. Illustrative target, no 3DGS convergence claim.')

def gradient_statistics():
    fig,axs=figure(4,[1,1,1,1.35],height=6.5)
    vectors=[(1,0),(-1,0),(0,2)]
    for k,(ax,g) in enumerate(zip(axs[:3],vectors)):
        clean(ax,(-1.5,1.5),(-.7,2.6),True);ax.set_title(f'视图 {k+1}',pad=23)
        ax.plot([-1.35,1.35],[0,0],color=GREY,lw=1.2);ax.plot([0,0],[-.45,2.4],color=GREY,lw=1.2)
        arrow(ax,(0,0),g,GREEN,lw=4);ax.scatter([0],[0],s=50,c=INK)
        note(ax,(0,-.55),str(g),size=25)
    ax=axs[3];ax.set_title('先求范数，再平均',pad=23)
    ax.barh([1,0],[4/3,2/3],color=[GREEN,GREY],height=.42)
    ax.set_yticks([]);ax.set_xlim(0,1.7);ax.set_xticks([0,2/3,4/3],['0','2/3','4/3'])
    note(ax,(.62,1.0),'范数的平均',color=BG,size=24)
    note(ax,(.82,.40),'平均向量的范数',size=24)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    ax.tick_params(axis='y',labelsize=23,length=0)
    fig.text(.5,.045,r'$\bar g=\frac{1+1+2}{3}=\frac{4}{3}\qquad\|\bar{\mathbf{g}}\|=\frac{2}{3}$',ha='center',fontsize=29)
    save(fig,'28-06-screen-gradient-statistics','先累加每次梯度模，再除以可统计观察次数，避免反方向向量直接抵消。','Eligible vectors (1,0), (-1,0), (0,2); mean norm 4/3, norm of mean vector 2/3, count n=3.')

def clone_split():
    fig,axs=figure(height=7.2);ax=axs[0];clean(ax,(0,16),(-.3,6.2))
    for row,y in [('clone',4.45),('split',1.5)]:
        note(ax,(1.3,y+1.15),r'$\bar g\geq\tau$'+('，小尺度：复制' if row=='clone' else '，大尺度：分裂'),size=28,ha='left')
        arrow(ax,(4.3,y),(6.0,y),GREY);arrow(ax,(10.0,y),(11.6,y),GREY)
    small=np.eye(2)*.45**2;big=rot(.3)@np.diag([1.3**2,.65**2])@rot(.3).T
    ellipse(ax,(2.8,4.4),small,color=GREEN);ax.scatter([2.8],[4.4],s=40,c=GREEN)
    ellipse(ax,(8,4.4),small,fill=False,color=GREEN,lw=5)
    ellipse(ax,(8,4.4),small,fill=False,color=ORANGE,lw=3,linestyle='--')
    ax.scatter([8],[4.4],s=50,c=INK);note(ax,(8,3.58),'同中心的两份参数',size=25)
    for mu,col in [((12.5,4.35),GREEN),((13.6,4.55),ORANGE)]:ellipse(ax,mu,small,color=col)
    note(ax,(13.1,3.58),'各自继续优化',size=25)
    ellipse(ax,(2.8,1.5),big,color=GREEN)
    child=big/1.6**2
    ellipse(ax,(8,1.5),big,fill=False,color=GREY,lw=2,linestyle='--')
    ax.plot([6.85,9.15],[.95,2.05],color=GREY,lw=2);ax.plot([6.85,9.15],[2.05,.95],color=GREY,lw=2)
    for mu,col in [((7.35,1.3),GREEN),((8.65,1.8),ORANGE)]:ellipse(ax,mu,child,color=col)
    note(ax,(8,.27),'采样子点，删除父点',size=25)
    for mu,col in [((12.45,1.3),GREEN),((13.6,1.9),ORANGE)]:ellipse(ax,mu,child,color=col)
    note(ax,(13.1,.27),r'子尺度 $\frac{s}{\gamma}$，$\gamma>1$',size=25)
    fig.text(.5,.045,'离散结构更新；不保证体积、不透明度或图像不变',ha='center',fontsize=28)
    save(fig,'28-07-clone-split','投影位置梯度较大的候选按三维尺度采用复制或分裂；复制时同中心，分裂后父点删除。','Clone uses identical centers/scales as early code. Split child scale s/1.6 for N=2; illustrative nonzero sampled offsets rotated to world coordinates, not a measured training result.')

def prune_reset():
    fig,axs=figure(height=7.4);ax=axs[0];clean(ax,(0,16),(-.1,6.3))
    note(ax,(.8,5.9),'剪枝：弱贡献点被删除',ha='left',size=29)
    for x,o,col in [(2.3,.75,GREEN),(3.8,.004,GREY),(5.3,.5,GREEN)]:
        ellipse(ax,(x,4.8),np.eye(2)*.43**2,color=col,alpha=max(.05,o*.5))
    ax.plot([3.45,4.15],[4.45,5.15],color=ORANGE);ax.plot([3.45,4.15],[5.15,4.45],color=ORANGE)
    note(ax,(3.8,3.95),r'$o$'+' 很小',GREY,25);arrow(ax,(6.3,4.8),(8.6,4.8),GREY)
    for x in [10.4,13.0]:ellipse(ax,(x,4.8),np.eye(2)*.43**2,color=GREEN,alpha=.35)
    note(ax,(11.7,3.95),'移除弱贡献基元',size=26)
    note(ax,(.8,3.35),'重置不透明度：保留高斯基元',ha='left',size=29)
    for base,ops,title in [(2.1,[.8,.5],'前景遮挡'),(7.7,[.01,.01],'重置'),(12.7,[.04,.65],'随后继续优化')]:
        for off,col,o in [(-.55,ORANGE,ops[0]),(.55,GREEN,ops[1])]:
            xx=base+off;ellipse(ax,(xx,1.8),np.diag([.22**2,.52**2]),color=col,alpha=max(.08,o*.8))
            note(ax,(xx,1.0),f'{o:g}',col,25)
        note(ax,(base,2.55),title,size=25)
    arrow(ax,(4.1,1.8),(5.9,1.8),GREY);arrow(ax,(9.4,1.8),(11.1,1.8),GREY)
    fig.text(.5,.045,r'$o^{\prime}=\min(o,\varepsilon_o),\quad\varepsilon_o=0.01$'+'（示例） 不删除基元',ha='center',fontsize=28)
    save(fig,'28-08-prune-opacity-reset','剪枝减少基元数量；重置仅降低不透明度，随后继续优化，并不保证恢复。','Prune illustrative low-opacity point. Reset caps both foreground and background opacity at .01. The last panel is one possible later opacity pattern, not guaranteed recovery.')

def novel_view_evaluation():
    fig,axs=figure(2,[1.15,1],height=6.7);ax=axs[0]
    clean(ax,(-3.8,3.8),(-3.0,3.0),True);ax.set_title('训练相机 / 独立留出相机',pad=22)
    for xx in [-.9,0,.9]:ellipse(ax,(xx,0),rot(.3)@np.diag([.5**2,.25**2])@rot(.3).T)
    for pos,col in [((-2.8,1.8),CYAN),((0,2.55),CYAN),((2.8,1.8),CYAN),((2.8,-1.8),ORANGE)]:
        start=camera(ax,pos,color=col,size=.32);arrow(ax,start,np.array(pos)*.35,col,lw=2)
    note(ax,(-1.9,-1.6),'训练',CYAN,29);note(ax,(2.6,-2.55),'留出',ORANGE,29)
    ax=axs[1];clean(ax,(0,6),(0,5));ax.set_title('外观泛化与几何分别判断',pad=22)
    ax.imshow(toy_image(),extent=[.3,2.65,2.25,4.6]);ax.imshow(toy_image(.12),extent=[3.1,5.45,2.25,4.6])
    note(ax,(1.48,1.85),'训练视图',CYAN,26);note(ax,(4.28,1.85),'新视角',ORANGE,26)
    note(ax,(2.9,.75),'PSNR / SSIM / LPIPS',size=27)
    fig.text(.5,.045,'图像质量好 ≠ 几何正确；几何需独立深度 / 表面参考',ha='center',fontsize=28)
    save(fig,'28-09-training-novel-view-evaluation','固定训练与留出相机分组，图像泛化评价不能代替几何参考。','Synthetic image examples and camera schematic, with no measured metric or geometric-accuracy claim.')

def main():
    copy_paper()
    for fn in [shape_parameters,coordinate_perspective,covariance_projection,
               footprint_alpha,sh_direction,front_to_back,tile_overlap,gradient_chain,
               sfm_initialization,parameter_constraints,multiview_loop,loss_fig,
               gradient_adam,gradient_statistics,clone_split,prune_reset,novel_view_evaluation]:
        fn()
    obj={'title':'Traditional 3DGS theory-step figures','created':'2026-10-11',
         'generator':'scripts/draw_3dgs_theory_figures.py','seedPolicy':'fixed deterministic examples',
         'sourcePaper':'https://arxiv.org/abs/2308.04079',
         'sourceCodeCommit':'b2ada78a779ba0455dfdc2b718bdf1726b05a1b6',
         'sourceRasterizerCommit':'8064f52ca233942bdec2d1a1451c026deedd320b',
         'selfDrawnNotice':'Scientific teaching illustrations, not original paper figures, experiment logs or fitted project results.',
         'figures':MANIFEST}
    (OUT/'figure-manifest.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'figures':len(MANIFEST),'files':[x['file'] for x in MANIFEST]},ensure_ascii=False))

if __name__=='__main__':
    main()

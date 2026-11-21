"""Actual 3-D mathematical scenes, not experimental reconstructions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LAB=Path(__file__).resolve().parents[1]
OUT=LAB/'outputs/3dgs-paper-course-001/figures'
OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':13,'axes.unicode_minus':False,
    'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'stix'})
BLUE='#347ea8'; GREEN='#3d927c'; RED='#ce6757'; GRAY='#80909c'
records=[]

def rot(ax,ay,az):
    cx,sx=np.cos(ax),np.sin(ax);cy,sy=np.cos(ay),np.sin(ay);cz,sz=np.cos(az),np.sin(az)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def ellipsoid(ax,mu,scales,R=np.eye(3),color=GREEN,alpha=.42,camera=False):
    u=np.linspace(0,2*np.pi,49);v=np.linspace(0,np.pi,25)
    xyz=np.array([np.outer(np.cos(u),np.sin(v)),np.outer(np.sin(u),np.sin(v)),np.outer(np.ones_like(u),np.cos(v))])
    xyz=(R@(xyz.reshape(3,-1)*np.asarray(scales)[:,None])).reshape(xyz.shape)+np.asarray(mu)[:,None,None]
    if camera:xyz=xyz[[2,0,1]]
    ax.plot_surface(*xyz,color=color,alpha=alpha,linewidth=.35,edgecolor=color,rstride=3,cstride=3,shade=True)
    return xyz

def setup(ax,lims=(-1.6,1.6),elev=23,azim=-60,camera=False):
    ax.set_facecolor('white');ax.view_init(elev=elev,azim=azim)
    ax.xaxis.pane.fill=False;ax.yaxis.pane.fill=False;ax.zaxis.pane.fill=False
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):a._axinfo['grid'].update(color=(.8,.85,.88,.65),linewidth=.55)
    if camera:
        ax.set_xlabel('$Z_c$',labelpad=9);ax.set_ylabel('$X_c$',labelpad=9);ax.set_zlabel('$Y_c$',labelpad=9)
    else:ax.set_xlabel('$X$',labelpad=6);ax.set_ylabel('$Y$',labelpad=6);ax.set_zlabel('$Z$',labelpad=6)
    ax.set_xlim(lims);ax.set_ylim(lims);ax.set_zlim(lims)
    ax.tick_params(labelsize=10,pad=0);ax.set_box_aspect((1,1,1))

def camera_scene(ax,end=5.3):
    setup(ax,camera=True,elev=16,azim=-61)
    ax.set_xlim(-.2,end);ax.set_ylim(-1.55,1.55);ax.set_zlim(-1.3,1.3)
    ax.set_box_aspect((end,3.1,2.6),zoom=1.15)
    ax.plot([0,end],[0,0],[0,0],ls='--',color=BLUE,lw=1.8)
    ax.quiver(end-.24,0,0,.24,0,0,color=BLUE,arrow_length_ratio=.7,lw=1.5)
    ax.scatter([0],[0],[0],color='black',s=35,depthshade=False)
    ax.text(0,-.48,-.55,'投影中心 $O$',fontsize=13)
    ax.text(end-.48,.15,-.32,'$+Z_c$',color=BLUE,fontsize=14)

def plane(ax,z,fov=(1.15,.78),alpha=.14):
    x,y=fov
    pts=np.array([[z,-x,-y],[z,x,-y],[z,x,y],[z,-x,y]])
    ax.add_collection3d(Poly3DCollection([pts],facecolors='#f1b774',edgecolors='#b58447',linewidths=1,alpha=alpha))
    return pts

def segment_cam(ax,a,b,**kw):
    a=np.asarray(a)[[2,0,1]];b=np.asarray(b)[[2,0,1]]
    ax.plot(*np.array([a,b]).T,**kw)

def save(fig,name,description,parameters):
    fig.savefig(OUT/name,dpi=150,bbox_inches=None)
    plt.close(fig)
    from PIL import Image
    records.append({'filename':name,'kind':'self-drawn-3d-scientific-illustration','pixelSize':list(Image.open(OUT/name).size),
        'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'description':description,
        'sourceTheory':'Kerbl et al. (2023), 3D Gaussian Splatting, Sections 4–6; additional elementary pinhole geometry definitions.',
        'parameters':parameters,'disclaimer':'教学三维示意，不是论文实验结果。绘出的高斯椭球是等值面，不是实体边界。'})

# 1. Rotate a genuine ellipsoid with the same semi-axis lengths.
fig=plt.figure(figsize=(11.6,6.1));fig.suptitle('三维高斯：标准差决定主轴长度，旋转决定空间方向',fontsize=19,y=.96)
s=np.array([1.12,.62,.28]);R=rot(.34,.62,.64)
for i,(r,title) in enumerate([(np.eye(3),'局部主轴与坐标轴对齐'),(R,'同一椭球旋转至世界坐标系')]):
    ax=fig.add_subplot(1,2,i+1,projection='3d');setup(ax);ellipsoid(ax,[0,0,0],s,r)
    ax.set_title(title,fontsize=15,pad=14)
    for k,col in enumerate([RED,BLUE,'#8665ad']):
        vec=r[:,k]*s[k]
        ax.quiver(0,0,0,*vec,color=col,arrow_length_ratio=.14,lw=2)
        labelpos=vec*1.22
        if k==2:labelpos+=np.array([-.35,-.15,.55])
        ax.text(*labelpos,'$s_'+str(k+1)+'$',color=col,fontsize=16)
    ax.scatter([0],[0],[0],color='black',s=25)
fig.text(.5,.055,r'$(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})=1$',ha='center',fontsize=19)
fig.text(.5,.01,'绘制的是同一高斯的等值面；高斯函数在椭球外仍有非零值。',ha='center',fontsize=12,color='#55636a')
fig.subplots_adjust(top=.84,bottom=.18,left=.04,right=.97,wspace=.02)
save(fig,'geometry-ellipsoid.png','同一标准差三元组的轴对齐和旋转椭球，以及旋转后的三个主轴。',{'standardDeviations':[1.12,.62,.28],'eulerAnglesRadians':[.34,.62,.64],'level':'Mahalanobis squared distance 1'})

# 2. A genuine perspective ray passing through a virtual image plane.
fig=plt.figure(figsize=(11.6,6.4));ax=fig.add_subplot(111,projection='3d');camera_scene(ax)
f=1.5;mu=np.array([.78,.48,4.25]);p=mu*f/mu[2];plane(ax,f)
ellipsoid(ax,mu,[.36,.24,.31],rot(.2,.3,.5),BLUE,.5,camera=True)
segment_cam(ax,[0,0,0],mu,color='#343b42',lw=2.1)
ax.scatter([p[2]],[p[0]],[p[1]],color=RED,s=45,depthshade=False)
ax.scatter([f],[0],[0],color=BLUE,s=25,depthshade=False)
ax.text(f,-1.32,1.12,'虚拟像平面 $Z_c=f$',fontsize=13,color='#8c602b')
ax.text(mu[2]-.5,mu[0]-.15,mu[1]+.56,r'三维中心 $\boldsymbol{\mu}_c$',fontsize=14)
ax.text(p[2]+.03,p[0]+.12,p[1]+.38,'投影点',fontsize=13,color=RED)
ax.text(f,.05,-.26,'主点',fontsize=12,color=BLUE)
ax.set_title('透视投影：三维点、投影中心和像平面投影点共线',fontsize=19,pad=8)
fig.text(.5,.065,'蓝色虚线为光轴；虚拟像平面垂直于相机 $Z_c$ 轴。',ha='center',fontsize=13)
fig.text(.5,.015,r'$X_{\mathrm{im}}=fX_c/Z_c,\qquad Y_{\mathrm{im}}=fY_c/Z_c,\qquad Z_c>0$',ha='center',fontsize=18)
fig.subplots_adjust(top=.88,bottom=.15,left=.03,right=.96)
save(fig,'geometry-perspective.png','相机轴、垂直虚拟像平面、三维高斯中心射线与实际透视投影点。',{'focalLengthInSceneUnits':f,'gaussianCenterCameraCoordinates':mu.tolist(),'imagePlane':'positive virtual plane; not physical sensor plane'})

# 3. First-order projected covariance, embedded in the same actual 3-D scene.
fig=plt.figure(figsize=(11.6,6.5));ax=fig.add_subplot(121,projection='3d');camera_scene(ax,5.7)
f=1.7;mu=np.array([.53,.25,4.35]);s=np.array([.68,.28,.4]);R=rot(.4,.6,-.45);cov=R@np.diag(s*s)@R.T
xyz=ellipsoid(ax,mu,s,R,GREEN,.42,camera=True);plane(ax,f,fov=(1.25,.8),alpha=.17)
J=np.array([[f/mu[2],0,-f*mu[0]/mu[2]**2],[0,f/mu[2],-f*mu[1]/mu[2]**2]])
cov2=J@cov@J.T;vals,V=np.linalg.eigh(cov2);theta=np.linspace(0,2*np.pi,180)
ellipse=V@np.array([np.sqrt(vals[0])*np.cos(theta),np.sqrt(vals[1])*np.sin(theta)])+f*mu[:2,None]/mu[2]
ax.plot(np.full(theta.shape,f),ellipse[0],ellipse[1],color=RED,lw=3)
segment_cam(ax,[0,0,0],mu,color='#263c40',lw=1.8)
ax.text(f,-1.28,1.12,'像平面',fontsize=13,color='#8c602b')
ax.text(mu[2]-.7,mu[0]-.15,mu[1]+.7,'三维协方差椭球',fontsize=13,color=GREEN)
ax.set_title('三维高斯及其投影',fontsize=15,pad=12)
bx=fig.add_subplot(122,projection='3d');setup(bx,camera=True,elev=16,azim=-61)
bx.set_xlim(f-.12,f+.12);bx.set_ylim(-.16,.57);bx.set_zlim(-.25,.43)
bx.set_box_aspect((.24,.73,.68),zoom=.95)
zoom_plane=np.array([[f,-.16,-.25],[f,.57,-.25],[f,.57,.43],[f,-.16,.43]])
bx.add_collection3d(Poly3DCollection([zoom_plane],facecolors='#f1b774',edgecolors='#b58447',linewidths=1,alpha=.17))
bx.plot(np.full(theta.shape,f),ellipse[0],ellipse[1],color=RED,lw=3)
bx.scatter([f],[f*mu[0]/mu[2]],[f*mu[1]/mu[2]],color='#333c42',s=20)
bx.set_xticks([f]);bx.set_yticks([0,.2,.4]);bx.set_zticks([-.2,0,.2,.4])
bx.set_xlabel('')
bx.set_title('同一像平面上的局部放大',fontsize=15,pad=12)
fig.suptitle('投影协方差：把三维局部变化映射至像平面',fontsize=19,y=.97)
fig.text(.74,.145,'红色：二维协方差椭圆（一阶近似）',ha='center',color=RED,fontsize=12)
fig.text(.5,.055,r'$\Sigma_{\mathrm{im}}\approx J\Sigma_cJ^\mathsf{T}$',ha='center',fontsize=21)
fig.text(.5,.014,'红色椭圆按中心处的雅可比计算；有限体积的透视投影不是严格线性变换。',ha='center',fontsize=12,color='#55636a')
fig.subplots_adjust(top=.84,bottom=.22,left=.03,right=.96,wspace=.05)
save(fig,'geometry-projection-covariance.png','真实三维椭球与其中心处雅可比一阶协方差投影椭圆，未声称为精确透视轮廓。',{'muCamera':mu.tolist(),'standardDeviations':s.tolist(),'f':f,'projectedCovariance':cov2.tolist(),'approximation':'first-order perspective projection at Gaussian mean'})

# 4. Fixed 3-D primitive, cameras orbit it; colors genuinely depend on direction.
fig=plt.figure(figsize=(11.6,6.5));ax=fig.add_subplot(111,projection='3d');setup(ax,lims=(-3.8,3.8),elev=24,azim=-52)
ellipsoid(ax,[0,0,0],[.82,.36,.55],rot(.12,.4,.22),'#9baab2',.38)
th=np.linspace(0,2*np.pi,160);ax.plot(3.05*np.cos(th),3.05*np.sin(th),np.full(th.shape,1.05),ls='--',color=GRAY,lw=1)
viewcolors=[]
for i,t in enumerate([-.3,1.2,2.7,4.2]):
    cam=np.array([3.05*np.cos(t),3.05*np.sin(t),1.05]);d=-cam/np.linalg.norm(cam)
    col=np.array([.5+.2*d[0],.5+.2*d[1],.5])
    viewcolors.append(col.tolist());u=np.cross(d,[0,0,1]);u/=np.linalg.norm(u);v=np.cross(d,u)
    cen=cam+.22*d;quad=np.array([cen+a*u+b*v for a,b in [(-.3,-.22),(.3,-.22),(.3,.22),(-.3,.22)]])
    ax.add_collection3d(Poly3DCollection([quad],facecolors=[col],edgecolors='#33444c',linewidths=1.3))
    ax.scatter(*cam,color='#33444c',s=15);ax.plot(*np.array([[0,0,0],cam]).T,color=GRAY,alpha=.55,lw=1)
    ax.text(*(cam+np.array([0,0,.4])),f'视角 {i+1}',fontsize=13)
ax.text(-.4,-.1,-1.15,'固定的三维高斯',fontsize=13)
ax.set_title('球谐函数：同一空间位置可随观察方向呈现不同颜色',fontsize=18,pad=8)
fig.text(.5,.075,r'$\mathbf{v}=(\boldsymbol{\mu}-\mathbf{c}_w)/\|\boldsymbol{\mu}-\mathbf{c}_w\|$',ha='center',fontsize=19)
fig.text(.5,.021,'相机旁的小色块表示对应方向的 RGB；几何位置和协方差始终固定。',ha='center',fontsize=12,color='#55636a')
fig.subplots_adjust(top=.88,bottom=.17,left=.06,right=.94)
save(fig,'geometry-sh-views.png','环绕固定三维高斯的四个相机位置及由零阶、一阶方向函数产生的RGB色块。',{'directionConvention':'camera center to Gaussian mean, matching chapter 27','colorFunction':'[0.5+0.2*v_x, 0.5+0.2*v_y, 0.5], matching chapter 27 (an illustrative degree <=1 real SH span)','colorsRGB':viewcolors,'notExperiment':True})

# 5. True composited central pixel next to the physical 3-D arrangement.
fig=plt.figure(figsize=(11.6,6.5));ax=fig.add_subplot(121,projection='3d');camera_scene(ax,5.7)
ellipsoid(ax,[0,0,2.3],[.53,.42,.44],rot(.2,.3,.25),RED,.48,camera=True)
ellipsoid(ax,[0,0,4.6],[.72,.5,.55],rot(.1,-.2,.3),BLUE,.42,camera=True)
segment_cam(ax,[0,0,0],[0,0,5.2],color='#37474f',lw=2)
ax.text(2.3,-.72,.8,'近处高斯',color=RED,fontsize=13);ax.text(4.6,-.65,.92,'远处高斯',color=BLUE,fontsize=13)
ax.set_title('同一像素射线上的空间顺序',fontsize=15,pad=10)
cred=np.array([.81,.4,.34]);cblue=np.array([.24,.51,.72]);bg=np.ones(3);a1=.6;a2=.7
out=a1*cred+(1-a1)*a2*cblue+(1-a1)*(1-a2)*bg
bx=fig.add_subplot(122);bx.set_xlim(0,1);bx.set_ylim(0,1);bx.axis('off')
bx.text(.5,.88,'中心像素的实际合成色',ha='center',fontsize=16)
bx.add_patch(plt.Rectangle((.29,.49),.42,.27,facecolor=out,edgecolor='#48545b',lw=1))
for y,col,txt in [(.36,cred,'近处：'+r'$\alpha_1=0.6$'),(.25,cblue,'远处：'+r'$\alpha_2=0.7$'),(.14,bg,'背景：白色')]:
    bx.add_patch(plt.Rectangle((.15,y),.08,.06,facecolor=col,edgecolor='#7d878b',lw=.6));bx.text(.28,y+.025,txt,va='center',fontsize=13)
fig.suptitle('遮挡与颜色合成：近处的不透明度减少远处的贡献',fontsize=19,y=.97)
fig.text(.5,.057,r'$\mathbf{C}=0.60\mathbf{c}_1+0.28\mathbf{c}_2+0.12\mathbf{c}_{\mathrm{bg}}$',ha='center',fontsize=21)
fig.text(.5,.013,'取像素位于两个投影中心，二维高斯核值均为 1；系数之和为 1。',ha='center',fontsize=12,color='#55636a')
fig.subplots_adjust(top=.85,bottom=.17,left=.02,right=.97,wspace=.04)
save(fig,'geometry-occlusion.png','共线的前后两个三维高斯及按alpha合成公式计算的实际中心像素颜色。',{'alpha':[a1,a2],'RGBnear':cred.tolist(),'RGBfar':cblue.tolist(),'RGBbackground':bg.tolist(),'RGBcomposite':out.tolist(),'weights':[.6,.28,.12]})

# 6. The underlying geometry is a 3-D curved strip, not a 2-D outline.
fig=plt.figure(figsize=(11.6,8));fig.suptitle('自适应密度控制：增加局部表达能力，再继续优化',fontsize=20,y=.97)
for i in range(4):
    ax=fig.add_subplot(2,2,i+1,projection='3d');setup(ax,lims=(-1.4,1.4),elev=22,azim=-62)
    x=np.linspace(-1.2,1.2,50);y=np.linspace(-.48,.48,15);xx,yy=np.meshgrid(x,y);zz=.29*np.sin(2.6*xx)+.1*yy
    ax.plot_surface(xx,yy,zz,color='#f0be76',alpha=.4,edgecolor='#ac9064',linewidth=.35,rstride=3,cstride=6)
    ax.set_xlim(-1.4,1.4);ax.set_ylim(-.95,.95);ax.set_zlim(-.9,.9);ax.set_box_aspect((2.8,1.9,1.8))
    if i==0:
        ellipsoid(ax,[-.48,0,.29*np.sin(-1.248)],[.24,.18,.075],rot(0,-.23,0),GREEN,.67)
        title='小高斯：局部表面覆盖不足'
    elif i==1:
        for px in [-.48,-.07]:ellipsoid(ax,[px,0,.29*np.sin(2.6*px)],[.24,.18,.075],rot(0,-.23,0),GREEN,.67)
        title='复制：新增同尺度高斯并调整位置'
    elif i==2:
        ellipsoid(ax,[0,0,.03],[.98,.45,.28],rot(0,-.13,0),BLUE,.52);title='大高斯：细小几何表示过粗'
    else:
        for px in [-.39,.38]:ellipsoid(ax,[px,0,.29*np.sin(2.6*px)],[.5,.23,.145],rot(0,-.35,0),BLUE,.56)
        title='分裂：用较小高斯继续优化'
    ax.set_title(title,fontsize=14,pad=8)
    ax.set_xticks([-1,0,1]);ax.set_yticks([-.5,0,.5]);ax.set_zticks([-.5,0,.5])
fig.text(.5,.038,'橙色曲面为局部目标表面，椭球为高斯等值面；新高斯的位置仅作教学示意。',ha='center',fontsize=12,color='#55636a')
fig.text(.5,.01,'复制与分裂会改变渲染结果；这里不假设体积、不透明度或颜色贡献严格守恒。',ha='center',fontsize=12,color='#55636a')
fig.subplots_adjust(top=.87,bottom=.13,left=.04,right=.96,wspace=.04,hspace=.29)
save(fig,'geometry-density-control.png','三维曲面上小高斯复制和大高斯分裂的前后空间示意。',{'surface':'z = 0.29 sin(2.6x) + 0.1y','interpretation':'qualitative geometry demonstration; not the official stochastic splitting distribution','conservationAssumed':False})

(OUT/'geometry-manifest.json').write_text(json.dumps({'assets':records,'generator':'scripts/draw_3dgs_paper_course_geometry.py','verifiedByVisualInspection':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'count':len(records),'filenames':[r['filename'] for r in records]},ensure_ascii=False))

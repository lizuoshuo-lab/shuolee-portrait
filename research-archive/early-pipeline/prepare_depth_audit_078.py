from pathlib import Path
import json,datetime,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/static-depth-mixture-audit-078'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
details=read(O/'fixed-ray-details.json')['rows'];epi=read(O/'epipolar-result.json');ray=read(O/'fixed-ray-result.json')
depth=read(O/'depth-result.json');grad=read(O/'gradient-result.json')
fig,axs=plt.subplots(1,2,figsize=(15,4.6),dpi=150,sharey=True)
for ax,g,title in zip(axs,[1,2],['下半脸：78 条轨迹，717 组对应','上半脸：63 条轨迹，588 组对应']):
    rr=[r for r in details if r['group']==g]
    for field,label,color,style in [
        ('current_error_px','077 高斯深度','#b44735','-'),
        ('error_px','单射线深度拟合（留出检查）','#176b8a','-'),
        ('perpendicular_px','逐对极线距离（诊断下界）','#777777','--')]:
        x=np.sort([r[field] for r in rr]);y=np.arange(1,len(x)+1)/len(x)
        ax.step(np.r_[0,x],np.r_[0,y],where='post',lw=2.1,color=color,ls=style,label=label)
    ax.set_xlim(0,43);ax.set_ylim(0,1.01);ax.set_xticks([0,10,20,30,40]);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_title(title,fontproperties=font,fontsize=15,pad=14)
    ax.set_xlabel('对应误差 / 完整去畸变图像像素',fontproperties=font,fontsize=13)
    ax.grid(alpha=.18);ax.legend(prop=font,loc='lower right',frameon=False)
axs[0].set_ylabel('累计比例',fontproperties=font,fontsize=13)
fig.tight_layout(pad=1.4);fig.savefig(O/'held-depth-ecdf.png',facecolor='white');plt.close(fig)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
decision=dict(updated=now,stage='078',classification='progress',new_gaussian_training=False,
    latest_gaussian_round='077',model_unchanged=True,optimizer_updates=0,root_cause_established=False,quality_passed=False,
    summary='175帧整图相机与人体初始化流程已完成。078冻结077继续诊断，未训练或替换高斯。面部对应误差主要沿极线方向，单射线深度的留出拟合可改善一部分误差，但仍有局部不一致，不能认定相机准确或唯一根因。',
    findings=[
        '48个脸外采样位置的高斯中心深度90%宽度中位数为人物稳健纵向跨度的3.03%，48个内部对照为1.52%。样本来自固定空间带，未按残差排序，不等同于精确鼻口语义。',
        '24个下半脸纹理位置的回投误差：混合平面8.02像素，中位平面8.63，最大贡献平面8.57。简单替换深度读出没有改善该组结果。该实验与717组独立对应的方向和采样不同，不合并统计。',
        '原100帧重影90%贡献由1630个高斯解释。它们在全175帧头部特征库中有可见贡献，不能解释为完全未被监督。',
        '4视图末态梯度中，新增对应项对625个下半脸纹理贡献点的位置梯度范数约为原项合计的1.35%至3.52%。梯度方向有同向也有反向，没有一致的整体抵消证据；这些不是历史Adam更新量。',
        '下半脸717组误差中位数6.172像素，垂直极线分量1.554，沿线分量5.823。逐对误差能量中沿线占比的中位数94.66%。极线方向允许通过改变深度移动，但这一几何自由度不能证明相机真值。',
        '每个源特征只拟合一个深度，使用非留出观测拟合，在连续中间三分之一留出观测检查：下半脸中位误差3.775像素，上半脸2.174。它不是新高斯训练结果，也没有消除全部不一致。'],
    rejected_shortcuts=['直接换成中位深度','仅把已有对应权重加大','把背景误差较小写成面部相机已准确','把逐对极线最短距离作为独立验证结果'],
    remaining=['局部相机误差与图像特征误差、人物微动仍未分离','混合平面的约束是否能充分作用于所有重影高斯','在遮挡和真实薄结构附近，表面收缩是否会损伤人物'],
    next_step='保持全景相机和静态人体初始化主线。先在完整头部对应库上检查约束覆盖与位置梯度，区分抽样不足和混合深度约束不足，再为一种针对可见高斯深度分布的改动设置单变量从头对照。当前不增加步数或扫描权重。',
    next_preflight=dict(gaussian_updates=0,scope='Frozen077, complete head correspondence bank; compare existing sampling expectation and exact full-bank gradient support on the same predetermined groups.',
        decision_gate='If full-bank still misses/conflicts on lower-texture or ghost contributors, increasing sample count alone is rejected. Any surface concentration term must preserve genuine occlusion and be checked in several views before fresh training.',
        preservation='No learned Gaussian, human fit, camera or optimizer edits in preflight. Next actual experiment must reinitialize200000points and train18000steps independently.'),
    open_source_reference=dict(name='2D Gaussian Splatting',repository='https://github.com/hbb1/2d-gaussian-splatting',
        project='https://surfsplatting.github.io/',status='Reference only; no download, integration, or reproduction in078.',
        relevance='Official2DGS combines a surfel rasterizer with depth distortion and normal consistency. It motivates testing per-contribution surface concentration.',
        distinction='PGSR3D Gaussian center variance is not equivalent to2DGS ray-surface depth distortion. Replacing a mean depth by median also does not implement that method.'),
    evidence=['depth-result.json','gradient-result.json','epipolar-result.json','fixed-ray-result.json'],
    graphs=dict(ecdf='held-depth-ecdf.png',provenance='Raw numerical arrays from fixed-ray-details.json, no generated visual content.',
        caveat='Pairs share tracks and are correlated. The gray curve uses target observations to compute the per-pair lower bound, unlike the held-out-depth fit.'))
(O/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sources=O/'source';sources.mkdir(exist_ok=True)
for name in ['audit_depth_mixture_078.py','audit_constraint_gradients_078.py','audit_epipolar_078.py','audit_fixed_ray_fit_078.py']:
    shutil.copy2(B.parents[2]/'repos/PGSR-shuolee'/name,sources/name)
print(json.dumps({'graph':str(O/'held-depth-ecdf.png'),'decision':str(O/'decision.json'),'classification':'progress'}))

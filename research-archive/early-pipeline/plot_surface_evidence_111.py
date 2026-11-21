"""Scientific diagnostic panel: original image samples with measured query positions."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/surface-point-alignment-111'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
records=read(O/'frozen-native-tangent.json')['records']
queries=read(B/'current/material-support-gradient-110/queries.json')
rows=read(B/'current/person-static-shared-face-fresh-098/data/training-inputs.json')['records']
font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
chosen=[(100,7429,'面部与眼镜'),(136,9609,'头顶'),(136,10412,'裤腿')]
fig,axes=plt.subplots(1,3,figsize=(12,4.3),dpi=150)
manifest=[]
for ax,(raw,tid,label) in zip(axes,chosen):
    q=next(q for q in queries if q['raw']==raw and q['track_id']==tid)
    r=next(r for r in records if r['raw']==raw and r['track_id']==tid)
    source=Path(rows[raw-1]['image']);im=Image.open(source).convert('RGB')
    u,v=q['observed_uv']
    left=int(np.clip(round(u)-200,0,im.width-400));top=int(np.clip(round(v)-140,0,im.height-280))
    ax.imshow(np.asarray(im),origin='upper')
    ax.set_xlim(left,left+400);ax.set_ylim(top+280,top)
    ax.plot(u,v,'+',color='#20F090',markersize=13,markeredgewidth=1.5)
    ax.scatter([u],[v],s=100,facecolors='none',edgecolors='#20F090',linewidths=1.5)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    ax.set_title(f'{label} · 实拍 {raw:06d}',fontproperties=font,fontsize=13,pad=10)
    status='达到本项支撑阈值' if r['admitted'] else '未达到本项支撑阈值'
    ax.set_xlabel(f"有效表面交点占贡献权重 {100*r['retained_mass_fraction']:.1f}%\n{status}",fontproperties=font,fontsize=12,labelpad=11)
    manifest.append(dict(raw=raw,track_id=tid,source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),view_box=[left,top,left+400,top+280],record=r))
fig.text(.5,.035,'固定上一轮的三个查询位置；仅核验冻结模型的支撑，不表示新训练效果。',ha='center',fontproperties=font,fontsize=11,color='#44505c')
fig.subplots_adjust(left=.01,right=.99,top=.88,bottom=.22,wspace=.035)
fig.savefig(O/'material-support-review.png',facecolor='white',dpi=150)
plt.close(fig)
(O/'visual-evidence.json').write_text(json.dumps(dict(selection='Same three predefined query locations used in 110, not selected by 111 residual',items=manifest),ensure_ascii=False,indent=2))
print(json.dumps(dict(figure=str(O/'material-support-review.png'),values=[r['record']['retained_mass_fraction'] for r in manifest])))

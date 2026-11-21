from pathlib import Path
import json, datetime as dt
import numpy as np
from PIL import Image, ImageDraw
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
H=B/'current/portrait-alignment-open-source-116/hsfm'; J=H/'joint-diagnostic-reserved'; E=J/'native-fit-cached/evaluation'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def local(p): return Path(p.replace('/mnt/c/','C:/'))
v=np.load(J/'frozen-validation.npz'); bg=np.flatnonzero(v['groups']==0)
arms=['original074','scene_only','scene_human']; data={a:np.load(E/f'{a}-frozen-errors.npz') for a in arms}
rank=bg[np.argsort(data['scene_human']['errors'][bg])]
selected=[int(rank[len(rank)//2]),int(rank[-2]),int(rank[-1])]
rows={r['raw_frame']:r for r in read(H/'input-pilot/inputs.json')['records']}
records=[]
for ordinal,ti in enumerate(selected):
    f=int(v['frames'][ti,1]); p=v['xy'][ti,1]; im=Image.open(local(rows[f]['image'])).convert('RGB')
    # Exact original-pixel crop, no rescaling; selected failures are explicitly labeled.
    box=np.rint([p[0]-160,p[1]-125,p[0]+160,p[1]+125]).astype(int)
    panel=Image.new('RGB',(960,300),'white'); d=ImageDraw.Draw(panel)
    for c,a in enumerate(arms):
        panel.paste(im.crop(tuple(box)),(c*320,50))
        d.text((c*320+7,5),a,fill='black'); d.text((c*320+7,23),f'raw{f:06d}  error={data[a]["errors"][ti]:.2f}px',fill='black')
        pp=p-box[:2]+[c*320,50]; q=data[a]['projected'][ti]-box[:2]+[c*320,50]
        d.line([tuple(pp),tuple(q)],fill='#f1a200',width=2)
        for pt,color in [(pp,'#008dd2'),(q,'#e43535')]:d.ellipse((pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4),outline=color,width=2)
    path=E/f'background-held-review-{ordinal+1}.jpg'; panel.save(path,quality=97)
    records.append(dict(file=path.name,frozen_index=ti,frame=f,selection='median' if ordinal==0 else 'largest_two_joint_errors',errors={a:float(data[a]['errors'][ti]) for a in arms}))
(E/'background-held-review.json').write_text(json.dumps(dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),selection='Diagnostic median and two largest errors, not representative prevalence; all65 remain in metrics',records=records),indent=2)+'\n')
print(json.dumps(records))

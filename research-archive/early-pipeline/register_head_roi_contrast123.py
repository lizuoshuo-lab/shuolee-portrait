"""Register one native-head-crop inference contrast before seeing new predictions."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
Q=R/'video10-correspondence-contrast'; N=R/'video10-semantic-role'; D=Q/'head-roi-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not D.exists(),'One registered contrast only'
old=read(Q/'plan.json'); audit=read(Q/'head-resolution-audit.json'); reg=read(N/'execution-admission.json')['regions']
nodes={}
for t in read(N/'secondary-observations.json')['tracks']:
    for o in t['observations']:nodes[o['frame'],o['feature']]=(t['family'],t['split'])
frames=[644,671,698];queries={};crops={};sources={};review=[]
for f in frames:
    fp=next(Path(n) for n in old['input_sha256'] if Path(n).name==f'features_{f:06d}.npz')
    ip=next(Path(n) for n in old['input_sha256'] if Path(n).name==f'native_frame_{f:06d}.png')
    assert all(sha(z)==old['input_sha256'][str(z)] for z in [fp,ip])
    with np.load(fp) as z:uv=z['uv'].copy()
    x0,y0,x1,y1=reg[str(f)]['head'];ids=np.flatnonzero((uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1))
    cx=int(np.floor((x0+x1)/2))-384;cy=int(np.floor((y0+y1)/2))-384
    cx=max(0,min(cx,2704-768));cy=max(0,min(cy,1520-768))
    crops[str(f)]=dict(source=str(ip),xyxy=[cx,cy,cx+768,cy+768],head_xyxy=[x0,y0,x1,y1])
    queries[str(f)]=[dict(feature=int(i),uv=uv[i].tolist(),family=nodes.get((f,int(i)),(-1,'unassigned'))[0],split=nodes.get((f,int(i)),(-1,'unassigned'))[1]) for i in ids]
    # Deterministic position-based review, never use current/old errors or gate confidence.
    chosen=[]
    for split,count in [('heldout',3),('fit',3),('unassigned',2)]:
        candidates=[q for q in queries[str(f)] if q['split']==split]
        for frac in np.linspace(.25,.75,count):
            target=np.array([x0+(x1-x0)*frac,y0+(y1-y0)*(1-frac)])
            left=[q for q in candidates if q['feature'] not in chosen]
            if not left:continue
            q=min(left,key=lambda q:(float(np.linalg.norm(np.array(q['uv'])-target)),q['feature']))
            chosen.append(q['feature']);review.append(dict(frame=f,**q))
    sources.update({str(fp):sha(fp),str(ip):sha(ip)})
sources.update({str(N/'secondary-observations.json'):sha(N/'secondary-observations.json'),str(Q/'head-resolution-audit.json'):sha(Q/'head-resolution-audit.json')})
D.mkdir();(D/'native-crops').mkdir()
for f,c in crops.items():
    im=Image.open(c['source']);assert im.size==(2704,1520)
    cp=D/'native-crops'/f'head_{int(f):06d}.png';im.crop(c['xyxy']).save(cp)
    c['crop_file']=str(cp);c['sha256']=sha(cp)
save(D/'queries.json',queries);save(D/'review-allocation.json',dict(rows=review,rule='3 HELD +3 FIT +2 unassigned per source where available, nearest frozen diagonal target positions; no inference errors/confidence used.'))
plan=dict(registered=datetime.now(timezone.utc).isoformat(),state='registered_not_executed',
    hypothesis='The earlier full-frame matcher allocated only about97-112 by130-159 coarse samples to each head rectangle. A native integer crop changes local context and effective feature resolution together. Test whether this improves original-pixel three-view consistency at the exact same frozen queries. This is not a pure resolution-only attribution.',
    frames=frames,pairs=[[644,671],[644,698],[671,698]],crops=crops,source_sha256=sources,
    queries_sha256=sha(D/'queries.json'),review_allocation_sha256=sha(D/'review-allocation.json'),
    model={**old['model'],'coarse_resolution_hw':[448,448],'upsample_resolution_hw':[768,768]},
    weight_root=old['weight_root'],weights=old['weights'],frozen_gate=old['frozen_gate'],
    budget=dict(pairs=3,GPU_attempts=1,wall_seconds=420,new_features=0,camera_fits=0,training=0),
    comparison='All600 frozen queries per frame. Baseline full-frame maps already saved. Original integer pixel centers, same bidirectional confidence/FB/cycle gates and8px family-conflict radius. Crop translation only; no training image change.',
    lineage='Every existing FIT/HELD/quarantine label remains fixed. Predicted opposite-split/quarantine proximity disqualifies use, but raw results retained. Unassigned queries stay unassigned. No old HELD point enters fitting.',
    acceptance='Report all pair and three-view gate transitions, displacement and the preregistered24 native crop rows. Self-consistency does not certify material identity or measurement accuracy. No camera/SMPL-X/Gaussian admission from this test.',
    stop='One inference only; no ROI/resolution/threshold sweep. Preserve failures and regressions. Native visual review required before any further material proposal.')
save(D/'plan.json',plan)
print(json.dumps(dict(plan=str(D/'plan.json'),queries=sum(map(len,queries.values())),review_queries=len(review)),ensure_ascii=False))

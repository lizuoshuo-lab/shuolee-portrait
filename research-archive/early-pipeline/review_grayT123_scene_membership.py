"""Disambiguate original appearance components from observations retained in BA.

Uses exactly the already selected FIT identities; no new sample based on error.
"""
from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grayT-scene-support-cross-run-review')
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
B=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'BA-membership-review.json').exists()
ar=read(A/'data/inputs.json')['records'];idx=read(O/'index.json');samples=read(O/'selected-samples.json')['samples'];output=[];tiles=[]
for run,root in [('research',A),('company00',B)]:
    rows=read(root/'data/inputs.json')['records'];z=np.load(root/'camera/solver-output.npz');ids=z['track_ids'].tolist()
    for sample in [s for s in samples if s['run']==run and s['split']=='FIT']:
        p=ids.index(sample['track_id']);kept=z['obs_cam'][z['obs_point']==p].tolist();chosen=np.unique(np.linspace(0,len(kept)-1,min(3,len(kept))).astype(int))
        nodes=dict(sample['nodes']);parts=[];sources=[]
        for k in chosen:
            i=kept[k];xy=np.load(root/f'camera/features-{i:03d}.npz')['xy'][nodes[i]]
            scale=np.array(rows[i]['wh'])/np.array(ar[i]['wh']);xy=(xy+.5)/scale-.5;cx,cy=np.rint(xy).astype(int);crop=[int(cx-128),int(cy-128),int(cx+128),int(cy+128)]
            ip=Path('//wsl.localhost/Ubuntu-20.04'+ar[i]['image']);tile=Image.open(ip).convert('RGB').crop(crop);parts.append(tile)
            sources.append(dict(frame=ar[i]['frame'],xy_native=xy.tolist(),crop=crop,used_in_BA=True))
        name=sample['sample']+'_actually_used_BA.png';image=Image.new('RGB',(768,256),'white')
        for k,tile in enumerate(parts):
            d=ImageDraw.Draw(tile);d.ellipse([123,123,133,133],outline=(0,230,255),width=1);image.paste(tile,(k*256,0))
        image.save(O/name);tiles.append(image)
        entry=dict(frame=sample['track_id'],region=run+'_实际进入BA的同一FIT轨迹；青圈定位，无图中文字',file=name,columns=[s['frame'] for s in sources],sources=sources)
        idx['images'].append(entry)
        output.append(dict(run=run,track_id=sample['track_id'],original_frames=[i+1 for i,j in sample['nodes']],BA_frames=[i+1 for i in kept],
            original_tail_in_BA=sample['nodes'][-1][0] in kept,first_mid_last_BA=[kept[k]+1 for k in chosen]))
sheet=Image.new('RGB',(768,len(tiles)*256),'white')
for i,t in enumerate(tiles):sheet.paste(t,(0,256*i))
sheet.save(O/'all3_actually_used_BA_contact.png');save(O/'index.json',idx);save(O/'BA-membership-review.json',dict(records=output,
    conclusion='Original component endpoint mismatches must not be claimed as BA contamination without membership checks. Original endpoints of the tree and road-to-roof components are not in saved BA.'))
print(json.dumps(output,ensure_ascii=False))

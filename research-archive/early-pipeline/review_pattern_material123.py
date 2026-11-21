"""Generate the preregistered original-pixel material review, without geometry."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/pattern-preflight/native-material';V=O/'visual-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not V.exists(),'Review crops already exist; inspect instead of regenerating.'
result=read(O/'result.json');assert result['state']=='complete_physical_material_review_pending'
plan=read(O/'review-plan.json');manifest=read(O/'feature-manifest.json')['frames']
tracks=read(O/'appearance-tracks.json')['records']
assert sha(O/'appearance-tracks.json')==result['preserved_files']['tracks_sha256']
assert sha(O/'duplicate-families.json')==result['preserved_files']['families_sha256']
assert plan['before_completed_tracks']
data=[]
for f in manifest:
    p=O/'features'/f['feature_file'];assert sha(p)==f['sha256']
    data.append(np.load(p,allow_pickle=False)['xy'])
def region(slot,point):
    x,y=data[slot][point];head,body=manifest[slot]['regions'][0]['bounds'],manifest[slot]['regions'][1]['bounds']
    return 'head' if head[0]<=x<head[2] and head[1]<=y<head[3] else ('person' if body[0]<=x<body[2] and body[1]<=y<body[3] else 'background')
cache={}
def image(slot):
    if slot not in cache:
        p=Path(manifest[slot]['image']);assert sha(p)==manifest[slot]['source_sha256'];cache[slot]=Image.open(p).convert('RGB')
    return cache[slot]
rows=[];seen=set();V.mkdir()
for slot in plan['source_slots']:
    for r in plan['regions']:
        for split in plan['splits']:
            possible=[]
            for tr in tracks:
                if tr['quarantined'] or tr['split']!=split or tr['family_id'] in seen:continue
                nodes=tr['nodes'];hits=[j for i,j in nodes if i==slot]
                if len(hits)==1 and region(slot,hits[0])==r:possible.append(tr)
            possible.sort(key=lambda tr:(tr['family_id'],tr['id']))
            row=dict(row=len(rows),selection_slot=slot,ROI_region=r,split=split,physical_label='pending',physical_subpixel_certified=False)
            if not possible:row.update(status='no_available_track');rows.append(row);continue
            tr=possible[0];seen.add(tr['family_id']);nodes=tr['nodes'];indices=[0,len(nodes)-1,next(i for i,n in enumerate(nodes) if n[0]==slot)]
            if len(set(indices))<3:indices.append(len(nodes)//2)
            indices=sorted(set(indices));assert len(indices)==3
            views=[];strip=Image.new('RGB',(576,192))
            for col,ii in enumerate(indices):
                f,j=nodes[ii];uv=data[f][j];left,top=np.floor(uv).astype(int)-96;box=[int(left),int(top),int(left+192),int(top+192)]
                strip.paste(image(f).crop(box),(col*192,0))
                views.append(dict(slot=f,source_frame=manifest[f]['source_frame'],keypoint=j,uv=uv.tolist(),crop_xyxy=box,source_sha256=manifest[f]['source_sha256']))
            name=f'row_{row["row"]:02d}_native.png';strip.save(V/name)
            row.update(status='ready',track_id=tr['id'],family_id=tr['family_id'],track_observation_count=len(nodes),track_nodes=nodes,views=views,image=name)
            rows.append(row)
for start in range(0,len(rows),5):
    subset=rows[start:start+5];sheet=Image.new('RGB',(576,192*len(subset)),(32,32,32))
    for rr,row in enumerate(subset):
        row['sheet']=f'materials_{start//5+1:02d}.png';row['sheet_row']=rr
        if row['status']=='ready':sheet.paste(Image.open(V/row['image']),(0,rr*192))
    sheet.save(V/subset[0]['sheet'])
save(V/'index.json',dict(updated=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(O/'review-plan.json'),
    all_queries=len(rows),available=sum(r['status']=='ready' for r in rows),rows=rows,crop_native_wh=[192,192],
    no_resize=True,no_text_on_images=True,source_observations_unchanged=True,camera_used=False,physical_face_HELD=0))
print(json.dumps(dict(rows=len(rows),available=sum(r['status']=='ready' for r in rows),path=str(V/'index.json')),ensure_ascii=False))

"""Prepare unchanged native crops for review; never modifies observations or candidate."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import match_video10_temporal_bridge123 as base
G=base.R/'video10-global-initializer';O=G/'joint-track-candidate';D=O/'visual-review';D.mkdir(exist_ok=True)
read=base.read;save=base.save
rows=read(O/'heldout-predictions.json');held=read(base.R/'video10-remaining-material-review/global-input-preflight/heldout-observations.json')
tracks={t['family']:t for t in held['tracks']};ranked=sorted(rows,key=lambda r:r.get('error_px',-1),reverse=True)
chosen=[]
for row in ranked:
    if row['family'] not in [r['family'] for r in chosen]:chosen.append(row)
    if len(chosen)==3:break
median=sorted(rows,key=lambda r:r['error_px'])[len(rows)//2]
if median['family'] not in [r['family'] for r in chosen]:chosen.append(median)
index=[]
for row in chosen:
    tr=tracks[row['family']];obs=tr['observations'];target=row['frame']
    others=sorted((o for o in obs if o['frame']!=target),key=lambda o:abs(o['frame']-target))[:2]
    observations=sorted([next(o for o in obs if o['frame']==target)]+others,key=lambda o:o['frame'])
    canvas=Image.new('RGB',(384*len(observations),384),(235,235,235));sources=[]
    for j,o in enumerate(observations):
        p=base.imgpath(o['frame']);im=Image.open(p).convert('RGB');x,y=np.rint(o['uv']).astype(int)
        box=(x-192,y-192,x+192,y+192);canvas.paste(im.crop(box),(384*j,0))
        sources.append(dict(frame=o['frame'],uv=o['uv'],path=str(p),crop=list(map(int,box)),sha256=base.sha(p)))
    fn=f'heldout_{row["family"]}_frames_'+ '_'.join(str(o['frame']) for o in observations)+'.png';canvas.save(D/fn)
    index.append(dict(family=row['family'],excluded_frame=target,error_px=row['error_px'],file=fn,sources=sources))
metrics=dict(total=len(rows),valid=sum(r['state']=='valid' for r in rows),strictly_below1px=sum(r['state']=='valid' and r['error_px']<1 for r in rows),at_least1px=sum(r.get('error_px',0)>=1 for r in rows),
    physical_face_HELD=0,selection='Three largest residuals with distinct families and the median residual. Explanatory posthoc inspection, no metric/data exclusion.',
    no_resize=True,no_text_added=True,observations_unchanged=True,candidate_sha256=base.sha(O/'diagnostic-candidate.npz'),rows=index)
save(D/'index.json',metrics);print(json.dumps(metrics,ensure_ascii=False))

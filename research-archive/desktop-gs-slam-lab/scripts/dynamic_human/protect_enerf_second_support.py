"""Conservative support clipping with frozen original hand/object and foot targets."""
from pathlib import Path
import json,hashlib
import numpy as np
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-support-correction-081';D=O/'accepted-input';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
poses={(r['camera'],r['frame']):r for r in read(S/'pose/observations.json')['records']};rows=[]
for f in sorted((O/'candidates').glob('*.npz')):
 m=read(f.with_suffix('.json'));c,t=m['camera'],m['frame'];r=poses[c,t];z=np.load(S/'native-input'/f.name);candidate=np.load(f);support=candidate['support'].copy();protect=np.zeros_like(support);boxes=[]
 def box(bb,label):
  x0,y0,x1,y1=np.rint(bb).astype(int);x0=max(0,x0);y0=max(0,y0);x1=min(1920,x1);y1=min(1080,y1);protect[y0:y1,x0:x1]=True;boxes.append(dict(label=label,bbox=[int(x0),int(y0),int(x1),int(y1)]))
 for d in r['detections']:
  k=np.array(d['keypoints']);scores=np.array(d['scores'])
  if scores[0]>.3:
   x,y=k[0,:2];box([x-90,y-90,x+90,y+90],'original_head_and_neck_support')
  for i in [9,10]:
   if scores[i]>.3:
    x,y=k[i,:2];box([x-155,y-115,x+155,y+165],'original_hand_and_carried_object_support')
  x0,y0,x1,y1=d['box'];box([x0-35,y1-220,x1+35,y1+70],'original_lower_leg_and_foot_support')
 if c in ['08','09','10','11','14','16']:protect[:]=True
 support|=protect;matte=z['matte']*support;removed=(z['matte']>.5)&~support
 np.savez_compressed(D/f.name,matte=matte,support=support,protected=protect)
 rows.append(dict(camera=c,frame=t,input_sha256=m['input_sha256'],draft_sha256=sha(f),training_support_sha256=sha(D/f.name),protected_regions=boxes,removed_foreground_pixels=int(removed.sum()),original_foreground_pixels=int((z['matte']>.5).sum()),visual_review='pending'))
save(O/'support-input-audit.json',dict(status='protected_candidates_visual_review_pending',records=rows,scope='Remove only released foreground outside SAM2 support AND protected hand/object/lower-leg windows. Original RGB, original matte inside protected windows, crop and heldout targets unchanged. Conservative: nearby background inside windows remains.',goal_complete=False));print(len(rows),sum(r['removed_foreground_pixels'] for r in rows))

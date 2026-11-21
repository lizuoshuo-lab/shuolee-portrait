"""Robust training-only wholebody landmarks. Predictions are not ground truth."""
from pathlib import Path
import itertools,json,hashlib,shutil
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-wholebody-059';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
obs=json.loads((OUT/'observations.json').read_text());types=['person_dark_clothing','person_light_clothing']
names=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_bigtoe','left_smalltoe','left_heel','right_bigtoe','right_smalltoe','right_heel']
indices=list(range(23))+[95,99,103,107,111,116,120,124,128,132]
names += [side+'_'+finger+'_tip' for side in ['left','right'] for finger in ['thumb','index','middle','ring','pinky']]
assert not (OUT/'triangulation.json').exists()
save(OUT/'triangulation-protocol.json',dict(status='fixed_before_triangulation',minimum_score=.4,max_reprojection_px=8,min_inliers=4,min_max_ray_angle_degrees=2,
 method='All-pair DLT, positive-depth reprojection consensus ranked by count then median; softL1 refinement f_scale3; three rounds of support recomputation.',
 observations='Separate fixed person boxes. No manual coordinate edits or heldout pixels. Fingertips diagnostic only; SMPL lacks individual finger articulation.',
 limitations='Back-facing face landmarks and occluded limbs may be hallucinations even with multiview consensus; acceptance is geometric, not semantic visibility certification.',goal_complete=False))
def project(x,o):
 c=o['W'][:3,:3]@x+o['W'][:3,3];q=o['K']@c;return q[:2]/q[2],c[2]
def errors(x,os):
 return np.array([np.linalg.norm(q-o['xy']) if z>0 else 1e9 for o in os for q,z in [project(x,o)]])
points=np.full((2,len(indices),3),np.nan);rows=[]
for pi,typ in enumerate(types):
 for ji,(idx,name) in enumerate(zip(indices,names)):
  os=[]
  for r in obs['records']:
   d=next(d for d in r['detections'] if d['person']==typ);s=d['scores'][idx]
   if s<.4:continue
   K=np.array(r['K']);W=np.array(r['w2c']);os.append(dict(camera=r['camera'],K=K,W=W,P=K@W[:3],xy=np.array(d['keypoints'][idx]),score=s))
  best=None
  for a,b in itertools.combinations(os,2):
   A=np.stack([xy*p[2]-p[j] for o in [a,b] for p in [o['P']] for j,xy in enumerate(o['xy'])]);_,_,v=np.linalg.svd(A)
   if abs(v[-1,3])<1e-12:continue
   x=v[-1,:3]/v[-1,3];e=errors(x,os);ins=e<8;key=(int(ins.sum()),-float(np.median(e[ins]))) if ins.any() else (0,-1e9)
   if best is None or key>best[0]:best=(key,x,ins)
  row=dict(person=typ,joint=name,index=idx,accepted=False,eligible_views=len(os))
  if best is None or best[0][0]<4:row['reason']='fewer_than4consistent_views';rows.append(row);continue
  _,x,ins=best
  for _ in range(3):
   x=least_squares(lambda x:np.concatenate([project(x,o)[0]-o['xy'] for o,b in zip(os,ins) if b]),x,loss='soft_l1',f_scale=3,max_nfev=100).x
   e=errors(x,os);newins=e<8
   if np.array_equal(newins,ins) or newins.sum()<4:ins=newins;break
   ins=newins
  dirs=[]
  for o,b in zip(os,ins):
   if b:
    center=-o['W'][:3,:3].T@o['W'][:3,3];v=x-center;dirs.append(v/np.linalg.norm(v))
  angle=max([float(np.degrees(np.arccos(np.clip(a@b,-1,1)))) for a,b in itertools.combinations(dirs,2)],default=0)
  row.update(accepted=bool(ins.sum()>=4 and angle>2),point=x.tolist(),inlier_count=int(ins.sum()),max_ray_angle_deg=angle,median_inlier_reprojection_px=float(np.median(e[ins])) if ins.any() else None,
   observations=[dict(camera=o['camera'],xy=o['xy'].tolist(),confidence=o['score'],error_px=float(err),inlier=bool(b)) for o,err,b in zip(os,e,ins)])
  if row['accepted']:points[pi,ji]=x
  rows.append(row)
np.savez_compressed(OUT/'triangulated-wholebody.npz',points=points,valid=np.isfinite(points).all(-1),joint_names=np.array(names),keypoint_indices=np.array(indices),persons=np.array(types))
for c in ['01','04','07','08','13','17']:
 z=np.load(BASE/f'train-camera-{c}.npz');im=Image.fromarray(z['rgb']);d=ImageDraw.Draw(im);o=dict(K=z['K'],W=z['w2c'])
 for row in rows:
  if not row['accepted'] or row['index']>=23:continue
  q,depth=project(np.array(row['point']),o)
  if depth<=0:continue
  x,y=q;col=(255,100,20) if row['person']==types[0] else (0,180,255)
  d.line([(x-4,y-4),(x+4,y+4)],fill=col,width=2);d.line([(x-4,y+4),(x+4,y-4)],fill=col,width=2);d.text((x+4,y-9),str(row['index']),fill=col)
  for ob in row['observations']:
   if ob['camera']==c:
    a,b=ob['xy'];d.ellipse([a-2,b-2,a+2,b+2],outline=col,width=1);d.line([(a,b),(x,y)],fill=(255,0,200),width=1)
 yy,xx=np.where(z['matte']>.5);im.crop((max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(1920,int(xx.max())+26),min(1080,int(yy.max())+26))).save(OUT/f'camera{c}-triangulated-projection.png')
save(OUT/'triangulation.json',dict(status='robust_predictions_require_semantic_projection_review',accepted=sum(r['accepted'] for r in rows),total=len(rows),records=rows,observations_sha256=hashlib.sha256((OUT/'observations.json').read_bytes()).hexdigest(),goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps([dict(person=r['person'],joint=r['joint'],accepted=r['accepted'],n=r.get('inlier_count'),median=r.get('median_inlier_reprojection_px')) for r in rows]))

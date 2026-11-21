from pathlib import Path
import json,itertools,hashlib,shutil
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-body-observations-057';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
assert not (OUT/'triangulation.json').exists()
obs=read(OUT/'observations.json');types=['person_dark_clothing','person_light_clothing'];names=obs['records'][0]['body_joint_names']
save(OUT/'triangulation-protocol.json',dict(status='fixed_before_triangulation',minimum_joint_logit=3.,max_reprojection_px=8.,min_inliers=4,min_max_ray_angle_degrees=2.,
 candidate='All distinct camera pairs, linear DLT, rank by inlier count then median error; refine with softL1 f_scale3 on inliers, recompute support',
 visual_review='All15 detector overlays reviewed. Dark person13 is a false face-only detection; scores below3 automatically exclude it. Occluded dark shoulders/arms in07/08/14/16/17 can be inaccurate. No manual coordinate edits.',
 limits='COCO predicted body joints, not surface points or GT. No left/right remapping; no heldout pixels. Geometry is a fitting hypothesis, requires projection review.',goal_complete=False))
cams={r['camera']:r for r in obs['records']};rows=[];points=np.full((2,12,3),np.nan)
def xyz_dlt(os):
 A=[]
 for o in os:
  p=o['P'];u,v=o['xy'];A.extend([u*p[2]-p[0],v*p[2]-p[1]])
 _,_,vt=np.linalg.svd(np.stack(A));x=vt[-1];return x[:3]/x[3]
def project(x,o):
 c=o['w2c'][:3,:3]@x+o['w2c'][:3,3];q=o['K']@c;return q[:2]/q[2],c[2]
for pi,typ in enumerate(types):
 for ji,name in enumerate(names):
  os=[]
  for r in obs['records']:
   aa=[a for a in r['associations'] if a['person']==typ]
   if not aa:continue
   d=r['candidates'][aa[0]['candidate']];confidence=d['body_confidence'][ji]
   if confidence<3:continue
   K=np.array(r['K']);W=np.array(r['w2c']);xy=np.array(d['body_xy'][ji]);P=K@W[:3]
   os.append(dict(camera=r['camera'],xy=xy,K=K,w2c=W,P=P,confidence=confidence))
  best=None
  for ia,ib in itertools.combinations(range(len(os)),2):
   x=xyz_dlt([os[ia],os[ib]])
   if not np.isfinite(x).all():continue
   projections=[project(x,o) for o in os];errors=np.array([np.linalg.norm(q-o['xy']) if z>0 else 1e9 for (q,z),o in zip(projections,os)])
   ins=errors<8.;n=int(ins.sum());key=(n,-float(np.median(errors[ins]))) if n else (0,-1e9)
   if best is None or key>best[0]:best=(key,x,ins)
  if best is None or best[0][0]<4:
   rows.append(dict(person=typ,joint=name,accepted=False,eligible_views=len(os),reason='fewer_than4consistent_views'));continue
  _,x,ins=best
  def residual(x):return np.concatenate([project(x,o)[0]-o['xy'] for o,b in zip(os,ins) if b])
  x=least_squares(residual,x,loss='soft_l1',f_scale=3,max_nfev=100).x
  errs=np.array([np.linalg.norm(project(x,o)[0]-o['xy']) if project(x,o)[1]>0 else 1e9 for o in os]);ins=errs<8.
  dirs=[]
  for o,b in zip(os,ins):
   if b:
    center=-o['w2c'][:3,:3].T@o['w2c'][:3,3];v=x-center;dirs.append(v/np.linalg.norm(v))
  angles=[float(np.degrees(np.arccos(np.clip(a@b,-1,1)))) for a,b in itertools.combinations(dirs,2)]
  accepted=bool(ins.sum()>=4 and max(angles,default=0)>2)
  if accepted:points[pi,ji]=x
  rows.append(dict(person=typ,joint=name,accepted=accepted,point=x.tolist(),eligible_views=len(os),inlier_count=int(ins.sum()),max_ray_angle_deg=max(angles,default=0),
   median_inlier_reprojection_px=float(np.median(errs[ins])) if ins.any() else None,
   observations=[dict(camera=o['camera'],xy=o['xy'].tolist(),confidence=o['confidence'],error_px=float(e),inlier=bool(b)) for o,e,b in zip(os,errs,ins)]))
np.savez_compressed(OUT/'triangulated-body-joints.npz',points=points,valid=np.isfinite(points).all(-1),joint_names=np.array(names),persons=np.array(types))
for c in ['01','07','08','11','17']:
 z=np.load(BASE/f'train-camera-{c}.npz');im=Image.fromarray(z['rgb']);d=ImageDraw.Draw(im);K=z['K'];W=z['w2c'];o=dict(K=K,w2c=W)
 for row in rows:
  if not row.get('accepted'):continue
  q,depth=project(np.array(row['point']),o)
  if depth<=0:continue
  x,y=q;color=(255,100,20) if row['person']==types[0] else (0,180,255)
  d.line([(x-5,y-5),(x+5,y+5)],fill=color,width=2);d.line([(x-5,y+5),(x+5,y-5)],fill=color,width=2)
  d.text((x+5,y-10),str(names.index(row['joint'])+5),fill=color)
  for ob in row['observations']:
   if ob['camera']==c:
    a,b=ob['xy'];d.ellipse([a-3,b-3,a+3,b+3],outline=color,width=2);d.line([(a,b),(x,y)],fill=(255,0,200),width=1)
 yy,xx=np.where(z['matte']>.5);box=(max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(1920,int(xx.max())+31),min(1080,int(yy.max())+31))
 im.crop(box).save(OUT/f'camera{c}-triangulated-projection.png')
report=dict(status='robust_multiview_body_hypotheses_projection_review_pending',accepted_joints=sum(r['accepted'] for r in rows),records=rows,
 observation_sha256=hashlib.sha256((OUT/'observations.json').read_bytes()).hexdigest(),goal_complete=False)
save(OUT/'triangulation.json',report);shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(dict(accepted=report['accepted_joints'],joints=[(r['person'],r['joint'],r.get('inlier_count'),r.get('median_inlier_reprojection_px')) for r in rows])))

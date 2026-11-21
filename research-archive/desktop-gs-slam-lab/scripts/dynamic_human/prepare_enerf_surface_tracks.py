"""Multi-view triangulation from existing train-only matching observations."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
LAB=Path(__file__).resolve().parents[2];SRC=LAB/'outputs/dynamic-human-enerf-foreground-geometry-045'
BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
OUT=LAB/'outputs/dynamic-human-enerf-surface-tracks-054';OUT.mkdir(exist_ok=True)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (OUT/'tracks.npz').exists()
save(OUT/'protocol.json',dict(status='fixed_before_depth_comparison',
    matching='Reuse045mutualSIFT0.7fixed observations. Pair links: positive depths,absSampson<1px,maxpairreprojection<1px.',
    tracks='Connected feature identity graph; require3distinctcameras, reject any component containing duplicatecameraobservations.',
    triangulation='Homogeneous DLT of all observations, unchangedK/R/T. Require positiveall, maxreprojection<1nativepixel, maxtriangulationangle>2degrees.',
    diagnostic='Compare triangulatedcameraZ with Gaussian expecteddepth at correspondingnativepixels, only alpha>0.9. No depth is used in training in054.',
    limits='Sparse texture-biased geometric hypotheses, not depth groundtruth. Matching mistakes, calibration/timing errors and foregroundmask mistakes may survive. Worldunits not metricallycertified.',
    target_frame=750,heldout_views_accessed=False,goal_complete=False))
parent={};coords={}
def key(c,xy):return (c,float(xy[0]),float(xy[1]))
def find(k):
 parent.setdefault(k,k)
 if parent[k]!=k:parent[k]=find(parent[k])
 return parent[k]
def join(a,b):parent[find(a)]=find(b)
files=[];links=0
for f in sorted(SRC.glob('foreground-*.npz')):
 _,a,b=f.stem.split('-');z=np.load(f);good=z['positive']&(np.abs(z['signed_sampson'])<1)&(z['reprojection'].max(1)<1)
 for x,y in zip(z['xy_a'][good],z['xy_b'][good]):join(key(a,x),key(b,y));links+=1
 files.append(dict(name=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),accepted_links=int(good.sum())))
groups={}
for k in parent:groups.setdefault(find(k),[]).append(k)
cams={c:np.load(BASE/f'train-camera-{c}.npz') for c in sorted({k[0] for k in parent})}
Ps={c:z['K']@z['w2c'][:3] for c,z in cams.items()}
centers={c:-z['w2c'][:3,:3].T@z['w2c'][:3,3] for c,z in cams.items()}
tracks=[];points=[];allobs=[];ids=[];errors=[];angles=[];rejected={'under3views':0,'duplicate_camera':0,'geometry':0}
for members in groups.values():
 if len(members)<3:rejected['under3views']+=1;continue
 if len({m[0] for m in members})<len(members):rejected['duplicate_camera']+=1;continue
 A=[]
 for c,x,y in members:
  P=Ps[c];A.extend([x*P[2]-P[0],y*P[2]-P[1]])
 _,_,vh=np.linalg.svd(np.array(A));X=vh[-1,:3]/vh[-1,3]
 es=[];depth=[];rays=[]
 for c,x,y in members:
  h=Ps[c]@np.r_[X,1];depth.append(h[2]);es.append(np.linalg.norm(h[:2]/h[2]-[x,y]))
  v=X-centers[c];rays.append(v/np.linalg.norm(v))
 angle=float(np.rad2deg(np.arccos(np.clip(np.min(np.array(rays)@np.array(rays).T),-1,1))))
 if min(depth)<=0 or max(es)>=1 or angle<=2 or not np.isfinite(X).all():rejected['geometry']+=1;continue
 idx=len(points);points.append(X);angles.append(angle)
 for m,e in zip(members,es):allobs.append(m);ids.append(idx);errors.append(e)
 tracks.append(dict(id=idx,cameras=[m[0] for m in members],max_reprojection_px=max(es),max_angle_degrees=angle))
assert len(points)>20,(len(points),rejected)
np.savez_compressed(OUT/'tracks.npz',points=np.array(points),camera=np.array([r[0] for r in allobs]),xy=np.array([r[1:] for r in allobs]),
 track_id=np.array(ids),reprojection=np.array(errors),angle=np.array(angles))
save(OUT/'preparation.json',dict(status='training_only_multiview_tracks_triangulated_depth_comparison_pending',pair_links=links,
 tracks=len(points),observations=len(ids),rejected=rejected,records=tracks,input_files=files,goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(dict(tracks=len(points),observations=len(ids),rejected=rejected,max_reprojection_px=max(errors))))

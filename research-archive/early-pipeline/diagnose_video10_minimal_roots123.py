"""One post-hoc algebraic ambiguity diagnostic; no model chosen using HELD."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib
import numpy as np
BASE='Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122'
R=Path('C:/'+BASE) if os.name=='nt' else Path('/mnt/c/'+BASE)
G=R/'video10-global-initializer';U=R/'video10-remaining-material-review/global-input-preflight';D=G/'minimal-root-diagnosis'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
inputs=[U/'fit-observations.json',U/'heldout-observations.json',U/'conditional-camera.json',G/'relative-graph/repair-invalid-pose/pairs.json']
if sys.argv[1:]==['--register']:
    D.mkdir(exist_ok=True);assert not (D/'plan.json').exists()
    save(D/'plan.json',dict(registered=now(),hypothesis='A calibrated five-correspondence pair can admit multiple essential roots with near-zero FIT residual but incompatible camera rotations and HELD predictions.',
        pair=[698,752],selection='Post-hoc explanatory case: exactly5FIT and largest observed HELD E residual. Not an independent performance estimate or camera selection.',
        source_sha256={p.name:sha(p) for p in inputs},method='One deterministic PoseLib relpose_5pt root enumeration, fixedK. Inspect all returned real roots, positive depth and every frozen HELD error. No root selection,threshold sweep,new features or BA.',
        budget=dict(pairs=1,CPU_seconds=30),training_started=False))
    print(str(D/'plan.json'));sys.exit(0)
assert sys.argv[1:]==[] and not (D/'result.json').exists()
plan=read(D/'plan.json')
for p in inputs:assert sha(p)==plan['source_sha256'][p.name]
sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib,cv2
from global_track_initializer123 import angle_between_rotations
K=np.asarray(read(U/'conditional-camera.json')['K_integer_centers']);inv=np.linalg.inv(K)
a,b=plan['pair'];pair=next(r for r in read(inputs[-1]) if r['a']==a and r['b']==b)
def observations(data,ids):
    tracks={t['family']:{o['frame']:o['uv'] for o in t['observations']} for t in data['tracks']}
    return np.array([tracks[k][a] for k in ids],float),np.array([tracks[k][b] for k in ids],float)
aa,bb=observations(read(inputs[0]),pair['families']);ha,hb=observations(read(inputs[1]),pair['heldout_families']);assert len(aa)==5
da=np.c_[aa,np.ones(5)]@inv.T;db=np.c_[bb,np.ones(5)]@inv.T
da/=np.linalg.norm(da,axis=1)[:,None];db/=np.linalg.norm(db,axis=1)[:,None]
poses=poselib.relpose_5pt(da,db);rows=[]
def errors(u,v,F):
    x=np.c_[u,np.ones(len(u))];y=np.c_[v,np.ones(len(v))];f=x@F.T;ft=y@F
    return np.abs(np.sum(y*f,axis=1))/np.sqrt(np.maximum(np.sum(f[:,:2]**2+ft[:,:2]**2,axis=1),1e-30))
for i,p in enumerate(poses):
    Q=np.array(p.R,copy=True);t=np.array(p.t,copy=True).reshape(3);t/=np.linalg.norm(t)
    E=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])@Q;F=inv.T@E@inv
    n1=(np.c_[aa,np.ones(5)]@inv.T)[:,:2];n2=(np.c_[bb,np.ones(5)]@inv.T)[:,:2]
    h=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[Q,t],n1.T.copy(),n2.T.copy()).T
    X=h[:,:3]/h[:,3,None];positive=(X[:,2]>0)&((X@Q.T+t)[:,2]>0)
    fe=errors(aa,bb,F);he=errors(ha,hb,F)
    rows.append(dict(root=i,R=Q.tolist(),t=t.tolist(),fit_sampson_max_px=float(fe.max()),fit_sampson_px=fe.tolist(),positive_depth_count=int(positive.sum()),
        heldout_families=pair['heldout_families'],heldout_sampson_px=he.tolist(),rotation_from_RANSAC_deg=float(np.degrees(angle_between_rotations(Q,np.asarray(pair['R_b_from_a']))))))
result=dict(updated=now(),state='minimal_roots_enumerated',pair=plan['pair'],fit_families=pair['families'],roots=rows,root_count=len(rows),
    positive_all_roots=sum(r['positive_depth_count']==5 for r in rows),plan_sha256=sha(D/'plan.json'),
    selection='No root chosen. HELD values explanatory only; original split/data unchanged.',real_global_cameras=0,training_started=False,quality_accepted=False)
save(D/'result.json',result)
print(json.dumps({**{k:v for k,v in result.items() if k!='roots'},'roots':[{k:v for k,v in r.items() if k not in ['R','t','fit_sampson_px']} for r in rows]},ensure_ascii=False))

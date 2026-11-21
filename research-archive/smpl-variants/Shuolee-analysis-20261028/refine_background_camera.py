"""Controlled camera refinement using existing background tracks only."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import hashlib,json,time,shutil
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
OUT=Path('/mnt/d/Projects/Shuolee-analysis-20261028/background-intrinsics-control')
OUT.mkdir(exist_ok=True)
src=ROOT/'camera/solver-output.npz'
digest=hashlib.sha256(src.read_bytes()).hexdigest()
data=dict(np.load(src))
E0=data['w2c'];K0=data['K'];X0=data['points'];ci=data['obs_cam'];pi=data['obs_point'];uv=data['observations']
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records']
wh=np.asarray(rows[0]['wh'])
seed=json.loads((ROOT/'camera/seed.json').read_text())['pair']
cp=np.c_[Rotation.from_matrix(E0[:,:3,:3]).as_rotvec(),E0[:,:3,3]]
free=np.ones_like(cp,dtype=bool);free[seed[0]]=False
free[seed[1],3+np.argmax(abs(cp[seed[1],3:]))]=False
cols=np.full(cp.shape,-1,int);cols[free]=np.arange(free.sum());nc=int(free.sum())
initial=np.r_[cp[free],X0.ravel(),0.,0.,0.]
S=lil_matrix((2*len(uv),len(initial)),dtype=np.int8);r=np.arange(len(uv))
for j in range(6):
    good=cols[ci,j]>=0;S[2*r[good],cols[ci[good],j]]=1;S[2*r[good]+1,cols[ci[good],j]]=1
for j in range(3):
    S[2*r,nc+3*pi+j]=1;S[2*r+1,nc+3*pi+j]=1
S[:,-3:]=1
lo=np.full(len(initial),-np.inf);hi=np.full(len(initial),np.inf)
lo[-3:]=[np.log(.8),-.05,-.05];hi[-3:]=[np.log(1.2),.05,.05]
started=time.monotonic();calls=0
def unpack(x):
    p=cp.copy();p[free]=x[:nc]
    E=np.tile(np.eye(4),(len(p),1,1));E[:,:3,:3]=Rotation.from_rotvec(p[:,:3]).as_matrix();E[:,:3,3]=p[:,3:]
    K=K0.copy();K[0,0]*=np.exp(x[-3]);K[1,1]*=np.exp(x[-3]);K[:2,2]+=x[-2:]*wh
    return E,K,x[nc:-3].reshape(-1,3)
def residual(x):
    global calls
    calls+=1
    if time.monotonic()-started>600:raise RuntimeError('Declared 600 second camera control budget exceeded')
    E,K,X=unpack(x)
    cam=np.einsum('nij,nj->ni',E[ci,:3,:3],X[pi])+E[ci,:3,3]
    q=cam@K.T
    result=q[:,:2]/np.maximum(q[:,2:],1e-6)-uv
    if calls%200==0:print(json.dumps(dict(residual_calls=calls,seconds=time.monotonic()-started,focal_factor=float(np.exp(x[-3])),principal_delta=(x[-2:]*wh).tolist())),flush=True)
    return result.ravel()
def stats(e):
    x=np.asarray(e);return dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(np.max(x)),over_1px=int((x>1).sum()))
before=stats(np.linalg.norm(residual(initial).reshape(-1,2),axis=1))
start_parameters=initial.copy()
continuing=(OUT/'candidate.npz').exists()
if continuing:
    previous=dict(np.load(OUT/'candidate.npz'))
    pc=np.c_[Rotation.from_matrix(previous['w2c'][:,:3,:3]).as_rotvec(),previous['w2c'][:,:3,3]]
    start_parameters=np.r_[pc[free],previous['points'].ravel(),np.log(previous['K'][0,0]/K0[0,0]),
                           (previous['K'][:2,2]-K0[:2,2])/wh]
    assert not (OUT/'initial-budget-report.json').exists()
    shutil.copyfile(OUT/'report.json',OUT/'initial-budget-report.json')
sol=least_squares(residual,start_parameters,bounds=(lo,hi),jac_sparsity=S.tocsr(),method='trf',tr_solver='lsmr',
                  loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=300,ftol=1e-6)
E,K,X=unpack(sol.x)
np.savez_compressed(OUT/'candidate.npz',w2c=E,K=K,points=X,obs_cam=ci,obs_point=pi,observations=uv)
held=json.loads((ROOT/'camera/held-evaluation.json').read_text())['records']
xy=[np.load(ROOT/f'camera/features-{i:03d}.npz')['xy'] for i in range(len(rows))]
def evaluate(matrices,intrinsics):
    result=[]
    for tr in held:
        nodes=tr['nodes'];anchors=nodes[::2];checks=nodes[1::2]
        A=[]
        for i,j in anchors:
            P=intrinsics@matrices[i,:3];u,v=xy[i][j];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
        _,_,vh=np.linalg.svd(A);point=vh[-1,:3]/vh[-1,3]
        errors=[];positive=True
        for i,j in checks:
            cam=matrices[i,:3,:3]@point+matrices[i,:3,3];q=intrinsics@cam
            errors.append(float(np.linalg.norm(q[:2]/q[2]-xy[i][j])));positive &= bool(cam[2]>0)
        result.append(dict(id=tr['id'],errors=errors,positive_depth=positive))
    return dict(summary=stats([e for row in result for e in row['errors']]),records=result)
old=evaluate(E0,K0);new=evaluate(E,K)
report=dict(experiment='Background-only joint camera, point, shared focal and principal-point refinement',
            input_sha256=digest,frames=len(rows),observations=len(uv),points=len(X0),
            original_K=K0.tolist(),candidate_K=K.tolist(),focal_factor=float(np.exp(sol.x[-3])),principal_delta_px=(sol.x[-2:]*wh).tolist(),
            train_before=before,train_after=stats(np.linalg.norm(residual(sol.x).reshape(-1,2),axis=1)),
            held_before=old['summary'],held_after=new['summary'],held_track_count=len(held),
            optimizer_success=bool(sol.success),message=sol.message,nfev=sol.nfev,continued_previous_budget=continuing,seconds=time.monotonic()-started,
            bound_active=bool(np.any(sol.active_mask)),human_observations_used=False,original_source_changed=hashlib.sha256(src.read_bytes()).hexdigest()!=digest,
            fitting_or_training_started=False,quality_accepted=False,
            limitation='Exploratory refinement; appearance-track correspondence is not independently measured geometry. Holdout used for comparison, not optimization.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'held-comparison.json').write_text(json.dumps(dict(before=old,after=new),indent=2)+'\n')
print(json.dumps(report),flush=True)

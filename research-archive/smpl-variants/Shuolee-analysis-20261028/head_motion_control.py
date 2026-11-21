"""Fixed-camera head-motion control, evaluated against a separately reserved skin track."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
OUT=Path('/mnt/d/Projects/Shuolee-analysis-20261028/head-motion-control')
OUT.mkdir(exist_ok=False)
fit=dict(np.load(ROOT/'fit/fitted.npz'));obs=dict(np.load(ROOT/'data/observations.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records'];n=len(rows)
E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);scale=float(fit['world_scale'])
template=np.load('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
tri=fit['faces'][template['lmk_faces_idx']]
face=(fit['vertices_world'][0,tri]*template['lmk_bary_coords'][:,:,None]).sum(1)
center=fit['joints_world'][0,15].astype(float)
target=obs['keypoints'][:,40:91].astype(float);w=weights['face'].astype(float);w/=max(float(w.mean()),1e-6)
keep=weights['face']>0
sigma=6*(obs['boxes'][:,3]-obs['boxes'][:,1])/1000
source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'fit/fitted.npz',ROOT/'fit/observation-weights.npz',ROOT/'data/training-inputs.json']}
def transform(x,points):
    p=x.reshape(n,6);R=Rotation.from_rotvec(p[:,:3]).as_matrix()
    return np.einsum('nij,pj->npi',R,points-center)+center+p[:,None,3:]*scale
def project(world):
    cam=np.einsum('nij,npj->npi',E[:,:3,:3],world)+E[:,None,:3,3]
    q=np.einsum('nij,npj->npi',K,cam)
    return q[...,:2]/np.maximum(q[...,2:],.01)-.5
def residual(x):
    p=x.reshape(n,6);e=(project(transform(x,face))-target)/sigma[:,None,None]
    robust=e*np.sqrt(2/(np.sqrt(1+e*e)+1))
    data=(robust*np.sqrt(w[...,None])).ravel()
    prior=np.concatenate([.3*p[:,:3]/.35,.3*p[:,3:]/.04],axis=1).ravel()
    accel=np.diff(p,n=2,axis=0)
    smooth=np.concatenate([.4*accel[:,:3]/.03,.4*accel[:,3:]/.004],axis=1).ravel()
    return np.r_[data,prior,smooth]
nd=n*51*2;npres=n*6;ns=(n-2)*6
S=lil_matrix((nd+npres+ns,n*6),dtype=np.int8)
for i in range(n):
    S[i*102:(i+1)*102,i*6:(i+1)*6]=1
    S[nd+i*6:nd+(i+1)*6,i*6:(i+1)*6]=1
for i in range(n-2):S[nd+npres+i*6:nd+npres+(i+1)*6,i*6:(i+3)*6]=1
lo=np.tile([-.6,-.6,-.6,-.12,-.12,-.12],n);hi=-lo
started=time.monotonic()
sol=least_squares(residual,np.zeros(n*6),jac_sparsity=S.tocsr(),bounds=(lo,hi),x_scale='jac',max_nfev=160,ftol=1e-6)
projection=project(transform(sol.x,face));baseline=project(transform(np.zeros(n*6),face))
np.savez_compressed(OUT/'candidate.npz',parameters=sol.x.reshape(n,6),center=center,world_scale=scale,projected_face51=projection,frame_ids=fit['frame_ids'])
def stats(x):
    x=np.asarray(x);return dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)))
source=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grayT-facial-articulation-review/cross-run-result.json')
evidence=json.loads(source.read_text());skin=evidence['company00_M0']['observations']
ids=np.array([r['frame_number']-1 for r in skin]);anchors=np.array([r['role']=='ANCHOR' for r in skin]);native_target=np.array([r['xy'] for r in skin])
native_wh=np.array([1520,2704]);decoded_wh=np.array(rows[0]['wh']);target_skin=(native_target+.5)*decoded_wh/native_wh-.5
def skin_evaluate(parameters):
    def uv(point):return project(transform(parameters,point[None]))[:,0][ids]
    def r(point):return (uv(point)[anchors]-target_skin[anchors]).ravel()
    original=np.asarray(evidence['company00_M0']['point'])
    result=least_squares(r,original,loss='huber',f_scale=1.,max_nfev=200)
    native=(uv(result.x)+.5)*native_wh/decoded_wh-.5
    errors=np.linalg.norm(native-native_target,axis=1)
    return dict(anchors=stats(errors[anchors]),held=stats(errors[~anchors]),point=result.x.tolist(),success=bool(result.success),
                observations=[dict(frame_number=int(i+1),role='ANCHOR' if a else 'HELD',target=t.tolist(),projected=p.tolist(),error_px=float(e)) for i,a,t,p,e in zip(ids,anchors,native_target,native,errors)])
old_skin=skin_evaluate(np.zeros(n*6));new_skin=skin_evaluate(sol.x)
report=dict(experiment='Per-frame rigid head motion from semantic face observations, fixed background cameras and fixed body template',
            optimizer_success=bool(sol.success),message=sol.message,nfev=sol.nfev,seconds=time.monotonic()-started,
            before=stats(np.linalg.norm(baseline-target,axis=-1)[keep]),after=stats(np.linalg.norm(projection-target,axis=-1)[keep]),
            head_rotation_degrees=stats(np.linalg.norm(sol.x.reshape(n,6)[:,:3],axis=1)*180/np.pi),
            head_translation_model_m=stats(np.linalg.norm(sol.x.reshape(n,6)[:,3:],axis=1)),
            bound_active=bool(np.any(sol.active_mask)),skin_source=str(source),skin_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            independent_skin_track_before=old_skin,independent_skin_track_after=new_skin,
            skin_track_used_in_head_motion_fit=False,cameras_changed=False,source_hashes=source_hashes,
            original_sources_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()),
            training_started=False,quality_accepted=False,
            limitation='Exploratory motion control. Learned keypoints may be biased; improvement on one reserved skin track is not complete physical calibration or reconstruction quality.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['optimizer_success','message','seconds','before','after','head_rotation_degrees','head_translation_model_m']}),flush=True)
print(json.dumps(dict(skin_before=old_skin['held'],skin_after=new_skin['held'])),flush=True)

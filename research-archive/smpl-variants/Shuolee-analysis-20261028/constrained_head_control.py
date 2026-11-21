"""Compare shared alignment and joint-constrained neck/head motion, with fixed cameras."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

BASE=Path('/mnt/d/Projects/Shuolee-analysis-20261028');ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
parser=argparse.ArgumentParser();parser.add_argument('--detector',choices=['rtmw','fan'],default='rtmw');args=parser.parse_args()
OUT=BASE/('constrained-head-control' if args.detector=='rtmw' else 'constrained-head-fan-control');OUT.mkdir(exist_ok=False)
fit=dict(np.load(ROOT/'fit/fitted.npz'));obs=dict(np.load(ROOT/'data/observations.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records'];n=len(rows)
E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);scale=float(fit['world_scale'])
template=np.load('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
tri=fit['faces'][template['lmk_faces_idx']];face=(fit['vertices_world'][0,tri]*template['lmk_bary_coords'][:,:,None]).sum(1)
center=fit['joints_world'][0,15].astype(float);neck=fit['joints_world'][0,12].astype(float)
target=obs['keypoints'][:,40:91].astype(float);w=weights['face'].astype(float);w/=max(float(w.mean()),1e-6);keep=weights['face']>0
if args.detector=='fan':
    fan=dict(np.load(BASE/'face-local-detector/observations.npz'));assert np.array_equal(fan['frame_ids'],fit['frame_ids'])
    target=fan['decoded_face68'][:,17:68].astype(float)
sigma=6*(obs['boxes'][:,3]-obs['boxes'][:,1])/1000
def world(parameters,points,kind):
    p=parameters.reshape(-1,6)
    if kind=='neck_head':
        rn=Rotation.from_rotvec(p[:,:3]).as_matrix();rh=Rotation.from_rotvec(p[:,3:]).as_matrix()
        head=np.einsum('nij,pj->npi',rh,points-center)+center
        return np.einsum('nij,npj->npi',rn,head-neck)+neck
    R=Rotation.from_rotvec(p[:,:3]).as_matrix()
    return np.einsum('nij,pj->npi',R,points-center)+center+p[:,None,3:]*scale
def project(points,matrices=E,intrinsics=K):
    cam=np.einsum('nij,npj->npi',matrices[:,:3,:3],points)+matrices[:,None,:3,3]
    q=np.einsum('nij,npj->npi',intrinsics,cam)
    return q[...,:2]/np.maximum(q[...,2:],.01)-.5
def residual(x,kind):
    p=x.reshape(-1,6);e=(project(world(x,face,kind))-target)/sigma[:,None,None]
    robust=e*np.sqrt(2/(np.sqrt(1+e*e)+1))
    data=(robust*np.sqrt(w[...,None])).ravel()
    if kind=='shared':return np.r_[data,.3*p[:,:3].ravel()/.35,.3*p[:,3:].ravel()/.04]
    prior=.3*p/.35;accel=.4*np.diff(p,n=2,axis=0)/.03
    return np.r_[data,prior.ravel(),accel.ravel()]
def stats(x):
    x=np.asarray(x);return dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)))
skin_report=json.loads((BASE/'head-motion-control/report.json').read_text())
skin=skin_report['independent_skin_track_before']['observations'];ids=np.array([r['frame_number']-1 for r in skin]);anchors=np.array([r['role']=='ANCHOR' for r in skin])
native_target=np.array([r['target'] for r in skin]);ratio=np.array([1520,2704])/np.array(rows[0]['wh']);target_skin=(native_target+.5)/ratio-.5
def skin_evaluate(x,kind,matrices=E,intrinsics=K):
    def uv(point):return project(world(x,point[None],kind),matrices,intrinsics)[:,0][ids]
    initial=np.array(skin_report['independent_skin_track_before']['point'])
    solved=least_squares(lambda p:(uv(p)[anchors]-target_skin[anchors]).ravel(),initial,loss='huber',f_scale=1.,max_nfev=200)
    errors=np.linalg.norm((uv(solved.x)+.5)*ratio-.5-native_target,axis=1)
    return dict(anchors=stats(errors[anchors]),held=stats(errors[~anchors]),success=bool(solved.success))
summaries={}
for kind in ['shared','neck_head']:
    start=time.monotonic();initial=np.zeros(6 if kind=='shared' else n*6)
    kwargs={}
    if kind=='neck_head':
        nd=n*51*2;npres=n*6;S=lil_matrix((nd+npres+(n-2)*6,n*6),dtype=np.int8)
        for i in range(n):S[i*102:(i+1)*102,i*6:(i+1)*6]=1;S[nd+i*6:nd+(i+1)*6,i*6:(i+1)*6]=1
        for i in range(n-2):S[nd+npres+i*6:nd+npres+(i+1)*6,i*6:(i+3)*6]=1
        kwargs['jac_sparsity']=S.tocsr()
        lo=np.tile([-.35]*3+[-.5]*3,n)
    else:lo=np.array([-.6]*3+[-.12]*3)
    result=least_squares(lambda x:residual(x,kind),initial,bounds=(lo,-lo),x_scale='jac',max_nfev=200,ftol=1e-6,**kwargs)
    projected=project(world(result.x,face,kind));errors=np.linalg.norm(projected-target,axis=-1)
    p=result.x.reshape(-1,6)
    summaries[kind]=dict(success=bool(result.success),message=result.message,seconds=time.monotonic()-start,
                         face=stats(errors[keep]),skin=skin_evaluate(result.x,kind),bound_active=bool(np.any(result.active_mask)),
                         first_rotation_degrees=stats(np.linalg.norm(p[:,:3],axis=1)*180/np.pi),
                         second_component_norm=stats(np.linalg.norm(p[:,3:],axis=1)),parameters=p.tolist())
    np.savez_compressed(OUT/f'{kind}.npz',parameters=p,center=center,neck=neck,world_scale=scale,projected_face51=projected,frame_ids=fit['frame_ids'])
    print(json.dumps({k:v for k,v in summaries[kind].items() if k!='parameters'}),flush=True)
camera=dict(np.load(BASE/'background-intrinsics-control/candidate.npz'));kc=np.tile(camera['K'],(n,1,1));kc[:,:2,2]+=.5
summaries['refined_background_skin']=skin_evaluate(np.zeros(6),'shared',camera['w2c'],kc)
summaries['scope']=dict(cameras_fixed_for_head_controls=True,no_subject_landmarks_in_background_control=True,
                        detector=args.detector,original_weights_unchanged=True,
                        no_training_or_source_modification=True,quality_accepted=False,
                        neck_to_head_distance_model_m=float(np.linalg.norm(center-neck)/scale))
(OUT/'report.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(dict(refined_background_skin=summaries['refined_background_skin'])),flush=True)

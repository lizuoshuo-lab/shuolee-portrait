"""Exact geometric decomposition on immutable 098 world-hit observations."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-shared-face-fresh-098'
O=B/'current/shared-point-ray-consistency-099'
O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[R/'shared-targets/shared-face-points.npz',R/'evaluation/same-feature-world-hits.npz']
hashes={str(p):sha(p) for p in paths}
a=np.load(paths[0]);h=np.load(paths[1]);ids=h['point_ids'];jj=np.array([np.flatnonzero(a['point_ids']==k)[0] for k in ids])
X=h['shared_xyz'];E=a['w2c'];K=a['K'];uv=a['observed_uv'][:,jj]
camray=np.concatenate([(uv+.5-K[:,None,:2,2])/np.stack([K[:,0,0],K[:,1,1]],1)[:,None],np.ones((*uv.shape[:2],1))],axis=-1)
ray=np.einsum('fji,fkj->fki',E[:,:3,:3],camray)
ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
C=-np.einsum('fji,fj->fi',E[:,:3,:3],E[:,:3,3])
along_X=np.sum((X[None]-C[:,None])*ray,axis=-1)
closest=C[:,None]+along_X[...,None]*ray
perp=np.linalg.norm(X[None]-closest,axis=-1)
valid=a['observation_valid'][:,jj].copy()
for arm in ('control','shared'):
    valid &= np.isfinite(h[arm+'_world']).all(-1)&(h[arm+'_alpha']>.8)
def stats(v):
    v=np.asarray(v);return dict(n=v.size,rms=float(np.sqrt(np.mean(v*v))),median=float(np.median(v)),p90=float(np.quantile(v,.9)))
results={};savearrays=dict(ray=ray,closest_target_on_ray=closest,perpendicular=perp,common_valid=valid,point_ids=ids)
for arm in ('control','shared'):
    Y=h[arm+'_world'];along=np.sum((Y-closest)*ray,axis=-1)
    point=np.linalg.norm(Y-X[None],axis=-1)
    ray_residual=np.linalg.norm((Y-closest)-along[...,None]*ray,axis=-1)
    pythagoras=np.abs(point**2-perp**2-along**2)
    assert np.max(ray_residual[valid])<2e-6
    assert np.max(pythagoras[valid])<2e-6
    record=dict(point_error=stats(point[valid]),along_ray_depth_error=stats(np.abs(along[valid])),
        along_ray_signed_bias=float(np.mean(along[valid])),
        fixed_ray_mismatch=stats(perp[valid]),
        fraction_squared_error_from_depth=float(np.sum(along[valid]**2)/np.sum(point[valid]**2)),
        ray_membership_max=float(np.max(ray_residual[valid])),pythagoras_max=float(np.max(pythagoras[valid])))
    groups={}
    for name,keep in [('nose',(ids>=27)&(ids<=35)),('mouth',ids>=48),('upper',(ids<27)|((ids>=36)&(ids<=47)))]:
        mask=valid&keep[None]
        groups[name]=dict(point_error=stats(point[mask]),along_ray_depth_error=stats(np.abs(along[mask])),
            along_ray_signed_bias=float(np.mean(along[mask])),fixed_ray_mismatch=stats(perp[mask]),
            fraction_squared_error_from_depth=float(np.sum(along[mask]**2)/np.sum(point[mask]**2)))
    record['regions']=groups;results[arm]=record
    savearrays[arm+'_along_error']=along;savearrays[arm+'_point_error']=point
np.savez_compressed(O/'ray-decomposition.npz',**savearrays)
result=dict(created=datetime.datetime.now().astimezone().isoformat(),source_hashes=hashes,
    definition='Yik is the observed-pixel ray intersection with the rendered blended plane. Qik is the closest point on that exact camera ray to the single global Xk. ||Y-X||^2 = ||X-Q||^2 + ||Y-Q||^2, since Y-Q is parallel to the ray.',
    interpretation='The perpendicular term is an irreducible lower bound if cameras and observed image locations stay fixed. The along-ray term is remaining depth disagreement with the same uncertain shared point. These are target-relative diagnostics, not independently measured truth.',
    common_observations=int(valid.sum()),source_frames=175,shared_feature_count=len(ids),arms=results,
    units='SfM world units, not calibrated meters',camera_updates=0,gaussian_updates=0,quality_accepted=False)
assert all(sha(Path(p))==v for p,v in hashes.items())
(O/'ray-decomposition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

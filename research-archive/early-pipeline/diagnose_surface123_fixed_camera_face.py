"""One bounded relaxation: fixed cameras/observations/weights, independent 3D face points.

This diagnoses the frozen SMPL-X constraint; points are not calibration truth,
new initialization, a new camera candidate, or a training artifact.
"""
from pathlib import Path
from datetime import datetime, timezone
import json,time,hashlib
import numpy as np
from scipy.optimize import minimize

ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'fixed-camera-face-relaxation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):
    x=np.asarray(x).ravel()
    return dict(n=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()),at_least_1px=int((x>=1).sum()))
def run(subject,lmk,bary):
    start=time.monotonic();root=ROOT/subject
    f=dict(np.load(root/'fit/fitted.npz'));o=dict(np.load(root/'data/observations.npz'));w=dict(np.load(root/'fit/observation-weights.npz'))
    rows=read(root/'data/inputs.json')['records'];E=np.asarray([r['w2c'] for r in rows]);K=np.asarray([r['K'] for r in rows]);R=E[:,:3,:3];t=E[:,:3,3]
    P=(f['vertices_world'][0][f['faces'][lmk]].astype(float)*bary[:,:,None]).sum(1)
    target=o['keypoints'][:,40:91];weights=w['face']*w['face_surface_visibility'];s=6*(o['boxes'][:,3]-o['boxes'][:,1])/1000
    fx,fy,cx,cy=K[:,0,0],K[:,1,1],K[:,0,2],K[:,1,2]
    def project(p):
        q=np.einsum('nij,j->ni',R,p)+t;z=q[:,2]
        if (z<=1e-7).any():raise FloatingPointError('Behind camera trial; do not force an invalid optimum.')
        uv=np.c_[fx*q[:,0]/z+cx-.5,fy*q[:,1]/z+cy-.5]
        jac=(np.stack([fx/z,0*z,-fx*q[:,0]/z**2,0*z,fy/z,-fy*q[:,1]/z**2],1).reshape(-1,2,3)@R)
        return uv,jac,z
    before=np.stack([project(p)[0] for p in P],1);after=before.copy();points=P.copy();reports=[]
    for j in range(51):
        if not weights[:,j].sum():continue
        ww=weights[:,j].astype(float);normal=weights.sum()
        def objective(p):
            if time.monotonic()-start>120:raise RuntimeError('Bounded120second diagnosis exceeded.')
            try:uv,J,z=project(p)
            except FloatingPointError:return 1e12,np.zeros(3)
            e=(uv-target[:,j])/s[:,None];den=np.sqrt(1+e*e)
            loss=float((ww[:,None]*(den-1)).sum()/normal)
            grad=np.einsum('n,na,nab->b',ww,e/den/s[:,None],J)/normal
            return loss,grad
        initial=objective(P[j])[0]
        sol=minimize(objective,P[j],jac=True,method='L-BFGS-B',options=dict(maxiter=300,ftol=1e-12,gtol=1e-9,maxls=40))
        uv,J,z=project(sol.x)
        if not np.isfinite(sol.fun) or sol.fun>initial+1e-9:raise RuntimeError('No valid decreasing relaxation; preserve evidence.')
        after[:,j]=uv;points[j]=sol.x
        # Independent central differences verify the actual objective gradient.
        at=P[j]+np.array([.001,-.0007,.0004])*float(f['world_scale']);analytic=objective(at)[1]
        eps=1e-5*float(f['world_scale']);numeric=np.array([(objective(at+eps*np.eye(3)[k])[0]-objective(at-eps*np.eye(3)[k])[0])/(2*eps) for k in range(3)])
        rel=float(np.linalg.norm(analytic-numeric)/max(np.linalg.norm(analytic),np.linalg.norm(numeric),1e-12))
        if rel>1e-5:raise RuntimeError(f'Objective gradient mismatch {rel}')
        reports.append(dict(face_iBUG=j+17,initial_objective=initial,relaxed_objective=float(sol.fun),optimizer_success=bool(sol.success),message=str(sol.message),iterations=int(sol.nit),jacobian_relative_error=rel,min_depth=float(z.min()),
            change_world_norm=float(np.linalg.norm(sol.x-P[j]))))
    b=np.linalg.norm(before-target,axis=-1);a=np.linalg.norm(after-target,axis=-1);active=weights>0
    result=dict(subject=subject,seconds=time.monotonic()-start,frames=len(rows),causal_variable='Remove only SMPL-X template/kinematic coupling for the same weighted facial points; fixed observations/K/E.',
        initial=stats(b[active]),relaxed=stats(a[active]),initial_exact_face_objective=sum(x['initial_objective'] for x in reports),relaxed_exact_face_objective=sum(x['relaxed_objective'] for x in reports),
        observations_or_weights_changed=False,cameras_changed=False,finite_local_optimum_not_proven_global_lower_bound=True,
        physical_material_identity_validated=False,no_point_promotion_or_training=True,points=reports,
        per_frame=[dict(frame=r['frame'],source_frame_index=r['source_frame_index'],initial=stats(b[i][active[i]]),relaxed=stats(a[i][active[i]])) for i,r in enumerate(rows)])
    np.savez_compressed(OUT/f'{subject}-diagnostic-only.npz',initial=P,relaxed=points,initial_projection=before,relaxed_projection=after,face_weights=weights)
    save(OUT/f'{subject}-result.json',result)
    return {k:v for k,v in result.items() if k not in ('points','per_frame')}
if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    save(OUT/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),subjects=['winterhat','whiteT','grayT'],budget='one frozen-initialization CPU solve per active face landmark;300iterations;120seconds perperson',
        variable='Shared3D facial points independent instead of tied SMPL-X; all95/93/112camera matrices and observations/weights frozen.',
        observation_rule='Keep every nonzero-weight observation; never discard by residual, never use to redefine holdout truth.',
        objective='Exact frozen fitter face pseudo-Huber objective, scale6*bodyHeight/1000 and unchanged face weights.',
        interpretation='Large improvement proves the frozen template constraints/fit leave avoidable detector residual at fixed cameras; remaining residual does not uniquely distinguish camera error, detector error, expression or motion.',
        no_gpu=True,no_new_camera=True,no_training=True,no_initialization_promotion=True))
    z=np.load('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz',allow_pickle=False)
    res=[run(s,z['lmk_faces_idx'],z['lmk_bary_coords']) for s in ['winterhat','whiteT','grayT']]
    save(OUT/'summary.json',res);print(json.dumps(res,ensure_ascii=False,indent=2))

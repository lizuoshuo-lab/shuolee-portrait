"""Diagnostic only: relax template coupling, keeping cameras/observations/weights."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

root=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
out=Path('/mnt/d/Projects/Shuolee-analysis-20261028/gray_current/face-template-control.json')
fit=dict(np.load(root/'fit/fitted.npz'));obs=dict(np.load(root/'data/observations.npz'))
weights=dict(np.load(root/'fit/observation-weights.npz'))
rows=json.loads((root/'data/training-inputs.json').read_text())['records']
E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows])
template=np.load('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
triangles=fit['faces'][template['lmk_faces_idx']]
initial=(fit['vertices_world'][0,triangles]*template['lmk_bary_coords'][:,:,None]).sum(1)
targets=obs['keypoints'][:,40:91];w=weights['face'].astype(float)
scale=6*(obs['boxes'][:,3]-obs['boxes'][:,1])/1000
def project(x):
    cam=np.einsum('nij,j->ni',E[:,:3,:3],x)+E[:,:3,3]
    q=np.einsum('nij,nj->ni',K,cam)
    return q[:,:2]/np.maximum(q[:,2:],.01)-.5
def residual(x,j):
    e=(project(x)-targets[:,j])/scale[:,None]
    return (e*np.sqrt(2/(np.sqrt(1+e*e)+1))*np.sqrt(w[:,j,None])).ravel()
before=np.zeros_like(targets);after=np.zeros_like(targets);point_reports=[]
for j in range(51):
    before[:,j]=project(initial[j]);after[:,j]=before[:,j]
    if not (w[:,j]>0).any():continue
    solved=least_squares(lambda x:residual(x,j),initial[j],max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    after[:,j]=project(solved.x)
    point_reports.append(dict(index=j,success=bool(solved.success),nfev=solved.nfev,
                              initial_objective=float(np.square(residual(initial[j],j)).sum()),
                              final_objective=float(np.square(residual(solved.x,j)).sum())))
keep=w>0
def statistics(uv):
    d=np.linalg.norm(uv-targets,axis=-1)
    e=(uv-targets)/scale[:,None,None]
    return dict(median_px=float(np.median(d[keep])),p90_px=float(np.quantile(d[keep],.9)),max_px=float(d[keep].max()),
                exact_face_objective=float(((np.sqrt(1+e*e)-1).sum(-1)*w).sum()/w.sum()),
                per_frame_median_px=[float(np.median(d[i,keep[i]])) for i in range(len(rows))])
report=dict(changed_variable='29 active face points may move independently in shared static 3D; no body-template coupling',
            before=statistics(before),after=statistics(after),points=point_reports,
            cameras_changed=False,observations_changed=False,weights_changed=False,
            source_fit_changed=False,new_training=False,held_out_validation=False,
            interpretation='Fitting control only; local optima do not establish a global lower bound or unique physical cause.')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{m:v[m] for m in ['median_px','p90_px','exact_face_objective']} for k,v in [('before',report['before']),('after',report['after'])]}))

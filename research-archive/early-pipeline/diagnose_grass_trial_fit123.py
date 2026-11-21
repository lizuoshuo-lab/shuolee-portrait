"""Read-only isolation of static ray support versus the current body optimizer."""
from trial_grass_surface123 import ROOT, OUT, read, save, sha, now
import numpy as np
from scipy.optimize import least_squares, linprog
from PIL import Image

rows=read(ROOT/'data/training-inputs.json')['records']
fit=dict(np.load(ROOT/'fit/fitted.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);K[:,:2,2]-=.5
P=K@E[:,:3,:];target=weights['body_target'];reliable=weights['body']*weights['visibility']>.25
out=[]
for j in range(17):
    ids=np.flatnonzero(reliable[:,j]);uv=target[ids,j];Ps=P[ids]
    if len(ids)<3:continue
    D=np.concatenate([uv[:,0,None]*Ps[:,2]-Ps[:,0],uv[:,1,None]*Ps[:,2]-Ps[:,1]],0)
    x=np.linalg.lstsq(D[:,:3],-D[:,3],rcond=None)[0]
    def residual(x):
        q=np.einsum('nij,j->ni',Ps[:,:,:3],x)+Ps[:,:,3]
        return (q[:,:2]/np.maximum(q[:,2:],1e-6)-uv).reshape(-1)
    opt=least_squares(residual,x,loss='soft_l1',f_scale=4.,max_nfev=200)
    d=np.linalg.norm(residual(opt.x).reshape(-1,2),axis=1)
    out.append(dict(joint=j,observations=len(ids),median_px=float(np.median(d)),p90_px=float(np.quantile(d,.9)),
                    max_px=float(d.max()),point_world=opt.x.tolist(),success=bool(opt.success)))
# Bounding rectangles are a necessary, not sufficient, condition for a common
# silhouette. No pixels, cameras, thresholds or training inputs are changed.
A=[];b=[]
for i,r in enumerate(rows):
    mask=np.array(Image.open(r['mask']).convert('L'))>127;y,x=np.where(mask)
    for axis,lo,hi in [(0,x.min()-2,x.max()+2),(1,y.min()-2,y.max()+2)]:
        for v in [P[i,axis]-hi*P[i,2],lo*P[i,2]-P[i,axis]]:
            A.append(v[:3]);b.append(-v[3])
    A.append(-E[i,2,:3]);b.append(E[i,2,3]-.001)
A=np.array(A);b=np.array(b);bounds=[]
for axis in range(3):
    low=linprog(np.eye(3)[axis],A_ub=A,b_ub=b,bounds=[(None,None)]*3,method='highs')
    high=linprog(-np.eye(3)[axis],A_ub=A,b_ub=b,bounds=[(None,None)]*3,method='highs')
    bounds.append(dict(axis=axis,min_status=int(low.status),max_status=int(high.status),
                       low=float(low.fun) if low.success else None,high=float(-high.fun) if high.success else None))
result=dict(updated=now(),purpose='Read-only diagnosis, these optimized detector joints are NOT physical truth or a new fit.',
            fit_sha256=sha(ROOT/'fit/fitted.npz'),camera_sha256=sha(ROOT/'data/training-inputs.json'),
            free_static_detector_joints=out,all94_mask_rectangle_intersection=bounds,
            mask_tolerance_native_px=2.,inputs_changed=False,training_started=False)
save(OUT/'fit-isolation.json',result)
print(result)

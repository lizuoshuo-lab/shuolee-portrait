"""One synthetic local-rank audit of the frozen reviewed FIT visibility pattern."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import time, json
from pathlib import Path
import numpy as np
import match_video10_temporal_bridge123 as b
U=b.R/'video10-remaining-material-review'; G=U/'generic-observability'
G.mkdir(exist_ok=True)
assert not (G/'plan.json').exists()
inputs=[U/'material-support.json',U/'ceiling-corrected-result.json',b.O/'plan.json']
plan=dict(registered=b.now(),hypothesis='A sequential 12-seed/8-extension rule can fail even if the reviewed visibility graph is generically locally observable. Separate algorithmic initialization failure from a structurally insufficient graph.',
    source_sha256={str(p):b.sha(p) for p in inputs},selection='All and only162 visually reviewed coarse FIT representative tracks, preserving their exact recorded frame identities. No HELD or unioned family branches.',
    parameters='One deterministic synthetic generic configuration, RNG12210. Known common pinhole K=I. This is not a candidate for any real video and uses no observed pixel values or old optimized geometry.',
    gauge='First camera R=I,t=0 fixed. Fix first synthetic 3D point z for the remaining global scale. Perturb others by left camera SE3 tangent and world XYZ.',
    checks='Positive depth, source hashes, first-camera and scale gauge, twelve central finite-difference columns with absolute error<1e-6. Normalize nonzero Jacobian columns then one SVD; rank threshold1e-9 times largest singular value chosen before execution.',
    budget=dict(CPU_seconds=120,random_configurations=1,SVD_runs=1,real_geometry_estimation=0,new_matches=0,training=0),
    interpretation='Full rank would only show that this synthetic local visibility pattern does not force a continuous ambiguity. It would not validate physical identities, numerical precision, K, real parallax, dynamic scene, global uniqueness or face error. Rank deficiency is a local synthetic result, not a universal impossibility proof.',
    stopping='Run once, save all singular values, per-frame null participation and finite differences. No threshold sweep or candidate selection. Do not train or change real cameras.')
b.save(G/'plan.json',plan); started=time.monotonic()
rows=[r for r in b.read(U/'material-support.json')['combined_rows'] if r['split']=='fit']
assert len(rows)==162 and len({r['family'] for r in rows})==162
frames=b.read(b.O/'plan.json')['frames']; assert len(frames)==33
obs=[(frames.index(f),i) for i,r in enumerate(rows) for f in sorted(set(r['frames']))]
rng=np.random.default_rng(12210); ncam=len(frames); npt=len(rows)

def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def rot(v):
    angle=np.linalg.norm(v); A=skew(v)
    if angle<1e-12: return np.eye(3)+A+0.5*A@A
    return np.eye(3)+np.sin(angle)/angle*A+(1-np.cos(angle))/angle**2*A@A

R=np.array([rot(v) for v in rng.uniform(-.15,.15,(ncam,3))]); R[0]=np.eye(3)
C=rng.uniform(-1.5,1.5,(ncam,3)); C[:,2]*=.1; C[0]=0
t=-np.einsum('nij,nj->ni',R,C)
X=rng.uniform(-1.5,1.5,(npt,3)); X[:,2]=rng.uniform(5,9,npt)
raw_n=6*(ncam-1)+3*npt; scale_column=6*(ncam-1)+2
keep=[i for i in range(raw_n) if i!=scale_column]; Jraw=np.zeros((len(obs)*2,raw_n))
depths=[]
for k,(ci,pi) in enumerate(obs):
    y=R[ci]@X[pi]+t[ci]; x,yv,z=y; depths.append(float(z))
    P=np.array([[1/z,0,-x/z**2],[0,1/z,-yv/z**2]])
    if ci:
        Jraw[2*k:2*k+2,6*(ci-1):6*ci]=P@np.column_stack((-skew(y),np.eye(3)))
    Jraw[2*k:2*k+2,6*(ncam-1)+3*pi:6*(ncam-1)+3*pi+3]=P@R[ci]
assert min(depths)>1
J=Jraw[:,keep]
def project(delta):
    raw=np.zeros(raw_n); raw[keep]=delta
    out=[]
    dX=raw[6*(ncam-1):].reshape(npt,3)
    for ci,pi in obs:
        y=R[ci]@(X[pi]+dX[pi])+t[ci]
        if ci:
            dv=raw[6*(ci-1):6*ci]; y=rot(dv[:3])@y+dv[3:]
        out.extend(y[:2]/y[2])
    return np.array(out)

fd=[]
cols=sorted(set(np.linspace(0,J.shape[1]-1,12,dtype=int).tolist()))
for c in cols:
    d=np.zeros(J.shape[1]); d[c]=1e-6
    numeric=(project(d)-project(-d))/(2e-6)
    err=float(np.max(np.abs(numeric-J[:,c]))); assert err<1e-6
    fd.append(dict(column=c,max_abs_error=err))
# Verify known similarity gauge of the unfixed-scale system before removing it.
scale=np.zeros(raw_n)
for ci in range(1,ncam): scale[6*(ci-1)+3:6*ci]=t[ci]
scale[6*(ncam-1):]=X.ravel()
assert np.max(np.abs(Jraw@scale))<1e-12
norm=np.linalg.norm(J,axis=0); nonzero=norm>1e-15
normalized=np.zeros_like(J); normalized[:,nonzero]=J[:,nonzero]/norm[nonzero]
left,singular,vt=np.linalg.svd(normalized,full_matrices=False)
threshold=float(singular[0]*1e-9); rank=int(np.sum(singular>threshold))
assert len(singular)==J.shape[1]
null=vt[rank:]; participation=[]
for ci in range(1,ncam):
    raw_cols=list(range(6*(ci-1),6*ci)); positions=[keep.index(c) for c in raw_cols]
    participation.append(dict(frame=frames[ci],normalized_null_energy=float(np.sum(null[:,positions]**2))))
elapsed=time.monotonic()-started; assert elapsed<120
for p,h in plan['source_sha256'].items(): assert b.sha(Path(p))==h
np.savez_compressed(G/'synthetic-input-and-jacobian.npz',R=R,t=t,X=X,observations=np.array(obs),frames=np.array(frames),families=np.array([r['family'] for r in rows]),jacobian=J,singular_values=singular)
result=dict(updated=b.now(),plan_sha256=b.sha(G/'plan.json'),elapsed_seconds=elapsed,fit_families=npt,frames=ncam,observations=len(obs),
    dimensions=list(J.shape),gauge_removed=7,rank=rank,nullity=J.shape[1]-rank,full_column_rank=rank==J.shape[1],
    threshold=threshold,largest_singular=float(singular[0]),smallest_singular=float(singular[-1]),singular_values=singular.tolist(),
    fd_controls=fd,scale_gauge_max_error=float(np.max(np.abs(Jraw@scale))),null_participation=participation,
    actual_pixels_used=False,real_camera_candidate=False,physical_calibration_accepted=False,training_admitted=False,
    limitation=plan['interpretation'])
b.save(G/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['singular_values','null_participation','fd_controls']},ensure_ascii=False))

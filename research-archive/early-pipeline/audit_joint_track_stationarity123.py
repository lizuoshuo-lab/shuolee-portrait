"""Read-only analytic Jacobian and observability audit of the frozen failed candidate."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.spatial.transform import Rotation
import json,hashlib,time,signal
root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer/joint-track-candidate')
out=root/'stationarity-audit';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (out/'plan.json').exists()
sources=[root/'diagnostic-candidate.npz',root/'result.json',root/'plan.json']
plan=dict(registered=now(),hypothesis='The one80-evaluation candidate stopped before convergence. Inspect the frozen actual FIT Jacobian and gradient to distinguish numerical stopping from an unavoidable local visibility nullspace; no further optimization.',
    source_sha256={str(p):sha(p) for p in sources},observations='All563FIT, no HELD input. Fixed conditionalK and existing failed candidate only as analysis object.',
    gauge='C0=0,R0=I; retain exact candidate fixed camera-center component. Left SO3 tangent for rotation derivatives, independent world center and point coordinates.',
    checks='Twelve preselected central finite-difference columns, epsilon1e-6, max relative error1e-5. Exactly one column-normalizedSVD; rank threshold1e-9 relative to largest, same as prior generic audit. Report robust gradient and all singular values.',
    budget=dict(CPU_seconds=60,Jacobian=1,SVD=1,BA=0,new_candidates=0,point_deletion=0,training=0),
    limitation='Local numerical rank of a nonstationary conditional solution is not physical calibration, statistical confidence, global uniqueness, proof of no static solution, or training admission.')
save(out/'plan.json',plan);started=time.monotonic()
def timeout(*args):raise TimeoutError('60second audit budget')
signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
result=read(root/'result.json')
with np.load(root/'diagnostic-candidate.npz',allow_pickle=False) as d:
    K,R,C,X,ids,uv,frames=[d[k] for k in ['K','R','C','X','ids','uv','frames']]
n,m=len(C),len(X);nr=3*(n-1);nc=3*(n-1);rawcols=nr+nc+3*m;fixed=result['scale_component_fixed'][0];drop=nr+fixed
keep=np.delete(np.arange(rawcols),drop);Jraw=np.zeros((2*len(ids),rawcols));camera=[]
def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
for k,(ci,pi) in enumerate(ids):
    p=R[ci]@(X[pi]-C[ci]);x,y,z=p;camera.append(p)
    P=np.array([[K[0,0]/z,0,-K[0,0]*x/z**2],[0,K[1,1]/z,-K[1,1]*y/z**2]])
    if ci:
        Jraw[2*k:2*k+2,3*(ci-1):3*ci]=-P@skew(p)
        Jraw[2*k:2*k+2,nr+3*(ci-1):nr+3*ci]=-P@R[ci]
    Jraw[2*k:2*k+2,nr+nc+3*pi:nr+nc+3*pi+3]=P@R[ci]
J=Jraw[:,keep];camera=np.asarray(camera);assert np.all(camera[:,2]>0)
pred=camera[:,:2]/camera[:,2,None]*np.diag(K)[:2]+K[:2,2];residual=(pred-uv).ravel()
def project_delta(v):
    raw=np.zeros(rawcols);raw[keep]=v
    dR=np.concatenate([np.eye(3)[None],Rotation.from_rotvec(raw[:nr].reshape(-1,3)).as_matrix()])
    dC=np.vstack([np.zeros(3),raw[nr:nr+nc].reshape(-1,3)]);dX=raw[nr+nc:].reshape(-1,3)
    p=np.einsum('nij,nj->ni',dR[ids[:,0]]@R[ids[:,0]],X[ids[:,1]]+dX[ids[:,1]]-C[ids[:,0]]-dC[ids[:,0]])
    return (p[:,:2]/p[:,2,None]*np.diag(K)[:2]+K[:2,2]).ravel()
fd=[]
for j in np.linspace(0,len(keep)-1,12,dtype=int):
    v=np.zeros(len(keep));v[j]=1e-6;num=(project_delta(v)-project_delta(-v))/2e-6
    absolute=float(np.max(np.abs(num-J[:,j])));relative=absolute/max(1,float(np.max(np.abs(J[:,j]))));assert relative<1e-5
    fd.append(dict(column=int(j),absolute_error=absolute,relative_error=relative))
# Similarity scale is a null direction before fixing one center coordinate.
scale=np.r_[np.zeros(nr),C[1:].ravel(),X.ravel()];gauge=float(np.max(np.abs(Jraw@scale)));assert gauge<1e-7
norm=np.linalg.norm(J,axis=0);assert np.all(norm>0)
_,s,vt=np.linalg.svd(J/norm,full_matrices=False);threshold=s[0]*1e-9;rank=int(np.sum(s>threshold))
gradient=J.T@(residual/np.sqrt(1+residual**2));weak=vt[-1];frame_energy=[]
for ci in range(1,n):
    cols=list(range(3*(ci-1),3*ci))+list(range(nr+3*(ci-1),nr+3*ci));indices=[int(np.flatnonzero(keep==c)[0]) for c in cols if c!=drop]
    frame_energy.append(dict(frame=int(frames[ci]),weakest_normalized_mode_energy=float(np.sum(weak[indices]**2))))
np.savez_compressed(out/'jacobian.npz',jacobian=J,singular_values=s,gradient=gradient,keep_columns=keep,column_norm=norm,weakest_normalized_vector=weak)
for p,h in plan['source_sha256'].items():assert sha(Path(p))==h
final=dict(updated=now(),elapsed_seconds=time.monotonic()-started,dimensions=list(J.shape),rank=rank,nullity=J.shape[1]-rank,
    normalized_condition=float(s[0]/s[-1]),smallest_singular=float(s[-1]),largest_singular=float(s[0]),threshold=float(threshold),
    robust_gradient_max_abs=float(np.max(np.abs(gradient))),parameterization='local left rotation/world centers/world XYZ; not identical to global Rodrigues optimizer optimality',
    solver_reported_converged=result['optimizer']['success'],solver_reported_optimality=result['optimizer']['optimality'],finite_difference=fd,scale_gauge_max_abs=gauge,
    weakest_mode_camera_energy=sorted(frame_energy,key=lambda r:-r['weakest_normalized_mode_energy']),
    physical_accuracy_accepted=False,parameters_changed=False,training_started=False,limitation=plan['limitation'])
save(out/'result.json',final);signal.alarm(0);print(json.dumps({k:v for k,v in final.items() if k not in ['finite_difference','weakest_mode_camera_energy']},ensure_ascii=False))

"""Execute the preregistered one-scene analytic joint solver control, no real candidate."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,signal,sys,traceback
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from analytic_fixedk_bundle123 import FixedKBundle
ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
P=ROOT/'video10-global-initializer/joint-track-candidate';O=P/'joint-synthetic-control';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
winpath=lambda s:Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/'))
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def state(s,**data):save(O/'progress.json',dict(updated=now(),stage=s,pid=os.getpid(),real_candidate=False,training_started=False,**data))
def stats(x):
    x=np.asarray(x,float);return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),p90=float(np.percentile(x,90)),max=float(x.max()))
assert not (O/'execution-admission.json').exists();plan=read(P/'joint-synthetic-plan.json')
for s,h in plan['source_sha256'].items():assert sha(winpath(s))==h
original_plan=read(P/'plan.json');kp=ROOT/'video10-remaining-material-review/global-input-preflight/conditional-camera.json'
kh=next(h for s,h in original_plan['source_sha256'].items() if s.endswith('conditional-camera.json'));assert sha(kp)==kh
save(O/'execution-admission.json',dict(updated=now(),pid=os.getpid(),python=sys.executable,plan_sha256=sha(P/'joint-synthetic-plan.json'),script_sha256=sha(Path(__file__)),module_sha256=sha(Path(__file__).with_name('analytic_fixedk_bundle123.py')),K_sha256=kh,training=False,GPU=False))
start=time.monotonic()
def timeout(*args):raise TimeoutError('60second registered synthetic budget')
signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
try:
    state('construct_frozen_synthetic_input');K=np.asarray(read(kp)['K_integer_centers'])
    with np.load(ROOT/'video10-remaining-material-review/generic-observability/synthetic-input-and-jacobian.npz',allow_pickle=False) as d:
        R,t,X,ids=[d[k] for k in ['R','t','X','observations']]
    C=-np.einsum('nji,nj->ni',R,t);fixed=int(np.argmax(np.abs(C[1:].ravel())))
    camera=np.einsum('nij,nj->ni',R[ids[:,0]],X[ids[:,1]]-C[ids[:,0]]);h=camera@K.T;truth=h[:,:2]/h[:,2,None]
    rng=np.random.default_rng(12312);uv=truth+rng.normal(0,.2,truth.shape)
    perturb=Rotation.from_rotvec(rng.normal(0,np.deg2rad(1),(len(R),3))).as_matrix();R0=perturb@R;R0[0]=R[0]
    C0=C+rng.normal(0,.02,C.shape);C0[0]=C[0];C0[1:].ravel()[fixed]=C[1:].ravel()[fixed]
    X0=X+rng.normal(0,.02,X.shape)
    model=FixedKBundle(K,R0,C0,X0,ids,uv,fixed,float(np.median(camera[:,2])))
    np.savez_compressed(O/'input.npz',K=K,R_truth=R,C_truth=C,X_truth=X,R_initial=R0,C_initial=C0,X_initial=X0,ids=ids,uv=uv,uv_truth=truth,fixed_center_component=fixed)
    state('finite_difference_control');J=model.jacobian(model.initial);checks=[]
    for col in np.linspace(0,len(model.initial)-1,12,dtype=int):
        delta=np.zeros_like(model.initial);delta[col]=1e-6
        numeric=(model.residual(model.initial+delta)-model.residual(model.initial-delta))/2e-6;analytic=J[:,col].toarray().ravel()
        absolute=float(np.max(np.abs(numeric-analytic)));relative=absolute/max(1,float(np.max(np.abs(analytic))))
        checks.append(dict(column=int(col),max_abs=absolute,max_relative=relative));assert relative<1e-5
    save(O/'finite-difference.json',dict(updated=now(),passed=True,columns=checks,jacobian_shape=list(J.shape)))
    initial=model.geometry(model.initial);initial_error=np.linalg.norm(initial[-1]-uv,axis=1)
    state('one_analytic_joint_BA',max_nfev=200)
    opt=least_squares(model.residual,model.initial,jac=model.jacobian,method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=1.,max_nfev=200,x_scale='jac')
    R1,C1,X1,_,cam,_,pred=model.geometry(opt.x);error=np.linalg.norm(pred-uv,axis=1)
    rot_deg=np.degrees(Rotation.from_matrix(R1@R.transpose(0,2,1)).magnitude())
    np.savez_compressed(O/'result-arrays.npz',K=K,R=R1,C=C1,X=X1,predicted_uv=pred,error=error,depth=cam[:,2],ids=ids,uv=uv,parameters=opt.x)
    result=dict(updated=now(),state='synthetic_joint_control_completed',elapsed_seconds=time.monotonic()-start,
        passed=bool(opt.success and error.mean()<.5 and error.max()<1.5 and np.all(cam[:,2]>0)),
        optimizer=dict(success=bool(opt.success),status=int(opt.status),message=opt.message,nfev=opt.nfev,njev=opt.njev,cost=float(opt.cost),optimality=float(opt.optimality)),
        initial_error_px=stats(initial_error),final_error_px=stats(error),rotation_error_deg=stats(rot_deg),center_error_scene_units=stats(np.linalg.norm(C1-C,axis=1)),point_error_scene_units=stats(np.linalg.norm(X1-X,axis=1)),
        all_depth_positive=bool(np.all(cam[:,2]>0)),first_camera_fixed=bool(np.array_equal(R1[0],R[0]) and np.array_equal(C1[0],C[0])),scale_component_fixed=bool(C1[1:].ravel()[fixed]==C[1:].ravel()[fixed]),
        finite_difference_passed=True,frames=len(R),points=len(X),observations=len(uv),variables=len(opt.x),real_candidate=False,physical_calibration_accepted=False,training_started=False,
        limitation='Synthetic implementation-only result at one known scene and perturbation. Does not evaluate physical materials, real camera branches, face accuracy or training admission.')
    for s,h in plan['source_sha256'].items():assert sha(winpath(s))==h
    save(O/'result.json',result);signal.alarm(0);save(O/'progress.json',dict(updated=now(),stage='complete',pid=None,real_candidate=False,training_started=False));print(json.dumps(result,ensure_ascii=False))
except BaseException as e:
    signal.alarm(0);save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));save(O/'progress.json',dict(updated=now(),stage='failed',pid=None,training_started=False));raise

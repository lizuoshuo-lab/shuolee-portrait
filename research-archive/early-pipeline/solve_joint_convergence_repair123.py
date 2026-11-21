"""One preregistered analytic convergence repair; same fresh start and frozen observations."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,signal,sys,traceback
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from analytic_fixedk_bundle123 import FixedKBundle
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');P=R/'video10-global-initializer/joint-track-candidate';O=P/'convergence-repair';U=R/'video10-remaining-material-review/global-input-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
winpath=lambda s:Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/'))
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def state(s,**v):save(O/'progress.json',dict(updated=now(),pid=os.getpid(),stage=s,training_started=False,**v))
def stats(a):
    a=np.asarray(a,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max())) if len(a) else dict(n=0)
assert not (O/'execution-admission.json').exists();plan=read(O/'plan.json')
for s,h in plan['source_sha256'].items():assert sha(winpath(s))==h
save(O/'execution-admission.json',dict(updated=now(),pid=os.getpid(),python=sys.executable,plan_sha256=sha(O/'plan.json'),script_sha256=sha(Path(__file__)),training=False,GPU=False))
start=time.monotonic()
def timeout(*args):raise TimeoutError('120second registered convergence repair budget')
signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
try:
    state('verify_original_fresh_start')
    with np.load(P/'initialization.npz',allow_pickle=False) as d:
        K,R0,C0,X0,ids,uv,frames,families=[d[k] for k in ['K','R','C','X','ids','uv','frames','families']]
    fit=read(U/'fit-observations.json');fi={int(f):i for i,f in enumerate(frames)}
    check_ids=[];check_uv=[]
    for pi,t in enumerate(fit['tracks']):
        assert t['family']==families[pi]
        for ob in t['observations']:check_ids.append([fi[ob['frame']],pi]);check_uv.append(ob['uv'])
    assert np.array_equal(ids,np.asarray(check_ids)) and np.array_equal(uv,np.asarray(check_uv))
    assert np.array_equal(R0[0],np.eye(3)) and np.array_equal(C0[0],np.zeros(3))
    cam=np.einsum('nij,nj->ni',R0[ids[:,0]],X0[ids[:,1]]-C0[ids[:,0]]);z=cam[:,2];depth_scale=float(np.median(z[z>0]))
    fixed=int(np.argmax(np.abs(C0[1:].ravel())));model=FixedKBundle(K,R0,C0,X0,ids,uv,fixed,depth_scale)
    # Independent replay of the previously used residual, using original gauge/order.
    nr=3*(len(frames)-1);cf=C0[1:].ravel();cfree=np.delete(np.arange(len(cf)),fixed);xs=nr+len(cfree)
    old_v0=np.r_[Rotation.from_matrix(R0[1:]).as_rotvec().ravel(),cf[cfree],X0.ravel()];assert np.array_equal(model.initial,old_v0)
    rot=np.concatenate([np.eye(3)[None],Rotation.from_rotvec(old_v0[:nr].reshape(-1,3)).as_matrix()]);cam_old=np.einsum('nij,nj->ni',rot[ids[:,0]],X0[ids[:,1]]-C0[ids[:,0]])
    zz=cam_old[:,2];den=np.where(np.abs(zz)>1e-8,zz,np.where(zz>=0,1e-8,-1e-8));pr=cam_old[:,:2]/den[:,None]*np.diag(K)[:2]+K[:2,2]
    reference=np.c_[pr-uv,1000*np.minimum(zz/depth_scale-1e-5,0.)].ravel();difference=float(np.max(np.abs(reference-model.residual(model.initial))));assert difference<1e-8
    state('analytic_jacobian_equivalence');J=model.jacobian(model.initial);checks=[]
    for j in np.linspace(0,len(old_v0)-1,12,dtype=int):
        delta=np.zeros_like(old_v0);delta[j]=1e-6;numeric=(model.residual(old_v0+delta)-model.residual(old_v0-delta))/2e-6;analytic=J[:,j].toarray().ravel()
        absolute=float(np.max(np.abs(numeric-analytic)));relative=absolute/max(1,float(np.max(np.abs(analytic))));checks.append(dict(column=int(j),absolute=absolute,relative=relative));assert relative<1e-5
    save(O/'equivalence.json',dict(updated=now(),initial_parameters_identical=True,all_initial_residual_max_difference=difference,finite_difference_passed=True,columns=checks,observations=563,old_optimized_candidate_loaded=False))
    count=[0]
    def residual(v):
        count[0]+=1
        if count[0]%100==0:state('one_analytic_BA',residual_calls=count[0],elapsed_seconds=time.monotonic()-start,max_nfev=2000)
        return model.residual(v)
    state('one_analytic_BA',max_nfev=2000)
    opt=least_squares(residual,old_v0,jac=model.jacobian,method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=1.,max_nfev=2000,x_scale='jac')
    R1,C1,X1,_,cam1,_,pred=model.geometry(opt.x);fiterr=np.linalg.norm(pred-uv,axis=1)
    np.savez_compressed(O/'diagnostic-candidate.npz',K=K,R=R1,C=C1,X=X1,ids=ids,uv=uv,predicted_uv=pred,depth=cam1[:,2],frames=frames,families=families,parameters=opt.x)
    freeze_hash=sha(O/'diagnostic-candidate.npz');save(O/'candidate-frozen.json',dict(updated=now(),sha256=freeze_hash,HELD_not_yet_loaded=True,optimizer_success=bool(opt.success)))
    state('frozen_HELD_evaluation');held=read(U/'heldout-observations.json');rows=[]
    for tr in held['tracks']:
        for excluded in tr['observations']:
            others=[o for o in tr['observations'] if o['frame']!=excluded['frame']];row=dict(family=tr['family'],frame=excluded['frame'],uv=excluded['uv'],fit_used=False)
            if len(others)<2:row['state']='insufficient_other_views';rows.append(row);continue
            A=[]
            for o in others:
                j=fi[o['frame']];projection=K@np.c_[R1[j],-R1[j]@C1[j]];u,v=o['uv'];A.extend([u*projection[2]-projection[0],v*projection[2]-projection[1]])
            _,_,vt=np.linalg.svd(np.asarray(A));h=vt[-1]
            if abs(h[3])<1e-12:row['state']='point_at_infinity';rows.append(row);continue
            xyz=h[:3]/h[3];j=fi[excluded['frame']];camh=R1[j]@(xyz-C1[j]);hp=K@camh
            if not np.all(np.isfinite(hp)) or abs(hp[2])<1e-12:row['state']='nonfinite_or_zero_depth_projection';rows.append(row);continue
            predh=hp[:2]/hp[2];err=float(np.linalg.norm(predh-np.asarray(excluded['uv'])))
            positive=all((R1[fi[o['frame']]]@(xyz-C1[fi[o['frame']]]))[2]>0 for o in others)
            row.update(state='valid' if camh[2]>0 and positive else 'behind_camera',error_px=err,predicted_uv=predh.tolist(),triangulation_frames=[o['frame'] for o in others],xyz=xyz.tolist());rows.append(row)
    save(O/'heldout-predictions.json',rows)
    baseline={(v['family'],v['frame']):v for v in read(P/'heldout-predictions.json')};assert len(baseline)==len(rows)==161
    comparisons=[]
    for row in rows:
        old=baseline[(row['family'],row['frame'])];assert row['uv']==old['uv']
        comparisons.append(dict(family=row['family'],frame=row['frame'],old_state=old['state'],new_state=row['state'],old_error_px=old.get('error_px'),new_error_px=row.get('error_px')))
    save(O/'all-observation-comparison.json',comparisons)
    per=[]
    for f in frames:
        ix=ids[:,0]==fi[int(f)];hh=[x for x in rows if x['frame']==f]
        per.append(dict(frame=int(f),fit_projection=stats(fiterr[ix]),held_total=len(hh),held_valid=sum(x['state']=='valid' for x in hh),held_errors=stats([x['error_px'] for x in hh if 'error_px' in x])))
    save(O/'per-frame.json',per)
    final=dict(updated=now(),state='one_convergence_repair_completed',elapsed_seconds=time.monotonic()-start,
        optimizer=dict(success=bool(opt.success),status=int(opt.status),message=opt.message,nfev=opt.nfev,njev=opt.njev,cost=float(opt.cost),optimality=float(opt.optimality)),
        fit_projection=stats(fiterr),fit_positive_depth_fraction=float(np.mean(cam1[:,2]>0)),heldout_total=len(rows),heldout_valid=sum(x['state']=='valid' for x in rows),
        heldout_error_px=stats([x['error_px'] for x in rows if 'error_px' in x]),heldout_strictly_below1px=sum(x['state']=='valid' and x['error_px']<1 for x in rows),
        heldout_all_valid_and_subpixel=all(x['state']=='valid' and x['error_px']<1 for x in rows),
        same_initialization=True,K_changed=False,observations_or_weights_changed=False,physical_face_HELD=0,full_training_frame_registration=False,
        formal_camera_admission=False,SMPLX_fit=False,training_started=False,quality_accepted=False,
        limitation='Numerical derivative and budget repair together. Solver termination is not proof of unique physical cameras or global optimum; HELD material independence and face coverage remain unverified.')
    for s,h in plan['source_sha256'].items():assert sha(winpath(s))==h
    assert sha(O/'diagnostic-candidate.npz')==freeze_hash
    save(O/'result.json',final);signal.alarm(0);save(O/'progress.json',dict(updated=now(),pid=None,stage='complete',training_started=False));print(json.dumps(final,ensure_ascii=False))
except BaseException as e:
    signal.alarm(0);save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));save(O/'progress.json',dict(updated=now(),pid=None,stage='failed',training_started=False));raise

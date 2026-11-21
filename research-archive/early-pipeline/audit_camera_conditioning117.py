"""Bounded source/geometry/K-sensitivity audit, keeping every camera immutable."""
from pathlib import Path
import datetime as dt, hashlib, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares

def local(p):
    s=str(p).replace('\\','/')
    return Path(s.replace('C:/','/mnt/c/')) if os.name!='nt' else Path(s.replace('/mnt/c/','C:/'))
B=local('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/portrait-causal-diagnosis-117';P=B/'current/background-architecture-109';D=O/'D1-conditioning'
REPO=local('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=len(a),q10=float(np.quantile(a,.1)),median=float(np.median(a)),q90=float(np.quantile(a,.9))) if len(a) else dict(n=0)
def project(X,E,K):
    v=np.einsum('fij,j->fi',E[:,:3,:3],X)+E[:,:3,3]
    uv=(v@K.T);return uv[:,:2]/uv[:,2:]-.5,v
def fit_fresh(uv,E,K):
    center=-E[0,:3,:3].T@E[0,:3,3]
    PP=E[:,:3].copy();PP[:,:,3]+=PP[:,:,:3]@center
    q=np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(K).T
    A=np.concatenate([q[:,0,None]*PP[:,2]-PP[:,0],q[:,1,None]*PP[:,2]-PP[:,1]])
    _,_,v=np.linalg.svd(A,full_matrices=False);x=v[-1,:3]/v[-1,3]+center
    opt=least_squares(lambda y:(project(y,E,K)[0]-uv).ravel(),x,loss='soft_l1',f_scale=2,
        max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    return opt.x,opt.success
def derivatives(X,E,K):
    uv,v=project(X,E,K);x,y,z=v.T;f=K[0,0]
    jlocal=np.zeros((len(z),2,3));jlocal[:,0,0]=f/z;jlocal[:,1,1]=f/z
    jlocal[:,0,2]=-f*x/z**2;jlocal[:,1,2]=-f*y/z**2
    Jx=np.einsum('fij,fjk->fik',jlocal,E[:,:3,:3]).reshape(-1,3)
    Jk=np.zeros((len(z),2,3));Jk[:,0,0]=.01*f*x/z;Jk[:,1,0]=.01*f*y/z
    Jk[:,0,1]=1;Jk[:,1,2]=1
    return Jx,Jk.reshape(-1,3)
def block_information(Jx,Jk,residual,robust):
    # Fixed radial SoftL1 block weights; a local sensitivity approximation, not a new camera solve.
    w=(1+np.sum(residual.reshape(-1,2)**2,axis=1)/4)**(-.25) if robust else np.ones(len(residual)//2)
    ww=np.repeat(w,2);A=Jx*ww[:,None];K=Jk*ww[:,None];r=residual*ww
    U,s,_=np.linalg.svd(A,full_matrices=False);U=U[:,s>max(s[0],1.)*1e-10]
    Q=K-U@(U.T@K);rr=r-U@(U.T@r)
    return Q.T@Q,Q.T@rr,K.T@K,np.linalg.norm(rr)**2
def matrix_report(H,g,raw):
    diag=np.sqrt(np.maximum(np.diag(H),1e-30));C=H/np.outer(diag,diag)
    vals=np.linalg.eigvalsh(C);rank=int(np.linalg.matrix_rank(H))
    return dict(rank=rank,column_normalized_eigenvalues=vals.tolist(),
        column_normalized_condition=float(vals[-1]/vals[0]) if vals[0]>1e-12 else None,
        xyz_marginalized_information=H.tolist(),fixed_xyz_information=raw.tolist(),
        fraction_after_eliminating_XYZ=(np.diag(H)/np.maximum(np.diag(raw),1e-30)).tolist(),
        local_linear_step_percent_f_cxpx_cypx=(-np.linalg.pinv(H,rcond=1e-12)@g).tolist(),
        meaning='Conditional locallinear direction at074, withpointmotion eliminated and extrinsicsfixed. Not an applied calibration, uncertainty interval, ortruephysicalK.')

def prepare():
    assert not (D/'preregistration.json').exists() and not (D/'start.json').exists(),'Do not duplicate an existing diagnostic.'
    D.mkdir(exist_ok=True);history=read(O/'historical-background-audit.json')
    for n,h in history['source_sha256'].items():assert sha(P/n)==h
    save(D/'preregistration.json',dict(created=now(),
        hypothesis='Sharedintrinsic error could leave depth-dependent residuals even when background-onlyR/trefinement fails. Test conditionalKsupport and whether ceiling/floor suggest compatible corrections before anycameraoptimization.',
        sole_variable='None applied: read-only linearized sensitivity using freshpoint-only fits and immutable074K/E.',
        source='109admittedraw2Dtracks.2127fit/557held preserved byoriginalIDmodulo5. No oldoptimizedXYZ/cameras are loaded.',
        geometry='FreshDLT and robustpoint-only refinement fromoriginal1092D; same originalK with +0.5 toCOLMAPcorners / -0.5 fromprojection tointegerpixelcenters.',
        intrinsics='Squarepixel/no-skew fixedrectification model: deltaf unit1percent oforiginalf, deltacx/cy unit1originalpixel. Distortion remains unchanged. Intrinsic variation is theoreticalsensitivity only.',
        measurement='AnalyticJacobian and finite-difference check; projectKJacobian off eachtrackXYZJacobian before aggregating. Report unweighted and fixedradialSoftL1weighted matrices, perregion/depth/time support; onlyfittracks determine primary linear directions.',
        comparisons='Ceiling vsfloor and near/middle/far estimatedtrackdepth strata. Depthbinquantiles andmediantrackframe bins are descriptive, not tuning/selection.',
        limits=['Extrinsic uncertainty NOT marginalized; fixedR/t conditionalrank is not globalself-calibration identifiability.',
                'No absolute units orcalibrationGT. Tracks retain sourcepositive-depthselection andimageidentity uncertainties.',
                'Oldsplit is validation reusedforanalysis, notnewunseen test. No newpoint/observation is dropped byresidual.'],
        admission='Analyticfixtures finite and maximumrelativeJacobianerror<1e-5; exact175ID,K/E agreement; allsourcehashesunchanged. Anyfit/nonpositivefailure isexplicit and blocks a calibrationcandidate.',
        stop='One CPUrun <=600seconds, noGPU, noK/Eupdates orautomaticGaussian training.',
        source_hashes={n:sha(P/n) for n in ['admitted-tracks.npz','solver-input.npz','solver-protocol.json']},
        code_sha256=sha(Path(__file__))))
    q=read(O/'queue-state.json');q.update(updated=now(),state='D1_conditioning_registered',stage='D1_conditioning',active_pid=None,
        next_action='Run once audit_camera_conditioning117.py run in existingWSLCPU environment, theninspect results andsourceimages.');save(O/'queue-state.json',q)
    print(json.dumps(dict(state='registered',root=str(D))),flush=True)

def run():
    assert (D/'preregistration.json').exists() and not (D/'start.json').exists()
    pr=read(D/'preregistration.json')
    for n,h in pr['source_hashes'].items():assert sha(P/n)==h
    save(D/'start.json',dict(created=now(),pid=os.getpid(),command=sys.argv,gpu=False,code_sha256=sha(Path(__file__))))
    q=read(O/'queue-state.json');q.update(updated=now(),state='D1_conditioning_running',active_pid=os.getpid(),active_environment='WSL',active_log='D1-conditioning/stdout.log');save(O/'queue-state.json',q)
    t0=time.monotonic();a=np.load(P/'admitted-tracks.npz');ob=a['observations'];off=a['offsets'];reg=a['regions'];ids=a['original_ids'];K=a['K'];E=a['w2c']
    manifest=read(B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json')['records']
    assert len(manifest)==175 and np.array_equal(E,np.array([r['w2c'] for r in manifest]))
    assert all(np.array_equal(K,r['K']) for r in manifest)
    assert all(np.all(np.diff(ob[lo:hi,0])>0) for lo,hi in zip(off[:-1],off[1:]))
    n=len(ids);XX=np.zeros((n,3));dep=[];ang=[];errors=[];invalid=[];tests=[]
    hess=np.zeros((2,n,3,3));grad=np.zeros((2,n,3));raw=np.zeros((2,n,3,3));cost=np.zeros((2,n));nobs=np.diff(off);seed=[]
    for j,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
        if time.monotonic()-t0>580:raise TimeoutError('CPUbudget exhausted; no automaticretry')
        o=ob[lo:hi];f=o[:,0].astype(int);ee=E[f];uv=o[:,1:]
        X,success=fit_fresh(uv,ee,K);pred,v=project(X,ee,K);z=v[:,2];res=(pred-uv).ravel();Jx,Jk=derivatives(X,ee,K)
        if not success or not np.isfinite(X).all() or not(z>0).all():invalid.append(int(ids[j]))
        if j<8:
            eps=1e-5
            for k in range(3):
                delta=np.eye(3)[k]*eps;fd=((project(X+delta,ee,K)[0]-project(X-delta,ee,K)[0])/(2*eps)).ravel()
                tests.append(float(np.linalg.norm(fd-Jx[:,k])/max(np.linalg.norm(Jx[:,k]),1)))
                kp=K.copy();km=K.copy()
                if k==0:kp[0,0]+=K[0,0]*.01*eps;kp[1,1]+=K[1,1]*.01*eps;km[0,0]-=K[0,0]*.01*eps;km[1,1]-=K[1,1]*.01*eps
                else:kp[k-1,2]+=eps;km[k-1,2]-=eps
                fd=((project(X,ee,kp)[0]-project(X,ee,km)[0])/(2*eps)).ravel()
                tests.append(float(np.linalg.norm(fd-Jk[:,k])/max(np.linalg.norm(Jk[:,k]),1)))
        for w in range(2):hess[w,j],grad[w,j],raw[w,j],cost[w,j]=block_information(Jx,Jk,res,bool(w))
        centers=-np.einsum('fji,fj->fi',ee[:,:3,:3],ee[:,:3,3]);rays=X-centers;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        theta=np.degrees(np.arccos(np.clip(rays[0]@rays[-1],-1,1)))
        XX[j]=X;dep.append(float(np.median(z)));ang.append(float(theta));errors.append(float(np.median(np.linalg.norm(res.reshape(-1,2),axis=1))));seed.append(float(np.median(f)))
        if j%300==0:
            save(D/'progress.json',dict(updated=now(),pid=os.getpid(),tracks=j+1,total=n,seconds=time.monotonic()-t0));print(f'track {j+1}/{n}',flush=True)
    fit=ids%5!=0;dep=np.array(dep);ang=np.array(ang);seed=np.array(seed);cuts=np.quantile(dep[fit],[1/3,2/3])
    groups={'fit_all':fit,'fit_ceiling':fit&(reg==0),'fit_floor':fit&(reg==1),
        'fit_near':fit&(dep<=cuts[0]),'fit_middle_depth':fit&(dep>cuts[0])&(dep<=cuts[1]),'fit_far':fit&(dep>cuts[1]),
        'fit_early':fit&(seed<58),'fit_middle_time':fit&(seed>=58)&(seed<117),'fit_late':fit&(seed>=117),
        'held_background_secondary':~fit}
    summaries={}
    for name,sel in groups.items():
        summaries[name]=dict(tracks=int(sel.sum()),observations=int(nobs[sel].sum()),
            camera_depth_arbitrary_units=stats(dep[sel]),endpoint_angle_degrees=stats(ang[sel]),point_fit_median_px=stats(np.array(errors)[sel]),
            linear={label:matrix_report(hess[w,sel].sum(0),grad[w,sel].sum(0),raw[w,sel].sum(0)) for w,label in enumerate(['unweighted','robust2px'])})
    checks=dict(analytic_finite_difference_relative_max=max(tests),jacobians_pass=max(tests)<1e-5,
        all175_K_E_exact_match=True,frame_ids_unique_per_track=True,no_old_XYZ_loaded=True,
        no_camera_parameter_changed=True,invalid_point_fit_tracks=invalid,sources_unchanged=all(sha(P/n)==h for n,h in pr['source_hashes'].items()))
    assert checks['jacobians_pass'] and checks['sources_unchanged']
    result=dict(updated=now(),state='conditioning_complete_review_pending',seconds=time.monotonic()-t0,
        tracks=n,fit_tracks=int(fit.sum()),held_tracks=int((~fit).sum()),fit_observations=int(nobs[fit].sum()),
        source_support=dict(ceiling_tracks=int((reg==0).sum()),floor_tracks=int((reg==1).sum()),
            observations_between_y650_and2050=int(((ob[:,2]>650)&(ob[:,2]<2050)).sum()),
            note='109admission selected top and floor bands. Coverage across175frames does not imply stableobjectsupport near faceheight.'),
        by_group=summaries,depth_cuts_arbitrary_units=cuts.tolist(),checks=checks,
        intrinsic_parameters=['f commonfxfy: unit1percent','cx: unit1originalpixel','cy: unit1originalpixel'],
        interpretation='Conditional linearized directions only; not applied asK. Rank is conditional onestimated074extrinsics. A newcamera candidate requires explicit separate preregistration andheld evaluation.',
        root_cause_resolved=False,camera_optimization=False,gaussian_training=False,new_model=False)
    np.savez_compressed(D/'conditioning.npz',xyz_fresh=XX,track_original_ids=ids,fit=fit,regions=reg,
        depth=dep,endpoint_angles=ang,hessian=hess,gradient=grad,raw_information=raw,cost=cost,point_fit_median_px=errors)
    save(D/'result.json',result);save(D/'progress.json',dict(updated=now(),state='completed',tracks=n,total=n,seconds=time.monotonic()-t0))
    q=read(O/'queue-state.json');q.update(updated=now(),state='D1_conditioning_review_pending',active_pid=None,stage='D1_review',
        next_action='Inspect conditionalKdirections/nearfarconditioning, acknowledge109onlytop/floor support, review fixed sourcecrop before deciding one boundednewKcandidate. Do notapply linearstep ascalibration.',result='D1-conditioning/result.json');save(O/'queue-state.json',q)
    print(json.dumps(dict(seconds=result['seconds'],checks=checks,groups={k:dict(tracks=v['tracks'],angle=v['endpoint_angle_degrees'],K=v['linear']['robust2px']) for k,v in summaries.items() if k in ['fit_all','fit_ceiling','fit_floor']})),flush=True)

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','run']
    globals()[sys.argv[1]]()

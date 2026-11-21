"""N6: one shared residual-radial output-domain hypothesis, not raw fisheye reprocessing."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import time,json,traceback
import numpy as np
from scipy.sparse import coo_matrix,hstack
from scipy.sparse.linalg import lsmr
from scipy.optimize import minimize_scalar
from audit_shared_solver122 import Problem,huber_cost
from repair_shared_solver122 import schur_step
import shared_background122 as old

O=old.B/'residual-radial';P=old.B/'residual-radial-plan.json';BASE=old.O/'solver-repair'
def distort(uv,K,k):
    c=K[:2,2];f=np.diag(K)[:2];xy=(np.asarray(uv)-c)/f
    return c+f*xy*(1+k*np.sum(xy**2,axis=-1,keepdims=True))
def inverse(uv,K,k):
    c=K[:2,2];f=np.diag(K)[:2];xy=(np.asarray(uv)-c)/f;rd=np.linalg.norm(xy,axis=-1,keepdims=True);r=rd.copy()
    for _ in range(12):r-=(r+k*r**3-rd)/(1+3*k*r**2)
    return c+f*xy*np.divide(r,rd,out=np.ones_like(r),where=rd>1e-15)

class RadialProblem:
    def __init__(self):
        self.p=Problem();x=np.load(BASE/'fit-diagnostic.npz')['parameters'];n=self.p.ncam
        self.x0=np.r_[x[:n],0.,x[n:]];self.ncam=n+1;self.n0=n;self.K=self.p.K;self.frames=self.p.frames
        self.indices=self.p.indices;self.groups=self.p.groups;self.target=self.p.target;self.tids=self.p.tids
        xy=(np.array([[0,0],[1519,0],[0,2703],[1519,2703]])-self.K[:2,2])/np.diag(self.K)[:2]
        self.k_bound=.05/np.max(np.sum(xy**2,axis=1))
    def base(self,x):return np.r_[x[:self.n0],x[self.ncam:]]
    def decode(self,x):return self.p.decode(self.base(x))
    def residual(self,x):
        uv=self.p.residual(self.base(x)).reshape(-1,2)+self.target
        return (distort(uv,self.K,x[self.n0])-self.target).ravel()
    def jacobian(self,x):
        base=self.base(x);uv=self.p.residual(base).reshape(-1,2)+self.target
        xy=(uv-self.K[:2,2])/np.diag(self.K)[:2];a,b=xy.T;r2=a*a+b*b;k=x[self.n0]
        d00=1+k*(r2+2*a*a);d11=1+k*(r2+2*b*b);d01=2*k*a*b*self.K[0,0]/self.K[1,1];d10=2*k*a*b*self.K[1,1]/self.K[0,0]
        j=self.p.jacobian(base).tocoo();ri=j.row//2;isx=j.row%2==0
        rr=np.r_[2*ri,2*ri+1];cc=np.r_[j.col,j.col];dd=np.r_[j.data*np.where(isx,d00[ri],d01[ri]),j.data*np.where(isx,d10[ri],d11[ri])]
        warped=coo_matrix((dd,(rr,cc)),shape=j.shape).tocsr()
        dk=(xy*r2[:,None]*np.diag(self.K)[:2]).ravel()
        return hstack([warped[:,:self.n0],coo_matrix(dk[:,None]),warped[:,self.n0:]],format='csr')

def evaluate(p,cams,tracks,k):
    errors=[]
    for t in tracks:
        obs=[o for o in t['observations'] if o['frame'] in cams]
        for target in obs:
            others=[(o['frame'],inverse(o['uv'],p.K,k)) for o in obs if o['frame']!=target['frame']]
            row=dict(track=t['track'],family=t.get('family'),target=target['frame'],support_frames=[f for f,_ in others],observed=target['uv'],valid_projection=False,error_px=None)
            if len(others)>=2:
                X=old.triangulate(p.K,cams,others)
                if np.isfinite(X).all():
                    uv,z=old.project(p.K,cams[target['frame']],X[None]);uv=distort(uv,p.K,k)
                    if z[0]>0 and all(old.project(p.K,cams[f],X[None])[1][0]>0 for f,_ in others):
                        row.update(valid_projection=True,predicted=uv[0].tolist(),delta=(uv[0]-target['uv']).tolist(),error_px=float(np.linalg.norm(uv[0]-target['uv'])))
            errors.append(row)
    return dict(summary=old.stats(errors),observations=errors)

def register():
    assert not P.exists() and not O.exists()
    base=old.read(BASE/'result.json');assert base['optimizer']['accepted_steps']>0
    old.save(P,dict(stage='N6_output_residual_radial',registered=old.now(),source_sha256=old.sha(Path(__file__)),
        user_steering='Test that the already rectified exported video may retain distortion; do not assume exact pinhole. ApproximateK is allowed for coarse diagnosis.',
        hypothesis='One time-invariant radial residual in exported MP4 pixels. u=c+f*[x,y]*(1+k1*r^2). k1 is NOT rawFISHEYE8 and not a second image undistortion.',
        causal_variable='Add only one shared output-domain radial coefficient; K,principalpoint,frames,materials,fit/holdout,gauge,Huber1px fixed. R/t andFIT3D reoptimized as identical nuisance parameters in both models. Not a unique causal proof.',
        initialization='Completed corrected N5c samecandidate; coefficient0. No old121/119 source, no face fit.',
        prior_controls=['Analytic derivatives versus central differences at0 and halfbound','Forward/inverse point roundtrip on25imagegrid','Frozen-geometry k1-onlyFIT diagnostic','Weighted local identifiability versus pose/point nuisance space'],
        identifiability='Fit-only weighted radialJac projected off nuisanceJ withLSMR. If retained norm ratio<0.001, stop jointfit as locally unresolved. This is an engineering rank screen, not physical calibration admission.',
        bounds='|k1| <=0.05/maxr2 of imagecorners: at most5percent corner radialscale; positive radial derivative. Boundhit means unsupported, do not expand.',
        budget=dict(candidates=1,joint_iterations=80,damping_trials=8,wall_seconds=180,GPU=0),
        validation='Same all681background and57uncertifiedface target-excluded3D observations, preserved noerrorfilter; output-space Euclideanpx. No threshold/model selection on validation. These datasets already informed development, not unseen testdata.',
        stop='Numeric failure, poor rank or boundhit -> preserve evidence. No higher polynomial/rayfield added automatically. No training or camera promotion.',
        source_inputs={str(f):old.sha(f) for f in [BASE/'result.json',BASE/'fit-diagnostic.npz',old.B/'observations.json',old.I/'native-temporal-material/observations.npz']}))
    print('N6 registered')

def run():
    plan=old.read(P);assert not O.exists();O.mkdir();start=time.monotonic()
    for f,h in plan['source_inputs'].items():assert old.sha(Path(f))==h
    p=RadialProblem();x=p.x0.copy();controls=[];rng=np.random.default_rng(1226)
    for k in [0,.5*p.k_bound]:
        q=x.copy();q[p.n0]=k;j=p.jacobian(q);v=rng.normal(size=len(q));v[p.n0]=.01;v/=np.max(abs(v))
        h=1e-6;ref=(p.residual(q+h*v)-p.residual(q-h*v))/(2*h);actual=j@v
        relative=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref));assert relative<1e-6
        controls.append(dict(name='analytic_derivative',k=k,relative_l2=relative))
    grid=np.array([[a,b] for a in np.linspace(0,1519,5) for b in np.linspace(0,2703,5)])
    for k in [-p.k_bound,p.k_bound]:
        err=float(np.max(abs(inverse(distort(grid,p.K,k),p.K,k)-grid)));assert err<1e-8
        controls.append(dict(name='ray_forward_inverse',k=k,max_pixel_error=err))
    j=p.jacobian(x);r=p.residual(x);w=np.sqrt(np.minimum(1,1/np.maximum(abs(r),1e-30)))
    a=hstack([j[:,:p.n0],j[:,p.ncam:]],format='csr').multiply(w[:,None]).tocsr();b=np.asarray(j[:,p.n0].toarray()).ravel()*w
    norms=np.sqrt(np.asarray(a.power(2).sum(axis=0))).ravel();a=a.multiply(1/np.maximum(norms,1e-12)[None,:]).tocsr()
    solved=lsmr(a,b,atol=1e-12,btol=1e-12,maxiter=5000);remaining=b-a@solved[0];ratio=float(np.linalg.norm(remaining)/np.linalg.norm(b))
    def scalar(k):q=x.copy();q[p.n0]=k;return huber_cost(p.residual(q))
    one=minimize_scalar(scalar,bounds=(-p.k_bound,p.k_bound),method='bounded',options={'xatol':1e-10,'maxiter':60})
    diag=dict(local_radial_retained_ratio=ratio,lsmr_status=int(solved[1]),lsmr_iterations=int(solved[2]),weighted_total_norm=float(np.linalg.norm(b)),weighted_profiled_norm=float(np.linalg.norm(remaining)),
        frozen_geometry_k1=float(one.x),frozen_geometry_cost=float(one.fun),pinhole_cost=scalar(0),k_bound=p.k_bound)
    old.save(O/'fit-identifiability.json',diag);old.save(O/'controls.json',dict(controls=controls))
    if ratio<.001:
        old.save(O/'result.json',dict(state='insufficient_local_identifiability',diagnostic=diag,training_started=False));print(json.dumps(diag));return
    cost=scalar(0);initial_cost=cost;damping=.001;history=[];reason='max_iterations';boundhit=False
    for it in range(80):
        if time.monotonic()-start>180:reason='wall_budget';break
        r=p.residual(x);j=p.jacobian(x);accepted=False
        for trial in range(8):
            dx,g,d=schur_step(j,r,p.ncam,damping);q=x+dx
            if abs(q[p.n0])>p.k_bound:
                damping*=10;boundhit=True;continue
            cams,xyz=p.decode(q);positive=all(np.all(old.project(p.K,cams[f],xyz[p.indices[ix]])[1]>0) for f,ix in p.groups.items())
            rr=p.residual(q);tc=huber_cost(rr) if np.isfinite(rr).all() and positive else float('inf')
            if tc<cost:prev=cost;x=q;cost=tc;accepted=True;used=damping;damping=max(damping/3,1e-12);break
            damping*=10
        history.append(dict(iteration=it,accepted=accepted,cost=cost,k1=float(x[p.n0]),gradient_inf=float(max(abs(g))),damping=float(used if accepted else damping),trials=trial+1))
        old.save(O/'progress.json',dict(state='running',pid=os.getpid(),updated=old.now(),last=history[-1]))
        if not accepted:reason='no_descent_with_registered_damping_budget';break
        if max(abs(dx))<1e-9 or max(abs(g))<1e-5:reason='numerical_stationarity';break
        if prev-cost<1e-9*max(1,prev):reason='relative_cost_change';break
    cams,xyz=p.decode(x);k=float(x[p.n0]);raw=old.read(old.B/'observations.json')['tracks'];held=[t for t in raw if t['split']=='heldout']
    face=np.load(old.I/'native-temporal-material/observations.npz')
    ft=[dict(track=t,family='N1_face_never_camera_fit',observations=[dict(frame=f,uv=face['uv'][f,t].tolist()) for f in p.frames if face['valid'][f,t]]) for t in range(9)]
    validation=dict(background=evaluate(p,cams,held,k),face_candidates=evaluate(p,cams,ft,k))
    delta=np.linalg.norm(distort(grid,p.K,k)-grid,axis=1)
    result=dict(state='completed_pending_review',updated=old.now(),elapsed_seconds=time.monotonic()-start,stage='N6',
        K_integer_centers=p.K.tolist(),K_optimized=False,k1=k,k_bound=float(p.k_bound),any_trial_crossed_bound=bool(boundhit),final_at_bound=bool(abs(k)>=.999*p.k_bound),
        optimizer=dict(reason=reason,initial_cost=initial_cost,final_cost=cost,gradient_inf=float(max(abs(p.jacobian(x).T@np.clip(p.residual(x),-1,1)))),history=history),
        diagnostic=diag,cameras=[dict(frame=f,w2c=cams[f].tolist()) for f in p.frames],validation=validation,
        fixed_ray_pixel_displacement=dict(grid25_max=float(delta.max()),grid25_median=float(np.median(delta))),
        actual_registered_frames=p.frames,physical_material_certified=False,all_training_frames_registered=False,training_started=False,
        limitations='Shared1parameter export-domain radial model only. Background support includes uncertain distant material and nearplane. Face candidates are not certifiedGT. No rawfisheye orimagewarp, nofinaladmission.')
    np.savez_compressed(O/'fit-diagnostic.npz',parameters=x,points=xyz,track_ids=p.tids);old.save(O/'result.json',result)
    old.save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=old.now()))
    print(json.dumps(dict(k1=k,cost=cost,diagnostic=diag,grid=result['fixed_ray_pixel_displacement'],validation={k:v['summary'] for k,v in validation.items()})),flush=True)
def repair_serialization():
    global O
    failed=O
    error=old.read(failed/'failure.json');assert 'bool_' in error['error']
    assert not (failed/'result.json').exists() and not (failed/'fit-diagnostic.npz').exists()
    assert not (failed/'serialization-repair').exists()
    old.save(failed/'serialization-repair-plan.json',dict(updated=old.now(),failure=error['error'],
        repair='Convert NumPy bool to Pythonbool and persist finalNPZ beforeJSON. No objective/model/solver/data/seed/budget change. Original computation completed but finalvector was lost atserialization; deterministic replay is necessary for recovery, not a new candidate.',
        old_source_sha256=old.sha(failed/'source-before-json-repair.py'),new_source_sha256=old.sha(Path(__file__)),
        original_final_progress=old.read(failed/'progress.json'),same_input_hashes=old.read(P)['source_inputs']))
    O=failed/'serialization-repair';run()
if __name__=='__main__':
    import sys
    try:globals()[sys.argv[1]]()
    except Exception as ex:
        if O.exists():old.save(O/'failure.json',dict(updated=old.now(),error=repr(ex),traceback=traceback.format_exc()))
        raise

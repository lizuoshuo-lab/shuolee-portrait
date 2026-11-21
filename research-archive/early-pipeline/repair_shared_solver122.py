"""N5c numerical repair: same Huber objective/FIT/K/gauge, explicit damped Schur IRLS."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import time,json,traceback
import numpy as np
from scipy.sparse import diags
from audit_shared_solver122 import Problem,huber_cost
import shared_background122 as old

O=old.O/'solver-repair';P=old.O/'solver-repair-plan.json'
def schur_step(j,r,ncam,damping):
    w=np.minimum(1.,1/np.maximum(abs(r),1e-30));a=j.multiply(np.sqrt(w)[:,None]).tocsr();b=r*np.sqrt(w)
    h=(a.T@a).tocsr();g=np.asarray(a.T@b).ravel();d=np.maximum(h.diagonal(),1e-12)
    h=h+diags(damping*d);hc=h[:ncam,:ncam].toarray();hp=h[:ncam,ncam:].toarray().reshape(ncam,-1,3)
    blocks=np.array([h[k:k+3,k:k+3].toarray() for k in range(ncam,h.shape[0],3)])
    inv=np.linalg.inv(blocks);gp=g[ncam:].reshape(-1,3)
    proj=np.einsum('cni,nij->cnj',hp,inv)
    s=hc-np.einsum('cni,dni->cd',proj,hp)
    rhs=-g[:ncam]+np.einsum('cni,ni->c',proj,gp)
    dc=np.linalg.solve(s,rhs)
    dp=-np.einsum('nij,nj->ni',inv,gp+np.einsum('cni,c->ni',hp,dc))
    return np.r_[dc,dp.ravel()],g,d

def linear_control():
    # Independent dense solution to a small block-sparse normal system.
    from scipy.sparse import csr_matrix
    rng=np.random.default_rng(122);j=np.zeros((16,8))
    j[:,:2]=rng.normal(size=(16,2));j[:8,2:5]=rng.normal(size=(8,3));j[8:,5:8]=rng.normal(size=(8,3));r=rng.normal(size=16)
    actual,_,_=schur_step(csr_matrix(j),r,2,.001)
    w=np.minimum(1,1/np.maximum(abs(r),1e-30));a=j*np.sqrt(w)[:,None];h=a.T@a;g=a.T@(r*np.sqrt(w))
    expected=np.linalg.solve(h+.001*np.diag(np.maximum(np.diag(h),1e-12)),-g)
    error=float(max(abs(expected-actual)));assert error<1e-10
    return error

def register():
    assert not P.exists() and not O.exists()
    a=old.read(old.O/'numerical-audit/result.json');assert a['small_descent_cost']<a['cost'] and a['gradient_inf']>1e5
    assert max(r['relative_l2'] for r in a['analytic_central_directional'])<1e-6
    old.save(P,dict(stage='N5c_numerical_repair',registered=old.now(),source_sha256=old.sha(Path(__file__)),
        failure='Original xtol status3 stopped after negligible change, gradient_inf155187; explicit small step lowers exactHuber cost. Jacobian error relative3.25e-8, so no evidence of projection derivative bug. Numerical solver termination failed; exact internal cause unproven.',
        unchanged=['All935FIT tracks and frozen whole-family splits','Same completeK and7 frames','Same projection/residual andcomponentwiseHuber1px','Same firstcamera/scale gauge','No heldout/face inoptimization'],
        repair='AnalyticJ checked against independent numerical derivatives; fixed damped Schur IRLS, exactHuber descent accepted only. This is solver repair of the same candidate, not a new physical camera hypothesis.',
        initialization='Saved stalledN5c parameter vector, effectively freshN5 initialization; no121/119 pose orgeometry. Preserve original result.',
        input_hashes={str(f):old.sha(f) for f in [old.O/'fit-diagnostic.npz',old.B/'observations.json',old.O/'result.json',old.O/'numerical-audit/result.json']},
        budget=dict(repaired_runs=1,max_iterations=80,max_damping_trials_per_iteration=8,wall_seconds=180,GPU=0),
        damping_rule='lambda starts0.001; exactHuber decrease accept anddivide3, reject multiply10; no model/threshold search or heldout selection.',
        output='Repaired optimizer evidence plus same target-excluded3D validation; no training/promotion.'))
    print('N5c solver repair registered')

def run():
    assert not O.exists();plan=old.read(P);O.mkdir();start=time.monotonic()
    for f,h in plan['input_hashes'].items():assert old.sha(Path(f))==h
    linear_error=linear_control();p=Problem();x=p.x0.copy();cost=huber_cost(p.residual(x));initial_cost=cost;damping=.001;history=[];reason='max_iterations'
    for it in range(80):
        if time.monotonic()-start>180:reason='wall_budget';break
        r=p.residual(x);j=p.jacobian(x);accepted=False
        for trial in range(8):
            dx,g,d=schur_step(j,r,p.ncam,damping);trial_r=p.residual(x+dx)
            cameras,xyz=p.decode(x+dx)
            positive=all(np.all(old.project(p.K,cameras[f],xyz[p.indices[ix]])[1]>0) for f,ix in p.groups.items())
            tc=huber_cost(trial_r) if np.isfinite(trial_r).all() and positive else float('inf')
            if tc<cost:
                prev=cost;x+=dx;cost=tc;accepted=True;used=damping;damping=max(damping/3,1e-12);break
            damping*=10
        history.append(dict(iteration=it,accepted=accepted,cost=cost,gradient_inf=float(np.max(abs(g))),damping=float(used if accepted else damping),trials=trial+1,
                            step_max=float(max(abs(dx))),elapsed_seconds=time.monotonic()-start))
        old.save(O/'progress.json',dict(state='optimizing',pid=os.getpid(),updated=old.now(),last=history[-1]))
        print(json.dumps(history[-1]),flush=True)
        if not accepted:reason='no_descent_with_registered_damping_budget';break
        if max(abs(dx))<1e-9 or np.max(abs(g))<1e-5:reason='numerical_stationarity';break
        if prev-cost<1e-9*max(1,prev):reason='relative_cost_change';break
    cams,xyz=p.decode(x);raw=old.read(old.B/'observations.json')['tracks'];held=[t for t in raw if t['split']=='heldout']
    face=np.load(old.I/'native-temporal-material/observations.npz')
    tracks=[dict(track=t,family='N1_face_never_camera_fit',observations=[dict(frame=f,uv=face['uv'][f,t].tolist()) for f in p.frames if face['valid'][f,t]]) for t in range(9)]
    final_gradient=np.max(abs(p.jacobian(x).T@np.clip(p.residual(x),-1,1)))
    result=dict(stage='N5c_solver_repair',updated=old.now(),elapsed_seconds=time.monotonic()-start,
        K_integer_centers=p.K.tolist(),K_optimized=False,actual_registered_frames=p.frames,
        optimizer=dict(reason=reason,initial_cost=initial_cost,final_cost=cost,final_gradient_inf=float(final_gradient),accepted_steps=sum(h['accepted'] for h in history),parameter_max_change=float(max(abs(x-p.x0))),history=history),
        controls=dict(schur_dense_difference=linear_error,analytic_directional_audit='numerical-audit/result.json'),
        cameras=[dict(frame=f,w2c=cams[f].tolist()) for f in p.frames],
        validation=dict(background=old.evaluate(p.K,cams,held),face_candidates=old.evaluate(p.K,cams,tracks)),
        physical_material_certified=False,all_training_frames_registered=False,training_started=False,
        limitations='Near-ground dominant support and ambiguous distanttracks. Face rawN1 candidates are not certifiedmaterials. All heldout observations retained, noerrorfilter.')
    old.save(O/'result.json',result);np.savez_compressed(O/'fit-diagnostic.npz',parameters=x,points=xyz,track_ids=p.tids)
    old.save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=old.now()))
    print(json.dumps(dict(reason=reason,cost=cost,gradient=float(final_gradient),validation={k:v['summary'] for k,v in result['validation'].items()})),flush=True)
if __name__=='__main__':
    import sys
    try:globals()[sys.argv[1]]()
    except Exception as e:
        if O.exists():old.save(O/'failure.json',dict(updated=old.now(),error=repr(e),traceback=traceback.format_exc()))
        raise

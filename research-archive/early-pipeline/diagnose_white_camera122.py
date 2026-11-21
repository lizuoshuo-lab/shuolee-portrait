"""N12h: read-only fixed-K background information and LOO transfer sensitivity."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, time, sys, traceback
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from audit_shared_solver122 import Problem, huber_cost
from shared_background122 import project, triangulate, camera_center

I=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
D=I/'clip03-fresh-geometry'; F=I/'clip03-roof-review'; G=I/'clip03-grid-coverage/array-cache-repair'
O=I/'clip03-camera-observability'; P=I/'clip03-camera-observability-plan.json'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def register():
    assert not O.exists() and not P.exists()
    inputs=[D/'diagnostic-structure.npz',D/'result.json',D/'validation.json',D/'visual-review.json',
            F/'reviewed-support.json',G/'visual-index.json',G/'visual-review.json',I/'clip03-fresh-geometry-plan.json']
    sources=[Path(__file__),Path(__file__).parent/'audit_shared_solver122.py',Path(__file__).parent/'shared_background122.py']
    save(P,dict(stage='N12h',registered=now(),
        hypothesis='With K and all observations frozen, weak background-only camera modes might transfer differently to head and feet. Quantify local sensitivity before attributing N12g face residuals to movement or changing calibration.',
        variable='Read-only linearization at the completed whiteT N12g candidate. No refit, new feature, K/pose/point update, residual-selected observation, or candidate export.',
        scope='Original216 gauge parameters from diagnostic-structure.npz. Its separately rebased184 points/cameras are used only for a projection-equivalence control. Never mix these gauges.',
        information='All40 frozen FIT physical families, equal per-coordinate unit variance, fixedK and fixed216 pose/unit216-232 baseline. Eliminate120 XYZ nuisance dimensions using QR/SVD projection; camera information35 dimensions. Report raw and column-normalized spectra and numerical rank without ridge.',
        transfer='All15 heldout background,6 coarse face and7 shoe/sock families. For each target, re-triangulate its other observations as in N12g. Compute camera derivative and independent localization derivative. Preserve every target. FIT-only information matrix and three weakest modes saved before validation targets are loaded.',
        noise_interpretation='Unit noise is a sensitivity convention, not measured uncertainty or a confidence interval. Camera covariance gain and independent test-track localization gain reported separately. Correlated bias, wrong material identity, uncertainK, noncentral camera, motion and model error are excluded from this local random-noise model.',
        controls='Full analytic FIT Jacobian versus central differences h=1e-6; gauge projection and baseline residual equality; Schur camera covariance versus full joint SVD; camera transfer h=1e-6 versus h=5e-7. Three FIT-derived weakest modes: nonlinear centered check at max parameter displacement1e-5, no face-residual mode fitting.',
        rank_tolerance='max(matrix.shape)*machineepsilon*largest singular value; report singular values. No regularization that hides null directions. Rank failure terminates with unbounded uncertainty rather than pseudo-certain covariance.',
        geometry='AllFIT triangulation depth and parallax grouped by reviewed ground/nonground, baseline-relative only; no physical metric scale claim.',
        budget=dict(runs=1,CPU_wall_seconds=120,GPU=0,training=0),
        stop='Any hash, identity, gauge, numerical control or budget failure -> preserve and stop. Fullrank/good numerical control cannot admit training or prove physical calibration. No parameter sweep or automatic follow-on fit.',
        input_hashes={str(p):sha(p) for p in inputs},code_hashes={str(p):sha(p) for p in sources},training_started=False))
    print('N12H_REGISTERED',flush=True)

def rank(s, shape): return int(np.sum(s>max(shape)*np.finfo(float).eps*s[0]))
def stats(a):
    a=np.array(a,dtype=float)
    return dict(min=float(a.min()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max()))

def build_problem(K,order,fit,ids,x):
    p=Problem.__new__(Problem) # Deliberately do not load historic hat data in __init__.
    p.K=K;p.frames=order;p.tids=ids;p.slices={};n=0
    for i,f in enumerate(order[1:],1):
        m=5 if i==1 else 6;p.slices[f]=slice(n,n+m);n+=m
    p.ncam=n;p.x0=x.copy();lookup={int(t):j for j,t in enumerate(ids)}
    obs=[(o['frame'],lookup[t['track']],o['uv']) for t in fit for o in t['observations']]
    p.target=np.array([uv for _,_,uv in obs]);p.indices=np.array([j for _,j,_ in obs])
    p.groups={f:np.flatnonzero([ff==f for ff,_,_ in obs]) for f in order}
    p.sparsity=coo_matrix((2*len(obs),len(x))).tocsr()
    assert n==35 and len(ids)==40 and len(x)==155
    return p,obs

def track_residual(K,cams,t):
    obs=t['observations'];rows=[]
    for target in obs:
        others=[(o['frame'],o['uv']) for o in obs if o['frame']!=target['frame']]
        assert len(others)>=2
        X=triangulate(K,cams,others)
        assert np.isfinite(X).all() and all(project(K,cams[o['frame']],X[None])[1][0]>0 for o in obs)
        rows.append(project(K,cams[target['frame']],X[None])[0][0]-target['uv'])
    return np.array(rows).ravel()

def run():
    assert not O.exists();plan=read(P)
    for path,h in {**plan['input_hashes'],**plan['code_hashes']}.items(): assert sha(Path(path))==h,path
    O.mkdir();started=time.monotonic();cv2.setNumThreads(1)
    def progress(state):
        assert time.monotonic()-started<120,'Registered120s budget exceeded'
        save(O/'progress.json',dict(state=state,pid=os.getpid(),updated=now(),elapsed_seconds=time.monotonic()-started))
    progress('FIT_only_information')
    base=read(D/'result.json');sourceplan=read(I/'clip03-fresh-geometry-plan.json')
    support=read(F/'reviewed-support.json')['tracks'];fit=[t for t in support if t['split']=='fit']
    assert len(fit)==40 and len(set(t['family'] for t in fit))==40
    archive=np.load(D/'diagnostic-structure.npz');x=archive['parameters'].copy();ids=archive['track_ids']
    assert sorted(ids.tolist())==sorted(t['track'] for t in fit)
    order=sourceplan['initial_pair']+sourceplan['registration_order'];K=np.array(base['K_integer_centers'])
    p,obs=build_problem(K,order,fit,ids,x);cams,xyz=p.decode(x)
    M184=cams[184];outcams={r['frame']:np.array(r['w2c']) for r in base['cameras']}
    gauge=max(np.max(abs(project(K,cams[f],xyz)[0]-project(K,outcams[f],archive['points'])[0])) for f in order)
    assert gauge<1e-7 and np.max(abs(archive['points']-(xyz@M184[:3,:3].T+M184[:3,3])))<1e-8
    residual=p.residual(x);cost=huber_cost(residual);assert abs(cost-base['optimizer']['final_cost'])<1e-8
    J=p.jacobian(x).toarray();h=1e-6
    Jn=np.column_stack([(p.residual(x+np.eye(1,len(x),i)[0]*h)-p.residual(x-np.eye(1,len(x),i)[0]*h))/(2*h) for i in range(len(x))])
    jerr=np.linalg.norm(J-Jn)/np.linalg.norm(Jn);assert jerr<1e-6
    B=J[:,:p.ncam];Q=J[:,p.ncam:];uq,sq,vq=np.linalg.svd(Q,full_matrices=False)
    assert rank(sq,Q.shape)==Q.shape[1], 'Point structure numerical rank deficient; no bounded camera covariance claimed'
    Qplus=(vq.T/sq)@uq.T;A=B-uq@(uq.T@B)
    scale=1/np.linalg.norm(B,axis=0);ua,sa,va=np.linalg.svd(A*scale,full_matrices=False)
    info=dict(stage='N12h',updated=now(),FIT_families=40,FIT_targets=len(obs),dimensions=dict(camera=p.ncam,points=Q.shape[1],residual=J.shape[0]),
        singular_values_camera_column_normalized=sa.tolist(),singular_values_camera_raw=np.linalg.svd(A,compute_uv=False).tolist(),
        numerical_rank=rank(sa,A.shape),rank_threshold=float(max(A.shape)*np.finfo(float).eps*sa[0]),
        scaled_condition=float(sa[0]/sa[-1]),gauge='216identity,216->232unitbaseline,35cameraDOF',noise='Unit independent FIT pixel-coordinate noise; sensitivity only',
        controls=dict(gauge_projection_max_px=float(gauge),base_huber_cost=cost,full_analytic_jacobian_relative=float(jerr)),
        validation_used_to_construct_information=False)
    save(O/'FIT-information.json',info)
    assert info['numerical_rank']==35,'Camera rank deficient; null directions not regularized'
    # Square root of covariance from FIT only. No heldout/person observations enter it.
    L=(scale[:,None]*va.T)/sa
    point_response=-Qplus@B@L
    ufull,sfull,vfull=np.linalg.svd(J,full_matrices=False)
    assert rank(sfull,J.shape)==len(x)
    Lfull=vfull.T/sfull;Cfull=Lfull[:p.ncam]@Lfull[:p.ncam].T;C=L@L.T
    cerr=np.linalg.norm(C-Cfull)/np.linalg.norm(Cfull);assert cerr<1e-5
    unitcost=np.linalg.norm(A@L,axis=0);assert max(abs(unitcost-1))<1e-7
    info['controls'].update(joint_covariance_relative=float(cerr),mode_unit_FIT_L2_max_error=float(max(abs(unitcost-1))))
    save(O/'FIT-information.json',info)
    np.savez_compressed(O/'FIT-linearization.npz',J=J,camera_covariance=C,camera_unit_modes=L,point_responses=point_response,singular_values=sa)
    progress('frozen_validation_transfer')
    visual=read(G/'visual-index.json');review=read(G/'visual-review.json')
    held=[t for t in support if t['split']=='heldout']
    face=[visual['rows'][r['row']] for r in review['rows'] if r['reviewed_material']=='face_surface_coarse']
    feet=[visual['rows'][r['row']] for r in review['rows'] if r['reviewed_material'] in ['shoe','sock']]
    assert not set(t['family'] for t in held)&set(t['family'] for t in fit)
    assert not set(t['family'] for t in face+feet)&set(t['family'] for t in fit)
    tracks=[dict(t,analysis_group=g) for g,ts in [('background',held),('face',face),('feet',feet)] for t in ts]
    assert [len(held),len(face),len(feet)]==[15,6,7]
    def values(xx):
        cc,_=p.decode(xx)
        return np.concatenate([track_residual(K,cc,t) for t in tracks])
    r0=values(x)
    def deriv(step):
        columns=[]
        for c in range(p.ncam):
            delta=np.zeros_like(x);delta[c]=step
            columns.append((values(x+delta)-values(x-delta))/(2*step))
        return np.column_stack(columns)
    H=deriv(1e-6);Hhalf=deriv(5e-7);herr=np.linalg.norm(H-Hhalf)/np.linalg.norm(Hhalf);assert herr<1e-5
    transfer=H@L;rows=[];start=0;saved=read(D/'validation.json')['final']
    for t in tracks:
        m=len(t['observations']);local=np.zeros((2*m,2*m))
        uv=np.array([o['uv'] for o in t['observations']])
        for c in range(2*m):
            results=[]
            for sign in [1,-1]:
                v=uv.copy();v.ravel()[c]+=sign*1e-3
                tt=dict(t,observations=[dict(o,uv=v[i].tolist()) for i,o in enumerate(t['observations'])])
                results.append(track_residual(K,cams,tt))
            local[:,c]=(results[0]-results[1])/.002
        groupname={'background':'background','face':'face_all_diagnostic','feet':'feet_diagnostic'}[t['analysis_group']]
        expected={(z['family'],z['target']):z for z in saved[groupname]['observations']}
        for j,o in enumerate(t['observations']):
            si=slice(start+2*j,start+2*j+2);cg=transfer[si]@transfer[si].T;lg=local[2*j:2*j+2]@local[2*j:2*j+2].T
            delta=r0[si];e=expected[t['family'],o['frame']];assert max(abs(delta-np.array(e['delta'])))<1e-6
            rows.append(dict(group=t['analysis_group'],family=t['family'],split=t['split'],frame=o['frame'],nonground=t.get('nonground'),
                observed=o['uv'],residual=delta.tolist(),error_px=float(np.linalg.norm(delta)),
                camera_unit_sigma=dict(x=float(np.sqrt(cg[0,0])),y=float(np.sqrt(cg[1,1])),major=float(np.sqrt(np.linalg.eigvalsh(cg)[-1])),radial_rms=float(np.sqrt(np.trace(cg)))),
                test_track_unit_sigma=dict(x=float(np.sqrt(lg[0,0])),y=float(np.sqrt(lg[1,1])),major=float(np.sqrt(np.linalg.eigvalsh(lg)[-1])),radial_rms=float(np.sqrt(np.trace(lg)))),
                weakest3_unit_FIT_L2_vectors=transfer[si,-3:].T.tolist()))
        start+=2*m
        progress('frozen_validation_transfer')
    assert start==len(r0)
    checks=[]
    for k in range(3):
        col=-1-k;direction=np.r_[L[:,col],point_response[:,col]]
        amp=1e-5/max(abs(direction));dx=direction*amp
        fit_numeric=(p.residual(x+dx)-p.residual(x-dx))/(2*amp)
        fit_exact=J@direction;val_numeric=(values(x+dx)-values(x-dx))/(2*amp);val_exact=transfer[:,col]
        fe=np.linalg.norm(fit_numeric-fit_exact)/np.linalg.norm(fit_exact)
        ve=np.linalg.norm(val_numeric-val_exact)/np.linalg.norm(val_exact)
        assert fe<1e-3 and ve<1e-3
        checks.append(dict(weakness_order=k+1,amplitude=float(amp),max_parameter_change=float(max(abs(dx))),
            FIT_directional_relative=float(fe),validation_directional_relative=float(ve),
            FIT_linear_L2_per_unit=float(np.linalg.norm(fit_exact)),
            per_group={g:stats(np.linalg.norm(val_exact.reshape(-1,2)[[i for i,z in enumerate(rows) if z['group']==g]],axis=1)) for g in ['background','face','feet']}))
    geom=[]
    lookup={int(t):j for j,t in enumerate(ids)}
    for t in fit:
        X=xyz[lookup[t['track']]];rays=[X-camera_center(cams[o['frame']]) for o in t['observations']]
        angles=[np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b))) for i,a in enumerate(rays) for b in rays[i+1:]]
        depths=[project(K,cams[o['frame']],X[None])[1][0] for o in t['observations']]
        geom.append(dict(family=t['family'],nonground=t['nonground'],material=t['material'],observations=len(depths),median_depth_baseline_units=float(np.median(depths)),max_angle_degrees=float(max(angles))))
    groups={g:[z for z in rows if (z['group']==g if g in ['background','face','feet'] else z['group']=='face' and z['split']=='heldout')] for g in ['background','face','feet','face_heldout']}
    summary={g:dict(n=len(rs),error_px=stats([z['error_px'] for z in rs]),camera_unit_sigma_major=stats([z['camera_unit_sigma']['major'] for z in rs]),
        camera_unit_sigma_y=stats([z['camera_unit_sigma']['y'] for z in rs]),test_track_unit_sigma_major=stats([z['test_track_unit_sigma']['major'] for z in rs]),
        per_frame=[dict(frame=f,n=len(a),median_residual_y=float(np.median([z['residual'][1] for z in a])),median_camera_unit_sigma_y=float(np.median([z['camera_unit_sigma']['y'] for z in a])),max_camera_unit_sigma_y=max(z['camera_unit_sigma']['y'] for z in a)) for f in sorted(order) if (a:=[z for z in rs if z['frame']==f])]) for g,rs in groups.items()}
    progress('preserve_read_only_result')
    for path,h0 in plan['input_hashes'].items():assert sha(Path(path))==h0
    save(O/'target-sensitivities.json',dict(noise_scale='Separate unit1px independent noise on FIT and test-track coordinates; gains are not measured error bars.',rows=rows))
    save(O/'structure-support.json',dict(rows=geom,groups={g:dict(depth=stats([z['median_depth_baseline_units'] for z in geom if z['nonground']==b]),parallax=stats([z['max_angle_degrees'] for z in geom if z['nonground']==b])) for g,b in [('ground',False),('nonground',True)]}))
    np.savez_compressed(O/'transfer-derivatives.npz',H=H,unit_modes_transfer=transfer,baseline_residual=r0)
    result=dict(stage='N12h',state='complete_pending_interpretation',updated=now(),elapsed_seconds=time.monotonic()-started,
        information=info,controls=dict(camera_transfer_step_agreement_relative=float(herr),weakest_modes=checks,input_hashes_unchanged=True),summary=summary,
        read_only=True,K_changed=False,poses_changed=False,points_changed=False,observations_changed=False,new_camera_candidate=False,training_started=False,physical_subpixel_accepted=False,
        limits=plan['noise_interpretation'])
    save(O/'result.json',result);save(O/'progress.json',dict(state=result['state'],pid=None,updated=now()))
    print(json.dumps(dict(elapsed=result['elapsed_seconds'],rank=info['numerical_rank'],condition=info['scaled_condition'],summary=summary)),flush=True)

if __name__=='__main__':
    try: globals()[sys.argv[1]]()
    except Exception as e:
        if O.exists():
            save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
            save(O/'progress.json',dict(state='failed_preserved',pid=None,updated=now()))
        raise

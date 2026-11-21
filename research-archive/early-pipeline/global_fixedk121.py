"""One bounded, fixed-K background SfM diagnostic from frozen native tracks.

No feature extraction, old cameras/geometry, image warp, training, or model promotion.
Background means outside conservative human ROIs, not independently proven static.
"""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, time, hashlib, argparse, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
sys.path.insert(0, '/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib

ROOT = Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121')
SUBJECTS = ['grass', 'sofa', 'portrait', 'indoor']

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)

def stats(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return {'n': len(a), 'median': float(np.median(a)), 'p90': float(np.percentile(a,90)), 'max': float(a.max()), 'fraction_le_3px': float(np.mean(a <= 3))} if len(a) else {'n':0}

def inside(p, box, pad=0):
    return box is not None and box[0]-pad <= p[0] <= box[2]+pad and box[1]-pad <= p[1] <= box[3]+pad

def region(o, roi):
    if inside(o['uv'], roi['face_box'], 12): return 'face_roi'
    if inside(o['uv'], roi['body_box'], 12): return 'body_roi'
    return 'background_candidate'

def input_data(subject):
    p = ROOT/subject/'roi-supplement/observations.json'
    obs = read(p); auditp = R/f'supplement-{subject}-track-family-audit.json'
    audit = read(auditp); retained = set(audit['retained_track_ids'])
    rois = {r['frame']:r for r in obs['rois']}
    frameids = {r['frame']:i for i,r in enumerate(obs['records'])}
    tracks = []
    for t in obs['tracks']:
        if t['track_id'] not in retained: continue
        assert t['split'] in ['fit', 'heldout']
        tt = dict(t); tt['by_frame'] = {frameids[o['frame']]:o for o in t['observations']}
        tt['regions'] = {frameids[o['frame']]:region(o,rois[o['frame']]) for o in t['observations']}
        tt['camera_fit_eligible'] = t['split']=='fit' and all(v=='background_candidate' for v in tt['regions'].values())
        tracks.append(tt)
    return obs, tracks, {'observations':sha(p), 'family_audit':sha(auditp)}

def apply_material_gate(subject, obs, tracks, plan):
    spec=plan['material_preflight'][subject]
    lower=spec.get('min_y_fraction')
    if lower is None: return
    h=obs['records'][0]['wh'][1]
    for t in tracks:
        is_background=all(v=='background_candidate' for v in t['regions'].values())
        allowed=is_background and all(o['uv'][1]>=lower*h for o in t['observations'])
        t['camera_fit_eligible']=t['camera_fit_eligible'] and allowed
        if is_background:
            t['regions']={k:('background_static_support_candidate' if allowed else 'background_excluded_material') for k in t['regions']}

def project(K, pose, X):
    rr, cc = pose
    x = np.atleast_2d(X) @ rr.T - (rr @ cc)[None]
    z = x[:,2]
    return (x[:,:2]/np.maximum(z[:,None],1e-12))*np.array([K[0,0],K[1,1]])+K[:2,2], z

def triangulate(K, poses, items, require_good=False):
    """Use the widest measured bearing angle; no test observation is supplied for LOO."""
    keys = sorted(k for k in items if k in poses)
    if len(keys)<2: return None
    inv = np.linalg.inv(K)
    rays = {k: poses[k][0].T @ (inv @ np.r_[items[k]['uv'],1.]) for k in keys}
    rays = {k:v/np.linalg.norm(v) for k,v in rays.items()}
    pairs = [(np.arccos(np.clip(rays[a]@rays[b],-1,1)),a,b) for j,a in enumerate(keys) for b in keys[j+1:]]
    angle,a,b = max(pairs)
    if np.degrees(angle)<.5: return None
    r0,c0=poses[a];r1,c1=poses[b]
    p0=K@np.c_[r0,-r0@c0];p1=K@np.c_[r1,-r1@c1]
    x=cv2.triangulatePoints(p0,p1,np.asarray(items[a]['uv']).reshape(2,1),np.asarray(items[b]['uv']).reshape(2,1))[:,0]
    if abs(x[3])<1e-12: return None
    x=x[:3]/x[3]
    if not np.isfinite(x).all(): return None
    errors=[]
    for k in keys:
        uv,z=project(K,poses[k],x)
        if z[0]<=0: return None
        errors.append(float(np.linalg.norm(uv[0]-items[k]['uv'])))
    if require_good and max(errors)>3.: return None
    return x, float(np.degrees(angle)), [a,b], errors

def components(n, edges):
    sets=[{i} for i in range(n)]
    for a,b in edges:
        ia=next(i for i,s in enumerate(sets) if a in s);ib=next(i for i,s in enumerate(sets) if b in s)
        if ia!=ib: sets[ia] |= sets[ib];sets.pop(ib)
    return [sorted(x) for x in sorted(sets,key=lambda s:min(s))]

def bundle(K, poses, points, tracks, seed, deadline):
    keys=sorted(poses); pids=sorted(points)
    camidx={k:i for i,k in enumerate(keys)}; ptidx={k:i for i,k in enumerate(pids)}
    base=np.array([np.r_[Rotation.from_matrix(poses[k][0]).as_rotvec(), poses[k][1]] for k in keys])
    # Exact similarity gauge: first full pose and one nonzero seed-center component.
    axis=int(np.argmax(np.abs(poses[seed][1])))
    assert abs(poses[seed][1][axis])>.1
    variable=np.ones(base.shape,dtype=bool);variable[camidx[0],:]=False;variable[camidx[seed],3+axis]=False
    camera_column=np.full(base.shape,-1,int);camera_column[variable]=np.arange(variable.sum())
    nc=int(variable.sum()); initial=np.r_[base[variable],np.array([points[k] for k in pids]).ravel()]
    obs=[]
    for tid in pids:
        for k,o in tracks[tid]['by_frame'].items():
            if k in poses: obs.append((camidx[k],ptidx[tid],o['uv']))
    ci=np.array([o[0] for o in obs]);pi=np.array([o[1] for o in obs]);uv=np.array([o[2] for o in obs])
    sparsity=lil_matrix((2*len(obs),len(initial)),dtype=np.int8)
    for j,(c,p,_) in enumerate(obs):
        cols=camera_column[c];cols=cols[cols>=0]
        sparsity[2*j:2*j+2,cols]=1;sparsity[2*j:2*j+2,nc+3*p:nc+3*p+3]=1
    def unpack(v):
        cams=base.copy();cams[variable]=v[:nc]
        return cams, v[nc:].reshape(-1,3)
    def residual(v):
        if time.monotonic()>deadline: raise TimeoutError('global fixed-K per-person CPU budget exceeded')
        cams,xyz=unpack(v);rot=Rotation.from_rotvec(cams[:,:3]).as_matrix()
        eye=np.einsum('nij,nj->ni',rot[ci],xyz[pi]-cams[ci,3:])
        pix=eye[:,:2]/np.maximum(eye[:,2,None],1e-10)*K[[0,1],[0,1]]+K[:2,2]
        return (pix-uv).ravel()
    before=stats(np.linalg.norm(residual(initial).reshape(-1,2),axis=1))
    fit=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),method='trf',tr_solver='lsmr',x_scale='jac',loss='soft_l1',f_scale=1.,max_nfev=80,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    cams,xyz=unpack(fit.x)
    newposes={k:(Rotation.from_rotvec(cams[camidx[k],:3]).as_matrix(),cams[camidx[k],3:]) for k in keys}
    newpoints={k:xyz[ptidx[k]] for k in pids}
    assert np.allclose(newposes[0][0],np.eye(3),atol=1e-12) and np.array_equal(newposes[0][1],np.zeros(3))
    assert newposes[seed][1][axis]==poses[seed][1][axis]
    return newposes,newpoints,dict(cost=float(fit.cost),nfev=fit.nfev,optimizer_success=bool(fit.success),message=fit.message,before=before,after=stats(np.linalg.norm(fit.fun.reshape(-1,2),axis=1)),gauge=dict(first_frame=0,seed_frame=seed,seed_center_component=axis,fixed_value=float(poses[seed][1][axis])))

def evaluate(K,poses,tracks,records,deadline):
    output=[]; summary={}
    for ti,t in enumerate(tracks):
        if time.monotonic()>deadline: raise TimeoutError('evaluation budget exceeded')
        # Camera-fit tracks are not an independent evaluation; keep only the whole-track heldout split.
        if t['split']!='heldout': continue
        for test,o in t['by_frame'].items():
            if test not in poses: continue
            other={k:v for k,v in t['by_frame'].items() if k!=test}
            tri=triangulate(K,poses,other)
            row=dict(track_id=t['track_id'],frame=records[test]['frame'],frame_index=test,region=t['regions'][test],observed_uv=o['uv'])
            if tri is None: row['state']='insufficient_parallax_or_cheirality'
            else:
                x,angle,pair,_=tri;pred,z=project(K,poses[test],x)
                row.update(state='predicted' if z[0]>0 else 'behind_test_camera',error_px=float(np.linalg.norm(pred[0]-o['uv'])),predicted_uv=pred[0].tolist(),parallax_degrees=angle,triangulation_frames=[records[k]['frame'] for k in pair])
            output.append(row)
    for kind in ['background_static_support_candidate','background_excluded_material','face_roi','body_roi']:
        rows=[r for r in output if r['region']==kind]; good=[r for r in rows if r['state']=='predicted']
        perframe={r['frame']:stats([x['error_px'] for x in good if x['frame']==r['frame']]) for r in records}
        summary[kind]=dict(attempted=len(rows),predicted=len(good),unusable=len(rows)-len(good),independent_track_count=len({r['track_id'] for r in good}),error_px=stats([r['error_px'] for r in good]),per_frame=perframe)
    return summary,output

def one(subject,plan):
    start=time.monotonic();deadline=start+plan['cpu_seconds_per_subject'];out=ROOT/subject/'global-fixedk'
    if out.exists(): raise RuntimeError(f'{subject}: global candidate directory exists; no duplicate run')
    out.mkdir();obs,tracks,hashes=input_data(subject);records=obs['records'];n=len(records)
    apply_material_gate(subject,obs,tracks,plan)
    K=np.array(plan['subjects'][subject]['K_integer_centers']);w,h=records[0]['wh']
    cam=dict(model='PINHOLE',width=w,height=h,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
    fitids=[i for i,t in enumerate(tracks) if t['camera_fit_eligible']]
    assert all(tracks[i]['split']=='fit' for i in fitids)
    result=dict(subject=subject,updated=now(),input_sha256=hashes,plan_sha256=sha(R/'global-fixedk-plan.json'),script_sha256=sha(Path(__file__)),K_integer_centers=K.tolist(),K_optimized=False,native_frame_count=n,background_fit_tracks=len(fitids),heldout_tracks=sum(t['split']=='heldout' for t in tracks),physical_static_background_verified=False,training_admitted=False,training_started=False,all_training_frames_registered=False)
    save(out/'input-contract.json',dict(**result,fit_track_ids=[tracks[i]['track_id'] for i in fitids]))
    if plan['material_preflight'][subject]['decision']=='stop_before_pose':
        result.update(state='stopped_material_support_preflight',registered_count=0,background_points=0,usable_background_pairs=0,reason=plan['material_preflight'][subject]['reason'],elapsed_seconds=time.monotonic()-start)
        save(out/'result.json',result);save(R/f'{subject}-global-fixedk.json',result)
        print(json.dumps({k:result[k] for k in ['subject','state','reason']},ensure_ascii=False),flush=True)
        return result
    rows=[];relative={};edges=[]
    opts={'max_epipolar_error':1.5,'min_iterations':1000,'max_iterations':10000,'seed':121}
    for p in obs['pairs']:
        if time.monotonic()>deadline: raise TimeoutError('relative pose budget exceeded')
        a,b=p['a'],p['b'];ids=[i for i in fitids if a in tracks[i]['by_frame'] and b in tracks[i]['by_frame']]
        row=dict(a=a,b=b,frames=p['frames'],background_fit_count=len(ids))
        if len(ids)<20: row['state']='insufficient_background_fit';rows.append(row);continue
        xx=[np.array([tracks[i]['by_frame'][k]['uv'] for i in ids]) for k in [a,b]]
        pose,info=poselib.estimate_relative_pose(*xx,cam,cam,opts,{})
        if pose is None or info['num_inliers']<20:
            row.update(state='insufficient_background_inliers',inliers=int(info.get('num_inliers',0)));rows.append(row);continue
        rr=np.asarray(pose.R);cc=-rr.T@np.asarray(pose.t);cc/=np.linalg.norm(cc)
        local={a:(np.eye(3),np.zeros(3)),b:(rr,cc)}
        ids_in=[i for i,ok in zip(ids,info['inliers']) if ok]
        tri=[triangulate(K,local,tracks[i]['by_frame'],True) for i in ids_in]
        valid=[t for t in tri if t is not None]
        row.update(inliers=int(info['num_inliers']),triangulable_inliers=len(valid),parallax_degrees=stats([v[1] for v in valid]))
        if len(valid)<20: row['state']='insufficient_triangulable_background';rows.append(row);continue
        row.update(state='usable_background_pair',R=rr.tolist(),C_direction=cc.tolist())
        relative[(a,b)]=(rr,cc,len(valid),float(np.median([v[1] for v in valid])))
        edges.append((a,b));rows.append(row)
    graph=components(n,edges);result.update(pair_graph_components=graph,usable_background_pairs=len(edges),pair_rows=rows)
    save(out/'relative-pairs.json',rows)
    poses={};points={};ba=None;steps=[]
    if len(graph)>1: result['state']='stopped_disconnected_background_graph'
    else:
        seedpairs=[(v[2]*np.sin(np.radians(v[3])),b,v) for (a,b),v in relative.items() if a==0]
        if not seedpairs: result['state']='stopped_no_first_frame_seed'
        else:
            _,seed,rv=max(seedpairs,key=lambda x:(x[0],x[1]));poses={0:(np.eye(3),np.zeros(3)),seed:(rv[0],rv[1])}
            attempted=set()
            while True:
                for i in fitids:
                    if i in points: continue
                    tr=triangulate(K,poses,tracks[i]['by_frame'],True)
                    if tr is not None: points[i]=tr[0]
                if len(poses)==n: break
                options=[]
                for k in range(n):
                    if k in poses or (k,len(poses)) in attempted: continue
                    ids=[i for i in points if k in tracks[i]['by_frame']]
                    options.append((len(ids),-k,k,ids))
                if not options or max(o[0] for o in options)<20: break
                _,_,k,ids=max(options);attempted.add((k,len(poses)))
                xx=np.array([tracks[i]['by_frame'][k]['uv'] for i in ids]);xyz=np.array([points[i] for i in ids])
                pose,info=poselib.estimate_absolute_pose(xx,xyz,cam,dict(max_reproj_error=2.,min_iterations=1000,max_iterations=10000,seed=121),{})
                step=dict(frame=records[k]['frame'],correspondences=len(ids),inliers=int(info.get('num_inliers',0)))
                if pose is None or info['num_inliers']<20: step['state']='pnp_failed'
                else: poses[k]=(np.asarray(pose.R),-np.asarray(pose.R).T@np.asarray(pose.t));step['state']='registered'
                steps.append(step)
                if time.monotonic()>deadline: raise TimeoutError('PnP budget exceeded')
            result['state']='registered_diagnostic_subset' if len(poses)==n else 'stopped_insufficient_3d_pnp_support'
            if len(poses)>=3 and len(points)>=20:
                poses,points,ba=bundle(K,poses,points,tracks,seed,deadline)
    result.update(registered_frames=[records[i]['frame'] for i in sorted(poses)],registered_count=len(poses),background_points=len(points),registration=steps,bundle_adjustment=ba)
    if poses:
        evaluation,evidence=evaluate(K,poses,tracks,records,deadline)
        result['heldout_leave_one_view_out']=evaluation
        save(out/'heldout-predictions.json',evidence)
        save(out/'cameras-diagnostic.json',[dict(frame=records[i]['frame'],frame_index=i,R_w2c=r.tolist(),C_world=c.tolist(),K_integer_centers=K.tolist()) for i,(r,c) in sorted(poses.items())])
        np.savez_compressed(out/'background-structure.npz',track_ids=np.array([tracks[i]['track_id'] for i in points]),xyz=np.array(list(points.values())))
    result['elapsed_seconds']=time.monotonic()-start
    result['updated']=now();result['next']='Review graph and heldout original-pixel evidence. Do not train or interpolate unregistered frames.'
    save(out/'result.json',result);save(R/f'{subject}-global-fixedk.json',result)
    print(json.dumps({k:result[k] for k in ['subject','state','registered_count','background_points','usable_background_pairs','elapsed_seconds']},ensure_ascii=False),flush=True)
    return {k:v for k,v in result.items() if k not in ['pair_rows','registration']}

def prepare():
    target=R/'global-fixedk-plan.json'
    if target.exists(): raise RuntimeError('Plan already registered')
    pp=read(R/'pair-probe-plan.json')
    plan=dict(registered=now(),hypothesis='With K fixed, one shared 3D reconstruction of non-person candidate materials can register the 24 native diagnostic frames; independent LOO person errors test whether apparent pair consistency extends globally.',subjects=pp['subjects'],main_variable='Global fixed-K shared geometry on the already supplemented, deduplicated native observation set. This combines supplementation and split hygiene; not a K-only effect.',camera_fit='Only retained FIT tracks whose every observation is outside the YOLO body rectangle expanded by12 native pixels. They are candidate background, not certified static; curtains/grass/reflections remain review risks.',no_new_features=True,old_pose_or_geometry_initialization=False,heldout_policy='Whole-track heldout and quarantine never enter relative/PnP/BA. Evaluation triangulates each heldout track excluding the tested view, predicts that view with fixed cameras.',seed='Only usable 0-1/0-2 pairs; select maximum fit triangulable count * sine(median parallax). First camera identity; initial seed baseline1. No heldout selection.',relative=dict(epipolar_px=1.5,min_fit_tracks=20,min_inliers=20,min_parallax_degrees=.5,seed=121,max_iterations=10000),registration=dict(PnP_min_points=20,PnP_min_inliers=20,PnP_error_px=2.,point_reprojection_gate_px=3.,no_interpolation=True),bundle=dict(K='fixed',first_camera='fixed R=I,C=0',scale='Fix the largest absolute initial seed camera-center component exactly',loss='soft_l1',scale_px=1.,max_nfev=80,max_global_BA=1),cpu_seconds_per_subject=600,one_candidate_per_person=True,stop=['Disconnected usable background pair graph: no fragment bridging with person tracks.','No seed or PnP>=20 material support: stop that reconstruction, preserve partial poses for diagnosis only.','Insufficient parallax/negative depth, large regional heldout error or ambiguous material identity: no training.','24 frames do not replace complete predeclared training-frame registration.'],acceptance=dict(diagnostic_only=True,required_before_full_registration=['All24 diagnostic frames in one background reconstruction','Every frame has10 heldout background predictions; median<=1px,p90<=3px','Manual original-pixel material review separates walls/floor from foliage/curtains/reflections','At least5 independent face and10 body heldout tracks, covered in3+frames with median<=1.5px,p90<=3px','Visibility/occlusion and person micro-motion assessed separately']),training_started=False,worker=str(Path(__file__)),worker_sha256=sha(Path(__file__)))
    summaries=[]
    for s in SUBJECTS:
        obs,tracks,hashes=input_data(s)
        frames=[]
        for k,r in enumerate(obs['records']):
            counts={split:{region:sum(t['split']==split and t['regions'].get(k)==region for t in tracks) for region in ['background_candidate','face_roi','body_roi']} for split in ['fit','heldout']}
            frames.append(dict(frame=r['frame'],counts=counts))
        summaries.append(dict(subject=s,input_sha256=hashes,background_fit_tracks=sum(t['camera_fit_eligible'] for t in tracks),per_frame=frames))
    save(target,plan);save(R/'global-fixedk-support-preflight.json',dict(updated=now(),subjects=summaries))
    print(json.dumps(dict(registered=str(target),support=[dict(subject=s['subject'],background_fit_tracks=s['background_fit_tracks']) for s in summaries])),flush=True)

def selftest():
    K=np.array([[1120.8,0,1352.8],[0,1119.8,763.7],[0,0,1.]])
    poses={0:(np.eye(3),np.zeros(3)),1:(Rotation.from_rotvec([.01,-.03,.02]).as_matrix(),np.array([1.,.02,.03])),2:(Rotation.from_rotvec([-.02,.04,-.01]).as_matrix(),np.array([-.3,.2,.1]))}
    rng=np.random.default_rng(121);xyz=rng.uniform([-1,-.8,4],[1,.8,6],(30,3));tracks=[]
    for x in xyz:
        d={k:dict(uv=project(K,p,x)[0][0].tolist()) for k,p in poses.items()};tracks.append(dict(by_frame=d,split='fit'))
        tri=triangulate(K,poses,d,True);assert tri is not None and np.linalg.norm(tri[0]-x)<1e-8
        pred=triangulate(K,poses,{k:o for k,o in d.items() if k!=1})
        assert np.linalg.norm(project(K,poses[1],pred[0])[0][0]-d[1]['uv'])<1e-7
    cam=dict(model='PINHOLE',width=2704,height=1520,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
    p,info=poselib.estimate_absolute_pose(np.array([t['by_frame'][2]['uv'] for t in tracks]),xyz,cam,dict(max_reproj_error=2.,seed=121),{})
    assert info['num_inliers']==30 and np.linalg.norm(-p.R.T@p.t-poses[2][1])<1e-7
    newposes,pts,b=bundle(K,poses,{i:x+rng.normal(0,.001,3) for i,x in enumerate(xyz)},tracks,1,time.monotonic()+60)
    assert b['after']['max']<1e-5
    print(json.dumps(dict(synthetic_projection_triangulation_pnp_and_gauge='passed',max_error_px=b['after']['max'],real_calibration_proven=False)),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    cv2.setNumThreads(2)
    if a.selftest: selftest();return
    if a.prepare: prepare();return
    import fcntl
    lock=open(ROOT/'global-fixedk.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (R/'global-fixedk-result.json').exists(): raise RuntimeError('Global results exist; do not rerun')
    plan=read(R/'global-fixedk-plan.json');assert plan['worker_sha256']==sha(Path(__file__))
    results=[]
    for s in SUBJECTS:
        save(R/'global-fixedk-progress.json',dict(updated=now(),pid=os.getpid(),state='running',subject=s,training_started=False))
        try: results.append(one(s,plan))
        except TimeoutError as e:
            failure=dict(subject=s,state='stopped_time_budget',error=str(e),training_started=False,training_admitted=False)
            save(R/f'{s}-global-fixedk.json',failure);results.append(failure)
    save(R/'global-fixedk-result.json',dict(updated=now(),subjects=results,training_started=False,requires_visual_review=True))
    save(R/'global-fixedk-progress.json',dict(updated=now(),pid=os.getpid(),state='complete_pending_visual_review',training_started=False))

if __name__=='__main__':
    try: main()
    except Exception as e:
        if '--selftest' not in sys.argv and '--prepare' not in sys.argv:
            save(R/'global-fixedk-failure.json',dict(updated=now(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),training_started=False))
        raise

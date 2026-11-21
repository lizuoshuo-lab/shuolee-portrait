"""Fixed-budget real relative geometry; no intrinsic, body, or Gaussian optimization."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time,traceback,itertools,collections
import numpy as np
import cv2
sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib
from global_track_initializer123 import average_rotations,angle_between_rotations,connected_components
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
G=R/'video10-global-initializer';D=G/'relative-graph';PLAN=D/'plan.json';U=R/'video10-remaining-material-review/global-input-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def stats(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max())) if len(a) else dict(n=0)
def winpath(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/'))
def normalized(uv,K):return (np.asarray(uv)-K[:2,2])/np.diag(K)[:2]
def skew(t):return np.array([[0.,-t[2],t[1]],[t[2],0.,-t[0]],[-t[1],t[0],0.]])
def sampson(uv1,uv2,F):
    if len(uv1)==0:return np.array([])
    a=np.c_[uv1,np.ones(len(uv1))];b=np.c_[uv2,np.ones(len(uv2))];Fa=a@F.T;Ftb=b@F
    return np.abs(np.sum(b*Fa,axis=1))/np.sqrt(np.maximum(np.sum(Fa[:,:2]**2+Ftb[:,:2]**2,axis=1),1e-30))
def triangulate(q1,q2,Q,t):
    if len(q1)==0:return np.empty((0,3)),np.array([],bool),np.array([])
    P1=np.c_[np.eye(3),np.zeros(3)];P2=np.c_[Q,t]
    h=cv2.triangulatePoints(P1,P2,q1.T.copy(),q2.T.copy()).T
    valid=np.abs(h[:,3])>1e-12;X=np.zeros((len(h),3));X[valid]=h[valid,:3]/h[valid,3,None]
    z2=(X@Q.T+t)[:,2];positive=valid&(X[:,2]>0)&(z2>0)
    d1=np.c_[q1,np.ones(len(q1))];d2=np.c_[q2,np.ones(len(q2))]@Q
    d1/=np.linalg.norm(d1,axis=1)[:,None];d2/=np.linalg.norm(d2,axis=1)[:,None]
    angle=np.degrees(np.arccos(np.clip(np.sum(d1*d2,axis=1),-1.,1.)))
    return X,positive,angle
def collect(data):
    byframe={f:{} for f in data['frames']}
    for tr in data['tracks']:
        for o in tr['observations']:byframe[o['frame']][tr['family']]=o['uv']
    return byframe
def all_cycles(n,rows):
    edges={(r['i'],r['j']):np.asarray(r['R_b_from_a']) for r in rows if 'R_b_from_a' in r}
    def rot(a,b):return edges[(a,b)] if a<b else edges[(b,a)].T
    out=[]
    for a,b,c in itertools.combinations(range(n),3):
        if all(e in edges for e in [(a,b),(b,c),(a,c)]):
            loop=rot(c,a)@rot(b,c)@rot(a,b)
            out.append(dict(indices=[a,b,c],frames=[frames[x] for x in [a,b,c]],rotation_closure_deg=float(np.degrees(angle_between_rotations(loop,np.eye(3))))))
    return out
if sys.argv[1:]==['--repair-readonly-pose']:
    failure=read(D/'failure.json')
    assert failure['error']=="ValueError('output array is read-only')" and failure['pairs_completed']==0
    D=D/'repair-readonly-pose';D.mkdir(exist_ok=True)
elif sys.argv[1:]==['--repair-invalid-pose']:
    repair=read(D/'repair-invalid-pose-plan.json')
    assert repair['previous_failure']=='PoseLib zero-translation failure pose was normalized; NaNs prevented final and failure serialization.'
    D=D/'repair-invalid-pose';D.mkdir(exist_ok=True)
elif sys.argv[1:]:raise ValueError('Unsupported execution argument')
assert not (D/'execution-admission.json').exists(),'Existing attempt; inspect instead of repeat.'
plan=read(PLAN);assert plan['budget']['pairs']==58
for p,h in plan['source_sha256'].items():assert sha(winpath(p))==h
fit=read(U/'fit-observations.json');held=read(U/'heldout-observations.json');K=np.asarray(read(U/'conditional-camera.json')['K_integer_centers']);invK=np.linalg.inv(K)
frames=fit['frames'];byframe={f:i for i,f in enumerate(frames)};fo=collect(fit);ho=collect(held)
pairs=read(U/'essential-minimal-graph.json')['edges'];assert len(pairs)==58
cam=dict(model='PINHOLE',width=2704,height=1520,params=[K[0,0],K[1,1],K[0,2],K[1,2]])
opts=dict(max_epipolar_error=plan['solver']['max_epipolar_error_px'],min_iterations=plan['solver']['min_iterations'],max_iterations=plan['solver']['max_iterations'],seed=plan['solver']['seed'])
save(D/'execution-admission.json',dict(updated=now(),pid=os.getpid(),plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),python=sys.executable,poselib=poselib.__version__,cv2=cv2.__version__,training_started=False,GPU=False,
    repair=sys.argv[1:] or None))
start=time.monotonic();rows=[];raw_poses=[]
try:
    for edge in pairs:
        assert time.monotonic()-start<plan['budget']['CPU_seconds']
        a,b=edge['a'],edge['b'];ids=sorted(set(fo[a])&set(fo[b]));assert len(ids)==edge['shared']
        aa=np.array([fo[a][f] for f in ids],float);bb=np.array([fo[b][f] for f in ids],float)
        row=dict(a=a,b=b,i=byframe[a],j=byframe[b],families=ids,count=len(ids),robust_admitted=False)
        pose,info=poselib.estimate_relative_pose(aa,bb,cam,cam,opts,{})
        if pose is None:
            row.update(state='no_relative_pose');rows.append(row);continue
        Q=np.array(pose.R,copy=True);t=np.array(pose.t,copy=True).reshape(3)
        mask=np.asarray(info['inliers'],bool);assert len(mask)==len(ids)
        raw_poses.append(dict(a=a,b=b,R=Q.tolist(),t=t.tolist(),num_inliers=int(mask.sum()),inliers=mask.tolist()))
        save(D/'raw-poses.json',raw_poses)
        if not np.isfinite(Q).all() or not np.isfinite(t).all() or np.linalg.norm(t)<1e-12 or mask.sum()<5:
            row.update(state='invalid_or_insufficient_pose',inlier_count=int(mask.sum()),translation_norm=float(np.linalg.norm(t)),rejection_reasons=['no_usable_pose'])
            rows.append(row);save(D/'pairs-partial.json',rows);continue
        t/=np.linalg.norm(t)
        E=skew(t)@Q;F=invK.T@E@invK;err=sampson(aa,bb,F)
        q1=normalized(aa[mask],K);q2=normalized(bb[mask],K)
        X,positive,angles=triangulate(q1,q2,Q,t)
        er1,er2,et=cv2.decomposeEssentialMat(E);branches=[]
        for r in [er1,er2]:
            for sign in [1.,-1.]:
                _,pos,ang=triangulate(q1,q2,r,et[:,0]*sign)
                branches.append(dict(positive=int(pos.sum()),rotation_difference_deg=float(np.degrees(angle_between_rotations(r,Q))),translation_dot=float(np.dot(et[:,0]*sign,t))))
        cv2.setRNGSeed(plan['solver']['seed'])
        Hom,hm=cv2.findHomography(aa,bb,cv2.RANSAC,plan['solver']['max_epipolar_error_px'],maxIters=10000,confidence=0.999)
        if Hom is not None and np.linalg.cond(Hom)<1e15:
            fa=cv2.perspectiveTransform(aa[None],Hom)[0];fb=cv2.perspectiveTransform(bb[None],np.linalg.inv(Hom))[0]
            he=np.maximum(np.linalg.norm(fa-bb,axis=1),np.linalg.norm(fb-aa,axis=1))
            hf=float(np.mean(he[mask]<=plan['solver']['max_epipolar_error_px'])) if mask.any() else 0.
            hd=dict(symmetric_max_error=stats(he),E_inlier_fraction_explained=hf,competing=hf>=plan['alternative']['competing_fraction'])
        else:hd=dict(competing=False,state='homography_failed',E_inlier_fraction_explained=0.)
        hids=sorted(set(ho[a])&set(ho[b]));ha=np.array([ho[a][f] for f in hids],float);hb=np.array([ho[b][f] for f in hids],float)
        herr=sampson(ha,hb,F)
        reasons=[];n=int(mask.sum());pf=float(positive.mean()) if n else 0.;par=float(np.median(angles[positive])) if positive.any() else 0.
        if len(ids)<12:reasons.append('fewer_than12sharedFIT')
        if n<12:reasons.append('fewer_than12inliers')
        if n/len(ids)<plan['robust_edge']['min_inlier_fraction']:reasons.append('low_inlier_fraction')
        if pf<plan['geometry']['min_positive_depth_fraction']:reasons.append('cheirality_ambiguous')
        if par<plan['geometry']['min_median_parallax_deg']:reasons.append('low_parallax')
        if hd['competing']:reasons.append('homography_competes')
        row.update(state='diagnostic_pose',R_b_from_a=Q.tolist(),t_direction=t.tolist(),inlier_count=n,inlier_families=[f for f,m in zip(ids,mask) if m],
            sampson_all_px=stats(err),sampson_inlier_px=stats(err[mask]),sampson_values_px=err.tolist(),
            positive_depth_fraction=pf,parallax_positive_deg=stats(angles[positive]),branches=branches,homography=hd,
            heldout_families=hids,heldout_sampson_px=stats(herr),heldout_sampson_values_px=herr.tolist(),
            robust_admitted=len(reasons)==0,rejection_reasons=reasons)
        rows.append(row)
        save(D/'pairs-partial.json',rows)
        save(D/'progress.json',dict(updated=now(),pid=os.getpid(),state='diagnosing_relative_pairs',pairs_completed=len(rows),total=len(pairs),training_started=False))
    cycles=all_cycles(len(frames),rows);accepted=[r for r in rows if r['robust_admitted']]
    comps=connected_components(len(frames),[(r['i'],r['j']) for r in accepted]);global_info=None
    if len(comps)==1:
        rotation,global_info=average_rotations(len(frames),[(r['i'],r['j'],np.array(r['R_b_from_a'])) for r in accepted])
        np.savez_compressed(D/'diagnostic-global-rotations.npz',frames=frames,R=rotation,K=K)
    for p,h in plan['source_sha256'].items():assert sha(winpath(p))==h
    save(D/'pairs.json',rows);save(D/'rotation-cycles.json',cycles)
    valid=[r for r in rows if 'R_b_from_a' in r]
    result=dict(updated=now(),state='relative_graph_diagnosis_complete',plan_sha256=sha(PLAN),elapsed_seconds=time.monotonic()-start,
        frames=len(frames),pairs=len(rows),estimated_relative_pairs=len(valid),robust_edges=len(accepted),
        rejection_counts=dict(collections.Counter(x for r in rows for x in r.get('rejection_reasons',[]))),
        robust_components=[[frames[k] for k in group] for group in comps],full_robust_graph_connected=len(comps)==1,
        rotation_cycles=stats([c['rotation_closure_deg'] for c in cycles]),cycles_over_screening_flag=sum(c['rotation_closure_deg']>plan['cycles']['screening_flag_deg'] for c in cycles),
        heldout_epipolar_px=stats([v for r in valid for v in r['heldout_sampson_values_px']]),
        global_rotation_diagnostic=global_info,absolute_cameras_estimated=0,old_pose_or_geometry_used=False,
        K_changed=False,heldout_used_for_fitting=False,body_fit=False,training_started=False,quality_accepted=False,
        limitation='These are coarse material candidates with conditionalK. E distance is not 3D reprojection. Face physicalHELD0. All58 mathematical edges are diagnosed but only robust edges can feed globalrotation; no camera interpolation or invented components.')
    save(D/'result.json',result);save(D/'progress.json',dict(updated=now(),pid=None,state='complete',result='result.json',training_started=False))
    print(json.dumps(result,ensure_ascii=False))
except Exception as e:
    save(D/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),pairs_completed=len(rows),training_started=False))
    save(D/'progress.json',dict(updated=now(),pid=None,state='failed',error=repr(e),training_started=False));raise

"""Fixed conditional-K native pair geometry. No camera interpolation, no training."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,os,hashlib,traceback
import numpy as np
import cv2
sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121')
ROOT=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
def now():return datetime.now(timezone.utc).isoformat()
def save(n,d):
    p=R/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)),p90=float(np.percentile(x,90)),fraction_le_1p5px=float(np.mean(x<=1.5)),fraction_le_3px=float(np.mean(x<=3))) if len(x) else dict(n=0)
def sampson(a,b,F):
    a=np.c_[a,np.ones(len(a))];b=np.c_[b,np.ones(len(b))];fa=a@F.T;fb=b@F
    return abs(np.sum(b*fa,axis=1))/np.sqrt(np.maximum(1e-20,np.sum(fa[:,:2]**2+fb[:,:2]**2,axis=1)))
def skew(t):return np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
def one(subject,plan):
    start=time.monotonic();op=ROOT/subject/'observations.json';obs=json.loads(op.read_text());K=np.asarray(plan['subjects'][subject]['K_integer_centers']);inv=np.linalg.inv(K)
    wh=obs['records'][0]['wh'];cam=dict(model='PINHOLE',width=wh[0],height=wh[1],params=[K[0,0],K[1,1],K[0,2],K[1,2]])
    tracks=[(t,{o['frame']:o for o in t['observations']}) for t in obs['tracks']];rows=[];poses={};held_errors=[]
    opt=plan['thresholds'];poseopts={k:opt[k] for k in ['max_epipolar_error_px','min_iterations','max_iterations']};poseopts['max_epipolar_error']=poseopts.pop('max_epipolar_error_px');poseopts['seed']=121
    for pair in obs['pairs']:
        if time.monotonic()-start>plan['max_seconds_per_subject']:raise RuntimeError('preregistered time budget exceeded')
        a,b=pair['a'],pair['b'];fa,fb=pair['frames'];shared=[(t,d) for t,d in tracks if fa in d and fb in d]
        fit=[(t,d) for t,d in shared if t['split']=='fit'];held=[(t,d) for t,d in shared if t['split']=='heldout']
        def xy(items,frame):return np.array([d[frame]['uv'] for _,d in items],float).reshape(-1,2)
        row=dict(a=a,b=b,frames=[fa,fb],fit_tracks=len(fit),heldout_tracks=len(held))
        if len(fit)<opt['min_fit_tracks']:row['state']='insufficient_fit_tracks';rows.append(row);continue
        pose,info=poselib.estimate_relative_pose(xy(fit,fa),xy(fit,fb),cam,cam,poseopts,{})
        if pose is None or int(info['num_inliers'])<opt['min_pair_inliers']:row.update(state='insufficient_geometric_inliers',num_inliers=int(info.get('num_inliers',0)));rows.append(row);continue
        F=inv.T@skew(pose.t)@pose.R@inv;ea=sampson(xy(held,fa),xy(held,fb),F);ef=sampson(xy(fit,fa),xy(fit,fb),F)
        poses[(a,b)]=dict(R=np.asarray(pose.R),t=np.asarray(pose.t));held_errors.extend(ea.tolist())
        positive=[];angles=[]
        if len(held):
            X=cv2.triangulatePoints(K@np.c_[np.eye(3),np.zeros(3)],K@np.c_[pose.R,pose.t],xy(held,fa).T,xy(held,fb).T);X=(X[:3]/X[3]).T
            c=-pose.R.T@pose.t;u=X;v=X-c;ang=np.degrees(np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.sum(u*v,axis=1)))
            good=np.isfinite(X).all(1)&(X[:,2]>0)&((X@pose.R.T+pose.t)[:,2]>0)
            positive=good.tolist();angles=ang.tolist()
        row.update(state='pair_pose_estimated',num_inliers=int(info['num_inliers']),R=pose.R.tolist(),t_direction=pose.t.tolist(),
            fit_sampson_px=stats(ef),heldout_sampson_px=stats(ea),heldout_evidence=[dict(track_id=t['track_id'],error_px=float(e),positive_depth=bool(p),parallax_degrees=float(g)) for (t,d),e,p,g in zip(held,ea,positive,angles)])
        rows.append(row);save('pair-probe-progress.json',dict(updated=now(),pid=os.getpid(),state='running',subject=subject,pairs_done=len(rows),training_started=False))
    cycles=[]
    for i in range(len(obs['records'])-2):
        if all(k in poses for k in [(i,i+1),(i+1,i+2),(i,i+2)]):
            d=poses[(i,i+2)]['R'].T@poses[(i+1,i+2)]['R']@poses[(i,i+1)]['R']
            cycles.append(dict(frames=[i,i+1,i+2],rotation_closure_degrees=float(np.degrees(np.arccos(np.clip((np.trace(d)-1)/2,-1,1))))))
    result=dict(subject=subject,updated=now(),K_integer_centers=K.tolist(),K_optimized=False,source_observations_sha256=hashlib.sha256(op.read_bytes()).hexdigest(),
        records=len(obs['records']),pairs=rows,estimated_pairs=len(poses),total_pairs=len(rows),heldout_sampson_px=stats(held_errors),rotation_cycles=cycles,
        physical_correspondences_verified=False,physical_calibration_accepted=False,global_poses_estimated=False,training_admitted=False,elapsed_seconds=time.monotonic()-start)
    save(f'{subject}-pair-probe.json',result)
    print(json.dumps(dict(subject=subject,estimated_pairs=len(poses),total_pairs=len(rows),heldout=stats(held_errors))),flush=True)
    return {k:v for k,v in result.items() if k not in ['pairs','rotation_cycles']}
def main():
    if (R/'pair-probe-result.json').exists():raise RuntimeError('Result already exists; do not rerun')
    import fcntl
    lock=open(ROOT/'pair-probe.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((R/'pair-probe-plan.json').read_text());results=[]
    for s in plan['subjects']:
        if (R/f'{s}-pair-probe.json').exists():raise RuntimeError('Subject evidence exists; inspect before resuming')
        results.append(one(s,plan))
    save('pair-probe-result.json',dict(updated=now(),subjects=results,training_started=False,complete_global_calibration=False))
    save('pair-probe-progress.json',dict(updated=now(),pid=os.getpid(),state='complete_pending_material_review',training_started=False))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save('pair-probe-failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

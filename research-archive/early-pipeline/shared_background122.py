"""N5c: one fixed-K shared-background bundle on the existing seven-frame diagnostic."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,hashlib,time,traceback,argparse
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';B=I/'background-geometry';O=B/'shared-path';P=B/'shared-path-plan.json'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def register():
    assert not P.exists() and not O.exists()
    inputs=[B/'observations.json',B/'geometry-result.json',B/'visual-review.json',I/'native-temporal-material/observations.npz']
    p=dict(stage='N5c',registered=now(),hypothesis='Pairwise epipolar agreement is insufficient for a shared static surface. With the SAME approximateK and frozen backgroundFIT families, couple all7 cameras to shared3D points and test target-excluded heldout reprojection.',
        source_inputs={str(f):sha(f) for f in inputs},frames=read(I/'background-geometry-plan.json')['frames'],
        causal_variable='Shared multi-frame 3D and global pose optimization instead of separate pairwise pose fits. No new image/feature/K/person fit.',
        initialization='Use currentN5 fixedK first-pair E only, never121/119 or oldoptimized poses. Firstcamera identity; unit firstbaseline. PnP each remaining camera on backgroundFIT points only.',
        gauge='FirstR/t fixed. Firstbaseline translation norm fixed1; its direction and rotation may optimize. Othercameras rigid6DOF, pointsXYZ, K fixed.',
        initialization_controls=dict(positive_depth=True,min_triangulation_angle_degrees=.2,PnP_threshold_px=2,PnP_iterations=1000,min_PnP_inliers=30),
        optimization=dict(method='scipy least_squares trf sparse numericJac',loss='huber',f_scale_px=1.,max_nfev=80,x_scale='jac'),
        heldout='All frozen heldout tracks retained. For each target use only other>=2 observations to triangulate a point; target is never in its own3D fit. All9 existing N1 face candidates evaluated if numeric-valid; these are not certified materials.',
        budget=dict(attempts=1,CPU_seconds=180,GPU=0),
        stop='Fewer than30PnP inliers or missing camera -> preserve and stop, no interpolation/threshold relax. Do not drop heldout observations by reprojection error.',
        limits='Near-ground planar/low-parallax degeneracy and material uncertainty remain. Sevenframe partial path is not all140 trainingframes; no training orcamera promotion.',
        K_optimized=False,training=False)
    save(P,p);print('N5c registered',flush=True)

def project(K,camera,X):
    v=X@camera[:3,:3].T+camera[:3,3]
    return v[:,:2]/v[:,2,None]@np.diag([K[0,0],K[1,1]])+[K[0,2],K[1,2]],v[:,2]

def triangulate(K,cameras,observations):
    rows=[]
    for fi,uv in observations:
        M=K@cameras[fi][:3];u,v=uv;rows.extend([u*M[2]-M[0],v*M[2]-M[1]])
    _,_,vt=np.linalg.svd(np.array(rows));q=vt[-1]
    return q[:3]/q[3] if abs(q[3])>1e-12 else np.full(3,np.nan)

def camera_center(M):return -M[:3,:3].T@M[:3,3]

def track_initial_point(K,cameras,t):
    ob=[(o['frame'],o['uv']) for o in t['observations'] if o['frame'] in cameras]
    if len(ob)<2:return None
    X=triangulate(K,cameras,ob)
    if not np.isfinite(X).all():return None
    if any(project(K,cameras[fi],X[None])[1][0]<=0 for fi,uv in ob):return None
    rays=[X-camera_center(cameras[fi]) for fi,uv in ob]
    angles=[np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b))) for i,a in enumerate(rays) for b in rays[i+1:]]
    return X if max(angles,default=0)>=.2 else None

def stats(errors):
    vals=[e['error_px'] for e in errors if e['valid_projection']]
    return dict(n_total=len(errors),n_valid=len(vals),invalid=len(errors)-len(vals),median=float(np.median(vals)) if vals else None,
                p90=float(np.percentile(vals,90)) if vals else None,max=float(max(vals)) if vals else None,
                all_targets_strictly_lt1=bool(vals) and len(vals)==len(errors) and max(vals)<1)

def evaluate(K,cameras,tracks):
    errors=[]
    for t in tracks:
        obs=[o for o in t['observations'] if o['frame'] in cameras]
        for target in obs:
            others=[(o['frame'],o['uv']) for o in obs if o['frame']!=target['frame']]
            row=dict(track=t['track'],family=t.get('family'),target=target['frame'],support_frames=[f for f,uv in others],observed=target['uv'],valid_projection=False,error_px=None)
            if len(others)>=2:
                X=triangulate(K,cameras,others)
                if np.isfinite(X).all():
                    uv,z=project(K,cameras[target['frame']],X[None])
                    if z[0]>0 and all(project(K,cameras[f],X[None])[1][0]>0 for f,_ in others):
                        row.update(valid_projection=True,predicted=uv[0].tolist(),delta=(uv[0]-target['uv']).tolist(),error_px=float(np.linalg.norm(uv[0]-target['uv'])))
            errors.append(row)
    return dict(summary=stats(errors),observations=errors)

def run():
    assert not O.exists();O.mkdir();plan=read(P);start=time.monotonic();cv2.setRNGSeed(1225);cv2.setNumThreads(4)
    for file,h in plan['source_inputs'].items():assert sha(Path(file))==h
    raw=read(B/'observations.json')['tracks'];fit=[t for t in raw if t['split']=='fit'];held=[t for t in raw if t['split']=='heldout']
    g=read(B/'geometry-result.json');K=np.array(g['K_integer_centers']);frames=plan['frames'];first=next(x for x in g['pairs'] if x['frames']==frames[:2])
    cameras={frames[0]:np.eye(4),frames[1]:np.eye(4)}
    cameras[frames[1]][:3,:3]=first['E_R'];cameras[frames[1]][:3,3]=np.array(first['E_t_direction'])/np.linalg.norm(first['E_t_direction'])
    points={};init_rows=[]
    for index,fi in enumerate(frames[2:],2):
        for t in fit:
            if t['track'] not in points:
                X=track_initial_point(K,cameras,t)
                if X is not None:points[t['track']]=X
        available=[(t['track'],o['uv']) for t in fit if t['track'] in points for o in t['observations'] if o['frame']==fi]
        assert len(available)>=30,'Insufficient freshPnP points'
        xyz=np.array([points[tid] for tid,uv in available]);uv=np.array([uv for tid,uv in available])
        ok,rv,tv,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        assert ok and inliers is not None and len(inliers)>=30,'PnP registration failed'
        mat=np.eye(4);mat[:3,:3]=cv2.Rodrigues(rv)[0];mat[:3,3]=tv.ravel();cameras[fi]=mat
        init_rows.append(dict(frame=fi,available=len(available),PnP_inliers=len(inliers)))
    # All FIT points have a fresh multiview DLT initialization after real camera registration.
    points={t['track']:X for t in fit if (X:=track_initial_point(K,cameras,t)) is not None}
    tids=sorted(points);point_lookup={tid:j for j,tid in enumerate(tids)}
    camera_slices={};params=[]
    for i,fi in enumerate(frames[1:],1):
        rv=cv2.Rodrigues(cameras[fi][:3,:3])[0].ravel();t=cameras[fi][:3,3]
        val=np.r_[rv,[np.arctan2(t[1],t[0]),np.arcsin(np.clip(t[2],-1,1))]] if i==1 else np.r_[rv,t]
        camera_slices[fi]=slice(len(params),len(params)+len(val));params.extend(val)
    ncam=len(params);x0=np.r_[params,np.array([points[t] for t in tids]).ravel()]
    observed=[]
    for t in fit:
        if t['track'] not in points:continue
        for o in t['observations']:observed.append((o['frame'],point_lookup[t['track']],o['uv']))
    target=np.array([v for fi,j,v in observed]);indices=np.array([j for fi,j,v in observed]);groups={fi:np.where([f==fi for f,j,v in observed])[0] for fi in frames}
    sparse=lil_matrix((len(observed)*2,len(x0)),dtype=int)
    for i,(fi,j,uv) in enumerate(observed):
        if fi in camera_slices:sparse[2*i:2*i+2,camera_slices[fi]]=1
        sparse[2*i:2*i+2,ncam+3*j:ncam+3*j+3]=1
    def decode(x):
        cams={frames[0]:np.eye(4)}
        for i,fi in enumerate(frames[1:],1):
            v=x[camera_slices[fi]];M=np.eye(4);M[:3,:3]=cv2.Rodrigues(v[:3])[0]
            M[:3,3]=[np.cos(v[4])*np.cos(v[3]),np.cos(v[4])*np.sin(v[3]),np.sin(v[4])] if i==1 else v[3:]
            cams[fi]=M
        return cams,x[ncam:].reshape(-1,3)
    def residual(x):
        if time.monotonic()-start>180:raise RuntimeError('Registered CPU wall budget exceeded')
        cams,xyz=decode(x);out=np.empty_like(target)
        for fi,ix in groups.items():out[ix]=project(K,cams[fi],xyz[indices[ix]])[0]-target[ix]
        if not np.isfinite(out).all():raise RuntimeError('Nonfinite shared projection')
        return out.ravel()
    save(O/'initialization.json',dict(updated=now(),frames=frames,PnP=init_rows,FIT_initialized_points=len(points),FIT_raw_tracks=len(fit),K_optimized=False,
        structural_init_exclusions='Only unavailable positive-depth/parallax initialization; all original observations retained. No heldout-error exclusion.'))
    save(O/'progress.json',dict(state='optimizing_shared_background',pid=os.getpid(),updated=now(),camera_count=len(cameras)))
    initial,_=decode(x0)
    opt=least_squares(residual,x0,jac_sparsity=sparse.tocsr(),method='trf',loss='huber',f_scale=1.,x_scale='jac',max_nfev=80,verbose=0)
    final,xyz=decode(opt.x)
    face=np.load(I/'native-temporal-material/observations.npz')
    face_tracks=[dict(track=tid,family='N1_face_never_camera_fit',observations=[dict(frame=fi,uv=face['uv'][fi,tid].tolist()) for fi in frames if face['valid'][fi,tid]]) for tid in range(9)]
    results={name:dict(background=evaluate(K,cams,held),face_candidates=evaluate(K,cams,face_tracks)) for name,cams in [('initial',initial),('shared',final)]}
    result=dict(stage='N5c',updated=now(),elapsed_seconds=time.monotonic()-start,K_integer_centers=K.tolist(),K_optimized=False,
        actual_registered_frames=frames,optimizer=dict(success=bool(opt.success),status=int(opt.status),message=opt.message,nfev=opt.nfev,cost=float(opt.cost),optimality=float(opt.optimality)),
        cameras=[dict(frame=fi,w2c=final[fi].tolist()) for fi in frames],validation=results,
        physical_material_certified=False,all_training_frames_registered=False,training_started=False,
        stage_scope='Approximate fixedK shared-background partial path; target-excluded Euclidean pixel errors. Face candidates have semantic/uncertainty limitations, not certifiedgt.')
    save(O/'result.json',result);np.savez_compressed(O/'fit-diagnostic.npz',points=xyz,track_ids=tids,parameters=opt.x)
    save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now()))
    print(json.dumps(dict(optimizer=result['optimizer'],validation={name:{group:r['summary'] for group,r in v.items()} for name,v in results.items()})),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);args=a.parse_args()
    try:globals()[args.stage]()
    except Exception as ex:
        if O.exists():save(O/'failure.json',dict(updated=now(),error=repr(ex),traceback=traceback.format_exc()))
        raise

import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,hashlib,signal,traceback
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from global_track_initializer123 import average_rotations,project
from joint_bearing_initializer123 import initialize
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');G=R/'video10-global-initializer';O=G/'joint-track-candidate';U=R/'video10-remaining-material-review/global-input-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def state(s,**extra):save(O/'progress.json',dict(updated=now(),pid=os.getpid(),stage=s,training_started=False,**extra))
def winpath(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/'))
def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max())) if len(a) else dict(n=0)
def timeout(*_):raise TimeoutError('registered300second wall budget')
assert not (O/'execution-admission.json').exists();plan=read(O/'plan.json')
for p,h in plan['source_sha256'].items():assert sha(winpath(p))==h
save(O/'execution-admission.json',dict(updated=now(),pid=os.getpid(),python=sys.executable,plan_sha256=sha(O/'plan.json'),script_sha256=sha(Path(__file__)),initializer_sha256=sha(Path(__file__).with_name('joint_bearing_initializer123.py')),GPU=False,training=False))
start=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['budget']['total_wall_seconds'])
try:
    K=np.asarray(read(U/'conditional-camera.json')['K_integer_centers']);state('noisy_synthetic_position_control')
    with np.load(R/'video10-remaining-material-review/generic-observability/synthetic-input-and-jacobian.npz',allow_pickle=False) as a:
        Rs=a['R'];ts=a['t'];Xs=a['X'];ids=a['observations']
    Cs=-np.einsum('nji,nj->ni',Rs,ts);truth,_=project(K,Rs,Cs,Xs,ids);noisy=truth+np.random.default_rng(12311).normal(0,.2,truth.shape)
    Cc,Xc,ci=initialize(K,Rs,ids,noisy,len(Xs),100);uvc,zc=project(K,Rs,Cc,Xc,ids);ce=np.linalg.norm(uvc-noisy,axis=1)
    control=dict(updated=now(),initializer=ci,error_px=stats(ce),positive_depth_fraction=float(np.mean(zc>0)),
        passed=bool(ci['success'] and ce.max()<1.5 and ce.mean()<.5 and np.all(zc>0)),synthetic_only=True)
    save(O/'control.json',control);assert control['passed'],'Noisy synthetic control failed; real candidate not run'
    state('global_rotation_start')
    fit=read(U/'fit-observations.json');frames=fit['frames'];fi={f:i for i,f in enumerate(frames)};tracks=fit['tracks'];families=[t['family'] for t in tracks];n=len(frames);m=len(tracks)
    ids=[];uv=[]
    for pi,t in enumerate(tracks):
        for o in t['observations']:ids.append([fi[o['frame']],pi]);uv.append(o['uv'])
    ids=np.asarray(ids,int);uv=np.asarray(uv,float)
    pairs=read(G/'relative-graph/repair-invalid-pose/pairs.json');edges=[(p['i'],p['j'],np.asarray(p['R_b_from_a'])) for p in pairs if 'R_b_from_a' in p]
    R0,ri=average_rotations(n,edges);save(O/'rotation-start.json',ri)
    state('joint_positive_bearings');C0,X0,position_info=initialize(K,R0,ids,uv,m,100);pu,z=project(K,R0,C0,X0,ids)
    save(O/'position-start.json',dict(updated=now(),initializer=position_info,projection_px=stats(np.linalg.norm(pu-uv,axis=1)),positive_depth_fraction=float(np.mean(z>0))))
    np.savez_compressed(O/'initialization.npz',K=K,R=R0,C=C0,X=X0,ids=ids,uv=uv,frames=frames,families=families)
    assert position_info['success'],'Real positive-bearing initializer did not converge'
    # One sharedXYZ per track, all563 observations retained throughout.
    rot0=Rotation.from_matrix(R0[1:]).as_rotvec().reshape(-1);cf=C0[1:].reshape(-1);fixed=int(np.argmax(np.abs(cf)));fixedvalue=float(cf[fixed]);assert abs(fixedvalue)>1e-8
    cfree=np.delete(np.arange(len(cf)),fixed);nr=len(rot0);nc=len(cfree);xs=nr+nc
    v0=np.r_[rot0,cf[cfree],X0.reshape(-1)];depth_scale=float(np.median(z[z>0]));assert depth_scale>1e-8
    def unpack(v):
        rot=np.concatenate([np.eye(3)[None],Rotation.from_rotvec(v[:nr].reshape(-1,3)).as_matrix()]);cc=np.empty(len(cf));cc[cfree]=v[nr:xs];cc[fixed]=fixedvalue
        return rot,np.vstack([np.zeros(3),cc.reshape(-1,3)]),v[xs:].reshape(m,3)
    def residual(v):
        rot,centers,points=unpack(v);cam=np.einsum('nij,nj->ni',rot[ids[:,0]],points[ids[:,1]]-centers[ids[:,0]])
        zz=cam[:,2];den=np.where(np.abs(zz)>1e-8,zz,np.where(zz>=0,1e-8,-1e-8))
        pr=cam[:,:2]/den[:,None]*np.diag(K)[:2]+K[:2,2]
        return np.c_[pr-uv,1000*np.minimum(zz/depth_scale-1e-5,0.)].reshape(-1)
    jac=sparse.lil_matrix((3*len(ids),len(v0)),dtype=np.int8);center_col={int(c):nr+i for i,c in enumerate(cfree)}
    for k,(cam,pt) in enumerate(ids):
        sl=slice(3*k,3*k+3)
        if cam:
            jac[sl,3*(cam-1):3*(cam-1)+3]=1
            for axis in range(3):
                raw=3*(cam-1)+axis
                if raw!=fixed:jac[sl,center_col[raw]]=1
        jac[sl,xs+3*pt:xs+3*pt+3]=1
    state('fixedK_joint_bundle',variables=len(v0),observations=len(ids),max_nfev=80)
    opt=least_squares(residual,v0,jac_sparsity=jac.tocsr(),method='trf',tr_solver='lsmr',loss='soft_l1',f_scale=1.,max_nfev=80,x_scale='jac')
    R1,C1,X1=unpack(opt.x);pred,depth=project(K,R1,C1,X1,ids);fiterr=np.linalg.norm(pred-uv,axis=1)
    np.savez_compressed(O/'diagnostic-candidate.npz',K=K,R=R1,C=C1,X=X1,ids=ids,uv=uv,predicted_uv=pred,depth=depth,frames=frames,families=families)
    state('frozen_candidate_heldout_evaluation')
    held=read(U/'heldout-observations.json');heldrows=[]
    for tr in held['tracks']:
        for excluded in tr['observations']:
            others=[o for o in tr['observations'] if o['frame']!=excluded['frame']];row=dict(family=tr['family'],frame=excluded['frame'],uv=excluded['uv'],fit_used=False)
            if len(others)<2:row['state']='insufficient_other_views';heldrows.append(row);continue
            A=[]
            for o in others:
                j=fi[o['frame']];P=K@np.c_[R1[j],-R1[j]@C1[j]];u,v=o['uv'];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
            _,_,vt=np.linalg.svd(np.asarray(A));h=vt[-1]
            if abs(h[3])<1e-12:row['state']='point_at_infinity';heldrows.append(row);continue
            xyz=h[:3]/h[3];j=fi[excluded['frame']];cam=R1[j]@(xyz-C1[j]);hp=K@cam
            if not np.all(np.isfinite(hp)) or abs(hp[2])<1e-12:
                row['state']='nonfinite_or_zero_depth_projection';heldrows.append(row);continue
            predh=hp[:2]/hp[2];err=float(np.linalg.norm(predh-np.asarray(excluded['uv'])))
            other_positive=all((R1[fi[o['frame']]]@(xyz-C1[fi[o['frame']]]))[2]>0 for o in others)
            row.update(state='valid' if cam[2]>0 and other_positive else 'behind_camera',error_px=err,predicted_uv=predh.tolist(),triangulation_frames=[o['frame'] for o in others],xyz=xyz.tolist());heldrows.append(row)
    save(O/'heldout-predictions.json',heldrows)
    per=[]
    for f in frames:
        fr=ids[:,0]==fi[f];hh=[r for r in heldrows if r['frame']==f]
        per.append(dict(frame=f,fit_projection=stats(fiterr[fr]),fit_positive_depth_fraction=float(np.mean(depth[fr]>0)),held_total=len(hh),held_valid=sum(r['state']=='valid' for r in hh),held_errors=stats([r['error_px'] for r in hh if 'error_px' in r]),held_invalid=sum(r['state']!='valid' for r in hh)))
    save(O/'per-frame.json',per)
    for p,h in plan['source_sha256'].items():assert sha(winpath(p))==h
    result=dict(updated=now(),state='single_joint_candidate_diagnosed',elapsed_seconds=time.monotonic()-start,
        optimizer=dict(success=bool(opt.success),status=int(opt.status),message=opt.message,nfev=opt.nfev,cost=float(opt.cost),optimality=float(opt.optimality)),
        frames=n,fit_families=m,fit_observations=len(ids),fit_projection=stats(fiterr),fit_positive_depth_fraction=float(np.mean(depth>0)),
        heldout_total=len(heldrows),heldout_valid=sum(r['state']=='valid' for r in heldrows),heldout_geometric_errors=stats([r['error_px'] for r in heldrows if 'error_px' in r]),
        heldout_all_valid_and_subpixel=all(r['state']=='valid' and r['error_px']<1 for r in heldrows),
        first_camera_fixed=True,scale_component_fixed=[int(fixed),fixedvalue],K_changed=False,old_camera_or_geometry_initialization=False,
        formal_camera_admission=False,quality_accepted=False,physical_face_HELD=0,SMPLX_fit=False,training_started=False,
        limitation='One fixed-K local optimizer from possibly wrong relative rotations, not proof of global optimum. Coarse background/material support, no full training-frame registration or independent face acceptance.')
    save(O/'result.json',result);signal.alarm(0);save(O/'progress.json',dict(updated=now(),pid=None,stage='complete',training_started=False));print(json.dumps(result,ensure_ascii=False))
except Exception as e:
    signal.alarm(0);save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,training_started=False));save(O/'progress.json',dict(updated=now(),pid=None,stage='failed',training_started=False));raise

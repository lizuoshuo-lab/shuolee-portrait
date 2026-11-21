"""Execute exactly the four registered synthetic controls, never real video fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from global_track_initializer123 import average_rotations,global_positions,project,angle_between_rotations,InitializationRejected
H=Path(__file__).parent;R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
G=R/'video10-global-initializer';U=R/'video10-remaining-material-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def state(name,**extra):
    value=dict(updated=now(),pid=os.getpid(),state=name,training_started=False,GPU=False)
    value.update(extra)
    save(G/'progress.json',value)
assert not (G/'execution-admission.json').exists(),'Existing attempt; inspect it instead of rerunning.'
plan=read(G/'plan.json')
for p,h in plan['source_sha256'].items():assert sha(Path(p))==h
ad=dict(registered=now(),pid=os.getpid(),command='Windows Python3.10 probe_video10_global_initializer123.py',plan_sha256=sha(G/'plan.json'),
 script_sha256=sha(Path(__file__)),implementation_sha256=sha(H/'global_track_initializer123.py'),
 compute='CPU only; prior process/nvidia inspection found no task worker. No real-camera estimates or training.')
save(G/'execution-admission.json',ad);start=time.monotonic();controls=[]
try:
    with np.load(U/'generic-observability/synthetic-input-and-jacobian.npz',allow_pickle=False) as a:
        Rt=a['R'];t=a['t'];Xt=a['X'];obs_ids=a['observations'];frames=a['frames'];families=a['families']
    K=np.asarray(read(U/'global-input-preflight/conditional-camera.json')['K_integer_centers'])
    Ct=-np.einsum('nji,nj->ni',Rt,t);assert np.array_equal(Ct[0],np.zeros(3))
    byframe={int(f):i for i,f in enumerate(frames)}
    pairdata=read(U/'global-input-preflight/essential-minimal-graph.json')['edges']
    edges=[(byframe[e['a']],byframe[e['b']],Rt[byframe[e['b']]]@Rt[byframe[e['a']]].T) for e in pairdata]
    uv,depth=project(K,Rt,Ct,Xt,obs_ids);assert np.all(depth>0)
    observations=[(int(c),int(p),q) for (c,p),q in zip(obs_ids,uv)]
    state('synthetic_rotation_averaging')
    Re,rotation_info=average_rotations(len(Rt),edges)
    angles=[angle_between_rotations(a,b) for a,b in zip(Re,Rt)]
    controls.append(dict(name='exact_rotation_average',passed=max(angles)<plan['accuracy']['exact_rotation_angle_max_rad'],
        max_angle_rad=max(angles),known_relative_rotations=True,real_pixels_used=False,diagnostic=rotation_info))
    assert controls[-1]['passed']
    state('synthetic_joint_positions')
    Ce,Xe,position_info=global_positions(K,Re,observations,len(Xt))
    reference_scale=float(np.median(np.linalg.norm(Ct[1:],axis=1)));Cref=Ct/reference_scale;Xref=Xt/reference_scale
    camera_error=np.linalg.norm(Ce-Cref,axis=1);point_error=np.linalg.norm(Xe-Xref,axis=1)
    relative_error=max(float(np.max(camera_error/np.maximum(1.,np.linalg.norm(Cref,axis=1)))),float(np.max(point_error/np.maximum(1.,np.linalg.norm(Xref,axis=1)))))
    predicted,ze=project(K,Re,Ce,Xe,obs_ids);pixel_error=np.linalg.norm(predicted-uv,axis=1)
    controls.append(dict(name='exact_global_bearing_positions',passed=relative_error<plan['accuracy']['exact_position_relative_max'] and float(pixel_error.max())<plan['accuracy']['exact_reprojection_max_px'],
        relative_error_max=relative_error,camera_error_baseline_units_max=float(camera_error.max()),point_error_baseline_units_max=float(point_error.max()),
        reprojection_max_px=float(pixel_error.max()),positive_depth_all=bool(np.all(ze>0)),diagnostic=position_info))
    assert controls[-1]['passed']
    np.savez_compressed(G/'synthetic-reconstruction.npz',K=K,R=Re,C=Ce,X=Xe,frames=frames,families=families,observation_ids=obs_ids,
        observation_uv=uv,projection_uv=predicted,true_R=Rt,true_C=Cref,true_X=Xref)
    state('synthetic_pure_rotation_negative')
    puv,pz=project(K,Rt,np.zeros_like(Ct),Xt,obs_ids)
    negative=[(int(c),int(p),q) for (c,p),q in zip(obs_ids,puv)]
    try:
        global_positions(K,Rt,negative,len(Xt))
    except InitializationRejected as e:
        controls.append(dict(name='pure_rotation_rejected',passed=e.reason=='position_geometry_not_unique_up_to_scale',reason=e.reason,diagnostic=e.diagnostic))
    else:raise AssertionError('Pure rotation incorrectly produced finite geometry')
    assert controls[-1]['passed']
    state('synthetic_disconnected_negative')
    split=len(Rt)//2;disconnected=[e for e in edges if (e[0]<split)==(e[1]<split)]
    try:average_rotations(len(Rt),disconnected)
    except InitializationRejected as e:
        controls.append(dict(name='disconnected_rotation_graph_rejected',passed=e.reason=='disconnected_rotation_graph',reason=e.reason,diagnostic=e.diagnostic))
    else:raise AssertionError('Disconnected graph incorrectly accepted')
    assert controls[-1]['passed'] and len(controls)==4
    elapsed=time.monotonic()-start;assert elapsed<plan['budget']['CPU_seconds']
    for p,h in plan['source_sha256'].items():assert sha(Path(p))==h
    result=dict(updated=now(),state='four_synthetic_controls_passed',elapsed_seconds=elapsed,controls=controls,plan_sha256=sha(G/'plan.json'),
        implementation_sha256=sha(H/'global_track_initializer123.py'),frames=len(frames),fit_families=len(families),observations=len(observations),
        heldout_used=False,real_pixels_used=False,relative_rotations_supplied_from_synthetic_truth=True,
        real_camera_estimates=0,new_body_fit=False,training_started=False,quality_accepted=False,
        remaining='Real essential-matrix branch estimation, outlier robustness, physical parallax, complete real-frame registration and independent face/background/body validation are not verified.')
    save(G/'result.json',result);state('complete',pid=None,result='result.json')
    print(json.dumps({k:v for k,v in result.items() if k!='controls'},ensure_ascii=False))
    print(json.dumps([dict(name=c['name'],passed=c['passed'],max_error=c.get('reprojection_max_px',c.get('max_angle_rad')),reason=c.get('reason')) for c in controls]))
except Exception as e:
    save(G/'result.json',dict(updated=now(),state='failed',error=repr(e),traceback=traceback.format_exc(),controls=controls,
         elapsed_seconds=time.monotonic()-start,training_started=False,real_camera_estimates=0))
    state('failed',error=repr(e));raise

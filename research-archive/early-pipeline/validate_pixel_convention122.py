"""One pre-registered R1d CPU check; never estimates a camera or reads holdouts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, signal, sys, time, traceback

H=Path(__file__).parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
K_PATH=R.parent/'raw-domain-121/pair-probe-plan.json'
PLAN=R/'pixel-convention-bridge-plan.json';O=R/'pixel-convention-bridge'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if O.exists():raise RuntimeError('R1d output already exists; inspect process/results, do not repeat')
plan=read(PLAN)
assert sha(K_PATH)==plan['K_source_sha256']
assert all(sha(H/name)==digest for name,digest in plan['source_hashes'].items())
O.mkdir()
save(O/'implementation.json',{'registered':datetime.now(timezone.utc).isoformat(),
 'plan_sha256':sha(PLAN),'K_sha256':sha(K_PATH),
 'source_sha256':{n:sha(H/n) for n in ['native_pixel_convention122.py','validate_pixel_convention122.py']},
 'scope':'Four stored conditional K matrices;25 deterministic grid pixels each. No feature, pose, K, residual selection, input image warp or GPU execution.'})
start=time.monotonic()
def progress(stage):save(O/'progress.json',{'stage':stage,'pid':os.getpid(),'updated':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-start})
def timeout(signum,frame):raise TimeoutError('R1d registered30second CPU budget exceeded')
signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['budget']['CPU_seconds'])
result={'passed':False,'stage':'R1d','subjects':[],'checks':[], 'K_refitted':False,
 'original_K_modified':False,'features_extracted':False,'images_resampled':False,
 'GPU_tasks':0,'heldout_used':False,'training_started':False,'physical_face_admitted':False}
def check(name,passed,**detail):
    result['checks'].append(dict(name=name,passed=bool(passed),**detail))
    if not passed:raise AssertionError(name)
progress('CPU_coordinate_identity_controls')
try:
    import numpy as np
    from native_pixel_convention122 import convert_pixel_convention, to_native_pixels
    source=read(K_PATH)
    for subject in ['grass','sofa','portrait','indoor']:
        k=np.asarray(source['subjects'][subject]['K_integer_centers'],dtype=np.float64)
        width,height=(1080,1920) if subject=='portrait' else (2704,1520)
        uv=np.stack(np.meshgrid(np.linspace(0,width-1,5),np.linspace(0,height-1,5)),axis=-1).reshape(-1,2)
        before_k=k.copy();before_uv=uv.copy()
        new_k,new_uv,receipt=to_native_pixels(k,uv,intrinsic_convention='integer_centers',observation_convention='integer_centers')
        back_k,back_uv,_=convert_pixel_convention(new_k,new_uv,intrinsic_convention='pixel_corners',observation_convention='pixel_corners',target_convention='integer_centers')
        def rays(mat,pix):
            a=np.column_stack([pix,np.ones(len(pix))])@np.linalg.inv(mat).T
            return a/np.linalg.norm(a,axis=1,keepdims=True)
        ray_error=float(np.abs(rays(k,uv)-rays(new_k,new_uv)).max())
        inverse_error=float(max(np.abs(k-back_k).max(),np.abs(uv-back_uv).max()))
        check(subject+'_same_rays',ray_error<plan['limits']['ray_max_abs_float64'],max_abs=ray_error)
        check(subject+'_inverse',inverse_error<plan['limits']['roundtrip_max_abs'],max_abs=inverse_error)
        check(subject+'_source_unchanged',np.array_equal(k,before_k) and np.array_equal(uv,before_uv))
        # Independent scalar projection of the original camera ray. If a corner
        # renderer receives the old integer-center K, its returned pixel-index
        # projection is lower by0.5 per axis, not a new estimated camera change.
        ray_xy=(uv-k[:2,2])/k[[0,1],[0,1]]
        native_prediction=ray_xy*new_k[[0,1],[0,1]]+new_k[:2,2]
        wrong_index_prediction=ray_xy*k[[0,1],[0,1]]+k[:2,2]-.5
        expected_wrong=wrong_index_prediction-uv
        correct_error=float(np.abs(native_prediction-.5-uv).max())
        negative_error=float(np.abs(expected_wrong+.5).max())
        check(subject+'_projection_translation',correct_error<1e-12 and negative_error<1e-12,
              corrected_projection_max_abs=correct_error,omitted_conversion_error_xy=[-.5,-.5],
              omitted_conversion_norm_px=float(np.linalg.norm(expected_wrong,axis=1).max()))
        result['subjects'].append({'subject':subject,'width':width,'height':height,'sample_count':25,
            'conversion':receipt,'ray_max_abs':ray_error,'roundtrip_max_abs':inverse_error,
            'physical_calibration_validated':False})
    negatives=[('missing_intrinsic',None,'integer_centers','pixel_corners'),
               ('missing_observation','integer_centers',None,'pixel_corners'),
               ('inconsistent','integer_centers','pixel_corners','pixel_corners'),
               ('unknown_target','integer_centers','integer_centers','unknown')]
    rejected=[]
    for name,a,b,c in negatives:
        try:convert_pixel_convention(k,uv,intrinsic_convention=a,observation_convention=b,target_convention=c)
        except ValueError as exc:rejected.append({'case':name,'message':str(exc)})
        else:raise AssertionError('Invalid convention accepted: '+name)
    check('unknown_or_mixed_conventions_rejected',len(rejected)==4,cases=rejected)
    check('frozen_inputs_unchanged',sha(K_PATH)==plan['K_source_sha256'] and all(sha(H/n)==h for n,h in plan['source_hashes'].items()))
    result.update(passed=True,state='completed_coordinate_representation_control',
     interpretation='An explicit change of coordinate origin preserves rays. Omission introduces0.5px per axis in a corner renderer. This does not show that existing121 solver residuals or historical119 ghosting contained that mismatch.121 feature/pose/residual math already uses the same integer-center convention.')
except Exception as exc:
    result.update(state='failed',error=str(exc),traceback=traceback.format_exc());print(result['traceback'])
finally:
    signal.alarm(0);result.update(seconds=time.monotonic()-start,updated=datetime.now(timezone.utc).isoformat())
    save(O/'result.json',result);progress(result['state'])
    print(json.dumps({'passed':result['passed'],'state':result['state'],'checks':len(result['checks']),'seconds':result['seconds']}))
if not result['passed']:sys.exit(1)

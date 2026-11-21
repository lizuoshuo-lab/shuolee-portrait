"""Admitted native inputs -> new official prior/SMPL-X fit -> free surface.

The old fitter is invoked from its pinned local project with fresh outputs.
Its anatomical detections are a coarse proposal prior, never material truth.
The new silhouette initializer uses explicit observed-valid support. No old
subject fit, Gaussian checkpoint, optimizer, re-estimated K or image warp.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,traceback
from copy import deepcopy
import numpy as np
from native_training_admission122_v2 import audit_input,assert_snapshot,checked_asset,read,save,sha,native_camera
from native_surface122_v2 import initialize_surface

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
SURFACE_CONFIG=dict(initial_points=200000,spacing_m=.012,padding_m=.4,max_voxels=5000000,
    mask_tolerance_px=2,min_observed_views=3,vote_fraction=.97,seed=6060)
FIT_CONFIG=dict(fit_steps=1200,seed=6060,pixel_center_convention='renderer_corner_to_detector_integer',
    landmark_image_support='mask_margin_4px',anatomical_visibility='continuous_mesh_rays_v2',
    max_fit_median_body_fraction=.04,max_fit_p90_body_fraction=.08)

def now():return datetime.now(timezone.utc).isoformat()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def status(root,**data):save(root/'preparation-status.json',dict(updated=now(),**data))

def build_fit_inputs(m,base,out):
    """Convert explicit native integer measurements to the old fitter contract.

    This function performs no inference. Fresh official priors are produced by
    the later GPU stage. The original observation arrays are copied unchanged.
    """
    def require_absolute_refs(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'} and not Path(value['path']).is_absolute():raise ValueError('Preparation requires absolute asset references; do not silently rewrite evidence bindings')
            for v in value.values():require_absolute_refs(v)
        elif isinstance(value,list):
            for v in value:require_absolute_refs(v)
    require_absolute_refs(m)
    spec=read(checked_asset(base,m['fit_inputs']));require_absolute_refs(spec)
    if spec.get('schema')!='shuolee-native-fit-input-v1':raise ValueError('Explicit fit-input schema required')
    for key,val in dict(source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],frame_plan_sha256=m['frame_plan']['sha256'],pixel_convention='integer_centers').items():
        if spec.get(key)!=val:raise ValueError('Fit input binding mismatch:'+key)
    observation_file=checked_asset(base,spec['observations'])
    camera_doc=read(checked_asset(base,m['cameras']));plan=read(checked_asset(base,m['frame_plan']))
    ids=plan['frame_ids'];clock=spec['frame_clock'];times=[r['time_seconds'] for r in clock]
    if [r['frame_id'] for r in clock]!=ids or ids!=sorted(ids) or len(ids)<3:raise ValueError('Exact chronological frame clock required')
    if not all(type(t) in (int,float) and np.isfinite(t) and t>=0 for t in times) or np.any(np.diff(times)<=0):raise ValueError('Monotonic native source time required')
    if len(set(np.diff(ids).tolist()))!=1:raise ValueError('Existing official temporal prior requires uniform frame stride; no silent temporal interpolation')
    checked_asset(base,spec['clock_evidence']);checked_asset(base,spec['observation_evidence'])
    if spec.get('old_individual_fit_loaded') is not False:raise ValueError('No prior personalized fit input')
    with np.load(observation_file,allow_pickle=False) as z:
        required={'frame_ids','boxes','keypoints','scores','flip_disagreement'}
        if not required<=set(z.files):raise ValueError('Raw wholebody detection arrays incomplete')
        obs={k:z[k].copy() for k in required}
    n=len(ids)
    shapes={'frame_ids':(n,),'boxes':(n,4),'keypoints':(n,133,2),'scores':(n,133),'flip_disagreement':(n,133)}
    if not np.issubdtype(obs['frame_ids'].dtype,np.integer) or not np.array_equal(obs['frame_ids'],ids) or any(obs[k].shape!=shape or not np.isfinite(obs[k]).all() for k,shape in shapes.items()):raise ValueError('Detection identity/shape/finite check failed')
    boxes=obs['boxes'];w=m['source']['width'];h=m['source']['height']
    if np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>w) or np.any(boxes[:,3]>h) or np.any(boxes[:,2:]<=boxes[:,:2]):raise ValueError('Native body boxes outside source raster')
    if np.any(obs['scores']<0) or np.any(obs['flip_disagreement']<0):raise ValueError('Invalid detector confidence/disagreement')
    cam={r['frame_id']:r for r in camera_doc['cameras']};frames={r['frame_id']:r for r in m['frames']}
    rows=[]
    for i,(f,t) in enumerate(zip(ids,times)):
        row=frames[f];camera,bridge=native_camera(cam[f],camera_doc,w,h,Path(row['rgb']['path']).name)
        rows.append(dict(frame=f,time_seconds=t,image=str(checked_asset(base,row['rgb'])),source_sha256=row['rgb']['sha256'],
            mask=str(checked_asset(base,row['alpha'])),mask_sha256=row['alpha']['sha256'],wh=[w,h],K=camera['K'],w2c=camera['w2c'],camera=camera,
            official_prediction_slot=i,initial_box_xyxy=boxes[i].tolist(),camera_kind='registered_fixed_native_K',observation_weight=1.,pixel_bridge=bridge))
    (out/'data').mkdir()
    save(out/'data/inputs.json',dict(records=rows));save(out/'data/training-inputs.json',dict(records=rows))
    # Bytes of all raw detector fields are retained; no target coordinate edits.
    import shutil
    shutil.copyfile(observation_file,out/'data/observations.npz')
    cfg=dict(FIT_CONFIG,frame_stride=int(ids[1]-ids[0]))
    save(out/'config.json',cfg)
    receipt=dict(source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],frame_plan_sha256=m['frame_plan']['sha256'],
        fit_inputs_sha256=m['fit_inputs']['sha256'],observation_sha256=sha(observation_file),observation_bytes_unchanged=sha(out/'data/observations.npz')==sha(observation_file),
        frame_ids=ids,times_seconds=times,complete_camera_KED_retained=True,cameras_reestimated=False,images_resampled=False,
        shared_static_body=True,coarse_anatomical_prior_not_material_truth=True,old_individual_geometry_loaded=False,
        external_occluders_in_fitter='Continuous self-visibility only; scene occluders unknown. Anatomical prior is coarse. Surface uses observed-valid mask.',visibility_policy='continuous_mesh_rays_v2',prior_is_coarse=True)
    save(out/'fit-input-binding.json',receipt);return receipt

def run_new_fit(input_path,output):
    path=Path(input_path).resolve(strict=True);out=Path(output)
    if out.exists():raise FileExistsError('New preparation root required')
    out.mkdir(parents=True)
    audit,m=audit_input(path);save(out/'admission.json',audit)
    status(out,state='input_rejected' if not audit['training_admitted'] else 'input_admitted',GPU_imported=False,prior_started=False,fit_started=False,surface_started=False,training_started=False)
    if not audit['training_admitted']:return 3
    if m['scope']!='person-reconstruction':raise ValueError('Synthetic audit fixtures cannot enter realperson prior/fit execution')
    assert_snapshot(audit,path);build_fit_inputs(m,path.parent,out)
    save(out/'input-reference.json',dict(path=str(path),sha256=sha(path),run_id=m['run_id']))
    if os.name!='posix':raise RuntimeError('Use the existing local Linux environment')
    import fcntl
    with open('/tmp/shuolee-native122-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU process already active; inspect rather than duplicate')
        sys.path.insert(0,str(PROJECT/'src'))
        from shuolee_portrait import runtime as rt
        rt.configure(ASSETS,out);rt.verify_assets()
        from shuolee_portrait.prior import infer
        from fitting_output_residual122 import main as fit
        from fit_gate_output_residual122 import evaluate
        # Same local public priors, a new inference and fresh shared fit only.
        status(out,state='running_fresh_prior',pid=os.getpid(),GPU_imported=True,prior_started=True,fit_started=False,surface_started=False,training_started=False)
        infer(out,read(out/'config.json'));assert_snapshot(audit,path)
        status(out,state='running_fresh_fit',pid=os.getpid(),GPU_imported=True,prior_started=True,fit_started=True,surface_started=False,training_started=False)
        fit(out,FIT_CONFIG['fit_steps']);evaluate(out,read(out/'config.json'));assert_snapshot(audit,path)
    fit_source=Path(__file__).parent/'fitting_output_residual122.py'
    receipt=dict(run_id=m['run_id'],source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],frame_plan_sha256=m['frame_plan']['sha256'],
        from_scratch=True,old_individual_geometry_loaded=False,executed_command=sys.argv,exit_code=0,
        fit_arrays_sha256=sha(out/'fit/fitted.npz'),fit_source_sha256=sha(fit_source),
        new_prior_receipt=ref(out/'inference/receipt.json'),new_prior_prediction=ref(out/'inference/prediction-postprocessed.npz'),
        original_init=ref(out/'fit/initialization.npz'),fit_result=ref(out/'fit/result.json'),fit_protocol=ref(out/'fit/protocol.json'))
    save(out/'fresh-fit-receipt.json',receipt)
    status(out,state='actual_fit_review_pending',GPU_imported=True,prior_started=True,fit_started=True,surface_started=False,training_started=False,
        fitted_sha256=receipt['fit_arrays_sha256'],required_review='fit-review.json binds fitted_sha256 and actual_same_camera_fit_reviewed,allow_surface,review_evidence')
    return 0

def build_surface(output):
    out=Path(output).resolve(strict=True);origin=read(out/'input-reference.json');path=Path(origin['path'])
    previous=read(out/'preparation-status.json')
    if previous['state']!='actual_fit_review_pending':raise RuntimeError('Only a completed new fit in this exact preparation run may continue')
    audit=read(out/'admission.json');assert_snapshot(audit,path);m=read(path)
    receipt=read(out/'fresh-fit-receipt.json')
    if sha(path)!=origin['sha256'] or receipt['run_id']!=m['run_id'] or receipt['fit_arrays_sha256']!=sha(out/'fit/fitted.npz'):raise ValueError('Fit/input modified')
    gate=read(out/'fit/geometry-gate.json');review=read(out/'fit-review.json')
    if not gate['coarse_initialization_gate_passed'] or gate['fit_sha256']!=receipt['fit_arrays_sha256']:raise ValueError('Coarse fit gate failed')
    if review.get('fitted_sha256')!=receipt['fit_arrays_sha256'] or review.get('actual_same_camera_fit_reviewed') is not True or review.get('allow_surface') is not True:raise ValueError('Actual same-camera fit review required')
    checked_asset(out,review['review_evidence'])
    with np.load(out/'fit/fitted.npz',allow_pickle=False) as z:fit={k:z[k].copy() for k in z.files}
    if not np.array_equal(fit['frame_ids'],read(checked_asset(path.parent,m['frame_plan']))['frame_ids']):raise ValueError('Fit frame identity mismatch')
    doc=read(checked_asset(path.parent,m['cameras']));cam={c['frame_id']:c['camera'] for c in doc['cameras']}
    rows=[dict(frame_id=f['frame_id'],alpha=f['alpha'],observation_valid=f['observation_valid']) for f in m['frames']]
    status(out,state='building_observed_surface',GPU_imported=False,prior_started=True,fit_started=True,surface_started=True,training_started=False)
    result=initialize_surface(fit,rows,cam,path.parent,out/'surface',SURFACE_CONFIG)
    assert_snapshot(audit,path)
    initial=dict(schema='shuolee-fresh-clothing-initialization-v1',run_id=m['run_id'],source_sha256=m['source']['asset']['sha256'],cameras_sha256=m['cameras']['sha256'],
        frame_plan_sha256=m['frame_plan']['sha256'],representation='free_world_clothing_surface',world_frame='right_handed_world_y_up',
        old_individual_geometry_loaded=False,old_gaussians_loaded=False,old_optimizer_loaded=False,training_steps=0,initial_points=200000,
        fresh_fit_receipt=ref(out/'fresh-fit-receipt.json'),fit_arrays=ref(out/'fit/fitted.npz'),fit_source=ref(Path(__file__).parent/'fitting_output_residual122.py'),
        initializer_source=ref(Path(__file__).parent/'native_surface122_v2.py'),arrays=ref(out/'surface/initial.npz'),executed_command=sys.argv,exit_code=0)
    save(out/'initialization.json',initial)
    status(out,state='actual_surface_review_pending',GPU_imported=False,prior_started=True,fit_started=True,surface_started=True,training_started=False,
        points=200000,arrays_sha256=result['arrays_sha256'],
        next='Actual surface coverage/centers review before train stage; no prepared training input or automatic dispatch yet.')
    return 0

def dispatch_training(output):
    raise RuntimeError('Residual preparation is versioned, but person optimizer orchestration is not yet connected. Do not fall back to the old pinhole entry.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fit','surface','train']);p.add_argument('--input');p.add_argument('--output',required=True);a=p.parse_args()
    if a.stage=='fit' and not a.input:p.error('--input required for new fit')
    root=Path(a.output);existed=root.exists()
    try:code=run_new_fit(a.input,root) if a.stage=='fit' else build_surface(root) if a.stage=='surface' else dispatch_training(root)
    except Exception as e:
        if root.exists() and (a.stage!='fit' or not existed):
            detail=dict(error=repr(e),traceback=traceback.format_exc(),updated=now())
            if not (root/'preparation-failure.json').exists():save(root/'preparation-failure.json',detail)
            state_file=root/'preparation-status.json'
            if state_file.exists():
                state=read(state_file)
                if state['state'] in {'running_fresh_prior','running_fresh_fit','building_observed_surface'}:
                    state.update(state='failed',updated=now(),error=repr(e));save(state_file,state)
        raise
    print(json.dumps(dict(exit_code=code,status=str(root/'preparation-status.json'))));return code
if __name__=='__main__':sys.exit(main())

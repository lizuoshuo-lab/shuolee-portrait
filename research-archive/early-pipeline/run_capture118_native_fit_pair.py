"""Two fresh fits, identical fixed-K raw input; only face/feet mesh visibility changes."""
from pathlib import Path
import json,os,sys,hashlib,subprocess,signal,shutil
from datetime import datetime,timezone
import numpy as np
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
BASE=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
VARIANT=BASE.parent/'shuolee-new-capture-118-native-visible'
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.cli import gpu_lock
from shuolee_portrait import runtime as rt
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kwargs):rt.save(RECORD/'queue-state.json',dict(updated=now(),supervisor_pid=os.getpid(),training_running=False,new_training_started_this_review=False,formal_training_admitted=False,quality_accepted=False,old_probe_steps=1000,**kwargs))

with gpu_lock():
    assert rt.read(BASE/'native-input-result.json')['coordinate_contract_passed']
    assert not (BASE/'fit').exists() and not VARIANT.exists(),'No fit resume or overwrite.'
    import torch
    from shuolee_portrait.surface_visibility import surface_supported
    points=torch.tensor([[0.,0,2],[0,0,3],[8,0,2],[0,0,2],[-2,0,2]])
    normals=torch.tensor([[0.,0,-1],[0,0,-1],[0,0,-1],[0,0,1],[0,0,-1]])
    K=torch.tensor([[1.,0,1.5],[0,1,1.5],[0,0,1.]])
    depth=torch.full((3,3),2.);depth[1,0]=0
    v,_=surface_supported(points,normals,K,torch.tensor([3.,3.]),depth,.1)
    assert v.tolist()==[1.,0,0,0,0],v
    for factor in (.5,2.):
        # Coordinate algebra must not acquire a half-pixel shift under image resizing.
        q=points@K.T;u=q[:,:2]/q[:,2:]
        kr=torch.diag(torch.tensor([factor,factor,1.]))@K
        qr=points@kr.T
        assert torch.allclose(qr[:,:2]/qr[:,2:]-.5, factor*(u-.5+.5)-.5)
    VARIANT.mkdir();(VARIANT/'data').mkdir();(VARIANT/'inference').mkdir();(VARIANT/'logs').mkdir()
    for p in (BASE/'data').iterdir():
        if p.is_file():shutil.copy2(p,VARIANT/'data'/p.name)
    for p in (BASE/'inference').iterdir():
        if p.is_file():shutil.copy2(p,VARIANT/'inference'/p.name)
    config=rt.read(BASE/'config.json');config['anatomical_visibility']='mesh_surface_zbuffer_v1';rt.save(VARIANT/'config.json',config)
    run=rt.read(BASE/'run.json');run.update(created=now(),variant='native_fixedK_mesh_surface_visibility',fresh_body_fit=True,
        old_fitted_person_loaded=False,old_gaussian_loaded=False);rt.save(VARIANT/'run.json',run)
    plan=dict(created=now(),hypothesis='Back-facing or out-of-frame anatomical labels can pull a shared head/foot mesh despite foreground-mask support.',
        single_changed_policy='Only face/feet observation weights multiply model-derived facing/depth/image support. Body weights, containment, priors, cameras and observations stay identical.',
        roots=[str(BASE),str(VARIANT)],steps_per_fit=1200,max_seconds_per_stage=600,
        source_sha256={x:sha(PROJECT/'src/shuolee_portrait'/x) for x in ['fitting.py','surface_visibility.py','fit_gate.py']},
        inputs_sha256=sha(BASE/'data/training-inputs.json'),observations_sha256=sha(BASE/'data/observations.npz'),
        prior_sha256=sha(BASE/'inference/prediction-postprocessed.npz'),synthetic_controls_passed=True,
        previous_fitted_body_loaded=False,model_visibility_not_image_visibility_truth=True,external_scene_occlusion_not_modeled=True,
        held_camera_failures_retained=True,no_gaussian_training=True)
    rt.save(RECORD/'native-fit-pair-plan.json',plan)
    gates={}
    for label,root in [('reference',BASE),('surface_visibility',VARIANT)]:
        rt.save(root/'fit-pair-plan.json',plan)
        for stage in ['fit','fit-gate']:
            command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',stage,'--output',str(root),'--assets',ASSETS]
            env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
            with (root/'logs'/f'{stage}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state(state='native_fixedK_fit_pair_running',stage=stage,arm=label,run_root=str(root),pid=proc.pid,command=command)
                print(json.dumps(dict(arm=label,stage=stage,pid=proc.pid,root=str(root))),flush=True)
                try:rc=proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                    rc=124
            if rc:
                coarse_failure=stage=='fit-gate' and (root/'fit/geometry-gate.json').is_file() and not rt.read(root/'fit/geometry-gate.json')['coarse_initialization_gate_passed']
                if not coarse_failure:
                    state(state='native_fixedK_fit_pair_failed',stage=stage,arm=label,run_root=str(root),pid=None,exit_code=rc)
                    raise SystemExit(rc)
        gates[label]=rt.read(root/'fit/geometry-gate.json')
    a=dict(np.load(BASE/'fit/initialization.npz'));b=dict(np.load(VARIANT/'fit/initialization.npz'))
    identical=a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
    assert identical,'Fresh initial meshes differ; this is not a single-policy comparison.'
    result=dict(created=now(),state='fresh_fit_pair_complete_pending_actual_review',same_initial_mesh=identical,
        same_camera_bytes=sha(BASE/'data/training-inputs.json')==sha(VARIANT/'data/training-inputs.json'),
        gates=gates,quality_accepted=False,formal_training_admitted=False,new_gaussian_training_started=False,
        next_action='Actual original-image nose/foot and six-view mesh review, retain all-observation errors and the rejected camera evidence before any Gaussian experiment.')
    rt.save(RECORD/'native-fit-pair-result.json',result)
    state(state='native_fixedK_fit_pair_review_pending',stage='fit_actual_review',pid=None,run_root=str(VARIANT),next_action=result['next_action'])
    print(json.dumps(result),flush=True)

"""One fresh single-variable fit: measured landmark foreground image support."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,subprocess,sys,time,hashlib
import numpy as np
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
BASE=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-support')
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.observation_support import image_support
from shuolee_portrait.cli import gpu_lock

# Synthetic occlusion/boundary controls, before touching the experiment.
mask=np.zeros((24,32),bool);mask[6:18,10:22]=1
points=np.array([[15,10],[9,10],[5,10],[-1,10],[31,23],[32,10]],float)
assert image_support(points,mask).tolist()==[1,1,0,0,0,0]
assert image_support(points,mask,margin_px=0).tolist()==[1,0,0,0,0,0]
assert not image_support(points,np.zeros_like(mask)).any()

with gpu_lock():
    assert not ROOT.exists(),'Do not replace or resume a previous body fit.'
    assert read(RECORD/'queue-state.json')['state']=='fresh_body_fit_pending_visual_review'
    ROOT.mkdir();(ROOT/'data').mkdir();(ROOT/'logs').mkdir();(ROOT/'inference').mkdir()
    for name in ['inputs.json','training-inputs.json','observations.npz']:
        shutil.copy2(BASE/'data'/name,ROOT/'data'/name)
    for name in ['prediction-postprocessed.npz','vitpose-keypoints.pt','receipt.json','crop-mapping.json']:
        shutil.copy2(BASE/'inference'/name,ROOT/'inference'/name)
    config=read(BASE/'config.json');config['landmark_image_support']='mask_margin_4px';save(ROOT/'config.json',config)
    run=read(BASE/'run.json');run.update(created=now(),variant='one_factor_landmark_image_support',
        parent_preprocessing=str(BASE),old_fitted_person_loaded=False,old_gaussian_loaded=False);save(ROOT/'run.json',run)
    plan=dict(created=now(),hypothesis='Off-image/background landmark targets can pull unobserved feet toward furniture even when aggregate body fit passes.',
        evidence=read(RECORD/'occluded-joint-audit.json')['groups'],
        variable='Only landmark image support: zero weight outside observed target mask plus fixed 4px existing diagnostic boundary tolerance; never move target pixels.',
        fresh_fit=True,previous_fitted_model_or_optimizer_loaded=False,
        official_detector_outputs_reused_for_same_video=True,
        camera_sha256=sha(ROOT/'data/training-inputs.json'),
        source_sha256={p:sha(PROJECT/'src/shuolee_portrait'/p) for p in ['fitting.py','observation_support.py']},
        synthetic_boundary_controls_passed=True,
        budget=dict(fit_steps=1200,max_fit_seconds=600),
        stop_conditions=['Any exception, failed coarse gate, or foreground/face deterioration in same-camera actual review.'],
        limitations=['Mask membership is not true 3D visibility or anatomical identity.','No threshold scan.','Not a camera calibration change.'])
    save(RECORD/'support-fit-plan.json',plan);save(ROOT/'single-variable-plan.json',plan)
    for stage in ['fit','fit-gate']:
        cmd=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',stage,'--output',str(ROOT),'--assets',ASSETS]
        env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
        with (ROOT/'logs'/f'{stage}.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(RECORD/'queue-state.json',dict(updated=now(),state='support_fit_running',stage=stage,pid=proc.pid,
                supervisor_pid=os.getpid(),command=cmd,run_root=str(ROOT),baseline_root=str(BASE),
                training_started=False,quality_accepted=False))
            print(json.dumps(dict(stage=stage,pid=proc.pid,log=str(ROOT/'logs'/f'{stage}.log'))),flush=True)
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                code=124
        if code:
            save(RECORD/'queue-state.json',dict(updated=now(),state='support_fit_failed',stage=stage,pid=None,
                run_root=str(ROOT),baseline_root=str(BASE),exit_code=code,training_started=False,quality_accepted=False))
            raise SystemExit(code)
    a=dict(np.load(BASE/'fit/initialization.npz'));b=dict(np.load(ROOT/'fit/initialization.npz'))
    assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a),'Initial mesh was not identical.'
    save(RECORD/'support-fit-comparison.json',dict(updated=now(),baseline=read(BASE/'fit/geometry-gate.json'),
        candidate=read(ROOT/'fit/geometry-gate.json'),same_initial_mesh=True,
        same_camera_bytes=sha(BASE/'data/training-inputs.json')==sha(ROOT/'data/training-inputs.json'),
        single_variable='landmark_image_support',visual_quality_review_pending=True,training_started=False))
    save(RECORD/'queue-state.json',dict(updated=now(),state='support_fit_pending_visual_review',stage='fit-gate',pid=None,
        run_root=str(ROOT),baseline_root=str(BASE),training_started=False,quality_accepted=False))
    print('Support-only fresh fit completed, same initialization and camera bytes verified.',flush=True)

"""Fresh raw-pixel observations and prior with the frozen fixed-K hypothesis.

No existing person parameters, fitted body, Gaussian or optimizer are loaded.
The output is a research input contract, not a physical calibration acceptance.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,hashlib,subprocess,signal
import numpy as np
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
BASE=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
CAND=BASE.parent/'shuolee-new-capture-118-fixedk-opencv-loopclosure'
ROOT=BASE.parent/'shuolee-new-capture-118-native-fixedk'
RECORD=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.cli import gpu_lock
from shuolee_portrait import runtime as rt
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_state(**kw):
    rt.save(RECORD/'queue-state.json',dict(updated=now(),run_root=str(ROOT),supervisor_pid=os.getpid(),training_running=False,new_training_started_this_review=False,formal_training_admitted=False,quality_accepted=False,old_probe_steps=1000,**kw))

with gpu_lock():
    assert not ROOT.exists(),'Fresh output only; inspect existing state instead of relaunching.'
    candidate=rt.read(CAND/'candidate-cameras.json')
    decoded=rt.read(BASE/'decoded.json')['records']
    assert len(decoded)==len(candidate['records'])==89
    ROOT.mkdir();(ROOT/'data').mkdir();(ROOT/'logs').mkdir()
    rows=[];worst=0.
    rng=np.random.default_rng(118)
    for i,(d,r) in enumerate(zip(decoded,candidate['records'])):
        assert d['frame']==r['frame'] and d['source_sha256']==r['source_sha256']
        assert sha(Path(r['image']))==r['source_sha256']
        Ki=np.asarray(r['K_integer_centers'],float);Kc=Ki.copy();Kc[:2,2]+=.5
        E=np.asarray(r['w2c'],float)
        assert np.isfinite(E).all() and abs(np.linalg.det(E[:3,:3])-1)<1e-6
        q=np.c_[rng.normal(0,.4,(64,2)),rng.uniform(1,8,64)]
        pi=(q@Ki.T);pc=(q@Kc.T)
        error=np.max(np.abs(pi[:,:2]/pi[:,2:]-pc[:,:2]/pc[:,2:]+.5));worst=max(worst,float(error))
        assert error<1e-10
        rows.append(dict(**d,image=r['image'],K=Kc.tolist(),K_integer_centers=Ki.tolist(),w2c=r['w2c'],mask=None,mask_sha256=None,
            official_prediction_slot=i,camera_kind='fixed_factory_K_opencv_research',camera_registration=dict(interpolation_used=False),observation_weight=1.))
    cfg=rt.read(BASE/'config.json')
    cfg.update(pixel_center_convention='renderer_corner_to_detector_integer',landmark_image_support='mask_margin_4px',
               camera_estimator='OpenCV essential/PnP and sparse BA; no COLMAP pose optimization',physical_calibration_validated=False)
    rt.save(ROOT/'config.json',cfg)
    rt.save(ROOT/'data/inputs.json',dict(records=rows,pixel_domain='decoded original exported video, no new distortion remap',
        intrinsic_origin='provided factory K, centered crop/resize hypothesis, fixed during OpenCV pose solve',
        coordinate_contract='integer-centre detector coordinates; K stored in corner convention; rasterizer projects with -0.5',
        distortion='No D8 formula assumption; zero distortion is an unverified rectified-export hypothesis',
        camera_candidate_sha256=sha(CAND/'candidate-cameras.json'),physical_calibration_accepted=False,from_scratch=True))
    run=rt.read(BASE/'run.json');run.update(created=now(),variant='native_pixel_fixedK_fresh_inference',old_subject_loaded=False,
        old_fitted_person_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,physical_calibration_accepted=False)
    rt.save(ROOT/'run.json',run)
    plan=dict(created=now(),camera_sha256=sha(CAND/'candidate-cameras.json'),config_sha256=sha(ROOT/'config.json'),
        input_sha256=sha(ROOT/'data/inputs.json'),projection_equivalence_max_px=worst,frames=89,
        source_sha256={name:sha(PROJECT/'src/shuolee_portrait'/name) for name in ['observations.py','prior.py','crops.py','runtime.py']},
        stages=['observations','prior'],budget_seconds=1800,old_personalized_parameters_loaded=False,
        physical_calibration_accepted=False,gaussian_training_started=False)
    rt.save(ROOT/'native-input-plan.json',plan);rt.save(RECORD/'native-input-plan.json',plan)
    for stage in ['observations','prior']:
        command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',stage,'--output',str(ROOT),'--assets',ASSETS]
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
        with (ROOT/'logs'/f'{stage}.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_state(state='native_fixedK_fresh_input_running',stage=stage,pid=proc.pid,command=command)
            print(json.dumps(dict(stage=stage,pid=proc.pid,root=str(ROOT))),flush=True)
            try:rc=proc.wait(timeout=1200 if stage=='observations' else 600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                rc=124
        if rc:
            save_state(state='native_fixedK_fresh_input_failed',stage=stage,pid=None,exit_code=rc)
            raise SystemExit(rc)
    obs=dict(np.load(ROOT/'data/observations.npz'));pred=dict(np.load(ROOT/'inference/prediction-postprocessed.npz'))
    assert obs['keypoints'].shape==(89,133,2) and all(np.isfinite(v).all() for v in pred.values())
    result=dict(created=now(),state='fresh_native_observations_prior_completed',frames=89,coordinate_contract_passed=True,
        observations_sha256=sha(ROOT/'data/observations.npz'),prior_sha256=sha(ROOT/'inference/prediction-postprocessed.npz'),
        physical_calibration_accepted=False,training_started=False,next_action='Actual nose/feet visibility review and two bounded fresh body fits; do not launch Gaussians automatically.')
    rt.save(ROOT/'native-input-result.json',result);rt.save(RECORD/'native-input-result.json',result)
    save_state(state='native_fixedK_fresh_input_ready',stage='observation_visibility_review',pid=None,next_action=result['next_action'])
    print(json.dumps(result),flush=True)

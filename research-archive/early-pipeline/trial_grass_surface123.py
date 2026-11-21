"""One explicitly authorized exploratory sofa-style reconstruction, fresh only.

The user's 2026-10-29 request permits a trial before strict physical subpixel
acceptance. This runner records that distinction rather than forging admission.
The unchanged frozen sofa runtime supplies the coarse extent and free surface.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,os,sys,shutil,subprocess,time,traceback,fcntl
import numpy as np
H=Path(__file__).parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
PRE=REC/'grass-preflight';OUT=REC/'grass-exploratory';BASE=Path('/opt/gs-slam-lab/shuolee-local-surface-123')
ROOT=BASE/'grass-exploratory';PROJECT=BASE/'runtime';ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def status(stage,**kw):
    item=dict(updated=now(),stage=stage,pid=os.getpid(),command=sys.argv,exploratory=True,quality_accepted=False,physical_camera_accepted=False,**kw)
    save(OUT/'progress.json',item)
    for folder in [REC,R]:
        q=read(folder/'queue-state.json');q.update(updated=now(),current_stage='grass_exploratory_'+stage,active_subject='grass',current_subject='grass',worker_pid=os.getpid(),worker_command=sys.argv,
            training_running=stage=='gaussian_training',training_started=(ROOT/'gaussian-run').exists(),finite_research_training_authorized=True,formal_training_admitted=False,latest_trial=str(OUT),next='Complete user-authorized fresh exploratory Gaussian trial, review at1000steps.')
        save(folder/'queue-state.json',q)
    print(json.dumps(item,ensure_ascii=False),flush=True)
def verify_runtime():
    for r in read(REC/'runtime-source-manifest.json')['files']:
        assert sha(PROJECT/r['path'])==r['sha256'],'Frozen runtime changed: '+r['path']
def prepare_camera():
    assert not ROOT.exists() and not (OUT/'plan.json').exists(),'Existing trial; inspect rather than restart.'
    verify_runtime();plan=read(PRE/'coverage-plan.json');source=Path(plan['run_root']);decoded=read(source/'decoded.json')
    assert len(decoded['records'])==94 and decoded['video_sha256']==plan['input_sha256']
    old=read(REC/'protocol.json');subject=next(r for r in old['subjects'] if r['id']=='grass')
    assert subject['video_sha256']==plan['input_sha256']
    trial=dict(registered=now(),user_authorization=['能有差不多1个像素的效果已经很好了，试着训练一下','做高斯训练试一下'],
        purpose='Exploratory Gaussian trial, not acceptance or a subpixel claim.',
        replaces_pretraining_physical_subpixel_gate_for_this_trial=True,source=plan['input_sha256'],frames=94,
        camera='Single fresh fixed-conditional-K OpenCV essential/PnP/scipy BA from original94decoded images. NoCOLMAP, nooldcamera or interpolation.',
        camera_source_sha256=sha(H/'local_surface123_camera.py'),camera_budget_seconds=1800,
        fit='Unchanged frozen sofa-style coarse staticSMPL-X initialization, used as extent/scale for silhouette envelope. Not the new fine alignment candidate or a claimed precise face fit.',
        point_representation='200000 free static world points sampled on fresh silhouette clothing envelope; no naked body binding.',
        steps_max=18000,review_step=1000,seed=6060,old_checkpoint_loaded=False,
        source_observations_reused_for_forward=True,new_missing_observations='SAM2masks and flipRTMW consistency',
        stop=['Unregistered selected frame','Nonfinite or clearly invalid coarse geometry','Empty or clipped clothing volume','1000step review fails'],
        approximate_one_pixel_quality_is_not_yet_measured=True,formal_acceptance=False)
    save(OUT/'plan.json',trial);ROOT.mkdir()
    for d in ['raw','data','camera','logs','reviews']:(ROOT/d).mkdir()
    for row in decoded['records']:
        p=source/'raw'/row['name'];assert sha(p)==row['source_sha256'];os.link(p,ROOT/'raw'/row['name'])
    save(ROOT/'decoded.json',decoded)
    cfg=read(PROJECT/'configs/server.json');cfg.update(frame_stride=5,max_image_size=2704,min_frames=12,max_frames=300,training_height=round(1520*1600/2704),
        seed=6060,initial_points=200000,iterations=18000,fit_steps=1200,pixel_center_convention='renderer_corner_to_detector_integer',landmark_image_support='mask_margin_4px',
        gaussian_representation='static_clothed_world',surface_spacing_m=.012,surface_padding_m=.40,surface_vote_fraction=.97,surface_mask_tolerance_px=2.,export_frame_mode='scene_up_keep_axis_order',
        camera_estimator='Fresh fixed conditionalK exploratory',physical_calibration_validated=False,finite_research_training_authorized=True,
        quality_accepted=False,fit_backend='legacy',exploratory_fit_role='coarse_extent_only_same_sofa_reference')
    save(ROOT/'config.json',cfg)
    save(ROOT/'run.json',dict(created=now(),video='/mnt/d/00/工作/数据集/video5.mp4',video_sha256=subject['video_sha256'],subject=subject,old_subject_loaded=False,old_fitted_person_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,exploratory=True))
    factory=old['calibration'];fw,fh=factory['image_wh'];w,h=2704,1520;cw,ch=fw,fw/(w/h);x0,y0=(fw-cw)/2,(fh-ch)/2
    A=np.array([[w/cw,0,w/cw*(.5-x0)-.5],[0,h/ch,h/ch*(.5-y0)-.5],[0,0,1.]])
    K=A@np.asarray(factory['K']);save(ROOT/'intrinsic-contract.json',dict(factory_sha256=factory['sha256'],factory_K=factory['K'],raw_K_integer=K.tolist(),pixel_map=A.tolist(),
        wh=[w,h],crop_xywh=[x0,y0,cw,ch],crop_is_assumption=True,D8_applied=False,calibration_applicability_confirmed=False,principal_center_convention='integer;renderer+0.5'))
    save(ROOT/'frame-plan.json',dict(native_video_frames=470,selected_frames=94,frame_stride=5,selected_source_indices=[r['source_frame_index'] for r in decoded['records']],decoded_sha256=sha(ROOT/'decoded.json'),no_interpolation=True))
    for folder in [REC,R]:
        q=read(folder/'protocol.json');q['latest_user_steering']=dict(updated=now(),statements=trial['user_authorization'],trial_plan=str(OUT/'plan.json'),exploratory_training_authorized_before_strict_subpixel_acceptance=True,formal_quality_accepted=False);save(folder/'protocol.json',q)
    status('fresh_camera_start',registered=0,total=94)
    import local_surface123_camera as camera_module
    camera_module.state=lambda subject,state=None,**kw: status(state or 'camera',**kw)
    camera_module.camera(ROOT)
    q=read(ROOT/'camera/result.json');assert q['frames']==94
    save(OUT/'camera-result.json',q);status('camera_complete',cameras=94,next='Generate missing masks and flip observations, fresh prior and coarse extent. No training has run yet.')

def stage(name):
    verify_runtime();rt.configure(ASSETS,ROOT);status(name)
    if name=='observations':
        from observe_from_coverage123 import observe
        observe(ROOT,PRE/'coverage')
    else:
        from shuolee_portrait.cli import stage as run_stage
        run_stage(name,ROOT,ASSETS)
    status(name+'_complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['camera','observations','prior','fit','fit-gate'],required=True);args=p.parse_args()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.phase=='camera':prepare_camera()
        else:stage(args.phase)
    except Exception as exc:
        save(OUT/'failure.json',dict(updated=now(),phase=args.phase,error=repr(exc),traceback=traceback.format_exc(),training_started=(ROOT/'gaussian-run').exists(),auto_retry=False))
        status('failed_preserved',error=repr(exc));raise

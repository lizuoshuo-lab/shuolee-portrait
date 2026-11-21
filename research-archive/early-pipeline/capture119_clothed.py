"""Finite static clothing-surface research, always a new fit and optimizer."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, time, traceback

PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ASSETS='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'
ROOT=Path('/opt/gs-slam-lab/shuolee-clothed-surface-119')
BASE=Path('/opt/gs-slam-lab/shuolee-capture118-all-persons-20261027')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/clothed-surface-119')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock

def status(subject,stage,**extra):
    d=dict(updated=rt.now(),subject=subject,stage=stage,pid=os.getpid(),command=sys.argv,
        quality_accepted=False,physical_calibration_accepted=False,**extra)
    rt.save(REC/f'{subject}-state.json',d);print(json.dumps(d,ensure_ascii=False),flush=True)

def init(subject):
    REC.mkdir(parents=True,exist_ok=True)
    protocol=REC/'protocol.json'
    if not protocol.exists():
        rt.save(protocol,dict(created=rt.now(),authorization='User requested points on skin/clothes instead of around SMPL initialization; previous all-person scope retained.',
            subjects=['portrait','sofa','grass','indoor118'],execution_order=['portrait','sofa'],
            hypothesis='Face-local position regularization and naked-mesh seeding restrict loose-clothing support. Test the earlier shared-static-world clothing representation.',
            intervention='A combined representation change: fresh silhouette surface seeds, unbound world XYZ/covariance, coordinate-appropriate learning rates, no body attraction or triangle quotas. Cannot identify the effect of each component separately.',
            controls='Same raw pixels, original masks, fixed factory-crop K/OpenCV extrinsics, frame identities and seed6060; fresh official prior, neutral SMPLX fit, Gaussians and empty optimizer.',
            physical_calibration_accepted=False,unknown_FISHEYE_8_applied=False,images_warped=False,dynamic_model=False,
            budgets=dict(new_fit_steps=1200,initial_points=200000,max_train_steps=18000,max_new_runs_per_subject=1,max_points=650000),
            first_review='Actual fresh fit and untrained surface before training; actual1000-step fullbody/face and centers before continuation.',
            deferred_subjects={'grass':'Existing fixed-camera all-frame fit contradicts observed body; do not waive gate because representation changed.',
                'indoor118':'External furniture occludes lower legs; full-mask hull may erase them. Do not apply previously rejected occluder mask.'},
            stop_conditions=['No silhouette surface / bounding-box clipping / catastrophic input displacement -> no Gaussian training.',
                'No actual1000-step decision in30min -> orderly stop with snapshot; never automatically approve.',
                'No new weights, remote training, threshold sweeps or old learned subject initialization.',
                'Same-camera and36novel-view review, world/+Y covariance+SH parity, actualOpenFile Gaussian/points before promotion.'],
            limitations=['Silhouette hull is an exterior envelope, not measured skin/clothing depth or concave geometry.',
                'Known camera applicability and subject micro-motion uncertainties are held fixed, not solved by this change.']))
    source=BASE/subject;root=ROOT/subject
    root.mkdir(parents=True,exist_ok=False)
    for name in ['data','logs','reviews']: (root/name).mkdir()
    for name in ['inputs.json','training-inputs.json','observations.npz']:
        shutil.copy2(source/'data'/name,root/'data'/name)
    for name in ['intrinsic-contract.json','run.json']:
        shutil.copy2(source/name,root/name)
    cfg=rt.read(source/'config.json');cfg.update(gaussian_representation='static_clothed_world',
        initial_points=200000,iterations=18000,fit_steps=1200,seed=6060,
        surface_spacing_m=.012,surface_padding_m=.40,surface_vote_fraction=.97,surface_mask_tolerance_px=2.)
    rt.save(root/'config.json',cfg)
    rt.save(root/'source-inputs.json',dict(created=rt.now(),source=str(source),
        reused_only_raw_image_camera_mask_and_2D_observations=True,
        old_subject_fit_loaded=False,old_gaussians_loaded=False,old_optimizer_loaded=False,
        image_camera_sha256=rt.sha(root/'data/training-inputs.json'),observations_sha256=rt.sha(root/'data/observations.npz'),
        source_camera_sha256=rt.sha(source/'camera/candidate-cameras.json'),
        new_official_prior_required=True,physical_calibration_accepted=False))
    status(subject,'initialized',run_root=str(root),training_running=False)

def run_stages(subject,names):
    root=ROOT/subject
    for name in names:
        path=root/'logs'/f'{name}.log'
        command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage',name,'--output',str(root),'--assets',ASSETS]
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
        with path.open('x') as log:
            proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(subject,name+'_running',child_pid=proc.pid,child_command=command,training_running=False)
            rc=proc.wait(timeout=1800)
            if rc:raise RuntimeError(f'{name} failed exit{rc}; see {path}')

def review(subject,step,snapshot):
    if step!=1000:return True
    root=ROOT/subject;path=root/'reviews/gaussian-step-001000-decision.json'
    digest=rt.sha(snapshot/'world.ply')
    status(subject,'gaussian_review_pending',step=step,training_running=True,optimizer_waiting=True,snapshot_sha256=digest)
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if path.exists():
            d=rt.read(path)
            assert d['snapshot_sha256']==digest and d['actual_same_camera_review'] and d['actual_center_distribution_review']
            return d['continue_same_run'] is True
        time.sleep(5)
    return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',choices=['portrait','sofa'],required=True)
    p.add_argument('--stage',choices=['init','prepare','surface','train','export'],required=True);a=p.parse_args()
    if a.stage=='init':init(a.subject);return
    root=ROOT/a.subject;cfg=rt.read(root/'config.json')
    with gpu_lock():
        try:
            if a.stage=='prepare':
                assert not (root/'inference').exists() and not (root/'fit').exists()
                run_stages(a.subject,['prior','fit','fit-gate'])
                status(a.subject,'fresh_fit_review_pending',training_running=False,fit_sha256=rt.sha(root/'fit/fitted.npz'))
            else:
                rt.configure(ASSETS,root)
                if a.stage=='surface':
                    d=rt.read(root/'reviews/fit-decision.json')
                    assert d['actual_same_camera_fit_reviewed'] and d['allow_surface_initialization'] and d['fit_sha256']==rt.sha(root/'fit/fitted.npz')
                    from shuolee_portrait.clothed_surface import initialize
                    initialize(root,cfg)
                    status(a.subject,'surface_review_pending',training_running=False)
                elif a.stage=='train':
                    d=rt.read(root/'reviews/surface-decision.json')
                    assert d['actual_initialization_review'] and d['allow_finite_research_training']
                    assert d['points_sha256']==rt.sha(root/'clothing-init/points.npz')
                    assert not (root/'gaussian-run').exists()
                    from shuolee_portrait.clothed_training import train_clothed
                    status(a.subject,'training',training_running=True,step=0)
                    train_clothed(root,cfg,checkpoint_review=lambda step,snapshot:review(a.subject,step,snapshot))
                    status(a.subject,'trained_export_review_pending',training_running=False,result=rt.read(root/'gaussian-run/result.json'))
                else:
                    assert not (root/'export').exists()
                    from shuolee_portrait.export import export
                    cfg['export_frame_mode']='scene_up_keep_axis_order'
                    export(root,cfg)
                    status(a.subject,'exported_actual_review_pending',training_running=False)
        except BaseException as exc:
            status(a.subject,a.stage+'_failed',training_running=False,error=repr(exc));raise

if __name__=='__main__':main()

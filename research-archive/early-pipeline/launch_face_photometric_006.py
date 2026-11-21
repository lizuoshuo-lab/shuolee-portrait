"""Run one exact-fit 006 Gaussian face photometric comparison after registration."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch_unified_reused_fit_016_006 as base


SUBJECT='006'
BASE=base.common.root_for(SUBJECT)/'run'
ROOT=base.common.LAB/'shuolee-smpl-face-photometric-006'
PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-photometric')
BASE_COMMIT=base.COMMIT
COMMIT='187988f49cd31966e28278374a42cf1778fe8e2e'
ACCEPTANCE=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001'
                '/shuolee/current/robust-portrait-122/smpl-face-photometric-preregistration.json')
ACCEPTANCE_SHA='e62aa89be98ebec30efc7cae01585cd3bfeb842d018c3606a00016a203e22d02'
BASE_RESULT='18333c3592caa278dfdd9cb8c11ee3ec2b561e94688178587824b31208c6faf7'
REFERENCE='frame_000033'
DRIVER='7aa4d97afa7177f5bd326a9ed24bbd8612d4b9af45495c2d712de35ee8f96a57'
HELPER='5195801e98475198d74a17e3cfe0bcf76b5d2d2520d4d8f75378553fe766e93c'
CAMERA_PINS={
    'gaussian-run/training-cameras.json':'20116a3ca1f873e54a1e0a5b5b2bddb6cd4e3648d8164377008f787ab05d8248',
    'export/cameras.json':'d24a129ac7036101ee5d9499b7eb2aecb0f53ec4df8c5fa54e4cc7d96a3f8474',
}
need,read,sha,pin=base.need,base.read,base.sha,base.pin


def rebound(function,**overrides):
    return FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__)


common=SimpleNamespace(**(vars(base.common)|dict(PROJECT=PROJECT,root_for=lambda subject:ROOT)))
imports=rebound(base.imports,common=common)
stage=rebound(base.stage,common=common,imports=imports)
stock_worker=rebound(base.worker,common=common,COMMIT=COMMIT,imports=imports)


def verify_source():
    pin(Path(base.__file__).resolve(),DRIVER)
    pin(Path(base.common.__file__).resolve(),HELPER)
    rebound(base.common.git_state,PROJECT=PROJECT)(COMMIT)
    pointer=(PROJECT/'.git').read_text(encoding='utf-8').strip().removeprefix('gitdir: ')
    gitdir=subprocess.check_output(['/usr/bin/wslpath','-u',pointer],text=True).strip()
    git=['/usr/bin/git','--git-dir',gitdir,'--work-tree',str(PROJECT)]
    subprocess.run([*git,'merge-base','--is-ancestor',BASE_COMMIT,COMMIT],check=True)
    changes=subprocess.check_output([*git,'diff','--name-only',BASE_COMMIT,COMMIT],text=True).splitlines()
    need(set(changes)=={'src/shuolee_portrait/training.py','tests/test_sh0_foreground_alpha.py'},
         'Candidate contains changes outside the registered photometric term and tests.')
    return changes


def verify_acceptance():
    pin(ACCEPTANCE,ACCEPTANCE_SHA)
    plan=read(ACCEPTANCE)
    fixed={
        'baseline_terminal':BASE/'result.json','fixed_fit':BASE/'fit/fitted.npz',
        'fixed_rows':BASE/'data/training-inputs.json','fixed_selection':BASE/'fit/selection.json',
    }
    need(plan['state']=='registered_before_candidate_training' and plan['subject']==SUBJECT
         and plan['baseline_commit']==BASE_COMMIT
         and plan['baseline_terminal']['sha256']==BASE_RESULT
         and plan['change']['applied_weight']==.05 and plan['change']['extra_raster_passes']==0
         and plan['change']['preserve_existing_loss'] is True
         and plan['budget']==dict(gaussian_iterations=10000,schedule_iterations=18000,seed=0,
             review_snapshot=1000,one_candidate=True,empty_optimizer=True,fit_updates=0,
             continuous_through_export=True),'Independent acceptance scope changed.')
    for key,path in fixed.items():
        need(Path(plan[key]['path'])==path,f'Unexpected registered {key}.')
    proof={}
    for key in (*fixed,'fixed_face_cameras_and_regions','fixed_spill_bands','fixed_formal_holdouts'):
        row=plan[key]
        pin(Path(row['path']),row['sha256'])
        proof[row['path']]=row['sha256']
    return proof


def check():
    changes=verify_source()
    proof=verify_acceptance()
    seq,_,effective,verify=imports()
    accepted=verify(BASE)
    baseline=read(BASE/'result.json')
    meta=read(BASE/'run.json')
    receipt=read(BASE/'fit/official-fit-result.json')
    selection=read(BASE/'fit/selection.json')
    cfg=read(BASE/'config.json')
    terminal=read(BASE.parent/'wrapper-terminal.json')
    prior_profile=read(BASE/'gaussian-run/config.json')
    need(terminal['state']=='completed' and terminal['result_sha256']==BASE_RESULT
         and terminal['candidate_commit']==BASE_COMMIT and baseline['fit_reused'] is True
         and baseline['gaussian_from_scratch'] is True and baseline['fit_calls']==0
         and baseline['reference_frame_id']==selection['reference_frame']==REFERENCE
         and baseline['frames']==len(selection['selected_frames'])==24
         and accepted['fit_sha256']==receipt['fit_sha256'] and receipt['selection']==selection,
         'Fixed 006 baseline delivery or fit identity differs.')
    profile=effective(face_photometric=True)
    need(profile['face_photometric'] is True and profile['face_photometric_weight']==.05
         and all(prior_profile[k]==v for k,v in profile.items()
                 if k not in ('face_photometric','face_photometric_weight','initial_points','image_cache_frames'))
         and prior_profile['initial_points']==20908 and prior_profile['optimizer_initial_state_entries']==0
         and prior_profile['image_cache_frames']==24 and profile['optimization']['position_lr_max_steps']==18000,
         'Candidate changed more than the declared face loss or the empty-optimizer profile.')
    # Precompute exactly the same native target ROIs using CPU tensors.
    import torch
    from shuolee_portrait.training import _face_interior_pixels,_native_foreground_target
    torch.set_num_threads(1)
    rows=read(BASE/'data/training-inputs.json')['records']
    with np.load(BASE/'fit/fitted.npz',allow_pickle=False) as fit:
        ordered=fit['frame_ids'].tolist()
        counts={}
        for row in rows:
            _,alpha,box=_native_foreground_target(row)
            pixels=_face_interior_pixels(fit['observed_native_keypoints'][ordered.index(row['frame']),67:118],alpha,box)
            counts[row['frame']]=0 if pixels is None else int(pixels.shape[1])
    need(len(counts)==24 and min(counts.values())==3790 and max(counts.values())==4839
         and counts[REFERENCE]==4644,'Actual native training ROI support differs.')
    pins={str(BASE/n):sha(BASE/n) for n in ('run.json','config.json','result.json',*base.COPY)}
    pins.update(proof)
    for name,digest in CAMERA_PINS.items():
        pin(BASE/name,digest)
        pins[str(BASE/name)]=digest
    plan=dict(schema='shuolee-006-face-photometric-frozen-fit-v1',subject=SUBJECT,candidate_commit=COMMIT,
        baseline_commit=BASE_COMMIT,baseline_run=str(BASE),baseline_result_sha256=BASE_RESULT,
        acceptance_plan=str(ACCEPTANCE),acceptance_plan_sha256=ACCEPTANCE_SHA,
        changed_paths=changes,selected_frames=selection['selected_frames'],
        retained_holdouts=selection['retained_holdouts'],reference_frame=REFERENCE,
        coverage=selection['coverage'],full_bin_coverage=selection['full_bin_coverage'],
        fit_reused=True,fit_calls=0,cold_video_clock=False,gaussian_from_scratch=True,
        fresh_gaussian_optimizer=True,face_photometric=True,face_photometric_weight=.05,
        face_photometric_roi_pixels=counts,camera_pins=CAMERA_PINS,input_sha256=pins,
        asset_sha256=rebound(base.common.official_pins,PROJECT=PROJECT)(),
        runtime=common.cpu_runtime(common.clean_env()),
        code_sha256={name:sha(PROJECT/name) for name in common.SOURCE_FILES},
        script_sha256=sha(Path(__file__).resolve()),reused_driver_sha256=DRIVER,launcher_helper_sha256=HELPER)
    return plan,None,(BASE,receipt,selection,meta,cfg),profile


def launch():
    verify_source()
    verify_acceptance()
    plan=read(ROOT/'preregistration.json')
    run=ROOT/'run'
    need(plan['candidate_commit']==COMMIT and plan['acceptance_plan_sha256']==ACCEPTANCE_SHA
         and plan['script_sha256']==sha(Path(__file__).resolve())
         and plan['runtime']==common.cpu_runtime(common.clean_env()),'Registered candidate or runtime changed.')
    for names,directory in ((plan['staged_sha256'],run),(plan['code_sha256'],PROJECT)):
        for name,digest in names.items():
            pin(directory/name,digest)
    for path,digest in (plan['input_sha256']|plan['asset_sha256']).items():
        pin(Path(path),digest)
    with common.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not any((run/name).exists() for name in ('gaussian-run','export','audit','result.json','status.json'))
             and not any((ROOT/name).exists() for name in ('launch.json','entry.log','trial-terminal.json')),
             'Fresh single-candidate trial required. Existing attempts are preserved.')
        common.write(ROOT/'launch.json',dict(state='launching',subject=SUBJECT,candidate_commit=COMMIT,
            started=common.now(),acceptance_plan_sha256=ACCEPTANCE_SHA,
            preregistration_sha256=sha(ROOT/'preregistration.json')))
        args=[str(common.ENV/'bin/python'),'-I','-B','-u',str(Path(__file__).resolve()),'worker']
        with (ROOT/'entry.log').open('xb') as log:
            rc=subprocess.run(['/usr/bin/timeout','--kill-after=15s','3585s',*args],
                env=common.clean_env(),cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:
            return rc
        terminal=read(ROOT/'wrapper-terminal.json')
        result=read(run/'result.json')
        need(terminal['state']=='completed' and terminal['candidate_commit']==COMMIT
             and terminal['result_sha256']==sha(run/'result.json') and result['fit_calls']==0
             and result['fit_reused'] is True and result['gaussian_from_scratch'] is True
             and result['frames']==24 and result['reference_frame_id']==REFERENCE,
             'Trial delivery is incomplete.')
        trained=read(run/'gaussian-run/config.json')
        need(trained['face_photometric'] is True and trained['face_photometric_weight']==.05
             and trained['face_photometric_roi_pixels']==plan['face_photometric_roi_pixels'],
             'Actual trained photometric configuration differs from registered native support.')
        for name,digest in CAMERA_PINS.items():
            pin(run/name,digest)
        common.write(ROOT/'trial-terminal.json',dict(terminal,
            state='delivered_pending_quality_review',face_photometric=True,face_photometric_weight=.05,
            acceptance_plan_sha256=ACCEPTANCE_SHA,preregistration_sha256=sha(ROOT/'preregistration.json'),
            actual_face_roi_pixels=trained['face_photometric_roi_pixels'],
            verified_camera_sha256=CAMERA_PINS))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','launch','worker'))
    args=parser.parse_args()
    if args.action=='worker':
        stock_worker(SUBJECT)
    elif args.action=='launch':
        raise SystemExit(launch())
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        data=check()
        if args.action=='stage':
            stage(SUBJECT,data)
        else:
            print(json.dumps(dict(state='face_trial_inputs_verified',gpu_calls=0,
                candidate_commit=COMMIT,face_roi_pixels=data[0]['face_photometric_roi_pixels'])),flush=True)


if __name__=='__main__':
    main()

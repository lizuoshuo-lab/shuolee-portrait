"""Fresh unified SH0 training on pinned 016/006 official pose archives."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch_unified_cold_016_006 as common


COMMIT = '79b906fff9b62b6833041194fcc2ea3f64a1d464'
RESULT_PINS = {
    '016':'9f1fae99da88a0c5f1c7a927bd58cef2b1dabb2d0f5d9f72a5071c91df86abaf',
    '006':'4b228f050170a172997a12fa7e4c2d06361877098c03232fa8011202e0f25683',
}
COPY = ('fit/fitted.npz','fit/official-fit-result.json','fit/selection.json',
        'fit/shared-body.json','data/training-inputs.json')
need, read, sha, pin = common.need, common.read, common.sha, common.pin


def imports():
    sys.path.insert(0,str(common.PROJECT/'src'))
    from shuolee_portrait import official_sequence as seq, stable_selection as selector
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit
    return seq,selector,smpl_sh0_effective_config,verify_official_fit


def old_run(subject):
    return common.LAB/common.PREVIOUS[subject]['parent']/'run'


def stock(subject):
    seq,selector,profile,verify = imports()
    old=old_run(subject)
    pin(old/'result.json',RESULT_PINS[subject])
    result=read(old/'result.json')
    receipt=read(old/'fit/official-fit-result.json')
    checked=verify(old)
    selection=read(old/'fit/selection.json')
    metadata=read(old/'run.json')
    cfg=read(old/'config.json')
    original=common.PREVIOUS[subject]
    need(result['state']=='completed' and result['format_passed'] is True and receipt['selection']==selection
         and checked['fit_sha256']==receipt['fit_sha256'] and metadata['assets']==str(common.ASSETS)
         and metadata['video_sha256']==original['video_sha256']
         and metadata['camera_config_sha256']==original['camera_sha256']
         and selection['retained_holdouts']==original['named_holdouts'] and selection['arc']=='half',
         'Changed archived official fit, named source camera, or heldouts.')
    return old,receipt,selection,metadata,cfg


def reselect_016(old,old_receipt,cfg):
    import numpy as np
    seq,selector,_,_=imports()
    inputs=seq.load_inputs(old/'data/inputs.json',old/'data/observations.npz',
        camera_length_contract=cfg['camera_length_contract'],reference_frame='frame_000027',
        capture_layout=cfg['capture_layout'])
    need(inputs['confidence']==old_receipt['source']['confidence']
         and len(inputs['ids'])==56 and len(inputs['scope']['fit_frame_ids'])==52
         and not any(inputs['targets'][inputs['ids'].index(f)].any() for f in inputs['scope']['retained_holdouts']),
         'Archived fitting scope or exact detector confidence recipe changed.')
    original_input=old_receipt['source']['input_files']
    for name in ('data/inputs.json','data/observations.npz'):
        pin(old/name,original_input[str(old/name)])
    shared=read(old/'fit/shared-body.json')
    pin(old/'fit/shared-body.json',old_receipt['shared_body_sha256'])
    beta=np.asarray(shared['betas'],dtype=np.float32).reshape(1,10)
    winners, reports, descriptors, paths = {}, {}, {}, {}
    for frame in inputs['scope']['fit_frame_ids']:
        base=old/'fit/frames'/frame
        f, report_path=base/'winner-camera-space.npz',base/'result.json'
        with np.load(f,allow_pickle=False) as z:
            winners[frame]=dict(z)
        report=reports[frame]=read(report_path)
        row=inputs['rows'][inputs['ids'].index(frame)]
        seq.check_result(report,row)
        need(np.array_equal(winners[frame]['betas'],beta),'Archived winner changed shared shape.')
        paths.update({str(f):sha(f),str(report_path):sha(report_path)})
        descriptors[frame]=dict(frame_id=frame,path=str(report_path),sha256=paths[str(report_path)])
    selection=selector.select_frames(inputs,winners,scope=inputs['scope'],reports=reports,shared_betas=beta)
    return inputs,selection,winners,descriptors,beta,paths


def assemble_016(run,old,old_receipt,cfg,packed):
    import smplx
    import torch
    seq,_,_,_=imports()
    inputs,selection,winners,descriptors,beta,paths=packed
    reference=selection['reference_frame']
    ordered=[reference]+[f for f in selection['selected_frames'] if f!=reference]
    chosen=seq._selection_inputs(inputs,ordered,reference)
    torch.set_num_threads(1)
    body=smplx.SMPLX(str(common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,
        num_expression_coeffs=10,use_pca=True,num_pca_comps=12,flat_hand_mean=False,
        use_face_contour=False,create_body_pose=False,create_transl=False,dtype=torch.float32).eval()
    for folder in ('fit','data'):
        (run/folder).mkdir()
    for path,digest in paths.items():
        pin(Path(path),digest)
    seq.write_results(run,chosen,{f:winners[f] for f in ordered},beta,seq.pelvis_basis(body),
        [descriptors[f] for f in ordered],execution_kind='historical_readonly_assembly',
        extra=dict(selection=selection,fitted_candidate_frames=len(winners),
            shared_body_sha256=old_receipt['shared_body_sha256'],fit_reused=True,fit_calls=0))
    common.write(run/'fit/selection.json',selection)
    shutil.copyfile(old/'fit/shared-body.json',run/'fit/shared-body.json')


def preflight(subject):
    common.git_state(COMMIT)
    env=common.cpu_runtime(common.clean_env())
    assets=common.official_pins()
    old,receipt,selection,metadata,cfg=stock(subject)
    packed=reselect_016(old,receipt,cfg) if subject=='016' else None
    if packed is not None:
        selection=packed[1]
    else:
        need(selection['coverage_basis']==imports()[1].SOURCE_BIN_COVERAGE_BASIS,
             '006 needs original selector replay before re-use.')
    profile=imports()[2]()
    need(profile['opacity_reset_steps']==[1800,3600,5400,7200] and profile['iterations']==10000
         and profile['sampling_policy']=='seeded_equal_occupied_source_bin_cycles'
         and selection['retained_holdouts']==common.PREVIOUS[subject]['named_holdouts'],
         'Integrated profile or protected roster differs.')
    pins={str(old/n):sha(old/n) for n in ('run.json','config.json','result.json',*COPY)}
    if packed is not None:
        pins.update(packed[-1])
        pins.update({str(old/n):sha(old/n) for n in ('data/inputs.json','data/observations.npz')})
    plan=dict(schema='shuolee-unified-reused-fit-v1',subject=subject,candidate_commit=COMMIT,
        baseline_run=str(old),baseline_result_sha256=RESULT_PINS[subject],
        source_video_sha256=metadata['video_sha256'],camera_config_sha256=metadata['camera_config_sha256'],
        selector_replayed_from_original_candidates=subject=='016',selected_frames=selection['selected_frames'],
        retained_holdouts=selection['retained_holdouts'],reference_frame=selection['reference_frame'],
        coverage=selection['coverage'],full_bin_coverage=selection['full_bin_coverage'],
        copied_fitting_input_from=str(old),fit_reused=True,fit_calls=0,
        cold_video_clock=False,fresh_gaussian_optimizer=True,gaussian_from_scratch=True,
        code_sha256={n:sha(common.PROJECT/n) for n in common.SOURCE_FILES},
        input_sha256=pins,asset_sha256=assets,runtime=env,
        script_sha256=sha(Path(__file__).resolve()),launcher_helper_sha256=sha(Path(common.__file__).resolve()))
    return plan,packed,(old,receipt,selection,metadata,cfg),profile


def stage(subject,state):
    plan,packed,(old,receipt,selection,metadata,cfg),profile=state
    root=common.root_for(subject)
    need(not root.exists(),'Preserve existing output and preregistration.')
    root.mkdir()
    run=root/'run'
    run.mkdir()
    if packed is None:
        for name in COPY:
            target=run/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(old/name,target)
            pin(target,plan['input_sha256'][str(old/name)])
    else:
        assemble_016(run,old,receipt,cfg,packed)
    common.write(run/'run.json',dict(metadata,created=common.now(),fit_reused=True,fit_calls=0,
          copied_fitting_input_from=str(old),cold_video_clock=False,gaussian_from_scratch=True))
    common.write(run/'config.json',dict(cfg,**profile))
    verified=imports()[3](run)
    fitted=read(run/'fit/official-fit-result.json')
    need(fitted['selection']==selection and verified['fit_sha256']==sha(run/'fit/fitted.npz'),
         'Reassembled official fit does not bind the selected original roster.')
    staged={name:sha(run/name) for name in (*COPY,'run.json','config.json')}
    common.write(root/'preregistration.json',dict(plan,registered_at=common.now(),staged_sha256=staged))
    print(json.dumps(dict(state='fit_reuse_staged',subject=subject,run=str(run),
        preregistration_sha256=sha(root/'preregistration.json'),selected_frames=selection['selected_frames'],
        coverage=selection['coverage'])),flush=True)


def execute(subject):
    root=common.root_for(subject)
    run=root/'run'
    plan=read(root/'preregistration.json')
    common.git_state(COMMIT)
    need(plan['candidate_commit']==COMMIT and plan['script_sha256']==sha(Path(__file__).resolve())
         and plan['launcher_helper_sha256']==sha(Path(common.__file__).resolve())
         and plan['runtime']==common.cpu_runtime(common.clean_env()),'Registered code or runtime differs.')
    for name,digest in plan['staged_sha256'].items():
        pin(run/name,digest)
    for name,digest in plan['code_sha256'].items():
        pin(common.PROJECT/name,digest)
    for name,digest in (plan['input_sha256']|plan['asset_sha256']).items():
        pin(Path(name),digest)
    with common.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subject=='006':
            predecessor=common.root_for('016')
            done=read(predecessor/'wrapper-terminal.json')
            need(done['state']=='completed' and done['candidate_commit']==COMMIT,
                 '016 must complete first on the unified revision.')
            pin(predecessor/'run/result.json',done['result_sha256'])
        need(not any((run/p).exists() for p in ('gaussian-run','export','audit','result.json','status.json'))
             and not (root/'entry.log').exists(),'Fresh Gaussian work only; existing attempts are preserved.')
        common.write(root/'launch.json',dict(state='launching',subject=subject,started=common.now(),
              candidate_commit=COMMIT,fit_reused=True,fit_calls=0,
              preregistration_sha256=sha(root/'preregistration.json')))
        cmd=[str(common.ENV/'bin/python'),'-I','-B','-u',str(Path(__file__).resolve()),'worker',subject]
        with (root/'entry.log').open('xb') as log:
            return subprocess.run(['/usr/bin/timeout','--kill-after=15s','3585s',*cmd],
                env=common.clean_env(),cwd=common.PROJECT,stdout=log,stderr=subprocess.STDOUT).returncode


def worker(subject):
    import sys
    root=common.root_for(subject)
    run=root/'run'
    plan=read(root/'preregistration.json')
    # Require the outer process to own the shared lease. Direct worker execution is rejected.
    with common.LEASE.open('r+') as lease:
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError('Use the launch action to establish the exclusive GPU lease.')
    need(dict(os.environ)==common.clean_env() and sys.flags.isolated==1,
         'Worker environment is contaminated.')
    seq,_,_,verify=imports()
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import train_official,replay_official_bound
    from shuolee_portrait.export import export
    from shuolee_portrait.bound_review_panels import write_panels
    rt=seq.rt
    start=time.monotonic()
    cfg=read(run/'config.json')
    status=dict(state='running',stage='gaussian_inputs',updated=rt.now(),fit_reused=True,
                fitting_complete=True,training_started=False,quality_accepted=False)
    with gpu_lock():
        rt.configure(common.ASSETS,root=run)
        verify(run)
        rt.save(run/'status.json',status)
        try:
            trained=train_official(run,cfg)
            status=read(run/'status.json')
            status.update(stage='export',training_complete=True,updated=rt.now())
            rt.save(run/'status.json',status)
            export(run,cfg)
            static_seconds=time.monotonic()-start
            status.update(stage='bound_review',static_export_passed=True,static_export_completed_at=rt.now())
            rt.save(run/'status.json',status)
            result=read(run/'result.json')
            replay=replay_official_bound(run,replay=trained)
            replay_path=run/'gaussian-run'/f'step-{result["steps"]:06d}'/'posed-replay/result.json'
            panels=write_panels(run,replay_path)
            panel_path=run/'audit/bound-panels/result.json'
            selected=set(plan['selected_frames'])
            gauss=read(run/'gaussian-run/config.json')
            need(result['state']=='completed' and result['format_passed'] is True
                 and result['reference_frame_id']==plan['reference_frame'] and result['steps']==10000
                 and {row['frame'] for row in replay['records']}=={row['frame'] for row in panels['panels']}==selected
                 and gauss['optimizer_initial_state_entries']==0 and gauss['initial_points']==20908
                 and gauss['opacity_reset_steps']==[1800,3600,5400,7200],
                 'Integrated SH0 or same-frame evidence did not pass.')
            result.update(from_scratch=False,gaussian_from_scratch=True,fit_reused=True,fit_calls=0,
                cold_video_clock=False,reused_fit_preregistration_sha256=sha(root/'preregistration.json'),
                bound_replay=replay_path.relative_to(run).as_posix(),bound_replay_sha256=sha(replay_path),
                bound_panels=panel_path.relative_to(run).as_posix(),bound_panels_sha256=sha(panel_path))
            rt.save(run/'result.json',result)
            status.update(state='completed',stage='complete',updated=rt.now(),format_passed=True,
                          reference_frame=result['reference_frame_id'],ply=str(run/result['ply']),sha256=result['sha256'])
            rt.save(run/'status.json',status)
            common.write(root/'wrapper-terminal.json',dict(state='completed',candidate_commit=COMMIT,
                result_sha256=sha(run/'result.json'),reference_frame=result['reference_frame_id'],points=result['points'],
                selected_frames=plan['selected_frames'],coverage=plan['coverage'],quality_accepted=False,
                fit_reused=True,fit_calls=0,gaussian_from_scratch=True,cold_video_clock=False,
                reuse_to_static_export_seconds=static_seconds,reuse_to_panels_seconds=time.monotonic()-start))
        except BaseException as exc:
            status=read(run/'status.json')
            status.update(state='failed',error=str(exc),updated=rt.now())
            rt.save(run/'status.json',status)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','launch','worker'))
    parser.add_argument('subject',choices=RESULT_PINS)
    args=parser.parse_args()
    if args.action=='worker':
        worker(args.subject)
    elif args.action=='launch':
        raise SystemExit(execute(args.subject))
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        state=preflight(args.subject)
        if args.action=='stage':
            stage(args.subject,state)
        else:
            print(json.dumps(dict(state='fit_reuse_verified',subject=args.subject,
                selected_frames=state[0]['selected_frames'],coverage=state[0]['coverage'])),flush=True)


if __name__=='__main__':
    main()

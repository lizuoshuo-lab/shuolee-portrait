"""Verify and replay independent archived poses against new unified deliveries."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch_unified_reused_fit_queue as queue
import review_005_formal_holdouts as five
import review_capture_formal_holdouts as capture


common=queue.common
need,read,sha,pin=queue.need,queue.read,queue.sha,queue.pin
ARCHIVE_PINS={
    '001':dict(inputs='6f9e306b98c13c991ce475525089f6a3c946e946285e1eade37cc9c528f43959',
        fitted='f35ec55f1d5bf9a399fe5381ef99e88823e7f760958bcd119f183afc54c8760a',
        fit='e88375cb03ff084ad33fd50890b5c5afe85ffadec187f6c0413eea336b596765',
        scope='99749a739e6ce9538b20ebb42a74fb726c095363f955564b300acd0b65a039fd',
        renders='7d9f2a211f85d6fe42fcf12bf12ea7350b66ed3580b99283dee150210ad7a74c'),
    '005':dict(inputs='1ed3942a80017a81e0d21c3cc215ede46267539113edbd1979e0b65117f7a6ba',
        fitted='56ee461a693d176708bad5593eac4ee96f854328e8acc2d303ce5eac0d733a3b',
        fit='0ef086056ec1459bd82ed44c2cbf5f84b59ed020091c7063131507c456019531',
        scope='2d388dbac00d104814cc85c31d42ead80c19e3ff7045850361dff6d8de55deec',
        renders='b4fad17398d85a6809712caa11e66f8059c5436ae3968109d9e7a3d9d3300dc6'),
}
PRIMARY={'001':['frame_000033','frame_000049'],'005':['frame_000027','frame_000042']}
FILES={'inputs':'inputs.json','fitted':'evaluation-fitted.npz','fit':'fit-result.json',
       'scope':'evaluation-scope.json','renders':'renders/result.json'}


def preflight(subject,require_terminal):
    queue.verify_driver()
    common.git_state(queue.COMMIT)
    seq,_,profile,verify=queue.base.imports()
    root=common.root_for(subject)
    run=root/'run'
    old=common.LAB/queue.SUBJECTS[subject]['parent']/'run'
    original=common.LAB/f'shuolee-smpl-coverage-{subject}/run'
    archive=common.LAB/f'shuolee-smpl-coverage-{subject}-formal'
    asset=common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    plan=read(root/'preregistration.json')
    need(plan['candidate_commit']==queue.COMMIT and plan['fit_reused'] is True and plan['fit_calls']==0
         and plan['baseline_run']==str(old),'Candidate does not use the registered frozen fit.')
    pins={str(archive/FILES[k]):v for k,v in ARCHIVE_PINS[subject].items()}
    pins.update({str(root/'preregistration.json'):sha(root/'preregistration.json'),
                 str(five.EVALUATOR):five.EVALUATOR_SHA})
    for name,digest in plan['staged_sha256'].items():
        path=run/name
        if name=='config.json':
            initial=dict(read(old/'config.json'),**profile())
            encoded=(json.dumps(initial,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
            resolved=dict(initial,initial_points=20908,image_cache_frames=len(plan['selected_frames']))
            need(hashlib.sha256(encoded).hexdigest()==digest and read(path) in (initial,resolved),
                 'Only the stock SH0 initialization and cache fields may be resolved after staging.')
            pins[str(path)]=sha(path)
            continue
        pin(path,digest)
        pins[str(path)]=digest
        if name in queue.base.COPY:
            pin(old/name,digest)
            pins[str(old/name)]=digest
    for path,digest in pins.items():
        pin(Path(path),digest)
    actual=read(run/'fit/official-fit-result.json')
    checked=verify(run)
    metadata=read(run/'run.json')
    fitted=read(archive/'fit-result.json')
    pose_inputs=read(archive/'inputs.json')
    scope=read(archive/'evaluation-scope.json')
    rendered=read(archive/'renders/result.json')
    held=scope['protected_frames']
    frames=list(held)
    need(frames==plan['retained_holdouts']==actual['selection']['retained_holdouts']
         and held==pose_inputs['protected_frames']==fitted['protected_frames']==rendered['protected_frames']
         and fitted['evaluation_only'] is True and fitted['gaussian_optimizer_updates']==0
         and fitted['fitted_sha256']==scope['fitted_sha256']==ARCHIVE_PINS[subject]['fitted']
         and fitted['input_receipt_sha256']==ARCHIVE_PINS[subject]['inputs']
         and rendered['fit_result_sha256']==ARCHIVE_PINS[subject]['fit'],
         'Independent evaluation fit or holdout identities changed.')
    shared=read(run/'fit/shared-body.json')
    original_shared=read(original/'fit/shared-body.json')
    need(all(shared[k]==original_shared[k] for k in ('betas','pose_latent','anchor_groups','valid_anchors'))
         and sha(run/'fit/shared-body.json')==actual['shared_body_sha256'],
         'Frozen shape or shared body latent differs.')
    beta=np.asarray(shared['betas'],dtype=np.float32).reshape(1,10)
    anchor={frame for group in shared['anchor_groups'].values() for frame in group}
    posed={path.name for path in (old/'fit/frames').iterdir() if path.is_dir()}
    need(not set(frames)&(set(plan['selected_frames'])|anchor|posed),'Heldout entered the fitting or Gaussian roster.')
    report_hashes=set()
    for r in checked['official_frame_results']:
        path=Path(r['path'])
        if not path.is_absolute():
            path=run/path
        report_hashes.add(read(path)['shared_body_target_sha256'])
        pins[str(path)]=r['sha256']
    need(report_hashes=={fitted['shared_target_sha256']},'Fitting and evaluation shared body targets differ.')
    q90=pose_inputs['frozen_q90']
    need(q90==actual['source']['confidence']['per_joint_q90']
         ==read(original/'fit/official-fit-result.json')['source']['confidence']['per_joint_q90'],
         'Frozen detector normalization differs.')
    obs=old/'data/observations.npz'
    obs0=original/'data/observations.npz'
    obs_sha=actual['source']['input_files'][str(obs)]
    need(obs_sha==pose_inputs['input_sha256'][str(obs0)],'Raw observation archive differs.')
    pins.update({str(obs):obs_sha,str(obs0):obs_sha})
    live={r['frame']:r for r in read(old/'data/inputs.json')['records']}
    original_rows={r['frame']:r for r in read(original/'data/inputs.json')['records']}
    rows=[live[f] for f in frames]
    launch=dict(candidate_commit=plan['candidate_commit'],source_sha256=plan['source_video_sha256'])
    if subject=='001':
        history=capture.benchmark(subject,held)
        evaluator=capture.load_evaluator(run,common.PROJECT,launch,held,frames,history)
    else:
        evaluator=five.load_evaluator(run,common.PROJECT,launch,held)
    helper,scoring,old_records,history_pins=evaluator.old_benchmark()
    need([r['frame'] for r in scoring]==frames,'Historical scoring order differs.')
    pins.update({str(p):h for p,h in history_pins.items()})
    with np.load(obs,allow_pickle=False) as z:
        positions=[z['frame_ids'].tolist().index(f) for f in frames]
        targets=evaluator.with_frozen_q90(seq.official,
            *(z[k][positions] for k in ('keypoints','scores','boxes','flip_disagreement')),q90)
    for i,row in enumerate(rows):
        x,y=targets[i,:,:2].T
        w,h=row['wh']
        targets[i,(x<-.5)|(x>=w-.5)|(y<-.5)|(y>=h-.5),2]=0.
    records=[]
    with np.load(archive/'evaluation-fitted.npz',allow_pickle=False) as evaluation, \
            np.load(run/'fit/fitted.npz',allow_pickle=False) as training:
        need(np.array_equal(evaluation['shared_betas'],beta) and np.array_equal(training['shared_betas'],beta)
             and evaluation['frame_ids'].tolist()==frames
             and evaluation['source_frames'].tolist()==list(held.values())
             and np.array_equal(evaluation['observed_native_keypoints'],targets),
             'Exact shared beta or regenerated native 118-point heldout targets differ.')
        for i,(row,score,prior,diag) in enumerate(zip(rows,scoring,rendered['records'],fitted['diagnostics'],strict=True)):
            frame=row['frame']
            earlier=original_rows[frame]
            need(frame==score['frame']==prior['frame']==diag['frame']
                 and all(row[k]==earlier[k]==score[k] for k in capture.SOURCE_FIELDS)
                 and np.array_equal(evaluation['K_integer_centers'][i],row['K_integer_centers'])
                 and np.array_equal(evaluation['w2c'][i],row['w2c'])
                 and evaluation['image_sha256'][i]==row['source_sha256']
                 and evaluation['mask_sha256'][i]==prior['fit_mask_sha256']==earlier['mask_sha256']
                 and prior['scoring_mask_sha256']==score['mask_sha256']
                 and prior['pose_screen_passed']==diag['pose_screen_passed']
                 and prior['pose_screen_flags']==diag['pose_screen_flags'],
                 f'Heldout RGB, full K, mask or archived pose differs: {frame}.')
            pins.update({str(Path(row['image'])):row['source_sha256'],
                         str(Path(row['mask'])):row['mask_sha256'],
                         str(Path(earlier['mask'])):earlier['mask_sha256'],
                         str(helper['relocate'](score['mask'])):score['mask_sha256'],
                         str(archive/'frames'/frame/'winner-camera-space.npz'):diag['winner_sha256'],
                         str(archive/'frames'/frame/'result.json'):diag['report_sha256']})
            role=('outside_arc_diagnostic' if frame in scope['outside_capture_arc'] else
                  'primary_pose_pass' if diag['pose_screen_passed'] else 'in_arc_pose_screen_failed')
            records.append(dict(frame=frame,role=role,source_sha256=row['source_sha256'],
                historical_fit_mask_sha256=earlier['mask_sha256'],
                reused_run_source_mask_sha256=row['mask_sha256'],
                fixed_scoring_mask_sha256=score['mask_sha256']))
    need([r['frame'] for r in records if r['role']=='primary_pose_pass']==PRIMARY[subject],
         'Primary pose-passing archive cohort changed.')
    if subject=='001':
        need(all(r['historical_fit_mask_sha256']==r['reused_run_source_mask_sha256'] for r in records),
             '001 reused archive changed native mask pixels.')
    else:
        parity=common.LAB/'shuolee-smpl-sixworker-ffprobe-005-formal/parity.json'
        pins[str(parity)]='216da3d9bff70fe291e70a186659ff8c1fac1528d122d677c3e62e1c8c2422a1'
        pin(parity,pins[str(parity)])
        p=read(parity)
        need(p['raw_observation_npz_bytes_equal'] is True and not p['evaluation_pose_mismatches']
             and p['fitted_archive_differences']==['mask_sha256']
             and all(p['mask_provenance'][r['frame']]=={
                'archived_fitting_mask_sha256':r['historical_fit_mask_sha256'],
                'candidate_preprocessed_sha256':r['reused_run_source_mask_sha256'],
                'fixed_scoring_mask_sha256':r['fixed_scoring_mask_sha256']} for r in records),
             '005 fresh mask differences cannot be reconciled with frozen scoring.')
    result=None
    snapshot=run/'gaussian-run/step-010000'
    if require_terminal:
        terminal=read(root/'wrapper-terminal.json')
        result=read(run/'result.json')
        need(terminal['state']=='completed' and terminal['result_sha256']==sha(run/'result.json')
             and result['fit_reused'] is True and result['fit_calls']==0 and result['gaussian_from_scratch'] is True
             and result['steps']==10000 and result['reference_frame_id']==plan['reference_frame'],
             'Verified unified delivery is required before native holdout replay.')
        from shuolee_portrait.bound_replay import _inputs
        manifest=read(snapshot/'bound-replay.json')
        _inputs(run,snapshot,asset,manifest)
        pin(snapshot/'bound-local.ply',manifest['bound_ply_sha256'])
        pins.update({str(run/'result.json'):terminal['result_sha256'],
                     str(snapshot/'bound-replay.json'):sha(snapshot/'bound-replay.json'),
                     str(snapshot/'bound-local.ply'):manifest['bound_ply_sha256'],
                     str(run/result['ply']):result['sha256'],str(run/result['scene']):result['scene_sha256'],
                     str(run/'export/validation.json'):sha(run/'export/validation.json')})
    for path,digest in pins.items():
        pin(Path(path),digest)
    evidence=dict(state='frozen_evaluation_inputs_verified',subject=subject,active_commit=queue.COMMIT,
        archive=str(archive),evaluation_fit_sha256=ARCHIVE_PINS[subject]['fitted'],
        shared_body_target_sha256=fitted['shared_target_sha256'],raw_observation_sha256=obs_sha,
        exact_shared_beta=True,exact_frozen_q90=True,exact_adapted_native_118_targets=True,
        exact_full_K_and_pose=True,zero_refits=True,quality_accepted=False,records=records)
    state=SimpleNamespace(seq=seq,run=metadata,result=result,asset=asset,snapshot=snapshot,
        subset=dict(rows=rows),beta=beta,target_hash=fitted['shared_target_sha256'],
        helper=helper,old_rows=scoring,old_records=old_records,pins=pose_inputs['input_sha256'])
    return evidence,evaluator,state,archive,rendered,pins


def render(subject,data):
    evidence,evaluator,state,archive,previous,pins=data
    root=common.root_for(subject)
    output=root/'formal-heldout'
    need(not output.exists(),'Preserve an existing formal replay.')
    with common.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        need(torch.cuda.device_count()==1,'Expose exactly one GPU for frozen replay.')
        with TemporaryDirectory(prefix='.formal-frozen-',dir=root) as scratch:
            stage=Path(scratch)
            for name in ('evaluation-fitted.npz','fit-result.json','inputs.json'):
                (stage/name).symlink_to(archive/name)
            rendered=evaluator.render_holdouts(state,stage)
            old={row['frame']:row for row in previous['records']}
            facts={row['frame']:row for row in evidence['records']}
            for row in rendered['records']:
                before=old[row['frame']]
                need(row['posed_camera']==dict(before['posed_camera'],pose_index=row['posed_camera']['pose_index'])
                     and row['scoring_mask_sha256']==before['scoring_mask_sha256']
                     and row['native_source_sha256']==before['native_source_sha256']
                     and row['pose_screen_flags']==before['pose_screen_flags'],
                     'Actual native replay camera, scoring mask or pose flags differ.')
                row.update(role=facts[row['frame']]['role'],old_posed_metrics=before['metrics']['posed'])
            for path,digest in pins.items():
                pin(Path(path),digest)
            output.mkdir()
            (stage/'renders').rename(output/'renders')
            common.write(output/'result.json',dict(evidence,state='formal_frozen_replay_complete',
                records=rendered['records'],optimizer_updates=0,evaluation_refits=0,
                source_pins=pins,script_sha256=sha(Path(__file__).resolve())))
    print(output/'result.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=ARCHIVE_PINS)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--render',action='store_true')
    mode.add_argument('--terminal-check',action='store_true')
    args=parser.parse_args()
    if not args.render:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    data=preflight(args.subject,args.render or args.terminal_check)
    if args.render:
        render(args.subject,data)
    else:
        print(json.dumps(data[0],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

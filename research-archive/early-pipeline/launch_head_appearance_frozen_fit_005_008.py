"""Extend the registered frozen-fit head appearance transfer to 005 and 008."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_frozen_fit as prior


ACCEPTANCE = prior.ACCEPTANCE.with_name('smpl-head-appearance-transfer-extension-preregistration.json')
ACCEPTANCE_SHA = 'c9b0d89bb6d5b9c7b6693c3cc0abbcf76e5aaf232e6b3530ce396c5662f7c913'
COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
PRIOR_SHA = '03742754dae69c35608830827a06e3c749f1637857a87f7c5c0f4dbb83ce188d'
ORDER = ('005','008')
base, common = prior.base, prior.common
need, read, sha, pin = prior.need, prior.read, prior.sha, prior.pin
checked_sha = prior.checked_sha


def acceptance(digest):
    need(digest == ACCEPTANCE_SHA,'Use the exact registered extension plan.')
    pin(Path(prior.__file__),PRIOR_SHA)
    pin(prior.ACCEPTANCE,prior.ACCEPTANCE_SHA)
    pin(ACCEPTANCE,digest)
    value = read(ACCEPTANCE)
    need(value['state'] == 'registered_before_005_008_common_candidate_training'
         and value['baseline_commit'] == base.COMMIT and value['candidate_commit'] == COMMIT
         and value['prior_transfer_plan_sha256'] == prior.ACCEPTANCE_SHA and tuple(value['order']) == ORDER
         and value['body_steps'] == 10000 and value['body_schedule_steps'] == 18000
         and value['head_appearance_steps'] == 600 and value['face_weight'] == .05
         and value['sh_degree'] == 0 and value['no_refit'] is True and value['use_exact_archived_fit_and_camera'] is True,
         'Registered transfer v2 differs from the common frozen-fit workflow.')
    return value


def prepare(subject, digest):
    policy = acceptance(digest)
    changed = prior.check_source(COMMIT)
    _, selector, profile, verify = prior.imports()
    spec = prior.queue.SUBJECTS[subject]
    registered = policy['subjects'][subject]
    source = base.common.root_for(subject)
    old = source/'run'
    pins = {str(ACCEPTANCE):digest, str(prior.ACCEPTANCE):prior.ACCEPTANCE_SHA, str(Path(prior.__file__)):PRIOR_SHA}
    for record in [*registered.values(),*policy['previous_deliveries'].values()]:
        if isinstance(record,dict) and set(record) == {'path','sha256'}:
            pins[record['path']] = checked_sha(Path(record['path']),record['sha256'])
    for label,path in (('baseline_wrapper',source/'wrapper-terminal.json'),('baseline_terminal',old/'result.json'),
                       ('fit',old/'fit/fitted.npz'),('training_inputs',old/'data/training-inputs.json'),
                       ('selection',old/'fit/selection.json'),('training_cameras',old/prior.CAMERA_FILES[0]),
                       ('portable_camera',old/prior.CAMERA_FILES[1])):
        need(registered[label]['path'] == str(path) and str(path) in pins, 'Registered source path differs.')
    result = read(old/'result.json')
    terminal = read(source/'wrapper-terminal.json')
    receipt = read(old/'fit/official-fit-result.json')
    selection = read(old/'fit/selection.json')
    metadata, cfg = (read(old/name) for name in ('run.json','config.json'))
    rows = read(old/'data/training-inputs.json')['records']
    selected, held, reference = (selection[key] for key in ('selected_frames','retained_holdouts','reference_frame'))
    need(result['state'] == terminal['state'] == 'completed' and terminal['candidate_commit'] == base.COMMIT
         and terminal['result_sha256'] == pins[str(old/'result.json')] and result['format_passed'] is True
         and result['fit_reused'] is True and result['fit_calls'] == 0 and result['steps'] == 10000
         and verify(old)['fit_sha256'] == pins[str(old/'fit/fitted.npz')] and receipt['selection'] == selection
         and selection['coverage_basis'] == selector.SOURCE_BIN_COVERAGE_BASIS
         and selection['arc'] == spec['arc'] == registered['arc'] and selection['coverage'] == registered['coverage']
         and metadata['assets'] == str(common.ASSETS) and metadata['video_sha256'] == spec['video']
         and metadata['camera_config_sha256'] == spec['camera']
         and reference == spec['reference'] == registered['reference_frame'] == result['reference_frame_id']
         and tuple(held) == spec['holdouts'] and held == registered['retained_holdouts'] and set(held).isdisjoint(selected)
         and selected == registered['selected_frames'] and len(set(selected)) == len(rows) == result['frames'] == spec['frames']
         and [reference]+[frame for frame in selected if frame != reference] == [row['frame'] for row in rows],
         'The original unified pose, shared shape, native source or protected roster changed.')
    formal = registered['formal_holdouts']
    need(set(formal) <= set(held),'Formal holdouts must stay out of the original fit and training.')
    if formal:
        gate = read(Path(registered['prior_formal_receipt']['path']))
        passed = {row['frame'] for row in gate['records'] if row['pose_screen_passed'] is True}
        need(gate['exact_full_K_and_pose'] is True and set(formal) <= passed,'Unavailable exact formal holdout poses.')
    replay_path = old/result['bound_replay']
    pins[str(replay_path)] = checked_sha(replay_path,result['bound_replay_sha256'])
    replay = read(replay_path)
    need(all(all(record['camera'][key] == row[key] for key in ('frame','K','w2c','wh'))
             for row,record in zip(rows,replay['records'],strict=True)), 'Archived native camera records differ.')
    cameras = {name:pins[str(old/name)] for name in prior.CAMERA_FILES}
    need(cameras[prior.CAMERA_FILES[0]] == replay['training_camera_sha256'],'Training camera lineage changed.')
    pins.update({str(old/name):sha(old/name) for name in ('run.json','config.json',*base.COPY) if str(old/name) not in pins})
    for row in rows:
        for field,digest_field in (('image','source_sha256'),('mask','mask_sha256')):
            pins[row[field]] = checked_sha(Path(row[field]),row[digest_field])
    effective = profile()
    stock = read(old/'gaussian-run/config.json')
    need(effective['head_appearance_steps'] == 600 and effective['head_appearance_weight'] == .05
         and all(stock[key] == value for key,value in effective.items()
                 if key not in ('head_appearance_steps','head_appearance_weight','initial_points','image_cache_frames'))
         and stock['optimizer_initial_state_entries'] == 0 and stock['initial_points'] == 20908,
         'Common candidate changed the initial-body Gaussian profile.')
    plan = dict(schema='shuolee-head-appearance-frozen-fit-transfer-v1',subject=subject,candidate_commit=COMMIT,
        baseline_commit=base.COMMIT,baseline_run=str(old),baseline_result_sha256=pins[str(old/'result.json')],
        acceptance_plan_sha256=digest,changed_paths=changed,camera_pins=cameras,formal_holdouts=formal,
        source_video_sha256=spec['video'],camera_config_sha256=spec['camera'],selected_frames=selected,training_slots=len(rows),
        retained_holdouts=held,reference_frame=reference,coverage=selection['coverage'],full_bin_coverage=selection['full_bin_coverage'],
        fit_reused=True,fit_calls=0,cold_video_clock=False,gaussian_from_scratch=True,fresh_gaussian_optimizer=True,
        head_appearance_steps=600,head_appearance_weight=.05,input_sha256=pins,asset_sha256=common.official_pins(),
        runtime=common.cpu_runtime(common.clean_env()),code_sha256={name:sha(prior.PROJECT/name) for name in common.SOURCE_FILES},
        script_sha256=sha(Path(__file__)),reused_transfer_launcher_sha256=PRIOR_SHA,
        launcher_helper_sha256=prior.queue.HELPER,reused_driver_sha256=prior.queue.DRIVER)
    return plan,None,(old,receipt,selection,metadata,cfg),effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','launch','worker'))
    parser.add_argument('subject',choices=ORDER)
    parser.add_argument('--acceptance-sha256',default=ACCEPTANCE_SHA)
    args = parser.parse_args()
    if args.action == 'worker':
        prior.worker(args.subject)
        return
    if args.action == 'launch':
        plan = read(prior.root_for(args.subject)/'preregistration.json')
        digest = plan['acceptance_plan_sha256']
        need(args.acceptance_sha256 == digest,'Registered policy SHA differs.')
        acceptance(digest)
        prior.check_source(COMMIT)
        if args.subject == ORDER[1]:
            earlier = prior.root_for(ORDER[0])
            done = read(earlier/'wrapper-terminal.json')
            need(done['state'] == 'completed' and done['candidate_commit'] == COMMIT,'Complete 005 delivery first.')
            pin(earlier/'run/result.json',done['result_sha256'])
        launch = prior.rebound(base.execute,common=common,COMMIT=COMMIT,__file__=__file__)
        raise SystemExit(launch(args.subject))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    state = prepare(args.subject,args.acceptance_sha256)
    if args.action == 'stage':
        prior.stage(args.subject,state)
    else:
        plan = state[0]
        print(json.dumps(dict(state='fit_reuse_verified',subject=args.subject,candidate_commit=COMMIT,
            reference_frame=plan['reference_frame'],selected_frames=plan['selected_frames'],formal_holdouts=plan['formal_holdouts'],
            fit_calls=0,gpu_calls=0)),flush=True)


if __name__ == '__main__':
    main()

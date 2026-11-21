"""Validate the common SH0 head tail on exact existing 001 or 016 official fits."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_unified_reused_fit_queue as queue


base = queue.base
need, read, sha, pin = base.need, base.read, base.sha, base.pin
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-appearance-20261110')
HEAD_MODULE = 'src/shuolee_portrait/head_appearance.py'
ACCEPTANCE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/'
                  'shuolee/current/robust-portrait-122/smpl-head-appearance-transfer-preregistration.json')
ACCEPTANCE_SHA = 'd8d9ff902f1f0d312233d14076548e48554ed766f2ba29de1d6382ccf4dde55b'
CAMERA_FILES = ('gaussian-run/training-cameras.json','export/cameras.json')
BASELINES = {
    '001': dict(result='ff7194a163a71b0526bdf7a587822057775c5b09143fc62a94f2646d5729a712',
                fit='76562e2d61da381abe1a318491d3288af1591c58fe124c1a5068e64437b01467',
                rows='2f34216373600983a8a8933c551a115810141ce2a24a99d5906185cef4c172ae',
                selection='2e09381484fb17e5ff30b94833c342ea3e2559edda0e0a5dba11efaedf7ca2d9',
                reference='frame_000056', holdouts=['frame_000013','frame_000033','frame_000049','frame_000065']),
    '016': dict(result='a1042ccba8a4399a97497bedcd3c049fb15f698e434f746213e263340bebc696',
                fit='8a98a5bc4c162844d8d505c0da1325e65f44c2a4d693e150705899f139bb7b2b',
                rows='0e9a13864bba32354dba862f18a4a2a59b2f32176a151ec87923d669d5a9a530',
                selection='ce82eed1133265e7bfb73794f341e97be384f22f48498bd8410693d0b86d0842',
                reference='frame_000027', holdouts=['frame_000011','frame_000028','frame_000040','frame_000052']),
}


def root_for(subject):
    return base.common.LAB/f'shuolee-smpl-head-appearance-{subject}'


def rebound(function, **values):
    return FunctionType(function.__code__, dict(function.__globals__, **values), function.__name__)


def checked_sha(path, digest):
    pin(path,digest)
    return digest


common = SimpleNamespace(**(vars(base.common) | dict(PROJECT=PROJECT, root_for=root_for,
    SOURCE_FILES=(*base.common.SOURCE_FILES, HEAD_MODULE),
    git_state=rebound(base.common.git_state, PROJECT=PROJECT),
    official_pins=rebound(base.common.official_pins, PROJECT=PROJECT))))
imports = rebound(base.imports, common=common)
stage = rebound(base.stage, common=common, imports=imports)


def check_source(commit):
    queue.verify_driver()
    common.git_state(commit)
    pointer = (PROJECT/'.git').read_text(encoding='utf-8').strip().removeprefix('gitdir: ')
    gitdir = subprocess.check_output(['/usr/bin/wslpath','-u',pointer], text=True).strip()
    git = ['/usr/bin/git','--git-dir',gitdir,'--work-tree',str(PROJECT)]
    subprocess.run([*git,'merge-base','--is-ancestor',base.COMMIT,commit], check=True)
    changed = subprocess.check_output([*git,'diff','--name-only',base.COMMIT,commit], text=True).splitlines()
    allowed = {HEAD_MODULE, 'src/shuolee_portrait/training.py', 'src/shuolee_portrait/export.py',
               'src/shuolee_portrait/bound_review_panels.py',
               'src/shuolee_portrait/sequence_pipeline.py', 'tests/test_head_appearance.py'}
    need(HEAD_MODULE in changed and set(changed) <= allowed, 'Candidate changes exceed the scoped head tail.')
    return changed


def prepare(subject, commit):
    changed = check_source(commit)
    _, selector, profile, verify = imports()
    spec = BASELINES[subject]
    pin(ACCEPTANCE,ACCEPTANCE_SHA)
    acceptance = read(ACCEPTANCE)
    registered = acceptance['subjects'][subject]
    need(acceptance['state'] == 'registered_before_common_candidate_training' and acceptance['baseline_commit'] == base.COMMIT
         and acceptance['order'] == list(BASELINES) and acceptance['body_steps'] == 10000
         and acceptance['body_schedule_steps'] == 18000 and acceptance['head_appearance_steps'] == 600
         and acceptance['face_weight'] == .05 and acceptance['sh_degree'] == 0 and acceptance['no_refit'] is True
         and acceptance['use_exact_archived_fit_and_camera'] is True, 'Registered common-transfer scope changed.')
    old_root = base.common.root_for(subject)
    old = old_root/'run'
    pins = {str(ACCEPTANCE):ACCEPTANCE_SHA}
    for key,label,name in (('result','baseline_terminal','result.json'),('fit','fit','fit/fitted.npz'),
                           ('rows','training_inputs','data/training-inputs.json'),('selection','selection','fit/selection.json')):
        need(registered[label] == {'path':str(old/name), 'sha256':spec[key]}, 'Registered frozen source differs.')
        pins[str(old/name)] = checked_sha(old/name,spec[key])
    gate = registered['baseline_holdout_receipt' if subject == '001' else 'prior_reference_face_receipt']
    pilot = acceptance['pilot_evidence']
    for proof in (gate,pilot):
        pins[proof['path']] = checked_sha(Path(proof['path']),proof['sha256'])
    prior = read(old/'result.json')
    terminal = read(old_root/'wrapper-terminal.json')
    receipt = read(old/'fit/official-fit-result.json')
    selection = read(old/'fit/selection.json')
    metadata, cfg = (read(old/name) for name in ('run.json','config.json'))
    rows = read(old/'data/training-inputs.json')['records']
    selected, holdouts, reference = (selection[key] for key in ('selected_frames','retained_holdouts','reference_frame'))
    need(prior['state'] == terminal['state'] == 'completed' and terminal['candidate_commit'] == base.COMMIT
         and terminal['result_sha256'] == spec['result'] and prior['format_passed'] is True
         and prior['fit_reused'] is True and prior['fit_calls'] == 0 and prior['steps'] == 10000
         and metadata['assets'] == str(common.ASSETS) and verify(old)['fit_sha256'] == spec['fit']
         and receipt['selection'] == selection and selection['arc'] == 'half'
         and selection['coverage_basis'] == selector.SOURCE_BIN_COVERAGE_BASIS
         and reference == spec['reference'] == prior['reference_frame_id'] == terminal['reference_frame'] == registered['reference_frame']
         and holdouts == spec['holdouts'] and set(holdouts).isdisjoint(selected)
         and len(set(selected)) == prior['frames'] == len(rows) == 24 and selected == registered['selected_frames']
         and [reference]+[frame for frame in selected if frame != reference] == [row['frame'] for row in rows],
         'Existing unified fit, native roster or protected holdouts changed.')
    replay_path = old/prior['bound_replay']
    pins[str(replay_path)] = checked_sha(replay_path,prior['bound_replay_sha256'])
    replay = read(replay_path)
    need(all(all(record['camera'][key] == row[key] for key in ('frame','K','w2c','wh'))
             for row,record in zip(rows,replay['records'],strict=True)), 'Frozen native cameras no longer match delivered views.')
    camera_pins = {CAMERA_FILES[0]:checked_sha(old/CAMERA_FILES[0],replay['training_camera_sha256']),
                   CAMERA_FILES[1]:sha(old/CAMERA_FILES[1])}
    pins.update({str(old/name):digest for name,digest in camera_pins.items()})
    prior_gate = read(Path(gate['path']))
    if subject == '001':
        passed = {item['frame'] for item in prior_gate['records'] if item['pose_screen_passed'] is True}
        need(prior_gate['exact_full_K_and_pose'] is True and set(registered['formal_holdouts']) <= (set(holdouts)&passed),
             'Registered independent holdout poses are unavailable.')
    else:
        need(prior_gate['native_K'] == rows[0]['K'] and prior_gate['native_w2c'] == rows[0]['w2c']
             and prior_gate['reference'] == reference and not registered['formal_holdouts'],
             'Registered reference camera changed or an unseen view was promoted.')
    pins.update({str(old/name):sha(old/name) for name in ('run.json','config.json',*base.COPY) if str(old/name) not in pins})
    for row in rows:
        for field in ('image','mask'):
            pins[row[field]] = checked_sha(Path(row[field]), row['source_sha256' if field == 'image' else 'mask_sha256'])
    effective = profile()
    prior_profile = read(old/'gaussian-run/config.json')
    need(effective['head_appearance_steps'] == 600 and effective['head_appearance_weight'] == .05
         and all(prior_profile[key] == value for key,value in effective.items()
                 if key not in ('head_appearance_steps','head_appearance_weight','initial_points','image_cache_frames'))
         and prior_profile['optimizer_initial_state_entries'] == 0 and prior_profile['initial_points'] == 20908,
         'New workflow changed the frozen body Gaussian profile.')
    plan = dict(schema='shuolee-head-appearance-frozen-fit-transfer-v1', subject=subject, candidate_commit=commit,
        baseline_commit=base.COMMIT, baseline_run=str(old), baseline_result_sha256=spec['result'], changed_paths=changed,
        acceptance_plan_sha256=ACCEPTANCE_SHA, camera_pins=camera_pins, formal_holdouts=registered['formal_holdouts'],
        source_video_sha256=metadata['video_sha256'], camera_config_sha256=metadata['camera_config_sha256'],
        selected_frames=selected, retained_holdouts=holdouts, reference_frame=reference,
        coverage=selection['coverage'], full_bin_coverage=selection['full_bin_coverage'], fit_reused=True,
        fit_calls=0, cold_video_clock=False, gaussian_from_scratch=True, fresh_gaussian_optimizer=True,
        head_appearance_steps=600, head_appearance_weight=.05,
        code_sha256={name:sha(PROJECT/name) for name in common.SOURCE_FILES},
        input_sha256=pins, asset_sha256=common.official_pins(), runtime=common.cpu_runtime(common.clean_env()),
        script_sha256=sha(Path(__file__)), launcher_helper_sha256=queue.HELPER, reused_driver_sha256=queue.DRIVER)
    return plan,None,(old,receipt,selection,metadata,cfg),effective


def worker(subject):
    root = root_for(subject)
    run = root/'run'
    plan = read(root/'preregistration.json')
    with common.LEASE.open('r+') as lease:
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError('Run the launch action to hold the machine GPU lease.')
    need(dict(os.environ) == common.clean_env() and sys.flags.isolated == 1, 'Worker environment changed.')
    seq, _, _, verify = imports()
    from shuolee_portrait.bound_review_panels import write_panels
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.export import export
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.training import replay_official_bound, train_official

    rt = seq.rt
    cfg = read(run/'config.json')
    start = time.monotonic()

    def progress(state, step, **fields):
        rt.save(run/'status.json', dict(state=state, stage=step, updated=rt.now(), fit_reused=True,
                                      fitting_complete=True, quality_accepted=False, **fields))

    with gpu_lock():
        rt.configure(common.ASSETS, root=run)
        verify(run)
        progress('running','gaussian_inputs',training_started=False)
        try:
            trained = train_official(run,cfg)
            progress('running','export',training_complete=True)
            export(run,cfg)
            static_seconds = time.monotonic()-start
            result = read(run/'result.json')
            gaussian = read(run/'gaussian-run/result.json')
            snapshot = snapshot_name(gaussian)
            appearance = gaussian['head_appearance']
            for name,digest in plan['camera_pins'].items():
                pin(run/name,digest)
            need(result['state'] == 'completed' and result['format_passed'] is True and result['steps'] == 10000
                 and result['reference_frame_id'] == plan['reference_frame'] and result['final_snapshot'] == snapshot
                 and result['head_appearance'] == appearance and (appearance['state'] != 'completed'
                     or appearance['steps'] == 600 and appearance['frozen_ply_bytes_verified'] is True),
                 'Portable delivery does not bind the validated final appearance snapshot.')
            progress('running','bound_review',static_export_passed=True,static_export_completed_at=rt.now())
            replay = replay_official_bound(run,replay=trained)
            replay_path = run/'gaussian-run'/snapshot/'posed-replay/result.json'
            panels = write_panels(run,replay_path)
            panel_path = run/'audit/bound-panels/result.json'
            expected = set(plan['selected_frames'])
            need({row['frame'] for row in replay['records']} == {row['frame'] for row in panels['panels']} == expected
                 and len(replay['records']) == len(panels['panels']) == len(expected), 'Selected pose review roster changed.')
            result.update(from_scratch=False, gaussian_from_scratch=True, fit_reused=True, fit_calls=0, cold_video_clock=False,
                reused_fit_preregistration_sha256=sha(root/'preregistration.json'),
                bound_replay=replay_path.relative_to(run).as_posix(), bound_replay_sha256=sha(replay_path),
                bound_panels=panel_path.relative_to(run).as_posix(), bound_panels_sha256=sha(panel_path))
            rt.save(run/'result.json',result)
            progress('completed','complete',static_export_passed=True,format_passed=True,
                     reference_frame=result['reference_frame_id'],ply=str(run/result['ply']),sha256=result['sha256'])
            common.write(root/'wrapper-terminal.json',dict(state='completed', candidate_commit=plan['candidate_commit'],
                result_sha256=sha(run/'result.json'), reference_frame=result['reference_frame_id'], points=result['points'],
                selected_frames=plan['selected_frames'], coverage=plan['coverage'], final_snapshot=snapshot,
                head_appearance=appearance, quality_accepted=False, fit_reused=True, fit_calls=0,
                gaussian_from_scratch=True, cold_video_clock=False, reuse_to_static_export_seconds=static_seconds,
                reuse_to_panels_seconds=time.monotonic()-start))
        except Exception as error:
            progress('failed',read(run/'status.json')['stage'],error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','launch','worker'))
    parser.add_argument('subject',choices=tuple(BASELINES))
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.subject)
    elif args.action == 'launch':
        plan = read(root_for(args.subject)/'preregistration.json')
        commit = plan['candidate_commit']
        need(args.commit is None or args.commit == commit, 'Commit differs from the staged transfer.')
        check_source(commit)
        if args.subject == '016':
            first = root_for('001')
            terminal = read(first/'wrapper-terminal.json')
            need(terminal['state'] == 'completed' and terminal['candidate_commit'] == commit,
                 'Complete the first 001 integration delivery before 016.')
            pin(first/'run/result.json',terminal['result_sha256'])
        invoke = rebound(base.execute, common=common, COMMIT=commit, __file__=__file__)
        raise SystemExit(invoke(args.subject))
    else:
        if not args.commit:
            parser.error('--commit is required to verify exact candidate source')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        state = prepare(args.subject,args.commit)
        if args.action == 'stage':
            stage(args.subject,state)
        else:
            plan = state[0]
            print(json.dumps(dict(state='fit_reuse_verified', subject=args.subject, candidate_commit=args.commit,
                                  selected_frames=plan['selected_frames'], retained_holdouts=plan['retained_holdouts'],
                                  fit_calls=0, gpu_calls=0)),flush=True)


if __name__ == '__main__':
    main()

"""Time one fresh 001 video-to-package run through the exact official 762d6d0 CLI."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_frozen_fit as transfer


common = transfer.common
need, read, sha, pin = transfer.need, transfer.read, transfer.sha, transfer.pin
COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
ROOT = common.LAB/'shuolee-smpl-head-appearance-cold-001'
RUN = ROOT/'run'
PRIOR = common.LAB/'shuolee-smpl-coverage-001'
VIDEO = common.SOURCE/'video1.mp4'
CAMERA = PRIOR/'camera.json'
DEPLOYMENT = PRIOR/'deployment.json'
SOURCE_PINS = {
    VIDEO:'32be30636adabfc4c25a1c8387cf19302d3ca39b4f040072081646d36523317c',
    CAMERA:'28c838033eb699c2497ba292793516596f1dfaba7fb8d4ae64fc9a11316a65b5',
    DEPLOYMENT:'c470238761681ae93530c126f01b75ab7e80caf2e8d5f95999512b7774cc1330',
}
HOLDOUTS = ['frame_000013','frame_000033','frame_000049','frame_000065']
HELD_SOURCE = [96,256,384,513]
ENDPOINT = transfer.root_for('008')


def command():
    return [str(common.ENV/'bin/python'),'-I','-B','-u',str(common.PROJECT/'run_smplx.py'),'train',str(VIDEO),
            '--deployment',str(DEPLOYMENT),'--camera-config',str(CAMERA),'--output',str(RUN)]


def check():
    need(sys.platform == 'linux' and sys.flags.isolated == 1 and Path(sys.prefix).resolve() == common.ENV.resolve(),
         'Use the pinned WSL Python in isolated mode.')
    changed = transfer.check_source(COMMIT)
    for path,digest in SOURCE_PINS.items():
        pin(path,digest)
    camera, deployment = read(CAMERA), read(DEPLOYMENT)
    need(camera['capture_layout'] == dict(arc='half',video_sha256=SOURCE_PINS[VIDEO],front_source_ranges=[[168,489]],
            rear_source_ranges=[],held_source_indices=HELD_SOURCE) and camera['sampling']['sample_fps'] == 3
         and camera['intrinsics']['image_wh'] == [1520,2704] and camera['intrinsics']['input_domain'] == 'rectified_pinhole'
         and camera['intrinsics']['pixel_convention'] == 'integer_centers'
         and camera['official_smplifyx'] == {'shape_policy':{'mode':'fit_reference'}}
         and deployment == common.deployment(PRIOR), 'The archived official raw-video configuration changed.')
    need(min(shutil.disk_usage(common.LAB).free,shutil.disk_usage('/mnt/c').free) > 8*1024**3,
         'The fresh output needs more than 8 GiB of available local disk.')
    return dict(schema='shuolee-head-appearance-cold-video-v1',subject='001',candidate_commit=COMMIT,changed_paths=changed,
        code_sha256={name:sha(common.PROJECT/name) for name in common.SOURCE_FILES},
        wrapper_sha256=sha(Path(__file__)),reused_cold_helper_sha256=transfer.queue.HELPER,
        source_sha256={str(path):digest for path,digest in SOURCE_PINS.items()},
        entry_command=command(),entry_environment=common.clean_env(),runtime=common.cpu_runtime(common.clean_env()),
        official_and_asset_sha256=common.official_pins(),output=str(RUN),candidate_native_samples=70,fit_candidates=39,
        shape_anchors=24,expected_holdouts=HOLDOUTS,expected_held_source_indices=HELD_SOURCE,automatic_reference=True,
        no_preprocessing_fit_or_gaussian_reuse=True,steps=10000,head_steps=600,schedule_steps=18000,sh_degree=0,
        cpu_fit_workers=6,ffprobe_decoder_threads=4,initial_points=20908,max_wall_seconds=common.MAX_WALL_S,
        budget_seconds=900.,requires_008_verified_delivery=str(ENDPOINT/'delivery-review.json'),
        input_domain='Native 1520x2704 already rectified. Conditional K and model-relative camera gauge.',quality_accepted=False)


def after_008(digest):
    path = ENDPOINT/'delivery-review.json'
    pin(path,digest)
    receipt = read(path)
    for key,expected in (('terminal',ENDPOINT/'run/result.json'),('wrapper',ENDPOINT/'wrapper-terminal.json')):
        item = receipt[key]
        need(Path(item['path']) == expected, '008 delivery receipt points outside the registered transfer.')
        pin(expected,item['sha256'])
    terminal = read(ENDPOINT/'run/result.json')
    wrapper = read(ENDPOINT/'wrapper-terminal.json')
    need(wrapper['state'] == terminal['state'] == 'completed' and wrapper['candidate_commit'] == COMMIT
         and wrapper['result_sha256'] == receipt['terminal']['sha256'] and terminal['format_passed'] is True
         and terminal['head_appearance']['state'] == 'completed' and terminal['head_appearance']['steps'] == 600
         and receipt['scene'] == {'path':str(ENDPOINT/'run'/terminal['scene']), 'sha256':terminal['scene_sha256']},
         '008 has no verified complete transfer delivery on the exact candidate.')
    pin(Path(receipt['scene']['path']),receipt['scene']['sha256'])
    return {'path':str(path),'sha256':digest,'result_sha256':receipt['terminal']['sha256']}


def located(relative):
    path = (RUN/relative).resolve()
    need(path.is_relative_to(RUN.resolve()),'Unexpected cold artifact outside the fresh run.')
    return path


def check_fit_desc(descriptor, folder):
    path = located(descriptor['path'])
    need(path.is_relative_to(RUN/folder) and path.parent.name == descriptor['frame_id'], 'Fit or shape-anchor output was reused.')
    pin(path,descriptor['sha256'])


def validate(plan, status, spawned, elapsed):
    result, run_input, selection, fit, shared, gaussian, profile, initialization = (read(RUN/name) for name in (
        'result.json','run.json','fit/selection.json','fit/official-fit-result.json','fit/shared-body.json',
        'gaussian-run/result.json','gaussian-run/config.json','gaussian-run/preflight.json'))
    selected, held = selection['selected_frames'], selection['retained_holdouts']
    native_rows = read(RUN/'data/inputs.json')['records']
    intrinsics = read(CAMERA)['intrinsics']
    need(status['state'] == result['state'] == gaussian['state'] == 'completed' and status['static_export_passed'] is True
         and result['format_passed'] is True and all(v['quality_accepted'] is False for v in (status,result,gaussian,fit))
         and result['from_scratch'] is gaussian['from_scratch'] is True
         and run_input['video'] == str(VIDEO) and run_input['video_sha256'] == SOURCE_PINS[VIDEO]
         and run_input['camera_config_sha256'] == SOURCE_PINS[CAMERA] and run_input['reference_frame'] is None
         and len(native_rows) == plan['candidate_native_samples'] and read(RUN/'camera.json') == read(CAMERA)
         and fit['new_fitting_started'] is True and fit['execution_kind'] == 'fresh_official_sequence'
         and fit['fitted_candidate_frames'] == plan['fit_candidates'] and fit['selection'] == selection
         and len(shared['anchor_receipts']) == sum(map(len,shared['anchor_groups'].values())) == plan['shape_anchors']
         and len(set(selected)) == len(selected) == len(fit['official_frame_results']) == result['frames']
         and set(selected).isdisjoint(held) and held == HOLDOUTS and selection['arc'] == 'half'
         and selection['reference_frame'] == result['reference_frame_id'] == status['reference_frame'],
         'The terminal run does not prove the declared fresh preprocessing and official fitting path.')
    by_id = {row['frame']:row for row in native_rows}
    need(len(by_id) == len(native_rows) and [by_id[frame]['source_frame_index'] for frame in held] == HELD_SOURCE
         and all(row['wh'] == intrinsics['image_wh'] and row['K_integer_centers'] == intrinsics['K'] for row in native_rows),
         'Cold native heldout source identities or camera mapping changed.')
    for row in native_rows:
        for field,digest in (('image','source_sha256'),('mask','mask_sha256')):
            pin(located(row[field]),row[digest])
    for name in ('data/inputs.json','data/observations.npz'):
        path = RUN/name
        pin(path,fit['source']['input_files'][str(path)])
    for descriptor in fit['official_frame_results']:
        check_fit_desc(descriptor,'fit/frames')
    for descriptor in shared['anchor_receipts']:
        check_fit_desc(descriptor,'fit/shape-anchors')
    for name,digest in (('fit/fitted.npz',fit['fit_sha256']),('data/training-inputs.json',fit['training_inputs_sha256']),
                        ('fit/shared-body.json',fit['shared_body_sha256'])):
        pin(RUN/name,digest)
    appearance = gaussian['head_appearance']
    expected_snapshot = 'step-010000-head-000600'
    need(result['steps'] == gaussian['steps'] == sum(gaussian['per_frame_updates']) == plan['steps']
         and gaussian['schedule_steps'] == plan['schedule_steps'] and result['sh_degree'] == gaussian['sh_degree'] == 0
         and profile['optimizer_initial_state_entries'] == initialization['optimizer_updates'] == 0
         and profile['initial_points'] == plan['initial_points'] and profile['snapshot_steps'] == [1000,10000]
         and profile['opacity_reset_steps'] == [1800,3600,5400,7200] and initialization['finite_backward'] is True
         and gaussian['stock_snapshot'] == 'step-010000' and result['final_snapshot'] == gaussian['final_snapshot'] == expected_snapshot
         and appearance == result['head_appearance'] and appearance['state'] == 'completed'
         and appearance['steps'] == sum(appearance['frame_draws'].values()) == plan['head_steps']
         and set(appearance['frame_draws']) == set(selected) and appearance['snapshot'] == expected_snapshot
         and appearance['frozen_ply_bytes_verified'] is True, 'The required fresh 10k body plus 600 head tail did not complete.')
    artifacts = {}
    for key,digest_key in (('ply','sha256'),('scene','scene_sha256'),('bound_replay','bound_replay_sha256'),
                           ('bound_panels','bound_panels_sha256')):
        path = located(result[key])
        pin(path,result[digest_key])
        artifacts[key] = {'path':str(path),'sha256':result[digest_key]}
    camera = located(result['camera'])
    camera_hash = sha(camera)
    with zipfile.ZipFile(located(result['scene'])) as bundle:
        with bundle.open('cameras.json') as packed_camera, bundle.open('point_cloud/iteration_10000/point_cloud.ply') as packed_ply:
            need(common.stream_sha(packed_camera) == camera_hash and common.stream_sha(packed_ply) == result['sha256'],
                 'Scene ZIP camera or PLY differs from the verified portable artifacts.')
    artifacts['camera'] = {'path':str(camera),'sha256':camera_hash}
    replay, panels = read(located(result['bound_replay'])), read(located(result['bound_panels']))
    need(len(replay['records']) == len(panels['panels']) == len(selected)
         and {row['frame'] for row in replay['records']} == {row['frame'] for row in panels['panels']} == set(selected),
         'The exact native bound replay or panel roster changed.')
    milestone = RUN/'gaussian-run/step-001000'
    parity_path = milestone/'native-ply-parity.json'
    parity = read(parity_path)
    need(parity['passed'] is True and parity['reference_frame_id'] == result['reference_frame_id']
         and (milestone/'reference-render.png').is_file(), 'No verified continuous 1,000-step snapshot.')
    pin(milestone/'world.ply',parity['sha256'])
    pin(milestone/'reference-mesh.npz',parity['mesh_sha256'])
    with (RUN/'gaussian-run/loss.jsonl').open() as stream:
        progress = {row['step']:row for row in map(json.loads,stream)}
    to = lambda stamp: (datetime.fromisoformat(stamp)-spawned).total_seconds()
    static = to(status['static_export_completed_at'])
    one_k = to(progress[1000]['updated'])
    one_k_verified = parity_path.stat().st_mtime_ns/1e9-spawned.timestamp()
    body_end = to(progress[10000]['updated'])
    replay_panel_end = to(status['updated'])
    need(0 < one_k <= one_k_verified <= body_end <= static <= replay_panel_end <= elapsed+1.,
         'Command-spawn, 1k, static export, or terminal panel timing is inconsistent.')
    for name,digest in plan['code_sha256'].items():
        pin(common.PROJECT/name,digest)
    for name,digest in plan['source_sha256'].items():
        pin(Path(name),digest)
    return dict(result_sha256=sha(RUN/'result.json'),reference_frame=result['reference_frame_id'],selected_frames=selected,
        coverage=selection['coverage'],points=result['points'],artifacts=artifacts,head_appearance=appearance,
        raw_video_to_static_export_seconds=static,static_export_within_15_minutes=static<=plan['budget_seconds'],
        full_cli_within_15_minutes=elapsed<=plan['budget_seconds'],replay_and_panels_seconds=replay_panel_end-static,
        command_to_bound_review_complete_seconds=replay_panel_end,body_training_start_to_1000_seconds=progress[1000]['seconds'],
        body_training_start_to_10000_seconds=progress[10000]['seconds'],command_to_1000_step_seconds=one_k,
        command_to_verified_1000_snapshot_seconds=one_k_verified,command_to_body_step_10000_seconds=body_end,
        official_fit_seconds=fit['elapsed_seconds'],snapshot_1000_parity_sha256=sha(parity_path),from_scratch=True,
        fixed_fit_reused=False,checkpoint_reused=False,quality_accepted=False)


def launch(prerequisite_sha):
    plan = read(ROOT/'preregistration.json')
    need({k:v for k,v in plan.items() if k != 'registered_at'} == check(), 'Registered cold inputs or runtime changed.')
    with common.LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous = after_008(prerequisite_sha)
        need(not any((ROOT/name).exists() for name in ('run','launch.json','entry.log','wrapper-terminal.json')),
             'The cold output must be fresh; preserve previous attempts.')
        common.write(ROOT/'launch.json',dict(state='launching',registered_launch_at=common.now(),entry_command=command(),
            preregistration_sha256=sha(ROOT/'preregistration.json'),prerequisite_008=previous))
        args = ['/usr/bin/timeout','--kill-after=15s',f'{common.MAX_WALL_S-15}s',*command()]
        with (ROOT/'entry.log').open('xb') as log:
            spawned, start = datetime.now(timezone.utc), time.monotonic()
            process = subprocess.Popen(args,env=common.clean_env(),cwd=common.PROJECT,stdout=log,stderr=subprocess.STDOUT)
            rc = process.wait()
            elapsed = time.monotonic()-start
        status = read(RUN/'status.json') if (RUN/'status.json').is_file() else {}
        result = dict(state='failed' if rc else 'incomplete',subject='001',candidate_commit=COMMIT,returncode=rc,
            command_spawned_at=spawned.isoformat(),command_to_cli_end_seconds=elapsed,process_status=status,
            command=command(),prerequisite_008=previous,preregistration_sha256=sha(ROOT/'preregistration.json'),quality_accepted=False)
        if status.get('static_export_passed') is True and status.get('static_export_completed_at'):
            result['application_reported_static_export_seconds'] = (
                datetime.fromisoformat(status['static_export_completed_at'])-spawned).total_seconds()
        if not rc:
            try:
                result.update(validate(plan,status,spawned,elapsed),state='completed')
            except Exception as error:
                result['validation_error'] = str(error)
        receipt = ROOT/'wrapper-terminal.json'
        common.write(receipt,result)
        print(json.dumps(dict(state=result['state'],receipt=str(receipt),sha256=sha(receipt))),flush=True)
        return 0 if result['state'] == 'completed' else (rc or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','stage','launch'))
    parser.add_argument('--after-008-sha256')
    args = parser.parse_args()
    if args.action == 'launch':
        if not args.after_008_sha256:
            parser.error('--after-008-sha256 requires the verified delivery receipt hash')
        raise SystemExit(launch(args.after_008_sha256))
    plan = check()
    receipt = ROOT/'preregistration.json'
    if args.action == 'stage':
        ROOT.mkdir(exist_ok=False)
        common.write(receipt,dict(plan,registered_at=common.now()))
    print(json.dumps(dict(state='cpu_preflight_passed',staged=args.action == 'stage',
        preregistration=str(receipt),preregistration_sha256=sha(receipt) if receipt.is_file() else None,
        verified_local_assets=len(plan['official_and_asset_sha256']),entry_command=command(),gpu_calls=0),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

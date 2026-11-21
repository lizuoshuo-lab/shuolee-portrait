"""Train the public SH0 pipeline from its stock Gaussian initialization on frozen fits."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round4 as round4


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-main-20261111')
DELIVERY = round4.DELIVERY.parent.parent/'smpl-delivery-round4/receipt.json'
DELIVERY_SHA = '9526e9cf8cfe01a662c7e0b9dcd1243537a1114278adca298c154c71e1a4ba99'
DIRECTOR = round4.DIRECTOR.with_name('round4_director_review_20261111.json')
DIRECTOR_SHA = '44d4a214bd3e618488415c2c5843deb77e260b358fbc5993520d2ae6590cc86a'
BASE = 'ac2cfe68379de9f16a35784f2320fefcb88843e8'
BASE_SNAPSHOT = f'{round4.BASE_SNAPSHOT}-shape-{round4.STEPS:06d}'
RESULT_PINS = {
    '016': '4013a65ea0d37cbd13432e20ad538f0c8086c7084ddae6454941fd236d96c585',
    '006': '76a608680b5e6f9a47bdcc3d7102a1b695d66fe1ab2eb220dad5340b9f3bad76',
}
MODULE = 'src/shuolee_portrait/training.py'
ALLOWED = {MODULE, 'src/shuolee_portrait/cli.py', 'src/shuolee_portrait/sh0_main.py',
           'src/shuolee_portrait/coverage_refinement.py', 'src/shuolee_portrait/shape_refinement.py',
           'src/shuolee_portrait/sequence_pipeline.py', 'src/shuolee_portrait/export.py',
           'tests/test_sh0_main.py'}
FINAL = 'step-010000'
FROZEN = 'fit/frozen-surface.npz'
SCHEDULE = dict(warmup_steps=2400, coverage_steps=[4800, 8400], stock_densify_through=7800)
FIT_FILES = round4.round2.previous.base.COPY
CAMERA_FILES = round4.round2.CAMERA_FILES
OLD_TAILS = ('head_appearance', 'surface_refinement', 'coverage_refinement', 'shape_refinement')
common, rebound = round4.common, round4.rebound
need, read, sha, pin = round4.need, round4.read, round4.sha, round4.pin
check_source = rebound(round4.round2.check_source, PROJECT=PROJECT, BASE=BASE, SURFACE=MODULE, ALLOWED=ALLOWED)
imports = rebound(round4.round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round5-{subject}'


def baseline(subject):
    old_root = round4.root_for(subject)
    old = old_root/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-four input changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 4
         and set(delivery['subjects']) == set(RESULT_PINS), 'Unexpected round-four director review.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    need(archive['result'] == str(result_path) and archive['result_sha256'] == result_sha,
         'Round-four terminal differs from delivery.')
    record(old_root/'wrapper-terminal.json')
    wrapper, result = read(old_root/'wrapper-terminal.json'), read(result_path)
    record(old_root/'preregistration.json', wrapper['preregistration_sha256'])
    for name in (*FIT_FILES, 'run.json', 'config.json', 'gaussian-run/config.json'):
        record(old/name)
    fitted = read(old/'fit/official-fit-result.json')
    rows = read(old/'data/training-inputs.json')['records']
    selection = read(old/'fit/selection.json')
    snapshot = old/'gaussian-run'/BASE_SNAPSHOT
    replay_path = snapshot/'posed-replay/result.json'
    reference = selection['reference_frame']
    selected, held = selection['selected_frames'], selection['retained_holdouts']
    need(wrapper['state'] == result['state'] == 'completed' and wrapper['candidate_commit'] == archive['commit'] == BASE
         and wrapper['result_sha256'] == result_sha and result['format_passed'] is True
         and result['fit_reused'] is True and result['fit_calls'] == 0 and result['frozen_mesh_verified'] is True
         and result['final_snapshot'] == wrapper['final_snapshot'] == snapshot.name
         and result['points'] == archive['points'] and result['steps'] == 10000 and result['sh_degree'] == 0
         and selected == wrapper['selected_frames'] and len(selected) == len(rows) == result['frames'] == 24
         and not set(selected) & set(held) and [reference]+[f for f in selected if f != reference] == [r['frame'] for r in rows]
         and reference == result['reference_frame_id'] == wrapper['reference_frame']
         and fitted['selection'] == selection and fitted['fit_sha256'] == pins[str(old/'fit/fitted.npz')]
         and fitted['training_inputs_sha256'] == pins[str(old/'data/training-inputs.json')]
         and old/result['bound_replay'] == replay_path and str(replay_path) == archive['posed_replay'],
         'Round-four frozen fit, terminal, or selected roster differs.')
    record(replay_path, result['bound_replay_sha256'])
    replay = read(replay_path)
    bound_path = snapshot/'bound-replay.json'
    record(bound_path, replay['bound_replay_sha256'])
    bound = read(bound_path)
    frozen = snapshot/'reference-mesh.npz'
    frozen_sha = record(frozen, bound['inputs_sha256']['reference_mesh'])
    old_mesh = old/result['mesh']
    mesh_sha = record(old_mesh, read(old/'export/scene.json')['referenceMesh']['sha256'])
    posed_mesh = replay_path.parent/replay['mesh_file']
    record(posed_mesh, replay['mesh_sha256'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    record(asset, bound['asset_sha256'])
    camera_pins = {name: record(old/name) for name in CAMERA_FILES}
    need(camera_pins[CAMERA_FILES[0]] == replay['training_camera_sha256']
         and all(all(item['camera'][key] == row[key] for key in ('frame', 'K', 'w2c', 'wh'))
                 for item, row in zip(replay['records'], rows, strict=True)), 'Round-four camera or row identity differs.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS), 'Round-four asset root differs.')
    record(Path(meta['video']), meta['video_sha256'])
    return dict(subject=subject, baseline_commit=BASE, baseline_run=str(old), baseline_result_sha256=result_sha,
        director_review_sha256=DIRECTOR_SHA, baseline_posed_mesh=str(posed_mesh), baseline_points=result['points'],
        baseline_portable_mesh_sha256=mesh_sha, frozen_surface_source=str(frozen),
        frozen_surface=dict(file=FROZEN, sha256=frozen_sha), camera_pins=camera_pins,
        source_video_sha256=meta['video_sha256'], camera_config_sha256=meta['camera_config_sha256'],
        reference_frame=reference, selected_frames=selected, retained_holdouts=held,
        coverage=selection['coverage'], input_sha256=pins), meta


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    plan, meta = baseline(subject)
    old = Path(plan['baseline_run'])
    official = verify_official_fit(old)
    old_profile = read(old/'gaussian-run/config.json')
    need(official == old_profile['official_fit_provenance'], 'Official frozen fit does not match round four.')
    profile = smpl_sh0_effective_config()
    need(profile['gaussian_profile'] == 'smpl_sh0' and profile['iterations'] == 10000
         and profile['sh_degree'] == 0 and profile['initial_points_policy'] == 'one_per_triangle'
         and profile['head_appearance_steps'] == profile['head_appearance_weight'] == 0
         and profile['opacity_reset_steps'] == [1800, 3600, 5400, 7200]
         and all(profile['unified_main'][key] == value for key, value in SCHEDULE.items()),
         'Candidate profile does not describe fresh 10k SH0 main training.')
    previous_config = read(old/'config.json')
    config = {key: value for key, value in previous_config.items() if key not in old_profile}
    config.update(profile, frozen_surface=plan['frozen_surface'])
    code_files = set(common.SOURCE_FILES) | {name for name in changed if name.startswith('src/')}
    plan.update(schema='shuolee-frozen-fit-fresh-gaussian-v1', candidate_commit=commit, changed_paths=changed,
        fit_reused=True, fit_calls=0, gaussian_from_scratch=True, cold_video_clock=False,
        steps=10000, final_snapshot=FINAL, effective_profile=profile,
        code_sha256={name: sha(PROJECT/name) for name in sorted(code_files)},
        runtime=common.cpu_runtime(common.clean_env()), script_sha256=sha(Path(__file__).resolve()))
    return plan, meta, config


def stage(plan, meta, config):
    from shuolee_portrait.training import verify_official_fit

    root = root_for(plan['subject'])
    root.mkdir(exist_ok=False)
    run, old = root/'run', Path(plan['baseline_run'])
    copies = {name: old/name for name in FIT_FILES}
    copies[FROZEN] = Path(plan['frozen_surface_source'])
    staged = {}
    for name, source in copies.items():
        target = run/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        staged[name] = plan['input_sha256'][str(source)]
        pin(target, staged[name])
    common.write(run/'run.json', dict(meta, created=common.now(), copied_fitting_input_from=str(old),
        fit_reused=True, fit_calls=0, gaussian_from_scratch=True, cold_video_clock=False))
    common.write(run/'config.json', config)
    staged.update({name: sha(run/name) for name in ('run.json', 'config.json')})
    need(verify_official_fit(run) == verify_official_fit(old), 'Staging changed exact official fitting provenance.')
    receipt = root/'preregistration.json'
    common.write(receipt, dict(plan, registered_at=common.now(), staged_sha256=staged))
    print(json.dumps(dict(state='fresh_main_training_staged', subject=plan['subject'], run=str(run),
        preregistration_sha256=sha(receipt), frozen_fit_frames=len(plan['selected_frames']),
        staged_gaussian_files=0)), flush=True)


def verify_delivery(root, plan, elapsed):
    run = root/'run'
    result, status = read(run/'result.json'), read(run/'status.json')
    training = read(run/'gaussian-run/result.json')
    effective = read(run/'gaussian-run/config.json')
    preflight = read(run/'gaussian-run/preflight.json')
    integration_path = run/'gaussian-run/integration.json'
    integration = read(integration_path)
    need(result['state'] == status['state'] == training['state'] == 'completed' and result['format_passed'] is True
         and result['fit_reused'] is True and result['fit_calls'] == 0 and result['gaussian_from_scratch'] is True
         and result['unified_main_training'] == training['unified_main_training'] == integration
         and not any(key in training or key in result for key in OLD_TAILS)
         and result['steps'] == training['steps'] == plan['steps'] == 10000
         and result['final_snapshot'] == training['final_snapshot'] == plan['final_snapshot'] == FINAL
         and result['sh_degree'] == training['sh_degree'] == effective['sh_degree'] == 0
         and result['points'] == training['points'] and result['frames'] == len(plan['selected_frames']) == 24
         and result['reference_frame_id'] == plan['reference_frame']
         and effective['initial_points_policy'] == 'one_per_triangle' and effective['initial_points'] == 20908
         and effective['optimizer_initial_state_entries'] == 0 and preflight['finite_backward'] is True
         and preflight['optimizer_updates'] == 0 and preflight['static_offset_gradient_norm'] is None
         and sum(training['per_frame_updates']) == 10000
         and min(training['per_frame_updates']) > 0 and len(training['per_frame_updates']) == result['frames']
         and integration['state'] == 'completed' and integration['optimizer_updates'] == 10000
         and integration['initial_points'] == 20908 and integration['optimizer_initial_state_entries'] == 0
         and integration['no_prior_gaussian_loaded'] is True
         and integration['frozen_surface_sha256'] == plan['frozen_surface']['sha256']
         and integration['raw_native_rgb_target'] is True and integration['prior_render_used_as_target'] is False
         and integration['schedule'] == plan['effective_profile']['unified_main']
         and [event['step'] for event in integration['coverage_events']] == SCHEDULE['coverage_steps'],
         'Public CLI did not deliver the registered fresh Gaussian terminal.')
    for name in (*FIT_FILES, FROZEN):
        pin(run/name, plan['staged_sha256'][name])
    for name, digest in plan['camera_pins'].items():
        pin(run/name, digest)
    pin(run/result['mesh'], plan['baseline_portable_mesh_sha256'])
    pin(run/result['ply'], result['sha256'])
    pin(run/result['scene'], result['scene_sha256'])
    snapshot = run/'gaussian-run'/FINAL
    replay_path = snapshot/'posed-replay/result.json'
    panel_path = run/'audit/bound-panels/result.json'
    need(run/result['bound_replay'] == replay_path and run/result['bound_panels'] == panel_path,
         'Public CLI delivery replay paths differ.')
    pin(replay_path, result['bound_replay_sha256'])
    pin(panel_path, result['bound_panels_sha256'])
    replay, panels = read(replay_path), read(panel_path)
    posed = replay_path.parent/replay['mesh_file']
    pin(posed, replay['mesh_sha256'])
    round4.round2.verify_posed_mesh(Path(plan['baseline_posed_mesh']), posed)
    pin(snapshot/'reference-mesh.npz', plan['frozen_surface']['sha256'])
    rows = read(run/'data/training-inputs.json')['records']
    need([item['frame'] for item in replay['records']] == [item['frame'] for item in panels['panels']]
         == [row['frame'] for row in rows] and set(replay['retained_holdouts']) == set(plan['retained_holdouts']),
         'Public CLI changed the selected frames or retained holdouts.')
    common.write(root/'wrapper-terminal.json', dict(state='completed', subject=plan['subject'],
        candidate_commit=plan['candidate_commit'], baseline_result_sha256=plan['baseline_result_sha256'],
        result_sha256=sha(run/'result.json'), preregistration_sha256=sha(root/'preregistration.json'),
        integration_sha256=sha(integration_path), points=result['points'], final_snapshot=FINAL,
        selected_frames=plan['selected_frames'], coverage=plan['coverage'], reference_frame=plan['reference_frame'],
        frozen_mesh_verified=True, fit_reused=True, fit_calls=0, gaussian_from_scratch=True,
        cold_video_clock=False, quality_accepted=False, unified_main_training=result['unified_main_training'],
        public_cli_to_panels_seconds=elapsed))


def launch(subject, commit):
    root = root_for(subject)
    run = root/'run'
    receipt = root/'preregistration.json'
    plan = read(receipt)
    check_source(commit)
    env = common.clean_env()
    need(plan['subject'] == subject and plan['candidate_commit'] == commit
         and plan['script_sha256'] == sha(Path(__file__).resolve())
         and plan['runtime'] == common.cpu_runtime(env), 'Staged code or runtime changed.')
    for pins, prefix in ((plan['input_sha256'], Path('/')), (plan['staged_sha256'], run),
                         (plan['code_sha256'], PROJECT)):
        for name, digest in pins.items():
            pin(prefix/name, digest)
    with common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((run/name).exists() for name in ('gaussian-run', 'audit', 'export', 'result.json', 'status.json'))
             and not any((root/name).exists() for name in ('entry.log', 'wrapper-terminal.json', 'launch.json')),
             'Fresh training requires a previously unlaunched staged root.')
        cmd = [str(common.ENV/'bin/python'), '-B', '-u', '-m', 'shuolee_portrait.cli',
               'train-fitted', '--run', str(run)]
        common.write(root/'launch.json', dict(state='launching', subject=subject, candidate_commit=commit,
            preregistration_sha256=sha(receipt), started=common.now(), command=cmd, cwd=str(PROJECT/'src'),
            fit_calls=0, gaussian_from_scratch=True))
        started = time.monotonic()
        with (root/'entry.log').open('xb') as log:
            completed = subprocess.run(['/usr/bin/timeout', '--kill-after=15s', '3600s', *cmd],
                env=env, cwd=PROJECT/'src', stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            common.write(root/'wrapper-failure.json', dict(state='failed', return_code=completed.returncode,
                         stage='train-fitted', elapsed_seconds=time.monotonic()-started))
            return completed.returncode
        try:
            verify_delivery(root, plan, time.monotonic()-started)
        except Exception as error:
            common.write(root/'wrapper-failure.json', dict(state='failed', error=str(error),
                         stage='delivery_verify', elapsed_seconds=time.monotonic()-started))
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'stage', 'launch'))
    parser.add_argument('subject', choices=tuple(RESULT_PINS))
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.action == 'launch':
        raise SystemExit(launch(args.subject, args.commit))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan, meta, config = prepare(args.subject, args.commit)
    if args.action == 'stage':
        stage(plan, meta, config)
    else:
        print(json.dumps(dict(state='fresh_main_training_verified', subject=args.subject,
            frozen_fit_frames=len(plan['selected_frames']), fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

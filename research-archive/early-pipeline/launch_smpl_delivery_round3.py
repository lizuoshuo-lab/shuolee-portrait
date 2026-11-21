"""Refine coverage on the two exact round-two Gaussian deliveries."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round2 as round2


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-coverage-20261111')
DELIVERY = round2.DELIVERY.parent.parent/'smpl-delivery-round2/receipt.json'
DELIVERY_SHA = '9598cc638c8e035e37a0472992e08c58461f961d9fc23c6653fe3bccecb8ef56'
DIRECTOR = round2.DIRECTOR.with_name('round2_director_review_20261111.json')
DIRECTOR_SHA = '169395c89136c946eb22e1aebbab6ea213adc6e8b801be016e5ad62d4524dc5b'
BASE = '9a0ef65026e027f1ba2020fdf0a411b7091d80c6'
BASE_SNAPSHOT = 'step-010000-head-000600-surface-001800'
RESULT_PINS = {
    '016': '6e2ef8e25ce0efc105e467cbc1ca6ac33a920ab05ef11a0d4876f19154fbdb4f',
    '006': '9abd56ba6798f113faf87c9e0b309ad59a2b322a3eed5504bfbdc36215aed0a8',
}
COVERAGE = 'src/shuolee_portrait/coverage_refinement.py'
ALLOWED = {COVERAGE, 'src/shuolee_portrait/head_appearance.py', 'src/shuolee_portrait/export.py',
           'tests/test_coverage_refinement.py'}
STEPS = 1500
common = round2.common
need, read, sha, pin = round2.need, round2.read, round2.sha, round2.pin
rebound = round2.previous.rebound
check_source = rebound(round2.check_source, PROJECT=PROJECT, BASE=BASE, SURFACE=COVERAGE, ALLOWED=ALLOWED)
imports = rebound(round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round3-{subject}'


launch = rebound(round2.launch, root_for=root_for, check_source=check_source, PROJECT=PROJECT, __file__=__file__)


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.training import verify_official_fit

    old_root = round2.root_for(subject)
    old = old_root/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-two artifact changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    record(Path(round2.__file__).resolve())
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 2
         and set(delivery['subjects']) == set(RESULT_PINS), 'Unexpected round-two director review.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    need(archive['result'] == str(result_path) and archive['result_sha256'] == result_sha,
         'Round-two terminal differs from delivery.')
    record(old_root/'wrapper-terminal.json')
    wrapper, result = read(old_root/'wrapper-terminal.json'), read(result_path)
    record(old_root/'preregistration.json', wrapper['preregistration_sha256'])
    for name in round2.STAGED_FILES:
        record(old/name)
    training = read(old/'gaussian-run/result.json')
    config = read(old/'gaussian-run/config.json')
    rows = read(old/'data/training-inputs.json')['records']
    selection = read(old/'fit/selection.json')
    snapshot = old/'gaussian-run'/snapshot_name(training)
    surface = training['surface_refinement']
    replay_path = snapshot/'posed-replay/result.json'
    need(wrapper['state'] == result['state'] == training['state'] == 'completed'
         and wrapper['candidate_commit'] == archive['commit'] == BASE and wrapper['result_sha256'] == result_sha
         and result['format_passed'] is True and result['fit_reused'] is True and result['fit_calls'] == 0
         and result['frozen_mesh_verified'] is True and surface['frozen_mesh_verified'] is True
         and wrapper['surface_refinement'] == result['surface_refinement'] == surface
         and surface['steps'] == 1800 and surface['state'] == 'completed'
         and result['final_snapshot'] == wrapper['final_snapshot'] == snapshot.name == BASE_SNAPSHOT
         and training['head_appearance']['state'] == 'completed'
         and result['points'] == surface['points'] == archive['points']
         and training['steps'] == 10000 and training['sh_degree'] == 0
         and min(training['per_frame_updates']) > 0 and sum(training['per_frame_updates']) == training['steps']
         and len(training['per_frame_updates']) == len(rows) == result['frames'] == 24
         and config['official_fit_provenance'] == verify_official_fit(old)
         and selection['selected_frames'] == wrapper['selected_frames']
         and result['reference_frame_id'] == selection['reference_frame'] == wrapper['reference_frame']
         and set(selection['selected_frames']).isdisjoint(selection['retained_holdouts'])
         and old/result['bound_replay'] == replay_path and str(replay_path) == archive['posed_replay'],
         'Round-two pose, training or terminal identity differs.')
    record(replay_path, result['bound_replay_sha256'])
    record(old/result['bound_panels'], result['bound_panels_sha256'])
    record(old/result['scene'], result['scene_sha256'])
    record(old/result['ply'], result['sha256'])
    replay = read(replay_path)
    bound_path = snapshot/'bound-replay.json'
    need(str(bound_path) == archive['bound_terminal'], 'Round-two local Gaussian identity differs.')
    record(bound_path, replay['bound_replay_sha256'])
    bound = read(bound_path)
    record(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    for name, key in (('world.ply', 'world_ply'), ('reference-mesh.npz', 'reference_mesh'),
                      ('native-ply-parity.json', 'native_ply_parity')):
        record(snapshot/name, bound['inputs_sha256'][key])
    posed_mesh = replay_path.parent/replay['mesh_file']
    record(posed_mesh, replay['mesh_sha256'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit, _, facts = _inputs(old, snapshot, asset, bound)
    ids, _ = inspect_bound_ply(snapshot/'bound-local.ply', len(fit['faces']))
    need(len(ids) == bound['points'] == result['points']
         and [item['frame'] for item in facts['frames']] == [row['frame'] for row in rows]
         and all(all(item['camera'][key] == row[key] for key in ('frame', 'K', 'w2c', 'wh'))
                 for item, row in zip(replay['records'], rows, strict=True)), 'Round-two mesh or camera differs.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    camera_pins = {name: record(old/name) for name in round2.CAMERA_FILES}
    need(camera_pins[round2.CAMERA_FILES[0]] == replay['training_camera_sha256'], 'Round-two camera bytes changed.')
    old_mesh = old/result['mesh']
    mesh_sha = record(old_mesh, read(old/'export/scene.json')['referenceMesh']['sha256'])
    need(str(old_mesh) == archive['mesh'], 'Round-two packaged reference mesh differs.')
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS), 'Round-two asset root differs.')
    record(Path(meta['video']), meta['video_sha256'])
    code_files = set(common.SOURCE_FILES) | {COVERAGE, round2.SURFACE}
    return dict(schema='shuolee-fixed-mesh-coverage-refinement-v1', subject=subject,
        candidate_commit=commit, baseline_commit=BASE, baseline_run=str(old), baseline_snapshot=str(snapshot),
        baseline_result_sha256=result_sha, director_review_sha256=DIRECTOR_SHA, changed_paths=changed,
        baseline_posed_mesh=str(posed_mesh), baseline_reference_mesh_sha256=bound['inputs_sha256']['reference_mesh'],
        baseline_portable_mesh_sha256=mesh_sha, baseline_points=result['points'], camera_pins=camera_pins,
        source_video_sha256=meta['video_sha256'], camera_config_sha256=meta['camera_config_sha256'],
        reference_frame=selection['reference_frame'], selected_frames=selection['selected_frames'],
        retained_holdouts=selection['retained_holdouts'], coverage=selection['coverage'], fit_reused=True,
        fit_calls=0, gaussian_from_scratch=False, cold_video_clock=False, base_steps=10000,
        head_appearance_steps=600, surface_refinement_steps=1800, coverage_refinement_steps=STEPS,
        input_sha256=pins, runtime=common.cpu_runtime(common.clean_env()),
        code_sha256={name: sha(PROJECT/name) for name in sorted(code_files)},
        script_sha256=sha(Path(__file__).resolve()))


def stage(plan):
    from shuolee_portrait.training import verify_official_fit

    root = root_for(plan['subject'])
    root.mkdir(exist_ok=False)
    run, old = root/'run', Path(plan['baseline_run'])
    staged = {}
    for name in round2.STAGED_FILES:
        target = run/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old/name, target)
        staged[name] = plan['input_sha256'][str(old/name)]
        pin(target, staged[name])
    need(verify_official_fit(run) == read(run/'gaussian-run/config.json')['official_fit_provenance'],
         'Staging changed exact official fitting provenance.')
    receipt = root/'preregistration.json'
    common.write(receipt, dict(plan, registered_at=common.now(), staged_sha256=staged))
    print(json.dumps(dict(state='coverage_refinement_staged', subject=plan['subject'], run=str(run),
        preregistration_sha256=sha(receipt), baseline_points=plan['baseline_points'],
        selected_frames=plan['selected_frames'])), flush=True)


def worker(subject, commit):
    root = root_for(subject)
    run, plan_path = root/'run', root/'preregistration.json'
    plan = read(plan_path)
    need(dict(os.environ) == common.clean_env() and sys.flags.isolated == 1
         and plan['candidate_commit'] == commit and plan['subject'] == subject
         and read(root/'launch.json')['preregistration_sha256'] == sha(plan_path), 'Worker launch identity differs.')
    with common.LEASE.open('r+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Launch the worker under the registered GPU lease.')
    rt = imports()
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.bound_review_panels import write_panels
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.coverage_refinement import refine
    from shuolee_portrait.export import export
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.training import replay_official_bound, verify_official_fit

    start = time.monotonic()
    cfg = read(run/'config.json')
    training_path = run/'gaussian-run/result.json'
    training = read(training_path)
    stock = Path(plan['baseline_snapshot'])
    step = 'coverage_refinement'

    def status(state, **fields):
        rt.save(run/'status.json', dict(state=state, stage=step, updated=rt.now(), fitting_complete=True,
            fit_reused=True, fit_calls=0, quality_accepted=False, **fields))

    with gpu_lock():
        rt.configure(common.ASSETS, root=run)
        status('running', training_started=True)
        try:
            effective = read(run/'gaussian-run/config.json')
            need(verify_official_fit(run) == effective['official_fit_provenance'],
                 'Frozen fit changed before refinement.')
            model, _ = load_bound_snapshot(run, stock, asset_path=common.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            repaired, coverage = refine(run, stock, model, effective)
            del model
            snapshot = run/'gaussian-run'/coverage['snapshot']
            need(coverage['state'] == 'completed' and coverage['steps'] == STEPS
                 and coverage['snapshot'] == f'{stock.name}-coverage-{STEPS:06d}'
                 and coverage['baseline_snapshot'] == stock.name and coverage['frozen_mesh_verified'] is True
                 and coverage['points'] == len(repaired[0].binding) > 0, 'Coverage refinement did not complete.')
            pin(snapshot/'reference-mesh.npz', plan['baseline_reference_mesh_sha256'])
            training.update(coverage_refinement=coverage, final_snapshot=snapshot.name, points=coverage['points'],
                            warm_started_from_round2=True, coverage_baseline_snapshot=str(stock), from_scratch=False)
            need(snapshot_name(training) == snapshot.name, 'Terminal coverage snapshot selection differs.')
            rt.save(training_path, training)
            training = read(training_path)
            coverage = training['coverage_refinement']
            step = 'export'
            status('running', training_complete=True)
            export(run, cfg)
            export_seconds = time.monotonic()-start
            result = read(run/'result.json')
            need(result['state'] == 'completed' and result['format_passed'] is True
                 and result['steps'] == plan['base_steps'] and result['points'] == coverage['points']
                 and result['coverage_refinement'] == coverage and result['final_snapshot'] == snapshot.name
                 and result['surface_refinement'] == training['surface_refinement']
                 and result['reference_frame_id'] == plan['reference_frame'], 'Export differs from the coverage terminal.')
            for name, digest in plan['camera_pins'].items():
                pin(run/name, digest)
            pin(run/result['mesh'], plan['baseline_portable_mesh_sha256'])
            step = 'bound_review'
            status('running', static_export_passed=True)
            replay = replay_official_bound(run, replay=repaired)
            replay_path = snapshot/'posed-replay/result.json'
            round2.verify_posed_mesh(Path(plan['baseline_posed_mesh']), replay_path.parent/replay['mesh_file'])
            panels = write_panels(run, replay_path)
            panel_path = run/'audit/bound-panels/result.json'
            expected = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
            need([row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']] == expected,
                 'Refined posed review changed the source roster.')
            result.update(from_scratch=False, gaussian_from_scratch=False, warm_started_from_round2=True,
                fit_reused=True, fit_calls=0, cold_video_clock=False, frozen_mesh_verified=True,
                baseline_result_sha256=plan['baseline_result_sha256'], baseline_points=plan['baseline_points'],
                reused_fit_preregistration_sha256=sha(plan_path), bound_replay=replay_path.relative_to(run).as_posix(),
                bound_replay_sha256=sha(replay_path), bound_panels=panel_path.relative_to(run).as_posix(),
                bound_panels_sha256=sha(panel_path))
            rt.save(run/'result.json', result)
            step = 'complete'
            status('completed', static_export_passed=True, format_passed=True, frozen_mesh_verified=True,
                   reference_frame=result['reference_frame_id'], ply=str(run/result['ply']), sha256=result['sha256'])
            common.write(root/'wrapper-terminal.json', dict(state='completed', subject=subject, candidate_commit=commit,
                result_sha256=sha(run/'result.json'), preregistration_sha256=sha(plan_path),
                baseline_result_sha256=plan['baseline_result_sha256'], reference_frame=result['reference_frame_id'],
                points=result['points'], selected_frames=plan['selected_frames'], coverage=plan['coverage'],
                final_snapshot=snapshot.name, head_appearance=training['head_appearance'],
                surface_refinement=training['surface_refinement'], coverage_refinement=coverage,
                frozen_mesh_verified=True, quality_accepted=False, fit_reused=True, fit_calls=0,
                gaussian_from_scratch=False, cold_video_clock=False, warm_started_from_round2=True,
                reuse_to_static_export_seconds=export_seconds, reuse_to_panels_seconds=time.monotonic()-start))
        except Exception as error:
            status('failed', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'stage', 'launch', 'worker'))
    parser.add_argument('subject', choices=tuple(RESULT_PINS))
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.subject, args.commit)
    elif args.action == 'launch':
        raise SystemExit(launch(args.subject, args.commit, resume=False))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        plan = prepare(args.subject, args.commit)
        if args.action == 'stage':
            stage(plan)
        else:
            print(json.dumps(dict(state='coverage_refinement_verified', subject=args.subject,
                baseline_points=plan['baseline_points'], selected_frames=plan['selected_frames'],
                fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

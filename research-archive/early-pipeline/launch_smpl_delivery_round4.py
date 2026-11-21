"""Run fixed-mesh shape refinement on the two delivered round-three models."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round3 as round3


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-shape-20261111')
DELIVERY = round3.DELIVERY.parent.parent/'smpl-delivery-round3/receipt.json'
DELIVERY_SHA = 'ed54c1fa0e80d32b1126728f9b6b77b086a2887706198b99d27f1475e6ce3ef9'
DIRECTOR = round3.DIRECTOR.with_name('round3_director_review_20261111.json')
DIRECTOR_SHA = 'c649493544fa27cba124890ad7e03b1218a27939930d3bd90aae10b7db14a954'
BASE = '967c286d023c47ae752991f5edfe4c09afc6c6f9'
BASE_SNAPSHOT = f'{round3.BASE_SNAPSHOT}-coverage-{round3.STEPS:06d}'
RESULT_PINS = {
    '016': '79cb21d3bae9cf18db8bc8c2f309e161e9b30e75d1dfd07727cf52d7311237c7',
    '006': '4f49c44e1892179363ebd8174b6ba64a1104eb350d0d450b2dd787ea0a986a1b',
}
MODULE = 'src/shuolee_portrait/shape_refinement.py'
ALLOWED = {MODULE, 'src/shuolee_portrait/head_appearance.py', 'src/shuolee_portrait/export.py',
           'tests/test_shape_refinement.py'}
STEPS = 1800
round2, common, rebound = round3.round2, round3.common, round3.rebound
need, read, sha, pin = round3.need, round3.read, round3.sha, round3.pin
check_source = rebound(round2.check_source, PROJECT=PROJECT, BASE=BASE, SURFACE=MODULE, ALLOWED=ALLOWED)
imports = rebound(round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round4-{subject}'


launch = rebound(round2.launch, root_for=root_for, check_source=check_source, PROJECT=PROJECT, __file__=__file__)


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.training import verify_official_fit

    old_root = round3.root_for(subject)
    old = old_root/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-three input changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    record(Path(round2.__file__).resolve())
    record(Path(round3.__file__).resolve())
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 3
         and set(delivery['subjects']) == set(RESULT_PINS), 'Unexpected round-three director review.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    need(archive['result'] == str(result_path) and archive['result_sha256'] == result_sha,
         'Round-three terminal differs from delivery.')
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
    coverage = training['coverage_refinement']
    replay_path = snapshot/'posed-replay/result.json'
    need(wrapper['state'] == result['state'] == training['state'] == 'completed'
         and wrapper['candidate_commit'] == archive['commit'] == BASE and wrapper['result_sha256'] == result_sha
         and result['format_passed'] is True and result['fit_reused'] is True and result['fit_calls'] == 0
         and result['frozen_mesh_verified'] is True and coverage['frozen_mesh_verified'] is True
         and all(wrapper[key] == result[key] == training[key]
                 for key in ('head_appearance', 'surface_refinement', 'coverage_refinement'))
         and coverage['steps'] == round3.STEPS and coverage['state'] == 'completed'
         and result['final_snapshot'] == wrapper['final_snapshot'] == snapshot.name == BASE_SNAPSHOT
         and result['points'] == coverage['points'] == archive['points']
         and training['steps'] == 10000 and training['sh_degree'] == 0
         and min(training['per_frame_updates']) > 0 and sum(training['per_frame_updates']) == training['steps']
         and len(training['per_frame_updates']) == len(rows) == result['frames'] == 24
         and config['official_fit_provenance'] == verify_official_fit(old)
         and selection['selected_frames'] == wrapper['selected_frames']
         and result['reference_frame_id'] == selection['reference_frame'] == wrapper['reference_frame']
         and set(selection['selected_frames']).isdisjoint(selection['retained_holdouts'])
         and old/result['bound_replay'] == replay_path and str(replay_path) == archive['posed_replay'],
         'Round-three frozen pose, training or terminal identity differs.')
    record(replay_path, result['bound_replay_sha256'])
    record(old/result['bound_panels'], result['bound_panels_sha256'])
    record(old/result['scene'], result['scene_sha256'])
    record(old/result['ply'], result['sha256'])
    replay = read(replay_path)
    bound_path = snapshot/'bound-replay.json'
    need(str(bound_path) == archive['bound_terminal'], 'Round-three local Gaussian identity differs.')
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
                 for item, row in zip(replay['records'], rows, strict=True)), 'Round-three mesh or camera differs.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    camera_pins = {name: record(old/name) for name in round2.CAMERA_FILES}
    need(camera_pins[round2.CAMERA_FILES[0]] == replay['training_camera_sha256'], 'Round-three camera bytes changed.')
    old_mesh = old/result['mesh']
    mesh_sha = record(old_mesh, read(old/'export/scene.json')['referenceMesh']['sha256'])
    need(str(old_mesh) == archive['mesh'], 'Round-three packaged reference mesh differs.')
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS), 'Round-three asset root differs.')
    record(Path(meta['video']), meta['video_sha256'])
    code_files = set(common.SOURCE_FILES) | {MODULE, round3.COVERAGE, round2.SURFACE}
    return dict(schema='shuolee-fixed-mesh-shape-refinement-v1', subject=subject, candidate_commit=commit,
        baseline_commit=BASE, baseline_run=str(old), baseline_snapshot=str(snapshot), baseline_result_sha256=result_sha,
        director_review_sha256=DIRECTOR_SHA, changed_paths=changed, baseline_posed_mesh=str(posed_mesh),
        baseline_reference_mesh_sha256=bound['inputs_sha256']['reference_mesh'], baseline_portable_mesh_sha256=mesh_sha,
        baseline_points=result['points'], camera_pins=camera_pins, source_video_sha256=meta['video_sha256'],
        camera_config_sha256=meta['camera_config_sha256'], reference_frame=selection['reference_frame'],
        selected_frames=selection['selected_frames'], retained_holdouts=selection['retained_holdouts'],
        coverage=selection['coverage'], fit_reused=True, fit_calls=0, gaussian_from_scratch=False, cold_video_clock=False,
        base_steps=10000, head_appearance_steps=600, surface_refinement_steps=1800, coverage_refinement_steps=round3.STEPS,
        shape_refinement_steps=STEPS, input_sha256=pins, runtime=common.cpu_runtime(common.clean_env()),
        code_sha256={name: sha(PROJECT/name) for name in sorted(code_files)}, script_sha256=sha(Path(__file__).resolve()))


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
    print(json.dumps(dict(state='shape_refinement_staged', subject=plan['subject'], run=str(run),
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
    from shuolee_portrait.export import export
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.shape_refinement import refine
    from shuolee_portrait.training import replay_official_bound, verify_official_fit

    start = time.monotonic()
    cfg = read(run/'config.json')
    training_path = run/'gaussian-run/result.json'
    training = read(training_path)
    stock = Path(plan['baseline_snapshot'])
    step = 'shape_refinement'

    def status(state, **fields):
        rt.save(run/'status.json', dict(state=state, stage=step, updated=rt.now(), fitting_complete=True,
            fit_reused=True, fit_calls=0, quality_accepted=False, **fields))

    with gpu_lock():
        rt.configure(common.ASSETS, root=run)
        status('running', training_started=True)
        try:
            effective = read(run/'gaussian-run/config.json')
            need(verify_official_fit(run) == effective['official_fit_provenance'], 'Frozen fit changed before refinement.')
            model, _ = load_bound_snapshot(run, stock, asset_path=common.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            repaired, shape = refine(run, stock, model, effective)
            del model
            snapshot = run/'gaussian-run'/shape['snapshot']
            need(shape['state'] == 'completed' and shape['steps'] == STEPS
                 and shape['snapshot'] == f'{stock.name}-shape-{STEPS:06d}'
                 and shape['baseline_snapshot'] == stock.name and shape['frozen_mesh_verified'] is True
                 and shape['initial_points'] == plan['baseline_points']
                 and shape['points'] == len(repaired[0].binding) > 0, 'Shape refinement did not complete.')
            pin(snapshot/'reference-mesh.npz', plan['baseline_reference_mesh_sha256'])
            training.update(shape_refinement=shape, final_snapshot=snapshot.name, points=shape['points'],
                            warm_started_from_round3=True, shape_baseline_snapshot=str(stock), from_scratch=False)
            need(snapshot_name(training) == snapshot.name, 'Terminal shape snapshot selection differs.')
            rt.save(training_path, training)
            training = read(training_path)
            shape = training['shape_refinement']
            step = 'export'
            status('running', training_complete=True)
            export(run, cfg)
            export_seconds = time.monotonic()-start
            result = read(run/'result.json')
            prior = ('head_appearance', 'surface_refinement', 'coverage_refinement')
            need(result['state'] == 'completed' and result['format_passed'] is True
                 and result['steps'] == plan['base_steps'] and result['points'] == shape['points']
                 and result['shape_refinement'] == shape and result['final_snapshot'] == snapshot.name
                 and all(result[key] == training[key] for key in prior)
                 and result['reference_frame_id'] == plan['reference_frame'], 'Export differs from the shape terminal.')
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
            result.update(from_scratch=False, gaussian_from_scratch=False, warm_started_from_round3=True,
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
                final_snapshot=snapshot.name, shape_refinement=shape, **{key:training[key] for key in prior},
                frozen_mesh_verified=True, quality_accepted=False, fit_reused=True, fit_calls=0,
                gaussian_from_scratch=False, cold_video_clock=False, warm_started_from_round3=True,
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
            print(json.dumps(dict(state='shape_refinement_verified', subject=args.subject,
                baseline_points=plan['baseline_points'], selected_frames=plan['selected_frames'],
                fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

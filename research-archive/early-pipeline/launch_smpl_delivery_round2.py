"""Deliver fixed-mesh Gaussian refinement for the two verified round-one models."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_frozen_fit as previous
import launch_smpl_delivery_round1 as round1


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-surface-20261111')
DELIVERY = Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round1/receipt.json')
DELIVERY_SHA = '0cd5b6afaaa6bc273646ce264fdb5d39c8dfcac834940f30f77492f367bc3732'
DIRECTOR = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs/round1_director_review_20261111.json')
DIRECTOR_SHA = 'e6e7b0c7110f6d77d9ed536f6703d9066fdbe5bf397815091d7cf933b950276f'
BASE = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
RESULT_PINS = {
    '016': '0f8ef05bd574afb8c360ac5cbbade89ed51cd8ae7b2feeb88337a95840069e27',
    '006': 'e5990409d18546a35044352662173f7c0bb3cc0d8a9af7e3e40a09887a4b8dea',
}
SURFACE = 'src/shuolee_portrait/surface_refinement.py'
ALLOWED = {SURFACE, 'src/shuolee_portrait/head_appearance.py', 'src/shuolee_portrait/export.py',
           'tests/test_surface_refinement.py', 'tests/test_head_appearance.py'}
STAGED_FILES = (*previous.base.COPY, 'run.json', 'config.json', 'gaussian-run/result.json',
                'gaussian-run/config.json', 'gaussian-run/training-cameras.json')
CAMERA_FILES = previous.CAMERA_FILES
STEPS = 1800
common = previous.common
need, read, sha, pin = previous.need, previous.read, previous.sha, previous.pin


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round2-{subject}'


def check_source(commit):
    need(re.fullmatch(r'[0-9a-f]{40}', commit) is not None and commit != BASE,
         'Pass the exact committed round-two revision.')
    previous.rebound(common.git_state, PROJECT=PROJECT)(commit)
    pointer = (PROJECT/'.git').read_text(encoding='utf-8').strip().removeprefix('gitdir: ')
    gitdir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer], text=True).strip()
    git = ['/usr/bin/git', '--git-dir', gitdir, '--work-tree', str(PROJECT)]
    subprocess.run([*git, 'merge-base', '--is-ancestor', BASE, commit], check=True)
    changed = subprocess.check_output([*git, 'diff', '--name-only', BASE, commit], text=True).splitlines()
    need(SURFACE in changed and set(changed) <= ALLOWED, f'Unexpected round-two production changes: {changed}')
    return changed


def imports():
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    need(rt.PROJECT == PROJECT, 'Wrong project for fixed-mesh replay.')
    return rt


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.head_appearance import snapshot_name
    from shuolee_portrait.training import verify_official_fit

    old_root = round1.root_for(subject)
    old = old_root/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-one artifact changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 1
         and tuple(director['subjects']) == tuple(RESULT_PINS) and set(delivery['subjects']) == set(RESULT_PINS),
         'Unexpected director review batch.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    need(archive['result'] == str(result_path) and archive['result_sha256'] == result_sha,
         'Baseline terminal does not match the delivered receipt.')
    record(old_root/'wrapper-terminal.json')
    wrapper, result = read(old_root/'wrapper-terminal.json'), read(result_path)
    for name in STAGED_FILES:
        record(old/name)
    official = verify_official_fit(old)
    training = read(old/'gaussian-run/result.json')
    config = read(old/'gaussian-run/config.json')
    rows = read(old/'data/training-inputs.json')['records']
    selection = read(old/'fit/selection.json')
    snapshot = old/'gaussian-run'/snapshot_name(training)
    replay_path = snapshot/'posed-replay/result.json'
    need(wrapper['state'] == result['state'] == training['state'] == 'completed'
         and wrapper['candidate_commit'] == BASE and wrapper['result_sha256'] == result_sha
         and result['format_passed'] is True and result['fit_reused'] is True and result['fit_calls'] == 0
         and result['final_snapshot'] == wrapper['final_snapshot'] == snapshot.name == 'step-010000-head-000600'
         and training['head_appearance']['state'] == 'completed' and result['points'] == training['points']
         and training['steps'] == 10000 and training['sh_degree'] == 0
         and min(training['per_frame_updates']) > 0 and sum(training['per_frame_updates']) == 10000
         and len(training['per_frame_updates']) == len(rows) == result['frames'] == 24
         and config['official_fit_provenance'] == official and selection['selected_frames'] == wrapper['selected_frames']
         and result['reference_frame_id'] == selection['reference_frame'] == wrapper['reference_frame']
         and set(selection['selected_frames']).isdisjoint(selection['retained_holdouts'])
         and (old/result['bound_replay']) == replay_path, 'Round-one fit, training or terminal identity differs.')
    record(replay_path, result['bound_replay_sha256'])
    record(old/result['bound_panels'], result['bound_panels_sha256'])
    record(old/result['scene'], result['scene_sha256'])
    record(old/result['ply'], result['sha256'])
    replay = read(replay_path)
    record(snapshot/'bound-replay.json', replay['bound_replay_sha256'])
    bound = read(snapshot/'bound-replay.json')
    record(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    for name, key in (('world.ply', 'world_ply'), ('reference-mesh.npz', 'reference_mesh'),
                      ('native-ply-parity.json', 'native_ply_parity')):
        record(snapshot/name, bound['inputs_sha256'][key])
    record(snapshot/'reference-render.png')
    posed_mesh = replay_path.parent/replay['mesh_file']
    record(posed_mesh, replay['mesh_sha256'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    fit, _, facts = _inputs(old, snapshot, asset, bound)
    ids, _ = inspect_bound_ply(snapshot/'bound-local.ply', len(fit['faces']))
    need(len(ids) == bound['points'] == result['points'] and len(facts['frames']) == len(rows)
         and [item['frame'] for item in facts['frames']] == [row['frame'] for row in rows]
         and all(all(record['camera'][key] == row[key] for key in ('frame', 'K', 'w2c', 'wh'))
                 for record, row in zip(replay['records'], rows, strict=True)), 'Round-one bound mesh or camera mismatch.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    camera_pins = {name: record(old/name) for name in CAMERA_FILES}
    need(camera_pins[CAMERA_FILES[0]] == replay['training_camera_sha256'], 'Round-one camera bytes changed.')
    old_mesh = old/result['mesh']
    mesh_sha = record(old_mesh)
    mesh_delivery = read(old/'export/scene.json')['referenceMesh']
    need(mesh_delivery['sha256'] == mesh_sha, 'Round-one packaged reference mesh differs.')
    baseline_meta = read(old/'run.json')
    need(baseline_meta['assets'] == str(common.ASSETS), 'Round-one asset root differs.')
    record(Path(baseline_meta['video']), baseline_meta['video_sha256'])
    registered_codes = set(common.SOURCE_FILES) | {SURFACE}
    return dict(schema='shuolee-fixed-mesh-surface-refinement-v1', subject=subject,
        candidate_commit=commit, baseline_commit=BASE, baseline_run=str(old), baseline_snapshot=str(snapshot),
        baseline_result_sha256=result_sha, director_review_sha256=DIRECTOR_SHA, changed_paths=changed,
        baseline_posed_mesh=str(posed_mesh), baseline_reference_mesh_sha256=bound['inputs_sha256']['reference_mesh'],
        baseline_portable_mesh_sha256=mesh_sha, baseline_points=result['points'], camera_pins=camera_pins,
        source_video_sha256=baseline_meta['video_sha256'], camera_config_sha256=baseline_meta['camera_config_sha256'],
        reference_frame=selection['reference_frame'], selected_frames=selection['selected_frames'],
        retained_holdouts=selection['retained_holdouts'], coverage=selection['coverage'],
        fit_reused=True, fit_calls=0, gaussian_from_scratch=False, cold_video_clock=False,
        base_steps=10000, head_appearance_steps=600, surface_refinement_steps=STEPS,
        input_sha256=pins, runtime=common.cpu_runtime(common.clean_env()),
        code_sha256={name: sha(PROJECT/name) for name in sorted(registered_codes)},
        script_sha256=sha(Path(__file__).resolve()))


def stage(plan):
    root = root_for(plan['subject'])
    root.mkdir(exist_ok=False)
    run = root/'run'
    old = Path(plan['baseline_run'])
    staged = {}
    for name in STAGED_FILES:
        target = run/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old/name, target)
        staged[name] = plan['input_sha256'][str(old/name)]
        pin(target, staged[name])
    from shuolee_portrait.training import verify_official_fit

    need(verify_official_fit(run) == read(run/'gaussian-run/config.json')['official_fit_provenance'],
         'New root does not preserve exact official fitting provenance.')
    receipt = root/'preregistration.json'
    common.write(receipt, dict(plan, registered_at=common.now(), staged_sha256=staged))
    print(json.dumps(dict(state='surface_refinement_staged', subject=plan['subject'], run=str(run),
                          preregistration_sha256=sha(receipt), baseline_points=plan['baseline_points'],
                          selected_frames=plan['selected_frames'])), flush=True)


def launch(subject, commit, *, resume=False):
    root = root_for(subject)
    run = root/'run'
    receipt = root/'preregistration.json'
    plan = read(receipt)
    check_source(commit)
    launch_script = root/'runner-before-json-fix.py' if resume else Path(__file__).resolve()
    need(plan['subject'] == subject and plan['candidate_commit'] == commit
         and plan['script_sha256'] == sha(launch_script)
         and plan['runtime'] == common.cpu_runtime(common.clean_env()), 'Staged code or runtime changed.')
    for pins, prefix in ((plan['input_sha256'], Path('/')), (plan['staged_sha256'], run),
                         (plan['code_sha256'], PROJECT)):
        for name, digest in pins.items():
            if resume and prefix == run and name == 'gaussian-run/result.json':
                continue
            pin(prefix/name, digest)
    with common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            previous_status = read(run/'status.json')
            need(previous_status.get('state') == 'failed' and previous_status.get('stage') == 'export'
                 and previous_status.get('error') == 'Export differs from the refined terminal.'
                 and not (root/'resume-entry.log').exists() and not (run/'audit').exists(),
                 'Only the verified post-export receipt failure is recoverable here.')
            shutil.copyfile(run/'status.json', root/'failed-export-status.json')
        else:
            need(not any((run/name).exists() for name in ('audit', 'export', 'result.json', 'status.json'))
                 and not any((root/name).exists() for name in ('entry.log', 'wrapper-terminal.json', 'launch.json')),
                 'Preserve the existing repair attempt.')
        launch_name = 'resume-launch.json' if resume else 'launch.json'
        log_name = 'resume-entry.log' if resume else 'entry.log'
        common.write(root/launch_name, dict(state='launching', subject=subject, candidate_commit=commit,
                     preregistration_sha256=sha(receipt), started=common.now(), fit_calls=0))
        cmd = [str(common.ENV/'bin/python'), '-I', '-B', '-u', str(Path(__file__).resolve()),
               'resume-worker' if resume else 'worker', subject, '--commit', commit]
        with (root/log_name).open('xb') as log:
            return subprocess.run(['/usr/bin/timeout', '--kill-after=15s', '1800s', *cmd],
                                  env=common.clean_env(), cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT).returncode


def verify_posed_mesh(before_path, after_path):
    import numpy as np

    with np.load(before_path, allow_pickle=False) as before, np.load(after_path, allow_pickle=False) as after:
        need(before.files == after.files, 'Posed mesh archive fields changed.')
        for key in before.files:
            a, b = before[key], after[key]
            need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                 f'Frozen posed mesh bytes changed: {key}')


def worker(subject, commit, *, resume=False):
    root = root_for(subject)
    run = root/'run'
    plan_path = root/'preregistration.json'
    plan = read(plan_path)
    need(dict(os.environ) == common.clean_env() and sys.flags.isolated == 1
         and plan['candidate_commit'] == commit and plan['subject'] == subject
         and read(root/('resume-launch.json' if resume else 'launch.json'))['preregistration_sha256'] == sha(plan_path),
         'Worker launch identity differs.')
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
    from shuolee_portrait.surface_refinement import refine
    from shuolee_portrait.training import replay_official_bound, verify_official_fit

    start = time.monotonic()
    cfg = read(run/'config.json')
    training_path = run/'gaussian-run/result.json'
    training = read(training_path)
    stock = Path(plan['baseline_snapshot'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    step = 'export' if resume else 'surface_refinement'

    def status(state, **fields):
        rt.save(run/'status.json', dict(state=state, stage=step, updated=rt.now(), fitting_complete=True,
                                      fit_reused=True, fit_calls=0, quality_accepted=False, **fields))

    with gpu_lock():
        rt.configure(common.ASSETS, root=run)
        status('running', training_started=True)
        try:
            effective = read(run/'gaussian-run/config.json')
            need(verify_official_fit(run) == effective['official_fit_provenance'],
                 'Frozen fit changed before repair.')
            if resume:
                refinement = training['surface_refinement']
                repaired = load_bound_snapshot(run, run/'gaussian-run'/snapshot_name(training), asset_path=asset)
            else:
                model, _ = load_bound_snapshot(run, stock, asset_path=asset)
                repaired, refinement = refine(run, stock, model, effective)
                del model
            snapshot = run/'gaussian-run'/refinement['snapshot']
            need(refinement['state'] == 'completed' and refinement['steps'] == STEPS
                 and refinement['snapshot'] == f'{stock.name}-surface-{STEPS:06d}'
                 and refinement['points'] == len(repaired[0].binding) > 0
                 and refinement['frozen_mesh_verified'] is True, 'Fixed-mesh surface refinement did not complete.')
            pin(snapshot/'reference-mesh.npz', plan['baseline_reference_mesh_sha256'])
            training.update(surface_refinement=refinement, final_snapshot=snapshot.name, points=refinement['points'],
                            warm_started_from_round1=True, baseline_snapshot=str(stock), from_scratch=False)
            need(snapshot_name(training) == snapshot.name, 'Terminal surface snapshot selection differs.')
            if not resume:
                rt.save(training_path, training)
                # JSON stores dictionary keys as strings. Compare the same
                # persisted representation the exporter reads and writes.
                training = read(training_path)
                refinement = training['surface_refinement']
                step = 'export'
                status('running', training_complete=True)
                export(run, cfg)
            export_seconds = None if resume else time.monotonic()-start
            result = read(run/'result.json')
            need(result['state'] == 'completed' and result['format_passed'] is True
                 and result['steps'] == plan['base_steps'] and result['points'] == refinement['points']
                 and result['surface_refinement'] == refinement and result['final_snapshot'] == snapshot.name
                 and result['reference_frame_id'] == plan['reference_frame'], 'Export differs from the refined terminal.')
            for name, digest in plan['camera_pins'].items():
                pin(run/name, digest)
            pin(run/result['mesh'], plan['baseline_portable_mesh_sha256'])
            step = 'bound_review'
            status('running', static_export_passed=True)
            replay = replay_official_bound(run, replay=repaired)
            replay_path = snapshot/'posed-replay/result.json'
            verify_posed_mesh(Path(plan['baseline_posed_mesh']), replay_path.parent/replay['mesh_file'])
            panels = write_panels(run, replay_path)
            panel_path = run/'audit/bound-panels/result.json'
            expected = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
            need([row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']] == expected,
                 'Refined posed review changed the source roster.')
            result.update(from_scratch=False, gaussian_from_scratch=False, warm_started_from_round1=True,
                fit_reused=True, fit_calls=0, cold_video_clock=False, frozen_mesh_verified=True,
                baseline_result_sha256=plan['baseline_result_sha256'], baseline_points=plan['baseline_points'],
                reused_fit_preregistration_sha256=sha(plan_path),
                bound_replay=replay_path.relative_to(run).as_posix(), bound_replay_sha256=sha(replay_path),
                bound_panels=panel_path.relative_to(run).as_posix(), bound_panels_sha256=sha(panel_path))
            rt.save(run/'result.json', result)
            step = 'complete'
            status('completed', static_export_passed=True, format_passed=True, frozen_mesh_verified=True,
                   reference_frame=result['reference_frame_id'], ply=str(run/result['ply']), sha256=result['sha256'])
            common.write(root/'wrapper-terminal.json', dict(state='completed', subject=subject, candidate_commit=commit,
                result_sha256=sha(run/'result.json'), preregistration_sha256=sha(plan_path),
                baseline_result_sha256=plan['baseline_result_sha256'], reference_frame=result['reference_frame_id'],
                points=result['points'], selected_frames=plan['selected_frames'], coverage=plan['coverage'],
                final_snapshot=snapshot.name, head_appearance=training['head_appearance'],
                surface_refinement=refinement, frozen_mesh_verified=True, quality_accepted=False,
                fit_reused=True, fit_calls=0, gaussian_from_scratch=False, cold_video_clock=False,
                warm_started_from_round1=True, reuse_to_static_export_seconds=export_seconds,
                reuse_to_panels_seconds=None if resume else time.monotonic()-start,
                export_receipt_recovered=resume, recovery_to_panels_seconds=time.monotonic()-start if resume else None))
        except Exception as error:
            status('failed', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'stage', 'launch', 'worker', 'resume', 'resume-worker'))
    parser.add_argument('subject', choices=tuple(RESULT_PINS))
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.action in ('worker', 'resume-worker'):
        worker(args.subject, args.commit, resume=args.action == 'resume-worker')
    elif args.action in ('launch', 'resume'):
        raise SystemExit(launch(args.subject, args.commit, resume=args.action == 'resume'))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        plan = prepare(args.subject, args.commit)
        if args.action == 'stage':
            stage(plan)
        else:
            print(json.dumps(dict(state='surface_refinement_verified', subject=args.subject,
                baseline_points=plan['baseline_points'], selected_frames=plan['selected_frames'],
                fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

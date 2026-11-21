"""Run fresh world-scale-controlled SH0 training on the exact round-five inputs."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round5 as round5


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-scale-20261111')
DELIVERY = round5.DELIVERY.parent.parent/'smpl-delivery-round5/receipt.json'
DELIVERY_SHA = '6db51a1c18534fa9b7965383d6162107c69ccf3e0335fa32cb05ebafca31cf5d'
DIRECTOR = round5.DIRECTOR.with_name('round5_director_review_20261111.json')
DIRECTOR_SHA = 'f50e7766b668ca343db640771fc727c8410f64261af9ffac07fac58ed8eb9d66'
BASE = '9013c26bc1b0758a5798ae2c3b0cf28e4d1a72c0'
RESULT_PINS = {
    '016': '39ce6bb640c66f7cac3226e1edf9e95aa48405453428ebded43d38e0d2edcd31',
    '006': '2ac4df42a040fd0485a8a93c24a8d061c253bc2385e9a918b43fd2c17baba410',
}
MODULE = 'src/shuolee_portrait/world_scale.py'
ALLOWED = {round5.MODULE, MODULE, 'src/shuolee_portrait/sh0_main.py', 'tests/test_world_scale.py'}
WORLD_SCALE = 'gaussian-run/world-scale.json'
INIT = 'gaussian-run/initialization.json'
common, rebound = round5.common, round5.rebound
need, read, sha, pin = round5.need, round5.read, round5.sha, round5.pin
check_source = rebound(round5.round4.round2.check_source, PROJECT=PROJECT, BASE=BASE,
                       SURFACE=MODULE, ALLOWED=ALLOWED)
imports = rebound(round5.round4.round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round6-{subject}'


def baseline(subject):
    from archive_smpl_delivery_round2 import fresh_proof

    old_root = round5.root_for(subject)
    old = old_root/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-five input changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    record(Path(round5.__file__).resolve())
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 5
         and director['receipt_sha256_verified'] == DELIVERY_SHA and set(delivery['subjects']) == set(RESULT_PINS),
         'Unexpected round-five director review.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    need(archive['result'] == str(result_path) and archive['result_sha256'] == result_sha,
         'Round-five terminal differs from published delivery.')
    record(old_root/'wrapper-terminal.json')
    wrapper, result = read(old_root/'wrapper-terminal.json'), read(result_path)
    plan_path = old_root/'preregistration.json'
    record(plan_path, wrapper['preregistration_sha256'])
    old_plan = read(plan_path)
    snapshot = old/'gaussian-run'/round5.FINAL
    need(wrapper['state'] == result['state'] == 'completed'
         and wrapper['candidate_commit'] == archive['commit'] == BASE and wrapper['result_sha256'] == result_sha
         and wrapper['frozen_mesh_verified'] is result['format_passed'] is True
         and result['gaussian_from_scratch'] is result['fit_reused'] is True and result['fit_calls'] == 0
         and result['final_snapshot'] == snapshot.name and result['points'] == archive['points'],
         'Round-five fresh delivery identity differs.')
    _, _, proofs = fresh_proof(old_root, old, snapshot, result, wrapper, old_plan)
    for name, (path, digest) in proofs.items():
        record(path, digest)
        published = archive['training_proof'][name]
        need(published['sha256'] == digest, f'Published round-five {name} differs.')
        record(Path(published['path']), digest)
    for name in (*round5.FIT_FILES, round5.FROZEN):
        record(old/name, old_plan['staged_sha256'][name])
    for name in ('run.json', 'config.json', 'gaussian-run/config.json'):
        record(old/name)
    rows = read(old/'data/training-inputs.json')['records']
    selection = read(old/'fit/selection.json')
    reference = selection['reference_frame']
    selected, held = selection['selected_frames'], selection['retained_holdouts']
    replay_path = old/result['bound_replay']
    record(replay_path, result['bound_replay_sha256'])
    replay = read(replay_path)
    posed_mesh = replay_path.parent/replay['mesh_file']
    record(posed_mesh, replay['mesh_sha256'])
    camera_pins = {name: record(old/name, digest) for name, digest in old_plan['camera_pins'].items()}
    need(camera_pins[round5.CAMERA_FILES[0]] == replay['training_camera_sha256']
         and reference == old_plan['reference_frame'] == result['reference_frame_id']
         and selected == old_plan['selected_frames'] == wrapper['selected_frames'] == archive['selected_frame_ids']
         and held == old_plan['retained_holdouts'] == replay['retained_holdouts']
         and len(rows) == len(selected) == result['frames'] == 24 and not set(selected) & set(held)
         and [reference]+[frame for frame in selected if frame != reference] == [row['frame'] for row in rows]
         and all(all(item['camera'][key] == row[key] for key in ('frame', 'K', 'w2c', 'wh'))
                 for item, row in zip(replay['records'], rows, strict=True)),
         'Round-five selected frames, retained holdouts or cameras differ.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    old_mesh = old/result['mesh']
    mesh_sha = record(old_mesh, old_plan['baseline_portable_mesh_sha256'])
    frozen = old/round5.FROZEN
    need(sha(frozen) == old_plan['frozen_surface']['sha256'] == sha(snapshot/'reference-mesh.npz'),
         'Round-five frozen surface bytes differ.')
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS) and meta['video_sha256'] == old_plan['source_video_sha256']
         and meta['camera_config_sha256'] == old_plan['camera_config_sha256'],
         'Round-five video, camera configuration or asset root differs.')
    record(Path(meta['video']), meta['video_sha256'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    record(asset, old_plan['input_sha256'][str(asset)])
    return dict(subject=subject, baseline_commit=BASE, baseline_run=str(old), baseline_result_sha256=result_sha,
        director_review_sha256=DIRECTOR_SHA, baseline_posed_mesh=str(posed_mesh), baseline_points=result['points'],
        baseline_portable_mesh_sha256=mesh_sha, frozen_surface_source=str(frozen),
        frozen_surface=old_plan['frozen_surface'], camera_pins=camera_pins,
        baseline_initialization=dict(file=str(old/INIT), sha256=proofs['initialization'][1]),
        source_video_sha256=meta['video_sha256'], camera_config_sha256=meta['camera_config_sha256'],
        reference_frame=reference, selected_frames=selected, retained_holdouts=held,
        coverage=selection['coverage'], input_sha256=pins), meta, old_plan


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    plan, meta, old_plan = baseline(subject)
    old = Path(plan['baseline_run'])
    old_profile = read(old/'gaussian-run/config.json')
    need(verify_official_fit(old) == old_profile['official_fit_provenance'],
         'Official frozen fit does not match round five.')
    profile = smpl_sh0_effective_config()
    schedule = profile['unified_main']
    previous_schedule = {key:value for key,value in schedule.items() if key != 'world_scale'}
    need(dict(profile, unified_main=previous_schedule) == old_plan['effective_profile']
         and schedule['world_scale'], 'Candidate changed the stock round-five schedule or lacks world-scale controls.')
    previous_config = read(old/'config.json')
    config = {key:value for key,value in previous_config.items() if key not in old_profile}
    config.update(profile, frozen_surface=plan['frozen_surface'])
    code_files = set(common.SOURCE_FILES) | {name for name in changed if name.startswith('src/')}
    plan.update(schema='shuolee-world-scale-fresh-gaussian-v1', candidate_commit=commit, changed_paths=changed,
        fit_reused=True, fit_calls=0, gaussian_from_scratch=True, cold_video_clock=False,
        steps=10000, final_snapshot=round5.FINAL, effective_profile=profile,
        code_sha256={name:sha(PROJECT/name) for name in sorted(code_files)},
        runtime=common.cpu_runtime(common.clean_env()), script_sha256=sha(Path(__file__).resolve()))
    return plan, meta, config


def world_scale_proof(run, integration, plan):
    import numpy as np
    from plyfile import PlyData

    path = run/WORLD_SCALE
    digest = sha(path)
    descriptor = integration['world_scale']
    need(descriptor['file'] == path.name and descriptor['sha256'] == digest,
         'World-scale audit is not bound to the completed integration receipt.')
    audit = read(path)
    terminal = audit['terminal']
    config = plan['effective_profile']['unified_main']['world_scale']
    result = read(run/'result.json')
    need(audit['state'] == 'completed' and audit['config'] == config and terminal == descriptor['terminal']
         and audit['pose_count'] == len(plan['selected_frames'])
         and audit['reference_pose_index'] == result['reference_pose_index']
         and audit['initial']['step'] == 0 and terminal['step'] == plan['steps']
         and 0 < audit['limits_m']['min'] <= audit['limits_m']['max'] <= audit['extent_m']
         and terminal['reference']['all']['points'] == terminal['all_poses_max']['all']['points'] == result['points']
         and terminal['all_poses_max']['all']['over_limit'] == 0 and audit['checkpoints']
         and [event['step'] for event in audit['splits']] == config['split_steps']
         and all(event['children'] == 2*event['split_parents'] <= 2*event['assessed'] <= 2*config['max_parents']
                 for event in audit['splits']), 'World-scale audit is incomplete.')

    native = PlyData.read(run/'gaussian-run'/result['final_snapshot']/'world.ply')['vertex'].data
    portable = PlyData.read(run/result['ply'])['vertex'].data
    axes = [f'scale_{i}' for i in range(3)]
    need(all(native[key].tobytes() == portable[key].tobytes() for key in (*axes, 'opacity')),
         'Portable PLY changed the reference covariance or opacity.')
    sigma = np.exp(np.stack([portable[key] for key in axes], axis=1).max(axis=1).astype(np.float64))
    cutoff = np.log(config['active_opacity']/(1-config['active_opacity']))
    definite = portable['opacity'] > cutoff+1e-6
    possible = portable['opacity'] >= cutoff-1e-6
    active = terminal['reference']['active']
    need(int(definite.sum()) <= active['points'] <= int(possible.sum()), 'Portable PLY active-point count differs.')
    scopes = [('all', sigma)]
    if np.array_equal(definite, possible) and definite.any():
        scopes.append(('active', sigma[definite]))
    for name, values in scopes:
        record = terminal['reference'][name]
        tolerance = 2e-7
        need(np.allclose(np.quantile(values, [.50, .95, .99, 1.]),
                         [record[key] for key in ('p50_m', 'p95_m', 'p99_m', 'max_m')], rtol=0, atol=tolerance),
             f'Portable PLY {name} world sigma quantiles differ from the reference receipt.')
        for threshold, key in zip(config['audit_thresholds_m'], ('over_005m', 'over_010m'), strict=True):
            need(int((values > threshold+tolerance).sum()) <= record[key] <= int((values > threshold-tolerance).sum()),
                 f'Portable PLY {name} over-threshold count differs.')

    initial = plan['baseline_initialization']
    pin(Path(initial['file']), initial['sha256'])
    pin(run/INIT, initial['sha256'])
    return path, digest


def verify_delivery(root, plan, elapsed):
    run = root/'run'
    integration = read(run/'gaussian-run/integration.json')
    world_scale_proof(run, integration, plan)
    round5.verify_delivery(root, plan, elapsed)


stage = rebound(round5.stage, root_for=root_for)
launch = rebound(round5.launch, root_for=root_for, check_source=check_source, verify_delivery=verify_delivery,
                 PROJECT=PROJECT, __file__=__file__)


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
        print(json.dumps(dict(state='world_scale_training_verified', subject=args.subject,
            frozen_fit_frames=len(plan['selected_frames']), fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

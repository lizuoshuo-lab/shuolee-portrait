"""Stage and verify two fresh runs with bounded native replacement validation."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round8 as round8


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-replacement-20261111')
DELIVERY = round8.DELIVERY.parent.parent/'smpl-delivery-round8/receipt.json'
DELIVERY_SHA = '208df5371dff11026a294ec0facfea6f6aa4200d2b717583f29f2b56c21709bf'
DIRECTOR = round8.DIRECTOR.with_name('round8_director_review_20261111.json')
DIRECTOR_SHA = '91b8fff0579a5c0694b461bfa78114233434567c308c02591014b3d1742a2b32'
ENDPOINT = DELIVERY.parent.parent/'smpl-delivery-round9/endpoint-check/receipt.json'
BASE = '2793338736d2957325f54023d4cbabbdc2e1aa99'
RESULT_PINS = {
    '016': '41e0d7d15c9f71e3536421a4237c4812b5a452c9ed29ed414f1ed3d3e11abb4d',
    '006': '0bda41d0bdca6f83b57a156ad9808ddcfd532410b7d318fd54111ed32b94b36a',
}
round6, round5 = round8.round6, round8.round5
MODULE = round8.MODULE
ALLOWED = {
    MODULE, round6.MODULE, 'src/shuolee_portrait/replacement_selection.py',
    'src/shuolee_portrait/sh0_main.py', 'tests/test_render_support.py',
    'tests/test_round9_coverage.py', 'tests/test_replacement_selection.py', 'tests/test_world_scale.py',
}
ADDED_OPTIONS = {
    'coverage_max_trials', 'coverage_tile_px', 'coverage_local_deficit_px',
    'coverage_region_deficit_px', 'coverage_rgb_mae_increase', 'coverage_local_rgb_mae_increase',
}
common, rebound = round8.common, round8.rebound
need, read, sha, pin = round8.need, round8.read, round8.sha, round8.pin
check_source = rebound(round5.round4.round2.check_source, PROJECT=PROJECT, BASE=BASE,
                       SURFACE=MODULE, ALLOWED=ALLOWED)
imports = rebound(round5.round4.round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round9-{subject}'


def baseline(subject):
    from archive_smpl_delivery_round2 import fresh_proof

    prior = round8.root_for(subject)
    old = prior/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-eight input changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    record(Path(round5.__file__).resolve())
    record(Path(round6.__file__).resolve())
    record(Path(round8.__file__).resolve())
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 8
         and director['receipt_sha256_verified'] == DELIVERY_SHA and set(delivery['subjects']) == set(RESULT_PINS),
         'Unexpected round-eight director review.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    wrapper = read(prior/'wrapper-terminal.json')
    record(prior/'wrapper-terminal.json')
    record(prior/'preregistration.json', wrapper['preregistration_sha256'])
    result = read(result_path)
    old_plan = read(prior/'preregistration.json')
    snapshot = old/'gaussian-run'/round5.FINAL
    need(archive['result'] == str(result_path) and archive['result_sha256'] == wrapper['result_sha256'] == result_sha
         and wrapper['candidate_commit'] == archive['commit'] == BASE
         and wrapper['state'] == result['state'] == 'completed'
         and wrapper['frozen_mesh_verified'] is result['format_passed'] is True
         and result['gaussian_from_scratch'] is result['fit_reused'] is True and result['fit_calls'] == 0
         and result['final_snapshot'] == snapshot.name and result['points'] == archive['points'],
         'Round-eight fresh delivery identity differs.')
    _, integration, proofs = fresh_proof(prior, old, snapshot, result, wrapper, old_plan)
    proofs['world_scale'] = round8.support_proof(old, integration, old_plan)
    for name, (path, digest) in proofs.items():
        record(path, digest)
        published = archive['training_proof'][name]
        need(published['sha256'] == digest, f'Published round-eight {name} differs.')
        record(Path(published['path']), digest)
    for name in (*round5.FIT_FILES, round5.FROZEN):
        record(old/name, old_plan['staged_sha256'][name])
    for name in ('run.json', 'config.json', 'gaussian-run/config.json'):
        record(old/name)
    selection = read(old/'fit/selection.json')
    rows = read(old/'data/training-inputs.json')['records']
    replay_path = old/result['bound_replay']
    record(replay_path, result['bound_replay_sha256'])
    replay = read(replay_path)
    posed_mesh = replay_path.parent/replay['mesh_file']
    record(posed_mesh, replay['mesh_sha256'])
    selected, held, reference = (selection[key] for key in ('selected_frames', 'retained_holdouts', 'reference_frame'))
    need(selected == old_plan['selected_frames'] == wrapper['selected_frames'] == archive['selected_frame_ids']
         and held == old_plan['retained_holdouts'] == replay['retained_holdouts']
         and reference == old_plan['reference_frame'] == result['reference_frame_id']
         and len(rows) == len(selected) == result['frames'] == 24 and set(selected).isdisjoint(held)
         and [reference]+[frame for frame in selected if frame != reference] == [row['frame'] for row in rows],
         'Round-eight selected or retained holdout identities differ.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    cameras = {name: record(old/name, digest) for name, digest in old_plan['camera_pins'].items()}
    need(cameras[round5.CAMERA_FILES[0]] == replay['training_camera_sha256'], 'Round-eight camera receipt differs.')
    mesh_sha = record(old/result['mesh'], old_plan['baseline_portable_mesh_sha256'])
    frozen = old/round5.FROZEN
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS) and meta['video_sha256'] == old_plan['source_video_sha256']
         and meta['camera_config_sha256'] == old_plan['camera_config_sha256'],
         'Round-eight video, camera configuration or assets differ.')
    record(Path(meta['video']), meta['video_sha256'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    record(asset, old_plan['input_sha256'][str(asset)])
    return dict(subject=subject, baseline_commit=BASE, baseline_run=str(old), baseline_result_sha256=result_sha,
        director_review_sha256=DIRECTOR_SHA, baseline_posed_mesh=str(posed_mesh), baseline_points=result['points'],
        baseline_portable_mesh_sha256=mesh_sha, frozen_surface_source=str(frozen),
        frozen_surface=old_plan['frozen_surface'], camera_pins=cameras,
        baseline_initialization=dict(file=str(old/round6.INIT), sha256=proofs['initialization'][1]),
        source_video_sha256=meta['video_sha256'], camera_config_sha256=meta['camera_config_sha256'],
        reference_frame=reference, selected_frames=selected, retained_holdouts=held,
        coverage=selection['coverage'], input_sha256=pins), meta, old_plan


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    tree = ast.parse((PROJECT/MODULE).read_text(encoding='utf-8'))
    [color_weights] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'color_weights']
    need(hashlib.sha256(ast.dump(color_weights, include_attributes=False).encode()).hexdigest()
         == round8.COLOR_WEIGHTS_SHA, 'The native compositing contribution measurement changed.')

    plan, meta, old_plan = baseline(subject)
    endpoint_sha = sha(ENDPOINT)
    endpoint = read(ENDPOINT)
    checks = endpoint['subjects']
    need(endpoint['state'] == 'completed' and endpoint['candidate_commit'] == commit
         and endpoint['baseline_commit'] == BASE and endpoint['baseline_receipt_sha256'] == DELIVERY_SHA
         and endpoint['optimizer_updates'] == 0 and endpoint['previous_run_artifacts_changed'] is False
         and set(checks) == set(RESULT_PINS),
         'The exact candidate has not completed both real-endpoint checks.')
    pin(ENDPOINT, endpoint_sha)
    archived = read(DELIVERY)['subjects']
    for name, item in checks.items():
        path = ENDPOINT.parent/f'{name}.json'
        need(Path(item['file']) == path and path.resolve() == path, 'Unexpected endpoint evidence path.')
        pin(path, item['sha256'])
        detail = read(path)
        event = detail['replacement_event']
        repaired = event['split_parents'] + event['needle']['support_replaced']
        need(detail['subject'] == name and item['state'] == 'verified' and item['optimizer_updates'] == 0
             and item['points_before'] == archived[name]['points']
             and item['points_after'] == item['points_before'] + repaired
             and all(item[key] == detail[key] for key in ('state', 'optimizer_updates', 'points_before',
                                                          'points_after', 'limits'))
             and item['limits']['inherited_children'] == 2*repaired,
             'Endpoint evidence has inconsistent replacements or source point counts.')
        plan['input_sha256'][str(path)] = item['sha256']
    plan['input_sha256'][str(ENDPOINT)] = endpoint_sha
    plan['replacement_endpoint_check'] = dict(file=str(ENDPOINT), sha256=endpoint_sha)

    old = Path(plan['baseline_run'])
    old_profile = read(old/'gaussian-run/config.json')
    need(verify_official_fit(old) == old_profile['official_fit_provenance'],
         'Official frozen fit does not match round eight.')
    profile = smpl_sh0_effective_config()
    schedule = profile['unified_main']
    scales = schedule['world_scale']
    support = scales['render_support']
    old_support = old_plan['effective_profile']['unified_main']['world_scale']['render_support']
    unchanged = {key: value for key, value in support.items() if key not in ADDED_OPTIONS}
    prior_scales = dict(scales, render_support=unchanged)
    need(set(support)-set(old_support) == ADDED_OPTIONS
         and dict(profile, unified_main=dict(schedule, world_scale=prior_scales)) == old_plan['effective_profile']
         and support['replacement_budget'] == 16 and support['coverage_max_trials'] == 32,
         'Candidate changed the round-eight training policy outside replacement validation.')
    config = {key: value for key, value in read(old/'config.json').items() if key not in old_profile}
    config.update(profile, frozen_surface=plan['frozen_surface'])
    code_files = set(old_plan['code_sha256']) | {name for name in changed if name.startswith('src/')}
    plan.update(schema='shuolee-replacement-validation-fresh-gaussian-v1',
        candidate_commit=commit, changed_paths=changed, fit_reused=True, fit_calls=0, gaussian_from_scratch=True,
        cold_video_clock=False, steps=10000, final_snapshot=round5.FINAL, effective_profile=profile,
        code_sha256={name:sha(PROJECT/name) for name in sorted(code_files)},
        runtime=common.cpu_runtime(common.clean_env()), script_sha256=sha(Path(__file__).resolve()))
    return plan, meta, config


def replacement_trial_proof(trial, replaced, frames, options):
    if trial is None:
        need(replaced == 0, 'A replacement has no native validation.')
        return

    records = trial['trials']
    first = records[0]
    count = trial['initial_proposed']
    roster = first['candidate_ids']
    need(first['phase'] == 'initial' and first['added_candidate_ids'] == roster
         and 0 < count == first['proposed'] == len(roster) == len(set(roster)) <= options['replacement_budget']
         and len(records) == trial['trial_count'] <= trial['max_trials'] == options['coverage_max_trials'],
         'The replacement shortlist or evaluation budget differs.')
    for index, record in enumerate(records):
        candidates = record['candidate_ids']
        source = record['frames']
        need(record['index'] == index and record['proposed'] == len(candidates)
             and candidates == record['accepted_prefix_ids'] + record['added_candidate_ids']
             and len(candidates) == len(set(candidates))
             and set(candidates) <= set(roster) and [row['frame'] for row in source] == frames
             and record['accepted'] is (not any(row['failures'] for row in source))
             and all('item' in failure for row in source for failure in row['failures']),
             'A replacement trial lacks native evidence from all registered source views.')
    selected = trial['selected_candidate_ids']
    final = trial['final_trial']
    need(trial['accepted'] is bool(replaced) and len(selected) == replaced
         and trial['proposed'] == (replaced or count)
         and trial['budget_exhausted'] is bool(trial['unresolved_candidate_ids']),
         'The replacement selector and actual point changes differ.')
    if replaced:
        need(final == records[-1] and final['accepted'] is True and final['candidate_ids'] == selected
             and final['phase'] == ('initial' if first['accepted'] else 'final'),
             'The committed combination does not match the final native validation.')
    elif final is not None:
        need(final == records[-1] and final['phase'] == 'final' and final['accepted'] is False,
             'A safe final validation was recorded with no replacement.')
    for rejected in trial['rejected_candidates']:
        failed = records[rejected['trial_index']]
        need(failed['added_candidate_ids'] == [rejected['candidate_id']] and not failed['accepted'],
             'A candidate rejection lacks evidence from its exact tested context.')


def support_proof(run, integration, plan):
    path, digest = round8.support_proof(run, integration, plan)
    audit = read(path)
    rows = read(run/'data/training-inputs.json')['records']
    frames = [row['frame'] for row in rows]
    need(len(frames) == len(set(frames)) == 24 and set(frames) == set(plan['selected_frames']),
         'Replacement proof does not use the frozen 24-frame roster.')
    growth = {(row['step'], row['phase']): row['tracks'] for row in audit['support_history']}
    options = audit['config']['render_support']
    for split in audit['splits']:
        support = split['render_support']
        replacement_trial_proof(support['coverage_trial'], split['needle']['support_replaced'], frames, options)
        replacement_trial_proof(support['near_limit_trial'], split['split_parents'], frames, options)
        near = support['near_limit_interventions']
        tracks = growth[split['step'], 'after_repair']
        need(len(near) == split['split_parents']
             and all(item['action'] == 'near_limit_replace' for item in near)
             and all(str(item['lineage']) in tracks for item in near + support['interventions']),
             'A replacement is missing its persistent lineage after the repair.')
    return path, digest


def verify_delivery(root, plan, elapsed):
    run = root/'run'
    support_proof(run, read(run/'gaussian-run/integration.json'), plan)
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
        print(json.dumps(dict(state='replacement_validation_training_verified', subject=args.subject,
            frozen_fit_frames=len(plan['selected_frames']), fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

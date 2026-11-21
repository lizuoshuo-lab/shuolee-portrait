"""Run the common render-support training policy on the exact round-seven fits."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round7 as round7


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-support-20261111')
DELIVERY = round7.DELIVERY.parent.parent/'smpl-delivery-round7/receipt.json'
DELIVERY_SHA = '465196daa6a388809b8c57b2f050a6ea97f5f59141cedfd7c53e0a9a400f7f4a'
DIRECTOR = round7.DIRECTOR.with_name('round7_director_review_20261111.json')
DIRECTOR_SHA = '0fa21ef3a9cceb995ebff5307528656a506c4e36e8336f2ff143ac830e2ad26d'
DIAGNOSTIC = DELIVERY.parent.parent/'smpl-delivery-round8/render-support-diagnostic/receipt.json'
DIAGNOSTIC_SHA = 'c782eae5f035248bd95986fb5c7897d4ee4af9ad180acc31b3cc6fc00ba9b580'
COLOR_WEIGHTS_SHA = '2dabbeff07a33bc0706e67498f2253613e63ccdc8253cd5d3f2e01a21644b388'
BASE = 'dd99e967e9c8f146b99a45df0855f3f7f4bade78'
RESULT_PINS = {
    '016': '1253d15b429171756f818d84f6afbc2932ff259277461413c507f69152986b17',
    '006': '6402db8b466e87a9eba67bbda4d13d0819260a221a02db3dc03cab4fd1278bd4',
}
MODULE = 'src/shuolee_portrait/render_support.py'
round6, round5 = round7.round6, round7.round5
ALLOWED = {
    MODULE, round6.MODULE, round5.MODULE, 'src/shuolee_portrait/sh0_main.py',
    'src/shuolee_portrait/shape_refinement.py', 'src/avatar_core/scene/gaussian_model.py',
    'src/avatar_core/scene/smplx_gaussian_model.py', 'tests/test_world_scale.py', 'tests/test_render_support.py',
}
common, rebound = round7.common, round7.rebound
need, read, sha, pin = round7.need, round7.read, round7.sha, round7.pin
check_source = rebound(round5.round4.round2.check_source, PROJECT=PROJECT, BASE=BASE,
                       SURFACE=MODULE, ALLOWED=ALLOWED)
imports = rebound(round5.round4.round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round8-{subject}'


def baseline(subject):
    from archive_smpl_delivery_round2 import fresh_proof

    prior = round7.root_for(subject)
    old = prior/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-seven input changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    record(Path(round5.__file__).resolve())
    record(Path(round6.__file__).resolve())
    record(Path(round7.__file__).resolve())
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 7
         and director['receipt_sha256_verified'] == DELIVERY_SHA and set(delivery['subjects']) == set(RESULT_PINS),
         'Unexpected round-seven director review.')
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
         'Round-seven fresh delivery identity differs.')
    _, integration, proofs = fresh_proof(prior, old, snapshot, result, wrapper, old_plan)
    proofs['world_scale'] = round7.needle_proof(old, integration, old_plan)
    for name, (path, digest) in proofs.items():
        record(path, digest)
        published = archive['training_proof'][name]
        need(published['sha256'] == digest, f'Published round-seven {name} differs.')
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
         'Round-seven selected or retained holdout identities differ.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    cameras = {name: record(old/name, digest) for name, digest in old_plan['camera_pins'].items()}
    need(cameras[round5.CAMERA_FILES[0]] == replay['training_camera_sha256'], 'Round-seven camera receipt differs.')
    mesh_sha = record(old/result['mesh'], old_plan['baseline_portable_mesh_sha256'])
    frozen = old/round5.FROZEN
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS) and meta['video_sha256'] == old_plan['source_video_sha256']
         and meta['camera_config_sha256'] == old_plan['camera_config_sha256'],
         'Round-seven video, camera configuration or assets differ.')
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


def bind_diagnostic(plan):
    pin(DIAGNOSTIC, DIAGNOSTIC_SHA)
    evidence = read(DIAGNOSTIC)
    synthetic = evidence['synthetic']
    need(evidence['schema'] == 'shuolee-round8-native-render-support-diagnostic-v1'
         and evidence['state'] == 'completed' and evidence['optimizer_updates'] == 0
         and evidence['r6_delivery_sha256'] == round7.DELIVERY_SHA and evidence['r7_delivery_sha256'] == DELIVERY_SHA
         and evidence['r7_causal_receipt_sha256'] == round7.CAUSAL_SHA
         and evidence['candidate_color_helper_path'] == str(PROJECT/MODULE)
         and synthetic['state'] == 'validated' and synthetic['restored_max_abs'] == 0
         and synthetic['expected_center_front_back'] == synthetic['measured_center_front_back']
         and synthetic['finite_difference_max_abs'] < .0001
         and synthetic['candidate_helper_under_no_grad_max_abs'] < .00001,
         'The bounded render-support diagnosis is incomplete.')
    subject = plan['subject']
    item = evidence['subjects'][subject]
    files = dict(receipt=dict(path=str(DIAGNOSTIC), sha256=DIAGNOSTIC_SHA),
                 subject_result=dict(path=item['path'], sha256=item['sha256']), panel=item['panel'])
    for proof in files.values():
        path = Path(proof['path'])
        need(path.is_absolute() and path.resolve().parent == DIAGNOSTIC.parent,
             'Unexpected render-support diagnosis location.')
        pin(path, proof['sha256'])
        plan['input_sha256'][str(path)] = proof['sha256']
    detail = read(Path(item['path']))
    old = Path(plan['baseline_run'])
    frames = [row['frame'] for row in detail['frames']]
    need(detail['schema'] == evidence['schema'] and detail['state'] == 'completed'
         and detail['subject'] == subject and detail['r6_result_sha256'] == round7.RESULT_PINS[subject]
         and detail['no_gaussian_state_change'] is True and detail['optimizer_updates'] == 0
         and detail['training_inputs_sha256'] == plan['input_sha256'][str(old/'data/training-inputs.json')]
         and detail['training_camera_sha256'] == plan['camera_pins'][round5.CAMERA_FILES[0]]
         and detail['retained_holdouts'] == plan['retained_holdouts'] and detail['panel'] == item['panel']
         and detail['frame_count'] == item['frame_count'] == len(frames) == len(set(frames)) == 24
         and set(frames) == set(plan['selected_frames']),
         'Render-support evidence differs from the registered camera or fit roster.')
    plan['render_support_diagnostic'] = dict(files=files, selected_frames=frames,
                                             color_weights_function_sha256=COLOR_WEIGHTS_SHA)


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    tree = ast.parse((PROJECT/MODULE).read_text(encoding='utf-8'))
    [color_weights] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'color_weights']
    need(hashlib.sha256(ast.dump(color_weights, include_attributes=False).encode()).hexdigest() == COLOR_WEIGHTS_SHA,
         'Candidate changed the color derivative verified by the occlusion fixture.')
    plan, meta, old_plan = baseline(subject)
    bind_diagnostic(plan)
    old = Path(plan['baseline_run'])
    old_profile = read(old/'gaussian-run/config.json')
    need(verify_official_fit(old) == old_profile['official_fit_provenance'],
         'Official frozen fit does not match round seven.')
    profile = smpl_sh0_effective_config()
    schedule = profile['unified_main']
    scales = schedule['world_scale']
    previous_scales = {key:value for key,value in scales.items() if key != 'render_support'}
    need(dict(profile, unified_main=dict(schedule, world_scale=previous_scales)) == old_plan['effective_profile']
         and scales['render_support'], 'Candidate changed the stock round-seven schedule or lacks render-support controls.')
    config = {key:value for key,value in read(old/'config.json').items() if key not in old_profile}
    config.update(profile, frozen_surface=plan['frozen_surface'])
    code_files = set(common.SOURCE_FILES) | {name for name in changed if name.startswith('src/')}
    plan.update(schema='shuolee-render-support-fresh-gaussian-v1', candidate_commit=commit, changed_paths=changed,
        fit_reused=True, fit_calls=0, gaussian_from_scratch=True, cold_video_clock=False,
        steps=10000, final_snapshot=round5.FINAL, effective_profile=profile,
        code_sha256={name:sha(PROJECT/name) for name in sorted(code_files)},
        runtime=common.cpu_runtime(common.clean_env()), script_sha256=sha(Path(__file__).resolve()))
    return plan, meta, config


def support_proof(run, integration, plan):
    path, digest = round7.needle_proof(run, integration, plan)
    audit = read(path)
    for split in audit['splits']:
        support, needle = split['render_support'], split['needle']
        interventions = support['interventions']
        counts = ('foreground_supported', 'center_unobserved_with_foreground', 'ambiguous')
        masses = ('foreground_equivalent_pixels', 'background_equivalent_pixels', 'total_equivalent_pixels')
        need(all(type(support[key]) is int and 0 <= support[key] <= needle['assessed'] for key in counts)
             and support['center_unobserved_with_foreground'] <= support['foreground_supported']
             and all(math.isfinite(support[key]) and support[key] >= 0 for key in masses)
             and len(interventions) == needle['width_trimmed'] + needle['support_replaced']
             and sum(item['action'] == 'trim' for item in interventions) == needle['width_trimmed']
             and sum(item['action'] == 'replace' for item in interventions) == needle['support_replaced'],
             f'Incomplete render-support event at step {split["step"]}.')
        for replaced, trial in ((needle['support_replaced'], support['coverage_trial']),
                                (split['split_parents'], support['near_limit_trial'])):
            if replaced:
                need(trial['accepted'] is True and trial['proposed'] == replaced,
                     'A support replacement lacks its recorded native-alpha check.')
    summary = audit['terminal']['render_support']
    keys = ('lineages_registered', 'live_points', 'projected_point_updates', 'inherited_children', 'pruned_points')
    growth = summary['max_preprojection_growth_ratio']
    need(all(type(summary[key]) is int and summary[key] >= 0 for key in keys)
         and summary['live_points'] == summary['lineages_registered'] + summary['inherited_children'] - summary['pruned_points']
         and math.isfinite(growth) and growth >= 1,
         'Render-support lineage accounting differs.')
    expected = [(step, phase) for step in audit['config']['split_steps'] for phase in ('before_repair', 'after_repair')]
    expected.append((plan['steps'], 'terminal'))
    history = audit['support_history']
    need([(row['step'], row['phase']) for row in history] == expected
         and sum(track['points'] for track in history[-1]['tracks'].values()) == summary['live_points']
         and all(0 <= row['approved'] <= row['proposed'] for row in audit['support_prunes']),
         'Persistent render-support measurements are incomplete.')
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
        print(json.dumps(dict(state='render_support_training_verified', subject=args.subject,
            frozen_fit_frames=len(plan['selected_frames']), fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

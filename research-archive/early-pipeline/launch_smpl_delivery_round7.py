"""Run fresh SH0 with the registered needle policy on the exact round-six fits."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round6 as round6


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-needle-20261111')
DELIVERY = round6.DELIVERY.parent.parent/'smpl-delivery-round6/receipt.json'
DELIVERY_SHA = 'a728539832d89b4f4d11b644d1d2398f1eedb157f13e021eaf2234e3b5658c1c'
DIRECTOR = round6.DIRECTOR.with_name('round6_director_review_20261111.json')
DIRECTOR_SHA = 'e482373de61adf8bd785842a2005007cdc4d25d851ad3e0fe17fb98819a81450'
CAUSAL = DELIVERY.parent.parent/'smpl-delivery-round7/causal-diagnostic/receipt.json'
CAUSAL_SHA = '8d44e129dad6bfdacdb38dcb1c794723c930d4739b26e450df10f6f314a73939'
BASE = '8c722b9aabd7577803118c910bb9971a82f009d6'
RESULT_PINS = {
    '016': 'e182907e5dc067055f745927de1389ab141eb86560a0fbc0f082333aa495bfd6',
    '006': '668cb4614158dd5d6c6fe836673b9644631de794ddb0c789f903677706d51dfd',
}
ALLOWED = {round6.MODULE, 'tests/test_world_scale.py'}
round5, common, rebound = round6.round5, round6.common, round6.rebound
need, read, sha, pin = round6.need, round6.read, round6.sha, round6.pin
check_source = rebound(round5.round4.round2.check_source, PROJECT=PROJECT, BASE=BASE,
                       SURFACE=round6.MODULE, ALLOWED=ALLOWED)
imports = rebound(round5.round4.round2.imports, PROJECT=PROJECT)


def root_for(subject):
    return common.LAB/f'shuolee-smpl-delivery-round7-{subject}'


def baseline(subject):
    from archive_smpl_delivery_round2 import fresh_proof

    old_root = round6.root_for(subject)
    old = old_root/'run'
    pins = {}

    def record(path, expected=None):
        digest = sha(path)
        need((expected is None or digest == expected) and pins.get(str(path), digest) == digest,
             f'Round-six input changed: {path}')
        pins[str(path)] = digest
        return digest

    record(DELIVERY, DELIVERY_SHA)
    record(DIRECTOR, DIRECTOR_SHA)
    record(Path(round5.__file__).resolve())
    record(Path(round6.__file__).resolve())
    delivery, director = read(DELIVERY), read(DIRECTOR)
    need(delivery['commit'] == director['commit'] == BASE and director['round'] == 6
         and director['receipt_sha256_verified'] == DELIVERY_SHA and set(delivery['subjects']) == set(RESULT_PINS),
         'Unexpected round-six director review.')
    archive = delivery['subjects'][subject]
    result_path = old/'result.json'
    result_sha = record(result_path, RESULT_PINS[subject])
    need(archive['result'] == str(result_path) and archive['result_sha256'] == result_sha,
         'Round-six terminal differs from published delivery.')
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
         'Round-six fresh delivery identity differs.')
    _, integration, proofs = fresh_proof(old_root, old, snapshot, result, wrapper, old_plan)
    proofs['world_scale'] = round6.world_scale_proof(old, integration, old_plan)
    for name, (path, digest) in proofs.items():
        record(path, digest)
        published = archive['training_proof'][name]
        need(published['sha256'] == digest, f'Published round-six {name} differs.')
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
         'Round-six selected frames, retained holdouts or cameras differ.')
    for row in rows:
        record(Path(row['image']), row['source_sha256'])
        record(Path(row['mask']), row['mask_sha256'])
    mesh_sha = record(old/result['mesh'], old_plan['baseline_portable_mesh_sha256'])
    frozen = old/round5.FROZEN
    need(sha(frozen) == old_plan['frozen_surface']['sha256'] == sha(snapshot/'reference-mesh.npz'),
         'Round-six frozen surface bytes differ.')
    meta = read(old/'run.json')
    need(meta['assets'] == str(common.ASSETS) and meta['video_sha256'] == old_plan['source_video_sha256']
         and meta['camera_config_sha256'] == old_plan['camera_config_sha256'],
         'Round-six video, camera configuration or asset root differs.')
    record(Path(meta['video']), meta['video_sha256'])
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    record(asset, old_plan['input_sha256'][str(asset)])
    return dict(subject=subject, baseline_commit=BASE, baseline_run=str(old), baseline_result_sha256=result_sha,
        director_review_sha256=DIRECTOR_SHA, baseline_posed_mesh=str(posed_mesh), baseline_points=result['points'],
        baseline_portable_mesh_sha256=mesh_sha, frozen_surface_source=str(frozen),
        frozen_surface=old_plan['frozen_surface'], camera_pins=camera_pins,
        baseline_initialization=dict(file=str(old/round6.INIT), sha256=proofs['initialization'][1]),
        source_video_sha256=meta['video_sha256'], camera_config_sha256=meta['camera_config_sha256'],
        reference_frame=reference, selected_frames=selected, retained_holdouts=held,
        coverage=selection['coverage'], input_sha256=pins), meta, old_plan


def bind_causal(plan):
    pin(CAUSAL, CAUSAL_SHA)
    causal = read(CAUSAL)
    need(causal['schema'] == 'shuolee-round6-static-artifact-causal-v1' and causal['state'] == 'completed'
         and causal['delivery_sha256'] == DELIVERY_SHA and causal['optimizer_updates'] == 0
         and set(causal['subjects']) == set(RESULT_PINS), 'The bounded contribution diagnosis is incomplete.')
    subject = plan['subject']
    item = causal['subjects'][subject]
    causal_files = dict(receipt=dict(path=str(CAUSAL), sha256=CAUSAL_SHA),
                        subject_result=dict(path=item['path'], sha256=item['sha256']), panel=item['panel'])
    for proof in causal_files.values():
        path = Path(proof['path'])
        need(path.is_absolute() and path.resolve().parent == CAUSAL.parent,
             'Unexpected contribution diagnosis location.')
        pin(path, proof['sha256'])
        plan['input_sha256'][str(path)] = proof['sha256']
    detail = read(Path(item['path']))
    control = detail['interventions']
    old = Path(plan['baseline_run'])
    result = read(old/'result.json')
    need(detail['schema'] == causal['schema'] and detail['delivery_sha256'] == DELIVERY_SHA
         and detail['subject'] == subject and detail['view'] == item['view']
         and detail['ply'] == str(old/result['ply']) and detail['ply_sha256'] == result['sha256']
         and detail['optimizer_updates'] == 0 and len(detail['top']) == item['point_count']
         and control['panel'] == item['panel'] and control['restored_bitwise'] is True
         and control['black_white_alpha_exact_archive'] is True, 'Contribution evidence is not the restored round-six terminal.')
    plan['causal_diagnostic'] = dict(files=causal_files, view=item['view'], point_count=item['point_count'])


def prepare(subject, commit):
    changed = check_source(commit)
    imports()
    from shuolee_portrait.training import smpl_sh0_effective_config, verify_official_fit

    plan, meta, old_plan = baseline(subject)
    bind_causal(plan)
    old = Path(plan['baseline_run'])
    old_profile = read(old/'gaussian-run/config.json')
    need(verify_official_fit(old) == old_profile['official_fit_provenance'],
         'Official frozen fit does not match round six.')
    profile = smpl_sh0_effective_config()
    schedule = profile['unified_main']
    scales = schedule['world_scale']
    previous_scales = {key:value for key,value in scales.items() if key != 'needle'}
    need(dict(profile, unified_main=dict(schedule, world_scale=previous_scales)) == old_plan['effective_profile']
         and scales['needle'], 'Candidate changed the stock round-six schedule or lacks needle controls.')
    config = {key:value for key,value in read(old/'config.json').items() if key not in old_profile}
    config.update(profile, frozen_surface=plan['frozen_surface'])
    code_files = set(common.SOURCE_FILES) | {round6.MODULE, 'src/shuolee_portrait/sh0_main.py'}
    plan.update(schema='shuolee-needle-fresh-gaussian-v1', candidate_commit=commit, changed_paths=changed,
        fit_reused=True, fit_calls=0, gaussian_from_scratch=True, cold_video_clock=False,
        steps=10000, final_snapshot=round5.FINAL, effective_profile=profile,
        code_sha256={name:sha(PROJECT/name) for name in sorted(code_files)},
        runtime=common.cpu_runtime(common.clean_env()), script_sha256=sha(Path(__file__).resolve()))
    return plan, meta, config


def needle_proof(run, integration, plan):
    import numpy as np
    from plyfile import PlyData

    path, digest = round6.world_scale_proof(run, integration, plan)
    audit = read(path)
    terminal = audit['terminal']['needle']
    event_keys = ('displaced_candidates', 'assessed', 'conflicted', 'unobserved',
                  'support_replaced', 'children', 'width_trimmed', 'preserved')
    for split in audit['splits']:
        event = split['needle']
        need(all(type(event[key]) is int and event[key] >= 0 for key in event_keys)
             and event['assessed'] <= event['displaced_candidates']
             and event['children'] == 2*event['support_replaced']
             and event['assessed'] == event['support_replaced']+event['width_trimmed']+event['preserved'],
             f'Incomplete support-replacement accounting at step {split["step"]}.')

    result = read(run/'result.json')
    ply = PlyData.read(run/result['ply'])['vertex'].data
    log_scales = np.sort(np.stack([ply[f'scale_{i}'] for i in range(3)], axis=1).astype(np.float64), axis=1)
    ratios = np.exp(log_scales[:, 2]-log_scales[:, 1])
    opacity = audit['config']['active_opacity']
    cutoff = np.log(opacity/(1-opacity))
    definite = ply['opacity'] > cutoff+1e-6
    possible = ply['opacity'] >= cutoff-1e-6
    for threshold in (20, 40):
        tolerance = threshold*1e-6
        recorded = terminal[f'active_ratio_over_{threshold}']
        need(int((definite & (ratios > threshold+tolerance)).sum()) <= recorded
             <= int((possible & (ratios > threshold-tolerance)).sum()), 'Portable PLY axis ratio count differs.')
    if definite.any():
        lower = float(ratios[definite].max())
        upper = float(ratios[possible].max())
        maximum = terminal['active_ratio_max']
        need(lower*(1-1e-6) <= maximum <= upper*(1+1e-6), 'Portable PLY maximum active axis ratio differs.')
    return path, digest


def verify_delivery(root, plan, elapsed):
    run = root/'run'
    needle_proof(run, read(run/'gaussian-run/integration.json'), plan)
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
        print(json.dumps(dict(state='needle_training_verified', subject=args.subject,
            frozen_fit_frames=len(plan['selected_frames']), fit_calls=0, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()

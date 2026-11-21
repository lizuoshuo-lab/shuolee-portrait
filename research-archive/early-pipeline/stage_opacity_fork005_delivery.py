"""Stage a visually reviewed 005 omit-reset comparison using the existing exporter."""

import argparse
import json
import os
from pathlib import Path
from types import FunctionType

import review_opacity_fork005_holdouts as review
import run_opacity_fork005 as fork
import stage_opacity_fork001_delivery as existing
from review_bound_alpha_ab import pin, read, require, sha


OUT = fork.OUT.with_name(fork.OUT.name + '-delivery')
REFERENCE = 'frame_000020'
EARLY_PINS = {
    'native-ply-parity.json': 'e87e670f799169fcc1751958bec7e193417f52c2cc6bfb3cd1a1ea30db7aba8b',
    'reference-mesh.npz': '9be09f6f31e50e48c22a108392b943380cac3981b6c9aa979240e0d116c8ed8f',
    'reference-render.png': 'e5214330a2a2f7368b7b864208d090d13e63b2e2386a5bd445c9c52d5d51c579',
    'world.ply': '3d26c0f6b918bf0a7fbeb044774bace54a817027c276b5309e63b02346ac435b',
}
HELPER_SHA = '5546e4e29b7f11ccd12cd13bf001dd5533b90df960be8c7a6c519ee15d9d1049'
REVIEW_SHA = 'f7875c0b0a153ee397f67f1ff27de75cbd54b7c896cce87ebeb678d9e4e3e679'


def prepare(fork_sha, heldout_sha):
    require(not OUT.exists(), 'Preserve an existing comparison delivery.')
    pin(Path(existing.__file__).resolve(), HELPER_SHA)
    pin(Path(review.__file__).resolve(), REVIEW_SHA)
    pin(fork.OUT / 'result.json', fork_sha)
    holdout_path = review.OUT / 'result.json'
    pin(holdout_path, heldout_sha)
    held = read(holdout_path)
    plan, state, poses, views = review.frozen_inputs()
    require(plan == read(review.REGISTRATION), 'Independent heldout registration differs.')
    rt, _, arms, evidence = review.terminal_inputs(plan, state, poses, views)
    arm = arms[existing.ARM]
    require(held['state'] == 'matched_005_holdout_replay_complete_pending_visual_review'
            and held['fork_result_sha256'] == fork_sha == evidence['fork_result_sha256']
            and held['checkpoint_sha256'] == evidence['checkpoint_sha256']
            and held['holdout_registration_sha256'] == evidence['holdout_registration_sha256']
            and held['evaluation_fit_sha256'] == plan['evaluation_fitted_sha256']
            and held['optimizer_updates'] == held['evaluation_refits'] == 0
            and held['quality_accepted'] is False
            and list(held['terminal_models']) == list(arms)
            and all(held['terminal_models'][name] == item.terminal for name, item in arms.items())
            and [row['frame'] for row in held['primary_records']] == list(review.PRIMARY),
            'Comparison does not match the frozen pose and terminal arm.')
    review_panels = {str(review.OUT / row['image']): pin(review.OUT / row['image'], row['image_sha256'])
                     for row in held['records']}
    source, stock = arm.root, arms['stock'].root
    require(arm.facts['reference_frame_id'] == REFERENCE
            and read(source / 'run.json')['fit_reused'] is True
            and read(source / 'gaussian-run/result.json')['from_scratch'] is False
            and read(stock / 'gaussian-run/result.json')['from_scratch'] is True
            and arm.terminal['restored_exact'] is True
            and arm.terminal['new_optimizer_updates'] == 1000,
            'The comparison and original common-prefix provenance differ.')
    paths = {name: source / name for name in ('run.json', 'config.json', 'fit/fitted.npz',
        'fit/official-fit-result.json', 'fit/selection.json', 'fit/shared-body.json', 'data/training-inputs.json',
        'gaussian-run/config.json', 'gaussian-run/result.json', 'gaussian-run/loss.jsonl',
        'gaussian-run/training-cameras.json')}
    paths.update({f'{existing.TERMINAL}/{name}': source / existing.TERMINAL / name for name in (
        'world.ply', 'bound-local.ply', 'bound-replay.json', 'reference-mesh.npz',
        'reference-render.png', 'native-ply-parity.json')})
    for name, digest in EARLY_PINS.items():
        path = stock / existing.EARLY / name
        pin(path, digest)
        paths[f'{existing.EARLY}/{name}'] = path
    early = read(stock / existing.EARLY / 'native-ply-parity.json')
    require(early['passed'] is True and early['reference_frame_id'] == REFERENCE
            and early['reference_pose_index'] == 0 and early['sha256'] == EARLY_PINS['world.ply']
            and early['mesh_sha256'] == EARLY_PINS['reference-mesh.npz'],
            'Common-prefix native 1k render did not bind to its saved mesh and PLY.')
    lineage = dict(schema='shuolee-bound-comparison-v1', arm=existing.ARM, from_scratch=False,
        common_prefix_from_scratch=True, fit_reused=True, source_run=str(source),
        common_prefix_run=str(stock), source_fork=str(fork.OUT), checkpoint_sha256=evidence['checkpoint_sha256'],
        fork_result_sha256=fork_sha, heldout_review_sha256=heldout_sha, heldout_panel_sha256=review_panels,
        evaluation_fit_sha256=held['evaluation_fit_sha256'], reference_frame=REFERENCE, points=arm.terminal['points'],
        training_camera_sha256=arm.terminal['training_camera_sha256'], original_arm_optimizer_updates=1000,
        optimizer_updates=0, cold_video_clock=False, comparison_only=True, quality_accepted=False,
        stage_helper_sha256=HELPER_SHA, stage_driver_sha256=sha(Path(__file__).resolve()),
        copied_files_sha256={name: sha(path) for name, path in paths.items()})
    return rt, paths, lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork-result-sha256', required=True)
    parser.add_argument('--heldout-review-sha256', required=True)
    parser.add_argument('--stage-reviewed-comparison', action='store_true')
    args = parser.parse_args()
    if not args.stage_reviewed_comparison:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    rt, paths, lineage = prepare(args.fork_result_sha256, args.heldout_review_sha256)
    if not args.stage_reviewed_comparison:
        print(json.dumps(dict(state='comparison_delivery_inputs_verified', output=str(OUT), gpu_calls=0, **lineage)))
        return
    stage = FunctionType(existing.stage.__code__, dict(vars(existing), REFERENCE=REFERENCE))
    stage(OUT, rt, paths, lineage)


if __name__ == '__main__':
    main()

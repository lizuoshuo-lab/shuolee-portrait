"""Audit a cold 005 repeat and replay only its already-fitted independent holdouts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

import review_005_formal_holdouts as formal
import review_006_cold_candidate_holdouts as matched
from review_bound_alpha_ab import pin, read, require, sha


LAB = Path('/opt/gs-slam-lab')
BASE = LAB / 'shuolee-smpl-coverage-005/run'
RUN = LAB / 'shuolee-smpl-sixworker-ffprobe-005/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-sixworker-ffprobe-005')
ARCHIVE = LAB / 'shuolee-smpl-coverage-005-formal'
OUTPUT = LAB / 'shuolee-smpl-sixworker-ffprobe-005-formal'
HELD = {'frame_000011': 80, 'frame_000027': 208, 'frame_000042': 328, 'frame_000055': 433}
PRIMARY = tuple(HELD)[:3]
TARGET = '5fdcf3949d95c96f4804ff6b7e6dcd22b56689a160e7a7a051fbd2d5ad61df49'
FIXED = {
    RUN.parent / 'preregistration.json': 'ae6964dc89c16f6b8d581c9d21a9802048b9d08913a9ddf84ffb6edcde20936b',
    RUN.parent / 'launch.json': '37d34dba775cec5500dd1bf6cd8035cf4a1bf2d5a0d230dc69e353ed9e041c29',
    ARCHIVE / 'evaluation-fitted.npz': '56ee461a693d176708bad5593eac4ee96f854328e8acc2d303ce5eac0d733a3b',
    ARCHIVE / 'fit-result.json': '0ef086056ec1459bd82ed44c2cbf5f84b59ed020091c7063131507c456019531',
    ARCHIVE / 'inputs.json': '1ed3942a80017a81e0d21c3cc215ede46267539113edbd1979e0b65117f7a6ba',
    ARCHIVE / 'evaluation-scope.json': '2d388dbac00d104814cc85c31d42ead80c19e3ff7045850361dff6d8de55deec',
    ARCHIVE / 'renders/result.json': 'b4fad17398d85a6809712caa11e66f8059c5436ae3968109d9e7a3d9d3300dc6',
    Path(formal.__file__): '55cb2d9d52e3ed36522374eea0a48218a0861b364387379c04f87a5ba4c57cd3',
    Path(matched.__file__): 'fba497752658df234da5049daaf129ae812d5aec2df0d8c1755358d18688b5b9',
}
ROW_PATHS = {'image', 'mask', 'mask_sha256'}


def rows_parity(base, new, pins):
    require(base.keys() == new.keys(), 'Native source frame roster differs.')
    differences, masks = {}, {}
    for frame, a in base.items():
        b = new[frame]
        differences[frame] = sorted(key for key in (a.keys() | b.keys()) - ROW_PATHS
                                    if a.get(key) != b.get(key))
        for row in (a, b):
            pins[Path(row['image'])] = pin(Path(row['image']), row['source_sha256'])
            pins[Path(row['mask'])] = pin(Path(row['mask']), row['mask_sha256'])
        if a['mask_sha256'] != b['mask_sha256']:
            with Image.open(a['mask']) as left, Image.open(b['mask']) as right:
                p, q = (np.asarray(im.convert('L')) for im in (left, right))
                require(p.shape == q.shape, 'Fresh mask raster differs.')
                pos, other = p >= 128, q >= 128
                union = np.count_nonzero(pos | other)
                masks[frame] = dict(base_sha256=a['mask_sha256'], candidate_sha256=b['mask_sha256'],
                                    grayscale_pixels_equal=np.array_equal(p, q),
                                    mask_iou=float(np.count_nonzero(pos & other) / union) if union else None)
    return {frame: keys for frame, keys in differences.items() if keys}, masks


def preflight():
    for path, digest in FIXED.items():
        pin(path, digest)
    launch, cfg, _, held, primary, outside = formal.cohort(RUN, PROJECT)
    base_cfg = read(BASE / 'config.json')
    static_cfg = cfg.keys() | base_cfg.keys()
    require(held == HELD and tuple(primary) == PRIMARY and outside == [next(reversed(HELD))]
            and launch['candidate_commit'] == '1f490565ee6074cd8ad541686d3bce6585c8f7fc'
            and launch['cpu_workers'] == 6 and launch['ffprobe_decoder_threads'] == 4
            and cfg.keys() == base_cfg.keys()
            and all(cfg[key] == base_cfg[key] for key in static_cfg - {'image_cache_frames'}),
            'Wrong cold repeat, camera, holdouts, or resolved training options.')
    pin(RUN.parent / 'deployment.json', launch['deployment_config_sha256'])
    pin(BASE / 'result.json', launch['baseline_result_sha256'])
    pin(BASE / 'fit/selection.json', launch['baseline_selection_sha256'])
    archive_input = read(ARCHIVE / 'inputs.json')
    archive_fit = read(ARCHIVE / 'fit-result.json')
    archive_render = read(ARCHIVE / 'renders/result.json')
    archive_scope = read(ARCHIVE / 'evaluation-scope.json')
    require(archive_fit['evaluation_only'] is archive_input['evaluation_only'] is True
            and archive_fit['gaussian_optimizer_updates'] == archive_input['training_or_model_updates'] == 0
            and archive_fit['protected_frames'] == archive_input['protected_frames'] == archive_render['protected_frames'] == HELD
            and archive_scope['evaluated_frames'] == list(HELD)
            and archive_scope['primary_within_capture_arc'] == list(PRIMARY)
            and archive_render['fit_result_sha256'] == FIXED[ARCHIVE / 'fit-result.json']
            and archive_fit['input_receipt_sha256'] == FIXED[ARCHIVE / 'inputs.json']
            and archive_fit['fitted_sha256'] == archive_scope['fitted_sha256'] == FIXED[ARCHIVE / 'evaluation-fitted.npz']
            and archive_fit['shared_target_sha256'] == TARGET,
            'Archived independent 005 evaluation identity differs.')
    pins = dict(FIXED)
    pins.update({Path(path): pin(Path(path), digest) for path, digest in archive_input['input_sha256'].items()})
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    base_rows, new_rows = (matched.source_rows(root, 'data/inputs.json') for root in (BASE, RUN))
    row_changes, masks = rows_parity(base_rows, new_rows, pins)
    a_obs, b_obs = (root / 'data/observations.npz' for root in (BASE, RUN))
    obs_changes = matched.compare_observations(a_obs, b_obs)
    a_fit, b_fit = (read(root / 'fit/official-fit-result.json') for root in (BASE, RUN))
    a_shared, b_shared = (read(root / 'fit/shared-body.json') for root in (BASE, RUN))
    a_selection, b_selection = (read(root / 'fit/selection.json') for root in (BASE, RUN))
    for root, receipt, obs in ((BASE, a_fit, a_obs), (RUN, b_fit, b_obs)):
        pins[obs] = pin(obs, receipt['source']['input_files'][str(obs)])
        pins[root / 'fit/shared-body.json'] = pin(root / 'fit/shared-body.json', receipt['shared_body_sha256'])
        pins[root / 'fit/fitted.npz'] = pin(root / 'fit/fitted.npz', receipt['fit_sha256'])
        pins[root / 'data/training-inputs.json'] = pin(root / 'data/training-inputs.json', receipt['training_inputs_sha256'])
    fit_changes = matched.compare_observations(BASE / 'fit/fitted.npz', RUN / 'fit/fitted.npz')
    boundary = matched.verify_mask_boundary(PROJECT, SimpleNamespace(pose_archive=ARCHIVE))
    boundary['scope'] = ('Original CPU pose fit consumes RGB, keypoints/confidence, complete K and fixed body inputs. '
                         'Mask file identity and dimensions are checked. Scoring uses the exact archived 252 masks.')
    pins.update({Path(path): digest for path, digest in boundary['source_pins'].items()})
    b_result = read(RUN / 'result.json')
    a_profile, b_profile = (read(root / 'gaussian-run/config.json') for root in (BASE, RUN))
    profile_keys = (static_cfg & a_profile.keys()) - {'image_cache_frames'}
    require(all(a_profile[key] == b_profile[key] for key in profile_keys)
            and a_profile['optimizer_initial_state_entries'] == b_profile['optimizer_initial_state_entries'] == 0
            and a_profile['image_cache_frames'] == base_cfg['image_cache_frames'] == len(a_selection['selected_frames'])
            and b_profile['image_cache_frames'] == cfg['image_cache_frames'] == len(b_selection['selected_frames'])
            and b_result['from_scratch'] is True and b_fit['execution_kind'] == 'fresh_official_sequence',
            'Training profile or cold initialization differs.')
    b_anchors = {frame for group in b_shared['anchor_groups'].values() for frame in group}
    actual_anchors = {p.name for p in (RUN / 'fit/shape-anchors').iterdir() if p.is_dir()}
    fitted = {p.name for p in (RUN / 'fit/frames').iterdir() if p.is_dir()}
    selected = set(b_selection['selected_frames'])
    require(len(new_rows) == launch['expected_native_samples'] and len(fitted) == launch['expected_fit_candidates']
            and len(b_anchors) == launch['expected_shape_anchors'] and b_anchors == actual_anchors <= fitted
            and b_selection['retained_holdouts'] == list(HELD)
            and not set(HELD) & (fitted | b_anchors | selected),
            'Protected views overlap or cold fit roster differs.')
    target_hashes = set()
    for frame in sorted(fitted):
        path = RUN / 'fit/frames' / frame / 'result.json'
        report = read(path)
        require(report['frame_id'] == frame and report['source_frame_index'] == new_rows[frame]['source_frame_index']
                and report['source_image_sha256'] == new_rows[frame]['source_sha256']
                and report['freeze_betas'] is True and report['body_placement_parameters'] == 3
                and report['shared_body_weight'] == 30. and report['shared_root_weight'] == 0.,
                f'Original body fitting contract differs: {frame}')
        pins[path] = sha(path)
        target_hashes.add(report['shared_body_target_sha256'])
    q90 = b_fit['source']['confidence']['per_joint_q90']
    eval_mismatches = []
    for label, test in (
        ('shared_betas', a_shared['betas'] == b_shared['betas']),
        ('shared_pose_latent', a_shared['pose_latent'] == b_shared['pose_latent']),
        ('shared_pose_target_sha256', target_hashes == {TARGET}),
        ('fit_only_q90', q90 == a_fit['source']['confidence']['per_joint_q90'] == archive_input['frozen_q90']),
        ('raw_observation_arrays', not obs_changes),
        ('heldout_native_row_fields', not set(row_changes) & set(HELD)),
    ):
        if not test:
            eval_mismatches.append(label)
    rows_by_frame = {record['frame']: record for record in archive_render['records']}
    diagnostics = {record['frame']: record for record in archive_fit['diagnostics']}
    require(list(rows_by_frame) == list(diagnostics) == list(HELD), 'Archive row order differs.')
    holdout_masks = {}
    for frame, record in rows_by_frame.items():
        baseline = base_rows[frame]
        require(record['source_frame_index'] == baseline['source_frame_index'] == HELD[frame]
                and record['native_source_sha256'] == baseline['source_sha256']
                and record['fit_mask_sha256'] == baseline['mask_sha256']
                and all(record[key] == diagnostics[frame][key] for key in ('pose_screen_passed', 'pose_screen_flags'))
                and all(record['posed_camera'][key] == baseline[key] for key in ('K', 'w2c', 'wh')),
                f'Original evaluation image, mask, camera, or screen differs: {frame}')
        for name, key in (('winner-camera-space.npz', 'winner_sha256'), ('result.json', 'report_sha256')):
            path = ARCHIVE / 'frames' / frame / name
            pins[path] = pin(path, diagnostics[frame][key])
        path = ARCHIVE / 'renders' / record['image']
        pins[path] = pin(path, record['image_sha256'])
        holdout_masks[frame] = dict(candidate_preprocessed_sha256=new_rows[frame]['mask_sha256'],
            archived_fitting_mask_sha256=record['fit_mask_sha256'], fixed_scoring_mask_sha256=record['scoring_mask_sha256'])
    evaluator = state = None
    if not eval_mismatches:
        evaluator = formal.load_evaluator(RUN, PROJECT, launch, HELD)
        state = evaluator.preflight()
        with np.load(ARCHIVE / 'evaluation-fitted.npz', allow_pickle=False) as poses:
            require(poses['frame_ids'].tolist() == list(HELD) and poses['source_frames'].tolist() == list(HELD.values())
                    and poses['image_sha256'].tolist() == [r['source_sha256'] for r in state.subset['rows']]
                    and poses['mask_sha256'].tolist() == [r['fit_mask_sha256'] for r in rows_by_frame.values()],
                    'Archive does not contain the exact protected pose roster.')
            for label, left, right in (
                ('terminal_shared_beta_vs_heldout', state.beta, poses['shared_betas']),
                ('heldout_adapted_118_point_targets', state.subset['targets'], poses['observed_native_keypoints']),
                ('heldout_integer_center_K', state.subset['K'], poses['K_integer_centers']),
                ('heldout_extrinsics', np.array([r['w2c'] for r in state.subset['rows']]), poses['w2c']),
            ):
                if not np.array_equal(left, right):
                    eval_mismatches.append(label)
        for row in state.old_rows:
            frame = row['frame']
            require(row['mask_sha256'] == holdout_masks[frame]['fixed_scoring_mask_sha256'],
                    f'Fixed scoring mask changed: {frame}')
        pins.update({Path(path): digest for path, digest in state.pins.items()})
        require(state.target_hash == TARGET and state.q90 == q90, 'Candidate evaluation recipe changed.')
    mask_pixel_changes = [f for f, v in masks.items() if not v['grayscale_pixels_equal']]
    strict = (not row_changes and not obs_changes and not masks and not fit_changes
              and a_selection == b_selection and a_shared['anchor_groups'] == b_shared['anchor_groups']
              and not eval_mismatches)
    receipt = dict(state='matched_pose_replay_ready' if not eval_mismatches else 'archive_pose_reuse_refused',
        strict_cold_input_parity=strict, input_manifest_differences=row_changes, raw_observation_differences=obs_changes,
        raw_observation_npz_bytes_equal=sha(a_obs) == sha(b_obs), fitted_archive_differences=fit_changes,
        native_mask_byte_differences=masks, native_mask_pixel_differences=mask_pixel_changes,
        mask_provenance=holdout_masks, evaluation_pose_mismatches=eval_mismatches, mask_boundary=boundary,
        selection_exact=a_selection == b_selection, anchor_grouping_exact=a_shared['anchor_groups'] == b_shared['anchor_groups'],
        candidate_reference=b_selection['reference_frame'], baseline_reference=a_selection['reference_frame'],
        candidate_result_sha256=sha(RUN / 'result.json'), archived_evaluation_fit_sha256=archive_fit['fitted_sha256'],
        protected_frames=HELD, primary=primary, outside_arc_diagnostic=outside, pins={str(p): h for p, h in pins.items()},
        evaluation_refits=0, gpu_calls=0, quality_accepted=False)
    return receipt, evaluator, state, rows_by_frame


def render(receipt, evaluator, state, baseline):
    require(state is not None and not receipt['evaluation_pose_mismatches'], 'Independent pose reuse refused.')
    pins = receipt['pins']
    evaluator.check_pins(pins)
    # Same unmodified renderer/scorer, with links only to the already-fitted archive.
    # The renderer checks the archive's original input receipt; this wrapper also
    # checks the complete candidate and baseline pins before and after replay.
    state.pins = read(ARCHIVE / 'inputs.json')['input_sha256']
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
        with TemporaryDirectory(prefix='.archived-pose-', dir=OUTPUT) as temporary:
            scratch = Path(temporary)
            for name in ('evaluation-fitted.npz', 'fit-result.json', 'inputs.json'):
                (scratch / name).symlink_to(ARCHIVE / name)
            rendered = evaluator.render_holdouts(state, scratch)
            for current in rendered['records']:
                old = baseline[current['frame']]
                require(current['posed_camera'] == dict(old['posed_camera'], pose_index=current['posed_camera']['pose_index'])
                        and current['scoring_mask_sha256'] == old['scoring_mask_sha256']
                        and current['native_source_sha256'] == old['native_source_sha256']
                        and current['pose_screen_flags'] == old['pose_screen_flags'],
                        'Actual render did not use the archived posed scoring camera, image or mask.')
                current['role'] = 'primary' if current['frame'] in PRIMARY else 'outside_arc_diagnostic'
                current['baseline_posed_metrics'] = old['metrics']['posed']
                current['posed_delta'] = {name: value - old['metrics']['posed'][name]
                                          for name, value in current['metrics']['posed'].items()}
                current['fixed_camera_exact'] = current['frozen_camera'] == old['frozen_camera']
            evaluator.check_pins(pins)
            (scratch / 'renders').rename(OUTPUT / 'renders')
    return dict(state='cold_005_matched_independent_holdouts_rendered', records=rendered['records'],
        parity_sha256=sha(OUTPUT / 'parity.json'), strict_cold_input_parity=receipt['strict_cold_input_parity'],
        evaluation_fit_sha256=receipt['archived_evaluation_fit_sha256'], optimizer_updates=0, evaluation_refits=0,
        quality_accepted=False, comparison_scope='Posed scores share independent saved poses, full K, RGB and masks. '
        'Out-of-arc scores are separate. Fresh mask pixels can confound timing or visual attribution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'render'))
    mode = parser.parse_args().mode
    status = read(RUN / 'status.json')
    if status['state'] != 'completed':
        print(json.dumps(dict(state='terminal_not_ready', stage=status['stage'], gpu_calls=0, evaluation_refits=0)))
        return
    if mode == 'check':
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    receipt, evaluator, state, old = preflight()
    parity = OUTPUT / 'parity.json'
    if mode == 'check':
        OUTPUT.mkdir(exist_ok=True)
        if parity.exists():
            require(read(parity) == receipt, 'Persisted parity differs. Preserve existing evidence.')
        else:
            with parity.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        print(json.dumps(dict(state=receipt['state'], receipt=str(parity), sha256=sha(parity),
                              strict=receipt['strict_cold_input_parity'],
                              mask_pixel_changes=len(receipt['native_mask_pixel_differences']),
                              mismatches=receipt['evaluation_pose_mismatches'])))
    else:
        require(read(parity) == receipt and not (OUTPUT / 'renders').exists()
                and not (OUTPUT / 'result.json').exists(), 'Run the exact CPU check before a fresh render.')
        result = render(receipt, evaluator, state, old)
        state.seq.rt.save(OUTPUT / 'result.json', result)
        print(json.dumps(dict(state=result['state'], receipt=str(OUTPUT / 'result.json'))))


if __name__ == '__main__':
    main()

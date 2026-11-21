"""Replay the saved 001 formal poses on the same-fit opacity-tail candidate."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import numpy as np

import review_006_area_formal_holdouts as paired
import review_capture_formal_holdouts as capture


BASE = paired.LAB/'shuolee-smpl-coverage-001/run'
CANDIDATE = paired.LAB/'shuolee-smpl-opacity-tail-001/run'
POSES = paired.LAB/'shuolee-smpl-coverage-001-formal'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-opacity-tail')
BASE_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-view-coverage')
LEASE = Path('/tmp/shuolee-native122-gpu.lock')
RESET_STEPS = [1800, 3600, 5400, 7200]
TRAINING = 'src/shuolee_portrait/training.py'
TRAINING_SHA = 'fe73a6f19838ed55f723df5bffe0d480a513cf3ebacf3d91b3affc15e9fd665a'
DRIVER_SHA = 'c9e8ea446cd0b358aa1d41036464249b2bd1b87085b72919e33fefe44baec4e1'
PINS = {
    Path(paired.__file__): 'cded6c28e424ba6832e4b076bcc9100a2cd776759a41d2ff5c65f9a0f32a26f6',
    Path(capture.__file__): 'b8e10f9e033b5fc253d938ded25674794ef5831b5c5ba04105185a70b17b1a6a',
    capture.COMMON: capture.COMMON_SHA,
    paired.EVALUATOR: paired.FIXED_PINS[paired.EVALUATOR],
    BASE/'result.json': 'f02291d953b2dc3df627c613c2e80a7a968330d9c41cce674015776068815e67',
    POSES/'inputs.json': '6f9e306b98c13c991ce475525089f6a3c946e946285e1eade37cc9c528f43959',
    POSES/'evaluation-fitted.npz': 'f35ec55f1d5bf9a399fe5381ef99e88823e7f760958bcd119f183afc54c8760a',
    POSES/'fit-result.json': 'e88375cb03ff084ad33fd50890b5c5afe85ffadec187f6c0413eea336b596765',
    POSES/'evaluation-scope.json': '99749a739e6ce9538b20ebb42a74fb726c095363f955564b300acd0b65a039fd',
    POSES/'renders/result.json': '7d9f2a211f85d6fe42fcf12bf12ea7350b66ed3580b99283dee150210ad7a74c',
    PROJECT/TRAINING: TRAINING_SHA,
    Path(__file__).with_name('run_same_fit_opacity_tail001.py'): DRIVER_SHA,
}
need, read, sha, check_pins = paired.need, paired.read, paired.sha, paired.check_pins


def rebound(function, **overrides):
    """Reuse the verified paired renderer while keeping its original module intact."""
    return FunctionType(function.__code__, dict(function.__globals__, **overrides), function.__name__)


def preflight(check_baseline_only):
    check_pins(PINS)
    _, _, protected, primary, outside = capture.registered_capture(BASE)
    history, helper, score_rows, old_records, unavailable, history_pins = capture.benchmark('001', protected)
    need(history == 'winterhat-automatic-sh0-253' and not unavailable and
         primary == ['frame_000033', 'frame_000049'] and outside == ['frame_000013', 'frame_000065'],
         'The original 001 evaluation roster changed.')
    scope, inputs, fit, prior = (read(POSES/path) for path in
        ('evaluation-scope.json', 'inputs.json', 'fit-result.json', 'renders/result.json'))
    need(scope['primary_within_capture_arc'] == primary and scope['outside_arc_diagnostic_frames'] == outside
         and scope['protected_frames'] == inputs['protected_frames'] == fit['protected_frames'] == prior['protected_frames'] == protected
         and scope['fitted_sha256'] == fit['fitted_sha256'] == PINS[POSES/'evaluation-fitted.npz']
         and fit['input_receipt_sha256'] == PINS[POSES/'inputs.json']
         and prior['fit_result_sha256'] == PINS[POSES/'fit-result.json']
         and fit['evaluation_only'] and fit['gaussian_optimizer_updates'] == 0,
         'The existing evaluation pose receipt is stale.')
    official = read(BASE/'fit/official-fit-result.json')
    selection = official['selection']
    shared = read(BASE/'fit/shared-body.json')
    anchors = {frame for group in shared['anchor_groups'].values() for frame in group}
    frame_paths = {path.name for path in (BASE/'fit/frames').iterdir() if path.is_dir()}
    need(selection['retained_holdouts'] == list(protected)
         and not set(protected) & (set(selection['selected_frames']) | anchors | frame_paths)
         and inputs['frozen_q90'] == official['source']['confidence']['per_joint_q90'],
         'Heldout pose data or normalization leaked into fitting.')
    beta = np.asarray(shared['betas'], dtype=np.float32)
    native = {row['frame']:row for row in read(BASE/'data/inputs.json')['records']}
    rows = [native[frame] for frame in protected]
    with np.load(POSES/'evaluation-fitted.npz', allow_pickle=False) as arrays:
        need(arrays['frame_ids'].tolist() == list(protected) and arrays['source_frames'].tolist() == list(protected.values())
             and np.array_equal(arrays['shared_betas'], beta)
             and np.array_equal(arrays['K_integer_centers'], [row['K_integer_centers'] for row in rows])
             and np.array_equal(arrays['w2c'], [row['w2c'] for row in rows])
             and arrays['image_sha256'].tolist() == [row['source_sha256'] for row in rows]
             and arrays['mask_sha256'].tolist() == [row['mask_sha256'] for row in rows], 'Frozen pose inputs differ.')
    pins = PINS | history_pins | {Path(name): digest for name, digest in inputs['input_sha256'].items()}
    baseline = {}
    for row, rendered, diagnostic in zip(rows, prior['records'], fit['diagnostics'], strict=True):
        frame = row['frame']
        need(all(row[key] == score_rows[frame][key] for key in capture.SOURCE_FIELDS)
             and rendered['frame'] == diagnostic['frame'] == frame
             and rendered['source_frame_index'] == row['source_frame_index'] == protected[frame]
             and rendered['native_source_sha256'] == row['source_sha256']
             and rendered['fit_mask_sha256'] == row['mask_sha256']
             and rendered['scoring_mask_sha256'] == score_rows[frame]['mask_sha256']
             and rendered['pose_screen_flags'] == diagnostic['pose_screen_flags'], f'Comparison identity differs: {frame}')
        pins[POSES/'frames'/frame/'result.json'] = diagnostic['report_sha256']
        pins[POSES/'frames'/frame/'winner-camera-space.npz'] = diagnostic['winner_sha256']
        baseline[frame] = rendered
    base_result = read(BASE/'result.json')
    base_snapshot = BASE/'gaussian-run/step-010000'
    base_bound = read(base_snapshot/'bound-replay.json')
    pins.update({PROJECT/name: digest for name, digest in base_bound['source_files_sha256'].items() if name != TRAINING})
    run = BASE if check_baseline_only else CANDIDATE
    result, metadata = read(run/'result.json'), read(run/'run.json')
    snapshot = run/'gaussian-run/step-010000'
    bound = read(snapshot/'bound-replay.json')
    if not check_baseline_only:
        plan, comparison = (read(CANDIDATE.parent/name) for name in ('preregistration.json', 'comparison.json'))
        need(plan['baseline'] == metadata['copied_fitting_input_from'] == str(BASE) and plan['same_fit'] and metadata['fit_reused']
             and plan['source_sha256'] == bound['source_files_sha256'][TRAINING] == TRAINING_SHA
             and plan['driver_sha256'] == DRIVER_SHA and plan['baseline_result_sha256'] == PINS[BASE/'result.json']
             and plan['baseline_bound_sha256'] == sha(base_snapshot/'bound-replay.json')
             and plan['baseline_reset_steps'] == RESET_STEPS+[9000] and plan['candidate_reset_steps'] == RESET_STEPS
             and plan['seed'] == 0 and plan['terminal'] == 10000, 'Candidate does not implement the registered isolated reset.')
        need(comparison['result_sha256'] == sha(run/'result.json') and comparison['same_fit'] and comparison['from_scratch']
             and comparison['training_camera_exact'] and comparison['through_logged_step_9000_exact']
             and len(comparison['step_1000_exact']) == 3 and all(comparison['step_1000_exact'].values()),
             'The archived baseline did not reproduce before the opacity intervention.')
        for name, digest in plan['pins'].items():
            pins[BASE/name] = digest
            if name not in ('config.json', 'run.json'):
                pins[run/name] = digest
        old, new = (read(root/'gaussian-run/config.json') for root in (BASE, run))
        need(new.pop('opacity_reset_steps') == RESET_STEPS and new == old,
             'The resolved candidate profile changed outside the reset schedule.')
        for name in ('preregistration.json', 'comparison.json'):
            path = CANDIDATE.parent/name
            pins[path] = sha(path)
    terminal = read(run/'gaussian-run/result.json')
    need(read(run/'status.json')['state'] == result['state'] == terminal['state'] == 'completed'
         and result['format_passed'] and result['steps'] == terminal['steps'] == 10000
         and result['sh_degree'] == terminal['sh_degree'] == 0
         and result['reference_frame_id'] == base_result['reference_frame_id'] == selection['reference_frame']
         and bound['frames'] == base_bound['frames'], 'Terminal named reference or bound roster differs.')
    for name in ('rotation', 'origin_world', 'scale', 'reference_frame_id'):
        need(read(run/'export/validation.json')[name] == read(BASE/'export/validation.json')[name],
             'The candidate static gauge differs from the baseline.')
    asset = Path(metadata['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    pins.update({run/'export/point_cloud.ply': result['sha256'], run/'export/scene.zip': result['scene_sha256'],
                 run/result['bound_replay']: result['bound_replay_sha256'], run/result['bound_panels']: result['bound_panels_sha256'],
                 run/'gaussian-run/training-cameras.json': sha(BASE/'gaussian-run/training-cameras.json'),
                 run/'export/cameras.json': sha(BASE/'export/cameras.json'), asset: bound['asset_sha256'],
                 snapshot/'bound-local.ply': bound['bound_ply_sha256']})
    for path in (run/'status.json', run/'result.json', run/'run.json', run/'gaussian-run/config.json',
                 run/'gaussian-run/result.json', run/'export/validation.json', snapshot/'bound-replay.json', Path(__file__)):
        pins[path] = sha(path)
    for key, path in dict(fitted=run/'fit/fitted.npz', training_rows=run/'data/training-inputs.json',
                          reference_mesh=snapshot/'reference-mesh.npz', world_ply=snapshot/'world.ply',
                          native_ply_parity=snapshot/'native-ply-parity.json').items():
        pins[path] = bound['inputs_sha256'][key]
    check_pins(pins)
    state = SimpleNamespace(run=metadata, result=result, snapshot=snapshot, asset=asset, subset={'rows':rows},
        helper=helper, old_rows=[score_rows[frame] for frame in protected], old_records=[old_records[frame] for frame in protected],
        beta=beta, pose_archive=POSES, pose_pins=inputs['input_sha256'], target_hash=fit['shared_target_sha256'],
        baseline=baseline, pins={str(path): digest for path, digest in pins.items()})
    return run, base_result, protected, primary, outside, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check-baseline', 'check', 'render'))
    parser.add_argument('--output', type=Path, default=paired.LAB/'shuolee-smpl-opacity-tail-001-formal-paired')
    args = parser.parse_args()
    out = args.output.resolve()
    need(not out.exists() and not any(out.is_relative_to(path) for path in (BASE.parent, CANDIDATE.parent, POSES, PROJECT)),
         'Use a fresh directory separate from model and pose archives.')
    run, base, protected, primary, outside, state = preflight(args.mode == 'check-baseline')
    project = BASE_PROJECT if run == BASE else PROJECT
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    need(rt.PROJECT == project, 'Wrong bound renderer worktree.')
    rebound(paired.check_cameras, AREA=run, POSE_ARCHIVE=POSES)(state)
    replay = rebound(paired.load_replay, AREA=run, HELD=protected, BASE_PLY=base['sha256'])()
    if args.mode != 'render':
        print(json.dumps(dict(state='frozen_evaluation_inputs_verified', model=str(run), primary=primary, outside=outside,
                              exact_baseline_camera_pairs=len(protected), pins=len(state.pins), evaluation_refits=0, gpu_calls=0)))
        return

    def save_receipt(path, receipt):
        receipt['candidate_model_sha256'] = receipt.pop('area_model_sha256')
        receipt['matched_baseline_model_sha256'] = receipt.pop('baseline006_model_sha256')
        receipt.update(primary_frames=primary, outside_arc_diagnostics=outside, opacity_reset_steps=RESET_STEPS,
            scope='Same saved 001 poses, native cameras, source RGB and archived 253 score masks. The candidate skips only the step 9000 reset. Outside-arc views are diagnostic.')
        for record in receipt['records']:
            record['matched_baseline_metrics'] = record.pop('baseline006_metrics')
            record['historical_253_metrics'] = record.pop('archived269_metrics')
            record['evaluation_role'] = 'primary_within_capture_arc' if record['frame'] in primary else 'outside_arc_diagnostic'
        rt.save(path, receipt)

    state.seq = SimpleNamespace(rt=SimpleNamespace(configure=rt.configure, save=save_receipt))
    with LEASE.open('rb') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        paired.no_live_training()
        out.mkdir(parents=True, exist_ok=False)
        print(replay(state, out)['state'])


if __name__ == '__main__':
    main()

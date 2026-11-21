"""Replay the original four independent 005 poses across the three reset arms."""

import argparse
from inspect import getsource
import json
import os
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
from PIL import Image

import review_005_formal_holdouts as original
import review_opacity_fork001 as fork_review
import review_opacity_fork001_holdouts as native
import run_opacity_fork005 as fork
from review_bound_alpha_ab import pin, read, require, sha


OUT = fork.OUT.with_name(fork.OUT.name + '-holdouts')
REGISTRATION = OUT.with_name(OUT.name + '-registration.json')
PRIMARY = ('frame_000027', 'frame_000042')
FAILED_POSE = ('frame_000011',)
OUTSIDE = ('frame_000055',)
ORDER = PRIMARY + FAILED_POSE + OUTSIDE
HELD = dict(zip(fork.HELD, (80, 208, 328, 433), strict=True))
AUDIT = fork.LAB / 'shuolee-scale-coverage-005/result.json'
FIXED = {
    fork.REGISTRATION: 'e584874880002e22f5b7dd82817de442f4dab506a26e36b7e4472b80965084c9',
    Path(fork.__file__).resolve(): '0bce946470ef3a3b375a8b19d5a3af01d58197ea7ae62f288d78ec97e0f35215',
    Path(native.__file__).resolve(): '10738369d1f0ecf03cd1061d0b46944581647dcdc04521bf333f3ab8505aa952',
    Path(fork_review.__file__).resolve(): '635ed0146f1e48789f44689de6986c3a8668b9d60bc4704fc00ee5f071285c78',
    Path(original.__file__).resolve(): '55cb2d9d52e3ed36522374eea0a48218a0861b364387379c04f87a5ba4c57cd3',
    fork.FORMAL / 'evaluation-fitted.npz': fork.FIXED[fork.FORMAL / 'evaluation-fitted.npz'],
    fork.FORMAL / 'renders/result.json': fork.FIXED[fork.FORMAL / 'renders/result.json'],
    fork.FORMAL / 'inputs.json': '1ed3942a80017a81e0d21c3cc215ede46267539113edbd1979e0b65117f7a6ba',
    fork.FORMAL / 'fit-result.json': '0ef086056ec1459bd82ed44c2cbf5f84b59ed020091c7063131507c456019531',
    fork.FORMAL / 'evaluation-scope.json': '2d388dbac00d104814cc85c31d42ead80c19e3ff7045850361dff6d8de55deec',
}


def frozen_inputs():
    pins = {str(path): pin(path, digest) for path, digest in FIXED.items()}
    pins[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    fit = read(fork.FORMAL / 'fit-result.json')
    scope = read(fork.FORMAL / 'evaluation-scope.json')
    inputs = read(fork.FORMAL / 'inputs.json')
    baseline = read(fork.FORMAL / 'renders/result.json')
    common = read(fork.BASE / 'fit/shared-body.json')
    official = read(fork.BASE / 'fit/official-fit-result.json')
    require(fit['state'] == 'evaluation_fit_complete' and fit['evaluation_only'] is True
            and fit['protected_frames'] == inputs['protected_frames'] == baseline['protected_frames'] == HELD
            and fit['gaussian_optimizer_updates'] == inputs['training_or_model_updates'] == baseline['gaussian_optimizer_updates'] == 0
            and fit['input_receipt_sha256'] == FIXED[fork.FORMAL / 'inputs.json']
            and baseline['fit_result_sha256'] == FIXED[fork.FORMAL / 'fit-result.json']
            and fit['fitted_sha256'] == scope['fitted_sha256'] == FIXED[fork.FORMAL / 'evaluation-fitted.npz']
            and inputs['frozen_q90'] == official['source']['confidence']['per_joint_q90']
            and scope['evaluated_frames'] == list(HELD)
            and scope['primary_within_capture_arc'] == [FAILED_POSE[0], *PRIMARY]
            and scope['outside_capture_arc'] == list(OUTSIDE),
            'Independent historical fitting contract changed.')
    for path, digest in inputs['input_sha256'].items():
        pins[path] = pin(Path(path), digest)
    launch = read(fork.BASE.parent / 'preregistration.json')
    evaluator = original.load_evaluator(fork.BASE, fork.PROJECT, launch, HELD)
    helper, score_rows, score_archive, score_pins = evaluator.old_benchmark()
    pins.update({str(path): digest for path, digest in score_pins.items()})
    with np.load(fork.FORMAL / 'evaluation-fitted.npz', allow_pickle=False) as packed:
        poses = dict(packed)
    require(poses['frame_ids'].tolist() == list(HELD) and poses['source_frames'].tolist() == list(HELD.values())
            and np.array_equal(poses['shared_betas'], np.asarray(common['betas'], dtype=np.float32)),
            'Archived independent poses do not use the original shared body.')
    native_rows = {row['frame']: row for row in read(fork.BASE / 'data/inputs.json')['records']}
    bound = read(fork.SNAPSHOT / 'bound-replay.json')
    n_train = len(bound['frames'])
    require(n_train == 23 and set(HELD).isdisjoint(frame['frame'] for frame in bound['frames']),
            'Protected sources overlap the frozen model fitting roster.')
    records = {}
    for i, (saved, score, earlier, diagnostic) in enumerate(zip(
            baseline['records'], score_rows, score_archive, fit['diagnostics'], strict=True)):
        frame = saved['frame']
        row = native_rows[frame]
        expected = saved['posed_camera']
        require(frame == list(HELD)[i] == row['frame'] == score['frame'] == diagnostic['frame']
                and saved['source_frame_index'] == row['source_frame_index'] == HELD[frame]
                and expected['pose_index'] == n_train+i
                and all(expected[key] == row[key] == score[key] for key in ('K', 'wh', 'w2c'))
                and saved['native_source_sha256'] == row['source_sha256'] == score['source_sha256'] == poses['image_sha256'][i]
                and saved['fit_mask_sha256'] == row['mask_sha256'] == poses['mask_sha256'][i]
                and saved['scoring_mask_sha256'] == score['mask_sha256']
                and np.array_equal(poses['K_integer_centers'][i], row['K_integer_centers'])
                and np.array_equal(poses['w2c'][i], row['w2c'])
                and saved['pose_screen_passed'] == diagnostic['pose_screen_passed'] == (frame in PRIMARY)
                and saved['pose_screen_flags'] == diagnostic['pose_screen_flags'],
                f'Archived independent native scoring inputs differ: {frame}')
        for name, key in (('winner-camera-space.npz', 'winner_sha256'), ('result.json', 'report_sha256')):
            path = fork.FORMAL / 'frames' / frame / name
            pins[str(path)] = pin(path, diagnostic[key])
        image_path, mask_path = Path(row['image']), helper['relocate'](score['mask'])
        pins[str(image_path)] = pin(image_path, row['source_sha256'])
        pins[str(mask_path)] = pin(mask_path, score['mask_sha256'])
        with Image.open(image_path) as source, Image.open(mask_path) as silhouette:
            require(list(source.size) == list(silhouette.size) == row['wh'], 'Native scoring raster differs.')
            image, mask = source.convert('RGB'), np.asarray(silhouette.convert('L')) >= 128
        crop = tuple(earlier['crop_xyxy'])
        x0, y0, x1, y1 = crop
        require(0 <= x0 < x1 <= row['wh'][0] and 0 <= y0 < y1 <= row['wh'][1] and mask.any(),
                f'Invalid original crop or scoring mask: {frame}')
        records[frame] = SimpleNamespace(i=i, row=row, pose=n_train+i, expected=expected,
                                         prior=saved, image=image, mask=mask, crop=crop)
    records = {frame: records[frame] for frame in ORDER}
    plan = dict(state='registered_frozen_independent_005_holdout_replay', protected_frames=HELD,
                primary_pose_screen_pass=list(PRIMARY), in_arc_pose_screen_failed=list(FAILED_POSE),
                outside_arc_diagnostic=list(OUTSIDE), fixed_scoring_masks='Exact archived 252 uint8 masks ≥128',
                evaluation_fitted_sha256=fit['fitted_sha256'],
                frames=[dict(frame=frame, native_source_sha256=row.row['source_sha256'],
                             scoring_mask_sha256=row.prior['scoring_mask_sha256'], camera=row.expected)
                        for frame, row in records.items()],
                arms=list(native.ARMS), max_native_render_calls=36, evaluation_refits=0, optimizer_updates=0,
                input_sha256=pins, quality_accepted=False)
    return plan, SimpleNamespace(helper=helper, pins=pins), poses, records


def reference_identities(baseline, audit_path):
    require(baseline == fork.BASE and audit_path == AUDIT, 'Unexpected original 005 training baseline.')
    pin(audit_path, fork.FIXED[AUDIT])
    pin(fork.SNAPSHOT / 'bound-replay.json', fork.FIXED[fork.SNAPSHOT / 'bound-replay.json'])
    audit = read(audit_path)
    identities = read(fork.SNAPSHOT / 'bound-replay.json')['frames']
    return audit, {row['frame']: (i, SimpleNamespace(identity=row)) for i, row in enumerate(identities)}


def terminal_inputs(plan, state, poses, records):
    path = fork.OUT / 'preregistration.json'
    registration = read(fork.REGISTRATION)
    require(read(path) == registration and read(fork.OUT / 'result.json')['preregistration_sha256'] == sha(path),
            'Training terminal belongs to a different registered fork.')
    source = getsource(fork_review.validate_fork)
    old = "manifest['reference_frame_id'] == 'frame_000056'"
    require(source.count(old) == 1, 'Original terminal verification gate changed.')
    source = source.replace(old, "manifest['reference_frame_id'] == 'frame_000020'")
    scope = dict(vars(fork_review), BASELINE_RESULT_SHA=fork.BASE_RESULT,
                 REFERENCE_AUDIT_SHA=fork.FIXED[AUDIT], reference_frames=reference_identities)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), scope)
    rt, asset, _, arms, evidence = scope['validate_fork'](fork.OUT, fork.PROJECT, AUDIT)
    from shuolee_portrait.training import camera

    require(all(np.array_equal(arm.fitted['shared_betas'], poses['shared_betas'])
                and set(HELD).isdisjoint(row['frame'] for row in arm.facts['frames']) for arm in arms.values()),
            'The archived evaluation identity or training protection differs.')
    require(all(camera(view.row, view.i, pose_index=view.pose)[1] == view.expected for view in records.values()),
            'Native camera replay differs from independent scoring archive.')
    evidence.update(script_sha256=sha(Path(__file__).resolve()),
                    holdout_registration_sha256=sha(REGISTRATION), evaluation_fit_sha256=plan['evaluation_fitted_sha256'])
    state.pins = dict(state.pins, **{
        str(path): sha(path), str(fork.OUT / 'result.json'): evidence['fork_result_sha256'],
        str(fork.OUT / 'shared-after-densify-step9000.pt'): evidence['checkpoint_sha256'],
    })
    return rt, asset, arms, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'check', 'render'))
    mode = parser.parse_args().mode
    require(not OUT.exists(), 'Preserve an existing heldout comparison.')
    if mode != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if mode != 'prepare' and not (fork.OUT / 'result.json').is_file():
        print(json.dumps(dict(state='terminal_not_ready', gpu_calls=0, evaluation_refits=0)))
        return
    plan, state, poses, views = frozen_inputs()
    if mode == 'prepare':
        with REGISTRATION.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(registration=str(REGISTRATION), sha256=sha(REGISTRATION), gpu_calls=0)))
        return
    require(read(REGISTRATION) == plan, 'Frozen evaluation sources differ from registration.')
    rt, asset, arms, evidence = terminal_inputs(plan, state, poses, views)
    if mode == 'check':
        print(json.dumps(dict(state='frozen_terminal_inputs_verified', arms=list(arms), gpu_calls=0,
                              evaluation_refits=0, **evidence)))
        return
    # Reuse the exact 001 rasterization/scoring implementation and panel layout.
    replay = FunctionType(native.replay.__code__, dict(vars(native), PRIMARY=ORDER))
    records, parities = replay(rt, asset, arms, state, poses, views, OUT)
    for row in records:
        frame = row['frame']
        row['role'] = ('primary_pose_pass' if frame in PRIMARY else 'in_arc_pose_failed'
                       if frame in FAILED_POSE else 'outside_arc_diagnostic')
        row['original_coverage_stock_metrics'] = views[frame].prior['metrics']['posed']
    target = OUT / 'result.json'
    rt.save(target, dict(state='matched_005_holdout_replay_complete_pending_visual_review',
        primary_records=[row for row in records if row['frame'] in PRIMARY],
        separate_diagnostic_records=[row for row in records if row['frame'] not in PRIMARY],
        records=records, load_parity=parities, terminal_models={key: arm.terminal for key, arm in arms.items()},
        optimizer_updates=0, evaluation_refits=0, quality_accepted=False, training_admission_changed=False,
        scope='Exact archived independent 005 poses, native camera and 252 scoring masks. '
              'This controlled terminal comparison measures opacity and image differences. '
              'Pose failures and out-of-arc results are separate, and this does not establish quality acceptance.',
        **evidence))
    print(json.dumps(dict(receipt=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()

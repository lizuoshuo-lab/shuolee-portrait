"""Evaluate 008 with only the source-confirmed hidden ankle/foot weights withheld."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

import review_capture_formal_holdouts as base


WRAPPER = Path(base.__file__).resolve()
WRAPPER_SHA = 'b8e10f9e033b5fc253d938ded25674794ef5831b5c5ba04105185a70b17b1a6a'
EVIDENCE = base.ARCHIVE/'indoor-native/leg-visibility179/result.json'
EVIDENCE_SHA = '42fabdee45f5a359725f8abb540da22b51f325ff53311c0fc516a7eebdbdc6f1'
ARCHIVED_INPUTS = Path('/opt/gs-slam-lab/shuolee-cross-subject-124/indoor/data/inputs.json')
GROUPS = {'left_foot': (14, 19, 20, 21), 'right_foot': (11, 22, 23, 24)}


def target_sha(targets):
    return hashlib.sha256(targets.tobytes()).hexdigest()


def setup(run, project):
    base.check_pins({WRAPPER: WRAPPER_SHA, EVIDENCE: EVIDENCE_SHA})
    launch, cfg, rows, protected, primary, outside = base.cohort(run, project)
    history = base.benchmark('008', protected)
    evidence = base.read(EVIDENCE)
    observations = run/'data/observations.npz'
    base.need(evidence['actual_visual_review'] is True and evidence['source_video_sha256'] == launch['source_sha256']
              and base.sha(observations) == evidence['observations_sha256']
              and base.sha(ARCHIVED_INPUTS) == evidence['observation_manifest_sha256'],
              'The source visibility receipt and native observations do not agree.')
    archived = {row['frame']: row for row in base.read(ARCHIVED_INPUTS)['records']}
    absent = {row['frame']: row for row in evidence['records']}
    base.need(len(rows) == len(archived) == len(absent) == len(evidence['records'])
              and all(all(row[k] == archived[row['frame']][k] for k in base.SOURCE_FIELDS)
                      and row['source_sha256'] == absent[row['frame']]['source_sha256'] for row in rows)
              and all(set(absent[frame]['absent_groups']) == set(GROUPS) for frame in primary)
              and set(outside) == set(history[4]), 'The frozen visibility evidence does not match the source roster.')
    evaluator = base.load_evaluator(run, project, launch, protected, primary, history)
    return evaluator, cfg, rows, absent, observations, evidence


def held_targets(evaluator, cfg, rows, observations):
    from shuolee_portrait import official_smplifyx as official, stable_selection as selection

    manifest = base.read(evaluator.RUN/'data/inputs.json')
    layout = cfg['capture_layout']
    scope = selection.selection_scope(dict(rows=rows, source_manifest=manifest),
                                      arc=layout['arc'], capture_layout=layout)
    base.need(scope['retained_holdouts'] == list(evaluator.HOLDOUTS)
              and set(scope['fit_frame_ids']).isdisjoint(evaluator.HOLDOUTS), 'Protected fit scope changed.')
    with np.load(observations, allow_pickle=False) as a:
        ids = a['frame_ids'].astype(str).tolist()
        base.need(len(ids) == len(rows) and set(ids) == {r['frame'] for r in rows}, 'Observation roster differs.')
        training = [ids.index(frame) for frame in scope['fit_frame_ids']]
        held = [ids.index(frame) for frame in evaluator.EVALUATED]
        _, q90 = official.relative_confidence(a['scores'][training], a['boxes'][training],
                                              a['flip_disagreement'][training])
        adapted = evaluator.with_frozen_q90(official, *(a[k][held] for k in
            ('keypoints', 'scores', 'boxes', 'flip_disagreement')), q90)
    live = {row['frame']: row for row in rows}
    for i, frame in enumerate(evaluator.EVALUATED):
        row = live[frame]
        x, y = adapted[i, :, :2].T
        w, h = row['wh']
        adapted[i, (x < -.5) | (x >= w-.5) | (y < -.5) | (y >= h-.5), 2] = 0
    return adapted, q90.tolist()


def without_hidden_feet(evaluator, targets, absent, q90):
    controlled = targets.copy()
    changed = {}
    for i, frame in enumerate(evaluator.EVALUATED):
        indices = [j for group in absent[frame]['absent_groups'] for j in GROUPS[group]]
        changed[frame] = [int(j) for j in sorted(indices) if targets[i, j, 2] > 0]
        controlled[i, indices, 2] = 0.
    base.need(all(changed.values()) and np.array_equal(controlled[..., :2], targets[..., :2]),
              'The visibility control must change only actual ankle/foot weights.')
    return controlled, dict(variant='evaluation_only_source_confirmed_hidden_feet_withheld',
        source_visibility_sha256=EVIDENCE_SHA, protected_frames=evaluator.HOLDOUTS,
        evaluated_frames=evaluator.EVALUATED, changed_confidence_slots=changed,
        original_targets_sha256=target_sha(targets), controlled_targets_sha256=target_sha(controlled),
        frozen_q90=q90, gaussian_optimizer_updates=0, quality_accepted=False)


def read_delivery(output, state, targets, pins):
    fit = base.read(output/'fit-result.json')
    review = base.read(output/'evaluation-review.json')
    renders = output/'renders/result.json'
    data = output/'evaluation-fitted.npz'
    inputs = output/'inputs.json'
    frame_ids = [row['frame'] for row in state.subset['rows']]
    base.need(base.read(inputs)['input_sha256'] == pins and base.read(inputs)['frozen_q90'] == state.q90
              and fit['state'] == 'evaluation_fit_complete' and fit['evaluation_only'] is True
              and fit['shared_target_sha256'] == state.target_hash and fit['input_receipt_sha256'] == base.sha(inputs)
              and fit['fitted_sha256'] == base.sha(data) and review['render_receipt_sha256'] == base.sha(renders),
              'Evaluation fit or render receipt differs from the current frozen delivery.')
    render = base.read(renders)
    base.need(render['state'] == 'formal_heldout_replay_complete' and render['model_changed'] is False
              and render['gaussian_optimizer_updates'] == 0 and render['fit_result_sha256'] == base.sha(output/'fit-result.json')
              and [r['frame'] for r in render['records']] == frame_ids, 'Evaluation render roster or optimizer state changed.')
    with np.load(data, allow_pickle=False) as saved:
        poses = dict(saved)
    base.need(poses['frame_ids'].tolist() == frame_ids and np.array_equal(poses['shared_betas'], state.beta)
              and np.array_equal(poses['observed_native_keypoints'], targets), 'Evaluation pose inputs changed.')
    for i, (row, old, record) in enumerate(zip(state.subset['rows'], state.old_rows, render['records'], strict=True)):
        base.need(record['native_source_sha256'] == row['source_sha256'] == poses['image_sha256'][i]
                  and record['fit_mask_sha256'] == row['mask_sha256'] == poses['mask_sha256'][i]
                  and record['scoring_mask_sha256'] == old['mask_sha256']
                  and poses['source_frames'][i] == row['source_frame_index']
                  and np.array_equal(poses['K_integer_centers'][i], row['K_integer_centers'])
                  and np.array_equal(poses['w2c'][i], row['w2c'])
                  and base.sha(output/'renders'/record['image']) == record['image_sha256'],
                  f'Evaluation RGB, mask, camera or panel differs: {row["frame"]}')
    return poses, render['records']


def compare(state, a, b, controlled):
    from shuolee_portrait.stable_selection import BODY

    records = []
    for i, row in enumerate(state.subset['rows']):
        target = controlled[i]
        indices = BODY[target[BODY, 2] > 0]
        weights = target[indices, 2].astype(np.float64)**2
        values = {}
        for label, (poses, scores) in (('A_stock', a), ('B_hidden_feet_withheld', b)):
            residual = np.linalg.norm(poses['joints_projected_native'][i, indices]-target[indices, :2], axis=1)
            values[label] = dict(common_body_detector_rms_px=float(np.sqrt(np.average(residual**2, weights=weights))),
                common_body_detector_max_px=float(residual.max()), rendered_scores=scores[i]['metrics'])
        rotation = b[0]['root_rotation'][i] @ a[0]['root_rotation'][i].T
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))
        shift = np.linalg.norm(b[0]['world_translation'][i]-a[0]['world_translation'][i])
        records.append(dict(frame=row['frame'], source_frame_index=row['source_frame_index'],
            shared_body_detector_slots=indices.tolist(), root_rotation_change_degrees=float(angle),
            root_pelvis_change_m=float(shift), feet_visibility='unobserved', **values))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check-inputs', 'fit', 'render', 'compare'))
    for name in ('run', 'project'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('a-output', 'b-output'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.mode != 'render':
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    run, project = args.run.resolve(), args.project.resolve()
    evaluator, cfg, rows, absent, obs, evidence = setup(run, project)
    targets, q90 = held_targets(evaluator, cfg, rows, obs)
    controlled, scope = without_hidden_feet(evaluator, targets, absent, q90)
    if args.mode == 'check-inputs':
        print(json.dumps(dict(state='visibility_control_inputs_verified', fit_calls=0, gpu_calls=0,
                              **{k: v for k, v in scope.items() if k != 'frozen_q90'}), ensure_ascii=False))
        return
    base.need(args.a_output is not None and args.b_output is not None, 'Supply the A and B output directories.')
    a, b = args.a_output.resolve(), args.b_output.resolve()
    base.need(not a.is_relative_to(b) and not b.is_relative_to(a)
              and all(not out.is_relative_to(run.parent) and not out.is_relative_to(project) for out in (a, b)),
              'Use disjoint evaluation outputs outside the reconstruction and its checkout.')
    state = evaluator.preflight()
    base.need(np.array_equal(state.subset['targets'], targets) and state.q90 == q90,
              'Evaluation targets or training-only q90 differ from the generic evaluator.')
    stock = read_delivery(a, state, targets, state.pins)
    state.pins.update({str(EVIDENCE): EVIDENCE_SHA, str(ARCHIVED_INPUTS): evidence['observation_manifest_sha256'],
                       str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())})
    for name in ('evaluation-scope.json', 'fit-result.json', 'evaluation-fitted.npz', 'renders/result.json'):
        state.pins[str(a/name)] = base.sha(a/name)
    state.subset['targets'] = controlled
    if args.mode == 'fit':
        fit = evaluator.fit_holdouts(state, b)
        state.seq.rt.save(b/'visibility-scope.json', dict(**scope, fitted_sha256=fit['fitted_sha256']))
        print(fit['state'])
        return
    fit = base.read(b/'fit-result.json')
    base.need(base.read(b/'visibility-scope.json') == dict(**scope, fitted_sha256=fit['fitted_sha256']),
              'The visibility-controlled fit no longer matches the registered intervention.')
    if args.mode == 'render':
        with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = evaluator.render_holdouts(state, b)
        state.seq.rt.save(b/'evaluation-review.json', dict(**scope,
            render_receipt_sha256=base.sha(b/'renders/result.json'), records=receipt['records']))
        print(receipt['state'])
        return
    intervention = read_delivery(b, state, controlled, state.pins)
    print(json.dumps(dict(state='paired_visibility_review', gaussian_optimizer_updates=0, quality_accepted=False,
        scope='One frozen model. The pair measures evaluation pose and rendered appearance sensitivity to withheld '
              'ankle/foot observations. It supplies no hidden-geometry or training evidence.',
        records=compare(state, stock, intervention, controlled)), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()

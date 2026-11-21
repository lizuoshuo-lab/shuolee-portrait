"""Evaluate the frozen visibility-aware 008 delivery on its three in-arc holdouts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

import review_008_occlusion_control as control
import review_capture_formal_holdouts as base


COMMIT = 'd080021d3c9d17a986e9fb60a93a0a8243c4a210'
CONTROL_SHA = '30acde0b0857573915ca5e9b129da76e9fd61224a4f31fdff7b460e1fa1c4e27'


def registration(run):
    parent = run.parent
    launch, camera, deployment, correction = (parent/f'{name}.json' for name in
        ('preregistration', 'camera', 'deployment', 'registration-correction'))
    pins = {str(path): base.sha(path) for path in (launch, camera, deployment, correction)}
    candidate = base.read(launch)
    base.need(candidate['candidate_commit'] == COMMIT and base.read(deployment)['runs'] == str(parent)
              and candidate['camera_config_sha256'] == pins[str(camera)],
              'Visibility-aware candidate launch identity differs.')
    changed = base.read(correction)
    expected = dict(preregistration_sha256=pins[str(launch)], old_output=candidate['output'],
                    actual_output=str(run), candidate_commit=COMMIT,
                    camera_config_sha256=pins[str(camera)], deployment_sha256=pins[str(deployment)])
    base.need(all(changed.get(k) == v for k, v in expected.items())
              and isinstance(changed.get('reason'), str) and changed['reason'].strip(),
              'Missing or invalid correction for the old preregistration output path.')
    return pins


def attach_visibility_replay(evaluator, cfg, rows, observations):
    """Adapt the archived evaluator's q90 replay at its observation boundary."""
    from shuolee_portrait import official_sequence as seq, stable_selection

    manifest_path = evaluator.RUN/'data/inputs.json'
    manifest = base.read(manifest_path)
    layout = cfg['capture_layout']
    spec = layout['visibility_annotations']
    expected = dict(path=str(control.EVIDENCE), sha256=control.EVIDENCE_SHA)
    base.need(spec == expected, 'Expected the registered 179 source visibility receipt.')
    fit_scope = stable_selection.selection_scope(dict(rows=rows, source_manifest=manifest),
        arc=layout['arc'], capture_layout=layout)
    fit_frames = set(fit_scope['fit_frame_ids'])
    base.need(fit_scope['retained_holdouts'] == list(evaluator.HOLDOUTS)
              and fit_frames.isdisjoint(evaluator.HOLDOUTS), 'A protected frame reached the training fit scope.')
    pins = {str(observations): base.sha(observations), str(manifest_path): base.sha(manifest_path)}
    absent = seq._source_visibility(spec, manifest, pins[str(observations)], pins)
    considered = fit_frames | set(evaluator.EVALUATED)
    original = evaluator.with_frozen_q90

    def with_visibility(official, points, scores, boxes, flip, q90):
        targets = original(official, points, scores, boxes, flip, q90)
        base.need(len(targets) == len(rows), 'Expected every native row in archived q90 replay order.')
        for i, row in enumerate(rows):
            if row['frame'] in considered:
                targets[i, absent[row['frame']], 2] = 0.
        return targets

    evaluator.with_frozen_q90 = with_visibility
    return fit_scope, absent, pins


def check_inputs(evaluator, cfg, rows, observations, fit_scope, absent):
    from shuolee_portrait import official_sequence as seq

    inputs = seq.load_inputs(evaluator.RUN/'data/inputs.json', observations,
        camera_length_contract=cfg['camera_length_contract'], reference_frame=rows[0]['frame'],
        capture_layout=cfg['capture_layout'])
    with np.load(observations, allow_pickle=False) as raw:
        ids = raw['frame_ids'].astype(str).tolist()
        order = [ids.index(frame) for frame in inputs['ids']]
        adapted = evaluator.with_frozen_q90(seq.official, *(raw[k][order] for k in
            ('keypoints', 'scores', 'boxes', 'flip_disagreement')), inputs['confidence']['per_joint_q90'])
    for i, row in enumerate(rows):
        x, y = adapted[i, :, :2].T
        w, h = row['wh']
        adapted[i, (x < -.5) | (x >= w-.5) | (y < -.5) | (y >= h-.5), 2] = 0
    fit = [inputs['ids'].index(frame) for frame in fit_scope['fit_frame_ids']]
    base.need(np.array_equal(adapted[fit], inputs['targets'][fit]) and
              all(not inputs['targets'][inputs['ids'].index(frame)].any() for frame in evaluator.HOLDOUTS) and
              all(not adapted[inputs['ids'].index(frame), absent[frame], 2].any() for frame in evaluator.EVALUATED),
              'Training visibility parity or evaluation isolation failed.')
    return dict(source_rows=len(rows), fit_candidates=len(fit), frozen_q90_replay_matches_training=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check-inputs', 'check', 'fit', 'render'))
    for name in ('run', 'project'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode != 'render':
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    run, project = args.run.resolve(), args.project.resolve()
    pins = registration(run)
    control_path = Path(control.__file__).resolve()
    base.check_pins({control_path: CONTROL_SHA})
    evaluator, cfg, rows, _, observations, _ = control.setup(run, project)
    base.need(cfg['capture_layout'] == base.read(run.parent/'camera.json')['capture_layout'],
              'Runtime capture layout differs from the registered camera.')
    _, historical_rows, _, _ = evaluator.old_benchmark()
    live_rows = {row['frame']: row for row in rows}
    base.need(all(all(live_rows[row['frame']][key] == row[key] for key in base.SOURCE_FIELDS)
                  for row in historical_rows), 'Historical scoring frame RGB or camera differs.')
    fit_scope, absent, source_pins = attach_visibility_replay(evaluator, cfg, rows, observations)
    pins.update(source_pins)
    pins.update({str(control_path): CONTROL_SHA, str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())})
    scope = dict(historical_baseline='indoor-automatic-sh0-254', protected_frames=evaluator.HOLDOUTS,
                 evaluated_frames=evaluator.EVALUATED,
                 unscored_outside_capture_arc=[f for f in evaluator.HOLDOUTS if f not in evaluator.EVALUATED],
                 hidden_feet_source_sha256=control.EVIDENCE_SHA,
                 excluded_detector_slots={f: sorted(absent[f]) for f in evaluator.EVALUATED},
                 scoring='Same native RGB, complete K/camera and exact archived 254 scoring masks. '
                         'Current evaluation poses are fitted with source-confirmed hidden feet withheld. '
                         'Historical and current fitting recipes differ. Hidden legs and feet remain unobserved.')
    if args.mode == 'check-inputs':
        facts = check_inputs(evaluator, cfg, rows, observations, fit_scope, absent)
        print(json.dumps(dict(state='visibility_inputs_verified', fit_calls=0, gpu_calls=0, **facts, **scope)))
        return
    state = evaluator.preflight()
    fit = base.read(run/'fit/official-fit-result.json')
    base.need(fit['source']['input_files'][str(control.EVIDENCE)] == control.EVIDENCE_SHA and
              all(not state.subset['targets'][i, absent[f], 2].any() for i, f in enumerate(evaluator.EVALUATED)),
              'Training source or heldout visibility receipt differs.')
    state.pins.update(pins)
    scope['evaluation_targets_sha256'] = control.target_sha(state.subset['targets'])
    if args.mode == 'check':
        print(json.dumps(dict(state='terminal_visibility_inputs_verified', fit_calls=0, gpu_calls=0, **scope)))
        return
    base.need(args.output is not None, 'Use a separate evaluation output directory.')
    out = args.output.resolve()
    base.need(not out.is_relative_to(run.parent) and not out.is_relative_to(project),
              'Use an output outside the reconstruction and its checkout.')
    if args.mode == 'fit':
        fitted = evaluator.fit_holdouts(state, out)
        state.seq.rt.save(out/'evaluation-scope.json', dict(**scope, fitted_sha256=fitted['fitted_sha256']))
        print(fitted['state'])
        return
    fitted = base.read(out/'fit-result.json')
    base.need(base.read(out/'evaluation-scope.json') == dict(**scope, fitted_sha256=fitted['fitted_sha256']),
              'Render request differs from the frozen visibility-aware evaluation fit.')
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rendered = evaluator.render_holdouts(state, out)
    state.seq.rt.save(out/'evaluation-review.json', dict(**scope, optimizer_updates=0, quality_accepted=False,
        render_receipt_sha256=base.sha(out/'renders/result.json'), records=rendered['records']))
    print(rendered['state'])


if __name__ == '__main__':
    main()

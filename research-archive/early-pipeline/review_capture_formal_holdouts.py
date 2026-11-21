"""Evaluate frozen bound SH0 deliveries against the archived 001 or 008 holdouts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType

sys.dont_write_bytecode = True
from review_005_formal_holdouts import ARCHIVE, EVALUATOR, EVALUATOR_SHA, METRIC_HELPER
from review_005_formal_holdouts import cohort, need, read, sha, within_arc


COMMON = Path(sys.modules[cohort.__module__].__file__).resolve()
COMMON_SHA = '55cb2d9d52e3ed36522374eea0a48218a0861b364387379c04f87a5ba4c57cd3'
BENCHMARKS = {
    '001': ('winterhat-automatic-sh0-253', (
        '1df0a7bb6ae46a9a122bf577ed55c590a95d100d212ac81e01049f8a1bee80a0',
        '31979b6e1dbedb2ae93270438c87757af6f3fe76ec0e67508c8b5077e558aaa4',
        'd661f1f6b535dbe8d75369cac4b05bdc392d9943ebe6023c1f4dd08ab068897c',
        '9a554c62ccaf550b0c169ca1acda1397d35c295b7a2989880a5e2d6fe4d48ded',
    )),
    '008': ('indoor-automatic-sh0-254', (
        'bcf02cd2683926102ca3df8ad926424a99221d093fd6bf27a0a0136c14e93c22',
        '8dc273fb08ac388cdf1ffa2cdd60d5ce8c642e5a5043c27055195765c3982618',
        'aa12f84894bf9efe2b90ee112305665f27db4d38b97eee41f3cef3546fced15f',
        '5488b7ef74f35aac82a7af0c3d06ecb2665ccfa5b57777b05a0865885d9147b9',
    )),
}
SOURCE_FIELDS = ('frame', 'source_frame_index', 'source_sha256', 'wh', 'K_integer_centers', 'K', 'w2c')


def check_pins(pins):
    for path, digest in pins.items():
        need(sha(path) == digest, f'Changed input: {path}')


def benchmark(subject, protected):
    name, digests = BENCHMARKS[subject]
    fit = ARCHIVE/(name+'-formal-heldout-fit262')
    render = ARCHIVE/(name+'-formal-render262')/'result.json'
    contacts, prereg, result = (fit/f'{stem}.json' for stem in ('contacts', 'preregistration', 'result'))
    pins = dict(zip((contacts, prereg, result, render), digests, strict=True))
    pins[METRIC_HELPER] = '72fb6738b6b6807dc0fe6d8124be41791eb31cdd3fc478a4349a139809a29f8a'
    check_pins(pins)
    prior_fit, prior_render = read(result), read(render)
    rows = {row['frame']: row for row in read(contacts)['records']}
    records = {record['frame']: record for record in prior_render['records']}
    unavailable = prior_fit.get('unavailable_frames', {})
    need(prior_fit['protected_training_holdouts'] == prior_render['protected_training_holdouts'] == list(protected)
         and list(records) == list(protected) and set(rows) == set(protected)-set(unavailable),
         'Historical roster or unavailable-frame declaration differs.')
    helper = runpy.run_path(str(METRIC_HELPER))
    for frame, row in rows.items():
        record = records[frame]
        need(record['rendered'] is True and row['source_frame_index'] == record['source_frame_index'] == protected[frame]
             and row['source_sha256'] == record['image_sha256'] and row['mask_sha256'] == record['mask_sha256']
             and row['K'] == record['fitted_camera']['K'] and row['w2c'] == record['fitted_camera']['w2c'],
             f'Historical RGB, mask or camera metadata differs: {frame}')
        pins[helper['relocate'](row['image'])] = row['source_sha256']
        pins[helper['relocate'](row['mask'])] = row['mask_sha256']
    for frame, reason in unavailable.items():
        need(records[frame]['rendered'] is False and records[frame]['reason'] == reason,
             f'Historical missing-result declarations differ: {frame}')
    check_pins(pins)
    return name, helper, rows, records, unavailable, pins


def registered_capture(run):
    launch_path, camera_path = (run.parent/f'{name}.json' for name in ('preregistration', 'camera'))
    launch, camera = read(launch_path), read(camera_path)
    layout = camera['capture_layout']
    protected = dict(zip(launch['expected_holdouts'], launch['expected_held_source_indices'], strict=True))
    need(len(protected) == len(layout['held_source_indices']) and
         list(protected.values()) == layout['held_source_indices'] and
         sha(camera_path) == launch['camera_config_sha256'] and
         layout['video_sha256'] == launch['source_sha256'] == sha(Path(launch['video'])),
         'Registered video, camera or protected indices differ.')
    intervals = layout['front_source_ranges'] + (layout['rear_source_ranges'] if layout['arc'] == 'full' else [])
    primary = [frame for frame, index in protected.items() if any(a <= index <= b for a, b in intervals)]
    outside = [frame for frame in protected if frame not in primary]
    need(primary and Path(launch['output']).resolve() == run, 'No in-arc holdout or launch directory differs.')
    return launch, camera, protected, primary, outside


def load_evaluator(run, project, launch, protected, evaluated, history):
    need(sha(EVALUATOR) == EVALUATOR_SHA, 'The archived evaluator has changed.')
    src = EVALUATOR.read_text(encoding='utf-8-sig')
    # Adapt only roster and receipt wording. Fit, rendering and scoring reuse the pinned implementation.
    replacements = (
        ('fitted == set(inputs["ids"])-set(HOLDOUTS)', 'fitted == within_arc(inputs["rows"], cfg)-set(HOLDOUTS)'),
        ('and len(fitted) == 52 and 0 < len(selected) <= 24',
         'and 0 < len(selected) <= sum(scope["view_targets"].values())'),
        ('slots = [inputs["ids"].index(frame) for frame in HOLDOUTS]',
         'slots = [inputs["ids"].index(frame) for frame in EVALUATED]'),
        ('seq._selection_inputs(inputs, list(HOLDOUTS), next(iter(HOLDOUTS)))',
         'seq._selection_inputs(inputs, EVALUATED, EVALUATED[0])'),
        ('poses["frame_ids"].tolist() == list(HOLDOUTS)', 'poses["frame_ids"].tolist() == EVALUATED'),
        ('archived269_metrics=archived["metrics"]', 'archived_baseline_metrics=archived["metrics"]'),
        ('Four frozen Gaussian holdouts;', 'Registered frozen Gaussian holdouts;'),
        ('exact archived 269 masks', 'exact archived benchmark masks'),
        ('Original 269 pixels/camera do not match:', 'Original benchmark pixels/camera do not match:'),
        ('complete 006 bound delivery', 'complete candidate bound delivery'),
        ('registered 006 training command', 'registered candidate training command'),
    )
    for old, new in replacements:
        need(src.count(old) == 1, f'Cannot match pinned evaluation interface: {old}')
        src = src.replace(old, new)
    evaluator = ModuleType('_archived_bound_evaluator_for_capture')
    evaluator.__file__ = str(EVALUATOR)
    previous = sys.path[:]
    try:
        exec(compile(src, str(EVALUATOR), 'exec'), evaluator.__dict__)
    finally:
        sys.path[:] = [str(project/'src'), *previous]
    _, helper, rows, records, _, pins = history
    pins = dict(pins)
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    pins[COMMON] = COMMON_SHA
    evaluator.__dict__.update(
        RUN=run, PROJECT=project, COMMIT=launch['candidate_commit'], VIDEO_SHA=launch['source_sha256'],
        HOLDOUTS=protected, EVALUATED=evaluated, within_arc=within_arc,
        old_benchmark=lambda: (helper, [rows[f] for f in evaluated], [records[f] for f in evaluated], pins),
    )
    return evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check-config', 'check-inputs', 'check', 'fit', 'render'))
    parser.add_argument('--subject', choices=BENCHMARKS, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--outside-arc-diagnostic', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    check_pins({COMMON: COMMON_SHA})
    if args.mode != 'render':
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    launch, camera, protected, primary, outside = registered_capture(run)
    history = benchmark(args.subject, protected)
    name, _, archived_rows, _, unavailable, _ = history
    need(set(primary) <= set(archived_rows) and set(unavailable) <= set(outside),
         'In-arc holdouts must have complete historical scoring inputs.')
    intrinsics = camera['intrinsics']
    need(intrinsics['pixel_convention'] == 'integer_centers' and
         all(row['wh'] == intrinsics['image_wh'] and row['K_integer_centers'] == intrinsics['K']
             for row in archived_rows.values()), 'Historical raster or K differs from the registered camera.')
    diagnostics = [frame for frame in outside if args.outside_arc_diagnostic and frame in archived_rows]
    evaluated = [frame for frame in protected if frame in primary or frame in diagnostics]
    scope = dict(historical_baseline=name, protected_frames=protected, evaluated_frames=evaluated,
                 primary_within_capture_arc=primary, outside_capture_arc=outside,
                 outside_arc_diagnostic_frames=diagnostics, historical_score_frames=list(archived_rows),
                 historically_unavailable_frames=unavailable,
                 score_mask_source='Exact archived baseline scoring masks',
                 comparison_scope='Comparison requires matching native RGB, camera gauge and scoring masks. Historical and current '
                                  'pose recipes, training inputs, objectives and iteration counts may differ. '
                                  'Evaluation poses use heldout 2D observations; their RGB was excluded from Gaussian training.')
    if args.mode == 'check-config':
        print(json.dumps(dict(state='configuration_and_historical_inputs_verified', live_intake_checked=False,
                              fit_calls=0, gpu_calls=0, **scope), ensure_ascii=False))
        return
    need(args.project is not None, 'A registered candidate checkout is required.')
    project = args.project.resolve()
    current_launch, cfg, rows, live_protected, live_primary, live_outside = cohort(run, project)
    need((current_launch, live_protected, live_primary, live_outside) == (launch, protected, primary, outside),
         'Current intake and registered capture roster differ.')
    live = {row['frame']: row for row in rows}
    for frame, old in archived_rows.items():
        need(all(live[frame][key] == old[key] for key in SOURCE_FIELDS), f'Historical RGB or camera differs: {frame}')
    evaluator = load_evaluator(run, project, launch, protected, evaluated, history)
    if args.mode == 'check-inputs':
        from shuolee_portrait import official_sequence as seq

        inputs = seq.load_inputs(run/'data/inputs.json', run/'data/observations.npz',
            camera_length_contract=cfg['camera_length_contract'], reference_frame=rows[0]['frame'],
            capture_layout=cfg['capture_layout'])
        need(inputs['scope']['retained_holdouts'] == list(protected) and
             set(inputs['scope']['fit_frame_ids']).isdisjoint(protected) and
             all(not inputs['targets'][inputs['ids'].index(frame)].any() for frame in protected),
             'Protected observations reached the training targets.')
        print(json.dumps(dict(state='live_inputs_verified', source_rows=len(rows),
                              fit_candidates=len(inputs['scope']['fit_frame_ids']), fit_calls=0, gpu_calls=0,
                              **scope), ensure_ascii=False))
        return
    state = evaluator.preflight()
    if args.mode == 'check':
        print(json.dumps(dict(state='terminal_inputs_verified', fit_calls=0, gpu_calls=0, **scope), ensure_ascii=False))
        return
    need(args.output is not None, 'Use a separate evaluation output directory.')
    out = args.output.resolve()
    need(not out.is_relative_to(run.parent) and not out.is_relative_to(project),
         'Use an output outside the reconstruction and its checkout.')
    if args.mode == 'fit':
        fit = evaluator.fit_holdouts(state, out)
        state.seq.rt.save(out/'evaluation-scope.json', dict(**scope, fitted_sha256=fit['fitted_sha256']))
        print(fit['state'])
        return
    fit = read(out/'fit-result.json')
    need(read(out/'evaluation-scope.json') == dict(**scope, fitted_sha256=fit['fitted_sha256']),
         'Render request differs from the frozen evaluation fit scope.')
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = evaluator.render_holdouts(state, out)
    state.seq.rt.save(out/'evaluation-review.json', dict(**scope, optimizer_updates=0, quality_accepted=False,
        render_receipt_sha256=sha(out/'renders/result.json'),
        primary_records=[r for r in receipt['records'] if r['frame'] in primary],
        outside_arc_records=[r for r in receipt['records'] if r['frame'] in diagnostics]))
    print(receipt['state'])


if __name__ == '__main__':
    main()

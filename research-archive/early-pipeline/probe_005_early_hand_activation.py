"""005 CPU control for activating the stock weak hand data one stage earlier."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import observe_005_official_fit_stages as observer
import numpy as np


ab, seq, need = observer.ab, observer.seq, observer.need
OUTPUT = ab.LAB / 'shuolee-smpl-coverage-005-early-hand-activation'
STOCK = [0., 0., 0., .1, 2.]
EARLY = [0., 0., .1, .1, 2.]
CONFIG_KEY = 'hand_joints_weights'
ARCHIVED_RESULT = '3ab534ee66dc47508861f29c4bfaddfb12fa42469cf777bd122e5485eb56ef9f'
ARCHIVED_PLAN = 'b32be769e80004ffb97b8dca86b4d8d9d5278fbecc8d028b5ffe7649e42bb77b'
OBSERVER_SHA = '230b894dd8abb6302a07ed812b43903584673292af49261c2df19adde65a323f'
REGION_LIMITS = dict(non_target_body=1., feet=1., face=.5, left_hand=1., right_hand=1.)


def activate_earlier(config):
    need(config[CONFIG_KEY] == STOCK, 'Expected the pinned stock hand data schedule.')
    candidate = dict(config, hand_joints_weights=EARLY.copy())
    need([key for key in config if config[key] != candidate[key]] == [CONFIG_KEY],
         'The sole config delta must be stage-2 hand data activation.')
    return candidate


def prepare():
    archived_root = observer.OUTPUT
    pins = {archived_root / 'result.json': ARCHIVED_RESULT, archived_root / 'preregistration.json': ARCHIVED_PLAN,
            Path(observer.__file__).resolve(): OBSERVER_SHA, Path(__file__).resolve(): seq.rt.sha(__file__)}
    seq.check_bound(pins)
    archived = seq.rt.read(archived_root / 'result.json')
    need(archived['state'] == 'stage_observation_complete' and archived['exact_stock_and_seed_parity']
         and archived['source_unchanged'] and not archived['objective_changed'],
         'Expected the completed immutable exact 005 stage observation.')
    prepared = observer.prepare()
    old_plan, pairs, checked, beta, pose, official, assets = prepared
    need(old_plan == seq.rt.read(archived_root / 'preregistration.json'),
         'Observer source plan changed after the original execution.')
    activate_earlier(seq.official.load_official_config(official))
    reference = {}
    for frame, _ in pairs:
        record = archived['records'][f'stock-zero/{frame}']
        path = archived_root / 'stock-zero' / frame / 'stage-state.npz'
        need(record['exact_archived_terminal'] and Path(record['stage_arrays']) == path,
             'The stock observer record differs from the frozen case.')
        pins[path] = record['stage_arrays_sha256']
        reference[frame] = (record['stages'], ab.arrays_at(path))
    pins.update({Path(path): digest for path, digest in old_plan['source_pins'].items()})
    seq.check_bound(pins)
    plan = dict(state='preregistered', frames=old_plan['pairs'], protected_frames=old_plan['protected_frames'],
        hand_joints_weights=dict(stock=STOCK, candidate=EARLY),
        only_change='Add the existing weak 0.1 hand-data weight in zero-based stage 2. '
                    'Every later stage, including the terminal objective, retains stock weights.',
        initialization='Stock zero latent only in both arms.',
        control='Archived exact 005 stock observer; check actual camera/stages 0–1 arrays byte-for-byte.',
        guards='Require unchanged fit provenance, shape, K, all other stage weights and full native screen. '
               'Compare source-relative elbow/wrist order, skin/limb visuals, protected region medians and maxima.',
        region_median_growth_limits_px=REGION_LIMITS, fit_calls=3, workers=3, gpu_calls=0,
        source_pins={str(path): digest for path, digest in pins.items()})
    return plan, pairs, checked, beta, pose, official, assets, reference


def early_job(item):
    load = seq.official.load_official_config
    executed = []

    def earlier(root):
        candidate = activate_earlier(load(root))
        executed.append(candidate[CONFIG_KEY])
        return candidate

    with patch.object(seq.official, 'load_official_config', new=earlier):
        result = observer.observed_job(item)
    need(executed == [EARLY], 'Expected the changed configuration to be consumed once.')
    return result


def validate_stages(stages, new_arrays, old_stages, old_arrays):
    need(new_arrays.keys() == old_arrays.keys(), 'Captured stage field names changed.')
    for key, value in new_arrays.items():
        stage = int(key.split('_')[1])
        expected = old_arrays[key]
        if key == 'stage_2_joint_weights':
            expected = expected.copy()
            expected[:, 25:67] = EARLY[2]
        invariant = stage <= 1 or key.endswith(('_K', '_betas', '_joint_weights', '_joints_confidence'))
        if invariant:
            need(ab.exact(value, expected), f'Unexpected change in recorded {key}.')
    for stage, baseline in zip(stages, old_stages, strict=True):
        need(stage['stage'] == baseline['stage'] and stage['weights'] == baseline['weights'],
             'Non-hand loss buffers or stage ordering changed.')
    for index in range(3):
        need(stages[index] == old_stages[index], 'Camera or early stage metrics lost exact parity.')


def panel(path, case, stock, candidate):
    from PIL import Image, ImageDraw
    from shuolee_portrait.bound_review_panels import _crop, _features, _mesh_panel

    row, target, K = case['row'], case['target'], case['K']
    box = _crop(row)
    with Image.open(row['image']) as source:
        rgb = source.convert('RGB').crop(box)
    parts = [rgb] + [_mesh_panel(rgb, box, pose['vertices_camera'][0], pose['faces'], K)
                     for pose in (stock, candidate)] + [
        _features(rgb, box, target, pose['joints_projected_native'][0]) for pose in (stock, candidate)]
    labels = ('Native source', 'Stock mesh', 'Early hand mesh', 'Stock 2D / fit', 'Early hand 2D / fit')
    canvas = Image.new('RGB', (5 * rgb.width, rgb.height + 22), '#181818')
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, parts, strict=True)):
        canvas.paste(image, (index * rgb.width, 22))
        draw.text((index * rgb.width + 6, 4), label, fill='white')
    canvas.save(path)
    return dict(path=str(path), sha256=seq.rt.sha(path), native_crop_xyxy=list(box),
                visual_review_completed=False)


def benchmark(prepared, output):
    plan, pairs, checked, beta, pose, official, assets, references = prepared
    jobs = [((checked[frame]['row'], output / frame, checked[frame]['target'][None],
              checked[frame]['K'], beta, True, pose), None) for frame, _ in pairs]
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'),
                             initializer=seq._init_cpu_fit, initargs=(official, assets)) as pool:
        results = list(pool.map(early_job, jobs))
    body, vposer, _ = seq.official.create_models(official_root=official, assets=assets,
                                                device='cpu', static_topology=False)
    utils, _, _, _ = seq.official.load_official(official)
    records = {}
    for (frame, _), (arrays, receipt, snapshots) in zip(pairs, results, strict=True):
        case, old = checked[frame], checked[frame]['old']
        report, archived = seq.rt.read(receipt['path']), case['report']
        seq.check_result(report, case['row'])
        changed_allowed = {'winning_loss', 'shared_body_loss_calls',
                           'last_evaluated_body_loss', 'body_placement_final_m'}
        need(report.keys() == archived.keys()
             and all(report[key] == archived[key] for key in report.keys() - changed_allowed)
             and all(ab.exact(arrays[key], old[key])
                     for key in ('betas', 'K_integer_centers', 'K_compute_float32', 'faces')),
             'Original fit identity, camera initialization, orientation or common body inputs differ.')
        path = output / frame / 'stage-state.npz'
        stages = observer.replay(snapshots, case, body.eval(), vposer, utils.GMoF, arrays, path)
        old_stages, old_arrays = references[frame]
        validate_stages(stages, ab.arrays_at(path), old_stages, old_arrays)
        initial, candidate = ab.score(case, old, archived, beta), ab.score(case, arrays, report, beta)
        medians = {name: (initial[name], candidate[name]) for name in ('non_target_body', 'feet', 'face')}
        medians.update({name: (old_stages[-1]['regions'][name], stages[-1]['regions'][name])
                        for name in ('left_hand', 'right_hand')})
        protections = {name: (before['median_px'] is None and after['median_px'] is None
                              or before['median_px'] is not None and after['median_px'] is not None
                              and after['median_px'] - before['median_px'] <= REGION_LIMITS[name])
                       for name, (before, after) in medians.items()}
        observed_order = stages[-1]['target_elbow_minus_body_wrist_native_y_px']
        fitted_order = stages[-1]['elbow_minus_body_wrist_native_y_px']
        records[frame] = dict(stock=initial, candidate=candidate,
            phase_parity_through_stage_1=True, only_executed_weight_delta='stage2 hand 0 to 0.1',
            original_gate_passed=not candidate['screen_flags'], region_protections=protections,
            terminal_source_elbow_order_matches=bool(observed_order * fitted_order > 0),
            stock_stages=old_stages, candidate_stages=stages,
            stage_arrays=str(path), stage_arrays_sha256=seq.rt.sha(path),
            panel=panel(output / (frame + '.png'), case, old, arrays))
    seq.check_bound(plan['source_pins'])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'benchmark'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == ab.LAB and not output.exists(), 'Use a new isolated 005 diagnostic output.')
    prepared = prepare()
    plan = prepared[0]
    if args.mode == 'check':
        print(json.dumps(dict(state='early_hand_inputs_verified', frames=[r['target'] for r in plan['frames']],
                              pins=len(plan['source_pins']), fit_calls=0, gpu_calls=0)))
        return
    with ab.LOCK.open('rb') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        need(available >= 8 * 1024**2 and len(os.sched_getaffinity(0)) >= 3,
             'Need 8 GiB free WSL memory and three CPU cores.')
        output.mkdir()
        seq.rt.save(output / 'preregistration.json', plan)
        start = perf_counter()
        try:
            records = benchmark(prepared, output)
            seq.rt.save(output / 'result.json', dict(state='early_hand_control_complete', records=records,
                fit_calls=3, gpu_calls=0, gaussian_updates=0, physical_depth_validated=False,
                candidate_promoted=False, formal_inputs_changed=False, quality_accepted=False,
                elapsed_seconds=perf_counter()-start))
            print('005_EARLY_HAND_CONTROL_COMPLETE', len(records), flush=True)
        except BaseException as exc:
            seq.rt.save(output / 'failure.json', dict(state='failed', error=str(exc),
                                                     elapsed_seconds=perf_counter()-start))
            raise


if __name__ == '__main__':
    main()

"""Compare stock and own-frame anchor latent starts on frozen 008 inputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import perf_counter

import probe_008_frozen_shape_swap as shape
import probe_005_nearest_admitted_latent as seed
import numpy as np


# The reused 005 worker must resolve every project module from the pinned 008 tree.
sys.path.remove(str(seed.PROJECT / 'src'))
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
seq, screen = shape.seq, shape.screen
require = shape.require
LAB, RUN = shape.LAB, shape.RUN
PROJECT = shape.PROJECTS / 'Shuolee-smpl-source-visibility'
OUTPUT = LAB / 'shuolee-smpl-visibility-008-own-anchor-latent'
SWAP = shape.OUTPUT / 'result.json'
SWAP_PLAN = SWAP.with_name('preregistration.json')
TARGET_SHA = '2da253e55a6fded5601e722098e85bae5f2f1d753941f160c629d28e184b992d'
HELD = ('frame_000015', 'frame_000043', 'frame_000071', 'frame_000098')
EXPECTED = tuple(f'frame_{i:06d}' for i in (5, 7, 10, 101, 105, 106, 107, 108))
CONTROL_PINS = (
    '93aa16b2e3e5024d1b7cca5db350a5e34bbf2251f2011145eee27b7ad6c9eb54',
    '8d117769189284bd8b5bfefb2ffe64268953906f6b850d8910f0bf019a7168d3',
)
SHOULDERS = np.array([2, 5])
WORKERS = 4


def prepare():
    here = Path(__file__).resolve().parent
    pins = {
        SWAP: '81c1eb15e903dc2aaf38e5f0e8247dd834add6bfbf1b2b725d8b33812dd05333',
        SWAP_PLAN: '0b90c3e76da066f575641e146bd2a12c2155c300dbde6cb6373635097f2ca590',
        here / 'probe_008_frozen_shape_swap.py': '1a6eb2608ea7e8e0d5bff19cabdadbe2ee8929c3df42c7db86f63149090cca2a',
        here / 'probe_005_nearest_admitted_latent.py': '656e065cf5b11f674c835e08bd845a3c8ae290fe50eedf91c82c490f6644d9bd',
        here / 'probe_local_articulation_feasibility.py': 'daaecb941f7d1191850b9a835da57e6c07488b18762d66eff66655d309a04443',
        RUN.parent / 'registration-correction.json': 'd3f8cb1f8bee2c4c910eddc23fe530a285b52cffe2b718881b575557973bf6e7',
    }
    require(seq is seed.seq and screen is seed.screen and seq.rt.PROJECT == PROJECT,
            'Expected one pinned 008 runtime in the helper and driver.')
    seq.check_bound(pins)
    parent, overlap, official, assets = shape.prepare()
    pins.update({Path(path): digest for path, digest in parent['source_pins'].items()})
    selection = seq.rt.read(RUN / 'fit/selection.json')
    cfg = seq.rt.read(RUN / 'config.json')
    fit = seq.rt.read(RUN / 'fit/official-fit-result.json')
    shared = seq.rt.read(RUN / 'fit/shared-body.json')
    launch = seq.rt.read(RUN.parent / 'preregistration.json')
    pins.update({Path(launch['video']): launch['source_sha256'],
                 RUN / 'fit/fitted.npz': fit['fit_sha256'],
                 RUN / 'data/training-inputs.json': fit['training_inputs_sha256'],
                 Path(__file__).resolve(): seq.rt.sha(__file__)})
    inputs = seq.load_inputs(RUN / 'data/inputs.json', RUN / 'data/observations.npz',
                             camera_length_contract=cfg['camera_length_contract'],
                             reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
    require(tuple(inputs['scope']['retained_holdouts']) == tuple(parent['protected_holdouts']) == HELD
            and len(inputs['ids']) == 111 and len(inputs['scope']['fit_frame_ids']) == 60
            and inputs['confidence'] == fit['source']['confidence'], 'Frozen confidence or fit scope differs.')
    result = seq.rt.read(SWAP)
    require(result['state'] == 'shape_effect_measured' and result['fit_calls'] == 0
            and result['preregistration_sha256'] == pins[SWAP_PLAN]
            and seq.rt.read(SWAP_PLAN)['source_pins'] == parent['source_pins'],
            'Expected the archived forward-only shape comparison.')
    wins = {row['frame']: row['cases'] for row in result['records']}
    pairs = {row['frame']: (row, target, K, poses, reason)
             for row, target, K, poses, _, reason in overlap}
    require(wins.keys() == pairs.keys(), 'The original overlap and its completed replay differ.')
    rejected = [frame for frame in pairs
                if not wins[frame]['anchor_pose_anchor_beta']['screen_flags']
                and wins[frame]['frozen_pose_frozen_beta']['screen_flags']]
    first = min(selection['reasons'][frame]['source_index'] for frame in rejected)
    left_range, right_range = cfg['capture_layout']['front_source_ranges']
    controls = [frame for frame in selection['selected_frames']
                if selection['reasons'][frame]['side'] == 'front'
                and not selection['reasons'][frame]['screen_flags']
                and left_range[0] <= selection['reasons'][frame]['source_index'] < first
                and (RUN / 'fit/shape-anchors' / frame / 'winner-camera-space.npz').is_file()]
    control = max(controls, key=lambda frame: (selection['reasons'][frame]['source_index'], frame))
    frames = tuple(sorted([control] + rejected))
    require(frames == EXPECTED and set(frames).isdisjoint(HELD)
            and set(frames) <= set(inputs['scope']['fit_frame_ids']),
            'Preregistered cases or holdout separation changed.')
    i = inputs['ids'].index(control)
    row, target, K = inputs['rows'][i], inputs['targets'][i], inputs['K'][i]
    control_poses = {}
    for arm, folder, digest in zip(('anchor', 'frozen'), ('shape-anchors', 'frames'), CONTROL_PINS, strict=True):
        path = RUN / 'fit' / folder / control / 'winner-camera-space.npz'
        control_poses[arm], receipt = shape.load_pose(path, row, digest, arm == 'frozen')
        pins.update({path: digest, receipt: seq.rt.sha(receipt)})
    pairs[control] = (row, target, K, control_poses, selection['reasons'][control])

    beta = np.asarray(shared['betas'], dtype=np.float32).reshape(1, 10)
    latent = np.asarray(shared['pose_latent'], dtype=np.float32).reshape(1, 32)
    entry, _ = seq.official.adapted_entry_source(official, refine_body_placement=True)
    cases = {}
    samples = []
    for frame in frames:
        row, observed, K, poses, reason = pairs[frame]
        old, anchor = poses['frozen'], poses['anchor']
        parent_dir = RUN / 'fit/frames' / frame
        report = seq.rt.read(parent_dir / 'result.json')
        anchor_report = seq.rt.read(RUN / 'fit/shape-anchors' / frame / 'result.json')
        seq.check_result(report, row)
        scores, flags, _, _ = screen._candidate(row, old, observed, K, beta, None, report, True)
        _, anchor_flags, _, _ = screen._candidate(row, anchor, observed, K, None, None, None, True)
        require(scores == reason['scores'] and flags == reason['screen_flags']
                and (not flags if frame == control else flags and not anchor_flags)
                and np.all(observed[SHOULDERS, 2] > 0)
                and anchor_report['freeze_betas'] is False and report['freeze_betas'] is True
                and report['shared_body_target_sha256'] == TARGET_SHA
                and report['shared_body_weight'] == 30. and report['shared_root_weight'] == 0.
                and report['body_placement_parameters'] == 3 and report['runtime_optimization'] is None,
                f'Archived gate, source or shared fitter contract changed for {frame}.')
        raw_seed = anchor['pose_embedding']
        seed.add_seed_hook(entry, raw_seed)
        index = row['source_frame_index']
        edge = 'early' if left_range[0] <= index <= left_range[1] else 'late'
        require(edge == 'early' or right_range[0] <= index <= right_range[1],
                'A case left the frozen front capture ranges.')
        pins.update({Path(row['image']): row['source_sha256'], Path(row['mask']): row['mask_sha256']})
        cases[frame] = dict(row=row, target=observed, K=K, old=old, report=report, seed=raw_seed)
        samples.append(dict(frame=frame, role='admitted_control' if frame == control else 'shoulder_reject',
                            edge=edge, source_index=index, source_bin=reason['source_bin'],
                            source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                            adapted_observation_sha256=sha256(observed.tobytes()).hexdigest(),
                            K_integer_centers=K.tolist(), source_w2c=row['w2c'],
                            anchor_latent_sha256=sha256(raw_seed.tobytes()).hexdigest(),
                            stock_zero_adapter_sha256=seq.rt.sha(parent_dir / 'input-output-adapter.diff')))
        pins[parent_dir / 'input-output-adapter.diff'] = samples[-1]['stock_zero_adapter_sha256']
    seq.check_bound(pins)
    plan = dict(state='preregistered', cases=samples, protected_frames=HELD, control=control,
                role_rule='All archive overlap rows passing the independent anchor native screen and failing '
                          'the shared stock screen. Control is the last earlier admitted front frame with an own archive.',
                only_change='Replace the single post-camera body-stage zero VPoser latent with this frame’s own '
                            'independent shape-anchor latent. Keep shared beta, camera init, joint targets/weights, '
                            'root branches, original five stages, placement policy, prior and optimizer.',
                output_scope='CPU diagnosis only. Alternate fit JSON is the underlying stock report schema; '
                             'the paired wrapper and adapter diff specify the actual seed. No training admission.',
                guards='First reproduce all eight zero-start archived arrays and reports byte for byte. '
                       'Original gate, smaller shoulder maximum and nonworse face RMS/max for reject cases. '
                       'Control must still pass with nonworse face RMS/max. A potential coverage gain requires '
                       'a new bin in both source ranges. All changes require later source and side visual review.',
                limits=dict(workers=WORKERS, stock_fit_calls=len(cases), alternate_fit_calls=len(cases),
                            cpu_threads_per_worker=1, gpu_calls=0, gaussian_updates=0),
                source_pins={str(path): digest for path, digest in pins.items()},
                shared_betas=beta.tolist(), shared_body_target_sha256=TARGET_SHA,
                per_joint_q90=inputs['confidence']['per_joint_q90'],
                existing_front_bins=selection['coverage']['front']['occupied_source_bins'])
    return plan, cases, beta, latent, official, assets


def run_arm(folder, cases, beta, shared_pose, official, assets, seeded):
    folder.mkdir()
    jobs = [((case['row'], folder / frame, case['target'][None], case['K'], beta, True, shared_pose),
             case['seed'] if seeded else None) for frame, case in cases.items()]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context('spawn'),
                             initializer=seq._init_cpu_fit, initargs=(official, assets)) as pool:
        results = list(pool.map(seed.fit_job, jobs))
    completed = {}
    for (frame, case), (arrays, receipt) in zip(cases.items(), results, strict=True):
        path = Path(receipt['path'])
        require(path == folder / frame / 'result.json' and seq.rt.sha(path) == receipt['sha256'],
                'Unexpected per-frame fit output.')
        report = seq.rt.read(path)
        seq.check_result(report, case['row'])
        archive, old = case['report'], case['old']
        require(all(report[key] == archive[key] for key in (
                    'official_commit', 'source_hashes', 'five_stages', 'original_optimizer_and_monitor',
                    'shared_body_target_sha256', 'shared_body_weight', 'shared_root_weight',
                    'body_placement_parameters', 'body_placement_initial_m', 'orientation_count',
                    'initialization', 'camera_initialization', 'runtime_optimization'))
                and all(seed.exact(arrays[key], old[key]) for key in
                        ('betas', 'K_integer_centers', 'K_compute_float32', 'faces')),
                f'{frame}: camera prefix, shared shape or original fitting contract changed.')
        diff = path.with_name('input-output-adapter.diff')
        observed_diff = diff.read_bytes()
        stock_diff = (RUN / 'fit/frames' / frame / diff.name).read_bytes()
        if seeded:
            delta = observed_diff[len(stock_diff):].decode().splitlines()
            added = [line[1:].strip() for line in delta if line.startswith('+') and not line.startswith('+++')]
            removed = [line[1:].strip() for line in delta if line.startswith('-') and not line.startswith('---')]
            expected = 'pose_embedding.copy_(pose_embedding.new_tensor(' + repr(case['seed'].tolist()) + '))'
            require(observed_diff.startswith(stock_diff)
                    and added == [expected] and removed == ['pose_embedding.fill_(0)'],
                    f'{frame}: alternate adapter does not prove the exact anchor latent.')
        else:
            require(observed_diff == stock_diff and report == archive and arrays.keys() == old.keys()
                    and all(seed.exact(value, old[key]) for key, value in arrays.items()),
                    f'{frame}: original zero start did not exactly replay.')
        receipt.update(adapter_diff_sha256=seq.rt.sha(diff),
                       winner_sha256=seq.rt.sha(path.with_name('winner-camera-space.npz')))
        completed[frame] = (arrays, report, receipt)
    return completed


def score(case, arrays, report, beta):
    observed = case['target']
    scores, flags, _, _ = screen._candidate(case['row'], arrays, observed, case['K'], beta,
        (case['old']['vertices_camera'].shape, case['old']['faces']), report, True)
    uv = arrays['joints_projected_native'][0]
    error = np.linalg.norm(uv - observed[:, :2], axis=1)
    joints = arrays['joints_camera'][0, SHOULDERS]
    maximum = {}
    for name, indices in (('body', screen.BODY), ('face', screen.FACE), ('feet', screen.FEET)):
        active = indices[observed[indices, 2] > 0]
        maximum[name] = float(error[active].max()) if len(active) else None
    return dict(scores=scores, screen_flags=flags, max_supported_error_px=maximum,
                shoulder_max_px=float(error[SHOULDERS].max()),
                shoulder_span_px=float(abs(np.diff(uv[SHOULDERS, 0])[0])),
                observed_shoulder_span_px=float(abs(np.diff(observed[SHOULDERS, 0])[0])),
                shoulder_center_depth_m=float(joints[:, 2].mean()),
                shoulder_distance_m=float(np.linalg.norm(joints[0] - joints[1])),
                placement_initial_m=report['body_placement_initial_m'],
                placement_final_m=report['body_placement_final_m'],
                final_original_objective=report['winning_loss'])


def benchmark(plan, cases, beta, latent, official, assets):
    import torch

    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, vposer, _ = seq.official.create_models(official_root=official, assets=assets,
                                             device='cpu', static_topology=False)
    shared_pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    require(sha256(shared_pose.tobytes()).hexdigest() == TARGET_SHA, 'Frozen shared target changed.')
    del vposer
    stock = run_arm(OUTPUT / 'stock-zero', cases, beta, shared_pose, official, assets, False)
    print('STOCK_EXACT', list(stock), flush=True)
    candidate = run_arm(OUTPUT / 'own-anchor', cases, beta, shared_pose, official, assets, True)
    records = []
    for sample in plan['cases']:
        frame = sample['frame']
        a, b = stock[frame], candidate[frame]
        before, after = score(cases[frame], *a[:2], beta), score(cases[frame], *b[:2], beta)
        face_ok = (after['scores']['face']['rms_over_height'] <= before['scores']['face']['rms_over_height']
                   and after['max_supported_error_px']['face'] <= before['max_supported_error_px']['face'])
        records.append(dict(sample, stock=before, seeded=after, face_nonworse=face_ok,
                            passes_diagnostic_numeric=(not after['screen_flags'] and face_ok
                                and (sample['role'] == 'admitted_control'
                                     or after['shoulder_max_px'] < before['shoulder_max_px'])),
                            stock_receipt=a[2], seeded_receipt=b[2],
                            panel=seed.write_panel(OUTPUT / (frame + '.png'), cases[frame], a[0], b[0])))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'benchmark'))
    args = parser.parse_args()
    require(not OUTPUT.exists(), 'Use a fresh diagnostic directory and preserve prior evidence.')
    if args.mode == 'check':
        plan, *_ = prepare()
        print(json.dumps(dict(state='pinned_inputs_and_seed_hook_verified', cases=plan['cases'],
                              protected_frames=HELD, pins=len(plan['source_pins']), fit_calls=0, gpu_calls=0)))
        return
    with shape.LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        require(available >= 8 * 1024**2 and len(os.sched_getaffinity(0)) >= WORKERS,
                'Need at least 8 GiB free WSL memory and four available CPU cores.')
        plan, *inputs = prepare()
        OUTPUT.mkdir()
        seq.rt.save(OUTPUT / 'preregistration.json', plan)
        started = perf_counter()
        try:
            records = benchmark(plan, *inputs)
            seq.check_bound(plan['source_pins'])
            control = next(row for row in records if row['role'] == 'admitted_control')
            eligible = [row for row in records if row['role'] == 'shoulder_reject'
                        and row['passes_diagnostic_numeric']]
            new_bins = {edge: sorted({row['source_bin'] for row in eligible if row['edge'] == edge}
                                      - set(plan['existing_front_bins'])) for edge in ('early', 'late')}
            seq.rt.save(OUTPUT / 'result.json', dict(state='paired_cpu_comparison_complete', stock_exact=True,
                preregistration_sha256=seq.rt.sha(OUTPUT / 'preregistration.json'),
                elapsed_seconds=perf_counter() - started, records=records,
                numeric_candidate_frames=[row['frame'] for row in eligible], numeric_new_front_bins=new_bins,
                control_numeric_pass=control['passes_diagnostic_numeric'],
                numeric_rule_supports_visual_review=control['passes_diagnostic_numeric'] and all(new_bins.values()),
                fit_calls=2 * len(records), gpu_calls=0, gaussian_updates=0, new_training_admissions=0,
                quality_accepted=False, physical_depth_validated=False, candidate_promoted=False))
            print('ANCHOR_LATENT_COMPARISON_COMPLETE', len(records), flush=True)
        except BaseException as exc:
            seq.rt.save(OUTPUT / 'failure.json', dict(state='failed', error=str(exc),
                                                       elapsed_seconds=perf_counter() - started))
            raise


if __name__ == '__main__':
    main()

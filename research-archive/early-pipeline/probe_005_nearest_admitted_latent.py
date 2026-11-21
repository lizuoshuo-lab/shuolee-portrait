"""Paired CPU-only 005 control for the nearest admitted same-half body latent."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from difflib import unified_diff
import fcntl
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import numpy as np


LAB = Path('/opt/gs-slam-lab')
RUN = LAB / 'shuolee-smpl-coverage-005/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-view-coverage')
OUTPUT = LAB / 'shuolee-smpl-coverage-005-nearest-admitted-latent'
LOCK = Path('/tmp/shuolee-native122-gpu.lock')
WITNESS = RUN.parent / 'pose-coverage-review/result.json'
TARGET_SHA = '5fdcf3949d95c96f4804ff6b7e6dcd22b56689a160e7a7a051fbd2d5ad61df49'
HELD = ('frame_000011', 'frame_000027', 'frame_000042', 'frame_000055')
EXPECTED = (('frame_000013', 'frame_000018'), ('frame_000022', 'frame_000020'),
            ('frame_000020', 'frame_000019'))
PINNED = {
    RUN / 'result.json': 'f55b70846760238b595cf1cb00803254991d98bffd6caaa730c9007d38e71946',
    RUN / 'config.json': 'ef46a20e1bacb905ef3d3a66a2749b2b88a046dfbd900ffd52a43208bc7c85bd',
    RUN / 'run.json': 'd9b693de3f5e4fe0476ef054b19ea6cc25a31e47793d00bc01d538c70d352673',
    RUN / 'fit/official-fit-result.json': '08f8e551d9c06d5aeb8593018d54c429fb913bee8158ca95d9ea9a0f9d77ab13',
    RUN / 'fit/selection.json': '56d9af357c8442c23a6b0f8cfba2d1909cc0eacd855782e7cd33445d21dd3251',
    RUN / 'fit/shared-body.json': 'd762b97063e9d4c1e4eb925f366ca50f6ee16d7da5e48427f7906249239073f8',
    RUN.parent / 'preregistration.json': '17cf148806f5186d6c9a26cd83f5617e845fb8e783f451508d9eb4d8962cfc1a',
    RUN.parent / 'camera.json': '65bc4ccfbf6d53f57cf7f47cc782ffe793d9000cab3580d7aa896c00d764fdd4',
    WITNESS: '26c48a5831edf428ed63f3320d8aa3c57ce77320a9114b70c024a1615e526139',
}
WINNERS = {
    'frame_000013': '654f853e0e4c6a78025d7b8efbe5d627441782f6b93aec94e7ce1e7cbc25ca4b',
    'frame_000018': '20cbe2de1428346d98e70763b5df7bbdd46ecbd5d2a0d595219da220c1ab17f8',
    'frame_000022': 'c7f5a183efb388a51b718845cd7a5637db7871fb6bad6eb21a5b6d39d3a0e4f3',
    'frame_000020': '2a304c4f59d31cd0217cd8d811d1a8fb01132f3a33f6f92a9047dc9432b079d3',
    'frame_000019': 'a9faa06077049890a10e8271934171db9dce73e59492ef094220554412035ac2',
}
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import official_sequence as seq, stable_selection as screen


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def arrays_at(path):
    with np.load(path, allow_pickle=False) as file:
        return dict(file)


def nearest_admitted(target, selection, layout):
    reasons = selection['reasons']
    reason = reasons[target]
    side, index = reason['side'], reason['source_index']
    interval = [bounds for bounds in layout[side + '_source_ranges'] if bounds[0] <= index <= bounds[1]]
    need(len(interval) == 1, 'Expected a single registered source interval per target.')
    start, stop = interval[0]
    donors = [frame for frame in selection['selected_frames']
              if frame != target and frame not in selection['retained_holdouts']
              and reasons[frame]['side'] == side and not reasons[frame]['screen_flags']
              and start <= reasons[frame]['source_index'] <= stop]
    need(donors, 'No admitted same-interval donor.')
    return min(donors, key=lambda frame: (abs(index - reasons[frame]['source_index']),
                                         reasons[frame]['source_index'], frame))


def add_seed_hook(source, seed):
    old = 'pose_embedding.fill_(0)'
    need(source.count(old) == 1
         and source.index('cam_init_loss_val = monitor.run_fitting(') < source.index(old)
         < source.index('for opt_idx, curr_weights in enumerate('),
         'Expected one stock zero reset between camera initialization and body stages.')
    need(seed.shape == (1, 32) and seed.dtype == np.float32 and np.isfinite(seed).all(),
         'Invalid donor body latent.')
    changed = source.replace(old, 'pose_embedding.copy_(pose_embedding.new_tensor(' + repr(seed.tolist()) + '))')
    compile(changed, 'diagnostic-nearest-admitted-latent-only', 'exec')
    return changed


def prepare():
    seq.check_bound(PINNED)
    terminal, cfg, source, fit, selection, common, launch, camera, witnesses = (
        seq.rt.read(path) for path in PINNED)
    layout = cfg['capture_layout']
    need(seq.rt.PROJECT == PROJECT and terminal['state'] == 'completed' and terminal['steps'] == 10_000
         and fit['selection'] == selection and fit['shared_identity']
         and fit['execution_kind'] == 'fresh_official_sequence'
         and fit['shared_body_sha256'] == PINNED[RUN / 'fit/shared-body.json']
         and layout == camera['capture_layout'] and layout['arc'] == 'half'
         and tuple(selection['retained_holdouts']) == tuple(launch['expected_holdouts']) == HELD,
         'Expected the original 005 cold fit and protected capture layout.')
    official = Path(cfg['official_smplifyx']['source_root'])
    assets = Path(source['assets'])
    pins = PINNED | {Path(path): digest for path, digest in fit['source']['input_files'].items()}
    pins.update({PROJECT / path: digest for path, digest in launch['code_sha256'].items()})
    pins.update({official / path: digest for path, digest in seq.official.PINNED.items()})
    pins.update({assets / row['path']: row['sha256'] for row in seq.rt.read(PROJECT / 'configs/server-assets.json')['files']
                 if row['path'].startswith(('smplx/', 'vposer/'))})
    pins.update({RUN / 'fit/fitted.npz': fit['fit_sha256'], RUN / 'data/training-inputs.json': fit['training_inputs_sha256'],
                 Path(launch['video']): launch['source_sha256'], Path(__file__).resolve(): seq.rt.sha(__file__)})
    for name in ('src/avatar_core/shuolee_vposer.py',
                 'third_party/human_body_prior/human_body_prior/models/vposer_model.py'):
        pins[PROJECT / name] = seq.rt.sha(PROJECT / name)
    seq.check_bound(pins)
    inputs = seq.load_inputs(RUN / 'data/inputs.json', RUN / 'data/observations.npz',
        camera_length_contract=cfg['camera_length_contract'], reference_frame=selection['reference_frame'],
        capture_layout=layout)
    need(inputs['confidence'] == fit['source']['confidence']
         and tuple(inputs['scope']['retained_holdouts']) == HELD
         and set(HELD).isdisjoint(inputs['scope']['fit_frame_ids'])
         and all(not inputs['targets'][inputs['ids'].index(frame)].any() for frame in HELD),
         'Protected native observations or the frozen confidence recipe differs.')
    pairs = tuple((frame, nearest_admitted(frame, selection, layout)) for frame, _ in EXPECTED)
    need(pairs == EXPECTED, 'Predeclared nearest-admitted source pairs changed.')
    raw = arrays_at(RUN / 'data/observations.npz')
    raw_ids = raw['frame_ids'].astype(str).tolist()
    beta = np.asarray(common['betas'], dtype=np.float32).reshape(1, 10)
    latent = np.asarray(common['pose_latent'], dtype=np.float32).reshape(1, 32)
    checked = {}
    evidence = {row['frame']: row for row in witnesses['frames']}
    for frame, winner_sha in WINNERS.items():
        index = inputs['ids'].index(frame)
        row, target, K = inputs['rows'][index], inputs['targets'][index], inputs['K'][index]
        path = RUN / 'fit/frames' / frame / 'winner-camera-space.npz'
        receipt = path.with_name('result.json')
        seq.check_bound({path: winner_sha})
        old, report = arrays_at(path), seq.rt.read(receipt)
        seq.check_result(report, row)
        scores, flags, _, _ = screen._candidate(row, old, target, K, beta, None, report, True)
        reason = selection['reasons'][frame]
        need(frame not in HELD and frame in inputs['scope']['fit_frame_ids']
             and scores == reason['scores'] and flags == reason['screen_flags']
             and reason['source_index'] == row['source_frame_index']
             and report['shared_body_target_sha256'] == TARGET_SHA and report['shared_body_weight'] == 30.
             and report['shared_root_weight'] == 0. and report['body_placement_parameters'] == 3,
             f'Archived source identity, original screen, or shared prior differs for {frame}.')
        if frame in ('frame_000013', 'frame_000022'):
            witness = evidence[frame]
            need(reason['status'] == 'pose_screen_rejection' and witness['body_point'] == 'R_elbow'
                 and witness['winner_sha256'] == winner_sha and witness['source_sha256'] == row['source_sha256'],
                 'Rejected case differs from its original sleeve witness.')
            pins[WITNESS.parent / (frame + '.jpg')] = witness['image_sha256']
        else:
            need(not flags and frame in selection['selected_frames'], 'Donor or control is no longer admitted.')
        pins.update({path: winner_sha, receipt: seq.rt.sha(receipt),
                     Path(row['image']): row['source_sha256'], Path(row['mask']): row['mask_sha256']})
        raw_index = raw_ids.index(frame)
        origin = {key: sha256(raw[key][raw_index].tobytes()).hexdigest()
                  for key in ('keypoints', 'scores', 'boxes', 'flip_disagreement')}
        checked[frame] = dict(row=row, target=target, K=K, old=old, report=report, reason=reason,
            source=dict(source_index=row['source_frame_index'], raw_observation_index=raw_index,
                native_source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'], raw_fields_sha256=origin,
                adapted_target_sha256=sha256(target.tobytes()).hexdigest(), K_integer_centers=K.tolist(),
                source_w2c=row['w2c'], status=reason['status'], source_bin=reason['source_bin'],
                side=reason['side'], winner_sha256=winner_sha))
    seq.check_bound(pins)
    import torch

    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, vposer, _ = seq.official.create_models(
        official_root=official, assets=assets, device='cpu', static_topology=False)
    shared_pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    need(sha256(shared_pose.tobytes()).hexdigest() == TARGET_SHA, 'Archived shared body target does not replay.')
    entry, _ = seq.official.adapted_entry_source(official, refine_body_placement=True)
    for _, donor in pairs:
        add_seed_hook(entry, checked[donor]['old']['pose_embedding'])
    plan = dict(
        state='preregistered', scope='Three paired CPU target fits with unchanged observations and original five-stage objective.',
        only_change='Replace the post-camera zero VPoser body latent with the selected donor latent. '
                    'Keep the target camera initialization, root orientation branches, beta, shared pose target, '
                    'loss weights, stages, optimizer, source resolution and point confidences.',
        donor_rule='From original admitted frames in the same registered interval and side, exclude self and heldouts. '
                   'Minimize source-index separation, break ties by earlier source index then frame identifier.',
        protected_frames=list(HELD), pairs=[
            dict(target=frame, donor=donor, target_source=checked[frame]['source'], donor_source=checked[donor]['source'],
                 target_role='admitted_control' if frame == 'frame_000020' else 'rejected_visible_elbow',
                 donor_latent_sha256=sha256(checked[donor]['old']['pose_embedding'].tobytes()).hexdigest())
            for frame, donor in pairs],
        protections='Require original supported body/feet screen and compare both native residual profiles. '
                    'Diagnostic non-target body and feet median growth limit 1 native px, face median growth limit 0.5 px. '
                    'Candidate requires independent pose visual review including unresolved monocular depth.',
        limits=dict(workers=3, threads_per_worker=1, full_stock_replays=3, seeded_fits=3, gpu_calls=0,
                    gaussian_updates=0),
        source_pins={str(path): digest for path, digest in pins.items()}, shared_body_target_sha256=TARGET_SHA)
    return plan, pairs, checked, beta, shared_pose, official, assets


def fit_job(item):
    job, seed = item
    if seed is None:
        return seq._fit_cpu_job(job)
    old_hook = seq.official.adapted_entry_source

    def seeded(root, *, refine_body_placement=False):
        source, diff = old_hook(root, refine_body_placement=refine_body_placement)
        need(refine_body_placement, 'Stock placement adapter unexpectedly disabled.')
        changed = add_seed_hook(source, seed)
        extra = ''.join(unified_diff(source.splitlines(True), changed.splitlines(True),
                                     fromfile='stock', tofile='nearest-admitted-latent'))
        return changed, diff + '\n' + extra

    seq.official.adapted_entry_source = seeded
    try:
        return seq._fit_cpu_job(job)
    finally:
        seq.official.adapted_entry_source = old_hook


def run_arm(folder, pairs, checked, beta, pose, official, assets, seeded):
    folder.mkdir()
    jobs = []
    for frame, donor in pairs:
        case = checked[frame]
        job = (case['row'], folder / frame, case['target'][None], case['K'], beta, True, pose)
        jobs.append((job, checked[donor]['old']['pose_embedding'] if seeded else None))
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'),
                             initializer=seq._init_cpu_fit, initargs=(official, assets)) as pool:
        completed = list(pool.map(fit_job, jobs))
    records = {}
    for (arrays, info), (frame, _) in zip(completed, pairs, strict=True):
        path = Path(info['path'])
        need(path == folder / frame / 'result.json' and seq.rt.sha(path) == info['sha256'],
             'Frame output receipt or directory differs.')
        case = checked[frame]
        old, archive = case['old'], case['report']
        report = seq.rt.read(path)
        seq.check_result(report, case['row'])
        need(all(report[key] == archive[key] for key in (
            'official_commit', 'source_hashes', 'five_stages', 'original_optimizer_and_monitor',
            'shared_body_target_sha256', 'shared_body_weight', 'shared_root_weight',
            'body_placement_parameters', 'body_placement_initial_m', 'orientation_count',
            'initialization', 'camera_initialization', 'runtime_optimization'))
            and all(exact(arrays[key], old[key]) for key in ('betas', 'K_integer_centers', 'K_compute_float32', 'faces')),
            f'{frame}: frozen K/shape/pose prior, stock camera start or orientation branches differ.')
        if not seeded:
            need(report == archive and arrays.keys() == old.keys() and all(
                exact(value, old[key]) for key, value in arrays.items()),
                f'{frame}: fresh stock output and report failed exact archive parity.')
        records[frame] = (arrays, report, info)
    return records


def score(case, arrays, report, beta):
    row, observed, K = case['row'], case['target'], case['K']
    original = case['old']
    scores, flags, _, _ = screen._candidate(row, arrays, observed, K, beta,
        (original['vertices_camera'].shape, original['faces']), report, True)
    residual = np.linalg.norm(arrays['joints_projected_native'][0] - observed[:, :2], axis=1)

    def summary(indices):
        active = indices[observed[indices, 2] > 0]
        return dict(points=len(active), median_px=float(np.median(residual[active])) if len(active) else None,
                    max_px=float(residual[active].max()) if len(active) else None)

    return dict(scores=scores, screen_flags=flags, elbow_px=float(residual[3]),
                non_target_body=summary(screen.BODY[screen.BODY != 3]),
                feet=summary(screen.FEET), face=summary(screen.FACE),
                right_wrist_relative_shoulder_depth_m=float(
                    arrays['joints_camera'][0, 4, 2] - arrays['joints_camera'][0, 2, 2]),
                placement_initial_m=report['body_placement_initial_m'],
                placement_final_m=report['body_placement_final_m'], optimizer_reported_loss=report['winning_loss'])


def write_panel(path, case, stock, alternate):
    from PIL import Image, ImageDraw
    from shuolee_portrait.bound_review_panels import _crop, _features, _mesh_panel

    row, observed, K = case['row'], case['target'], case['K']
    box = _crop(row)
    with Image.open(row['image']) as source:
        rgb = source.convert('RGB').crop(box)
    parts = [rgb] + [_mesh_panel(rgb, box, pose['vertices_camera'][0], pose['faces'], K)
                     for pose in (stock, alternate)] + [
        _features(rgb, box, observed, pose['joints_projected_native'][0]) for pose in (stock, alternate)]
    labels = ('Native source', 'Stock mesh', 'Seeded mesh', 'Stock 2D / fit', 'Seeded 2D / fit')
    panel = Image.new('RGB', (5 * rgb.width, rgb.height + 22), '#181818')
    draw = ImageDraw.Draw(panel)
    for i, (part, label) in enumerate(zip(parts, labels, strict=True)):
        panel.paste(part, (i * rgb.width, 22))
        draw.text((i * rgb.width + 6, 5), label, fill='white')
    panel.save(path)
    return dict(path=str(path), sha256=seq.rt.sha(path), crop_xyxy=list(box), visual_review_completed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'benchmark'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == LAB and not output.exists(), 'Use a fresh isolated local output directory.')
    plan, pairs, checked, beta, pose, official, assets = prepare()
    if args.mode == 'check':
        print(json.dumps(dict(state='pinned_inputs_and_seed_seam_verified', pairs=pairs, protected=HELD,
                              pins=len(plan['source_pins']), fit_calls=0, gpu_calls=0)))
        return
    with LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        need(available >= 8 * 1024**2 and len(os.sched_getaffinity(0)) >= 3,
             'Need at least 8 GiB free WSL memory and three available CPU cores.')
        output.mkdir()
        seq.rt.save(output / 'preregistration.json', plan)
        start = perf_counter()
        try:
            stock = run_arm(output / 'stock-zero', pairs, checked, beta, pose, official, assets, False)
            print('STOCK_EXACT', list(stock), flush=True)
            alternate = run_arm(output / 'nearest-admitted', pairs, checked, beta, pose, official, assets, True)
            results = []
            for frame, donor in pairs:
                case = checked[frame]
                a, b = stock[frame], alternate[frame]
                original, candidate = score(case, *a[:2], beta), score(case, *b[:2], beta)
                protections = {}
                for region, limit in (('non_target_body', 1.), ('feet', 1.), ('face', .5)):
                    old_median = original[region]['median_px']
                    new_median = candidate[region]['median_px']
                    protections[region] = None if old_median is None else new_median - old_median <= limit
                results.append(dict(frame=frame, donor=donor, stock=original, seeded=candidate,
                    original_screen_passed=not original['screen_flags'],
                    seeded_original_screen_passed=not candidate['screen_flags'],
                    non_target_median_protections=protections, stock_receipt=a[2], seeded_receipt=b[2],
                    panel=write_panel(output / (frame + '.png'), case, a[0], b[0])))
            seq.check_bound(plan['source_pins'])
            seq.rt.save(output / 'result.json', dict(state='paired_cpu_comparison_complete', stock_exact=True,
                preregistration_sha256=seq.rt.sha(output / 'preregistration.json'), elapsed_seconds=perf_counter() - start,
                records=results, fit_calls=6, gpu_calls=0, gaussian_updates=0, formal_inputs_changed=False,
                quality_accepted=False, candidate_promoted=False, physical_depth_validated=False))
            print('SEED_COMPARISON_COMPLETE', len(results), flush=True)
        except BaseException as exc:
            seq.rt.save(output / 'failure.json', dict(state='failed', error=str(exc), elapsed_seconds=perf_counter() - start))
            raise


if __name__ == '__main__':
    main()

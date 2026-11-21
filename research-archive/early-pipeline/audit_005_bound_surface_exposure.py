"""Measure source-only exposure of the original 005 bound surface."""

import argparse
import fcntl
from itertools import islice
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from review_bound_alpha_ab import pin, read, require, sha


LAB = Path('/opt/gs-slam-lab')
RUN = LAB / 'shuolee-smpl-coverage-005/run'
SNAPSHOT = RUN / 'gaussian-run/step-010000'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-view-coverage')
OUTPUT = LAB / 'shuolee-smpl-coverage-005-bound-exposure'
PLAN = OUTPUT / 'preregistration.json'
RESULT = OUTPUT / 'result.json'
LOCK = Path('/tmp/shuolee-native122-gpu.lock')
PINS = {
    RUN / 'result.json': 'f55b70846760238b595cf1cb00803254991d98bffd6caaa730c9007d38e71946',
    RUN / 'fit/selection.json': '56d9af357c8442c23a6b0f8cfba2d1909cc0eacd855782e7cd33445d21dd3251',
    RUN / 'gaussian-run/result.json': '81e2c3b1abd4fc8316a9f5ee10f3552f60537bf7ece575fa3b450dad1ca39392',
    RUN / 'gaussian-run/training-cameras.json': 'bae82c15ebc310615a36deda46307b0e0dd9f4bd55cc71de8d43fe34ff820d87',
    SNAPSHOT / 'bound-replay.json': '02711fbfba9e483370057d77a880ed50060e5b0abc595f56303a2383497077f3',
    PROJECT / 'src/shuolee_portrait/stable_selection.py': '7243843203365098921a85ffb1146378acb2689bc896bef7fa995b40fd7a334a',
}
HELD = ('frame_000011', 'frame_000027', 'frame_000042', 'frame_000055')
ALPHA_MIN = 253
REDISTRIBUTION_MIX = .25
MIN_GAIN = 1.10
sys.path.insert(0, str(PROJECT / 'src'))


def prepare():
    from shuolee_portrait import runtime as rt, stable_selection as selection_api
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply

    require(rt.PROJECT == PROJECT, 'Loaded another implementation.')
    pins = {path: pin(path, digest) for path, digest in PINS.items()}
    terminal = read(RUN / 'result.json')
    training = read(RUN / 'gaussian-run/result.json')
    camera_receipt = read(RUN / 'gaussian-run/training-cameras.json')
    bound = read(SNAPSHOT / 'bound-replay.json')
    selection = read(RUN / 'fit/selection.json')
    asset = Path(read(RUN / 'run.json')['assets']) / 'smplx/SMPLX_NEUTRAL.npz'
    fit, mesh, facts = _inputs(RUN, SNAPSHOT, asset, bound)
    bound_ply = SNAPSHOT / 'bound-local.ply'
    pins[bound_ply] = pin(bound_ply, bound['bound_ply_sha256'])
    pins[asset] = pin(asset, bound['asset_sha256'])
    pins.update({PROJECT / path: digest for path, digest in bound['source_files_sha256'].items()})
    files = dict(fitted=RUN / 'fit/fitted.npz', training_rows=RUN / 'data/training-inputs.json',
                 reference_mesh=SNAPSHOT / 'reference-mesh.npz', world_ply=SNAPSHOT / 'world.ply',
                 native_ply_parity=SNAPSHOT / 'native-ply-parity.json')
    pins.update({path: facts['inputs_sha256'][key] for key, path in files.items()})
    pins[Path(__file__).resolve()] = sha(Path(__file__).resolve())
    rows = read(files['training_rows'])['records']
    frames = [row['frame'] for row in rows]
    cameras = camera_receipt['records']
    pose_indices = training['pose_layout']['pose_indices']
    require(terminal['state'] == training['state'] == 'completed'
            and training['steps'] == terminal['steps'] == 10000
            and training['points'] == terminal['points'] == bound['points'] == 87792
            and bound['optimizer_updates'] == 0
            and terminal['sh_degree'] == training['sh_degree'] == bound['sh_degree'] == 0
            and facts['reference_frame_id'] == 'frame_000020'
            and frames == training['pose_layout']['frame_ids'] == fit['frame_ids'].tolist()
            and frames == [record['frame'] for record in cameras] == [record['frame'] for record in facts['frames']]
            and pose_indices == [record['pose_index'] for record in cameras]
            and set(frames) == set(selection['selected_frames']) and len(frames) == 23
            and tuple(selection['retained_holdouts']) == HELD and not set(frames) & set(HELD)
            and camera_receipt['source_sha256'] == facts['inputs_sha256']['training_rows']
            and camera_receipt['interpolated_cameras'] == 0,
            'The original training delivery, camera order or heldout protection changed.')
    for row, camera in zip(rows, cameras, strict=True):
        require(row['wh'] == [2704, 1520] and camera['pixel_convention'] == 'pixel_corners'
                and all(row[key] == camera[key] for key in ('wh', 'K', 'w2c')),
                f'Native camera mismatch: {row["frame"]}')
        for name, digest in (('image', 'source_sha256'), ('mask', 'mask_sha256')):
            pins[Path(row[name])] = pin(Path(row[name]), row[digest])
    bin_ids, order = selection_api.source_bin_training_order(rows, selection, seed=0)
    schedule = np.fromiter(islice(order, training['steps']), dtype=np.int64)
    draws = np.asarray(training['per_frame_updates'], dtype=np.int64)
    require(np.array_equal(np.bincount(schedule, minlength=len(rows)), draws)
            and len(set(bin_ids)) == 13 and int(draws.sum()) == 10000
            and {str(b): int(draws[np.asarray(bin_ids) == b].sum()) for b in set(bin_ids)}
                == training['source_bin_draws'],
            'The original deterministic schedule does not replay its saved draw counts.')
    _, binding_counts = inspect_bound_ply(bound_ply, len(fit['faces']))
    require(fit['faces'].shape == (20908, 3) and int(binding_counts.sum()) == training['points'],
            'Terminal face topology or bound counts differ.')
    plan = dict(state='registered_source_only_exposure_audit', original_run=str(RUN),
                frames=[dict(frame=row['frame'], source_index=row['source_frame_index'],
                             bin=bin_id, draw_count=int(count), camera=cam)
                        for row, bin_id, count, cam in zip(rows, bin_ids, draws, cameras, strict=True)],
                excluded_heldouts=list(HELD), full_raster_wh=[2704, 1520], native_pixels=2704 * 1520 * len(rows),
                face_count=len(fit['faces']), confidence_mask_min_uint8=ALPHA_MIN,
                stock_draw_count=int(draws.sum()), final_draw_frame=frames[schedule[-1]],
                scope='Original terminal shared-offset surface with exact archived named poses and native source cameras. '
                      'One depth-tested triangle-ID raster per selected view. This is a terminal geometric exposure proxy, '
                      'not an audit of material identity, Gaussian support or historical visibility at each optimizer step.',
                count_definition='Record visible native mask-interior pixels per triangle and view, '
                                 'distinct occupied source bins, and exact saved raster draw counts over the resulting binary visibility.',
                only_candidate_rule='Within each existing source bin, blend 75% stock frame draw counts and '
                    '25% allocation proportional to the sum of reference area divided by stock exposure over visible, '
                    'bound, fully clothing-tagged faces observed in at least two bins. Use largest remainder with '
                    'chronological slot ties, keeping the exact bin total and at least one draw for each original view. '
                    'Keep original counts when the entire bin has zero score.',
                go_rule='Eligible clothing area-weighted exposure p10 must rise at least 10%. Positive-stock '
                        'protected head and hand area-weighted p10 and integrated exposure must not fall. '
                        'This only identifies whether a later sampling A/B merits registration.',
                gpu_rasters=25, evaluation_refits=0, optimizer_updates=0,
                input_sha256={str(path): digest for path, digest in pins.items()})
    return plan, rt, asset, rows, cameras, fit, mesh, facts, draws, np.asarray(bin_ids), binding_counts


def raster_ids(ctx, rasterize, torch, points, faces, K, E, wh):
    w, h = wh
    K = torch.as_tensor(K, dtype=torch.float32, device=points.device)
    E = torch.as_tensor(E, dtype=torch.float32, device=points.device)
    camera = points @ E[:3, :3].T + E[:3, 3]
    x, y, z = camera.unbind(-1)
    # Native full-K rectified pixels. Flip the standard GL raster back to
    # source-image y-down. K is already in the integer-edge convention.
    clip = torch.stack((2 * K[0, 0] / w * x + (2 * K[0, 2] / w - 1) * z,
                        -2 * K[1, 1] / h * y + (1 - 2 * K[1, 2] / h) * z,
                        1.00002 * z - .0200002, z), -1)
    raster, _ = rasterize(ctx, clip[None].contiguous(), faces, (h, w), grad_db=False)
    return raster[0, :, :, 3].flip(0)


def fixture(ctx, dr, torch, camera):
    wh = [64, 48]
    K = np.asarray([[40., 0., 22.75], [0., 41., 19.25], [0., 0., 1.]])
    E = np.eye(4)
    uv = np.array([[5, 6], [21, 6], [5, 22]], np.float64)
    rays = np.c_[(uv + .5 - K[:2, 2]) / K.diagonal()[:2], np.ones(3)]
    points = torch.as_tensor(np.r_[2 * rays, .8 * rays], dtype=torch.float32, device='cuda')
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device='cuda')
    row = dict(frame='fixture', wh=wh, K=K.tolist(), w2c=E.tolist())
    stock, camera_receipt = camera(row, 0)
    xyz = points.cpu().numpy()
    clip = np.c_[xyz, np.ones(len(xyz))] @ stock.full_proj_transform.numpy()
    xy = ((clip[:, :2] / clip[:, 3:] + 1) * wh - 1) / 2
    error = float(abs(xy - np.tile(uv, (2, 1))).max())
    far = raster_ids(ctx, dr.rasterize, torch, points, faces[:1].contiguous(), K, E, wh)
    both = raster_ids(ctx, dr.rasterize, torch, points, faces, K, E, wh)
    require(error < .001 and far[8, 8].item() == 1 and both[8, 8].item() == 2
            and both[h_mirror := wh[1] - 1 - 8, 8].item() == 0,
            'Native off-centre projection, y direction or frontmost triangle fixture failed.')
    return dict(stock_projection_error_px=error, camera_receipt=camera_receipt,
                far_only_id=1, overlapping_near_id=2, mirrored_background_row=h_mirror)


def weighted_p10(values, area):
    order = np.argsort(values, kind='stable')
    weights = np.cumsum(area[order])
    return int(values[order[np.searchsorted(weights, .1 * weights[-1])]])


def exposure_readout(pixels, draws, bins, binding, area, clothing, head, hands):
    visible = pixels > 0
    observed_bins = np.stack([visible[bins == group].any(axis=0) for group in sorted(set(bins))]).sum(axis=0)
    stock = draws @ visible
    eligible = (observed_bins >= 2) & (binding > 0) & clothing & (area > 0)
    alternative = draws.copy()
    if eligible.any():
        score = visible[:, eligible] @ (area[eligible] / stock[eligible])
        for group in sorted(set(bins)):
            idx = np.flatnonzero(bins == group)
            total = draws[idx].sum()
            if score[idx].sum() == 0:
                continue
            proposal = (1 - REDISTRIBUTION_MIX) * draws[idx] + REDISTRIBUTION_MIX * total * score[idx] / score[idx].sum()
            whole = np.floor(proposal).astype(np.int64)
            order = np.argsort(whole - proposal, kind='stable')
            whole[order[:total - whole.sum()]] += 1
            require((whole > 0).all() and whole.sum() == total, 'Changed roster or per-bin budget.')
            alternative[idx] = whole
    expected = alternative @ visible

    def stats(region):
        active = region & (area > 0) & (stock > 0)
        if not active.any():
            return dict(faces=0, stock_p10=None, proposed_p10=None, stock_area_draws=None, proposed_area_draws=None)
        return dict(faces=int(active.sum()),
                    stock_p10=weighted_p10(stock[active], area[active]),
                    proposed_p10=weighted_p10(expected[active], area[active]),
                    stock_area_draws=float(area[active] @ stock[active]),
                    proposed_area_draws=float(area[active] @ expected[active]))

    regions = dict(eligible_clothing=stats(eligible), protected_head=stats(head), protected_hands=stats(hands))
    c, h, a = regions.values()
    protect = all(r['faces'] and r['proposed_p10'] >= r['stock_p10']
                  and r['proposed_area_draws'] >= r['stock_area_draws'] for r in (h, a))
    go = bool(c['faces'] and c['proposed_p10'] >= MIN_GAIN * c['stock_p10'] and protect)
    return dict(stock_draws=draws.tolist(), proposed_draws=alternative.tolist(), regions=regions,
                bound_faces_by_visible_bin_count=np.bincount(observed_bins[binding > 0]).tolist(),
                unobserved_bound_faces=int(((observed_bins == 0) & (binding > 0)).sum()),
                protection_nonworse=protect, sampling_ab_merits_registration=go), stock, expected, observed_bins


def audit(plan, rt, asset, rows, cameras, fit, mesh, facts, draws, bins, binding):
    import nvdiffrast.torch as dr
    import torch
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import _frozen_smpl_surface, camera

    rt.configure(asset.parent.parent)
    require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    with gpu_lock(), torch.no_grad():
        base, linear, clothing, protected, base_error = _frozen_smpl_surface(fit, asset)
        require(not np.any(mesh['static_offset'][~clothing]), 'Saved offset crosses the clothing boundary.')
        faces = fit['faces']
        triangles = torch.as_tensor(faces, dtype=torch.int32, device='cuda').contiguous()
        ctx = dr.RasterizeCudaContext()
        projection = fixture(ctx, dr, torch, camera)
        delta = torch.from_numpy(mesh['static_offset']).cuda()[..., None]
        slot = facts['reference_pose_index']
        replay = base[slot].cuda() + (linear[slot].cuda() @ delta).squeeze(-1)
        ref_error = float((replay - torch.as_tensor(mesh['vertices_world'], device='cuda')).abs().max())
        require(ref_error < 2e-6, 'Terminal named mesh did not replay.')
        pixels, records = [], []
        for i, (row, expected) in enumerate(zip(rows, cameras, strict=True)):
            slot = expected['pose_index']
            _, camera_receipt = camera(row, i, pose_index=slot)
            require(camera_receipt == expected, f'Camera replay differs: {row["frame"]}')
            points = base[slot].cuda() + (linear[slot].cuda() @ delta).squeeze(-1)
            ids = raster_ids(ctx, dr.rasterize, torch, points, triangles, row['K'], row['w2c'], row['wh'])
            with Image.open(row['mask']) as source:
                require(list(source.size) == row['wh'], 'Native mask dimensions differ.')
                mask = torch.from_numpy(np.asarray(source.convert('L')).copy()).cuda() >= ALPHA_MIN
            support = torch.bincount(ids[mask].long(), minlength=len(faces) + 1).cpu().numpy()
            require(len(support) == len(faces) + 1 and support[1:].sum() > 0, 'Invalid native triangle support.')
            pixels.append(support[1:])
            records.append(dict(frame=row['frame'], native_mask_positive_pixels=int(support.sum()),
                                unmatched_mask_positive_pixels=int(support[0]),
                                supported_triangle_count=int(np.count_nonzero(support[1:]))))
    pixels = np.stack(pixels)
    v = mesh['vertices_world'][faces].astype(np.float64)
    area = .5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    with np.load(asset, allow_pickle=False) as source:
        dominant = source['weights'].argmax(axis=1)
    hands = np.isin(dominant, [20, 21, *range(25, 55)])[faces].any(axis=1)
    readout, stock, expected, n_bins = exposure_readout(pixels, draws, bins, binding, area,
        clothing[faces].all(axis=1), protected[faces].any(axis=1), hands)
    for path, digest in plan['input_sha256'].items():
        pin(Path(path), digest)
    with (OUTPUT / 'per-face-exposure.npz').open('xb') as file:
        np.savez_compressed(file, native_mask_pixels=pixels, frames=np.asarray([r['frame'] for r in rows]),
                            stock_exposure=stock, proposed_exposure=expected, visible_source_bins=n_bins,
                            bound_gaussian_counts=binding, area_m2=area)
    return dict(state='source_only_exposure_measured', observations=records, redistribution=readout,
                projection_fixture=projection, reference_replay_error_m=ref_error, base_replay_error_m=base_error,
                per_face_exposure_sha256=sha(OUTPUT / 'per-face-exposure.npz'),
                preregistration_sha256=sha(PLAN), optimizer_updates=0, evaluation_refits=0,
                new_training_admissions=0, quality_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'audit'))
    mode = parser.parse_args().mode
    if mode == 'register':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan, rt, *state = prepare()
    if mode == 'register':
        OUTPUT.mkdir(exist_ok=False)
        rt.save(PLAN, plan)
        print(json.dumps(dict(receipt=str(PLAN), sha256=sha(PLAN), gpu_calls=0)))
        return
    require(read(PLAN) == plan and not any((OUTPUT / name).exists() for name in
            ('result.json', 'per-face-exposure.npz', 'failure.json')),
            'No fresh audit or current inputs differ from preregistration.')
    with LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = perf_counter()
        try:
            report = audit(plan, rt, *state)
            rt.save(RESULT, dict(report, elapsed_seconds=perf_counter() - started))
            print(json.dumps(dict(receipt=str(RESULT), sha256=sha(RESULT))))
        except BaseException as exc:
            rt.save(OUTPUT / 'failure.json', dict(state='failed', error=str(exc)))
            raise


if __name__ == '__main__':
    main()

"""Measure the frozen 016 f051 ear polygon in the archived native 1x PNGs."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


ROUND4 = Path(__file__).resolve().parents[1] / 'outputs/smpl-delivery-round4'
EAR_SHA = '4fdc21d143f464c0e2505d42d159f650f017698cb5d70520e329d0e59db4b65b'
RECEIPT_SHA = '9526e9cf8cfe01a662c7e0b9dcd1243537a1114278adca298c154c71e1a4ba99'
FRAME = 'frame_000051'
OLD_ROW = 'round4_ac2cfe6'
COLUMNS = ['source_rgb', 'black', 'white', 'alpha']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path, expected):
    path = Path(path)
    require(sha(path) == expected, f'Changed artifact: {path}')
    return path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def choose(items, **fields):
    matches = [item for item in items if all(item.get(k) == v for k, v in fields.items())]
    require(len(matches) == 1, f'Expected one registered record: {fields}')
    return matches[0]


def panel(path, receipt, source, crop, rows):
    item = choose(receipt['panels'], view='side_head', scale=1.0, frame=FRAME)
    require(item['header_pixels'] == 0 and item['crop_xyxy'] == crop and item['columns'] == COLUMNS
            and item['rows'] == rows and Path(item['file']).name == item['file'], 'Native side-head panel layout changed.')
    path = pin(path.parent / item['file'], item['sha256'])
    with Image.open(path) as image:
        require(image.format == 'PNG', 'Use the lossless review PNG.')
        pixels = np.asarray(image.convert('RGB'))
    h, w = source.shape[:2]
    require(pixels.shape == (h * len(rows), w * 4, 3), 'Native panel dimensions differ.')
    strips = list(np.split(pixels, len(rows), axis=0))
    require(all(np.array_equal(strip[:, :w], source) for strip in strips), 'A source crop differs from native RGB bytes.')
    return path, strips


def camera_record(receipt, roi_audit):
    record = choose(receipt['records'], role='side', frame=FRAME)
    identity = roi_audit['identity']
    camera = record['native_camera']
    frozen = identity['training_camera']
    require(record['source_frame_index'] == roi_audit['scope']['source_frame_index']
            and record['source_sha256'] == identity['source_rgb']['sha256']
            and record['score_mask_sha256'] == record['own_mask_sha256'] == identity['score_mask']['sha256']
            and camera['K'] == frozen['K_pixel_corners'] and camera['w2c'] == frozen['w2c']
            and camera['wh'] == frozen['wh'] and camera['pose_index'] == roi_audit['scope']['training_slot']
            and camera['pixel_convention'] == 'pixel_corners', 'Source, matte, pose slot or native camera changed.')
    return record


def posed_mesh(run_result_path, audit, native):
    run = read(run_result_path)
    replay_path = pin(run_result_path.parent / run['bound_replay'], run['bound_replay_sha256'])
    replay = read(replay_path)
    mesh_path = pin(replay_path.parent / replay['mesh_file'], replay['mesh_sha256'])
    rec = choose(replay['records'], frame=FRAME, pose_index=audit['scope']['training_slot'])
    require(replay['state'] == 'completed' and replay['optimizer_updates'] == 0
            and rec['camera'] == native['native_camera'] and rec['source_frame_index'] == native['source_frame_index']
            and rec['source_sha256'] == native['source_sha256'] and rec['mask_sha256'] == native['score_mask_sha256'],
            'Posed replay source or camera differs.')
    with np.load(mesh_path, allow_pickle=False) as arrays:
        mesh = dict(arrays)
    slot = rec['pose_index']
    require(mesh['frame_ids'][slot] == FRAME and mesh['pose_indices'][slot] == slot
            and mesh['source_frames'][slot] == rec['source_frame_index'] and mesh['image_sha256'][slot] == rec['source_sha256']
            and mesh['mask_sha256'][slot] == rec['mask_sha256'], 'Posed mesh row differs.')
    return mesh_path, mesh


def mesh_coverage(vertices, faces, camera, xy):
    E, K = np.asarray(camera['w2c']), np.asarray(camera['K'])
    xyz = vertices @ E[:3, :3].T + E[:3, 3]
    q = xyz @ K.T
    triangles = (q[:, :2] / q[:, 2:] - .5)[faces]
    include = ((triangles.min(1) <= xy.max(0)) & (triangles.max(1) >= xy.min(0))).all(1)
    ids = np.flatnonzero(include & (xyz[faces, 2] > .2).all(1))
    require(len(ids) > 0, 'No triangles near the registered ear crop.')
    a, b, c = triangles[ids].transpose(1, 0, 2)
    p, u, v = xy[:, None] - a, b - a, c - a
    den = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        l1 = (p[..., 0] * v[:, 1] - p[..., 1] * v[:, 0]) / den
        l2 = (u[:, 0] * p[..., 1] - u[:, 1] * p[..., 0]) / den
        bary = np.stack((1-l1-l2, l1, l2), axis=-1) / xyz[faces[ids], 2]
        depth = np.where((bary >= -1e-8).all(-1), 1 / bary.sum(-1), np.inf)
    hit = np.isfinite(depth).any(1)
    return np.where(hit, ids[depth.argmin(1)], -1)


def metrics(strip, source, local_xy, on_mesh, review_path, native):
    entry = native['closeup']['panel']
    require(Path(entry['file']).name == entry['file'] and entry['header_pixels'] == 0,
            'The direct native crop layout differs.')
    direct_path = pin(review_path.parent / entry['file'], entry['sha256'])
    with Image.open(direct_path) as image:
        direct = np.asarray(image.convert('RGB'))
    h, w, channels = strip.shape
    require(direct.shape == (2*h, w, channels) and np.array_equal(direct[:h], strip),
            'The comparison row differs from the direct native 1x render panel.')
    float_error, _ = native['native_composition_max_abs']
    require(np.isfinite(float_error) and 0 <= float_error < 5e-5, 'The float renderer composition check failed.')
    _, black, white, alpha_rgb = np.split(strip, 4, axis=1)
    require(np.array_equal(alpha_rgb[..., 0], alpha_rgb[..., 1]) and np.array_equal(alpha_rgb[..., 0], alpha_rgb[..., 2]),
            'Expected a grayscale native alpha panel.')
    x, y = local_xy.T
    alpha = alpha_rgb[y, x, 0]
    black8, white8 = black[y, x].astype(int), white[y, x].astype(int)
    residual = 255 - (white8 - black8) - alpha[:, None]
    # Independent output clipping can only reduce white-minus-black. When both
    # values avoid a clipping endpoint, three half-code roundings plus the
    # float check's tolerance bound this integer residual to one code.
    clip_endpoint = (black8 == 0) | (white8 == 255)
    beyond_rounding = residual > 1
    require(int(residual.min()) >= -1 and np.all(clip_endpoint[beyond_rounding]),
            'Saved alpha has a discrepancy unexplained by RGB clipping and rounding.')
    i, channel = np.unravel_index(residual.argmax(), residual.shape)
    x0, y0, _, _ = entry['crop_xyxy']
    composition = dict(direct_head_panel_sha256=entry['sha256'], matched_direct_native_1x_pixels=True,
        float_renderer_composition_max_abs=float_error,
        signed_u8_residual_min_max=[int(residual.min()), int(residual.max())],
        rgb_channels_with_clip_endpoint=int(clip_endpoint.sum()),
        rgb_channels_above_rounding_bound=int(beyond_rounding.sum()),
        rgb_channels_above_old_five_code_gate=int((residual > 5).sum()),
        rgb_channels_above_rounding_without_clipped_endpoint=int((beyond_rounding & ~clip_endpoint).sum()),
        max_signed_residual_example=dict(native_xy=[int(x[i])+x0, int(y[i])+y0], channel='RGB'[channel],
            black_u8=int(black8[i, channel]), white_u8=int(white8[i, channel]), alpha_u8=int(alpha[i])),
        note='Independent RGB clipping can reduce white-minus-black. Alpha comes from the direct grayscale renderer pass, never from clipped RGB.')

    def subset(select):
        values = alpha[select]
        rgb = black[y[select], x[select]].astype(float) - source[y[select], x[select]]
        return dict(pixels=len(values), alpha_u8_min_p05_p50_p95_max=np.quantile(values, [0,.05,.5,.95,1]).tolist(),
                    alpha_u8_lt_166=int((values < 166).sum()), alpha_u8_lt_243=int((values < 243).sum()),
                    black_rgb_mae_01=float(np.abs(rgb).mean() / 255))

    return dict(all_roi=subset(np.ones(len(alpha), dtype=bool)), no_mesh=subset(~on_mesh),
                right_ear_owner_mesh=subset(on_mesh), alpha_composition=composition)


def measure(candidate=None):
    audit_path = pin(ROUND4 / 'ear-016-source-audit.json', EAR_SHA)
    audit = read(audit_path)
    old_subject = read(pin(ROUND4 / 'receipt.json', RECEIPT_SHA))['subjects']['016']
    old_review = pin(old_subject['review'], old_subject['review_sha256'])
    old_run = pin(old_subject['result'], old_subject['result_sha256'])
    old_receipt = read(old_review)
    require(old_receipt['state'] == 'round4_delivery_views_complete' and old_receipt['subject'] == '016', 'Round four is incomplete.')
    old_native = camera_record(old_receipt['new'], audit)
    old_mesh_path, old_mesh = posed_mesh(old_run, audit, old_native)
    coords = audit['roi']
    image = Image.new('1', tuple(coords['wh']))
    ImageDraw.Draw(image).polygon([tuple(p) for p in coords['polygon_xy_native_integer_centers_inclusive']], fill=1)
    roi = np.asarray(image, dtype=np.bool_)
    roi_sha = hashlib.sha256(roi.tobytes(order='C')).hexdigest()
    y, x = np.nonzero(roi)
    xy = np.column_stack((x, y))
    require(len(xy) == coords['pixels'] == 574 and roi_sha == coords['roi_sha256'], 'Frozen ear ROI changed.')
    source_id, mask_id = (audit['identity'][name] for name in ('source_rgb', 'score_mask'))
    crop = audit['identity']['archived_native_1x_panels']['new']['crop_xyxy']
    with Image.open(pin(source_id['path'], source_id['sha256'])) as original, Image.open(pin(mask_id['path'], mask_id['sha256'])) as matte:
        require(original.size == matte.size == tuple(coords['wh']), 'Native source or matte size differs.')
        source = np.asarray(original.convert('RGB').crop(crop))
        require(np.all(np.asarray(matte.convert('L'))[y, x] == 255), 'Frozen foreground support changed.')
    local_xy = xy - crop[:2]
    require(np.all(local_xy.min(0) >= 0) and np.all(local_xy.max(0) < np.asarray(source.shape[:2])[::-1]), 'ROI crosses review crop.')
    old_panel, old_strips = panel(old_review, old_receipt, source, crop, ['round1_762d6d0', 'round3_967c286', OLD_ROW])
    slot = audit['scope']['training_slot']
    owners = mesh_coverage(old_mesh['vertices_world'][slot], old_mesh['faces'], old_native['native_camera'], xy)
    on_mesh = owners >= 0
    require(int((~on_mesh).sum()) == audit['mesh_support']['no_nearest_triangle_pixels'] == 267
            and np.unique(owners[on_mesh]).tolist() == audit['mesh_support']['distinct_right_ear_owner_faces'],
            'Round-four nearest ear-mesh support differs from frozen audit.')
    baseline = metrics(old_strips[-1], source, local_xy, on_mesh, old_review, old_native)
    result = dict(schema='shuolee-smpl-round5-ear016-remeasurement-v1', state='baseline_verified', subject='016', frame=FRAME,
        source_frame_index=old_native['source_frame_index'], source_sha256=source_id['sha256'], mask_sha256=mask_id['sha256'],
        roi=dict(audit_sha256=EAR_SHA, sha256=roi_sha, pixels=len(xy), mask_255_pixels=len(xy), crop_xyxy=crop),
        native_camera=old_native['native_camera'], gpu_calls=0, optimizer_updates=0, quality_accepted=False,
        round4=dict(panel=str(old_panel), panel_sha256=sha(old_panel), review_sha256=sha(old_review),
                    posed_mesh_sha256=sha(old_mesh_path), metrics=baseline))
    if candidate is None:
        return result

    reviewed = read(candidate)
    plan = read(pin(candidate.parent / 'plan.json', reviewed['plan_sha256']))
    commit = plan['new_commit']
    require(reviewed['state'] == 'round5_delivery_views_complete' and reviewed['subject'] == '016'
            and reviewed['optimizer_updates'] == reviewed['refits'] == 0 and re.fullmatch('[0-9a-f]{40}', commit)
            and Path(plan['baseline_review']) == old_review and plan['baseline_review_sha256'] == sha(old_review),
            'Use the completed matched Round-five review.')
    new_native = camera_record(reviewed['new'], audit)
    require(new_native['native_camera'] == old_native['native_camera'] and new_native['pose_sha256'] == old_native['pose_sha256'],
            'Round five changed the camera or fit pose.')
    new_result_path = pin(Path(plan['new']['run']) / 'result.json', plan['new']['terminal_sha256'])
    new_mesh_path, new_mesh = posed_mesh(new_result_path, audit, new_native)
    require(old_mesh.keys() == new_mesh.keys() and all(old_mesh[k].shape == new_mesh[k].shape
            and old_mesh[k].dtype == new_mesh[k].dtype and old_mesh[k].tobytes() == new_mesh[k].tobytes() for k in old_mesh),
            'Round five changed the frozen posed mesh arrays.')
    new_panel, new_strips = panel(candidate, reviewed, source, crop, [OLD_ROW, f'round5_{commit[:7]}'])
    require(np.array_equal(new_strips[0], old_strips[-1]), 'Matched first row differs from the published Round-four PNG.')
    fresh = metrics(new_strips[1], source, local_xy, on_mesh, candidate, new_native)
    result.update(state='remeasured', frozen_posed_mesh_arrays_bitwise_equal=True, round5=dict(commit=commit,
        panel=str(new_panel), panel_sha256=sha(new_panel), review_sha256=sha(candidate), posed_mesh_sha256=sha(new_mesh_path),
        metrics=fresh), delta_round5_minus_round4={region: {key: fresh[region][key] - baseline[region][key]
        for key in ('alpha_u8_lt_166','alpha_u8_lt_243','black_rgb_mae_01')}
        for region in ('all_roi','no_mesh','right_ear_owner_mesh')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--review', type=Path)
    mode.add_argument('--check-baseline', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = measure(args.review)
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            output.write(data)
    print(data, end='')


if __name__ == '__main__':
    main()

"""Verify one ground plane against frozen background tracks in native pixels."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from audit_video21_multiview_holdout171 import BASE, CAMERA, REFERENCE, TRACKS, passes, statistics
from fit_video21_native148 import original
from prepare_video21_shoe_binding163 import FOOT_BONES
from review_full_sequence123 import bound_file, relocate
from shared_background122 import project, triangulate
from update_leafwoman129_state import atomic, read, sha


OUT = BASE / 'floor-contact174'
FIT = BASE / 'world-root-coherent173'
CAMERAS = CAMERA / 'cameras-diagnostic.json'
STRUCTURE = CAMERA / 'background-structure.npz'
TRACK_ARCHIVE = TRACKS / 'tracks.npz'
PATCHES = {
    'frame_000012': [[70, 1220, 1180, 1485], [1800, 1160, 2600, 1485]],
    'frame_000029': [[720, 1160, 1120, 1480], [1550, 1220, 1840, 1480], [90, 1370, 2620, 1485]],
    'frame_000044': [[85, 1110, 900, 1480], [1660, 1250, 2530, 1480], [940, 1370, 1650, 1490]],
    'frame_000053': [[80, 1110, 1030, 1480], [1580, 1185, 2040, 1480], [1040, 1360, 1530, 1490]],
}
CONTACT_CHECKS = ('frame_000044', 'frame_000053')


def select_pairs(uv, family, split, names, centers, xyz, source_ids):
    inside = np.zeros(uv.shape[:2], bool)
    for frame, boxes in PATCHES.items():
        i = names.index(frame)
        for x0, y0, x1, y1 in boxes:
            p = uv[i]
            inside[i] |= ((p[:, 0] >= x0) & (p[:, 0] < x1)
                          & (p[:, 1] >= y0) & (p[:, 1] < y1))
    seen = np.isfinite(uv).all(2)
    primary = np.array([name in source_ids for name in names])
    used, pairs = set(), {0: [], 1: []}
    for j in np.flatnonzero(inside.any(0)):
        group, part = int(family[j]), int(split[j])
        if part not in pairs or group in used or (part == 0 and j not in xyz):
            continue
        visible = np.flatnonzero(seen[:, j])
        i = int(np.flatnonzero(inside[:, j])[0])
        targets = visible[(visible != i) & primary[visible]]
        if len(visible) < 3 or not len(targets):
            continue
        lengths = np.linalg.norm(centers[targets] - centers[i], axis=1)
        k = int(targets[np.argmax(lengths)])
        used.add(group)
        pairs[part].append((int(j), group, i, k, visible))
    return pairs


def plane_projection(rows, uv, K, rotations, centers):
    tracks, _, a, b, _ = zip(*rows, strict=True)
    a, b = np.array(a), np.array(b)
    target = uv[b, tracks]
    ray_camera = np.c_[uv[a, tracks], np.ones(len(a))] @ np.linalg.inv(K).T
    ray = np.einsum('nij,ni->nj', rotations[a], ray_camera)
    origins = centers[a]

    def evaluate(p):
        denominator = ray @ p
        denominator = np.where(abs(denominator) < 1e-10, np.copysign(1e-10, denominator), denominator)
        depth = (1. - origins @ p) / denominator
        X = origins + depth[:, None] * ray
        camera = np.einsum('nij,nj->ni', rotations[b], X - centers[b])
        z = camera[:, 2]
        safe = np.where(abs(z) < 1e-10, np.copysign(1e-10, z), z)
        q = camera @ K.T
        pixels = q[:, :2] / safe[:, None]
        return pixels, (depth > 0.) & (z > 0.) & np.isfinite(pixels).all(1)

    return evaluate, target


def relative_contact(cameras, normal, weights, source):
    rows = []
    for frame in CONTACT_CHECKS:
        r = source[frame]
        archive = relocate(r['output'])
        assert sha(archive) == r['sha256']
        with np.load(archive, allow_pickle=False) as fit:
            v = fit['vertices_camera'][0]
        height = v @ (cameras[frame][:3, :3] @ normal)
        feet = {side: height[weights[:, ids].sum(1) > .5] for side, ids in FOOT_BONES.items()}
        lower = {side: float(np.percentile(h, 2)) for side, h in feet.items()}
        rows.append(dict(frame=frame, left_minus_right_model_units=lower['left'] - lower['right'],
                         foot_vertex_count={side: len(h) for side, h in feet.items()},
                         absolute_plane_alignment_inferred=False, source_pose_sha256=r['sha256']))
    return rows


def main():
    assert not OUT.exists()
    plan_fit = FIT / 'plan.json'
    fit_result = FIT / 'result.json'
    contact_doc = FIT / 'contacts.json'
    contacts = {r['frame']: r for r in read(contact_doc)['records']}
    plan = read(plan_fit)
    old_result = read(fit_result)
    assert sha(contact_doc) == plan['contacts_sha256'] and sha(plan_fit) == old_result['plan_sha256']
    before = read(CAMERA / 'result.json')
    pose_receipt = read(BASE / 'background-multiview-check171/result.json')
    assert sha(CAMERAS) == pose_receipt['source_pose_sha256']
    assert read(REFERENCE / 'process-exit.json')['result_sha256'] == sha(CAMERA / 'result.json')
    tracks = read(TRACKS / 'result.json')
    assert sha(TRACK_ARCHIVE) == tracks['tracks_sha256'] == read(REFERENCE / 'global-fixedk-plan.json')['tracks_sha256']
    K = np.asarray(before['K_integer_centers'])
    assert np.array_equal(K, plan['K_integer_centers']) and K[0, 1] == K[1, 0] == 0.
    with np.load(TRACK_ARCHIVE, allow_pickle=False) as a:
        names, uv, family, split = a['frames'].tolist(), a['uv'].astype(float), a['family'], a['split']
    assert len(names) == len(set(names)) == before['registered_count']
    assert not set(names) & set(plan['retained_holdouts'])
    assert set(family[split == 0]).isdisjoint(family[split == 1])
    camera_rows = {r['frame']: r for r in read(CAMERAS)}
    R = np.array([camera_rows[f]['R_w2c'] for f in names])
    C = np.array([camera_rows[f]['C_world'] for f in names])
    assert np.array_equal(C[0], [0., 0., 0.])
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-9)
    assert all(camera_rows[f]['frame_index'] == i and np.array_equal(camera_rows[f]['K_integer_centers'], K)
               for i, f in enumerate(names))
    E = np.tile(np.eye(4), (len(names), 1, 1))
    E[:, :3, :3], E[:, :3, 3] = R, -np.einsum('nij,nj->ni', R, C)
    with np.load(STRUCTURE, allow_pickle=False) as a:
        xyz = dict(zip(a['track_ids'], a['xyz'], strict=True))
    assert set(xyz) <= set(read(CAMERA / 'input-contract.json')['fit_track_ids'])
    assert all(split[j] == 0 for j in xyz)

    source = {r['frame']: r for r in old_result['frames']}
    sources = [Path(__file__), CAMERAS, STRUCTURE, TRACK_ARCHIVE, plan_fit, fit_result, contact_doc,
               Path(project.__code__.co_filename), original.ASSET]
    paths = {f: bound_file(contacts[f]['paths']['body']) for f in PATCHES}
    assert set(paths) <= source.keys()
    track_receipts = {r['frame']: r for r in tracks['camera_frame_receipts']}
    for frame, path in paths.items():
        with Image.open(path) as im:
            gray = cv2.cvtColor(np.asarray(im.convert('RGB')), cv2.COLOR_RGB2GRAY)
        receipt = track_receipts[frame]
        assert receipt['source_index'] == contacts[frame]['source_frame_index']
        assert receipt['grayscale_sha256'] == sha256(gray.tobytes()).hexdigest()
    sources.extend(paths.values())
    plan_path = OUT / 'plan.json'
    atomic(plan_path, dict(updated=datetime.now(timezone.utc).isoformat(), carpet_boxes_xyxy=PATCHES,
        source_hashes={str(f): sha(f) for f in sources}, source_holdouts=plan['retained_holdouts'],
        surface='Same visible carpet in four native frames. Boxes chosen from source thumbnails before fitting.',
        association='One lowest eligible track ID per original family. Earliest carpet anchor, largest camera-center '
                    'baseline among other real 3 fps source images, at least three total observations.',
        gauge='Only a three-parameter plane n0 dot X = 1. Frozen conditional K and saved rigid poses.',
        fit='SVD seed from original FIT-only background XYZ. soft_l1 1 native pixel, max 80 evaluations.',
        test='Original HELD families, no point deletion. Baseline triangulates only the other views.',
        threshold=dict(families_min=10, median_max_px=1., p90_max_px=3., missing_depth_allowed=0),
        relative_contact_frames=CONTACT_CHECKS, relative_contact='2nd percentile along estimated normal in each '
            'SMPL-X foot skinning group. Only relative; original bare-foot mesh and sole thickness differ.',
        camera_updates=0, body_updates=0, gaussian_started=False))
    pairs = select_pairs(uv, family, split, names, C, xyz, source.keys())
    result = dict(plan_sha256=sha(plan_path), fit_count=len(pairs[0]), held_count=len(pairs[1]),
                  camera_updates=0, body_updates=0, gaussian_started=False, physical_camera_accepted=False)
    if min(map(len, pairs.values())) < 10:
        result.update(state='insufficient_independent_carpet_tracks', plane_numeric_pass=False)
        atomic(OUT / 'result.json', result)
        print(result, flush=True)
        return

    X0 = np.array([xyz[j] for j, *_ in pairs[0]])
    origin = X0.mean(0)
    _, seed_singular, axes = np.linalg.svd(X0 - origin, full_matrices=False)
    n = axes[-1]
    assert abs(n @ origin) > 1e-8
    start = n / (n @ origin)
    fit_pixels, fit_uv = plane_projection(pairs[0], uv, K, R, C)
    held_pixels, held_uv = plane_projection(pairs[1], uv, K, R, C)
    fit = least_squares(lambda p: (fit_pixels(p)[0] - fit_uv).ravel(), start, loss='soft_l1',
                        f_scale=1., max_nfev=80)
    p = fit.x
    normal, offset = -p / np.linalg.norm(p), 1. / np.linalg.norm(p)
    predicted, valid = held_pixels(p)
    rows = []
    for i, (j, group, a, b, visible) in enumerate(pairs[1]):
        point = triangulate(K, E, [(k, uv[k, j]) for k in visible if k != b])
        q, z = project(K, E[b], point[None])
        free = bool(z[0] > 0. and np.isfinite(q).all())
        rows.append(dict(track=j, family=group, anchor=names[a], target=names[b],
            anchor_uv=uv[a, j].tolist(), target_uv=uv[b, j].tolist(),
            predicted_uv=predicted[i].tolist() if valid[i] else None,
            plane_error_px=float(np.linalg.norm(predicted[i]-held_uv[i])) if valid[i] else None,
            free_3d_error_px=float(np.linalg.norm(q[0]-held_uv[i])) if free else None))
    errors = [r['plane_error_px'] for r in rows if r['plane_error_px'] is not None]
    stats = statistics(errors)
    camera_height = C @ normal + offset
    numeric_pass = bool(fit.success and valid.all() and camera_height.min() > 0. and passes(stats))
    result.update(state='plane_test_complete', optimizer_success=bool(fit.success), evaluations=fit.nfev,
        seed_singular_values=seed_singular.tolist(), plane_normal_world=normal.tolist(),
        plane_offset_scene_units=offset, all_cameras_above_plane=bool(camera_height.min() > 0.),
        fit_px=statistics(np.linalg.norm(fit_pixels(p)[0] - fit_uv, axis=1)),
        held_plane_px=stats, held_free_3d_px=statistics([r['free_3d_error_px'] for r in rows if r['free_3d_error_px'] is not None]),
        invalid_held=int((~valid).sum()), plane_numeric_pass=numeric_pass,
        held_by_anchor={f: statistics([r['plane_error_px'] for r in rows if r['anchor']==f and r['plane_error_px'] is not None])
                        for f in PATCHES})
    if numeric_pass:
        with np.load(original.ASSET, allow_pickle=False) as a:
            weights = a['weights']
        result['contact_check'] = relative_contact(dict(zip(names, E, strict=True)), normal, weights, source)
    atomic(OUT / 'heldout.json', rows)
    result['heldout_sha256'] = sha(OUT / 'heldout.json')
    atomic(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()

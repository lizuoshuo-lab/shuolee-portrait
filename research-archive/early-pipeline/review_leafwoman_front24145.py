"""Compare complete 022 front-roster fits on the same native source pixels."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from official_smplifyx_entry123 import sha
from review_full_sequence123 import joined, mesh_renderer, relocate
from review_leafwoman_comparison144 import ear_comparison, metrics, side_comparison
from run_leafwoman_front24145 import ARCHIVE, ASSET, CONTACTS, OUT, PREVIOUS
from select_quality_frames124 import POLICY
from update_leafwoman129_state import atomic, read


def load_mode(mode, names, targets, K, faces):
    root = OUT/(mode+'-cpu')
    result = read(root/'result.json')
    assert result['successful'] and result['plan_sha256'] == sha(root/'plan.json')
    assert [r['frame'] for r in result['frames']] == names
    vertices, projections, betas, poses = [], [], [], []
    for i, row in enumerate(result['frames']):
        path = Path(row['output'])
        assert sha(path) == row['sha256'] and row['native_crop_bitexact']
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['K_integer_centers'], K) and np.array_equal(z['faces'], faces)
            assert np.array_equal(z['observed_native_keypoints'][0], targets[i])
            v, p = z['vertices_camera'][0], z['joints_projected_native'][0]
            assert v.shape == (10475, 3) and np.isfinite(v).all() and (v[:, 2] > .05).all()
            assert p.shape == (118, 2) and np.isfinite(p).all()
            vertices.append(v)
            projections.append(p)
            betas.append(z['betas'][0])
            poses.append(z['body_pose_axis_angle'][0])
    shape = np.stack(betas)
    if mode == 'shared':
        assert np.array_equal(shape, np.broadcast_to(shape[0], shape.shape))
    return np.stack(vertices), np.stack(projections), shape, np.stack(poses)


def leg_mask_comparison(item, box, meshes, K, selected):
    path = relocate(item['path'])
    assert sha(path) == item['sha256'] and item['crop_original_xyxy'] == box.tolist()
    with Image.open(path) as image:
        foreground = np.asarray(image.convert('L')) >= 128
    assert foreground.shape[::-1] == tuple(box[2:]-box[:2])
    distance = cv2.distanceTransform((~foreground).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    values = []
    for vertices in meshes:
        projected = vertices[0, selected]@K.T
        xy = np.rint(projected[:, :2]/projected[:, 2:]-box[:2]).astype(int)
        w, h = foreground.shape[::-1]
        valid = (projected[:, 2] > .05) & ((xy >= 0) & (xy < [w, h])).all(axis=1)
        result = np.full(len(xy), np.nan)
        result[valid] = distance[xy[valid, 1], xy[valid, 0]]
        values.append(result)
    common = np.isfinite(values).all(axis=0)
    assert common.any()
    return dict(common_vertices_in_crop=int(common.sum()), modes=[
        dict(inside_crop=int(np.isfinite(value).sum()), outside_fraction=float(np.mean(value[common] > 0)),
             outside_distance_p90_px=float(np.quantile(value[common], .9))) for value in values])


def review(through_shared, through_neutral=False):
    cv2.setNumThreads(1)
    roster = read(OUT.parent/'fit-plan.json')
    assert sha(ARCHIVE) == roster['previous_fit_sha256'] and sha(CONTACTS) == roster['source_contacts_sha256']
    out = OUT/('review-neutral' if through_neutral else 'review-shared' if through_shared else 'review-free')
    out.mkdir()
    contacts = {r['frame']: r for r in read(CONTACTS)['records']}
    with np.load(ARCHIVE, allow_pickle=False) as z:
        names = z['frame_ids'].tolist()
        targets, K, faces = z['observed_native_keypoints'], z['K_integer_centers'], z['faces']
        vertices = {'old_shared_pose': z['vertices_camera']}
        points = {'old_shared_pose': z['joints_projected_native']}
        poses = {'old_shared_pose': np.repeat(z['body_pose_axis_angle'], len(names), axis=0)}
    assert names == [r['frame'] for r in roster['front_arc']] and len(names) == 24
    methods = ('shared', 'neutral') if through_neutral else ('free', 'shared') if through_shared else ('free',)
    shapes = {}
    for mode in methods:
        vertices[mode], points[mode], shapes[mode], poses[mode] = load_mode(mode, names, targets, K, faces)
    errors = {k: np.linalg.norm(uv-targets[..., :2], axis=2) for k, uv in points.items()}
    arm = np.arange(2, 8)
    face = np.arange(67, 118)
    scopes = dict(arms=arm, inside_face=face)
    scores = {k: {group: metrics(x[:, ids][targets[:, ids, 2]>0]) for group, ids in scopes.items()} for k, x in errors.items()}
    assert all(group['arms']['count'] == 126 for group in scores.values())
    ears = {r['frame']: r['side'] for r in read(OUT.with_name('ear-region139')/'plan.json')['frames']}
    cuff = read(PREVIOUS/'visible-cuff-source-only.json')
    assert cuff['source_contacts_sha256'] == sha(CONTACTS)
    draw_mesh, renderer = mesh_renderer()
    records = []
    ear_stats = []
    pages = []
    selected = None
    if through_neutral:
        locked = read(OUT/'neutral-cpu/plan.json')['locked_unobserved_pose_joints']
        assert not poses['neutral'].reshape(24, 21, 3)[:, np.asarray(locked)-1].any()
        assert np.array_equal(shapes['shared'], shapes['neutral'])
        with np.load(ASSET, allow_pickle=False) as model:
            selected = model['weights'][:, locked].sum(axis=1) > .5
        assert 1000 < int(selected.sum()) < 7000
    for i, name in enumerate(names):
        row = contacts[name]
        item = row['paths']['body']
        src = relocate(item['path'])
        assert sha(src) == item['sha256']
        box = np.array(item['crop_original_xyxy'])
        with Image.open(src) as im:
            raw = im.convert('RGB')
        assert raw.size == tuple(box[2:]-box[:2])
        meshes = [v[i:i+1] for v in vertices.values()]
        panels = [raw, *[draw_mesh(raw, box, dict(vertices_camera=v, faces=faces, K_integer_centers=K))
                         for v in meshes]]
        board = joined(panels)
        path = out/f'{name}-native.png'
        board.save(path)
        if i % 4 == 0:
            sheet = Image.new('RGB', (1024, 1520), '#14171b')
            label = ImageDraw.Draw(sheet)
        board.thumbnail((1024, 358), Image.Resampling.LANCZOS)
        top = (i % 4)*380
        label.text((8, top+3), name, fill='white')
        sheet.paste(board, ((1024-board.width)//2, top+18))
        if i % 4 == 3:
            page = out/f'overview-{i//4+1:02d}.jpg'
            sheet.save(page, quality=90)
            pages.append(dict(file=page.name, sha256=sha(page), frames=names[i-3:i+1]))
        values = {k: {group: x[i, ids][targets[i, ids, 2]>0].tolist() for group, ids in scopes.items()} for k, x in errors.items()}
        rec = dict(frame=name, image=path.name, image_sha256=sha(path), errors=values)
        if through_neutral:
            rec['leg_mask_comparison'] = leg_mask_comparison(row['paths']['mask'], box, meshes[1:], K, selected)
        if name in ears:
            which = 17 if ears[name] == 'right' else 18
            im, measure = ear_comparison(meshes, faces, K, targets[i, which, :2], frame=name)
            ear_path = out/f'{name}-ear.png'
            im.save(ear_path)
            ear_stats.append(dict(measure, image=ear_path.name, image_sha256=sha(ear_path)))
        if name in cuff['data'] or through_neutral and name in ('frame_000011', 'frame_000024', 'frame_000027', 'frame_000038'):
            im = side_comparison(draw_mesh, meshes, faces)
            side = out/f'{name}-side.png'
            im.save(side)
            rec.update(side=side.name, side_sha256=sha(side),
                       wrist_pixels={k: uv[i, [4, 7]].tolist() for k, uv in points.items()})
        records.append(rec)
    with np.load(ASSET, allow_pickle=False) as body:
        directions = body['shapedirs'][..., :10].copy()
    shape_summary = {}
    for k, v in shapes.items():
        delta_beta = v - np.median(v, axis=0)
        distances = np.linalg.norm(delta_beta, axis=1)
        shape_delta = np.linalg.norm(np.einsum('ik,vck->ivc', delta_beta, directions), axis=2)
        shape_summary[k] = dict(unique=int(len(np.unique(v, axis=0))), beta_deviation_p50=float(np.median(distances)),
            beta_deviation_max=float(np.max(distances)), neutral_vertex_displacement_p90_model_m=float(np.quantile(shape_delta,.9)),
            neutral_vertex_displacement_max_model_m=float(np.max(shape_delta)))
    time = np.asarray([contacts[n]['time_seconds'] for n in names])
    assert np.all(np.diff(time) > 0)
    # SMPL-X pose channels are local. Global camera-relative root is excluded.
    arm_joint_indices = np.asarray([13, 14, 16, 17, 18, 19, 20, 21])-1
    temporal = {}
    limit = POLICY['joint_speed_deg_per_second']
    for method, angles in poses.items():
        local = angles.reshape(len(names), 21, 3)[:, arm_joint_indices]
        rotations = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(len(names), -1, 3, 3)
        delta = rotations[:-1].transpose(0, 1, 3, 2) @ rotations[1:]
        speed = np.rad2deg(Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude())
        speed = speed.reshape(len(names)-1, -1) / np.diff(time)[:, None]
        temporal[method] = dict(p50_deg_per_second=float(np.median(speed)), max_deg_per_second=float(speed.max()),
            transitions_above_existing_limit=[dict(pair=names[i:i+2], peak=float(speed[i].max()))
                                             for i in np.flatnonzero((speed > limit).any(axis=1))])
    mask_summary = None
    if through_neutral:
        rows = [record['leg_mask_comparison'] for record in records]
        mask_summary = {
            name: {stat: float(np.median([row['modes'][i][stat] for row in rows]))
                   for stat in ('outside_fraction', 'outside_distance_p90_px')}
            for i, name in enumerate(methods)
        }
    atomic(out/'result.json', dict(frames=records, method_order=list(vertices), pages=pages, ears=ear_stats,
        observed_detector_metrics=scores, shape_summary=shape_summary, upper_arm_temporal_screen=temporal,
        temporal_policy_deg_per_second=limit, temporal_screen_uses_fitted_local_rotation_only=True,
        leg_mask_comparison=mask_summary, leg_mask_scope='Common projected lower vertices inside exact native source crop. No hidden geometry or depth truth.',
        renderer_sha256=sha(renderer), fit_result_sha256={m: sha(OUT/(m+'-cpu')/'result.json') for m in methods},
        comparison_is_same_original_crops=True, synthetic_side_is_not_ground_truth=True,
        actual_visual_review=False, quality_accepted=False, gaussian_admitted=False))
    print('FRONT24_REVIEW', scores, shape_summary, mask_summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--neutral', action='store_true')
    args = parser.parse_args()
    assert not (args.shared and args.neutral)
    review(args.shared, args.neutral)

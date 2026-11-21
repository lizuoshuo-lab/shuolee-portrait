"""Read-only camera, triangulation and depth-prior audit for the shuolee project.

The sparse reconstruction is shared with training and is not geometry ground
truth. All uncertainty numbers condition on fixed calibrated cameras and 1 px
independent feature noise; camera and subject-motion uncertainty are excluded.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/mini-splatting2-computer00-001'
ROOT = LAB / 'outputs/mini-splatting2-mip-depth-001'
OUT = ROOT / 'shuolee/current/geometry-diagnosis'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), median=float(np.median(a)), p10=float(np.quantile(a, .1)),
                p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def sample(image, xy):
    return cv2.remap(image.astype('float32'), xy[:, 0].astype('float32').reshape(-1, 1),
                     xy[:, 1].astype('float32').reshape(-1, 1), cv2.INTER_LINEAR).ravel()


def inside(xy, box):
    return ((xy[:, 0] >= box[0]) & (xy[:, 0] < box[2]) &
            (xy[:, 1] >= box[1]) & (xy[:, 1] < box[3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    out = args.output
    if out.exists() and args.refresh:
        assert out.resolve() == OUT.resolve()
        assert json.loads((out / 'protocol.json').read_text())['owner'] == 'shuolee'
    out.mkdir(parents=True, exist_ok=args.refresh)
    start = time.time()
    cv2.setNumThreads(2)
    sys.path.insert(0, str(LAB / 'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    sfm = BASE / 'hover-sfm-001/dataset/sparse/0'
    camera_file = BASE / 'viewer/cameras.json'
    region_file = ROOT / 'training-head-regions-003/manifest.json'
    review_file = region_file.with_name('visual-review.json')
    review = json.loads(review_file.read_text())
    assert review['approved_for_training_attention_experiment']
    assert review['manifest_sha256'] == sha(region_file)
    boxes = {r['frame']: r['box'] for r in json.loads(region_file.read_text())['rows']}
    assert len(boxes) == 51 and all(int(n) % 8 for n in boxes)
    input_paths = [camera_file, region_file, review_file] + [sfm / n for n in ('cameras.bin', 'images.bin', 'points3D.bin')]
    fingerprints = {str(p): sha(p) for p in input_paths}
    protocol = dict(owner='shuolee', started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        input_sha256=fingerprints, source_sha256=sha(Path(__file__)),
        sparse_selection={'max_reprojection_error_px': 1., 'min_unique_images': 4},
        strong_triangulation_threshold_deg=3., minimum_train_track_length=3,
        depth_confidence_thresholds=[.25, .85], prior_fit_reserved='point_id % 5 == 0',
        region_definition='Existing visually reviewed training-photo head rectangles; include neck/background margin.',
        camera_comparison='Numeric projection implied by viewer.js, not a browser rasterizer parity test.',
        uncertainty='Fixed-camera local linear covariance (sum J^T J)^-1 at 1 px observation noise; optimistic, not measured accuracy.',
        train_validation_boundary='All 59 images contributed to SfM. Only 51 training images used for prior and triangulation analysis.',
        changes_models=False, creates_training=False)
    write_json(out / 'protocol.json', protocol)
    cs, images, points = rw.read_model(str(sfm), ext='.bin')
    cams = sorted(json.loads(camera_file.read_text()), key=lambda c: c['img_name'])
    byname = {im.name: im for im in images.values()}
    geom, camera_rows, raw_obs, equivalence = {}, [], [], []
    canonical = {}; duplicate_tracks = 0; duplicate_observations = 0; reciprocal_errors = 0
    for pid, point in points.items():
        chosen = {}
        for iid, idx in zip(point.image_ids, point.point2D_idxs):
            iid, idx = int(iid), int(idx)
            im = images[iid]; ic = cs[im.camera_id]
            reciprocal_errors += int(im.point3D_ids[idx] != pid)
            p = rw.qvec2rotmat(im.qvec) @ point.xyz + im.tvec
            uv = p[:2]/p[2]*ic.params[:2] + ic.params[2:4]
            err = float(np.linalg.norm(uv-im.xys[idx]))
            if iid not in chosen or err < chosen[iid][1]:
                chosen[iid] = (idx, err)
        duplicate_tracks += int(len(chosen) < len(point.image_ids))
        duplicate_observations += len(point.image_ids)-len(chosen)
        canonical[pid] = chosen
    assert reciprocal_errors == 0
    hygiene = dict(total_points=len(points), tracks_with_duplicate_image_observations=duplicate_tracks,
        duplicate_observations=duplicate_observations, reciprocal_reference_errors=reciprocal_errors,
        diagnostic_policy='One lowest-reprojection observation per image per point; source SfM unchanged.')
    for c in cams:
        name = c['img_name']; im = byname[name + '.jpg']; ic = cs[im.camera_id]
        assert ic.model == 'PINHOLE', f'Unsupported camera {ic.model}'
        R = rw.qvec2rotmat(im.qvec); C = -R.T @ im.tvec
        K = np.array([[ic.params[0], 0, ic.params[2]], [0, ic.params[1], ic.params[3]], [0, 0, 1.]])
        assert (ic.width, ic.height) == (c['width'], c['height'])
        assert Image.open(BASE / f'hover-sfm-001/dataset/images/{name}.jpg').size == (ic.width, ic.height)
        ids = np.array([int(x) for x in im.point3D_ids if int(x) in points], dtype=np.int64)
        good = np.array([int(x) in points for x in im.point3D_ids])
        xyz = np.array([points[int(x)].xyz for x in ids]); xy = im.xys[good]
        pc = xyz @ R.T + im.tvec
        uv = (pc @ K.T)[:, :2] / pc[:, 2, None]
        # Mathematical Three.js camera conversion used in viewer.js.
        Rv = np.asarray(c['rotation']); Cv = np.asarray(c['position'])
        gl = (xyz - Cv) @ (Rv @ np.diag([1., -1., -1.]))
        aspect = c['width'] / c['height'] * c['fy'] / c['fx']
        tan_y = c['height'] / (2 * c['fy'])
        ndc = gl[:, :2] / (-gl[:, 2, None]) / np.array([tan_y * aspect, tan_y])
        uv_view = np.stack([(ndc[:, 0] + 1) * c['width'] / 2, (1 - ndc[:, 1]) * c['height'] / 2], axis=1)
        eq = np.linalg.norm(uv_view - uv, axis=1)
        err = np.linalg.norm(uv - xy, axis=1)
        equivalence.extend(eq.tolist()); raw_obs.extend(err.tolist())
        camera_rows.append(dict(frame=name, position_max_abs=float(abs(C-Cv).max()),
            rotation_max_abs=float(abs(R.T-Rv).max()), principal_point_delta_px=[float(ic.params[2]-c['width']/2), float(ic.params[3]-c['height']/2)],
            viewer_projection_max_px=float(eq.max()), reprojection_error=stats(err)))
        geom[im.id] = dict(R=R, C=C, K=K, t=im.tvec, name=name, im=im)
    camera_result = dict(camera_count=len(cams), image_sizes_match=True,
        max_position_error=max(x['position_max_abs'] for x in camera_rows),
        max_rotation_error=max(x['rotation_max_abs'] for x in camera_rows),
        projection_difference_px=stats(equivalence), observed_feature_reprojection_px=stats(raw_obs), rows=camera_rows,
        numerical_projection_pass=bool(max(equivalence) < 1e-6), browser_rasterizer_parity_verified=False)
    write_json(out / 'camera-consistency.json', camera_result)
    print('camera projection max difference px', max(equivalence), flush=True)

    point_rows = []
    for pid, point in sorted(points.items()):
        if point.error > 1 or len(canonical[pid]) < 4:
            continue
        observations = [(i, val[0]) for i, val in canonical[pid].items() if geom[i]['name'] in boxes]
        if len(observations) < 3:
            continue
        centers = np.array([geom[i]['C'] for i, _ in observations])
        rays = point.xyz - centers; distances = np.linalg.norm(rays, axis=1); rays /= distances[:, None]
        angle = float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))
        H = np.zeros((3, 3)); head_hits = 0
        for iid, idx in observations:
            g = geom[iid]; p = g['R'] @ point.xyz + g['t']; x, y, z = p
            Jc = np.array([[g['K'][0, 0]/z, 0, -g['K'][0, 0]*x/z**2],
                           [0, g['K'][1, 1]/z, -g['K'][1, 1]*y/z**2]])
            J = Jc @ g['R']; H += J.T @ J
            head_hits += int(inside(g['im'].xys[idx:idx+1], boxes[g['name']])[0])
        eigenvalues = np.linalg.eigvalsh(H)
        worst_sigma_ratio = float(1/np.sqrt(max(eigenvalues[0], 1e-20))/np.median(distances))
        label = 'head_crop_supported' if head_hits >= 2 and head_hits/len(observations) >= .6 else ('outside_head_crop' if head_hits == 0 else 'mixed')
        point_rows.append(dict(point_id=int(pid), train_observations=len(observations), head_observations=head_hits,
            region=label, max_angle_deg=angle, reprojection_error=float(point.error),
            fixed_camera_worst_sigma_over_range_1px=worst_sigma_ratio))
    points_by_id = {r['point_id']: r for r in point_rows}
    tri = {}
    for region in ['all', 'head_crop_supported', 'outside_head_crop', 'mixed']:
        rr = point_rows if region == 'all' else [r for r in point_rows if r['region'] == region]
        tri[region] = dict(points=len(rr), max_angle_deg=stats([r['max_angle_deg'] for r in rr]),
            train_observations=stats([r['train_observations'] for r in rr]),
            fraction_under_3deg=float(np.mean([r['max_angle_deg'] < 3 for r in rr])) if rr else None,
            fixed_camera_worst_sigma_over_range_1px=stats([r['fixed_camera_worst_sigma_over_range_1px'] for r in rr]))
    write_json(out / 'triangulation.json', dict(aggregate=tri, track_hygiene=hygiene, rows=point_rows))
    print('triangulation', json.dumps(tri, separators=(',', ':')), flush=True)

    prior_rows, frame_rows, sparse_observations = [], [], []
    for g in geom.values():
        name = g['name']
        if name not in boxes:
            continue
        im = g['im']; good = np.array([int(pid) in points_by_id and canonical[int(pid)][im.id][0] == idx
                                     for idx, pid in enumerate(im.point3D_ids)])
        ids = np.asarray(im.point3D_ids[good], dtype=np.int64); xy = im.xys[good]
        xyz = np.array([points[int(pid)].xyz for pid in ids]); z = (xyz @ g['R'].T + g['t'])[:, 2]
        prior_path = ROOT / f'depth-priors/{name}.npz'
        with np.load(prior_path) as d:
            inv, weight = d['inverse_depth'].astype('float32'), d['weight'].astype('float32')
        prediction, conf = sample(inv, xy), sample(weight, xy)
        motion_path = ROOT / f'motion-masks-001/{name}.png'
        motion = sample(np.asarray(Image.open(motion_path).convert('L')), xy)
        head = inside(xy, boxes[name]); angles = np.array([points_by_id[int(pid)]['max_angle_deg'] for pid in ids])
        eligible = (z > .2) & (prediction > 0) & (motion == 0) & (angles >= 3)
        rel = np.abs(1/np.maximum(prediction, 1e-8)-z)/np.maximum(z, 1e-8)
        for j, pid in enumerate(ids):
            if not eligible[j]:
                continue
            row = dict(frame=name, point_id=int(pid), xy=xy[j].tolist(), z=float(z[j]),
                confidence=float(conf[j]), head_crop=bool(head[j]), prior_fit_reserved=bool(pid % 5 == 0),
                relative_depth_error=float(rel[j]), signed_depth_ratio=float(1/prediction[j]/z[j]),
                triangulation_deg=float(angles[j]))
            sparse_observations.append(row)
        for label, region in [('head_crop', head), ('outside_head_crop', ~head)]:
            for threshold in [.25, .85]:
                for reserved in [False, True]:
                    mask = eligible & region & (conf > threshold)
                    if reserved:
                        mask &= ids % 5 == 0
                    values = rel[mask]
                    prior_rows.append(dict(frame=name, region=label, min_confidence=threshold,
                        reserved_only=reserved, relative_depth_error=stats(values),
                        fraction_over_20pct=float((values > .2).mean()) if len(values) else None))
        frame_rows.append(dict(frame=name, robust_points=len(ids), head_crop_points=int(head.sum()),
            strong_head_points=int((head & (angles >= 3)).sum()), prior_sha256=sha(prior_path), motion_sha256=sha(motion_path)))
    aggregate = []
    for label in ['head_crop', 'outside_head_crop']:
        for threshold in [.25, .85]:
            for reserved in [False, True]:
                rows = [r for r in sparse_observations if r['head_crop'] == (label == 'head_crop')
                        and r['confidence'] > threshold and (not reserved or r['prior_fit_reserved'])]
                err = [r['relative_depth_error'] for r in rows]
                aggregate.append(dict(region=label, min_confidence=threshold, reserved_only=reserved,
                    observations=len(rows), unique_points=len({r['point_id'] for r in rows}),
                    relative_depth_error=stats(err), signed_depth_ratio=stats([r['signed_depth_ratio'] for r in rows]),
                    fraction_over_20pct=float(np.mean(np.asarray(err) > .2)) if err else None))
    write_json(out / 'prior-sparse-agreement.json', dict(aggregate=aggregate, rows=prior_rows, frames=frame_rows))
    write_json(out / 'sparse-observations.json', sparse_observations)
    assert all(sha(Path(p)) == digest for p, digest in fingerprints.items())
    summary = dict(status='complete', owner='shuolee', elapsed_seconds=time.time()-start,
        camera={k: v for k, v in camera_result.items() if k != 'rows'}, triangulation=tri, track_hygiene=hygiene,
        prior_sparse_agreement=aggregate, inputs_unchanged=True, models_modified=False, training_started=False,
        caveats=['Sparse points share SfM calibration and Gaussian initialization; not independent ground truth.',
                 'A head rectangle is not a head segmentation mask.',
                 'Small reprojection residual does not prove correct geometry or static subject.',
                 'Many pixels, including featureless surfaces, have no sparse-point validation.',
                 'Triangulation angle supports depth observability, not unseen-surface coverage.'])
    write_json(out / 'summary.json', summary)
    print('complete', out, 'elapsed', round(time.time()-start, 2), flush=True)


if __name__ == '__main__':
    main()

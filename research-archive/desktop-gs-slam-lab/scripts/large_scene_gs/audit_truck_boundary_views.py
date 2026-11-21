"""CPU-only sparse-observation boundary proxy and existing paired-image index.

Select holdout views by registered SfM boundary-proxy observation counts before
reading any rendered RGB. No render, model import, GPU, figure editing, or writes
to source data/evaluation products. RGB diagnostics use fixed geometry-selected
tiles and do not establish that a seam exists or that post optimization removes it.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
COMMIT = '677c8553dc64dfd62c272eca94a291a277733113'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dense_tiles(xy, width, height, size=32, number=3):
    """Deterministic nonoverlapping tiles; rank solely by observation counts."""
    counts = Counter((int(y // size), int(x // size)) for x, y in xy)
    result = []
    for (row, col), count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:number]:
        result.append({'bbox_xyxy_exclusive': [col * size, row * size,
                       min(width, (col + 1) * size), min(height, (row + 1) * size)],
                       'boundary_observation_count': count})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='truck-boundary-view-audit-001')
    parser.add_argument('--evaluation', default='truck-official-continuous-heldout-002')
    parser.add_argument('--views', type=int, default=6, choices=[4, 5, 6])
    args = parser.parse_args()
    for name in (args.run, args.evaluation):
        require(name.replace('-', '').isalnum(), 'Invalid output identifier')
    output = BASE / 'independent-evaluation' / args.run
    output.mkdir(parents=True, exist_ok=False)
    inputs = BASE / 'official-inputs/truck-four-chunks-001'
    training = BASE / 'training/truck-baseline-001'
    evaluation = BASE / 'independent-evaluation' / args.evaluation
    original_path = BASE / 'data-preparation/global-cameras-points.npz'
    model_reader = ROOT / 'repos/hierarchical-3d-gaussians/preprocess/read_write_model.py'
    ownership_source = ROOT / 'repos/gaussian-hierarchy-large-gs/hierarchy_explicit_loader.cpp'
    manifest_path, geometry_path = inputs / 'manifest.json', training / 'parser_geometry.npz'
    split_path = training / 'split.json'
    source_paths = [Path(__file__), manifest_path, geometry_path, split_path, original_path,
                    model_reader, ownership_source, evaluation / 'protocol.json',
                    evaluation / 'official-camera-audit.json']
    source_paths += [inputs / 'aligned/sparse/0' / n for n in ('cameras.bin', 'images.bin', 'points3D.bin', 'test.txt')]
    manifest = read(manifest_path)
    for chunk in manifest['chunks']:
        source_paths += [inputs / 'chunks' / chunk['name'] / n for n in ('center.txt', 'extent.txt')]
    before = {str(p): digest(p) for p in source_paths}
    require(before[str(geometry_path)] == manifest['baseline_geometry_sha256'], 'Frozen common camera geometry changed')
    with np.load(geometry_path, allow_pickle=False) as z:
        geometry = {k: z[k].copy() for k in z.files}
    with np.load(original_path, allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    split = read(split_path)
    test_indices = geometry['test_indices'].astype(int)
    require(geometry['image_names'][test_indices].tolist() == split['test_names'], 'Frozen holdout identities changed')
    require(len(test_indices) == 32, 'Require complete32-view holdout')
    require([manifest['original_to_png'][n] for n in split['test_names']] == manifest['test_names'], 'Derived holdout mapping changed')
    width, height = manifest['image_size']
    spec = importlib.util.spec_from_file_location('_truck_boundary_official_CPU_COLMAP_reader', model_reader)
    reader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reader
    spec.loader.exec_module(reader)
    cameras, images, points = reader.read_model(str(inputs / 'aligned/sparse/0'), ext='.bin')
    image_map = {image.name: image for image in images.values()}
    point_ids = np.array(sorted(points), dtype=np.uint64)
    xyz = np.array([points[int(pid)].xyz for pid in point_ids], dtype=np.float64)
    point_index = {int(pid): i for i, pid in enumerate(point_ids)}
    chunks = sorted(manifest['chunks'], key=lambda x: x['name'])
    centers = np.array([c['center'] for c in chunks], dtype=np.float64)
    extents = np.array([c['extent'] for c in chunks], dtype=np.float64)
    for j, c in enumerate(chunks):
        require(np.array_equal(np.loadtxt(inputs / 'chunks' / c['name'] / 'center.txt'), centers[j]), 'Chunk center changed')
        require(np.array_equal(np.loadtxt(inputs / 'chunks' / c['name'] / 'extent.txt'), extents[j]), 'Chunk extent changed')
    in_union = np.any(np.all(np.abs(xyz[:, None, :2] - centers[None, :, :2]) <= extents[None, :, :2] / 2, axis=2), axis=1)
    distances = np.linalg.norm(xyz[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(distances, axis=1, kind='stable')
    nearest, second = order[:, 0], order[:, 1]
    d1, d2 = distances[np.arange(len(xyz)), nearest], distances[np.arange(len(xyz)), second]
    ratio = np.divide(d1, d2, out=np.full(len(xyz), np.inf), where=d2 > 0)
    delta = .05
    candidate = in_union & (d2 > 0) & (ratio >= 1 - delta) & (ratio <= 1 + delta)
    reciprocal = np.divide(d2, d1, out=np.full(len(xyz), np.inf), where=d1 > 0)
    two_way = candidate & (reciprocal <= 1 + delta)
    candidate_ids = {int(p) for p in point_ids[candidate]}
    require(candidate_ids, 'No sparse boundary-proxy points inside rectangle union')
    original_index = {int(pid): i for i, pid in enumerate(original['point_ids'])}
    original_rows = np.array([original_index[int(pid)] for pid in point_ids])
    transformed = original['points_xyz'][original_rows].astype(np.float32).astype(np.float64) @ geometry['global_transform'][:3, :3].T + geometry['global_transform'][:3, 3]
    geometry_error = float(np.max(np.abs(transformed - xyz)))
    require(geometry_error < 1e-10, 'Aligned points do not use the frozen common transform')
    official_cameras = {x['name']: x for x in read(evaluation / 'official-camera-audit.json')['per_image']}
    all_views, coordinate_arrays = [], {}
    for ordinal, index in enumerate(test_indices):
        name = str(geometry['image_names'][index]); image = image_map[manifest['original_to_png'][name]]
        w2c = np.linalg.inv(geometry['camtoworlds'][index]); K = geometry['K'][index]
        cam = cameras[image.camera_id]
        camera_K = np.array([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]])
        require(np.array_equal(K, camera_K) and np.array_equal(K, official_cameras[name]['source_K']), 'Boundary image K differs from actual paired renderer K')
        pose_error = max(float(np.max(np.abs(reader.qvec2rotmat(image.qvec) - w2c[:3, :3]))), float(np.max(np.abs(image.tvec - w2c[:3, 3]))))
        require(pose_error < 1e-10, 'Derived observation camera differs from saved global camera')
        observation_ids = image.point3D_ids
        original_lo, original_hi = original['registered_point_observation_offsets'][index:index+2]
        registered = set(int(pid) for pid in observation_ids if pid >= 0)
        require(registered == set(original['registered_point_observation_ids'][original_lo:original_hi].tolist()), 'Actual registered SfM observations differ from frozen geometry')
        selected_rows = np.array([j for j, pid in enumerate(observation_ids) if int(pid) in candidate_ids], dtype=int)
        ids = observation_ids[selected_rows].astype(np.uint64)
        rows = np.array([point_index[int(pid)] for pid in ids], dtype=int)
        pixel_xy = image.xys[selected_rows].astype(np.float64)
        p_camera = xyz[rows] @ w2c[:3, :3].T + w2c[:3, 3]
        good = ((pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < width) & (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < height) & (p_camera[:, 2] > 0))
        ids, rows, pixel_xy, p_camera = ids[good], rows[good], pixel_xy[good], p_camera[good]
        projected = p_camera @ K.T
        projected = projected[:, :2] / projected[:, 2:3]
        residual = np.linalg.norm(projected - pixel_xy, axis=1)
        pairs = Counter('|'.join(sorted((chunks[nearest[r]]['name'], chunks[second[r]]['name']))) for r in rows)
        record = {'name': name, 'holdout_ordinal': ordinal, 'parser_index': int(index), 'width': width, 'height': height,
                  'registered_SfM_observations': len(registered), 'boundary_observation_count': len(ids),
                  'unique_boundary_observed_point_count': len(set(ids.tolist())), 'two_way_source_transition_observation_count': int(two_way[rows].sum()),
                  'boundary_observation_fraction': len(ids) / len(registered) if registered else None,
                  'nearest_two_chunk_pair_counts': dict(pairs), 'dense_geometry_only_tiles': dense_tiles(pixel_xy, width, height),
                  'boundary_observed_xy_min_max': [pixel_xy.min(axis=0).tolist(), pixel_xy.max(axis=0).tolist()] if len(ids) else [],
                  'reprojection_residual_px_min_median_max': [float(np.min(residual)), float(np.median(residual)), float(np.max(residual))] if len(ids) else [],
                  'K': K.tolist(), 'world_to_camera': w2c.tolist(), 'camera_pose_max_error': pose_error}
        all_views.append(record)
        prefix = f'view_{ordinal:02d}'
        coordinate_arrays.update({prefix + '_point_ids': ids, prefix + '_observed_pixel_xy': pixel_xy,
                                 prefix + '_reprojected_pixel_xy': projected, prefix + '_xyz_common_world': xyz[rows],
                                 prefix + '_nearest_two_chunks': order[rows, :2], prefix + '_nearest_distance_ratio': ratio[rows]})
    ranking = sorted((v for v in all_views if v['unique_boundary_observed_point_count'] > 0), key=lambda v: (-v['unique_boundary_observed_point_count'], v['name']))
    require(len(ranking) >= args.views, 'Too few holdouts with registered boundary proxy observations')
    chosen = ranking[:args.views]
    selection = {'selected_names': [v['name'] for v in chosen], 'rule': 'Descending unique registered boundary-proxy observation count; filename lexical tie break; no RGB/error score ranking',
                 'tile_rule': 'Fixed32x32 nonoverlapping image grid; top3 tiles by boundary observation count, then row/column order',
                 'candidate_view_count': len(ranking), 'all32_view_counts': all_views}
    write(output / 'selection.json', selection)  # Freeze before reading metric-path metadata or RGB.
    np.savez_compressed(output / 'boundary-observation-coordinates.npz', **coordinate_arrays)
    selected = []
    unchanged_images = {}
    metric_paths = {(model, setting): evaluation / model / setting / 'per-image-metrics.json'
                    for model in ('merged-unoptimized', 'merged-post') for setting in ('leaves', 'tau3')}
    metadata = {key: {row['name']: row for row in read(path)} for key, path in metric_paths.items()}
    before.update({str(p): digest(p) for p in metric_paths.values()})
    for v in chosen:
        paths, patches = {}, []
        target = None
        predictions = {}
        for (model, setting), records in metadata.items():
            row = records[v['name']]
            require(row['width'] == width and row['height'] == height, 'Paired image dimension differs from frozen K')
            target_path, prediction_path, paired_path = (Path(row[k]) for k in ('target_path', 'prediction_path', 'paired_path'))
            for path, expected in [(target_path, row['target_sha256']), (prediction_path, row['prediction_sha256']), (paired_path, row['paired_sha256'])]:
                require(digest(path) == expected, 'Original paired RGB source changed')
                unchanged_images[str(path)] = expected
            gt = np.asarray(Image.open(target_path).convert('RGB'))
            pred8 = np.asarray(Image.open(prediction_path).convert('RGB'))
            paired = np.asarray(Image.open(paired_path).convert('RGB'))
            require(np.array_equal(paired[:, :width], gt) and np.array_equal(paired[:, width:], pred8), 'Original paired panel convention differs')
            if target is None:
                target = gt.astype(np.float64) / 255.
            else:
                require(np.array_equal(target, gt.astype(np.float64) / 255.), 'Pre/post ground truth differs')
            float_path = prediction_path.parent.parent / 'float-frames' / (prediction_path.stem + '.npy')
            pred = np.load(float_path, allow_pickle=False)
            require(pred.shape == (height, width, 3) and np.isfinite(pred).all() and np.all((pred >= 0) & (pred <= 1)), 'Invalid original float RGB frame')
            require(np.array_equal((pred * 255).astype(np.uint8), pred8), 'Prediction PNG is not the exact saved float RGB quantization')
            unchanged_images[str(float_path)] = digest(float_path)
            label = model + '/' + setting
            predictions[label] = pred.astype(np.float64)
            paths[label] = {'target_path': str(target_path), 'predicted_path': str(prediction_path), 'paired_path': str(paired_path),
                            'paired_sha256': row['paired_sha256'], 'float_rgb_path': str(float_path), 'float_rgb_sha256': digest(float_path),
                            'paired_layout': 'Unmodified original: GT left, rendered RGB right, width490 height136; no annotation',
                            'pixel_ROI_in_left_GT_panel': [t['bbox_xyxy_exclusive'] for t in v['dense_geometry_only_tiles']],
                            'pixel_ROI_in_right_prediction_panel': [[b[0] + width, b[1], b[2] + width, b[3]] for b in [t['bbox_xyxy_exclusive'] for t in v['dense_geometry_only_tiles']]]}
        for tile in v['dense_geometry_only_tiles']:
            x0, y0, x1, y1 = tile['bbox_xyxy_exclusive']
            diagnostics = {}
            for label, pred in predictions.items():
                error = pred[y0:y1, x0:x1] - target[y0:y1, x0:x1]
                diagnostics[label] = {'RGB_components': int(error.size), 'MSE_float_RGB': float(np.mean(error ** 2)), 'MAE_float_RGB': float(np.mean(np.abs(error)))}
            leaf_diff = predictions['merged-post/leaves'][y0:y1, x0:x1] - predictions['merged-unoptimized/leaves'][y0:y1, x0:x1]
            patches.append({**tile, 'same_fixed_RGB_patch_diagnostic': diagnostics,
                            'pre_post_leaf_patch_max_abs_difference': float(np.max(np.abs(leaf_diff))),
                            'interpretation': 'Geometry-selected diagnostic patch, not seam mask, surface GT, causal post/ownership effect, or heldout tuning signal'})
        selected.append({**v, 'existing_original_paired_images': paths, 'fixed_patch_diagnostics': patches})
    for path, value in {**before, **unchanged_images}.items():
        require(digest(path) == value, 'Read-only input changed during CPU audit')
    shutil.copyfile(Path(__file__), output / 'source-snapshot.py')
    report = {'status': 'completed_CPU_boundary_proxy_original_image_index', 'gpu_started': False,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(Path(__file__)), 'command_argv': sys.argv,
              'selection': {k: value for k, value in selection.items() if k != 'all32_view_counts'}, 'selected_views': selected,
              'common_coordinate_frame': manifest['coordinate_frame'], 'chunk_centers': centers.tolist(), 'chunk_extents': extents.tolist(),
              'global_sparse_points': len(point_ids), 'in_XY_rectangle_union': int(in_union.sum()), 'boundary_proxy_points': int(candidate.sum()),
              'two_way_source_transition_points': int(two_way.sum()), 'aligned_original_transform_max_abs_error': geometry_error,
              'boundary_rule': {'delta': delta, 'unit': 'dimensionless', 'source_commit': COMMIT,
                  'source_uri': f'https://github.com/graphdeco-inria/gaussian-hierarchy/blob/{COMMIT}/hierarchy_explicit_loader.cpp#L22',
                  'formula': 'r=d_nearest/d_second; inside closed XY rectangle union; d_second>0; 1-delta<=r<=1+delta. Full3D Euclidean distances to original centers; nearest ties use fixed lexical chunk order.',
                  'second_ratio_note': 'Nearest ratio never exceeds1. The stricter two-way subset additionally requires d_second/d_nearest<=1+delta; this includes the endpoint where the second amplitude is zero. It is a closed transition interval, not a positive-opacity or visibility test.',
                  'scope': 'Apply official center-distance-ratio idea to sparse SfM points as an observation proxy; does not evaluate actual trained Gaussian ownership or support.'},
              'coordinate_arrays': str(output / 'boundary-observation-coordinates.npz'), 'all_view_counts': str(output / 'selection.json'),
              'sources': [{'path': path, 'sha256': value} for path, value in before.items()], 'all_read_sources_unchanged': True,
              'selected_image_source_count': len(unchanged_images),
              'limitations': ['Registered sparse points are estimated multi-view correspondences, not ground-truth surfaces or dense visibility; low-texture areas may have few or no observations.',
                  'The common COLMAP calibration/initialization used the full published capture, including test images. These views are held out from GS photometric optimization only.',
                  'Sparse-point center ratios are not trained Gaussian mean ownership, Gaussian support, opacity contributions, a seam mask, or proof of a geometric crack.',
                  'Distances use full3D norm and normalized scene units; high-Z points may satisfy a ratio band far from a narrow XY bisector. This is not a constant physical-width strip.',
                  'Observed feature pixels use the derived same-K COLMAP coordinates; source rasterizer projected-mean half-pixel convention and footprint can differ. ROI tiles are an inspection aid only.',
                  'RGB patch error mixes geometry, color, high-level approximation, antialiasing, occlusion and sampling. Without matched no-consolidation or boundary-specific controlled ablation it cannot prove a seam cause or its removal.',
                  'View/tile selection depends solely on registered observation counts. It is intentionally sparse-observation biased and need not represent every boundary or camera trajectory.',
                  'These unmodified paired originals still require visual inspection. This CPU report makes no claim that seams were eliminated; no plots, annotated images or replacement paper figures were generated.']}
    write(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'root': str(output), 'views': [(v['name'], v['unique_boundary_observed_point_count']) for v in selected],
                      'boundary_proxy_points': report['boundary_proxy_points'], 'GPU_started': False}))


if __name__ == '__main__':
    main()

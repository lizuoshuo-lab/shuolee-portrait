"""Freeze City inspection views from sparse geometry, before dense render scores.

CPU only; no torch/model/renderer/RGB imports or changes to frozen inputs.
Streaming COLMAP records retain only the 117 test image observations. Sparse
center-distance ratios are an inspection proxy, not a surface/seam GT mask.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
INPUTS = BASE / 'official-inputs/small-city-training-002'
OUTPUT = BASE / 'independent-evaluation/small-city-boundary-selection-CPU-001'
COARSE = BASE / 'independent-evaluation/small-city-local-coarse-003-flat-heldout-001'
MANIFEST_SHA = '2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def selected_images(path, wanted):
    """The official COLMAP binary schema; skip arrays of non-test views."""
    result = {}
    dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
    with Path(path).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        count = struct.unpack_from('<Q', b, 0)[0]
        offset = 8
        for _ in range(count):
            header = struct.unpack_from('<idddddddi', b, offset)
            offset += 64
            end = b.find(b'\x00', offset)
            require(end >= offset, 'Missing COLMAP image name terminator')
            name = b[offset:end].decode('utf-8')
            offset = end + 1
            observations = struct.unpack_from('<Q', b, offset)[0]
            offset += 8
            end = offset + 24 * observations
            require(end <= len(b), 'Truncated COLMAP observation array')
            if name in wanted:
                require(name not in result, 'Duplicate canonical image name')
                rows = np.frombuffer(b, dtype=dtype, count=observations, offset=offset).copy()
                result[name] = dict(id=header[0], qvec=np.array(header[1:5]),
                                    tvec=np.array(header[5:8]), camera_id=header[8], observations=rows)
            offset = end
        require(offset == len(b), 'Unexpected trailing COLMAP image bytes')
    require(count == 5822 and set(result) == wanted, 'Require all canonical117 in full5822 capture')
    return result


def sparse_positions(path):
    """Read IDs/XYZ, without allocating millions of per-point track objects."""
    with Path(path).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        count = struct.unpack_from('<Q', b, 0)[0]
        require(count == 2390040, 'Canonical City point count changed')
        ids = np.empty(count, dtype=np.uint64)
        xyz = np.empty((count, 3), dtype=np.float64)
        offset = 8
        for j in range(count):
            fields = struct.unpack_from('<QdddBBBdQ', b, offset)
            ids[j], xyz[j] = fields[0], fields[1:4]
            offset += 51 + 8 * fields[8]
            require(offset <= len(b), 'Truncated COLMAP point track array')
        require(offset == len(b), 'Unexpected trailing COLMAP point bytes')
    order = np.argsort(ids, kind='stable')
    ids, xyz = ids[order], xyz[order]
    require(np.all(ids[1:] > ids[:-1]) and np.isfinite(xyz).all(), 'Nonunique/nonfinite sparse geometry')
    return ids, xyz


def tiles(xy, width, height):
    bins = Counter((int(y // 32), int(x // 32)) for x, y in xy)
    return [dict(bbox_xyxy_exclusive=[32 * col, 32 * row,
                 min(width, 32 * (col + 1)), min(height, 32 * (row + 1))],
                 boundary_observation_count=count)
            for (row, col), count in sorted(bins.items(), key=lambda x: (-x[1], x[0]))[:3]]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    outcome = dict(status='running_CPU_selection', started_utc=datetime.now(timezone.utc).isoformat(),
                   CUDA_initialized=False, torch_imported=False, RGB_or_render_metrics_read=False)
    write(OUTPUT / 'outcome.json', outcome)
    protected = [Path(__file__), INPUTS / 'manifest.json', INPUTS / 'frozen-split.json',
                 INPUTS / 'source-copy-audit.json', COARSE / 'protocol.json']
    sparse = INPUTS / 'aligned/sparse/0'
    protected += [sparse / x for x in ('cameras.bin', 'images.bin', 'points3D.bin', 'test.txt')]
    reader_path = ROOT / 'repos/hierarchical-3d-gaussians/preprocess/read_write_model.py'
    ownership_path = ROOT / 'repos/gaussian-hierarchy-large-gs/hierarchy_explicit_loader.cpp'
    protected += [reader_path, ownership_path]
    try:
        manifest, protocol = read(INPUTS / 'manifest.json'), read(COARSE / 'protocol.json')
        require(digest(INPUTS / 'manifest.json') == MANIFEST_SHA, 'Frozen input manifest changed')
        require(digest(INPUTS / 'frozen-split.json') == manifest['frozen_split_sha256'], 'Frozen split changed')
        require(protocol['test_names'] == manifest['test_names'] and len(protocol['test_names']) == 117,
                'Canonical117 evaluation identity/order changed')
        arrays_path = Path(protocol['evaluated_input_npz_path'])
        protected += [arrays_path]
        require(digest(arrays_path) == protocol['evaluated_input_npz_sha256'], 'Frozen evaluation arrays changed')
        with np.load(arrays_path, allow_pickle=False) as z:
            names, Ks, c2ws = z['names'].copy(), z['K'].copy(), z['c2w'].copy()
        require(names.tolist() == protocol['test_names'], 'Canonical NPZ test order differs')
        metadata = {view['name']: view for view in protocol['views']}
        chunks = sorted(manifest['chunks'], key=lambda x: x['name'])
        centers = np.array([x['center'] for x in chunks], dtype=np.float64)
        extents = np.array([x['extent'] for x in chunks], dtype=np.float64)
        for i, chunk in enumerate(chunks):
            for filename, expected in [('center.txt', centers[i]), ('extent.txt', extents[i])]:
                path = INPUTS / 'chunks' / chunk['name'] / filename
                protected.append(path)
                require(np.array_equal(np.loadtxt(path), expected), 'Chunk center/extent changed')
        # Match the immutable preparation receipt for actual aligned metadata.
        copies = {str(Path(x['destination']).resolve()): x['destination_sha256']
                  for x in read(INPUTS / 'source-copy-audit.json')}
        before = {str(p): digest(p) for p in protected}
        for path in [sparse / x for x in ('cameras.bin', 'images.bin', 'points3D.bin')]:
            require(before[str(path)] == copies[str(path.resolve())], 'Canonical derived metadata changed')
        spec = importlib.util.spec_from_file_location('_city_boundary_COLMAP_cpu', reader_path)
        reader = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reader
        spec.loader.exec_module(reader)
        cameras = reader.read_cameras_binary(str(sparse / 'cameras.bin'))
        images = selected_images(sparse / 'images.bin', set(names.tolist()))
        point_ids, xyz = sparse_positions(sparse / 'points3D.bin')
        distances = np.linalg.norm(xyz[:, None, :] - centers[None, :, :], axis=2)
        order = np.argsort(distances, axis=1, kind='stable')
        d1 = np.take_along_axis(distances, order[:, :1], axis=1)[:, 0]
        d2 = np.take_along_axis(distances, order[:, 1:2], axis=1)[:, 0]
        ratio = np.divide(d1, d2, out=np.full(len(xyz), np.inf), where=d2 > 0)
        in_union = np.any(np.all(np.abs(xyz[:, None, :2] - centers[None, :, :2]) <= extents[None, :, :2] / 2, axis=2), axis=1)
        candidate = in_union & (d2 > 0) & (ratio >= .95) & (ratio <= 1.05)
        records, coordinates = [], {}
        for ordinal, name in enumerate(names.tolist()):
            image, meta = images[name], metadata[name]
            cam = cameras[image['camera_id']]
            require(cam.model == 'PINHOLE', 'Expected rectified PINHOLE camera')
            original_K = np.array([[cam.params[0], 0, cam.params[2]],
                                   [0, cam.params[1], cam.params[3]], [0, 0, 1]])
            require(np.array_equal(original_K, np.array(meta['original_K'])), 'Original K differs')
            width, height = meta['derived_dimensions']
            require((width, height) == (256, 172), 'Frozen r4 image size changed')
            require(np.array_equal(Ks[ordinal], np.array(meta['K'])) and
                    np.array_equal(c2ws[ordinal], np.array(meta['C2W'])), 'Canonical render calibration changed')
            w2c = np.linalg.inv(c2ws[ordinal])
            pose_error = max(float(np.max(np.abs(w2c[:3, :3] - reader.qvec2rotmat(image['qvec'])))),
                             float(np.max(np.abs(w2c[:3, 3] - image['tvec']))))
            require(pose_error < 1e-9, 'COLMAP pose differs from frozen rendered pose')
            obs = image['observations']
            obs = obs[obs['id'] >= 0]
            rows = np.searchsorted(point_ids, obs['id'].astype(np.uint64))
            require(np.all(rows < len(point_ids)), 'Unknown observed global sparse point ID')
            require(np.array_equal(point_ids[rows], obs['id'].astype(np.uint64)), 'Sparse observation ID mismatch')
            xy = np.column_stack([obs['x'], obs['y']]) * np.array([width / cam.width, height / cam.height])
            camera_xyz = xyz[rows] @ w2c[:3, :3].T + w2c[:3, 3]
            good = candidate[rows] & (camera_xyz[:, 2] > .2) & np.all((xy >= 0) & (xy < [width, height]), axis=1)
            chosen_rows, chosen_xy, chosen_camera = rows[good], xy[good], camera_xyz[good]
            projected = chosen_camera @ Ks[ordinal].T
            projected = projected[:, :2] / projected[:, 2:3]
            residual = np.linalg.norm(projected - chosen_xy, axis=1)
            pairs = Counter('|'.join(chunks[j]['name'] for j in sorted(order[r, :2])) for r in chosen_rows)
            records.append(dict(name=name, pass_group=name.split('_')[0], holdout_ordinal=ordinal,
                registered_SfM_observations=len(obs), boundary_observation_count=len(chosen_rows),
                unique_boundary_observed_point_count=len(np.unique(point_ids[chosen_rows])),
                nearest_two_chunk_pair_counts=dict(pairs), dense_geometry_only_tiles=tiles(chosen_xy, width, height),
                pose_max_abs_error=pose_error,
                scaled_reprojection_residual_px_min_median_max=[float(np.min(residual)), float(np.median(residual)), float(np.max(residual))] if len(residual) else [],
                K=Ks[ordinal].tolist(), C2W=c2ws[ordinal].tolist(), width=width, height=height))
            coordinates[name] = dict(ids=point_ids[chosen_rows], xy=chosen_xy, projected_xy=projected,
                world_xyz=xyz[chosen_rows], nearest_two_chunks=order[chosen_rows, :2], distance_ratio=ratio[chosen_rows])
        selected = []
        for group in ('pass1', 'pass2', 'pass3', 'pass4'):
            ranking = sorted([r for r in records if r['pass_group'] == group and r['unique_boundary_observed_point_count'] > 0],
                             key=lambda r: (-r['unique_boundary_observed_point_count'], r['name']))
            require(len(ranking) >= 2, 'Insufficient boundary observations for fixed per-pass2 rule')
            selected += ranking[:2]
        selection = dict(status='frozen_geometry_only_before_city_dense_results',
            selected_names=[r['name'] for r in selected], selected_views=selected, all117_view_counts=records,
            rule='Per filename pass1/2/3/4 choose top2 unique registered boundary-proxy observations; lexical filename tie break. Fixed eight views. No RGB/PSNR/SSIM/LPIPS ranking.',
            tile_rule='Fixed32x32 nonoverlapping image grid, top3 tiles by sparse boundary observation count then row/column. No residual/metric ranking.',
            pass_semantics='Filename capture-pass grouping, not demonstrated to equal paper rig camera identities.')
        write(OUTPUT / 'selection.json', selection)
        arrays = {f'view_{j:02d}_{key}': array for j, row in enumerate(selected) for key, array in coordinates[row['name']].items()}
        np.savez_compressed(OUTPUT / 'selected-sparse-observations.npz', **arrays)
        require(all(digest(path) == sha for path, sha in before.items()), 'Protected input changed during read-only CPU selection')
        require('torch' not in sys.modules, 'CPU selector must not import torch')
        report = dict(status='completed_actual_CPU_geometry_only_selection', created_utc=datetime.now(timezone.utc).isoformat(),
            source_hashes=before, protected_sources_unchanged=True, selection_sha256=digest(OUTPUT / 'selection.json'),
            coordinate_npz_sha256=digest(OUTPUT / 'selected-sparse-observations.npz'),
            sparse_points=len(point_ids), sparse_boundary_proxy_points=int(candidate.sum()),
            chunk_centers=centers.tolist(), chunk_extents=extents.tolist(), selected_names=selection['selected_names'],
            formula='d1,d2 are nearest two full3D Euclidean distances to fixed lexical-order original chunk centers; r=d1/d2; d2>0; 0.95<=r<=1.05; inside closed XY rectangle union. Units are original scene units, ratio dimensionless.',
            RGB_or_render_metrics_read=False, CUDA_initialized=False, torch_imported=False,
            limitations=['Sparse SfM observations and center ratios are an inspection proxy, not actual trained Gaussian ownership, dense support, geometric cracks, or surface GT.',
                'Full3D distance-ratio band has no fixed physical width. Scene coordinates are not independently verified metres.',
                'Selection favors textured observed surfaces; eight pass-stratified views do not prove all seams or all city quality.',
                'Observed feature pixels are scaled by256/1024 and172/690; Pillow/native projected-mean pixel-center conventions may differ. Tiles only guide inspection.',
                'Published full-capture SfM includes the117 test cameras. Holdout isolates Gaussian photometric optimization, not all camera calibration.',
                'Masks/dynamics, color, antialiasing, sampling and LOD approximations can all affect later fixed-patch RGB error. Do not infer seam cause or elimination.',
                'No city dense pre/post render exists in this selection report. Later paired originals need independent SHA checks, float RGB patch diagnostics and actual visual inspection.'])
        write(OUTPUT / 'report.json', report)
        shutil.copyfile(Path(__file__), OUTPUT / 'source-snapshot.py')
        outcome.update(status=report['status'], ended_utc=datetime.now(timezone.utc).isoformat(),
                       selected_names=selection['selected_names'], report_sha256=digest(OUTPUT / 'report.json'))
        write(OUTPUT / 'outcome.json', outcome)
        print(json.dumps(dict(status=outcome['status'], output=str(OUTPUT), views=[(r['name'],r['unique_boundary_observed_point_count']) for r in selected], CUDA_initialized=False)))
    except Exception as error:
        outcome.update(status='failed_preserved_CPU_selection', error=str(error), ended_utc=datetime.now(timezone.utc).isoformat())
        write(OUTPUT / 'outcome.json', outcome)
        raise


if __name__ == '__main__':
    main()

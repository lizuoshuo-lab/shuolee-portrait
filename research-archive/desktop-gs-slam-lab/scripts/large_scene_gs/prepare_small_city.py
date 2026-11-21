"""Read-only SmallCity readiness audit; freeze an explicit local proposed split.

No downloaded input is modified. Capture-pass grouping is a local choice, not
evidence of the paper's camera grouping or the pretrained model's train split.
Only geometry/split metadata is written, never a redistributable dataset copy.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'repos/hierarchical-3d-gaussians'
OUT = ROOT / 'outputs/large-scene-gs-001'
DATA = OUT / 'research/datasets/official-small-city/extracted/small_city/camera_calibration'
sys.path.insert(0, str(REPO / 'preprocess'))
from read_write_model import qvec2rotmat, read_cameras_binary


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ValueError('Truncated COLMAP file')
    return data


def images_metadata(path):
    result = {}
    with path.open('rb') as f:
        count, = struct.unpack('<Q', exact(f, 8))
        for _ in range(count):
            row = struct.unpack('<idddddddi', exact(f, 64))
            name = bytearray()
            while True:
                char = exact(f, 1)
                if char == b'\0':
                    break
                name.extend(char)
            name = name.decode('utf-8')
            if name in result or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Duplicate/unsafe COLMAP image name')
            observations, = struct.unpack('<Q', exact(f, 8))
            f.seek(observations * 24, 1)
            q = np.array(row[1:5], dtype=np.float64)
            t = np.array(row[5:8], dtype=np.float64)
            r = qvec2rotmat(q)
            if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-6:
                raise ValueError('Invalid camera quaternion')
            if np.max(np.abs(r @ r.T - np.eye(3))) > 1e-6 or np.linalg.det(r) < .999999:
                raise ValueError('Invalid camera rotation')
            w2c = np.eye(4)
            w2c[:3,:3], w2c[:3,3] = r, t
            result[name] = dict(id=row[0], camera_id=row[8], w2c=w2c,
                                c2w=np.linalg.inv(w2c), observations=observations)
        if f.tell() != path.stat().st_size:
            raise ValueError('COLMAP image length mismatch')
    return result


def points_arrays(path):
    with path.open('rb') as f:
        count, = struct.unpack('<Q', exact(f, 8))
        ids = np.empty(count, dtype=np.uint64)
        xyz = np.empty((count,3), dtype=np.float64)
        errors = np.empty(count, dtype=np.float64)
        tracks = np.empty(count, dtype=np.uint64)
        for i in range(count):
            row = struct.unpack('<QdddBBBdQ', exact(f, 51))
            ids[i], xyz[i], errors[i], tracks[i] = row[0], row[1:4], row[7], row[8]
            f.seek(int(tracks[i]) * 8, 1)
        if f.tell() != path.stat().st_size:
            raise ValueError('COLMAP point length mismatch')
    if len(np.unique(ids)) != count or not np.isfinite(xyz).all() or not np.isfinite(errors).all():
        raise ValueError('Invalid point identifiers or values')
    order = np.argsort(ids)
    return ids[order], xyz[order], errors[order], tracks[order]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    if not a.run.replace('-', '').isalnum():
        raise ValueError('Unsafe run identifier')
    output = OUT / 'official-inputs' / a.run
    output.mkdir(parents=True, exist_ok=False)
    config = dict(started_utc=datetime.now(timezone.utc).isoformat(), source=str(DATA),
                  script_sha256=digest(__file__), scope='CPU readiness and future local split; no training',
                  holdout_rule='Lexically sort each capture-pass filename prefix, choose index % 50 == 0',
                  holdout_limit='Capture pass is not demonstrated to equal a paper rig camera. Author pretrained training split is unknown.')
    write(output / 'configuration.json', config)
    try:
        global_sparse = DATA / 'aligned/sparse/0'
        cameras = read_cameras_binary(str(global_sparse / 'cameras.bin'))
        global_images = images_metadata(global_sparse / 'images.bin')
        names = sorted(global_images)
        groups = defaultdict(list)
        for name in names:
            groups[name.rsplit('_',1)[0]].append(name)
            for folder, suffix in [('images','.jpg'), ('masks','.png'), ('depths','.png')]:
                if not (DATA / 'rectified' / folder / Path(name).with_suffix(suffix)).is_file():
                    raise FileNotFoundError(name)
        test = sorted(name for group in groups.values() for i,name in enumerate(group) if i % 50 == 0)
        test_set = set(test)
        train = [name for name in names if name not in test_set]
        split = dict(rule=config['holdout_rule'], limitation=config['holdout_limit'],
                     groups={k:len(v) for k,v in groups.items()}, train_names=train, test_names=test)
        write(output / 'split.json', split)
        (output / 'proposed-test.txt').write_text('\n'.join(test)+'\n', encoding='utf-8')
        c2w = np.stack([global_images[n]['c2w'] for n in names])
        ks = []
        for name in names:
            camera = cameras[global_images[name]['camera_id']]
            if camera.model != 'PINHOLE':
                raise ValueError('Expected rectified PINHOLE camera')
            fx,fy,cx,cy = camera.params
            ks.append([[fx,0,cx],[0,fy,cy],[0,0,1]])
        np.savez_compressed(output / 'global-cameras.npz', names=names, c2w=c2w,
                            K=np.array(ks), train_names=train, test_names=test)
        ids,xyz,error,tracks = points_arrays(global_sparse / 'points3D.bin')
        model_records = []
        union = set()
        for chunk in sorted((DATA / 'chunks').iterdir()):
            sparse = chunk / 'sparse/0'
            ci = images_metadata(sparse / 'images.bin')
            cc = read_cameras_binary(str(sparse / 'cameras.bin'))
            if not set(ci) <= set(global_images):
                raise ValueError('Unknown chunk camera')
            translation, rotation, kdiff = [], [], []
            for name, image in ci.items():
                translation.append(float(np.linalg.norm(image['c2w'][:3,3]-global_images[name]['c2w'][:3,3])))
                delta = image['c2w'][:3,:3] @ global_images[name]['c2w'][:3,:3].T
                rotation.append(float(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))))
                kdiff.append(float(np.max(np.abs(cc[image['camera_id']].params-cameras[global_images[name]['camera_id']].params))))
            pid, pxyz, pe, pt = points_arrays(sparse / 'points3D.bin')
            j = np.searchsorted(ids,pid)
            matched = j < len(ids)
            matched[matched] &= ids[j[matched]] == pid[matched]
            displacements = np.linalg.norm(pxyz[matched]-xyz[j[matched]],axis=1)
            params = json.loads((sparse / 'depth_params.json').read_text(encoding='utf-8'))
            if set(params) != {n.rsplit('.',1)[0] for n in ci}:
                raise ValueError('Depth parameter membership mismatch')
            scales = np.array([v['scale'] for v in params.values()])
            offsets = np.array([v['offset'] for v in params.values()])
            if not np.isfinite(scales).all() or not np.isfinite(offsets).all():
                raise ValueError('Nonfinite depth calibration')
            union.update(ci)
            model_records.append(dict(chunk=chunk.name, images=len(ci), train_images=len(set(ci)-test_set),
                test_images=len(set(ci)&test_set), points=len(pid), numeric_point_ids_also_present_global=int(matched.sum()),
                max_camera_center_displacement_scene_units=max(translation),
                max_camera_rotation_difference_radians=max(rotation), max_intrinsics_difference=max(kdiff),
                same_numeric_id_distance_scene_units_not_correspondence=dict(median=float(np.median(displacements)),
                    p95=float(np.percentile(displacements,95)),max=float(displacements.max())),
                point_identity_limit='Chunk track associations are empty. Same numeric IDs are not established physical correspondences; distances must not be interpreted as BA displacement or drift.',
                saved_reprojection_error=dict(median=float(np.median(pe)),p95=float(np.percentile(pe,95))),
                zero_track_points=int((pt==0).sum()),
                depth_scale_min_max=[float(scales.min()),float(scales.max())],
                depth_offset_min_max=[float(offsets.min()),float(offsets.max())],
                center=np.loadtxt(chunk/'center.txt').tolist(), extent=np.loadtxt(chunk/'extent.txt').tolist()))
        manifest = dict(status='completed_cpu_readiness_no_training', dataset_images=len(names),
            proposed_train_images=len(train), proposed_test_images=len(test), captured_passes=split['groups'],
            original_input_unchanged=True, pretrained_split_unknown=True,
            global_points=len(ids), global_camera_center_bounds=[c2w[:,:3,3].min(axis=0).tolist(),c2w[:,:3,3].max(axis=0).tolist()],
            global_point_bounds=[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],
            global_reprojection_error_from_file=dict(median=float(np.median(error)),p95=float(np.percentile(error,95))),
            global_total_track_entries=int(tracks.sum()), chunk_geometry=model_records,
            chunk_image_union=len(union), global_images_outside_chunk_union=len(set(names)-union),
            cached_rgb_plus_mask_float32_lower_bound_GiB={str(r):len(names)*round(1024/r)*round(690/r)*4*4/2**30 for r in [1,2,4]},
            memory_scope='Lower bound RGB plus one mask only; depth, loader intermediates, model, optimizer and CUDA buffers are additional',
            geometric_units='Published aligned scene units; metric accuracy not independently calibrated',
            depth_encoding='Preserve original RGB uint8 maps with supplied per-chunk scale/offset and official /65536 decoder; not metric depth',
            geometry_provenance='Camera correspondence is established by unique filename. Point IDs are model-local and empty chunk tracks do not establish physical correspondences; no point-BA displacement conclusion.',
            holdout_limit=config['holdout_limit'],
            licenses='Dataset/model redistribution license unresolved; local learning only, no raw assets in website',
            source_files=[dict(path=str(global_sparse/n),sha256=digest(global_sparse/n)) for n in ['cameras.bin','images.bin','points3D.bin']],
            split_sha256=digest(output/'split.json'), camera_archive_sha256=digest(output/'global-cameras.npz'))
        write(output/'manifest.json',manifest)
        print(json.dumps({k:manifest[k] for k in ['status','dataset_images','proposed_train_images','proposed_test_images','global_points','chunk_image_union']},ensure_ascii=False))
    except Exception as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__ == '__main__':
    main()

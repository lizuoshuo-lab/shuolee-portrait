"""Prepare byte-preserving SmallCity inputs for unchanged official trainers.

CPU only. The split is frozen by protocol-002, not recomputed or claimed to be
the authors' split. Published coordinates and depth encodings are preserved.
"""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs' / 'large-scene-gs-001'
REPO = ROOT / 'repos' / 'hierarchical-3d-gaussians'
COMMIT = '596dc7081ee74e57a66502f4746328911f763b34'
CHUNKS = ('0_0', '0_1', '1_0', '1_1')
OFFICIAL_FILES = ('train_coarse.py', 'train_single.py', 'arguments/__init__.py',
                  'scene/__init__.py', 'scene/dataset_readers.py',
                  'scene/cameras.py', 'scene/gaussian_model.py',
                  'utils/camera_utils.py', 'utils/general_utils.py',
                  'scene/colmap_loader.py', 'utils/graphics_utils.py',
                  'gaussian_renderer/__init__.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_images(path):
    """Read poses/names, seek over observations without allocating them."""
    images = {}
    with Path(path).open('rb') as stream:
        count, = struct.unpack('<Q', stream.read(8))
        for _ in range(count):
            values = struct.unpack('<i7di', stream.read(64))
            name = bytearray()
            while True:
                char = stream.read(1)
                if char == b'\0':
                    break
                if not char:
                    raise ValueError('Truncated COLMAP image name')
                name.extend(char)
            name = name.decode('utf-8')
            if name in images or Path(name).name != name:
                raise ValueError('Duplicate or non-flat image name')
            observations, = struct.unpack('<Q', stream.read(8))
            stream.seek(24 * observations, 1)
            images[name] = {'q': np.asarray(values[1:5]), 't': np.asarray(values[5:8]),
                            'camera_id': values[8]}
        if stream.tell() != Path(path).stat().st_size:
            raise ValueError('COLMAP image length mismatch')
    return images


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def point_statistics(path, center, extent, rectangles):
    with Path(path).open('rb') as stream:
        count, = struct.unpack('<Q', stream.read(8))
    if Path(path).stat().st_size != 8 + 51 * count:
        raise ValueError('Expected published chunk points with empty tracks')
    dtype = np.dtype([('id', '<u8'), ('xyz', '<f8', (3,)), ('rgb', 'u1', (3,)),
                      ('error', '<f8'), ('tracks', '<u8')])
    points = np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(count,))
    xyz = points['xyz']
    if np.any(points['tracks'] != 0) or not np.isfinite(xyz).all():
        raise ValueError('Unexpected point tracks or invalid coordinates')
    own = np.all(np.abs(xyz[:, :2] - center[:2]) <= extent[:2] / 2, axis=1)
    memberships = {}
    other = np.zeros(count, dtype=bool)
    for name, (c, e) in rectangles.items():
        inside = np.all(np.abs(xyz[:, :2] - c[:2]) <= e[:2] / 2, axis=1)
        memberships[name] = {'count': int(inside.sum()), 'ratio': float(inside.mean())}
        if not np.array_equal(c, center):
            other |= inside
    return {'count': int(count), 'xyz_min': xyz.min(axis=0).tolist(),
            'xyz_max': xyz.max(axis=0).tolist(), 'empty_track_count': int(count),
            'inside_own_XY_rectangle_count': int(own.sum()),
            'inside_own_XY_rectangle_ratio': float(own.mean()),
            'outside_own_XY_rectangle_ratio': float((~own).mean()),
            'outside_own_inside_other_published_rectangle_ratio': float((~own & other).mean()),
            'XY_rectangle_memberships': memberships,
            'geometry_scope': 'Coordinate membership only; no cross-chunk physical point correspondence or numeric-ID overlap claim'}


def depth_statistics(path, names, train, test):
    params = json.loads(Path(path).read_text(encoding='utf-8'))
    scales = np.asarray([p['scale'] for p in params.values()], dtype=np.float64)
    if not np.isfinite(scales).all() or not np.isfinite([p['offset'] for p in params.values()]).all():
        raise ValueError('Nonfinite depth parameter')
    med = float(np.median(scales[scales > 0]))
    classes = {}
    for name in names:
        value = float(params[Path(name).stem]['scale'])
        classes[name] = ('nonpositive' if value <= 0 else 'below_0.2_median' if value < .2 * med
                         else 'above_5_median' if value > 5 * med else 'reliable')
    def counts(subset):
        return {key: sum(classes[n] == key for n in subset)
                for key in ('reliable', 'nonpositive', 'below_0.2_median', 'above_5_median')}
    return {'parameter_entries': len(params), 'positive_scale_median': med,
            'accepted_inclusive_scale_interval': [.2 * med, 5 * med],
            'all_images': counts(names), 'training_images': counts(train), 'test_images': counts(test),
            'official_rule': 'Median over every positive scale in this chunk depth_params.json; accept positive scales in [0.2*med_scale,5*med_scale]',
            'raw_decoder': 'RGB uint8 PNG -> float32 /65536, supplied scale+offset, resize, clamp negative to zero, channel0',
            'parameter_bytes_unchanged': True}


def audit_official_reader(out, rectified, train, test, chunks, copies):
    """Execute exact official reader AST definitions, without importing Scene.

    Torch is imported only for the unchanged fetchPt fallback; no CUDA device
    initialization/allocation, model, renderer, extension, or trainer is used.
    The official first-read bin-to-PLY conversion writes only new derived input.
    """
    import math
    import os
    import sys
    from typing import NamedTuple
    import torch
    from plyfile import PlyData, PlyElement

    if torch.cuda.is_initialized():
        raise ValueError('CPU reader audit must not initialize CUDA')
    source_paths = [REPO / 'scene/dataset_readers.py', REPO / 'scene/colmap_loader.py',
                    REPO / 'utils/graphics_utils.py']
    scope = {'np': np, 'math': math, 'os': os, 'sys': sys, 'json': json,
             'Path': Path, 'NamedTuple': NamedTuple, 'torch': torch,
             'PlyData': PlyData, 'PlyElement': PlyElement}
    loader = source_paths[1]
    exec(compile(ast.parse(loader.read_text(encoding='utf-8')), str(loader), 'exec'), scope)
    selections = {
        source_paths[2]: {'BasicPointCloud', 'getWorld2View2', 'focal2fov', 'fov2focal'},
        source_paths[0]: {'CameraInfo', 'SceneInfo', 'getNerfppNorm', 'readColmapCameras',
                         'fetchPly', 'fetchPt', 'storePly', 'readColmapSceneInfo'}}
    selected_definitions = []
    for path, names in selections.items():
        source = path.read_text(encoding='utf-8')
        nodes = [n for n in ast.parse(source).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError('Official CPU reader definition inventory changed')
        selected_definitions.extend({'path': str(path), 'name': n.name,
                                     'source_sha256': hashlib.sha256(ast.get_source_segment(source, n).encode('utf-8')).hexdigest()}
                                    for n in nodes)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    protected = list(dict.fromkeys([Path(c['source']) for c in copies] +
                                   [Path(c['destination']) for c in copies] + source_paths +
                                   [out / 'aligned/sparse/0/test.txt'] +
                                   [out / 'chunks' / n / 'sparse/0/test.txt' for n in CHUNKS]))
    before = {str(p): sha(p) for p in protected}
    rows, generated = [], []
    with (out / 'official-reader-CPU.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        for name in ('aligned', *CHUNKS):
            data = out / 'aligned' if name == 'aligned' else out / 'chunks' / name
            if any((data / 'sparse/0' / file).exists() for file in ('xyz.pt', 'rgb.pt')):
                raise ValueError('Unexpected PT point-cloud override')
            ply = data / 'sparse/0/points3D.ply'
            existed = ply.is_file()
            scene = scope['readColmapSceneInfo'](str(data), str(rectified / 'images'),
                str(rectified / 'masks'), '' if name == 'aligned' else str(rectified / 'depths'), True, False)
            entry = {'train_names': train, 'test_names': test} if name == 'aligned' else next(c for c in chunks if c['name'] == name)
            got_train = [c.image_name for c in scene.train_cameras]
            got_test = [c.image_name for c in scene.test_cameras]
            if got_train != sorted(entry['train_names']) or got_test != sorted(entry['test_names']):
                raise ValueError('Actual official reader train/test split mismatch')
            if not np.isfinite(scene.point_cloud.points).all():
                raise ValueError('Actual official reader has nonfinite points')
            cameras = scene.train_cameras + scene.test_cameras
            if any(not Path(c.image_path).is_file() or not Path(c.mask_path).is_file()
                   or (name != 'aligned' and (not Path(c.depth_path).is_file() or c.depth_params is None)) for c in cameras):
                raise ValueError('Actual official reader camera/reference path incomplete')
            if name == 'aligned' and any(c.depth_path or c.depth_params is not None for c in cameras):
                raise ValueError('Coarse depth must remain absent')
            if not existed:
                generated.append({'path': str(ply), 'sha256': sha(ply), 'bytes': ply.stat().st_size,
                                  'source_points3D_bin': str(data / 'sparse/0/points3D.bin'),
                                  'source_points3D_bin_sha256': sha(data / 'sparse/0/points3D.bin'),
                                  'producer': 'Unchanged official readColmapSceneInfo -> read_points3D_binary -> storePly'})
            rows.append({'stage_input': name, 'actual_train_count': len(got_train), 'actual_test_count': len(got_test),
                         'exact_frozen_name_lists_passed': True, 'train_test_overlap': len(set(got_train) & set(got_test)),
                         'actual_point_count': len(scene.point_cloud.points), 'point_cloud_PLY_sha256': sha(ply),
                         'official_test_path': str(data / 'sparse/0/test.txt'),
                         'official_test_sha256': sha(data / 'sparse/0/test.txt'),
                         'all_camera_reference_paths_exist': True,
                         'depth_parameters_present': name != 'aligned', 'nerf_radius': float(scene.nerf_normalization['radius'])})
            del scene
        negative = BASE / 'official-inputs/small-city-training-001/aligned'
        try:
            scope['readColmapSceneInfo'](str(negative), str(rectified / 'images'), str(rectified / 'masks'), '', True, False)
        except FileNotFoundError as exc:
            expected = str(negative / 'sparse/0/test.txt')
            if Path(exc.filename).resolve() != Path(expected).resolve():
                raise ValueError('Historical negative control failed at an unrelated path') from exc
            negative_control = {'historical_input': str(negative), 'failure': 'FileNotFoundError',
                                'actual_missing_path': expected, 'confirmed': True}
        else:
            raise ValueError('Historical missing-test-path negative control did not reject')
    after = {str(p): sha(p) for p in protected}
    if before != after or torch.cuda.is_initialized():
        raise ValueError('CPU reader audit mutated protected inputs or initialized CUDA')
    receipt = {'status': 'actual_official_reader_AST_CPU_passed', 'gpu_started': False,
               'CUDA_initialized': False, 'Scene_or_GaussianModel_instantiated': False,
               'execution_scope': 'Exact original AST definitions and full pure colmap_loader; CPU torch fetchPt missing-file fallback; official bin-to-PLY only in new derived input',
               'official_source_files': [{'path': str(p), 'sha256': sha(p)} for p in source_paths],
               'selected_original_definitions': selected_definitions, 'stages': rows,
               'generated_PLY': generated, 'historical_001_negative_control': negative_control,
               'protected_input_before_after_hashes': [{'path': p, 'before': h, 'after': after[p], 'unchanged': True} for p, h in before.items()]}
    save(out / 'official-reader-CPU-audit.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='small-city-training-002')
    args = parser.parse_args()
    if not args.output.replace('-', '').isalnum():
        raise ValueError('Invalid output identifier')
    dataset = BASE / 'research/datasets/official-small-city/extracted/small_city'
    original = dataset / 'camera_calibration'
    protocol = BASE / 'official-inputs/small-city-protocol-002'
    split_path = protocol / 'split.json'
    protocol_manifest = json.loads((protocol / 'manifest.json').read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    train, test = split['train_names'], split['test_names']
    if sha(split_path) != protocol_manifest['split_sha256'] or len(train) != 5705 or len(test) != 117:
        raise ValueError('Frozen protocol split verification failed')
    if set(train) & set(test) or len(set(train + test)) != 5822:
        raise ValueError('Invalid frozen split')
    if subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Official repository commit changed')
    source_files = []
    for relative in OFFICIAL_FILES:
        path = REPO / relative
        canonical = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{relative}'])
        if path.read_bytes().replace(b'\r\n', b'\n') != canonical.replace(b'\r\n', b'\n'):
            raise ValueError(f'Official source changed: {relative}')
        source_files.append({'relative_path': relative, 'path': str(path), 'sha256': sha(path)})
    global_images = read_images(original / 'aligned/sparse/0/images.bin')
    if set(global_images) != set(train + test):
        raise ValueError('Frozen split does not match registered global images')
    global_camera_bytes = (original / 'aligned/sparse/0/cameras.bin').read_bytes()
    rectangles = {name: (np.loadtxt(original / 'chunks' / name / 'center.txt'),
                          np.loadtxt(original / 'chunks' / name / 'extent.txt')) for name in CHUNKS}
    out = BASE / 'official-inputs' / args.output
    file_sources = [(p, out / 'aligned' / p.relative_to(original / 'aligned'))
                    for p in (original / 'aligned').rglob('*') if p.is_file()]
    for name in CHUNKS:
        file_sources.extend((p, out / 'chunks' / name / p.relative_to(original / 'chunks' / name))
                            for p in (original / 'chunks' / name).rglob('*') if p.is_file())
    copy_bytes = sum(p.stat().st_size for p, _ in file_sources)
    available = shutil.disk_usage(ROOT).free
    if available - copy_bytes < 80 * 1024**3:
        raise ValueError('Copy would violate 80 GiB system-drive reserve')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'preparation-state.json', {'status': 'preparing_cpu_only', 'gpu_started': False})
    try:
        copies = []
        for src, dst in file_sources:
            dst.parent.mkdir(parents=True, exist_ok=True)
            before = sha(src)
            shutil.copyfile(src, dst)
            after, copied = sha(src), sha(dst)
            if before != after or before != copied:
                raise ValueError(f'Immutable copy mismatch: {src}')
            copies.append({'source': str(src), 'destination': str(dst), 'bytes': dst.stat().st_size,
                           'source_sha256_before': before, 'source_sha256_after': after,
                           'destination_sha256': copied, 'byte_equal': True})
        shutil.copyfile(split_path, out / 'frozen-split.json')
        (out / 'aligned/sparse/0/test.txt').write_text('\n'.join(test) + '\n', encoding='utf-8')
        train_set, test_set = set(train), set(test)
        chunks = []
        for name in CHUNKS:
            data = out / 'chunks' / name
            images = read_images(data / 'sparse/0/images.bin')
            names = sorted(images)
            local_train = [n for n in names if n in train_set]
            local_test = [n for n in names if n in test_set]
            if len(local_train) + len(local_test) != len(names):
                raise ValueError('Chunk image outside frozen split')
            (data / 'sparse/0/test.txt').write_text('\n'.join(local_test) + '\n', encoding='utf-8')
            if (data / 'sparse/0/cameras.bin').read_bytes() != global_camera_bytes:
                raise ValueError('Global/chunk intrinsic bytes differ')
            center, extent = rectangles[name]
            max_center, max_rotation = 0., 0.
            for filename, pose in images.items():
                global_pose = global_images[filename]
                r, g = rotation(pose['q']), rotation(global_pose['q'])
                delta = np.linalg.norm(-r.T @ pose['t'] + g.T @ global_pose['t'])
                angle = np.arccos(np.clip((np.trace(r @ g.T) - 1.) / 2., -1., 1.))
                max_center, max_rotation = max(max_center, float(delta)), max(max_rotation, float(angle))
            if max_center > 1.e-4 or max_rotation > 1.e-6:
                raise ValueError('Global/chunk common coordinate frame check failed')
            chunks.append({'name': name, 'image_count': len(names), 'train_names': local_train,
                           'test_names': local_test, 'train_count': len(local_train), 'test_count': len(local_test),
                           'center': center.tolist(), 'extent': extent.tolist(),
                           'extent_Z_semantics': '2e12 is an effectively unbounded vertical partition sentinel',
                           'points': point_statistics(data / 'sparse/0/points3D.bin', center, extent, rectangles),
                           'camera_correspondence': {'key': 'unique image filename', 'intrinsics_byte_equal': True,
                            'max_center_difference_scene_units': max_center, 'max_rotation_difference_radians': max_rotation,
                            'passes': True, 'point_IDs_are_not_correspondence': True},
                           'depth_reliability': depth_statistics(data / 'sparse/0/depth_params.json', names,
                                                                  local_train, local_test)})
        rectified = original / 'rectified'
        for name in train + test:
            for subdir, filename in [('images', name), ('masks', Path(name).stem + '.png'),
                                      ('depths', Path(name).stem + '.png')]:
                if not (rectified / subdir / filename).is_file():
                    raise ValueError('Missing original image/mask/depth reference')
        if (out / 'aligned/sparse/0/depth_params.json').exists():
            raise ValueError('Expected no global depth parameters')
        reader_audit = audit_official_reader(out, rectified, train, test, chunks, copies)
        snapshot = out / 'official-source'
        for item in source_files:
            dest = snapshot / item['relative_path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item['path'], dest)
        shutil.copyfile(Path(__file__), out / 'preparation-source.py')
        manifest = {'status': 'ready_for_official_small_city_training_cpu_prepared',
                    'prepared_utc': datetime.now(timezone.utc).isoformat(), 'gpu_started': False,
                    'official_commit': COMMIT, 'official_source_files': source_files,
                    'dataset_original': str(dataset), 'original_input_unchanged': True,
                    'source_copy_audit': str(out / 'source-copy-audit.json'), 'metadata_copied_bytes': copy_bytes,
                    'images_absolute': str(rectified / 'images'), 'masks_absolute': str(rectified / 'masks'),
                    'depths_absolute': str(rectified / 'depths'), 'image_mask_depth_reference_files_exist': True,
                    'protocol_source': str(protocol), 'protocol_manifest_sha256': sha(protocol / 'manifest.json'),
                    'frozen_split_sha256': sha(split_path), 'train_names': train, 'test_names': test,
                    'test_list_relative_path': 'sparse/0/test.txt',
                    'official_reader_CPU_audit': str(out / 'official-reader-CPU-audit.json'),
                    'official_reader_CPU_audit_sha256': sha(out / 'official-reader-CPU-audit.json'),
                    'official_reader_actual_split_passed': reader_audit['status'] == 'actual_official_reader_AST_CPU_passed',
                    'train_count': len(train), 'test_count': len(test), 'chunks': chunks,
                    'split_rule': split['rule'], 'split_limitation': split['limitation'],
                    'author_pretrained_split_supplied': False,
                    'coordinates': 'Original published aligned scene units; no independent global/chunk normalization',
                    'coarse_depth': 'Disabled: no global depth_params.json supplied',
                    'coarse_skybox_count': 100000, 'coarse_SH_degree': 1,
                    'coarse_XYZ_contract': 'Learning rate fixed zero; require exact saved XYZ equality after 100000 sky prefix',
                    'chunk_depth': 'Official raw RGB uint8 /65536 and supplied unchanged scale/offset; original exponential loss schedule 1.0 to 0.01',
                    'chunk_exposure_default': 'Official optimizer learning rates 0.001 to 0.0001; train_test_exp=False',
                    'difference_from_truck': 'SmallCity retains masks and chunk depth and enables official chunk exposure optimization by default',
                    'default_resolution': 4, 'resolution4_pixels': [round(1024 / 4), round(690 / 4)],
                    'resolution4_RGB_plus_mask_if_all_cached_lower_bound_GiB': 5822 * 256 * 172 * 4 * 4 / 1024**3,
                    'memory_scope': 'Hypothetical full float32 RGB+one-mask cache lower bound; CameraDataset loads lazily with eight worker prefetch, additional depths/model/optimizer/buffers not included',
                    'local_training_is_author_pretrained_reproduction': False,
                    'licenses': 'Dataset/model redistribution license unresolved; local learning only; paper CC BY not transferred to data',
                    'system_drive_free_bytes_after_copy': shutil.disk_usage(ROOT).free,
                    'preparation_source_sha256': sha(Path(__file__))}
        save(out / 'source-copy-audit.json', copies)
        save(out / 'manifest.json', manifest)
        save(out / 'preparation-state.json', {'status': 'completed_cpu_only', 'gpu_started': False,
                                            'manifest_sha256': sha(out / 'manifest.json')})
        print(json.dumps({'status': manifest['status'], 'directory': str(out), 'train': len(train),
                          'test': len(test), 'copied_bytes': copy_bytes, 'gpu_started': False}))
    except Exception as exc:
        save(out / 'preparation-state.json', {'status': 'failed_cpu_preparation', 'gpu_started': False,
                                            'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()

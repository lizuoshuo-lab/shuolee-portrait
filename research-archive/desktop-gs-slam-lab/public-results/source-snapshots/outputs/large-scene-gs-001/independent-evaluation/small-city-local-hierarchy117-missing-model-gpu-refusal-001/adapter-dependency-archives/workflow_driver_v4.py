"""Independent SmallCity Creator -> official post -> paired Merger workflow.

plan is CPU-only. Other execution stages require --execute and the existing
run_local.py environment after explicit parent GPU scheduling. No Truck files
or official algorithm sources are changed. Reused CPU helpers are snapshotted
and imported in a separate namespace, with hashes frozen in configuration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/large-scene-gs-001'
REPO = ROOT / 'repos/hierarchical-3d-gaussians'
NATIVE_REPO = ROOT / 'repos/gaussian-hierarchy-large-gs'
BUILD = OUTPUT / 'hierarchy-build'
EXES = NATIVE_REPO / 'build-local/Release'
COMMIT = '596dc7081ee74e57a66502f4746328911f763b34'
NATIVE_COMMIT = '677c8553dc64dfd62c272eca94a291a277733113'
DRIVER_VERSION = 4
ENV_LAUNCHER = ROOT / 'scripts/large_scene_gs/run_local.py'
TRAINING_PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'
HELPER_NAMES = ('run_official_hierarchy.py', 'read_hierarchy.py', 'run_small_city_stage.py')
CPP_CONTRACT_FILES = ('loader.cpp', 'mainHierarchyCreator.cpp', 'mainHierarchyMerger.cpp',
                      'hierarchy_explicit_loader.cpp', 'hierarchy_loader.cpp', 'hierarchy_writer.cpp',
                      'runtime_switching.cu', 'types.h', 'common.h')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identifier(value):
    if not value.replace('-', '').isalnum() or len(value) > 100:
        raise ValueError('Invalid run identifier')
    return value


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_helpers(run_root):
    directory = run_root / 'helper-snapshots'
    records_path = directory / 'manifest.json'
    if not records_path.exists():
        directory.mkdir(parents=True, exist_ok=False)
        records = []
        for name in HELPER_NAMES:
            original = ROOT / 'scripts/large_scene_gs' / name
            data = original.read_bytes()
            path = directory / name
            path.write_bytes(data)
            if original.read_bytes() != data:
                raise RuntimeError('Helper changed during snapshot')
            records.append({'name': name, 'original_path': str(original), 'snapshot_path': str(path),
                            'sha256': hashlib.sha256(data).hexdigest()})
        write(records_path, records)
    records = read(records_path)
    for record in records:
        if digest(record['snapshot_path']) != record['sha256']:
            raise RuntimeError('Frozen helper bytes changed')
    codec = load_module('_smallcity_frozen_codec', directory / 'read_hierarchy.py')
    previous = sys.modules.get('read_hierarchy')
    sys.modules['read_hierarchy'] = codec
    try:
        helper = load_module('_smallcity_frozen_truck_cpu_helpers', directory / 'run_official_hierarchy.py')
    finally:
        if previous is None:
            del sys.modules['read_hierarchy']
        else:
            sys.modules['read_hierarchy'] = previous
    if helper.DRIVER_VERSION != 4:
        raise RuntimeError('Expected frozen Truck v4 anchor/half-storage contract')
    # Snapshots move __file__; preserve the original workspace globals explicitly.
    helper.ROOT, helper.OUTPUT, helper.REPO = ROOT, OUTPUT, REPO
    helper.BUILD, helper.EXES = BUILD, EXES
    training = load_module('_smallcity_frozen_training_cpu_helpers', directory / 'run_small_city_stage.py')
    return helper, codec, training, records


def source_audit(helper):
    records = helper.source_audit()  # CPU file/ZIP/hash checks; no torch import.
    for name in CPP_CONTRACT_FILES:
        path = NATIVE_REPO / name
        records.append({'path': str(path), 'sha256': digest(path)})
    for name in ('scene/cameras.py', 'scene/dataset_readers.py', 'utils/camera_utils.py', 'train_single.py', 'train_coarse.py'):
        path = REPO / name
        canonical = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{name}'])
        if path.read_bytes().replace(b'\r\n', b'\n') != canonical.replace(b'\r\n', b'\n'):
            raise RuntimeError('Official SmallCity data/training source changed')
        records.append({'path': str(path), 'sha256': digest(path)})
    records.append({'path': str(ENV_LAUNCHER), 'sha256': digest(ENV_LAUNCHER)})
    return records


def registered_image_names(path):
    """Read filenames without materializing the original COLMAP observations."""
    names = []
    with Path(path).open('rb') as stream:
        count, = struct.unpack('<Q', stream.read(8))
        for _ in range(count):
            if len(stream.read(64)) != 64:
                raise RuntimeError('Truncated original COLMAP image pose')
            name = bytearray()
            while True:
                char = stream.read(1)
                if char == b'\0':
                    break
                if not char:
                    raise RuntimeError('Truncated original COLMAP image filename')
                name.extend(char)
            names.append(name.decode('utf-8'))
            raw_count = stream.read(8)
            if len(raw_count) != 8:
                raise RuntimeError('Truncated original COLMAP observation count')
            observations, = struct.unpack('<Q', raw_count)
            stream.seek(24 * observations, 1)
        if stream.tell() != Path(path).stat().st_size or len(names) != len(set(names)):
            raise RuntimeError('Original COLMAP image payload/count mismatch')
    return set(names)


def input_split_audit(inputs, manifest):
    """Gate the actual official Reader location, including the global scaffold."""
    split = read(inputs / 'frozen-split.json')
    if split['train_names'] != manifest['train_names'] or split['test_names'] != manifest['test_names']:
        raise RuntimeError('Input manifest differs from the frozen global split')
    global_tests = set(manifest['test_names'])
    rows = [('aligned', inputs / 'aligned', manifest)]
    rows += [(c['name'], inputs / 'chunks' / c['name'], c) for c in manifest['chunks']]
    records = []
    for name, source, row in rows:
        if '360' in str(source):
            raise RuntimeError('Official Reader would override the explicit split with LLFF holdout')
        path = source / 'sparse/0/test.txt'
        if not path.is_file():
            raise RuntimeError(f'Official Reader test list missing at {path}')
        tests = path.read_text(encoding='utf-8').splitlines()
        expected_train, expected_test = set(row['train_names']), set(row['test_names'])
        registered = registered_image_names(source / 'sparse/0/images.bin')
        if (len(tests) != len(set(tests)) or set(tests) != expected_test
                or expected_train & expected_test or expected_train | expected_test != registered
                or expected_test != registered & global_tests):
            raise RuntimeError('Actual sparse/0/test.txt or COLMAP camera identities differ from frozen split')
        records.append({'stage': name, 'test_list_path': str(path), 'test_list_sha256': digest(path),
                        'train_count': len(expected_train), 'test_count': len(expected_test),
                        'registered_count': len(registered), 'train_test_overlap': 0,
                        'registered_names_partition_exact': True,
                        'official_reader_arguments': {'eval': True, 'train_test_exp': False, 'llffhold': None}})
    # The preparation owner runs the actual official CPU Reader before freezing 002.
    if (manifest.get('test_list_relative_path') != 'sparse/0/test.txt'
            or manifest.get('official_reader_actual_split_passed') is not True):
        raise RuntimeError('Require the corrected official Reader split contract and successful actual CPU Reader')
    reader_path = Path(manifest['official_reader_CPU_audit']).resolve()
    if not reader_path.is_relative_to(inputs.resolve()) or digest(reader_path) != manifest['official_reader_CPU_audit_sha256']:
        raise RuntimeError('Actual official CPU Reader audit missing, external, or changed')
    reader = read(reader_path)
    if (reader['status'] != 'actual_official_reader_AST_CPU_passed' or reader['gpu_started']
            or reader['CUDA_initialized'] or reader['Scene_or_GaussianModel_instantiated']):
        raise RuntimeError('Actual official CPU Reader audit did not pass')
    actual = {r['stage_input']: r for r in reader['stages']}
    if len(actual) != len(reader['stages']) or set(actual) != {r['stage'] for r in records}:
        raise RuntimeError('Actual official Reader stage coverage mismatch')
    for record in records:
        observed = actual[record['stage']]
        if (observed['actual_train_count'] != record['train_count']
                or observed['actual_test_count'] != record['test_count']
                or not observed['exact_frozen_name_lists_passed'] or observed['train_test_overlap'] != 0
                or Path(observed['official_test_path']).resolve() != Path(record['test_list_path']).resolve()
                or observed['official_test_sha256'] != record['test_list_sha256']):
            raise RuntimeError('Actual official Reader result differs from hierarchy input split')
    for item in reader['protected_input_before_after_hashes']:
        if not item['unchanged'] or item['before'] != item['after'] or digest(item['path']) != item['before']:
            raise RuntimeError('Official CPU Reader protected source or input changed')
    for item in reader['generated_PLY']:
        if (digest(item['path']) != item['sha256']
                or digest(item['source_points3D_bin']) != item['source_points3D_bin_sha256']):
            raise RuntimeError('Official CPU-derived PLY changed')
    return {'test_list_relative_path': 'sparse/0/test.txt', 'records': records,
            'actual_reader_audit_path': str(reader_path),
            'actual_reader_audit_sha256': digest(reader_path),
            'protected_reader_source_and_input_count': len(reader['protected_input_before_after_hashes']),
            'actual_reader_train_test_counts_and_paths_exact': True,
            'scope': 'Actual file location, filenames, membership and hashes; preparation separately executed the official CPU Reader. No Scene or GPU created here.'}


def train_pose_archive(source, target, train_names, test_names):
    """Preserve each original 64-byte COLMAP image pose and PINHOLE bytes.

    Seek over original 2D observations instead of allocating them. Empty
    observations/points are only for Creator appearance filtering, never SfM.
    """
    camera_path = source / 'sparse/0/cameras.bin'
    camera_bytes = camera_path.read_bytes()
    if len(camera_bytes) != 64 or struct.unpack('<Q', camera_bytes[:8])[0] != 1:
        raise RuntimeError('Expected supplied SmallCity single shared PINHOLE')
    camera_id, model_id, width, height, fx, fy, cx, cy = struct.unpack('<iiQQ4d', camera_bytes[8:])
    if model_id != 1:
        raise RuntimeError('AppearanceFilter requires four-parameter PINHOLE')
    selected, registered = {}, set()
    train_set, test_set = set(train_names), set(test_names)
    if train_set & test_set:
        raise RuntimeError('Train/test pose overlap')
    images_path = source / 'sparse/0/images.bin'
    with images_path.open('rb') as stream:
        count, = struct.unpack('<Q', stream.read(8))
        for _ in range(count):
            raw_pose = stream.read(64)
            if len(raw_pose) != 64:
                raise RuntimeError('Truncated camera pose')
            name = bytearray()
            while True:
                char = stream.read(1)
                if char == b'\0':
                    break
                if not char:
                    raise RuntimeError('Truncated camera name')
                name.extend(char)
            name = name.decode('utf-8')
            if name in registered:
                raise RuntimeError('Duplicate camera filename')
            registered.add(name)
            observations, = struct.unpack('<Q', stream.read(8))
            stream.seek(24 * observations, 1)
            if name in train_set:
                if struct.unpack('<i', raw_pose[60:64])[0] != camera_id:
                    raise RuntimeError('Unexpected image camera ID')
                selected[name] = raw_pose
        if stream.tell() != images_path.stat().st_size or set(selected) != train_set:
            raise RuntimeError('Pose archive source/count mismatch')
    sparse = target / 'sparse/0'
    sparse.mkdir(parents=True, exist_ok=False)
    (sparse / 'cameras.bin').write_bytes(camera_bytes)
    (sparse / 'points3D.bin').write_bytes(struct.pack('<Q', 0))
    with (sparse / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(selected)))
        for name, raw_pose in selected.items():
            stream.write(raw_pose + name.encode('utf-8') + b'\0' + struct.pack('<Q', 0))
    (sparse / 'cameras.txt').write_text(f'{camera_id} PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n', encoding='utf-8')
    lines = []
    for name, raw_pose in selected.items():
        values = struct.unpack('<i7di', raw_pose)
        lines += [str(values[0]) + ' ' + ' '.join(f'{v:.17g}' for v in values[1:8]) + f' {values[8]} {name}', '']
    (sparse / 'images.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (sparse / 'points3D.txt').write_text('', encoding='utf-8')
    result = {'source_images_sha256': digest(images_path), 'source_cameras_sha256': digest(camera_path),
              'train_only_pose_count': len(selected), 'heldout_pose_count': 0,
              'pose_64_byte_blocks_preserved_exactly': True, 'camera_bytes_preserved_exactly': True,
              'original_width_height': [width, height], 'original_fx_fy': [fx, fy],
              'scope': 'Original aligned-world W2C and K; train-only appearance-filter pose archive, empty point tracks; not an SfM reconstruction',
              'files': [{'path': str(p), 'sha256': digest(p)} for p in sorted(sparse.iterdir())]}
    write(target / 'manifest.json', result)
    return result


def binary_contract(path, skybox, codec):
    arrays = codec.read_gaussian_bin(path)
    count = len(arrays['positions'])
    if count <= 8_000_000 or not 0 < skybox < count:
        raise RuntimeError('Expected official >8M save_pt block-binary with sky prefix')
    for begin in range(0, count, 65536):
        for name, array in arrays.items():
            if not np.isfinite(array[begin:begin+65536]).all():
                raise RuntimeError(f'Nonfinite raw binary field: {name}')
        if np.any(np.linalg.norm(arrays['rotations_wxyz'][begin:begin+65536].astype(np.float64), axis=1) == 0):
            raise RuntimeError('Zero quaternion in raw binary')
        with np.errstate(over='ignore'):
            if not np.isfinite(np.exp(arrays['log_scales'][begin:begin+65536])).all():
                raise RuntimeError('Creator exp(log_scale) would overflow')
    return {'format': 'official_save_pt_block_binary_SH3', 'vertices': count, 'bytes': path.stat().st_size,
            'payload_contract': 'int32 P then float32 blocks XYZ(P,3), SH(P,16,3), opacity logits(P), log scales(P,3), wxyz rotations(P,4); 4+236P bytes',
            'skybox_prefix_skipped': skybox, 'sha256': digest(path),
            'loader_transform': 'loadBin sigmoid(logits), exp(log scales), normalize wxyz, copy basis-major RGB SH',
            'Creator_argument_rule': 'Pass absent sibling point_cloud.ply as official fallback sentinel; do not pass existing .bin to loadPly and do not manufacture any PLY',
            'source_contract': 'loader.cpp::loadBin and mainHierarchyCreator.cpp PLY failure -> replace final3 characters with bin'}


def shared_sky_audit(scaffold, chunk_input, saved, training, codec):
    skybox = int((scaffold / 'pc_info.txt').read_text())
    data = training.ply_vertices(scaffold / 'point_cloud.ply')
    center = np.loadtxt(chunk_input / 'center.txt', dtype=np.float32)
    extent = np.loadtxt(chunk_input / 'extent.txt', dtype=np.float32)
    xy = np.stack([data['x'], data['y']], axis=1)
    distance = np.max(np.abs(xy-center[:2]), axis=1)
    selected = (distance > np.float32(.5)*extent[0]) & (distance < np.float32(1.5)*extent[0])
    selected[:skybox] = True
    expected = {'positions': np.stack([data[n][:skybox] for n in ('x', 'y', 'z')], axis=1),
                'opacity_logits': data['opacity'][:skybox],
                'log_scales': np.stack([data[f'scale_{i}'][:skybox] for i in range(3)], axis=1),
                'rotations_wxyz': np.stack([data[f'rot_{i}'][:skybox] for i in range(4)], axis=1),
                'sh': np.zeros((skybox, 16, 3), dtype=np.float32)}
    expected['sh'][:, 0, :] = np.stack([data[f'f_dc_{i}'][:skybox] for i in range(3)], axis=1)
    for channel in range(3):
        for basis in range(3):
            expected['sh'][:, basis+1, channel] = data[f'f_rest_{channel*3+basis}'][:skybox]
    if (saved / 'point_cloud.ply').is_file():
        v = training.ply_vertices(saved / 'point_cloud.ply')
        actual = {'positions': np.stack([v[n][:skybox] for n in ('x', 'y', 'z')], axis=1),
                  'opacity_logits': v['opacity'][:skybox],
                  'log_scales': np.stack([v[f'scale_{i}'][:skybox] for i in range(3)], axis=1),
                  'rotations_wxyz': np.stack([v[f'rot_{i}'][:skybox] for i in range(4)], axis=1),
                  'sh': np.zeros((skybox, 16, 3), dtype=np.float32)}
        actual['sh'][:, 0, :] = np.stack([v[f'f_dc_{i}'][:skybox] for i in range(3)], axis=1)
        for channel in range(3):
            for basis in range(15):
                actual['sh'][:, basis+1, channel] = v[f'f_rest_{channel*15+basis}'][:skybox]
    else:
        actual = codec.read_gaussian_bin(saved / 'point_cloud.bin')
    equality = {name: bool(np.array_equal(reference, actual[name][:skybox])) for name, reference in expected.items()}
    if not all(equality.values()):
        raise RuntimeError('Chunk shared skybox raw fields/SH1-to-SH3 padding changed')
    return {'skybox_prefix': skybox, 'full_scaffold_prefix': int(selected.sum()),
            'non_sky_scaffold_ring': int(selected.sum())-skybox, 'raw_sky_fields_exact': equality,
            'ring_predicate': 'float32 max(abs(x-center.x),abs(y-center.y)) strictly >.5*extent.x and <1.5*extent.x, plus sky prefix',
            'note': 'Only sky prefix is frozen; full scaffold prefix excluded from densification; non-sky ring parameters may train'}


def completed_training(run_name, stage, chunk, config, manifest, training):
    directory = OUTPUT / 'official-training' / identifier(run_name)
    receipt, source_config = read(directory / 'outcome.json'), read(directory / 'configuration.json')
    if (receipt['status'] != 'completed' or source_config['stage'] != stage or source_config.get('chunk') != chunk
            or source_config['official_commit'] != COMMIT or source_config['input_manifest_sha256'] != config['input_manifest_sha256']
            or source_config['requested_iterations'] < 30000 or source_config['effective_official_arguments']['resolution'] != 4
            or source_config['train_test_exp'] or source_config['exposure_mode'] != 'official'):
        raise RuntimeError('Require real complete local r4 training with the same frozen SmallCity manifest and official exposure condition')
    iterations = source_config['requested_iterations']
    if (receipt.get('exit_code') != 0 or receipt.get('stage') != stage
            or receipt.get('requested_iterations') != iterations
            or receipt.get('optimizer_update_calls') != iterations-1
            or receipt.get('training_image_updates') != iterations-1
            or source_config.get('expected_optimizer_update_calls') != iterations-1
            or source_config['effective_official_arguments'].get('iterations') != iterations):
        raise RuntimeError('Completed stage receipt exit code, iteration budget, or update count differs from the official configured final stage')
    trainer = REPO / ('train_coarse.py' if stage == 'coarse' else 'train_single.py')
    if (source_config.get('source_changed') is not False
            or source_config.get('trainer_source_sha256') != digest(trainer)):
        raise RuntimeError('Completed training did not preserve the selected actual official trainer source')
    expected_train = manifest['train_names'] if stage == 'coarse' else next(c['train_names'] for c in manifest['chunks'] if c['name'] == chunk)
    if source_config['training_names'] != expected_train or set(expected_train) & set(manifest['test_names']):
        raise RuntimeError('Training camera identity differs from frozen split')
    arguments = source_config['effective_official_arguments']
    data = OUTPUT / 'official-inputs' / config['input']
    data = data / 'aligned' if stage == 'coarse' else data / 'chunks' / chunk
    expected_tests = manifest['test_names'] if stage == 'coarse' else next(c['test_names'] for c in manifest['chunks'] if c['name'] == chunk)
    if (not arguments.get('eval') or arguments.get('train_test_exp')
            or Path(arguments['source_path']).resolve() != data.resolve()
            or source_config.get('official_reader_CPU_audit_sha256') != config['input_split_audit']['actual_reader_audit_sha256']
            or Path(source_config.get('official_test_path', '')).resolve() != (data / 'sparse/0/test.txt').resolve()
            or source_config.get('official_test_sha256') != digest(data / 'sparse/0/test.txt')
            or source_config.get('stage_test_names') != expected_tests):
        raise RuntimeError('Completed local training did not use the actual corrected official Reader split')
    if arguments['images'] != manifest['images_absolute'] or arguments['alpha_masks'] != manifest['masks_absolute']:
        raise RuntimeError('Expected unchanged original images and masks')
    if stage == 'chunk' and (arguments['depths'] != manifest['depths_absolute'] or arguments['depth_l1_weight_init'] != 1.0 or arguments['depth_l1_weight_final'] != .01):
        raise RuntimeError('Expected actual chunk depth supervision with official loss schedule')
    if stage == 'chunk':
        # Bind the complete chunk to this plan's actual local coarse, not merely
        # to matching input metadata or a coincidentally identical sky prefix.
        coarse_run = identifier(config['coarse_run'])
        if source_config.get('coarse_run') != coarse_run:
            raise RuntimeError('Chunk coarse run differs from the selected hierarchy plan')
        coarse_directory = OUTPUT / 'official-training' / coarse_run
        scaffold, coarse_receipt, _ = completed_training(coarse_run, 'coarse', None, config, manifest, training)
        if source_config.get('coarse_outcome_sha256') != digest(coarse_directory / 'outcome.json'):
            raise RuntimeError('Chunk coarse receipt hash differs from the selected completed local coarse')
        if Path(arguments.get('scaffold_file', '')).resolve() != scaffold.resolve():
            raise RuntimeError('Chunk scaffold path differs from the selected completed local coarse model')
        coarse_ply = scaffold / 'point_cloud.ply'
        coarse_audit = read(coarse_directory / 'model-audit.json')
        if not coarse_ply.is_file():
            raise RuntimeError('Selected completed local coarse PLY missing')
        coarse_ply_sha256 = digest(coarse_ply)
        if (Path(coarse_audit.get('parameter_path', '')).resolve() != coarse_ply.resolve()
                or coarse_ply_sha256 != coarse_receipt.get('point_cloud_sha256')
                or coarse_ply_sha256 != coarse_receipt.get('model_parameter_sha256')
                or not any(Path(file['path']).resolve() == coarse_ply.resolve()
                           and file['sha256'] == coarse_ply_sha256 for file in coarse_audit['files'])):
            raise RuntimeError('Selected coarse PLY differs from its actual completed receipt or model audit')
    saved = Path(receipt['saved_model_directory']).resolve()
    if not saved.is_relative_to((directory / 'model').resolve()):
        raise RuntimeError('Saved model must be this local training output, not author pretrained assets')
    expected_saved = (directory / 'model/point_cloud' / f'iteration_{iterations}').resolve()
    if saved != expected_saved:
        raise RuntimeError('Completed stage saved model path differs from its configured final iteration')
    audit = read(directory / 'model-audit.json')
    for file in audit['files']:
        if digest(file['path']) != file['sha256']:
            raise RuntimeError('Saved local model changed after training audit')
    if stage == 'coarse' and not audit.get('exact_SfM_position_freeze_passed'):
        raise RuntimeError('Local coarse XYZ freeze not verified')
    if int((saved / 'pc_info.txt').read_text()) != 100000:
        raise RuntimeError('Require official common 100000-sky scaffold')
    return saved, receipt, source_config


def exposure_audit(path, train_names, tests):
    exposures = read(path)
    if set(exposures) != set(train_names) or set(exposures) & set(tests):
        raise RuntimeError('Exposure camera identities differ from training-only split')
    values = np.asarray(list(exposures.values()), dtype=np.float64)
    if values.shape != (len(train_names), 3, 4) or not np.isfinite(values).all():
        raise RuntimeError('Expected finite actual trained 3x4 exposure matrices')
    return {'path': str(path), 'sha256': digest(path), 'matrix_count': len(exposures),
            'nonidentity_matrix_count': int(np.any(values != np.eye(3, 4), axis=(1, 2)).sum()),
            'training_names_exact': True, 'test_overlap': 0,
            'policy': 'Retain actual trained chunk matrices; identity is not required. Post loads pretrained_exposures as frozen CUDA values and uses them for training renders; no exposure reoptimization.'}


def require_runtime():
    normal = lambda value: os.path.normcase(str(Path(value).resolve()))
    extensions = OUTPUT / 'hierarchy-build/python-extensions/cu128'
    if (normal(sys.executable) != normal(TRAINING_PYTHON)
            or normal(os.environ.get('TORCH_EXTENSIONS_DIR', '')) != normal(ROOT / '.cache/large_gs_torch_extensions')
            or normal(extensions) not in [normal(p) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
            or os.environ.get('CUDA_HOME') != r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'):
        raise RuntimeError('Execute through existing scripts/large_scene_gs/run_local.py environment')
    if shutil.disk_usage(ROOT).free < 80 * 1024**3:
        raise RuntimeError('System drive below 80 GiB reserve')


def execute(command, directory, receipt_path):
    if receipt_path.exists():
        raise RuntimeError('Stage receipt already exists; refuse overwrite')
    receipt = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'command_argv': command, 'cwd': str(REPO), 'log_path': str(directory / 'stage.log')}
    write(receipt_path, receipt)
    start = time.perf_counter()
    with (directory / 'stage.log').open('wb') as log:
        code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(status='process_exited_zero' if code == 0 else 'failed', exit_code=code,
                   process_seconds=time.perf_counter()-start)
    write(receipt_path, receipt)
    if code:
        raise SystemExit(code)
    return receipt


def post_command(directory, data, scaffold, manifest, iterations):
    return [str(TRAINING_PYTHON), '-u', str(REPO / 'train_post.py'), '-s', str(data),
            '-i', manifest['images_absolute'], '--alpha_masks', manifest['masks_absolute'],
            '--depths', manifest['depths_absolute'], '-m', str(directory / 'post-stage/model'),
            '-r', '4', '--data_device', 'cpu', '--eval', '--iterations', str(iterations),
            '--save_iterations', '-1', '--feature_lr', '0.0005', '--opacity_lr', '0.01', '--scaling_lr', '0.001',
            '--hierarchy', str(directory / 'hierarchy.hier'), '--scaffold_file', str(scaffold),
            '--skybox_locked', '--disable_viewer']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'creator', 'post', 'audit-post', 'merge-unoptimized', 'merge'], required=True)
    parser.add_argument('--run', default='small-city-hierarchy-002', type=identifier)
    parser.add_argument('--input', default='small-city-training-002', type=identifier)
    parser.add_argument('--coarse-run', default='small-city-coarse-i30000-r4-002', type=identifier)
    parser.add_argument('--chunk-run', action='append', required=True, help='One CHUNK=local-complete-run mapping per chunk')
    parser.add_argument('--chunk', choices=['0_0', '0_1', '1_0', '1_1'])
    parser.add_argument('--post-iterations', type=int, default=15000)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--revise-unexecuted-plan', action='store_true')
    args = parser.parse_args()
    if args.run == 'small-city-hierarchy-001' or args.input == 'small-city-training-001':
        raise RuntimeError('001 is a historical CPU plan with incorrect Reader test-list locations; use corrected 002 without overwriting 001')
    if args.post_iterations < 2:
        raise ValueError('At least two post iterations')
    if args.stage not in ['plan', 'audit-post'] and not args.execute:
        raise ValueError('Only plan/audit-post are CPU-only; real execution requires --execute after parent scheduling')
    if args.stage in ['creator', 'post', 'audit-post'] and not args.chunk:
        raise ValueError('Select chunk for individual stage')
    mappings = {}
    for value in args.chunk_run:
        chunk, run = value.split('=', 1)
        if chunk in mappings:
            raise ValueError('Duplicate chunk mapping')
        mappings[chunk] = identifier(run)
    inputs = OUTPUT / 'official-inputs' / args.input
    manifest_path = inputs / 'manifest.json'
    manifest = read(manifest_path)
    chunks = {c['name']: c for c in manifest['chunks']}
    if set(mappings) != set(chunks) or manifest['status'] != 'ready_for_official_small_city_training_cpu_prepared':
        raise RuntimeError('Require ready SmallCity manifest and exactly one local run per chunk')
    if digest(inputs / 'frozen-split.json') != manifest['frozen_split_sha256']:
        raise RuntimeError('Frozen split changed')
    for copy in read(manifest['source_copy_audit']):
        if digest(copy['destination']) != copy['destination_sha256']:
            raise RuntimeError('Derived aligned/chunk metadata changed')
    split_audit = input_split_audit(inputs, manifest)
    root = OUTPUT / 'official-hierarchy' / args.run
    root.mkdir(parents=True, exist_ok=True)
    helper, codec, training, helper_records = frozen_helpers(root)
    variant_path = BUILD / 'merger-unoptimized/manifest.json'
    variant = read(variant_path)
    if digest(variant['executable']['path']) != variant['executable']['sha256']:
        raise RuntimeError('Filename-only unoptimized merger executable changed')
    camera_archives = {}
    for name, chunk in chunks.items():
        target = root / 'creator-camera-archives' / name
        if (target / 'manifest.json').exists():
            archive = read(target / 'manifest.json')
            if archive['source_images_sha256'] != digest(inputs / 'chunks' / name / 'sparse/0/images.bin'):
                raise RuntimeError('Creator camera source changed')
            for file in archive['files']:
                if digest(file['path']) != file['sha256']:
                    raise RuntimeError('Frozen Creator camera archive changed')
        else:
            archive = train_pose_archive(inputs / 'chunks' / name, target, chunk['train_names'], manifest['test_names'])
        camera_archives[name] = archive
    fx = camera_archives[next(iter(chunks))]['original_fx_fy'][0] / 4
    config = {'driver_version': DRIVER_VERSION, 'driver_sha256': digest(Path(__file__)),
              'official_main_commit': COMMIT, 'official_hierarchy_commit': NATIVE_COMMIT,
              'input': args.input, 'input_manifest_sha256': digest(manifest_path),
               'input_split_audit': split_audit,
              'coarse_run': args.coarse_run, 'chunk_runs': mappings, 'resolution': 4,
              'pixels': [256, 172], 'render_fx_pixels': fx,
              'coordinate_frame': manifest['coordinates'], 'geometry_identity': 'Byte-preserved source metadata and filename-corresponding cameras; point IDs model-local with empty chunk tracks',
              'post_iterations': args.post_iterations, 'expected_post_optimizer_calls': args.post_iterations-1,
              'frozen_helper_records': helper_records, 'source_records': source_audit(helper),
              'environment_launcher_sha256': digest(ENV_LAUNCHER),
              'unoptimized_filename_variant_manifest_sha256': digest(variant_path),
              'chunk_conditions': 'Official original RGB+mask, rawRGB uint8 depth /65536 + original scale/offset, original depth reliability/loss schedule; trained exposure enabled, train_test_exp=False',
              'post_conditions': 'Unchanged official photo L1+SSIM loss; alpha mask and depth data supplied, but train_post has NO depth loss. Existing actual chunk pretrained_exposures frozen and used in rendering; no new exposure optimizer steps.',
              'post_full_train_defaults': {'feature_lr': .0005, 'opacity_lr': .01, 'scaling_lr': .001,
                'exposure_lr_init': .001, 'exposure_lr_final': .0001, 'dimensionless_log_uniform_limit': [.005, .1]},
              'limit_conversion': {'formula': 'theta=2*(tau+0.5)/fx; tau=theta*fx/2-0.5',
                'tau_max_pixels_at_r4': .1*fx/2-.5, 'formal_tau_min_pixels_at_r4': .005*fx/2-.5,
                'scope': 'Upper end approximately5.59px. Negative lower inverse is not a valid nonnegative pixel-tau; preserve original dimensionless training range, not a pixel sampling claim.'},
              'post_storage_contract': 'Decoded native anchors exact; one appended shared coarse sky tail expected half-writer roundtrip, two float32 ULP sigmoid allowance before quantization; original nodes/boxes retained',
              'merge_ownership': 'Same original centers, fixed chunk order, mode0 official nearest-center 5% opacity falloff for both outputs; not alpha-conserving deduplication',
              'merged_background': 'No sky in merged tree; use one shared100000 coarse sky during later rendering, not one per chunk',
              'author_pretrained_model_used': False, 'author_pretrained_split_known': False,
              'local_holdout_limitation': manifest['split_limitation']}
    config_path = root / 'configuration.json'
    if config_path.exists() and read(config_path) != config:
        if args.stage != 'plan' or not args.revise_unexecuted_plan or list(root.rglob('*-outcome.json')):
            raise RuntimeError('Configuration immutable after any stage receipt; only explicitly revised unexecuted CPU plan may change')
        old_sha = digest(config_path)
        shutil.copyfile(config_path, root / f'configuration-before-{old_sha[:12]}.json')
    write(config_path, config)
    shutil.copyfile(Path(__file__), root / 'driver-source.py')
    if args.stage == 'plan':
        preview = root / 'trained_chunks/0_0'
        command = post_command(preview, inputs / 'chunks/0_0', Path('<future-completed-local-coarse>'), manifest, args.post_iterations)
        pure = load_module('_smallcity_post_pure_args', REPO / 'arguments/__init__.py')
        p = argparse.ArgumentParser()
        pure.ModelParams(p); pure.OptimizationParams(p); pure.PipelineParams(p)
        p.add_argument('--save_iterations', nargs='+', type=int); p.add_argument('--disable_viewer', action='store_true')
        effective = vars(p.parse_args(command[3:]))
        if effective['train_test_exp'] or effective['resolution'] != 4:
            raise RuntimeError('Post holdout/r4 CLI contract failed')
        plan = {'status': 'CPU_prepared_no_GPU_execution', 'gpu_started': False,
                'configuration': str(config_path), 'configuration_sha256': digest(config_path),
                'dependencies': {'coarse': args.coarse_run, 'chunks': mappings, 'policy': 'Future complete local>=30000,r4 runs with same manifest; neither preflight nor author pretrained'},
                'camera_archives': camera_archives, 'post_command_preview': command, 'effective_post_args': effective,
                 'input_split_audit': split_audit,
                'post_depth_loss_present': False, 'post_exposure_optimizer_steps': 0,
                'paired_outputs': {'unoptimized': 'merged-unoptimized.hier from real hierarchy.hier via filename-only variant',
                  'post_optimized': 'merged.hier from real hierarchy.hier_opt via official Merger'},
                'binary_Creator_interface': 'If official save_pt emits point_cloud.bin (>8M): validate4+236P bytes/fields, pass absent point_cloud.ply fallback sentinel to unchanged Creator; create no fake PLY or hier_opt',
                'limits': config['limit_conversion'], 'system_drive_free_bytes': shutil.disk_usage(ROOT).free,
                'validation_scope': 'CPU source/hash, five actual sparse/0/test.txt splits and official CPU Reader receipt, exact train-only pose archives, CLI parse; runtime/model dependencies pending, no native Creator/post/Merger executed'}
        write(root / 'plan.json', plan)
        write(root / 'readiness.json', {'status': plan['status'], 'gpu_started': False,
              'driver_sha256': config['driver_sha256'], 'plan_sha256': digest(root / 'plan.json'),
              'actual_trained_dependencies_present': False, 'future_runtime_audits': ['PLY/bin input/sky fields', 'actual exposure matrices', 'anchors and post sky half-roundtrip', 'paired identical ownership']})
        print(json.dumps({'status': plan['status'], 'root': str(root), 'tau_max': config['limit_conversion']['tau_max_pixels_at_r4'], 'GPU_started': False}))
        return
    if args.stage != 'audit-post':
        require_runtime()
    scaffold, coarse_receipt, coarse_config = completed_training(args.coarse_run, 'coarse', None, config, manifest, training)
    skybox = int((scaffold / 'pc_info.txt').read_text())
    directory = root / 'trained_chunks' / str(args.chunk)
    if args.stage == 'creator':
        saved, training_receipt, training_config = completed_training(mappings[args.chunk], 'chunk', args.chunk, config, manifest, training)
        ply, binary = saved / 'point_cloud.ply', saved / 'point_cloud.bin'
        if ply.is_file():
            contract = helper.creator_ply_contract(ply, skybox)
            contract['format'] = 'official_SH3_PLY'
        else:
            contract = binary_contract(binary, skybox, codec)
        sky_audit = shared_sky_audit(scaffold, inputs / 'chunks' / args.chunk, saved, training, codec)
        exposure = OUTPUT / 'official-training' / mappings[args.chunk] / 'model/exposure.json'
        exposures = exposure_audit(exposure, chunks[args.chunk]['train_names'], manifest['test_names'])
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(exposure, directory / 'exposure.json')
        command = [str(EXES / 'GaussianHierarchyCreator.exe'), str(ply),
                   str(root / 'creator-camera-archives' / args.chunk), str(directory), str(scaffold)]
        receipt = execute(command, directory, directory / 'creator-outcome.json')
        h = codec.read_hierarchy(directory / 'hierarchy.hier')
        anchors = codec.read_anchors(directory / 'anchors.bin', len(h.positions))
        receipt.update(status='completed', source_kind='unoptimized', input_model=contract,
                       hierarchy=h.summary(), hierarchy_sha256=digest(h.path),
                       anchor_count=len(anchors), anchor_sha256=digest(directory / 'anchors.bin'),
                       source_training_receipt=training_receipt, source_exposure=exposures,
                       copied_exposure_sha256=digest(directory / 'exposure.json'),
                       creator_train_camera_archive=camera_archives[args.chunk], shared_skybox=sky_audit,
                       numeric_audit=helper.hierarchy_numeric_audit(h), configuration_sha256=digest(config_path))
        write(directory / 'creator-outcome.json', receipt)
    elif args.stage in ['post', 'audit-post']:
        creator = read(directory / 'creator-outcome.json')
        if creator['status'] != 'completed' or digest(directory / 'hierarchy.hier') != creator['hierarchy_sha256']:
            raise RuntimeError('Creator hierarchy changed or incomplete')
        if digest(directory / 'anchors.bin') != creator['anchor_sha256'] or digest(directory / 'exposure.json') != creator['copied_exposure_sha256']:
            raise RuntimeError('Creator anchors/exposures changed')
        before = codec.read_hierarchy(directory / 'hierarchy.hier')
        anchors = codec.read_anchors(directory / 'anchors.bin', len(before.positions))
        exposures = exposure_audit(directory / 'exposure.json', chunks[args.chunk]['train_names'], manifest['test_names'])
        stage_dir = directory / 'post-stage'
        command = post_command(directory, inputs / 'chunks' / args.chunk, scaffold, manifest, args.post_iterations)
        if args.stage == 'post':
            if (directory / 'hierarchy.hier_opt').exists():
                raise RuntimeError('Real post output exists; refuse overwrite')
            stage_dir.mkdir(exist_ok=False)
            write(stage_dir / 'post-input-audit.json', {'hierarchy_sha256': creator['hierarchy_sha256'],
                  'anchors_sha256': creator['anchor_sha256'], 'exposure': exposures,
                  'scaffold_sha256': digest(scaffold / 'point_cloud.ply'), 'expected_sky_tail': skybox,
                  'expected_optimizer_updates': args.post_iterations-1, 'CPU_inputs_verified_before_GPU': True})
            receipt = execute(command, stage_dir, directory / 'post-outcome.json')
        else:
            receipt = read(directory / 'post-outcome.json')
            if receipt['status'] != 'process_exited_zero' or receipt['exit_code'] != 0 or receipt['command_argv'] != command:
                raise RuntimeError('CPU audit requires actual matching exit0 post receipt')
        after = codec.read_hierarchy(directory / 'hierarchy.hier_opt', expected_unowned_tail=skybox)
        if digest(directory / 'hierarchy.hier') != creator['hierarchy_sha256'] or digest(directory / 'exposure.json') != exposures['sha256']:
            raise RuntimeError('Post modified original hierarchy or frozen pretrained exposure')
        freeze = helper.post_freeze_audit(before, after, anchors, scaffold, skybox)
        write(stage_dir / 'post-freeze-audit.json', freeze)
        receipt.update(status='completed', source_kind='post_optimized', hierarchy=after.summary(),
                       hierarchy_sha256=digest(after.path), frozen_fields=freeze, frozen_exposure=exposures,
                       optimizer_update_calls=args.post_iterations-1, requested_iterations=args.post_iterations,
                       numeric_audit=helper.hierarchy_numeric_audit(after), snapshot_boundary='Final backward/save I, return before optimizer step I; I-1 updates',
                       depth_loss_present_in_unchanged_post=False, exposure_optimizer_updates=0,
                       dimensionless_sampling=[.005, .1], pixel_conversion=config['limit_conversion'],
                       configuration_sha256=digest(config_path))
        write(directory / 'post-outcome.json', receipt)
    else:
        unoptimized = args.stage == 'merge-unoptimized'
        source_kind = 'unoptimized' if unoptimized else 'post_optimized'
        filename = 'hierarchy.hier' if unoptimized else 'hierarchy.hier_opt'
        receipts = []
        for chunk in chunks:
            stage_receipt = read(root / 'trained_chunks' / chunk / ('creator-outcome.json' if unoptimized else 'post-outcome.json'))
            path = root / 'trained_chunks' / chunk / filename
            if stage_receipt['status'] != 'completed' or stage_receipt['source_kind'] != source_kind or digest(path) != stage_receipt['hierarchy_sha256']:
                raise RuntimeError('Require all real completed matching merge inputs; no fake hier_opt')
            receipts.append({'chunk': chunk, 'path': str(path), 'sha256': digest(path), 'source_kind': source_kind})
        stem = 'merge-unoptimized' if unoptimized else 'merge'
        destination = root / ('merged-unoptimized.hier' if unoptimized else 'merged.hier')
        stage_dir = root / (stem + '-stage')
        stage_dir.mkdir(exist_ok=False)
        executable = variant['executable']['path'] if unoptimized else str(EXES / 'GaussianHierarchyMerger.exe')
        command = [str(executable), str(root / 'trained_chunks'), '0', str(inputs / 'chunks'), str(destination)] + list(chunks)
        receipt = execute(command, stage_dir, root / (stem + '-outcome.json'))
        h = codec.read_hierarchy(destination)
        receipt.update(status='completed', source_kind=source_kind, hierarchy=h.summary(), hierarchy_sha256=digest(destination),
                       input_hierarchies=receipts, chunk_order=list(chunks), chunk_centers={name: c['center'] for name, c in chunks.items()},
                       merger_executable_sha256=digest(executable), input_filename_only_variant=variant if unoptimized else None,
                       skybox_in_tree=0, shared_background_skybox=str(scaffold), shared_background_skybox_count=skybox,
                       ownership=config['merge_ownership'], numeric_audit=helper.hierarchy_numeric_audit(h),
                       configuration_sha256=digest(config_path))
        write(root / (stem + '-outcome.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

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

ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
OUTPUT = ROOT / 'outputs/large-scene-gs-001'
REPO = ROOT / 'repos/hierarchical-3d-gaussians'
NATIVE_REPO = ROOT / 'repos/gaussian-hierarchy-large-gs'
BUILD = OUTPUT / 'hierarchy-build'
EXES = NATIVE_REPO / 'build-local/Release'
COMMIT = '596dc7081ee74e57a66502f4746328911f763b34'
NATIVE_COMMIT = '677c8553dc64dfd62c272eca94a291a277733113'
DRIVER_VERSION = 5
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


BINDING_PATH = OUTPUT / 'official-hierarchy/small-city-hierarchy-006/binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
BINDING_SHA = '1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'
BINDER_PATH = ROOT / 'scripts/large_scene_gs/bind_small_city_lineage_006.py'
BINDER_SNAPSHOT = BINDING_PATH.parent / 'binder-source.py'
BINDER_SHA = 'c72fea79bb3b083ead1dfa58e9cf7e7d0134842d764dd3c5482162448dc4db9e'
FIXED_RUN = 'small-city-hierarchy-006'
ORDER = ('0_0', '0_1', '1_0', '1_1')
ACTIVE_STAGE = OUTPUT / 'official-hierarchy-stage-supervision-006/active-stage.lock.json'
SUPERVISOR_SOURCE = ROOT / 'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
POST64_SOURCE = ROOT / 'scripts/large_scene_gs/run_small_city_post_preflight_006.py'
PROCESS_HELPER = ROOT / 'scripts/large_scene_gs/observe_small_city_process_roles.py'
PROCESS_HELPER_SHA = 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a'


def process_birth(pid, owned_handle=None):
    """Read actual Windows FILETIME, protecting PID reuse without touching CUDA."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = wintypes.HANDLE(int(owned_handle)) if owned_handle is not None else kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError('Actual locked process handle unavailable: ' + str(pid))
    try:
        c, e, k, u = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, c, e, k, u):
            raise RuntimeError('Actual locked process birth unavailable')
        value = (int(c.dwHighDateTime)<<32) | int(c.dwLowDateTime)
        return {'pid':pid, 'creation_FILETIME_100ns':value, 'creation_Unix_ms':(value-116444736000000000)//10000}
    finally:
        if owned_handle is None:
            kernel.CloseHandle(handle)


def gpu_process_roles(rows, argv_parser):
    """Pure exact-argument classification used by actual and CPU negative controls."""
    owners = []
    wrappers = {'run_small_city_stage.py', 'run_small_city_observation_001.py',
                'run_small_city_hierarchy.py', 'run_small_city_hierarchy_006.py',
                'run_small_city_post_preflight_001.py', 'run_small_city_post_preflight_006.py',
                'supervise_small_city_stage.py', 'supervise_small_city_observation_001.py',
                'supervise_small_city_hierarchy_stage_002.py', 'supervise_small_city_hierarchy_stage_006.py'}
    for row in rows:
        argv = argv_parser(row.get('CommandLine') or '')
        names = {Path(a).name.lower() for a in argv[1:]}
        native = row['Name'].lower().startswith('gaussianhierarchy')
        trainer = bool(names & {'train_coarse.py', 'train_single.py', 'train_post.py', 'observe_small_city_train_single_001.py'})
        evaluator = any(n.startswith('evaluate_') and n.endswith('.py') for n in names) and ('--gpu' in argv or 'run_local.py' in names)
        wrapper = '--execute' in argv and bool(names & wrappers)
        if native or trainer or evaluator or wrapper:
            owners.append(row)
    return owners


def no_conflicting_gpu_stage():
    """Actual exact argv + locked birth/ancestor guard; CPU imports never acquire GPU."""
    if digest(PROCESS_HELPER) != PROCESS_HELPER_SHA:
        raise RuntimeError('Exact Windows argv helper changed')
    parser = load_module('_smallcity006_process_argv', PROCESS_HELPER).windows_argv
    query = "Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|GaussianHierarchy'} | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Compress"
    raw = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],
                         capture_output=True,text=True,check=True,timeout=30,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    rows = json.loads(raw) if raw else []
    rows = rows if isinstance(rows,list) else [rows]
    owners = gpu_process_roles(rows, parser)
    bypid = {int(r['ProcessId']):r for r in rows}
    allowed = set()
    lock = read(ACTIVE_STAGE) if ACTIVE_STAGE.exists() else None
    current = os.getpid()
    current_row = bypid.get(current)
    if current_row is None:
        raise RuntimeError('Cannot observe actual current006 CPU process')
    current_argv = parser(current_row.get('CommandLine') or '')
    current_names = {Path(a).name.lower() for a in current_argv[1:]}
    is_executing006 = '--execute' in current_argv and bool(current_names & {
        'run_small_city_hierarchy_006.py','run_small_city_post_preflight_006.py','supervise_small_city_hierarchy_stage_006.py'})
    if is_executing006:
        if lock is None:
            raise RuntimeError('006 execution lacks active supervised ownership lock')
        supervisor_pid = lock['supervisor_pid']
        keys=('pid','creation_FILETIME_100ns','creation_Unix_ms')
        if process_birth(supervisor_pid) != {k:lock['supervisor_birth'][k] for k in keys}:
            raise RuntimeError('006 supervisor PID birth differs from lock')
        supervisor = bypid.get(supervisor_pid)
        if supervisor is None or parser(supervisor['CommandLine']) != lock['supervisor_command_argv']:
            raise RuntimeError('006 actual supervisor command differs from active lock')
        if lock['supervisor_source_sha256'] != digest(SUPERVISOR_SOURCE):
            raise RuntimeError('006 ownership source changed')
        wrapper = lock['wrapper_command_argv']
        target = Path(wrapper[3]) if len(wrapper)>3 else Path('invalid')
        if (wrapper[:3] != [str(TRAINING_PYTHON),'-u',str(ENV_LAUNCHER)]
                or target not in [ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py',POST64_SOURCE]
                or '--execute' not in wrapper or lock['stage'] not in ['creator','post','merge-unoptimized','merge','post64']):
            raise RuntimeError('006 lock wrapper/stage identity differs')
        if target == POST64_SOURCE and lock['stage'] != 'post64':
            raise RuntimeError('006 short post may only own post64 stage')
        if target != POST64_SOURCE and lock['stage'] == 'post64':
            raise RuntimeError('006 formal workflow cannot claim post64 stage')
        # run_local.py forwards [PYTHON, DRIVER, remainingargs]; it does not forward its -u.
        legal_tails = [wrapper[1:], [str(target),*wrapper[4:]], ['-u',str(target),*wrapper[4:]],
                       lock['supervisor_command_argv'][1:]]
        python_paths = {os.path.normcase(str(TRAINING_PYTHON.resolve())),
                        os.path.normcase(str(Path(getattr(sys,'_base_executable',sys.executable)).resolve()))}
        chain=[]; cursor=current
        while cursor in bypid and cursor not in chain:
            chain.append(cursor)
            if cursor == supervisor_pid:
                break
            cursor=int(bypid[cursor]['ParentProcessId'])
        if supervisor_pid not in chain:
            raise RuntimeError('006 executing wrapper is not in locked supervisor descendant chain')
        if current != supervisor_pid:
            child_birth=lock.get('actual_child_birth')
            if (child_birth is None or child_birth['pid'] not in chain
                    or process_birth(child_birth['pid']) != {k:child_birth[k] for k in keys}):
                raise RuntimeError('006 Popen child birth or exact descendant chain differs')
        # Include only the supervisor's own venv shim ancestors with the same exact command.
        cursor=int(supervisor['ParentProcessId'])
        while cursor in bypid:
            argv=parser(bypid[cursor].get('CommandLine') or '')
            if argv[1:] != lock['supervisor_command_argv'][1:]:
                break
            chain.append(cursor); cursor=int(bypid[cursor]['ParentProcessId'])
        for pid in chain:
            argv=parser(bypid[pid].get('CommandLine') or '')
            if (not argv or os.path.normcase(str(Path(argv[0]).resolve())) not in python_paths
                    or argv[1:] not in legal_tails):
                raise RuntimeError('006 descendant argv/python path outside exact active ownership chain')
            allowed.add(pid)
    conflicts = [r for r in owners if int(r['ProcessId']) not in allowed]
    if conflicts:
        raise RuntimeError('Actual conflicting GPU-stage process: '+json.dumps(conflicts,ensure_ascii=False))
    if lock is not None and not is_executing006:
        raise RuntimeError('Another006 active ownership lock exists; CPU dispatch gate rejects')
    return {'queried_utc':datetime.now(timezone.utc).isoformat(), 'actual_snapshot':rows,
            'actual_conflicting_gpu_processes':[], 'allowed_current_supervised_ancestor_PIDs':sorted(allowed),
            'CUDA_query_performed':False, 'GPU_started':False}


def fixed_sources():
    """CPU-only immutable mixed-lineage contract; never authorizes CUDA."""
    for p, expected in [(BINDING_PATH, BINDING_SHA), (BINDER_PATH, BINDER_SHA), (BINDER_SNAPSHOT, BINDER_SHA)]:
        if digest(p) != expected:
            raise RuntimeError('Actual006 binder source or lineage receipt changed')
    b = read(BINDING_PATH)
    if (b['status'] != 'actual_complete_mixed_local_training_lineage_CPU_bound'
            or b['formal_observation_reclassification'] is not False
            or b['training_lineage_kind'] != 'three_legacy_local_plus_one_observed_local_official_training'
            or [r['chunk'] for r in b['ordered_chunks']] != list(ORDER)
            or b['ordered_chunks'][-1]['training_source_kind'] != 'observed_local_official_training'
            or b['ordered_chunks'][-1]['formal_training_run'] is not False):
        raise RuntimeError('006 actual mixed training origin contract differs')
    binder = load_module('_smallcity006_actual_CPU_binder', BINDER_SNAPSHOT)
    binder.ROOT, binder.BASE = ROOT, OUTPUT
    # This module is loaded from its audit snapshot, so restore all actual path globals.
    binder.LOCK = OUTPUT / 'official-hierarchy/small-city-hierarchy-006'
    binder.OLD = OUTPUT / 'official-hierarchy/small-city-hierarchy-005'
    binder.INPUT = OUTPUT / 'official-inputs/small-city-training-002'
    binder.OBS = OUTPUT / 'official-training-observation' / binder.OBS_RUN
    binder.SUP = OUTPUT / 'official-training-observation-supervision' / binder.OBS_RUN
    binder.PLAN = OUTPUT / 'official-training-observation-plans' / (binder.OBS_RUN + '-v3')
    binder.COARSE = OUTPUT / 'official-training' / binder.COARSE_RUN
    binder.LEGACY = {c: OUTPUT / 'official-training' / ('small-city-chunk' + c.replace('_', '') + '-i30000-r4-003') for c in ORDER[:3]}
    if binder.sources() != b['source_pins']:
        raise RuntimeError('Current actual006 source pins differ from bound bytes')
    proofs = {}
    _, contract = binder.observed_terminal(proofs)
    binder.observed_sources(contract)
    if proofs['observed_terminal_accounting'] != b['observed_terminal']:
        raise RuntimeError('Observed actual save/step/skip accounting changed')
    return binder, b


def lineage_gate(config, manifest, training, codec):
    """Recheck all five saved parameter arrays, raw finite/activations/Sky/exposures/source."""
    binder, binding = fixed_sources()
    if (config['training_lineage_binding_path'] != str(BINDING_PATH)
            or config['training_lineage_binding_sha256'] != BINDING_SHA
            or config['training_lineage_kind'] != binding['training_lineage_kind']
            or config['input_manifest_sha256'] != binder.PINS['manifest']):
        raise RuntimeError('006 configuration lost actual mixed training binding')
    # Exact pre-existing model receipt identities are independently recomputed rather than trusted.
    refs = [binding['coarse']] + binding['ordered_chunks']
    actual = []
    workflow = sys.modules[__name__]
    for ref in refs:
        directory = Path(ref['training_directory'])
        item = binder.bind_one(directory, ref['stage'], ref['chunk'], manifest, workflow, training, codec,
                               observed=ref['training_source_kind'] == 'observed_local_official_training')
        if item != ref:
            raise RuntimeError('Actual saved model/source/config/finite/Sky/exposure differs: ' + str(ref['chunk']))
        actual.append(item)
    input_split_audit(binder.INPUT, manifest)
    no_conflicting_gpu_stage()  # Exact006/legacy GPU roles; never rely on a versioned scan omitting this wrapper.
    return {'coarse': actual[0], 'ordered_chunks': actual[1:],
            'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,
            'all_five_saved_model_bindings_rechecked': True, 'GPU_started': False}


def completed_training(run_name, stage, chunk, config, manifest, training):
    """Compatibility interface with explicit origin, not formal-status relabelling."""
    _, binding = fixed_sources()
    ref = binding['coarse'] if stage == 'coarse' else next(r for r in binding['ordered_chunks'] if r['chunk'] == chunk)
    directory = Path(ref['training_directory'])
    if directory.name != run_name or stage != ref['stage']:
        raise RuntimeError('006 run mapping differs from immutable actual lineage')
    for p, expected in [(directory / 'outcome.json', ref['training_receipt']['sha256']),
                        (directory / 'configuration.json', ref['configuration']['sha256']),
                        (directory / 'model-audit.json', ref['model_audit_sha256']),
                        (Path(ref['model']['path']), ref['model']['sha256'])]:
        if digest(p) != expected:
            raise RuntimeError('Bound training model or receipt changed')
    return Path(ref['model']['path']).parent, read(directory / 'outcome.json'), read(directory / 'configuration.json')


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
    execution_source_gate()
    no_conflicting_gpu_stage()
    binary_sha = digest(command[0])
    receipt = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'command_argv': command, 'cwd': str(REPO), 'log_path': str(directory / 'stage.log'),
               'wrapper_pid': os.getpid(), 'executable_sha256': binary_sha,
               'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA}
    write(receipt_path, receipt)
    start = time.perf_counter()
    with (directory / 'stage.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['actual_child_pid'] = child.pid
        receipt['actual_child_birth'] = process_birth(child.pid, owned_handle=child._handle)
        receipt['actual_child_birth_source'] = 'Popen-owned handle GetProcessTimes, not inferred from a PID-only receipt'
        write(receipt_path, receipt)
        code = child.wait()
    receipt.update(status='process_exited_zero' if code == 0 else 'failed', exit_code=code,
                   process_seconds=time.perf_counter()-start,
                   process_ended_utc=datetime.now(timezone.utc).isoformat(),
                   actual_child_poll_exit_code=child.poll(), executable_after_sha256=digest(command[0]))
    write(receipt_path, receipt)
    if code or digest(command[0]) != binary_sha:
        if not code:
            receipt.update(status='failed_executable_source_changed', exit_code=None)
            write(receipt_path, receipt)
            raise RuntimeError('Native/training executable bytes changed during stage')
        raise SystemExit(code)
    execution_source_gate()
    no_conflicting_gpu_stage()
    return receipt


def execution_source_gate():
    """CPU current bytes check before/after native calls; never mutate official files."""
    root = OUTPUT / 'official-hierarchy' / FIXED_RUN
    config = read(root / 'configuration.json')
    if (digest(ROOT / 'scripts/large_scene_gs/run_small_city_hierarchy_006.py') != config['driver_sha256']
            or digest(root / 'driver-source.py') != config['driver_sha256']
            or config['training_lineage_binding_sha256'] != BINDING_SHA):
        raise RuntimeError('Frozen006 driver/config lineage changed')
    for row in config['source_records']:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Frozen006 official native/post runtime source changed')
    for row in config['frozen_helper_records']:
        if digest(row['snapshot_path']) != row['sha256']:
            raise RuntimeError('Frozen006 CPU helper changed')
    fixed_sources()
    return {'source_changed': False, 'configuration_sha256': digest(root / 'configuration.json')}


def creator_gate(chunk, config, manifest, training, codec, helper, camera_archives):
    """Require an actual completed006 Creator, exact argv and immutable source lineage."""
    root = OUTPUT / 'official-hierarchy' / FIXED_RUN
    directory = root / 'trained_chunks' / chunk
    for name in ('creator-outcome.json', 'hierarchy.hier', 'anchors.bin', 'exposure.json'):
        if not (directory / name).is_file():
            raise RuntimeError('Actual006 Creator stage missing: ' + chunk + '/' + name)
    receipt = read(directory / 'creator-outcome.json')
    _, binding = fixed_sources()
    ref = next(r for r in binding['ordered_chunks'] if r['chunk'] == chunk)
    saved = Path(ref['model']['path']).parent
    scaffold = Path(binding['coarse']['model']['path']).parent
    expected_command = [str(EXES / 'GaussianHierarchyCreator.exe'), str(saved / 'point_cloud.ply'),
                        str(root / 'creator-camera-archives' / chunk), str(directory), str(scaffold)]
    expected = {'status': 'completed', 'exit_code': 0, 'source_kind': 'unoptimized',
                'command_argv': expected_command, 'configuration_sha256': digest(root / 'configuration.json'),
                'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,
                'source_training_receipt': read(ref['training_receipt']['path']),
                'training_origin': ref, 'training_source_kind': ref['training_source_kind'],
                'formal_observation_reclassification': False,
                'creator_train_camera_archive': camera_archives[chunk]}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Actual006 Creator argv/source/config/origin receipt contract differs')
    if (receipt.get('actual_child_poll_exit_code') != 0 or not receipt.get('actual_child_pid')
            or receipt.get('executable_sha256') != digest(EXES / 'GaussianHierarchyCreator.exe')
            or receipt.get('executable_after_sha256') != receipt['executable_sha256']
            or receipt['input_model']['sha256'] != ref['model']['sha256']
            or receipt['copied_exposure_sha256'] != ref['exposure']['sha256']
            or receipt['source_exposure'] != ref['exposure']
            or camera_archives[chunk]['heldout_pose_count'] != 0):
        raise RuntimeError('Actual006 Creator child/model/exposure/archive identity differs')
    for row in camera_archives[chunk]['files']:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Actual006 train-only camera archive changed')
    if (digest(directory / 'hierarchy.hier') != receipt['hierarchy_sha256']
            or digest(directory / 'anchors.bin') != receipt['anchor_sha256']
            or digest(directory / 'exposure.json') != receipt['copied_exposure_sha256']):
        raise RuntimeError('Actual006 Creator output bytes changed')
    before = codec.read_hierarchy(directory / 'hierarchy.hier')
    anchors = codec.read_anchors(directory / 'anchors.bin', len(before.positions))
    if (before.summary() != receipt['hierarchy'] or len(anchors) != receipt['anchor_count']
            or helper.hierarchy_numeric_audit(before) != receipt['numeric_audit']):
        raise RuntimeError('Actual006 Creator decoded hierarchy/anchors/numeric evidence differs')
    sky = shared_sky_audit(scaffold, OUTPUT / 'official-inputs/small-city-training-002/chunks' / chunk,
                           saved, training, codec)
    if sky != receipt['shared_skybox']:
        raise RuntimeError('Actual006 Creator shared100k sky binding differs')
    return receipt


def post_gate(chunk, config, manifest, training, codec, helper, camera_archives):
    creator = creator_gate(chunk, config, manifest, training, codec, helper, camera_archives)
    root = OUTPUT / 'official-hierarchy' / FIXED_RUN
    directory = root / 'trained_chunks' / chunk
    receipt = read(directory / 'post-outcome.json')
    scaffold = Path(read(BINDING_PATH)['coarse']['model']['path']).parent
    command = post_command(directory, OUTPUT / 'official-inputs/small-city-training-002/chunks' / chunk,
                           scaffold, manifest, 15000)
    expected = {'status': 'completed', 'exit_code': 0, 'source_kind': 'post_optimized',
                'requested_iterations': 15000, 'optimizer_update_calls': 14999,
                'optimizer_call_count_source_derived': True,
                'command_argv': command, 'configuration_sha256': digest(root / 'configuration.json'),
                'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,
                'source_training_receipt': creator['source_training_receipt'], 'input_model': creator['input_model'],
                'training_origin': creator['training_origin'], 'training_source_kind': creator['training_source_kind'],
                'formal_observation_reclassification': False, 'exposure_optimizer_updates': 0,
                'depth_loss_present_in_unchanged_post': False}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Actual006 formal15000 post source/argv/budget/config contract differs')
    if (receipt.get('actual_child_poll_exit_code') != 0 or not receipt.get('actual_child_pid')
            or digest(directory / 'hierarchy.hier_opt') != receipt['hierarchy_sha256']
            or receipt['frozen_exposure']['sha256'] != creator['copied_exposure_sha256']):
        raise RuntimeError('Actual006 post real child/hierarchy/frozen exposure mismatch')
    text = (directory / 'post-stage/stage.log').read_text(encoding='utf-8', errors='replace')
    if '[ITER 15000] Saving Gaussians' not in text or 'Training complete.' not in text:
        raise RuntimeError('Actual006 post15000 final save/normal return absent')
    before = codec.read_hierarchy(directory / 'hierarchy.hier')
    after = codec.read_hierarchy(directory / 'hierarchy.hier_opt', expected_unowned_tail=100000)
    anchors = codec.read_anchors(directory / 'anchors.bin', len(before.positions))
    if (after.summary() != receipt['hierarchy']
            or helper.hierarchy_numeric_audit(after) != receipt['numeric_audit']
            or helper.post_freeze_audit(before, after, anchors, scaffold, 100000) != receipt['frozen_fields']):
        raise RuntimeError('Actual006 post decoded tree/anchors/singleSky numeric contract differs')
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
    parser.add_argument('--run', default=FIXED_RUN, type=identifier)
    parser.add_argument('--input', default='small-city-training-002', type=identifier)
    parser.add_argument('--coarse-run', default='small-city-coarse-i30000-r4-003', type=identifier)
    parser.add_argument('--chunk-run', action='append', required=True, help='One CHUNK=local-complete-run mapping per chunk')
    parser.add_argument('--chunk', choices=['0_0', '0_1', '1_0', '1_1'])
    parser.add_argument('--post-iterations', type=int, default=15000)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--revise-unexecuted-plan', action='store_true')
    args = parser.parse_args()
    if args.run == 'small-city-hierarchy-001' or args.input == 'small-city-training-001':
        raise RuntimeError('001 is a historical CPU plan with incorrect Reader test-list locations; use corrected 002 without overwriting 001')
    if args.post_iterations != 15000:
        raise ValueError('006 formal post budget is exactly15000; short64 uses independent post006 namespace')
    if args.run != FIXED_RUN or args.input != 'small-city-training-002':
        raise RuntimeError('006 driver is locked to its independent006 namespace and002 input')
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
    _, actual_binding = fixed_sources()
    expected_mappings = {r['chunk']: Path(r['training_directory']).name for r in actual_binding['ordered_chunks']}
    if mappings != expected_mappings or args.coarse_run != Path(actual_binding['coarse']['training_directory']).name:
        raise RuntimeError('006 CLI mappings differ from actual completed mixed lineage')
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
    if list(chunks) != list(ORDER):
        raise RuntimeError('Original chunk order changed')
    fx = camera_archives[next(iter(chunks))]['original_fx_fy'][0] / 4
    config = {'driver_version': DRIVER_VERSION, 'driver_sha256': digest(Path(__file__)),
              'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,
              'training_lineage_kind': actual_binding['training_lineage_kind'], 'formal_observation_reclassification': False,
              'training_origin_records': actual_binding['ordered_chunks'],
              'post_optimizer_call_accounting': '14999 derived from unchanged unconditional post step before finalI15000 save/return; not instrumented per-Gaussian nonzero updates',
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
    lineage = lineage_gate(config, manifest, training, codec)
    write(config_path, config)
    shutil.copyfile(Path(__file__), root / 'driver-source.py')
    if args.stage == 'plan':
        preview = root / 'trained_chunks/0_0'
        command = post_command(preview, inputs / 'chunks/0_0', Path(actual_binding['coarse']['model']['path']).parent, manifest, args.post_iterations)
        pure = load_module('_smallcity_post_pure_args', REPO / 'arguments/__init__.py')
        p = argparse.ArgumentParser()
        pure.ModelParams(p); pure.OptimizationParams(p); pure.PipelineParams(p)
        p.add_argument('--save_iterations', nargs='+', type=int); p.add_argument('--disable_viewer', action='store_true')
        effective = vars(p.parse_args(command[3:]))
        if effective['train_test_exp'] or effective['resolution'] != 4:
            raise RuntimeError('Post holdout/r4 CLI contract failed')
        plan = {'status': 'CPU_prepared_no_GPU_execution', 'gpu_started': False,
                'configuration': str(config_path), 'configuration_sha256': digest(config_path),
                'dependencies': {'coarse': args.coarse_run, 'chunks': mappings, 'policy': 'Actual3legacy+1observed complete finite local30k,r4 models; origin remains distinct, no formal relabelling', 'all_five_actual_bindings': lineage},
                'camera_archives': camera_archives, 'post_command_preview': command, 'effective_post_args': effective,
                 'input_split_audit': split_audit,
                'post_depth_loss_present': False, 'post_exposure_optimizer_steps': 0,
                'paired_outputs': {'unoptimized': 'merged-unoptimized.hier from real hierarchy.hier via filename-only variant',
                  'post_optimized': 'merged.hier from real hierarchy.hier_opt via official Merger'},
                'binary_Creator_interface': 'If official save_pt emits point_cloud.bin (>8M): validate4+236P bytes/fields, pass absent point_cloud.ply fallback sentinel to unchanged Creator; create no fake PLY or hier_opt',
                'limits': config['limit_conversion'], 'system_drive_free_bytes': shutil.disk_usage(ROOT).free,
                'validation_scope': 'CPU actual all5 saved model/source/finite/Sky/exposure binding, five splits, train-only camera archives and CLI; no native Creator/post/Merger executed',
                'training_lineage_binding_path': str(BINDING_PATH), 'training_lineage_binding_sha256': BINDING_SHA,
                'creator_command_previews': {c: [str(EXES / 'GaussianHierarchyCreator.exe'), str(Path(r['model']['path']).parent / 'point_cloud.ply'), str(root / 'creator-camera-archives' / c), str(root / 'trained_chunks' / c), str(Path(actual_binding['coarse']['model']['path']).parent)] for c,r in zip(ORDER, actual_binding['ordered_chunks'])}}
        write(root / 'plan.json', plan)
        write(root / 'readiness.json', {'status': plan['status'], 'gpu_started': False,
              'driver_sha256': config['driver_sha256'], 'plan_sha256': digest(root / 'plan.json'),
              'actual_trained_dependencies_present': True, 'all_five_bindings_rechecked': True, 'training_lineage_binding_sha256': BINDING_SHA, 'future_runtime_audits': ['PLY/bin input/sky fields', 'actual exposure matrices', 'anchors and post sky half-roundtrip', 'paired identical ownership']})
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
        training_origin = next(r for r in actual_binding['ordered_chunks'] if r['chunk'] == args.chunk)
        exposure = Path(training_origin['exposure']['path'])
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
                       training_origin=training_origin, training_source_kind=training_origin['training_source_kind'],
                       formal_observation_reclassification=False,
                       copied_exposure_sha256=digest(directory / 'exposure.json'),
                       creator_train_camera_archive=camera_archives[args.chunk], shared_skybox=sky_audit,
                       numeric_audit=helper.hierarchy_numeric_audit(h), configuration_sha256=digest(config_path),
                       training_lineage_binding_path=str(BINDING_PATH), training_lineage_binding_sha256=BINDING_SHA)
        write(directory / 'creator-outcome.json', receipt)
    elif args.stage in ['post', 'audit-post']:
        creator = creator_gate(args.chunk, config, manifest, training, codec, helper, camera_archives)
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
        final_text = (stage_dir / 'stage.log').read_text(encoding='utf-8', errors='replace')
        if '[ITER 15000] Saving Gaussians' not in final_text or 'Training complete.' not in final_text:
            raise RuntimeError('Actual post15000 final save/normal return log absent')
        if digest(directory / 'hierarchy.hier') != creator['hierarchy_sha256'] or digest(directory / 'exposure.json') != exposures['sha256']:
            raise RuntimeError('Post modified original hierarchy or frozen pretrained exposure')
        freeze = helper.post_freeze_audit(before, after, anchors, scaffold, skybox)
        write(stage_dir / 'post-freeze-audit.json', freeze)
        receipt.update(status='completed', source_kind='post_optimized', hierarchy=after.summary(),
                       hierarchy_sha256=digest(after.path), frozen_fields=freeze, frozen_exposure=exposures,
                       source_training_receipt=creator['source_training_receipt'], input_model=creator['input_model'],
                       training_origin=creator['training_origin'], training_source_kind=creator['training_source_kind'],
                       formal_observation_reclassification=False, optimizer_call_count_source_derived=True,
                       optimizer_update_calls=args.post_iterations-1, requested_iterations=args.post_iterations,
                       numeric_audit=helper.hierarchy_numeric_audit(after), snapshot_boundary='Final backward/save I, return before optimizer step I; I-1 updates',
                       depth_loss_present_in_unchanged_post=False, exposure_optimizer_updates=0,
                       dimensionless_sampling=[.005, .1], pixel_conversion=config['limit_conversion'],
                       configuration_sha256=digest(config_path),
                       training_lineage_binding_path=str(BINDING_PATH), training_lineage_binding_sha256=BINDING_SHA)
        write(directory / 'post-outcome.json', receipt)
    else:
        unoptimized = args.stage == 'merge-unoptimized'
        source_kind = 'unoptimized' if unoptimized else 'post_optimized'
        filename = 'hierarchy.hier' if unoptimized else 'hierarchy.hier_opt'
        receipts = []
        for chunk in chunks:
            stage_receipt = (creator_gate if unoptimized else post_gate)(chunk, config, manifest, training, codec, helper, camera_archives)
            path = root / 'trained_chunks' / chunk / filename
            if stage_receipt['status'] != 'completed' or stage_receipt['source_kind'] != source_kind or digest(path) != stage_receipt['hierarchy_sha256']:
                raise RuntimeError('Require all real completed matching merge inputs; no fake hier_opt')
            receipts.append({'chunk': chunk, 'path': str(path), 'sha256': digest(path), 'source_kind': source_kind,
                             'source_stage_receipt_sha256': digest(root / 'trained_chunks' / chunk / ('creator-outcome.json' if unoptimized else 'post-outcome.json')),
                             'training_source_kind': stage_receipt['training_source_kind'],
                             'training_lineage_binding_sha256': BINDING_SHA})
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
                       configuration_sha256=digest(config_path),
                       training_lineage_binding_path=str(BINDING_PATH), training_lineage_binding_sha256=BINDING_SHA)
        write(root / (stem + '-outcome.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

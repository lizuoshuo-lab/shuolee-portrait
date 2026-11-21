"""Independent full-budget observation wrapper derived from the preserved original stage.
All official input/model validation is retained; actual child argv and observation-only scope are explicit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs' / 'large-scene-gs-001'
REPO = ROOT / 'repos' / 'hierarchical-3d-gaussians'
COMMIT = '596dc7081ee74e57a66502f4746328911f763b34'
ENV_LAUNCHER = ROOT / 'scripts/large_scene_gs/run_local.py'
TRAINING_PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def ply_vertices(path):
    """CPU memmap of scalar-property binary little-endian PLY."""
    import numpy as np
    types = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'uchar': 'u1',
             'uint8': 'u1', 'int': '<i4', 'uint': '<u4', 'short': '<i2', 'ushort': '<u2'}
    fields, count, vertex_element = [], None, False
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            tokens = line.decode('ascii').strip().split()
            if tokens[:1] == ['format'] and tokens[1] != 'binary_little_endian':
                raise ValueError('Expected binary little-endian PLY')
            if tokens[:1] == ['element']:
                vertex_element = tokens[1] == 'vertex'
                if vertex_element:
                    count = int(tokens[2])
                elif int(tokens[2]) != 0:
                    raise ValueError('Unexpected nonempty nonvertex PLY element')
            if tokens[:1] == ['property'] and vertex_element:
                if tokens[1] not in types:
                    raise ValueError('Unsupported PLY property')
                fields.append((tokens[2], types[tokens[1]]))
            if tokens == ['end_header']:
                offset = stream.tell()
                break
    dtype = np.dtype(fields)
    if count is None or Path(path).stat().st_size != offset + count * dtype.itemsize:
        raise ValueError('PLY payload size mismatch')
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))


def audit_model(saved, stage, data, scaffold):
    """Read saved scalar arrays on CPU; never load pickle/checkpoint tensors."""
    import numpy as np
    info = (saved / 'pc_info.txt').read_text(encoding='utf-8').strip()
    if int(info) != 100000:
        raise ValueError('Saved shared skybox must contain 100000 Gaussians')
    ply, binary = saved / 'point_cloud.ply', saved / 'point_cloud.bin'
    if ply.is_file():
        vertices = ply_vertices(ply)
        fields = vertices.dtype.names
        count = len(vertices)
        rest = len([field for field in fields if field.startswith('f_rest_')])
        degree = int(round((rest / 3 + 1)**.5 - 1))
        expected = 9 if stage == 'coarse' else 45
        if rest != expected:
            raise ValueError('Saved spherical harmonic layout mismatch')
        arrays = [(name, vertices[name]) for name in fields]
        xyz = np.stack([vertices[n] for n in ('x', 'y', 'z')], axis=1)
        rotations = np.stack([vertices[f'rot_{i}'] for i in range(4)], axis=1)
        format_name, parameter_path = 'binary_little_endian_PLY', ply
    elif binary.is_file() and stage == 'chunk':
        with binary.open('rb') as stream:
            count, = struct.unpack('<i', stream.read(4))
        if count <= 8_000_000 or binary.stat().st_size != 4 + 236 * count:
            raise ValueError('Official SH3 block-binary layout mismatch')
        offset, arrays = 4, []
        for name, width in [('xyz', 3), ('SH', 48), ('opacity', 1), ('log_scale', 3), ('rotation', 4)]:
            array = np.memmap(binary, dtype='<f4', mode='r', offset=offset, shape=(count, width))
            arrays.append((name, array))
            offset += 4 * count * width
        xyz, rotations = arrays[0][1], arrays[-1][1]
        fields, degree = [a[0] for a in arrays], 3
        format_name, parameter_path = 'official_float32_block_binary_SH3', binary
    else:
        raise ValueError('Saved model parameters missing')
    if not all(np.isfinite(array).all() for _, array in arrays):
        raise ValueError('Saved model has nonfinite parameters')
    min_rotation_norm = float(np.linalg.norm(rotations, axis=1).min())
    if min_rotation_norm <= 0:
        raise ValueError('Saved model has zero quaternion')
    model_files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in sorted(saved.iterdir()) if p.is_file()]
    audit = {'saved_format': format_name, 'gaussian_count': int(count),
             'all_saved_parameter_arrays_finite': True, 'nonzero_quaternions': True,
             'minimum_raw_quaternion_norm': min_rotation_norm, 'quaternion_order': 'wxyz',
             'max_SH_degree': degree, 'saved_fields': list(fields), 'skybox_prefix_count': 100000,
             'pc_info': info, 'parameter_path': str(parameter_path), 'files': model_files}
    if stage == 'coarse':
        original = ply_vertices(data / 'sparse/0/points3D.ply')
        original_xyz = np.stack([original[n] for n in ('x', 'y', 'z')], axis=1).astype(np.float32)
        if len(xyz) != len(original_xyz) + 100000:
            raise ValueError('Scaffold vertex count differs from original SfM plus skybox')
        delta = float(np.max(np.abs(xyz[100000:] - original_xyz)))
        audit.update(scaffold_SfM_position_max_abs_change=delta,
                     exact_SfM_position_freeze_passed=delta == 0.,
                     reference='Byte-preserved authoritative source points3D.ply rows converted to float32',
                     optimizer_note='Coarse xyz learning rate zero; final forward/backward iteration has no optimizer step')
        if delta != 0.:
            raise ValueError('Coarse scaffold XYZ changed')
    else:
        shared = ply_vertices(scaffold / 'point_cloud.ply')
        shared_xyz = np.stack([shared[n][:100000] for n in ('x', 'y', 'z')], axis=1)
        delta = float(np.max(np.abs(xyz[:100000] - shared_xyz)))
        audit.update(shared_skybox_position_max_abs_change=delta, exact_shared_skybox_XYZ_passed=delta == 0.)
        if delta != 0.:
            raise ValueError('Locked shared skybox XYZ changed')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['coarse', 'chunk'], required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--input', default='small-city-training-002')
    parser.add_argument('--chunk', choices=['0_0', '0_1', '1_0', '1_1'])
    parser.add_argument('--coarse-run')
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--resolution', type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument('--exposure', choices=['official', 'disabled'], default='official')
    parser.add_argument('--port', type=int, default=6022)
    parser.add_argument('--execute', action='store_true', help='Launch GPU only after supervising authorization')
    args = parser.parse_args()
    if (args.stage, args.run, args.input, args.chunk, args.coarse_run, args.iterations, args.resolution, args.exposure, args.port) != (
            'chunk', 'small-city-chunk11-observation-i30000-r4-001', 'small-city-training-002', '1_1',
            'small-city-coarse-i30000-r4-003', 30000, 4, 'official', 6024):
        raise ValueError('Only the separately authorized full-budget observation contract is supported')
    for name in (args.run, args.input, args.coarse_run):
        if name is not None and (not name.replace('-', '').isalnum() or len(name) > 100):
            raise ValueError('Invalid run/input identifier')
    if args.iterations < 2 or not 1024 <= args.port <= 65535:
        raise ValueError('Invalid iteration budget or viewer port')
    inputs = BASE / 'official-inputs' / args.input
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'ready_for_official_small_city_training_cpu_prepared' or manifest['official_commit'] != COMMIT:
        raise ValueError('SmallCity inputs not ready')
    if sha(inputs / 'frozen-split.json') != manifest['frozen_split_sha256']:
        raise ValueError('Frozen split changed')
    if manifest.get('test_list_relative_path') != 'sparse/0/test.txt' or not manifest.get('official_reader_actual_split_passed'):
        raise ValueError('Require repaired inputs with the actual official CPU reader split audit')
    reader_path = Path(manifest['official_reader_CPU_audit'])
    if sha(reader_path) != manifest['official_reader_CPU_audit_sha256']:
        raise ValueError('Actual official CPU reader receipt changed')
    reader = json.loads(reader_path.read_text(encoding='utf-8'))
    if reader['status'] != 'actual_official_reader_AST_CPU_passed' or reader['gpu_started'] or reader['CUDA_initialized']:
        raise ValueError('Official CPU reader readiness contract failed')
    for item in reader['protected_input_before_after_hashes']:
        if sha(item['path']) != item['before'] or item['before'] != item['after'] or not item['unchanged']:
            raise ValueError('Official CPU reader protected input changed')
    for item in reader['generated_PLY']:
        if sha(item['path']) != item['sha256'] or sha(item['source_points3D_bin']) != item['source_points3D_bin_sha256']:
            raise ValueError('Official CPU-derived PLY changed')
    copies = json.loads(Path(manifest['source_copy_audit']).read_text(encoding='utf-8'))
    for copy in copies:
        if sha(copy['destination']) != copy['destination_sha256']:
            raise ValueError('Derived authoritative metadata changed')
    if subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('Official source commit changed')
    for item in manifest['official_source_files']:
        path = Path(item['path'])
        canonical = subprocess.check_output(['git', '-C', str(REPO), 'show', f"{COMMIT}:{item['relative_path']}"])
        if sha(path) != item['sha256'] or path.read_bytes().replace(b'\r\n', b'\n') != canonical.replace(b'\r\n', b'\n'):
            raise ValueError('Official source bytes changed')
    data = inputs / 'aligned' if args.stage == 'coarse' else inputs / 'chunks' / str(args.chunk)
    entry = manifest if args.stage == 'coarse' else next((c for c in manifest['chunks'] if c['name'] == args.chunk), None)
    if entry is None:
        raise ValueError('Chunk required for chunk stage')
    train_names, test_names = entry['train_names'], entry['test_names']
    if set(train_names) & set(manifest['test_names']):
        raise ValueError('Training names overlap frozen test set')
    if (data / 'sparse/0/test.txt').read_text(encoding='utf-8').splitlines() != test_names:
        raise ValueError('Stage test.txt changed')
    runs = BASE / 'official-training'
    out = BASE / 'official-training-observation' / args.run
    if out.exists():
        raise ValueError('Run already exists; refusing overwrite')
    scaffold = None
    receipt_hash = None
    if args.stage == 'chunk':
        if not args.coarse_run:
            if args.execute:
                raise ValueError('Completed local coarse run required')
        else:
            receipt_path = runs / args.coarse_run / 'outcome.json'
            if not receipt_path.is_file():
                if args.execute:
                    raise ValueError('Future full local coarse run has not completed')
            else:
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                receipt_hash = sha(receipt_path)
                config = json.loads((runs / args.coarse_run / 'configuration.json').read_text(encoding='utf-8'))
                if (receipt['status'] != 'completed' or config['stage'] != 'coarse'
                        or config['input_manifest_sha256'] != sha(manifest_path)
                        or config['requested_iterations'] < 30000):
                    raise ValueError('Require completed >=30000-iteration local coarse with the same frozen SmallCity input')
                scaffold = Path(receipt['saved_model_directory'])
                audit = json.loads((runs / args.coarse_run / 'model-audit.json').read_text(encoding='utf-8'))
                if not audit.get('exact_SfM_position_freeze_passed') or audit['skybox_prefix_count'] != 100000:
                    raise ValueError('Coarse freeze/shared-skybox audit failed')
                for item in audit['files']:
                    if sha(item['path']) != item['sha256']:
                        raise ValueError('Scaffold file changed since audit')
    trainer = REPO / ('train_coarse.py' if args.stage == 'coarse' else 'train_single.py')
    command = [str(TRAINING_PYTHON), '-u', str(trainer), '-s', str(data), '-i', manifest['images_absolute'],
               '--alpha_masks', manifest['masks_absolute'], '-m', str(out / 'model'),
               '-r', str(args.resolution), '--data_device', 'cpu', '--eval',
               '--iterations', str(args.iterations), '--save_iterations', '-1', '--port', str(args.port)]
    if args.stage == 'coarse':
        command += ['--skybox_num', '100000', '--depth_l1_weight_init', '0', '--depth_l1_weight_final', '0']
    else:
        command += ['--depths', manifest['depths_absolute'], '--bounds_file', str(data),
                    '--scaffold_file', str(scaffold) if scaffold else '<completed-local-coarse-run-required>',
                    '--skybox_locked', '--disable_viewer']
    if args.exposure == 'disabled':
        command += ['--exposure_lr_init', '0', '--exposure_lr_final', '0']
    # Import the pure argparse module only. Importing trainers would initialize CUDA.
    spec = importlib.util.spec_from_file_location('official_pure_arguments', REPO / 'arguments/__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official_parser = argparse.ArgumentParser()
    module.ModelParams(official_parser)
    module.OptimizationParams(official_parser)
    module.PipelineParams(official_parser)
    official_parser.add_argument('--port', type=int, default=6009)
    official_parser.add_argument('--save_iterations', nargs='+', type=int, default=[7000, 30000])
    if args.stage == 'chunk':
        official_parser.add_argument('--disable_viewer', action='store_true')
    effective = vars(official_parser.parse_args(command[3:]))
    if effective['train_test_exp'] or not effective['eval'] or effective['data_device'] != 'cpu':
        raise ValueError('Holdout/device contract violated')
    authorized_command = ['python', str(ENV_LAUNCHER), str(Path(__file__)),
                          '--stage', args.stage, '--run', args.run, '--input', args.input,
                          '--iterations', str(args.iterations), '--resolution', str(args.resolution),
                          '--exposure', args.exposure, '--port', str(args.port), '--execute']
    if args.stage == 'chunk':
        authorized_command += ['--chunk', args.chunk, '--coarse-run',
                               args.coarse_run or '<future-full-local-coarse-run-required>']
    binding_path = ROOT / 'scripts/large_scene_gs/verify_small_city_stage_binding_004.py'
    if sha(binding_path) != '8da91ef734ca6579d94d0016b2863816a2bd7eedb752eaaafba8d8f626a59e90':
        raise ValueError('Fixed actual completed-training binding gate changed')
    binding_spec = importlib.util.spec_from_file_location('actual_fixed005_coarse_gate', binding_path)
    binding_module = importlib.util.module_from_spec(binding_spec)
    binding_spec.loader.exec_module(binding_module)
    actual_coarse_gate = binding_module.completed_gate('small-city-coarse-i30000-r4-003', 'coarse', None)
    if (actual_coarse_gate['model_parameter_sha256'] != 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31' or
            actual_coarse_gate['stage_outcome_sha256'] != '8214314c1371002928ecf327b348883b93a45fdb5a8fbae180b5f3452ec6f7d2'):
        raise ValueError('Actual full local coarse receipt/model bytes differ')
    oracle_path = BASE / 'official-training-observation-plans/small-city-initial-suspect-oracle-001/expected-raw.json'
    if sha(oracle_path) != '3fc724de94b8609f443ec63e9cdb6705a2083ca4785298af57d208064899c236':
        raise ValueError('Actual initialized-row raw oracle changed')
    observer = ROOT / 'scripts/large_scene_gs/observe_small_city_train_single_001.py'
    if sha(observer) != '0dfb6a869cf4751b76b3e1b263b95cc4fe0f7de3798612a312949ceaab848386':
        raise ValueError('Frozen observation driver source changed')
    adam = REPO / 'scene/OurAdam.py'
    adam_sha = 'fb552b3c31f07ef541dd71a2ac2b8fef72833423a5ca7d9fdf4f17cc22e1c9ba'
    canonical_adam = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:scene/OurAdam.py'])
    if sha(adam) != adam_sha or adam.read_bytes().replace(b'\r\n', b'\n') != canonical_adam.replace(b'\r\n', b'\n'):
        raise ValueError('Original official OurAdam bytes changed')
    official_command = command[:]
    baseline_path = BASE / 'official-training/small-city-chunk11-i30000-r4-003/configuration.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    expected_command = baseline['command_argv'][:]
    expected_command[expected_command.index('-m') + 1] = str(out / 'model')
    if official_command != expected_command:
        raise ValueError('Observed original official argv differs beyond isolated output model path')
    command = [str(TRAINING_PYTHON), '-u', str(observer), '--execute', '--directory', str(out / 'observer'), '--', *official_command[2:]]
    source_files = manifest['official_source_files'] + [{'path': str(adam), 'relative_path': 'scene/OurAdam.py', 'sha256': adam_sha}]
    configuration = {'official_commit': COMMIT, 'official_source_files': source_files,
                     'actual_coarse_fixed005_gate': actual_coarse_gate,
                     'initial_raw_oracle': str(oracle_path), 'initial_raw_oracle_sha256': sha(oracle_path),
                     'original_official_command_argv': official_command, 'actual_observation_child_argv': command,
                     'observer_source': str(observer), 'observer_source_sha256': sha(observer), 'OurAdam_source_sha256': adam_sha,
                     'diagnostic_scope': 'Read-only observation hooks, not formal completion or reproduction timing',
                     'formal_training_run': False, 'seed': 0, 'seed_source': 'Unchanged official safe_state(args.quiet)',
                     'baseline_failed_configuration': str(baseline_path), 'baseline_failed_configuration_sha256': sha(baseline_path),
                     'trainer_source_sha256': sha(trainer), 'launcher_source_sha256': sha(Path(__file__)),
                     'source_changed': False, 'command_argv': command, 'cwd': str(REPO),
                     'stage': args.stage, 'chunk': args.chunk, 'coarse_run': args.coarse_run,
                     'chunk_coarse_dependency_minimum_iterations': 30000 if args.stage == 'chunk' else None,
                     'coarse_outcome_sha256': receipt_hash, 'input_manifest_sha256': sha(manifest_path),
                     'official_reader_CPU_audit_sha256': sha(reader_path),
                     'official_test_path': str(data / 'sparse/0/test.txt'),
                     'official_test_sha256': sha(data / 'sparse/0/test.txt'),
                     'training_names': train_names, 'stage_test_names': test_names,
                     'frozen_global_test_names': manifest['test_names'], 'train_test_exp': False,
                     'requested_iterations': args.iterations, 'expected_optimizer_update_calls': args.iterations - 1,
                     'snapshot_timing': 'Save after forward/backward at I; official return before optimizer step at I; update calls I-1',
                     'effective_official_arguments': effective, 'resolution_pixels': [round(1024 / args.resolution), round(690 / args.resolution)],
                     'depth_scope': 'Disabled without global parameters' if args.stage == 'coarse' else 'Supplied unchanged RGB uint8 /65536 + chunk scale/offset; official reliability and exponential loss schedule',
                     'exposure_mode': args.exposure,
                     'exposure_optimizer_steps_expected': 0 if args.stage == 'coarse' else args.iterations - 1,
                     'exposure_parameter_updates_enabled': args.stage == 'chunk' and args.exposure == 'official',
                     'exposure_limit': 'Official chunk exposure defaults enabled; test exposure never fitted with train_test_exp=False, so raw test scores differ from paper exposure protocol',
                     'schedule': 'Official defaults retained including position max_steps=30000, densification and SH schedule; only requested iterations/resolution altered',
                     'authorized_launch_argv': authorized_command,
                     'environment_launcher': str(ENV_LAUNCHER), 'environment_launcher_sha256': sha(ENV_LAUNCHER),
                     'runtime_contract': 'GPU execution must use scripts/large_scene_gs/run_local.py and existing .venv-large-gs/cu128 extension PATH; direct general .venv or bundled Python is rejected',
                     'geometry': manifest['coordinates'], 'split_limitation': manifest['split_limitation'],
                     'author_pretrained_reproduction': False, 'system_drive_free_bytes': shutil.disk_usage(ROOT).free}
    if not args.execute:
        plan = BASE / 'official-training-observation-plans' / f'{args.run}-v2'
        plan.mkdir(parents=True, exist_ok=False)
        save(plan / 'configuration.json', configuration)
        shutil.copyfile(Path(__file__), plan / 'launcher-source.py')
        shutil.copyfile(ENV_LAUNCHER, plan / 'environment-launcher-source.py')
        shutil.copyfile(observer, plan / 'observer-source.py')
        shutil.copyfile(oracle_path, plan / 'initial-raw-oracle.json')
        save(plan / 'outcome.json', {'status': 'cpu_prepared_not_executed', 'gpu_started': False,
                                   'chunk_scaffold_dependency_resolved': scaffold is not None if args.stage == 'chunk' else True})
        print(json.dumps({'status': 'cpu_prepared_not_executed', 'plan': str(plan), 'gpu_started': False}))
        return
    if configuration['system_drive_free_bytes'] < 80 * 1024**3:
        raise ValueError('System-drive reserve below 80 GiB')
    expected_extensions = ROOT / 'outputs/large-scene-gs-001/hierarchy-build/python-extensions/cu128'
    expected_cache = ROOT / '.cache/large_gs_torch_extensions'
    normal = lambda value: os.path.normcase(str(Path(value).resolve()))
    python_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if (normal(sys.executable) != normal(TRAINING_PYTHON)
            or normal(os.environ.get('TORCH_EXTENSIONS_DIR', '')) != normal(expected_cache)
            or normal(expected_extensions) not in [normal(path) for path in python_paths if path]
            or os.environ.get('CUDA_HOME') != r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'):
        raise ValueError('Execute through scripts/large_scene_gs/run_local.py; expected existing CUDA/extension environment missing')
    configuration['runtime_environment_checked'] = {'python': sys.executable,
        'CUDA_HOME': os.environ['CUDA_HOME'], 'TORCH_EXTENSIONS_DIR': os.environ['TORCH_EXTENSIONS_DIR'],
        'official_cu128_extensions_in_PYTHONPATH': True}
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'configuration.json', configuration)
    shutil.copyfile(Path(__file__), out / 'launcher-source.py')
    shutil.copyfile(ENV_LAUNCHER, out / 'environment-launcher-source.py')
    (out / 'observer').mkdir()
    shutil.copyfile(observer, out / 'observer-source.py')
    source_dir = out / 'official-source'
    for item in source_files:
        dest = source_dir / item['relative_path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item['path'], dest)
    start = time.perf_counter()
    outcome = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
               'stage': args.stage, 'requested_iterations': args.iterations, 'log_path': str(out / 'training.log')}
    save(out / 'outcome.json', outcome)
    code = 1
    try:
        with (out / 'training.log').open('wb') as log:
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        observed = json.loads((out / 'observer/observer-outcome.json').read_text(encoding='utf-8'))
        outcome.update(exit_code=code, status='failed', observer_outcome=observed,
                       diagnostic_observation_only=True, formal_completed_model_claimed=False)
        if code == 86 and observed['status'] == 'diagnostic_stopped_at_first_observed_nonfinite':
            outcome.update(status='diagnostic_stopped_first_observed_nonfinite_no_repair')
        if code == 0:
            if observed['status'] != 'official_full30000_observation_returned_no_detected_nonfinite_not_formal' or observed['original_optimizer_call_count'] != 29999:
                raise ValueError('Actual instrumented original step count/full observation return failed')
            saved = out / 'model/point_cloud' / f'iteration_{args.iterations}'
            audit = audit_model(saved, args.stage, data, scaffold)
            exposure_path = out / 'model/exposure.json'
            exposure = json.loads(exposure_path.read_text(encoding='utf-8'))
            import numpy as np
            if set(exposure) != set(train_names) or not np.isfinite(np.asarray(list(exposure.values()))).all():
                raise ValueError('Exposure names/finiteness audit failed')
            identity = np.eye(3, 4, dtype=np.float32)
            identity_passed = all(np.array_equal(np.asarray(value, dtype=np.float32), identity)
                                  for value in exposure.values())
            if (args.stage == 'coarse' or args.exposure == 'disabled') and not identity_passed:
                raise ValueError('Unfitted or disabled exposure must remain exact identity')
            audit['exposure'] = {'path': str(exposure_path), 'sha256': sha(exposure_path),
                                 'names_count': len(exposure), 'equals_training_names': True,
                                 'frozen_test_overlap': len(set(exposure) & set(manifest['test_names'])),
                                 'train_test_exp': False, 'exact_identity': identity_passed,
                                 'optimizer_step_calls_expected': configuration['exposure_optimizer_steps_expected'],
                                 'parameter_updates_enabled': configuration['exposure_parameter_updates_enabled']}
            save(out / 'model-audit.json', audit)
            training_text = (out / 'training.log').read_text(encoding='utf-8', errors='replace')
            peaks = [int(value) for value in re.findall(r'(?:Peak memory[=:]|peak memory:)\s*(\d+)', training_text)]
            outcome.update(status='completed_observation_only_final_finite_not_formal', saved_model_directory=str(saved), gaussian_count=audit['gaussian_count'],
                           optimizer_update_calls=args.iterations - 1, training_image_updates=args.iterations - 1,
                           update_count_scope='Original Adam bound calls instrumented externally, one unchanged original call per observation; 29999 calls required',
                           logged_peak_allocated_bytes=max(peaks) if peaks else None,
                           memory_scope='Torch peak includes read-only observation temporaries/synchronization; not an equivalent memory or timing benchmark of failed003',
                           model_audit=str(out / 'model-audit.json'), model_parameter_sha256=sha(audit['parameter_path']),
                           point_cloud_sha256=sha(saved / 'point_cloud.ply') if (saved / 'point_cloud.ply').is_file() else None,
                           saved_storage_format=audit['saved_format'])
    except Exception as exc:
        outcome.update(status='failed', error_type=type(exc).__name__, error=str(exc), exit_code=code)
    finally:
        changed = any(sha(item['path']) != item['sha256'] for item in source_files) or sha(observer) != configuration['observer_source_sha256']
        configuration['source_changed'] = changed
        save(out / 'configuration.json', configuration)
        outcome['source_changed'] = changed
        if changed:
            outcome.update(status='failed_source_changed_no_formal_claim')
        outcome.update(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start)
        save(out / 'outcome.json', outcome)
        print(json.dumps(outcome), flush=True)
    if outcome['status'] != 'completed_observation_only_final_finite_not_formal':
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()

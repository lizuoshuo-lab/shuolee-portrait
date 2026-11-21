"""CPU audit of a completed official full SmallCity chunk with frozen skybox.

Accepts the actual official SH3 PLY or >8M float32 block-binary layout. Does not
load pickle/tensors, mutate models or source, or initialize CUDA.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
NAMES = {'0_0': 'small-city-chunk00-i30000-r4-003', '0_1': 'small-city-chunk01-i30000-r4-003',
         '1_0': 'small-city-chunk10-i30000-r4-003', '1_1': 'small-city-chunk11-i30000-r4-003'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(vertices, fields, count):
    return np.stack([vertices[field][:count] for field in fields], axis=1).astype(np.float32)


def bit_audit(actual, expected):
    actual, expected = np.ascontiguousarray(actual), np.ascontiguousarray(expected)
    return {'shape': list(actual.shape), 'float32_bitexact': bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
            'actual_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
            'expected_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
            'max_abs_change': float(np.max(np.abs(actual - expected)))}


def audit(chunk_name):
    run = NAMES[chunk_name]
    provenance_module_spec = importlib.util.spec_from_file_location('binding004', ROOT / 'scripts/large_scene_gs/verify_small_city_stage_binding_004.py')
    provenance_module = importlib.util.module_from_spec(provenance_module_spec)
    provenance_module_spec.loader.exec_module(provenance_module)
    fixed005_gate = provenance_module.completed_gate(run, 'chunk', chunk_name)
    out = BASE / 'official-training' / run
    target = BASE / 'recovery-audit' / f'{run}-completion-004'
    if target.exists():
        raise ValueError('Existing completion audit; refusing overwrite')
    outcome, config, saved_audit = (read(out / filename) for filename in ('outcome.json', 'configuration.json', 'model-audit.json'))
    coarse_dir = BASE / 'official-training/small-city-coarse-i30000-r4-003'
    coarse_outcome, coarse_config, coarse_audit = (read(coarse_dir / filename) for filename in ('outcome.json', 'configuration.json', 'model-audit.json'))
    manifest = BASE / 'official-inputs/small-city-training-002/manifest.json'
    supervisor = read(BASE / 'official-training-supervision' / run / 'supervisor-outcome.json')
    if (outcome['status'] != 'completed' or outcome['exit_code'] != 0 or config['requested_iterations'] != 30000
        or config['stage'] != 'chunk' or config['chunk'] != chunk_name or config['coarse_run'] != coarse_dir.name
        or config['resolution_pixels'] != [256, 172] or config['exposure_mode'] != 'official'
        or supervisor['status'] != 'completed' or supervisor['child_exit_code'] != 0
        or coarse_outcome['status'] != 'completed' or coarse_config['requested_iterations'] < 30000
        or config['input_manifest_sha256'] != sha(manifest) or coarse_config['input_manifest_sha256'] != sha(manifest)
        or config['coarse_outcome_sha256'] != sha(coarse_dir / 'outcome.json')):
        raise ValueError('Actual full budget/input/coarse/supervisor completion contract failed')
    for model_audit in (coarse_audit, saved_audit):
        for item in model_audit['files']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Completed saved model bytes changed')
    stage = ROOT / 'scripts/large_scene_gs/run_small_city_stage.py'
    if sha(stage) != config['launcher_source_sha256']:
        raise ValueError('Stage source changed')
    for item in config['official_source_files']:
        if sha(item['path']) != item['sha256'] or sha(out / 'official-source' / item['relative_path']) != item['sha256']:
            raise ValueError('Official live/snapshotted source changed')
    spec = importlib.util.spec_from_file_location('pure_local_stage', stage)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    coarse = module.ply_vertices(coarse_audit['parameter_path'])
    count = int(coarse_audit['skybox_prefix_count'])
    parameter_path = Path(saved_audit['parameter_path'])
    if parameter_path.suffix == '.ply':
        vertices = module.ply_vertices(parameter_path)
        model_count = len(vertices)
        if len([f for f in vertices.dtype.names if f.startswith('f_rest_')]) != 45:
            raise ValueError('Expected SH3 raw PLY layout')
        sky_groups = {name: rows(vertices, fields, count) for name, fields in [
            ('xyz', ['x', 'y', 'z']), ('log_scale', [f'scale_{i}' for i in range(3)]),
            ('quaternion_wxyz', [f'rot_{i}' for i in range(4)]), ('opacity_logit', ['opacity']),
            ('SH_DC', [f'f_dc_{i}' for i in range(3)])]}
        sky_groups['SH_rest'] = rows(vertices, [f'f_rest_{i}' for i in range(45)], count).reshape(count, 3, 15).transpose(0, 2, 1)
        raw_finite = {field: bool(np.isfinite(vertices[field]).all()) for field in vertices.dtype.names}
        scales = np.exp(rows(vertices, [f'scale_{i}' for i in range(3)], model_count))
        norms = np.linalg.norm(rows(vertices, [f'rot_{i}' for i in range(4)], model_count), axis=1)
        SH2_nonzero = sum(int(np.count_nonzero(vertices[f'f_rest_{channel * 15 + i}'])) for channel in range(3) for i in range(3, 8))
        SH3_nonzero = sum(int(np.count_nonzero(vertices[f'f_rest_{channel * 15 + i}'])) for channel in range(3) for i in range(8, 15))
    elif parameter_path.suffix == '.bin':
        with parameter_path.open('rb') as stream:
            model_count, = struct.unpack('<i', stream.read(4))
        if model_count <= 8_000_000 or parameter_path.stat().st_size != 4 + 236 * model_count:
            raise ValueError('Expected official >8M SH3 block-binary layout')
        offset, arrays = 4, {}
        for name, width in [('xyz', 3), ('SH', 48), ('opacity_logit', 1), ('log_scale', 3), ('quaternion_wxyz', 4)]:
            arrays[name] = np.memmap(parameter_path, mode='r', dtype='<f4', shape=(model_count, width), offset=offset)
            offset += 4 * model_count * width
        sky_groups = {name: np.asarray(array[:count]) for name, array in arrays.items() if name != 'SH'}
        full_SH = arrays['SH'].reshape(model_count, 16, 3)
        sky_groups['SH_DC'] = np.asarray(full_SH[:count, 0, :])
        sky_groups['SH_rest'] = np.asarray(full_SH[:count, 1:, :])
        raw_finite = {name: bool(np.isfinite(array).all()) for name, array in arrays.items()}
        scales = np.exp(arrays['log_scale'])
        norms = np.linalg.norm(arrays['quaternion_wxyz'], axis=1)
        SH2_nonzero, SH3_nonzero = int(np.count_nonzero(full_SH[:, 4:9, :])), int(np.count_nonzero(full_SH[:, 9:16, :]))
    else:
        raise ValueError('Unsupported saved model layout')
    groups = {}
    for name, fields in [('xyz', ['x', 'y', 'z']), ('log_scale', [f'scale_{i}' for i in range(3)]),
                         ('quaternion_wxyz', [f'rot_{i}' for i in range(4)]), ('opacity_logit', ['opacity']),
                         ('SH_DC', [f'f_dc_{i}' for i in range(3)])]:
        groups[name] = bit_audit(sky_groups[name], rows(coarse, fields, count))
    expected_rest = np.zeros((count, 15, 3), dtype=np.float32)
    expected_rest[:, :3, :] = rows(coarse, [f'f_rest_{i}' for i in range(9)], count).reshape(count, 3, 3).transpose(0, 2, 1)
    groups['SH_rest_SH1_to_SH3_zero_padded'] = bit_audit(sky_groups['SH_rest'], expected_rest)
    if (not all(value['float32_bitexact'] for value in groups.values()) or not all(raw_finite.values())
        or not np.isfinite(scales).all() or not (scales > 0).all() or not np.isfinite(norms).all() or not (norms > 0).all()
        or SH2_nonzero == 0 or SH3_nonzero == 0):
        raise ValueError('Frozen sky/activated parameters/actual higher-SH learning audit failed')
    chunks = BASE / 'official-inputs/small-city-training-002/chunks' / chunk_name
    center = np.asarray([float(value) for value in (chunks / 'center.txt').read_text().split()], dtype=np.float32)
    extent = np.asarray([float(value) for value in (chunks / 'extent.txt').read_text().split()], dtype=np.float32)
    distance = np.max(np.abs(rows(coarse, ['x', 'y', 'z'], len(coarse))[:, :2] - center[:2]), axis=1)
    selected = (distance > np.float32(.5) * extent[0]) & (distance < np.float32(1.5) * extent[0])
    selected[:count] = True
    initial_prediction = int(selected.sum()) + len(module.ply_vertices(chunks / 'sparse/0/points3D.ply'))
    text = (out / 'training.log').read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'Number of points at initialisation\s*:\s*(\d+)', text)
    progress = [(int(i), int(b), int(g)) for i, b, g in re.findall(r'(\d+)/(\d+)\s*\[[^\r\n]*?Size=(\d+)', text)]
    if len(matches) != 1 or int(matches[0]) != initial_prediction or '[ITER 30000] Saving Gaussians' not in text or not progress:
        raise ValueError('Actual initialization/full save log evidence failed')
    exposure = read(out / 'model/exposure.json')
    identity = np.eye(3, 4, dtype=np.float32)
    manifest_obj = read(manifest)
    chunk_manifest = next(c for c in manifest_obj['chunks'] if c['name'] == chunk_name)
    if set(exposure) != set(chunk_manifest['train_names']) or set(exposure) & set(manifest_obj['test_names']):
        raise ValueError('Train-only exposure names violated')
    spec_process = importlib.util.spec_from_file_location('pure_process_roles', ROOT / 'scripts/large_scene_gs/observe_small_city_process_roles.py')
    process_module = importlib.util.module_from_spec(spec_process)
    spec_process.loader.exec_module(process_module)
    snapshot = process_module.observe(run)
    # This observer classifies coarse and stage/supervisor scripts exactly.
    # Also independently reject any real train_single.py argv while auditing.
    single_trainers = [p for p in snapshot['actual_python_processes'] if
                       'train_single.py' in {Path(a).name.lower() for a in process_module.windows_argv(p.get('CommandLine') or '')[1:]}]
    if snapshot['related_training_processes'] or single_trainers:
        raise ValueError('Official training process is still live')
    report = {'status': 'completed_actual_full30000_chunk_model_sky_source_and_terminal_CPU_audit_passed',
              'observed_utc': datetime.now(timezone.utc).isoformat(), 'run': run, 'chunk': chunk_name,
              'requested_iterations': 30000, 'optimizer_update_calls': outcome['optimizer_update_calls'],
              'optimizer_count_scope': outcome['update_count_scope'], 'coarse_run': coarse_dir.name,
              'input_manifest_sha256': sha(manifest), 'coarse_outcome_sha256': sha(coarse_dir / 'outcome.json'),
              'coarse_model_sha256': coarse_outcome['model_parameter_sha256'],
              'stage_configuration_sha256': sha(out / 'configuration.json'), 'stage_outcome_sha256': sha(out / 'outcome.json'),
              'model_parameter_sha256': sha(parameter_path), 'saved_format': saved_audit['saved_format'],
              'script_sha256': sha(__file__), 'actual_initial_gaussians': int(matches[0]),
              'predicted_initial_gaussians': initial_prediction, 'actual_saved_gaussians': model_count,
              'logged_gaussian_count_min_max': [min(p[2] for p in progress), max(p[2] for p in progress)],
              'actual_SH2_nonzero_scalar_coefficients': SH2_nonzero, 'actual_SH3_nonzero_scalar_coefficients': SH3_nonzero,
              'SH_learning_evidence': 'Initial higher-degree coefficients zero by unchanged official initialization; final nonzero l=2 and l=3 coefficients, not merely SH3 storage layout.',
              'shared_sky_all_parameter_groups_bitexact': groups, 'all_raw_arrays_finite': raw_finite,
              'activated_scale_min_max_scene_units': [float(scales.min()), float(scales.max())],
              'float32_quaternion_norm_min_max': [float(norms.min()), float(norms.max())],
              'train_exposure_names': len(exposure), 'train_exposure_nonidentity_names': sum(not np.array_equal(np.asarray(v, dtype=np.float32), identity) for v in exposure.values()),
              'frozen_test_exposure_names': 0, 'train_views': len(chunk_manifest['train_names']), 'test_views': len(chunk_manifest['test_names']),
              'logged_torch_peak_allocated_bytes': outcome['logged_peak_allocated_bytes'], 'memory_scope': outcome['memory_scope'],
              'elapsed_seconds': outcome['elapsed_seconds'], 'CUDA_initialized_by_this_audit': False,
              'torch_imported_by_this_audit': 'torch' in sys.modules, 'actual_related_python_processes': [],
              'actual_single_trainer_processes': single_trainers, 'actual_python_process_snapshot': snapshot['actual_python_processes'],
              'all_four_chunks_complete': False, 'Creator_post_Merger_or_LOD_complete': False}
    report['fixed005_actual_completed_training_gate'] = fixed005_gate
    if report['torch_imported_by_this_audit']:
        raise ValueError('CPU audit unexpectedly imported torch')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'completion-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(__file__, target / 'source.py')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', choices=list(NAMES), required=True)
    args = parser.parse_args()
    report = audit(args.chunk)
    print(json.dumps({'status': report['status'], 'run': report['run'], 'G': report['actual_saved_gaussians'],
                      'model_sha': report['model_parameter_sha256']}), flush=True)


if __name__ == '__main__':
    main()

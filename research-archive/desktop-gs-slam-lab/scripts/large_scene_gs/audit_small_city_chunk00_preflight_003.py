"""CPU verification of the real I64 chunk00 run, including every frozen sky field.

Never executes CUDA or changes source, saved model, training input, or config.
The completion gate is a short-run readiness result, not formal chunk training.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
RUN = 'small-city-chunk-0-0-preflight-r4-003'
COARSE = 'small-city-coarse-i30000-r4-003'


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
    return {'shape': list(actual.shape), 'float32_bitexact': bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
            'actual_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
            'expected_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
            'max_abs_change': float(np.max(np.abs(actual - expected)))}


def main():
    out = BASE / 'official-training' / RUN
    target = BASE / 'recovery-audit' / f'{RUN}-completion-001'
    if target.exists():
        raise ValueError('Existing completion audit; refusing overwrite')
    outcome, config, saved_audit = (read(out / filename) for filename in ('outcome.json', 'configuration.json', 'model-audit.json'))
    coarse_dir = BASE / 'official-training' / COARSE
    coarse_outcome, coarse_config, coarse_audit = (read(coarse_dir / filename) for filename in ('outcome.json', 'configuration.json', 'model-audit.json'))
    manifest = BASE / 'official-inputs/small-city-training-002/manifest.json'
    if (outcome['status'] != 'completed' or outcome['exit_code'] != 0 or config['requested_iterations'] != 64
        or config['stage'] != 'chunk' or config['chunk'] != '0_0' or config['coarse_run'] != COARSE
        or coarse_outcome['status'] != 'completed' or coarse_config['requested_iterations'] < 30000
        or config['input_manifest_sha256'] != sha(manifest) or coarse_config['input_manifest_sha256'] != sha(manifest)):
        raise ValueError('Actual completion/input/dependency contract failed')
    for audit in (coarse_audit, saved_audit):
        for item in audit['files']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Completed saved model bytes changed')
    stage = ROOT / 'scripts/large_scene_gs/run_small_city_stage.py'
    if sha(stage) != config['launcher_source_sha256']:
        raise ValueError('Stage source changed')
    spec = importlib.util.spec_from_file_location('pure_local_stage', stage)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    coarse = module.ply_vertices(coarse_audit['parameter_path'])
    chunk = module.ply_vertices(saved_audit['parameter_path'])
    if len([f for f in coarse.dtype.names if f.startswith('f_rest_')]) != 9 or len([f for f in chunk.dtype.names if f.startswith('f_rest_')]) != 45:
        raise ValueError('Expected coarse SH1 / chunk SH3 raw PLY layout')
    count = int(coarse_audit['skybox_prefix_count'])
    groups = {}
    for name, fields in [('xyz', ['x', 'y', 'z']), ('log_scale', [f'scale_{i}' for i in range(3)]),
                         ('quaternion_wxyz', [f'rot_{i}' for i in range(4)]), ('opacity_logit', ['opacity']),
                         ('SH_DC', [f'f_dc_{i}' for i in range(3)])]:
        groups[name] = bit_audit(rows(chunk, fields, count), rows(coarse, fields, count))
    original_rest = rows(coarse, [f'f_rest_{i}' for i in range(9)], count).reshape(count, 3, 3)
    expected_rest = np.zeros((count, 3, 15), dtype=np.float32)
    expected_rest[:, :, :3] = original_rest
    observed_rest = rows(chunk, [f'f_rest_{i}' for i in range(45)], count).reshape(count, 3, 15)
    groups['SH_rest_SH1_to_SH3_zero_padded'] = bit_audit(observed_rest, expected_rest)
    if not all(value['float32_bitexact'] for value in groups.values()):
        raise ValueError('Shared sky parameter freeze failed')
    chunks = BASE / 'official-inputs/small-city-training-002/chunks/0_0'
    center = np.asarray([float(value) for value in (chunks / 'center.txt').read_text().split()], dtype=np.float32)
    extent = np.asarray([float(value) for value in (chunks / 'extent.txt').read_text().split()], dtype=np.float32)
    distance = np.max(np.abs(rows(coarse, ['x', 'y', 'z'], len(coarse))[:, :2] - center[:2]), axis=1)
    selected = (distance > np.float32(.5) * extent[0]) & (distance < np.float32(1.5) * extent[0])
    selected[:count] = True
    chunk_points = module.ply_vertices(chunks / 'sparse/0/points3D.ply')
    predicted_count = int(selected.sum()) + len(chunk_points)
    text = (out / 'training.log').read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'Number of points at initialisation\s*:\s*(\d+)', text)
    if len(matches) != 1 or int(matches[0]) != predicted_count or len(chunk) != predicted_count:
        raise ValueError('Actual initialization/saved G does not match completed-coarse source predicate')
    finite_fields = {f: bool(np.isfinite(chunk[f]).all()) for f in chunk.dtype.names}
    log_scales = rows(chunk, [f'scale_{i}' for i in range(3)], len(chunk))
    with np.errstate(over='raise', under='ignore', invalid='raise'):
        scales = np.exp(log_scales)
    rotations = rows(chunk, [f'rot_{i}' for i in range(4)], len(chunk))
    norms = np.linalg.norm(rotations, axis=1)
    if not all(finite_fields.values()) or not np.isfinite(scales).all() or not (scales > 0).all() or not np.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError('Raw/activated parameter validity failed')
    exposure = read(out / 'model/exposure.json')
    identity = np.eye(3, 4, dtype=np.float32)
    nonidentity = sum(not np.array_equal(np.asarray(value, dtype=np.float32), identity) for value in exposure.values())
    losses = [float(value) for value in re.findall(r'(?<!Depth )Loss=([0-9.eE+-]+)', text)]
    if not losses or not np.isfinite(losses).all():
        raise ValueError('Training photo-loss observations not finite')
    report = {'status': 'completed_actual_I64_chunk00_preflight_CPU_audit_passed',
              'observed_utc': datetime.now(timezone.utc).isoformat(), 'run': RUN, 'coarse_run': COARSE,
              'CUDA_initialized_by_this_audit': False, 'torch_imported_by_this_audit': 'torch' in sys.modules,
              'requested_iterations': 64, 'optimizer_update_calls': outcome['optimizer_update_calls'],
              'optimizer_count_scope': outcome['update_count_scope'],
              'coarse_requested_iterations': coarse_config['requested_iterations'],
              'input_manifest_sha256': sha(manifest), 'chunk_configuration_sha256': sha(out / 'configuration.json'),
              'chunk_outcome_sha256': sha(out / 'outcome.json'), 'chunk_model_sha256': sha(saved_audit['parameter_path']),
              'coarse_model_sha256': sha(coarse_audit['parameter_path']), 'script_sha256': sha(__file__),
              'shared_sky_all_parameter_groups_bitexact': groups,
              'actual_initial_gaussians': int(matches[0]), 'predicted_initial_gaussians_from_completed_coarse': predicted_count,
              'actual_saved_gaussians': len(chunk), 'scaffold_selected_count': int(selected.sum()),
              'new_chunk_SfM_initial_points': len(chunk_points), 'skybox_prefix_count': count,
              'all_raw_fields_finite': finite_fields, 'activated_scale_min_max_scene_units': [float(scales.min()), float(scales.max())],
              'float32_quaternion_norm_min_max': [float(norms.min()), float(norms.max())],
              'observed_photo_losses_finite': True,
              'train_exposure_names': len(exposure), 'train_exposure_nonidentity_names': nonidentity,
              'train_exposure_parameter_updates_enabled': config['exposure_parameter_updates_enabled'],
              'test_exposure_names': saved_audit['exposure']['frozen_test_overlap'],
              'logged_torch_peak_allocated_bytes': outcome['logged_peak_allocated_bytes'],
              'memory_scope': outcome['memory_scope'], 'elapsed_seconds': outcome['elapsed_seconds'],
              'full30000_training_complete': False, 'full_training_GPU_authorized_by_this_audit': False,
              'limitations': ['I64 precedes official densification start at iteration500; it does not measure full training memory growth.',
                'No merged boundary image or LOD quality assertion is possible from one short chunk preflight.',
                'Non-sky scaffold context parameters may update; only shared skybox is required to remain frozen.',
                'Logged torch allocated peak excludes native transient allocations and is not minimum hardware requirement.']}
    if report['torch_imported_by_this_audit']:
        raise ValueError('CPU audit unexpectedly imported torch')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'completion-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(__file__, target / 'source.py')
    print(json.dumps({'audit': str(target / 'completion-audit.json'), 'status': report['status'], 'G': len(chunk),
                      'logged_peak': outcome['logged_peak_allocated_bytes']}))


if __name__ == '__main__':
    main()

"""Read-only CPU diagnosis of the failed formal11 final PLY; never repairs or retries."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
RUN = 'small-city-chunk11-i30000-r4-003'
OUT = BASE / 'official-training' / RUN
TARGET = BASE / 'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-CPU-001'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(filename, expected):
    path = ROOT / 'scripts/large_scene_gs' / filename
    if sha(path) != expected:
        raise ValueError('Frozen CPU reader/process helper changed')
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(directory):
    return [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(directory.rglob('*')) if p.is_file()]


def rows(v, fields, count=100000):
    return np.stack([v[f][:count] for f in fields], axis=1).astype(np.float32)


def bit_audit(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return {'shape': list(a.shape), 'float32_bitexact': bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
            'actual_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'expected_sha256': hashlib.sha256(b.tobytes()).hexdigest()}


def encoded(value):
    number = float(value)
    if np.isfinite(number):
        return number
    if np.isnan(number):
        return 'NaN'
    return '+Infinity' if number > 0 else '-Infinity'


def main():
    if TARGET.exists():
        raise ValueError('Preserve existing failure diagnosis; refusing overwrite')
    old_inventory = inventory(OUT)
    outcome = read(OUT / 'outcome.json')
    config = read(OUT / 'configuration.json')
    if outcome['status'] != 'failed' or outcome['error'] != 'Saved model has nonfinite parameters' or outcome['exit_code'] != 0:
        raise ValueError('Actual failure evidence changed')
    stage = load('run_small_city_stage.py', 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704')
    observer = load('observe_small_city_process_roles.py', 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a')
    snapshot = observer.observe(RUN)
    scripts = {'train_single.py', 'train_coarse.py', 'run_small_city_stage.py', 'supervise_small_city_stage.py', 'run_small_city_full_pipeline_005.py'}
    active = [p for p in snapshot['actual_python_processes'] if
              {Path(a).name.lower() for a in observer.windows_argv(p.get('CommandLine') or '')[1:]} & scripts]
    if active:
        raise ValueError('Actual original trainer/controller/supervisor still exists; do not diagnose an actively written model')
    parameter_path = OUT / 'model/point_cloud/iteration_30000/point_cloud.ply'
    v = stage.ply_vertices(parameter_path)
    invalid_rows = np.zeros(len(v), dtype=bool)
    fields = {}
    for field in v.dtype.names:
        a = v[field]
        bad = ~np.isfinite(a)
        invalid_rows |= bad
        indices = np.flatnonzero(bad)
        fields[field] = {'nonfinite_count': int(bad.sum()), 'NaN_count': int(np.isnan(a).sum()),
                         'positive_infinity_count': int(np.isposinf(a).sum()), 'negative_infinity_count': int(np.isneginf(a).sum()),
                         'sky_prefix_nonfinite_count': int(bad[:100000].sum()), 'first_20_row_indices_zero_based': indices[:20].tolist()}
    bad_indices = np.flatnonzero(invalid_rows)
    if not len(bad_indices):
        raise ValueError('Independent raw diagnosis did not reproduce the wrapper failure')
    samples = []
    for index in bad_indices[:20]:
        samples.append({'row_index_zero_based': int(index), 'is_shared_sky': bool(index < 100000),
                        'values': {f: encoded(v[f][index]) for f in v.dtype.names},
                        'nonfinite_fields_uint32_hex': {f: format(int(np.asarray(v[f][index]).view(np.uint32)), '08x')
                                                       for f in v.dtype.names if not np.isfinite(v[f][index])}})
    coarse_path = BASE / 'official-training/small-city-coarse-i30000-r4-003/model/point_cloud/iteration_30000/point_cloud.ply'
    coarse = stage.ply_vertices(coarse_path)
    sky = {}
    for name, names in [('xyz', ['x', 'y', 'z']), ('log_scale', [f'scale_{i}' for i in range(3)]),
                        ('quaternion_wxyz', [f'rot_{i}' for i in range(4)]), ('opacity_logit', ['opacity']),
                        ('SH_DC', [f'f_dc_{i}' for i in range(3)])]:
        sky[name] = bit_audit(rows(v, names), rows(coarse, names))
    actual_rest = rows(v, [f'f_rest_{i}' for i in range(45)]).reshape(100000, 3, 15).transpose(0, 2, 1)
    expected_rest = np.zeros((100000, 15, 3), dtype=np.float32)
    expected_rest[:, :3, :] = rows(coarse, [f'f_rest_{i}' for i in range(9)]).reshape(100000, 3, 3).transpose(0, 2, 1)
    sky['SH_rest_SH1_to_SH3_zero_padded'] = bit_audit(actual_rest, expected_rest)
    activated = {}
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        for i in range(3):
            a = np.exp(v[f'scale_{i}'])
            finite = np.isfinite(a)
            activated[f'scale_{i}'] = {'nonfinite_count': int((~finite).sum()), 'zero_count': int((a == 0).sum()),
                                      'finite_min': float(a[finite].min()), 'finite_max': float(a[finite].max())}
        q = rows(v, [f'rot_{i}' for i in range(4)], len(v))
        norms = np.linalg.norm(q, axis=1)
        finite = np.isfinite(norms)
        quaternion = {'nonfinite_norm_count': int((~finite).sum()), 'zero_norm_count': int((norms == 0).sum()),
                      'finite_min': float(norms[finite].min()), 'finite_max': float(norms[finite].max())}
    text = (OUT / 'training.log').read_text(encoding='utf-8', errors='replace')
    peaks = re.findall(r'peak memory:\s*(\d+)', text)
    log_nan_inf = [line for line in text.splitlines() if re.search(r'(?i)(?:Loss|Depth Loss)=(?:nan|[-+]?inf)', line)]
    source_checks = [{'relative_path': item['relative_path'], 'expected_sha256': item['sha256'],
                      'live_matches': sha(item['path']) == item['sha256'],
                      'snapshot_matches': sha(OUT / 'official-source' / item['relative_path']) == item['sha256']}
                     for item in config['official_source_files']]
    if not all(item['live_matches'] and item['snapshot_matches'] for item in source_checks):
        raise ValueError('Actual official source changed during the failed run')
    new_inventory = inventory(OUT)
    if old_inventory != new_inventory:
        raise ValueError('Original failed run bytes changed during diagnosis')
    result = {
        'status': 'actual_failed_full30k_chunk11_nonfinite_saved_PLY_reproduced_CPU_no_repair_no_retry',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
        'run': RUN, 'chunk': '1_1', 'requested_iterations': 30000, 'official_trainer_exit_code': outcome['exit_code'],
        'stage_outcome': outcome, 'stage_outcome_sha256': sha(OUT / 'outcome.json'),
        'saved_parameter_path': str(parameter_path), 'saved_parameter_sha256': sha(parameter_path),
        'saved_parameter_bytes': parameter_path.stat().st_size, 'saved_format': 'binary_little_endian_PLY',
        'actual_saved_G': len(v), 'PLY_payload_size_and_scalar_layout_valid': True,
        'invalid_gaussian_rows_count': int(invalid_rows.sum()), 'invalid_shared_sky_rows_count': int(invalid_rows[:100000].sum()),
        'field_nonfinite_counts': fields, 'first_20_invalid_rows': samples,
        'shared_sky_bit_audits': sky, 'activated_float32_scale_diagnostics': activated, 'float32_quaternion_norm_diagnostics': quaternion,
        'actual_official_final_save_log': '[ITER 30000] Saving Gaussians' in text,
        'actual_training_complete_log': 'Training complete.' in text,
        'logged_torch_peak_allocated_bytes': int(peaks[-1]) if peaks else None,
        'memory_scope': 'Official torch CUDA allocated peak only; not total/reserved/instantaneous GPU peak.',
        'logged_nonfinite_loss_lines': log_nan_inf[:20], 'logged_nonfinite_loss_line_count': len(log_nan_inf),
        'official_source_checks': source_checks, 'original_failed_run_files_preserved': old_inventory,
        'actual_original_training_controller_processes': active,
        'actual_python_process_snapshot': snapshot['actual_python_processes'],
        'cause': None, 'cause_scope': 'This audit locates saved numerical invalidity only; it does not establish first iteration or computational cause.',
        'automatic_retry_performed': False, 'parameters_repaired_or_rows_deleted': False,
        'CUDA_initialized': False, 'torch_imported': False,
        'all_four_full_training_complete': False, 'Creator_post_Merger_or_LOD_complete': False,
    }
    TARGET.mkdir(parents=True)
    path = TARGET / 'diagnosis.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (TARGET / 'diagnosis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path),
                      'saved_G': len(v), 'invalid_rows': int(invalid_rows.sum()),
                      'invalid_fields': {k: d for k, d in fields.items() if d['nonfinite_count']},
                      'shared_sky_six_groups_bitexact': all(d['float32_bitexact'] for d in sky.values()),
                      'CUDA_initialized': False}, allow_nan=False))


if __name__ == '__main__':
    main()

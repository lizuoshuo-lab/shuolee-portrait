"""CPU lineage/activation follow-up; preserves failed run and first diagnosis bytes."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
RUN = 'small-city-chunk11-i30000-r4-003'
OUT = BASE / 'official-training' / RUN
TARGET = BASE / 'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-lineage-CPU-002'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(a):
    values = np.asarray(a).flatten()
    return [float(v) if np.isfinite(v) else ('NaN' if np.isnan(v) else ('+Infinity' if v > 0 else '-Infinity')) for v in values]


def main():
    if TARGET.exists():
        raise ValueError('Preserve existing lineage report')
    prior_path = BASE / 'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-CPU-001/diagnosis.json'
    if sha(prior_path) != '338423bcf489386fbf7dd5a47c06281ad9ea01794d00eed2bfe8e347e56c13c7':
        raise ValueError('Prior numerical failure evidence changed')
    prior = read(prior_path)
    parameter = Path(prior['saved_parameter_path'])
    if sha(parameter) != prior['saved_parameter_sha256']:
        raise ValueError('Original failed PLY changed')
    config = read(OUT / 'configuration.json')
    official = {item['relative_path']: item for item in config['official_source_files']}
    sources = {}
    for name in ('scene/gaussian_model.py', 'train_single.py'):
        item = official[name]
        if sha(item['path']) != item['sha256'] or sha(OUT / 'official-source' / name) != item['sha256']:
            raise ValueError('Unchanged actual official source contract violated')
        sources[name] = {'live_source_sha256': item['sha256'], 'snapshot_source_sha256': sha(OUT / 'official-source' / name)}
    gaussian_text = (OUT / 'official-source/scene/gaussian_model.py').read_text(encoding='utf-8')
    if (gaussian_text.count('selected_pts_mask[:self.scaffold_points] = False') != 2 or
        gaussian_text.count('prune_mask[:self.scaffold_points] = False') != 1 or
        gaussian_text.count('self.prune_points(') != 2):
        raise ValueError('Expected official scaffold-prefix protection call sites differ')
    parser_path = ROOT / 'scripts/large_scene_gs/run_small_city_stage.py'
    if sha(parser_path) != 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704':
        raise ValueError('Frozen CPU PLY parser changed')
    spec = importlib.util.spec_from_file_location('pure_CPU_stage_parser', parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    coarse_path = BASE / 'official-training/small-city-coarse-i30000-r4-003/model/point_cloud/iteration_30000/point_cloud.ply'
    if sha(coarse_path) != 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31':
        raise ValueError('Completed coarse model changed')
    coarse = module.ply_vertices(coarse_path)
    v = module.ply_vertices(parameter)
    chunks = BASE / 'official-inputs/small-city-training-002/chunks/1_1'
    center = np.asarray([float(x) for x in (chunks / 'center.txt').read_text().split()], dtype=np.float32)
    extent = np.asarray([float(x) for x in (chunks / 'extent.txt').read_text().split()], dtype=np.float32)
    xyz = np.stack([coarse[f] for f in ('x', 'y', 'z')], axis=1).astype(np.float32)
    distance = np.max(np.abs(xyz[:, :2] - center[:2]), axis=1)
    selected = (distance > np.float32(.5) * extent[0]) & (distance < np.float32(1.5) * extent[0])
    selected[:100000] = True
    indices = np.flatnonzero(selected)
    initial_SfM = module.ply_vertices(chunks / 'sparse/0/points3D.ply')
    failed_rows = []
    for sample in prior['first_20_invalid_rows']:
        row = sample['row_index_zero_based']
        if row < 100000:
            kind = 'shared_sky_prefix'
        elif row < len(indices):
            kind = 'selected_coarse_scaffold_prefix'
        else:
            kind = 'new_SfM_or_densified_region_not_individually_identifiable_from_final_PLY'
        raw_scales = np.asarray([v[f'scale_{i}'][row] for i in range(3)], dtype=np.float32)
        raw_q = np.asarray([v[f'rot_{i}'][row] for i in range(4)], dtype=np.float32)
        raw_opacity = np.float32(v['opacity'][row])
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            scales = np.exp(raw_scales)
            opacity = np.float32(1) / (np.float32(1) + np.exp(-raw_opacity))
            qnorm = np.linalg.norm(raw_q)
            normalized_q = raw_q / qnorm
        original_index = int(indices[row]) if row < len(indices) else None
        failed_rows.append({
            'final_PLY_row_index_zero_based': row, 'classification': kind,
            'derived_original_coarse_PLY_row_index_zero_based': original_index,
            'classification_scope': 'Derived from unchanged official initial boolean selection/prepend order and protected scaffold prefix; no runtime point-ID instrumentation.',
            'original_coarse_all_raw_fields_finite': bool(all(np.isfinite(coarse[f][original_index]) for f in coarse.dtype.names)) if original_index is not None else None,
            'original_coarse_raw_values': {f: encoded(coarse[f][original_index])[0] for f in coarse.dtype.names} if original_index is not None else None,
            'raw_opacity_logit': encoded(raw_opacity)[0], 'float32_sigmoid_opacity': encoded(opacity)[0],
            'raw_log_scale': encoded(raw_scales), 'float32_exp_scale': encoded(scales),
            'raw_quaternion_wxyz': encoded(raw_q), 'float32_quaternion_norm': encoded(qnorm)[0],
            'float32_normalized_quaternion_wxyz': encoded(normalized_q),
            'finite_SH_DC_and_rest': bool(all(np.isfinite(v[f][row]) for f in v.dtype.names if f.startswith('f_'))),
        })
    if sha(parameter) != prior['saved_parameter_sha256'] or sha(prior_path) != '338423bcf489386fbf7dd5a47c06281ad9ea01794d00eed2bfe8e347e56c13c7':
        raise ValueError('Original numerical failure bytes changed during CPU mapping')
    result = {
        'status': 'failed11_single_nonSky_scaffold_Gaussian_NaN_lineage_and_activation_CPU_verified_no_repair',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
        'prior_diagnosis': str(prior_path), 'prior_diagnosis_sha256': sha(prior_path),
        'failed_PLY_sha256': sha(parameter), 'actual_initial_selected_scaffold_count': len(indices),
        'actual_initial_new_SfM_count': len(initial_SfM), 'predicted_initial_G': len(indices) + len(initial_SfM),
        'failed_rows': failed_rows, 'actual_official_source': sources,
        'source_prefix_invariants': [
            'create_from_pcd prepends ordered selected coarse rows before new chunk SfM.',
            'densify clone/split excludes selected scaffold prefix and appends new points.',
            'density prune clears its removal mask for every selected scaffold prefix row.',
            'all prune_points calls are the protected split and density prune call sites; boolean filtering preserves retained order.',
            'scale cap protects scaffold rows; changes neither row order nor identity.'
        ],
        'not_zero_opacity_minus_infinity_case': True,
        'first_nonfinite_iteration': None, 'computational_cause': None,
        'cause_scope': 'Initial source values finite and final geometry/opacity/scale/quaternion NaN; no intermediate checkpoints or gradient trace establish when or why.',
        'raw_finite_gate_relaxed': False, 'rows_deleted_or_repaired': False, 'automatic_GPU_retry': False,
        'CUDA_initialized': False, 'torch_imported': False,
        'all_four_full_training_complete': False, 'Creator_post_Merger_or_LOD_complete': False,
    }
    TARGET.mkdir(parents=True)
    path = TARGET / 'lineage-activation.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (TARGET / 'diagnosis-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path), 'scaffold_count': len(indices),
                      'initial_G': len(indices) + len(initial_SfM), 'failed_rows': failed_rows,
                      'CUDA_initialized': False}, allow_nan=False))


if __name__ == '__main__':
    main()

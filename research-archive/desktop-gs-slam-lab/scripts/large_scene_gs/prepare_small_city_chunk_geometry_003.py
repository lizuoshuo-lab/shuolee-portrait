"""CPU-only geometry forecast for chunks pending completed local coarse 003.

Uses the already completed local 64-iteration coarse only as a geometry planning
reference. It is never a training scaffold substitute. Actual chunk execution
requires the >=30000-iteration completed local 003 audit and parent GPU gate.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    plan = BASE / 'official-inputs/small-city-chunk-geometry-CPU-003'
    plan.mkdir(parents=True, exist_ok=False)
    stage = ROOT / 'scripts/large_scene_gs/run_small_city_stage.py'
    spec = importlib.util.spec_from_file_location('pure_local_stage', stage)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = BASE / 'official-inputs/small-city-training-002'
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    preflight = BASE / 'official-training/small-city-coarse-preflight-002'
    receipt = json.loads((preflight / 'outcome.json').read_text(encoding='utf-8'))
    audit = json.loads((preflight / 'model-audit.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'completed' or not audit['exact_SfM_position_freeze_passed']:
        raise ValueError('Completed local geometry reference missing')
    for item in audit['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Local reference saved bytes changed')
    xyz_fields = module.ply_vertices(audit['parameter_path'])
    xyz = np.stack([xyz_fields[n] for n in ('x', 'y', 'z')], axis=1).astype(np.float32)
    model_source = ROOT / 'repos/hierarchical-3d-gaussians/scene/gaussian_model.py'
    canonical = next(item for item in manifest['official_source_files'] if item['relative_path'] == 'scene/gaussian_model.py')
    if sha(model_source) != canonical['sha256']:
        raise ValueError('Official chunk initialization source changed')
    sky = int(audit['skybox_prefix_count'])
    records = []
    for chunk in manifest['chunks']:
        directory = inputs / 'chunks' / chunk['name']
        center_path, extent_path = directory / 'center.txt', directory / 'extent.txt'
        center = np.asarray([float(value) for value in center_path.read_text().split()], dtype=np.float32)
        extent = np.asarray([float(value) for value in extent_path.read_text().split()], dtype=np.float32)
        distances = np.abs(xyz - center)
        distance = np.maximum(distances[:, 0], distances[:, 1])
        inner = np.float32(.5) * extent[0]
        outer = np.float32(1.5) * extent[0]
        selected = (distance > inner) & (distance < outer)
        selected[:sky] = True
        chunk_points = module.ply_vertices(directory / 'sparse/0/points3D.ply')
        scaffold_count = int(selected.sum())
        min_gap = float(np.minimum(np.abs(distance[sky:] - inner), np.abs(distance[sky:] - outer)).min())
        records.append({'chunk': chunk['name'], 'train_views': len(chunk['train_names']), 'heldout_views': len(chunk['test_names']),
            'chunk_original_SfM_points': len(chunk_points), 'local_preflight_geometry_scaffold_selected': scaffold_count,
            'shared_skybox_prefix_count': sky, 'predicted_initial_gaussians': scaffold_count + len(chunk_points),
            'minimum_boundary_distance_scene_units': min_gap, 'center': center.tolist(), 'extent': extent.tolist(),
            'center_sha256': sha(center_path), 'extent_sha256': sha(extent_path),
            'chunk_points_PLY_sha256': sha(directory / 'sparse/0/points3D.ply'),
            'test_list_sha256': sha(directory / 'sparse/0/test.txt'),
            'future_supervised_preflight_argv': [str(ROOT / '.venv-large-gs/Scripts/python.exe'), '-u',
                str(ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py'), '--stage', 'chunk',
                '--run', 'small-city-chunk-' + chunk['name'].replace('_', '-') + '-preflight-r4-003',
                '--input', 'small-city-training-002', '--chunk', chunk['name'],
                '--coarse-run', 'small-city-coarse-i30000-r4-003', '--iterations', '64', '--resolution', '4',
                '--exposure', 'official', '--port', '6023']})
    report = {'status': 'CPU_geometry_forecast_only_actual_coarse003_binding_and_parent_GPU_gate_pending',
              'recorded_utc': datetime.now(timezone.utc).isoformat(), 'CUDA_initialized': False,
              'torch_imported': 'torch' in sys.modules, 'official_algorithm_modified': False,
              'input_manifest_sha256': sha(manifest_path), 'official_model_source_sha256': sha(model_source),
              'local_stage_wrapper_sha256': sha(stage), 'script_sha256': sha(__file__),
              'geometry_reference': str(preflight), 'geometry_reference_model_sha256': sha(audit['parameter_path']),
              'geometry_reference_requested_iterations': 64,
              'training_dependency': 'Actual completed local small-city-coarse-i30000-r4-003 with same input; 64-iteration reference is not allowed for chunk training',
              'source_predicate': 'max(abs(x-cx),abs(y-cy)) > 0.5*extent[0] and <1.5*extent[0]; skybox prefix forced true; no z condition',
              'limitations': ['CPU count uses float32 geometry reference; compare against actual GPU init after completed full local coarse.',
                  'Memory and densification growth cannot be inferred from initial count; use actual preflight and full training telemetry.',
                  'Source scaffold predicate uses extent[0] on both x/y and no z bound. Extents equal here; do not generalize to anisotropic chunks.'],
              'chunks': records}
    if report['torch_imported']:
        raise ValueError('CPU forecast unexpectedly imported torch')
    (plan / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(__file__, plan / 'source.py')
    print(json.dumps({'report': str(plan / 'report.json'), 'CUDA_initialized': False,
                      'initial_count_forecasts': {c['chunk']: c['predicted_initial_gaussians'] for c in records}}))


if __name__ == '__main__':
    main()

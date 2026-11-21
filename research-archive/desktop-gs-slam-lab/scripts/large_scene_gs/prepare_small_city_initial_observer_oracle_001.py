"""CPU-only exact raw initialization oracle from the actual completed coarse PLY."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = BASE / 'official-training-observation-plans/small-city-initial-suspect-oracle-001'
    if target.exists():
        raise ValueError('Preserve existing CPU oracle')
    coarse = BASE / 'official-training/small-city-coarse-i30000-r4-003/model/point_cloud/iteration_30000/point_cloud.ply'
    if sha(coarse) != 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31':
        raise ValueError('Actual completed coarse bytes changed')
    parser = ROOT / 'scripts/large_scene_gs/run_small_city_stage.py'
    if sha(parser) != 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704':
        raise ValueError('Frozen CPU PLY parser changed')
    spec = importlib.util.spec_from_file_location('pure_CPU_actual_PLY', parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v = module.ply_vertices(coarse)
    chunk = BASE / 'official-inputs/small-city-training-002/chunks/1_1'
    center = np.asarray([float(x) for x in (chunk / 'center.txt').read_text().split()], dtype=np.float32)
    extent = np.asarray([float(x) for x in (chunk / 'extent.txt').read_text().split()], dtype=np.float32)
    xyz = np.stack([v[f] for f in ('x', 'y', 'z')], axis=1)
    distance = np.max(np.abs(xyz[:, :2] - center[:2]), axis=1)
    selected = (distance > np.float32(.5) * extent[0]) & (distance < np.float32(1.5) * extent[0])
    selected[:100000] = True
    indices = np.flatnonzero(selected)
    if len(indices) != 1835543 or int(indices[1377084]) != 1778538:
        raise ValueError('Actual ordered protected-prefix mapping differs')
    i = 1778538
    def row(fields):
        return np.asarray([v[f][i] for f in fields], dtype=np.float32)
    rest = np.zeros((15, 3), dtype=np.float32)
    rest[:3, :] = row([f'f_rest_{j}' for j in range(9)]).reshape(3, 3).T
    values = {'xyz': row(['x', 'y', 'z']), 'f_dc': row([f'f_dc_{j}' for j in range(3)])[None, :], 'f_rest': rest,
              'opacity': row(['opacity']), 'scaling': row([f'scale_{j}' for j in range(3)]),
              'rotation': row([f'rot_{j}' for j in range(4)])}
    groups = {}
    for name, a in values.items():
        a = np.ascontiguousarray(a)
        if not np.isfinite(a).all():
            raise ValueError('Original coarse raw values not finite')
        groups[name] = {'shape': list(a.shape), 'dtype': 'float32', 'uint32_hex': [format(int(x), '08x') for x in a.view(np.uint32).reshape(-1)],
                        'values': a.tolist(), 'raw_group_bytes_sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    result = {'status': 'CPU_exact_actual_coarse_raw6group_to_initialized_suspect_oracle_passed',
              'observed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
              'actual_coarse_PLY': str(coarse), 'actual_coarse_PLY_sha256': sha(coarse),
              'suspect_selected_prefix_row_zero_based': 1377084, 'original_coarse_PLY_row_zero_based': 1778538,
              'selected_scaffold_count': len(indices), 'expected_initial_G': 2065281,
              'groups': groups, 'raw_semantics': 'No sigmoid/exp/quaternion normalization. Raw xyz/DC/logscale/logit/wxyz exact; SH1 basis rows copied and higher SH rows zero.',
              'CUDA_initialized': False, 'torch_imported': False}
    target.mkdir(parents=True)
    path = target / 'expected-raw.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (target / 'oracle-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path), 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()

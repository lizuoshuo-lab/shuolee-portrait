"""Numeric implementation checks; no captured subject, calibration fit or training."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import runtime as rt


def check():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from plyfile import PlyData, PlyElement
    # configure only local source modules; these numeric checks never read model assets.
    rt.configure(PROJECT / 'assets-unused-for-numeric-check')
    from shuolee_portrait.training import camera
    from shuolee_portrait.export import rotate_ply
    from utils.sh_utils import eval_sh
    E = np.eye(4)
    E[:3, :3] = Rotation.from_euler('xyz', [.3, -.4, .1]).as_matrix()
    E[:3, 3] = [.2, -.1, 3.]
    row = dict(frame='synthetic', wh=[731, 1297], K=[[980, 0, 351.4], [0, 987, 624.7], [0, 0, 1]], w2c=E.tolist())
    _, receipt = camera(row, 0, 768)
    checks = {'actual_camera_projection': receipt}
    rng = np.random.default_rng(15)
    n = 37
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] +
             [f'f_rest_{i}' for i in range(45)] + ['opacity'] +
             [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    values = np.zeros(n, dtype=[(k, '<f4') for k in names])
    for k in names:
        values[k] = rng.normal(size=n)
    for i in range(3):
        values[f'scale_{i}'] = rng.uniform(-5, -3, n)
    quats = Rotation.random(n, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    for i in range(4):
        values[f'rot_{i}'] = quats[:, i]
    with tempfile.TemporaryDirectory(prefix='shuolee-numeric-contract-') as temp:
        source, mapped = Path(temp) / 'synthetic.ply', Path(temp) / 'rotated.ply'
        PlyData([PlyElement.describe(values, 'vertex')]).write(source)
        R = Rotation.from_euler('xyz', [.4, .8, -.3]).as_matrix()
        checks['synthetic_ply_transform'] = rotate_ply(source, mapped, R, np.array([3., 4., 5.]))
        loaded = PlyData.read(mapped)['vertex'].data
        def coefficients(v):
            dc = np.column_stack([v[f'f_dc_{i}'] for i in range(3)])[:, None, :]
            rest = np.column_stack([v[f'f_rest_{i}'] for i in range(45)]).reshape(-1, 3, 15).transpose(0, 2, 1)
            return np.concatenate([dc, rest], axis=1).transpose(0, 2, 1)
        directions = rng.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        a = eval_sh(3, coefficients(values), directions @ R)
        b = eval_sh(3, coefficients(loaded), directions)
        error = float(abs(a-b).max())
        assert error < 1e-6
        checks['SH_color_rotation_max'] = error
    checks.update(passed=True, physical_camera_accuracy_proven=False, synthetic_fixtures_only=True,
                  model_weights_loaded=False, training_started=False, quality_accepted=False)
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check()
    rt.save(args.output, result)
    print(json.dumps(result, indent=2))

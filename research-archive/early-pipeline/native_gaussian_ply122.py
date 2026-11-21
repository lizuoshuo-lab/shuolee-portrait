"""Strict standard 3DGS PLY serialization for native, world-space Gaussians.

PLY does not declare its camera or renderer. Keep the native contract and cameras
in a separate manifest. This module performs no coordinate or SH basis rotation.
"""
from pathlib import Path
import math
import numpy as np
from plyfile import PlyData, PlyElement


def validate_parameters(means, quats, scales, opacities, sh):
    values = [np.asarray(v) for v in (means, quats, scales, opacities, sh)]
    means, quats, scales, opacities, sh = values
    n = len(means)
    if n == 0 or means.shape != (n, 3) or quats.shape != (n, 4) or scales.shape != (n, 3) or opacities.shape != (n,):
        raise ValueError('Expected nonempty N x 3 means/scales, N x 4 wxyz, N opacity')
    if sh.ndim != 3 or sh.shape[0] != n or sh.shape[2] != 3 or sh.shape[1] not in (1, 4, 9, 16):
        raise ValueError('Expected complete real SH degrees 0 through 3, N x K x RGB')
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError('Nonfinite Gaussian parameters are not silently removed')
    if np.any(scales <= 0) or np.any((opacities <= 0) | (opacities >= 1)):
        raise ValueError('Positive linear scales and opacity strictly between zero and one required for finite log/logit')
    norms = np.linalg.norm(quats.astype(np.float64), axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero quaternion')
    return int(math.isqrt(sh.shape[1]) - 1)


def write_standard_ply(path, *, means, quats, scales, opacities, sh):
    degree = validate_parameters(means, quats, scales, opacities, sh)
    means, quats, scales, opacities, sh = [np.asarray(v, dtype=np.float64) for v in (means, quats, scales, opacities, sh)]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rest = sh[:, 1:].transpose(0, 2, 1).reshape(len(means), -1)
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
             + [f'f_rest_{i}' for i in range(rest.shape[1])] + ['opacity']
             + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    values = np.column_stack((means, np.zeros_like(means), sh[:, 0], rest,
                              np.log(opacities) - np.log1p(-opacities), np.log(scales), quats)).astype('<f4')
    if not np.isfinite(values).all():
        raise ValueError('Float32 PLY cannot represent these parameters')
    vertices = np.empty(len(means), dtype=[(name, '<f4') for name in names])
    for i, name in enumerate(names):
        vertices[name] = values[:, i]
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False, byte_order='<',
            comments=['Standard world-space Gaussian attributes; renderer and cameras require the companion manifest']).write(path)
    return {'vertices': len(means), 'sh_degree': degree, 'position': 'world',
            'scale': 'natural-log standard-deviation', 'quaternion': 'wxyz local-to-world',
            'opacity': 'logit', 'sh_rest': 'RGB-channel-major, basis-index-minor',
            'coordinate_transform_applied': False}


def read_standard_ply(path):
    vertices = PlyData.read(path)['vertex'].data
    names = set(vertices.dtype.names)
    required = ['x', 'y', 'z', 'opacity'] + [f'f_dc_{i}' for i in range(3)] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    if any(name not in names for name in required):
        raise ValueError('Incomplete standard Gaussian PLY attributes')
    rest_names = [n for n in names if n.startswith('f_rest_')]
    if len(rest_names) not in (0, 9, 24, 45) or set(rest_names) != {f'f_rest_{i}' for i in range(len(rest_names))}:
        raise ValueError('SH rest fields are not a complete contiguous degree')
    cols = lambda fields: np.column_stack([vertices[n] for n in fields]).astype(np.float32)
    means = cols(['x', 'y', 'z'])
    quats = cols([f'rot_{i}' for i in range(4)])
    stored_scaling = cols([f'scale_{i}' for i in range(3)])
    stored_opacity = np.asarray(vertices['opacity'], dtype=np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        scales = np.exp(stored_scaling.astype(np.float64)).astype(np.float32)
        opacities = np.exp(-np.logaddexp(0, -stored_opacity)).astype(np.float32)
    sh0 = cols([f'f_dc_{i}' for i in range(3)])[:, None, :]
    rest = (cols([f'f_rest_{i}' for i in range(len(rest_names))]).reshape(len(vertices), 3, -1).transpose(0, 2, 1)
            if rest_names else np.empty((len(vertices), 0, 3), dtype=np.float32))
    sh = np.concatenate((sh0, rest), axis=1)
    degree = validate_parameters(means, quats, scales, opacities, sh)
    quats = (quats.astype(np.float64) / np.linalg.norm(quats.astype(np.float64), axis=1, keepdims=True)).astype(np.float32)
    return {'means': means, 'quats': quats, 'scales': scales, 'opacities': opacities, 'sh': sh, 'sh_degree': degree}

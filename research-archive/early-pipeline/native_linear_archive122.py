"""Lossless inference arrays. No logarithms, quaternion renormalization or resume.

Standard PLY remains a separate, explicitly lossy preview representation.
"""
from pathlib import Path
import math
import zipfile
import numpy as np

FIELDS = ('means', 'quats', 'scales', 'opacities', 'sh')
FORMAT = 'linear-float32-npz-no-renormalization-v1'


def validate(parameters):
    if set(parameters) != set(FIELDS):
        raise ValueError('Exactly five linear parameter arrays required')
    arrays = {k: np.asarray(parameters[k]) for k in FIELDS}
    n = len(arrays['means'])
    shapes = dict(means=(n, 3), quats=(n, 4), scales=(n, 3), opacities=(n,))
    if not 1 <= n <= 10000000:
        raise ValueError('Unsupported point count')
    for k, a in arrays.items():
        if a.dtype != np.dtype('float32') or not np.isfinite(a).all():
            raise ValueError('Finite float32 without implicit conversion required: ' + k)
        if k in shapes and a.shape != shapes[k]:
            raise ValueError('Invalid parameter shape: ' + k)
    s = arrays['sh'].shape
    if len(s) != 3 or s[0] != n or s[1] not in (1, 4, 9, 16) or s[2] != 3:
        raise ValueError('Complete SH degree 0..3 required')
    if (arrays['scales'] <= 0).any() or ((arrays['opacities'] <= 0) | (arrays['opacities'] >= 1)).any():
        raise ValueError('Positive scale and interior opacity required')
    norms = np.linalg.norm(arrays['quats'].astype(np.float64), axis=1)
    if np.max(np.abs(norms - 1)) > 2e-5:
        raise ValueError('Native export expects already normalized quaternions; never normalize during I/O')
    return n, math.isqrt(s[1]) - 1


def write(path, parameters):
    n, degree = validate(parameters)
    with Path(path).open('xb') as f:
        np.savez(f, **parameters)
    return dict(vertices=n, sh_degree=degree, format=FORMAT, optimizer_saved=False)


def read(path, *, vertices, degree):
    if type(vertices) is not int or not 1 <= vertices <= 10000000 or type(degree) is not int or degree not in range(4):
        raise ValueError('Bounded count and complete SH degree required')
    expected = vertices * (11 + 3 * (degree + 1) ** 2) * 4
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) != 5 or {i.filename for i in infos} != {k + '.npy' for k in FIELDS}:
            raise ValueError('Duplicate, missing or unexpected archive entries')
        if any(i.compress_type != zipfile.ZIP_STORED for i in infos) or sum(i.file_size for i in infos) > expected + 4096:
            raise ValueError('Unexpected compressed/oversized native archive')
    with np.load(path, allow_pickle=False) as z:
        p = {k: z[k].copy() for k in FIELDS}
    if validate(p) != (vertices, degree):
        raise ValueError('Native arrays disagree with declared count/SH')
    return p

"""CPU-only camera validation for the frame/pose input contract.

Preserve complete integer-center intrinsics and require the optional renderer
intrinsics to add 0.5 to the principal point exactly once. This module performs
no fitting, detector inference or training admission.
"""
import numpy as np


def need(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, shape, message):
    result = np.asarray(value, dtype=np.float64)
    need(result.shape == shape and np.isfinite(result).all(), message)
    return result


def validate_camera(row):
    """Return one complete K; absence is reported, never replaced with a guess."""
    present = {key for key in ('K_integer_centers', 'K', 'w2c') if key in row}
    if not present:
        return None
    need('K_integer_centers' in present and 'w2c' in present,
         'Partial or unlabelled camera: require integer-centre K and w2c.')
    K = numeric(row['K_integer_centers'], (3, 3), 'Invalid K.')
    E = numeric(row['w2c'], (4, 4), 'Invalid w2c.')
    need(K[0, 0] > 0 and K[1, 1] > 0 and np.allclose(K[2], [0, 0, 1], atol=1e-12),
         'Unsupported pinhole K.')
    need(np.allclose(E[3], [0, 0, 0, 1], atol=1e-9) and
         np.allclose(E[:3, :3].T @ E[:3, :3], np.eye(3), atol=1e-5) and
         abs(np.linalg.det(E[:3, :3]) - 1) < 1e-5, 'w2c is not a proper rigid transform.')
    edge = K.copy()
    edge[:2, 2] += .5
    if 'K' in row:
        need(np.allclose(numeric(row['K'], (3, 3), 'Invalid renderer K.'), edge,
                         atol=1e-9, rtol=0), 'Renderer K must differ by exactly +0.5 in cx/cy.')
    return K, E, edge

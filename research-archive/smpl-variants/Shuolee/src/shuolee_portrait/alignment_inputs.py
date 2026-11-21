"""CPU-only camera validation for the frame/pose input contract.

Preserve complete integer-center intrinsics and require the optional renderer
intrinsics to add 0.5 to the principal point exactly once. This module performs
no fitting, detector inference or training admission.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


BODY_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
              'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
FEET_NAMES = ['left_big_toe', 'left_small_toe', 'left_heel',
              'right_big_toe', 'right_small_toe', 'right_heel']
HAND_NAMES = [side + '_' + name for side in ('left', 'right') for name in
              ['hand_root'] + [f'{finger}{i}' for finger in
               ('thumb', 'forefinger', 'middle_finger', 'ring_finger', 'pinky_finger')
               for i in range(1, 5)]]
COCO133_NAMES = BODY_NAMES + FEET_NAMES + [f'face-{i}' for i in range(68)] + HAND_NAMES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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

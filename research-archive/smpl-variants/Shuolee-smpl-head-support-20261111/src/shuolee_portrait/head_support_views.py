"""Metric fixed-framing camera geometry for native SH0 head-support comparisons."""

import math

import numpy as np


AZIMUTHS_DEG = (-90, -45, 0, 45, 90)
PITCHES_DEG = (30, 15, 0, -15, -30)
NORMAL_DISTANCE_M = 3.0
CLOSE_DISTANCE_M = .65


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not math.isfinite(length) or length < 1e-12:
        raise ValueError('A camera direction must be a finite nonzero 3-vector.')
    return vector / length


def camera_center(w2c):
    """Return the optical center in the input metric world."""
    matrix = np.asarray(w2c, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.array_equal(matrix[3], [0., 0., 0., 1.])):
        raise ValueError('Expected a finite 4×4 world-to-camera matrix.')
    return -matrix[:3, :3].T @ matrix[:3, 3]


def orbit_basis(w2c, pivot_world, up_world):
    """Use the front reference optical center and a fixed metric up direction."""
    up = _unit(up_world)
    outward = camera_center(w2c) - np.asarray(pivot_world, dtype=np.float64)
    front = _unit(outward - np.dot(outward, up) * up)
    return front, up


def orbit_camera(pivot_world, front_world, up_world, azimuth_deg, pitch_deg, distance_m):
    """World-to-CV-camera at a metric radius; positive pitch raises the camera."""
    if not all(math.isfinite(value) for value in (azimuth_deg, pitch_deg, distance_m)) or distance_m <= 0:
        raise ValueError('Expected finite angles and a positive distance.')
    pivot = np.asarray(pivot_world, dtype=np.float64)
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError('Expected a finite world pivot.')
    up = _unit(up_world)
    front = np.asarray(front_world, dtype=np.float64)
    front = _unit(front - np.dot(front, up) * up)
    east = _unit(np.cross(up, front))
    azimuth, pitch = np.deg2rad([azimuth_deg, pitch_deg])
    horizontal = np.cos(azimuth) * front + np.sin(azimuth) * east
    outward = _unit(np.cos(pitch) * horizontal + np.sin(pitch) * up)
    forward = -outward
    right = _unit(np.cross(forward, up))
    down = np.cross(forward, right)
    center = pivot + distance_m * outward
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack((right, down, forward))
    matrix[:3, 3] = -matrix[:3, :3] @ center
    return matrix


def project_world(vertices, K, w2c):
    """Project full corner-convention K to integer pixel-center coordinates."""
    points = np.asarray(vertices, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    matrix = np.asarray(w2c, dtype=np.float64)
    camera = points @ matrix[:3, :3].T + matrix[:3, 3]
    homogeneous = camera @ K.T
    depth = camera[..., 2]
    uv = np.full(homogeneous.shape[:-1] + (2,), np.nan)
    np.divide(homogeneous[..., :2], depth[..., None], out=uv, where=depth[..., None] != 0)
    return uv - .5, depth


def to_portable(w2c, world_to_viewer):
    """Express the identical optical camera in a rigid portable PLY world."""
    if world_to_viewer['scale'] != 1.0:
        raise ValueError('Expected an unchanged metric portable PLY scale.')
    matrix = np.asarray(w2c, dtype=np.float64)
    rotation = np.asarray(world_to_viewer['rotation'], dtype=np.float64)
    origin = np.asarray(world_to_viewer['origin_world'], dtype=np.float64)
    portable = np.eye(4)
    portable[:3, :3] = matrix[:3, :3] @ rotation.T
    portable[:3, 3] = matrix[:3, 3] + matrix[:3, :3] @ origin
    return portable

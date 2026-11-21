"""Explicit pixel-origin conversion, without changing a camera ray or an image.

OpenCV integer_centers: top-left sample is (0, 0).
Native pixel_corners: the same sample is (0.5, 0.5).
K and observations must travel together. This does not calibrate a lens and
must not be used to silently relabel an unknown source convention.
"""
import numpy as np

CONVENTIONS = {'integer_centers': 0.0, 'pixel_corners': 0.5}


def convert_pixel_convention(K, observations, *, intrinsic_convention,
                             observation_convention, target_convention):
    for value in [intrinsic_convention, observation_convention, target_convention]:
        if value not in CONVENTIONS:
            raise ValueError('Explicit supported pixel convention required')
    if intrinsic_convention != observation_convention:
        raise ValueError('K and observations have inconsistent pixel conventions')
    k = np.asarray(K)
    uv = np.asarray(observations)
    if (k.shape != (3, 3) or k.dtype.kind not in 'ifu' or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0
            or not np.array_equal(k[2], [0, 0, 1])):
        raise ValueError('Finite full pinhole K with positive focal lengths required')
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.dtype.kind not in 'ifu' or not np.isfinite(uv).all():
        raise ValueError('Finite N x 2 pixel observations required')
    k = k.astype(np.float64, copy=True)
    uv = uv.astype(np.float64, copy=True)
    offset = CONVENTIONS[target_convention] - CONVENTIONS[intrinsic_convention]
    transform = np.eye(3, dtype=np.float64)
    transform[:2, 2] = offset
    # A K preserves the full matrix, including an off-center principal point.
    output_k = transform @ k
    output_uv = uv + offset
    receipt = {'schema': 'shuolee-pixel-origin-conversion-v1',
               'source_convention': intrinsic_convention, 'target_convention': target_convention,
               'translation_xy_pixels': [offset, offset], 'pixel_transform': transform.tolist(),
               'input_K': k.tolist(), 'output_K': output_k.tolist(),
               'observations_count': len(uv), 'images_resampled': False,
               'intrinsics_refitted': False, 'extrinsics_modified': False,
               'physical_calibration_validated': False}
    return output_k, output_uv, receipt


def to_native_pixels(K, observations, *, intrinsic_convention, observation_convention):
    """Adapter for the native renderer. Callers must state both input conventions."""
    return convert_pixel_convention(K, observations, intrinsic_convention=intrinsic_convention,
        observation_convention=observation_convention, target_convention='pixel_corners')

"""Explicit rectified -> resize -> crop -> training coordinates.

This module does not guess FISHEYE_8 or undo already-processed exported videos.
K_rectified must be the actual output pinhole K of the vendor's undistortion.
OpenCV integer-center pixels are used internally; renderer edge coordinates
are obtained by adding 0.5 exactly once to u/v and to the principal point.
"""
import numpy as np


def _matrix(a, shape, name):
    if a is None:
        raise ValueError(f'{name} is required; unknown calibration is not a default')
    a = np.asarray(a, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Invalid {name}')
    return a


def _size(a, name):
    a = _matrix(a, (2,), name)
    if np.any(a <= 0) or np.any(a != np.floor(a)):
        raise ValueError(f'{name} must be positive integer width/height')
    return a


def resize_matrix(source_wh, target_wh):
    source = _size(source_wh, 'source_wh')
    target = _size(target_wh, 'target_wh')
    sx, sy = target / source
    return np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])


def map_pixels(xy, transform):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('xy must contain finite 2-D points')
    a = _matrix(transform, (3, 3), 'pixel transform')
    out = np.c_[xy, np.ones(len(xy))] @ a.T
    if np.any(abs(out[:, 2]) < 1e-15):
        raise ValueError('Pixel mapped to infinity')
    return out[:, :2] / out[:, 2, None]


def rectified_contract(spec):
    if spec.get('input_pixel_domain') != 'rectified_pinhole':
        raise ValueError('A verified rectified pinhole output domain is required')
    if spec.get('pixel_convention') != 'integer_centers':
        raise ValueError('Declare zero-based integer-center pixel coordinates')
    if spec.get('fixed_processing_across_frames') is not True:
        raise ValueError('Fixed crop/resize across frames must be confirmed; EIS needs per-frame maps')
    K = _matrix(spec.get('K_rectified'), (3, 3), 'K_rectified')
    if min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]) or abs(K[1, 0])+abs(K[0, 1]) > 1e-12:
        raise ValueError('Expected positive, zero-skew pinhole K')
    rectified = _size(spec.get('rectified_wh'), 'rectified_wh')
    downsampled = _size(spec.get('downsampled_wh'), 'downsampled_wh')
    crop = _matrix(spec.get('crop_xywh'), (4,), 'crop_xywh after downsample')
    x, y, w, h = crop
    if np.any(crop != np.floor(crop)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > downsampled[0] or y+h > downsampled[1]:
        raise ValueError('Crop must be an actual integer ROI in the downsampled image')
    output = _size(spec.get('training_wh'), 'training_wh')
    S = resize_matrix(rectified, downsampled)
    C = np.array([[1., 0, -x], [0, 1, -y], [0, 0, 1]])
    T = resize_matrix([w, h], output)
    A = T @ C @ S
    corner = np.array([[1., 0, .5], [0, 1, .5], [0, 0, 1.]])
    R = _matrix(spec.get('rectification_R'), (3, 3), 'rectification_R')
    if not np.allclose(R@R.T, np.eye(3), atol=1e-8) or abs(np.linalg.det(R)-1) > 1e-8:
        raise ValueError('rectification_R must be a proper rotation')
    return dict(schema='shuolee-rectify-resize-crop-v1',
                order=['vendor_undistort', 'downsample', 'crop', 'training_resize'],
                K_rectified=K.tolist(), K_integer_centers=(A@K).tolist(),
                K_renderer_corners=(corner@A@K).tolist(),
                rectified_to_training=A.tolist(), training_to_rectified=np.linalg.inv(A).tolist(),
                integer_centers_to_renderer_corners=corner.tolist(),
                training_wh=output.astype(int).tolist(), rectification_R=R.tolist(),
                raw_to_rectified_nonlinear_map='must use vendor lens model/map; not represented by A',
                physical_calibration_validated=False, training_admitted=False)


def rectify_extrinsics(w2c_raw, rectification_R):
    e = _matrix(w2c_raw, (4, 4), 'w2c_raw')
    r = _matrix(rectification_R, (3, 3), 'rectification_R')
    if not np.allclose(e[3], [0, 0, 0, 1]) or not np.allclose(r@r.T, np.eye(3), atol=1e-8) or abs(np.linalg.det(r)-1)>1e-8:
        raise ValueError('Invalid rigid rectification')
    out = np.eye(4)
    out[:3] = r @ e[:3]
    return out

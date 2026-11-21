"""Recorded full-image to prior-crop mapping, in integer-centre pixel coordinates."""
import numpy as np


def prepare_prior_crop(rgb, bbox_xyxy):
    import cv2
    from hmr4d.network.hmr2.utils.preproc import crop_and_resize

    box = np.asarray(bbox_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Prior crop requires a finite, positive xyxy box.')
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Prior crop expects an H x W x 3 RGB image.')
    center = (box[:2] + box[2:]) / 2
    size = float(np.max(box[2:] - box[:2]) * 1.2)
    # The pinned upstream map spans pixel centres 0..255, not 0..256.
    # Reuse its rasterizer so HMR2 and ViTPose receive the same crop geometry.
    crop, center_size = crop_and_resize(rgb, center, size, 256, enlarge_ratio=1.0)
    half = size / 2
    src = np.stack([center-half, center+[half, -half], center]).astype(np.float32)
    dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
    transform = np.eye(3)
    transform[:2] = cv2.getAffineTransform(src, dst)
    receipt = dict(
        source_wh=[int(rgb.shape[1]), int(rgb.shape[0])], crop_wh=[256, 256],
        bbox_xyxy=box.tolist(), center_size=center_size.tolist(),
        source_to_crop=transform.tolist(), crop_to_source=np.linalg.inv(transform).tolist(),
        pixel_convention='Integer pixel centres in both source image and crop',
        upstream='hmr4d.network.hmr2.utils.preproc.crop_and_resize',
        affects='HMR2/ViTPose prior input only; full-image SfM and Gaussian images are unchanged',
        antialias_filter_changed=False,
    )
    return crop, center_size, receipt

"""Compare rendered RGB structure against a fixed region of an observed image."""

import cv2
import numpy as np


def _features(bgr, support, normalizer):
    gray = cv2.cvtColor(bgr.astype(np.float32) / 255., cv2.COLOR_BGR2GRAY)
    inside = gray * support
    mean = cv2.GaussianBlur(inside, (0, 0), 4.) / normalizer
    second = cv2.GaussianBlur(inside * gray, (0, 0), 4.) / normalizer
    sigma = np.sqrt(np.maximum(second - mean * mean, .02 ** 2))
    normalized = np.clip((gray - mean) / sigma, -3., 3.) * support
    filtered = cv2.GaussianBlur(normalized, (3, 3), .65)
    gradient = np.stack([
        cv2.Scharr(filtered, cv2.CV_32F, 1, 0, scale=1 / 32.),
        cv2.Scharr(filtered, cv2.CV_32F, 0, 1, scale=1 / 32.),
    ], axis=-1)
    return normalized, gradient, sigma


def _zncc(source, rendered, weights):
    a = source - np.average(source, axis=0, weights=weights[:, 0])
    b = rendered - np.average(rendered, axis=0, weights=weights[:, 0])
    denominator = np.sqrt((weights * a * a).sum() * (weights * b * b).sum())
    if denominator <= 1e-8:
        return 0.
    return float(np.clip((weights * a * b).sum() / denominator, -1., 1.))


def compare_profile_rgb(source_bgr, before_bgr, after_bgr, interior):
    """Score structure on one fixed photo-side head ROI. Higher is better.

    Images are corresponding uint8 BGR rasters at the same native or crop scale.
    Interior contains only trusted head pixels. It is eroded and never changes
    between the before and after estimates. These are image-alignment checks
    sensitive to both appearance and geometry, not measurements of yaw.
    """
    images = tuple(np.asarray(image) for image in (source_bgr, before_bgr, after_bgr))
    shape = images[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(
            image.shape != shape or image.dtype != np.uint8 for image in images):
        raise ValueError("Profile RGB check needs corresponding uint8 BGR images")
    inside = np.asarray(interior, dtype=bool)
    if inside.shape != shape[:2]:
        raise ValueError("Profile RGB support must be aligned with the images")
    inside = inside.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(inside)
    if not w or not h:
        return {"status": "insufficient_interior", "support_pixels": 0, "textured_pixels": 0}
    pad = 12
    region = np.s_[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    inside = inside[region]
    stable = cv2.erode(inside, np.ones((7, 7), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    support_count = int(stable.sum())
    if support_count < 100:
        return {"status": "insufficient_interior", "support_pixels": support_count, "textured_pixels": 0}
    support = inside.astype(np.float32)
    normalization = np.maximum(cv2.GaussianBlur(support, (0, 0), 4.), 1e-6)
    source_luma, source_grad, source_sigma = _features(images[0][region], support, normalization)
    strength = np.linalg.norm(source_grad, axis=-1)
    texture = stable & (source_sigma > .024) & (strength >= max(.05, np.percentile(strength[stable], 55)))
    texture_count = int(texture.sum())
    if texture_count < 48:
        return {"status": "insufficient_texture", "support_pixels": support_count, "textured_pixels": texture_count}

    source_luma = source_luma[texture, None]
    source_grad = source_grad[texture]
    strength = strength[texture]
    weights = np.minimum(strength, np.percentile(strength, 95))[:, None]

    def score(image):
        luma, grad, _ = _features(image[region], support, normalization)
        grad = grad[texture]
        lengths = np.linalg.norm(grad, axis=1)
        cosine = (source_grad * grad).sum(axis=1) / (strength * np.maximum(lengths, .05))
        return {
            "local_luma_zncc": _zncc(source_luma, luma[texture, None], weights),
            "gradient_zncc": _zncc(source_grad, grad, weights),
            "gradient_direction": float(np.clip(np.average(cosine, weights=weights[:, 0]), -1., 1.)),
        }

    before, after = score(images[1]), score(images[2])
    return {"status": "ok", "support_pixels": support_count, "textured_pixels": texture_count,
            "before": before, "after": after, "delta": {key: after[key] - before[key] for key in before}}

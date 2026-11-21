"""Isolate the visible upper head silhouette in a registered person matte."""

import cv2
import numpy as np


def profile_mask(person, coarse):
    """Return a full-size silhouette, its scoring support, and a preview crop.

    Both inputs are aligned uint8 rasters. The coarse mask locates the head and
    its lower cutoff. The candidate contour itself comes from the person matte.
    """
    if (person.ndim != 2 or person.shape != coarse.shape
            or person.dtype != np.uint8 or coarse.dtype != np.uint8):
        raise ValueError("Expected aligned uint8 grayscale masks")

    foreground = (person >= 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if count < 2:
        raise ValueError("The person matte has no foreground")
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    component = labels == largest
    head_hint = component & (coarse >= 128)
    rows = np.flatnonzero(head_hint.any(axis=1))
    if len(rows) < 8:
        raise ValueError("The coarse head mask has insufficient support in the person matte")

    top = int(rows[0])
    bottom = int(rows[-1]) + 1
    span = bottom - top
    # Reserve the bottom of the heuristic region so its artificial edge is unscored.
    guard = max(3, round(0.15 * span))
    limit = bottom - guard
    columns = np.flatnonzero(component[top:limit].any(axis=0))
    pad = max(4, round(0.12 * span))
    height, width = person.shape
    x = slice(max(0, int(columns[0]) - pad), min(width, int(columns[-1]) + pad + 1))
    y = slice(max(0, top - pad), min(height, bottom + pad))

    target = np.zeros(person.shape, dtype=bool)
    target[y.start:bottom, x] = component[y.start:bottom, x]
    valid = np.zeros_like(target)
    boundary_valid = np.zeros_like(target)
    band = np.s_[y.start:limit, x]
    matte_certain = (person[band] <= 95) | (person[band] >= 160)
    other_component = (labels[band] != 0) & (labels[band] != largest)
    valid[band] = matte_certain & ~other_component
    boundary_valid[band] = ~other_component
    if not (target & valid).any():
        raise ValueError("The person matte has no certain upper-head foreground")
    return {"target": target, "valid": valid, "boundary_valid": boundary_valid, "crop": (y, x)}

"""Use intervening video frames to bridge native-image head texture tracks."""

import cv2
import numpy as np

from side_tracks import track_features


def _timeline(records, stride):
    names = [record["name"] for record in records]
    indices = [int(record["source_index"]) for record in records]
    seconds = [float(record["timestamp_s"]) for record in records]
    if (len(records) < 2 or len(set(names)) != len(names) or stride < 1 or min(indices) < 0
            or not (np.diff(indices) > 0).all() or not np.isfinite(seconds).all()
            or not (np.diff(seconds) > 0).all()):
        raise ValueError("Expected ordered registered source images with unique names, indices and times")
    timeline = [(indices[0], names[0], seconds[0], None)]
    for i in range(len(records) - 1):
        gap = indices[i + 1] - indices[i]
        parts = (gap + stride - 1) // stride
        for j in range(1, parts):
            index = indices[i] + (j * gap + parts // 2) // parts
            fraction = (index - indices[i]) / gap
            t = seconds[i] * (1 - fraction) + seconds[i + 1] * fraction
            timeline.append((index, f"@source:{index}", t, (names[i], names[i + 1], fraction)))
        timeline.append((indices[i + 1], names[i + 1], seconds[i + 1], None))
    return timeline


def _bounds(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return np.array((x, y, x + w, y + h), dtype=np.float64) if w and h else None


def _transport_mask(mask, source, target):
    scale = (target[2:] - target[:2]) / (source[2:] - source[:2])
    shift = target[:2] - scale * source[:2] + .5 * (scale - 1)
    transform = np.column_stack((np.diag(scale), shift))
    return cv2.warpAffine(mask, transform, mask.shape[::-1], flags=cv2.INTER_NEAREST)


def _intermediate_support(masks, boxes, left, right, fraction):
    a, b = boxes[left], boxes[right]
    if a is None or b is None:
        return np.zeros_like(masks[left])
    target = a * (1 - fraction) + b * fraction
    # Restrict feature windows to the overlap of the two transported endpoint masks.
    return _transport_mask(masks[left], a, target) & _transport_mask(masks[right], b, target)


def _read_crops(video, timeline, reference, maps, source_size):
    slots = {index: (name, neighbors) for index, name, _, neighbors in timeline}
    registered = [index for index, _, _, neighbors in timeline if neighbors is None]
    checks = {registered[0], registered[len(registered) // 2], registered[-1]}
    crops, parity = {}, []
    cap = cv2.VideoCapture(str(video))
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        first, last = timeline[0][0], timeline[-1][0]
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise OSError(f"Could not open and seek source video {video}")
        for index in range(first, last + 1):
            if not cap.grab():
                raise OSError(f"Could not read original source frame {index}")
            if index not in slots:
                continue
            name, neighbors = slots[index]
            if neighbors is None and index not in checks:
                continue
            ok, rgb = cap.retrieve()
            if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise OSError(f"Decoder returned a different source frame than {index}")
            height, width = rgb.shape[:2]
            if int(height * source_size[0] / width) != source_size[1]:
                raise ValueError("Original video orientation and distorted COLMAP raster disagree")
            rgb = cv2.resize(rgb, source_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            crop = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR)
            if neighbors is None:
                error = float(np.mean(cv2.absdiff(crop, reference[name])))
                if error > 10:
                    raise ValueError(f"Intermediate raster does not match registered image {name}: MAE {error:.2f}")
                parity.append(error)
            else:
                crops[name] = crop
    finally:
        cap.release()
    return crops, parity


def dense_feature_tracks(video, source_records, gray, support, anchor_projections, bounds,
                         undistort_maps, source_size, *, max_stride=3):
    """Track on refined temporal sampling and return observations only in registered images."""
    records = sorted(source_records, key=lambda record: record["source_index"])
    timeline = _timeline(records, max_stride)
    names = [record["name"] for record in records]
    masks = {name: np.asarray(support[name], dtype=np.uint8) for name in names}
    height, width = gray[names[0]].shape
    if any(mask.shape != (height, width) for mask in masks.values()):
        raise ValueError("Registered head support must share the native pixel raster")
    if len(undistort_maps) != 2 or any(m.shape != (height, width) for m in undistort_maps):
        raise ValueError("The original-to-native video remap does not share the native pixel raster")
    observed_bounds = [bbox for mask in masks.values() if (bbox := _bounds(mask)) is not None]
    if not observed_bounds:
        raise ValueError("Missing head support for dense recovery")
    observed_bounds = np.array(observed_bounds)
    x0, y0 = np.maximum(observed_bounds[:, :2].min(axis=0) - 40, 0).astype(int)
    x1, y1 = np.minimum(observed_bounds[:, 2:].max(axis=0) + 40, [width, height]).astype(int)
    area = np.s_[y0:y1, x0:x1]
    cropped_gray = {name: np.ascontiguousarray(gray[name][area]) for name in names}
    cropped_masks = {name: np.ascontiguousarray(mask[area]) for name, mask in masks.items()}
    boxes = {name: _bounds(mask) for name, mask in cropped_masks.items()}
    middle, parity = _read_crops(video, timeline, cropped_gray,
                                 tuple(np.ascontiguousarray(m[area]) for m in undistort_maps),
                                 source_size)
    cropped_gray.update(middle)
    for _, name, _, neighbors in timeline:
        if neighbors is not None:
            cropped_masks[name] = _intermediate_support(cropped_masks, boxes, *neighbors)
    pixel_shift = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
    anchor_crops = {name: pixel_shift @ projection for name, projection in anchor_projections.items()}
    tracks = track_features([name for _, name, _, _ in timeline], cropped_gray, cropped_masks, anchor_crops,
                            bounds, timestamps={name: t for _, name, t, _ in timeline}, targets=names)
    for result in tracks.values():
        result["points2d"] += np.array((x0, y0))
    stats = {"registered_frames": len(names), "intermediate_frames": len(middle),
             "maximum_source_index_step": max(b[0] - a[0] for a, b in zip(timeline, timeline[1:])),
             "native_crop_xyxy": [int(v) for v in (x0, y0, x1, y1)],
             "registered_raster_check_mae": parity}
    return tracks, stats

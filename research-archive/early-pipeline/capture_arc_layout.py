"""Resolve declared capture intervals against the hash-bound native sample roster."""

import re


def _source_index(value):
    return type(value) is int and value >= 0


def _ranges(layout, side):
    name = f"{side}_source_ranges"
    ranges = layout.get(name)
    if not isinstance(ranges, list):
        raise ValueError(f"{name} must be a list of inclusive [first, last] source-index ranges.")
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(_source_index(value) for value in pair) or pair[0] > pair[1]):
            raise ValueError(f"Invalid inclusive source-index range in {name}: {pair!r}")
    return ranges


def source_layout(layout: dict, rows: list[dict], video_sha256: str, arc: str) -> dict:
    """Return frame IDs for the fitter's source groups and protected holdouts.

    Ranges use zero-based indices in the original MP4, before native sampling.
    Unassigned samples are omitted, and a holdout may be outside both groups.
    """
    if arc not in ("half", "full"):
        raise ValueError("Capture arc must be half or full.")
    if not isinstance(layout, dict):
        raise ValueError("Capture layout must be a JSON object.")
    recorded_sha = layout.get("video_sha256")
    if (not isinstance(recorded_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", recorded_sha)
            or not isinstance(video_sha256, str) or recorded_sha.lower() != video_sha256.lower()):
        raise ValueError("Capture layout video_sha256 does not match the original video.")

    front_ranges = _ranges(layout, "front")
    rear_ranges = _ranges(layout, "rear")
    all_ranges = sorted(front_ranges + rear_ranges)
    if any(previous[1] >= current[0] for previous, current in zip(all_ranges, all_ranges[1:])):
        raise ValueError("Capture source-index ranges must be disjoint across and within sides.")

    if not isinstance(rows, list) or not rows:
        raise ValueError("Capture layout requires a nonempty native sample roster.")
    frames, indices = set(), set()
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each native sample must be an object.")
        frame, index = row.get("frame"), row.get("source_frame_index")
        if not isinstance(frame, str) or not frame or not _source_index(index):
            raise ValueError("Each native sample must have a frame ID and a nonnegative integer source_frame_index.")
        if frame in frames or index in indices:
            raise ValueError("Native frame IDs and source indices must each be unique.")
        frames.add(frame)
        indices.add(index)
        samples.append((frame, index))

    held_values = layout.get("held_source_indices", [])
    if not isinstance(held_values, list) or not all(_source_index(index) for index in held_values):
        raise ValueError("held_source_indices must be a list of nonnegative integer source indices.")
    held = set(held_values)
    if len(held) != len(held_values) or not held <= indices:
        raise ValueError("Held source indices must be unique and exist in the native sample roster.")

    front = [frame for frame, index in samples if any(first <= index <= last for first, last in front_ranges)]
    rear = [frame for frame, index in samples if any(first <= index <= last for first, last in rear_ranges)]
    holdouts = [frame for frame, index in samples if index in held]
    held_frames = set(holdouts)
    if not set(front) - held_frames or (arc == "full" and not set(rear) - held_frames):
        raise ValueError("Capture layout must leave usable front samples and, for a full arc, usable rear samples.")

    return dict(front_frame_ids=front, rear_frame_ids=rear, rear_coverage_observed=bool(rear), retained_holdouts=holdouts)

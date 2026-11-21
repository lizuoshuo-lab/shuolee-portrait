"""Choose repeatable training frames and time-block holdouts from head observations."""

import math

import numpy as np


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _quality(rows):
    rankings = []
    for name, direction in (("face_score", 1), ("reprojection_px", -1),
                            ("blur_score", 1), ("visibility", 1)):
        values = np.array([row[name] if row[name] is not None else np.nan for row in rows])
        known = np.isfinite(values)
        if known.sum() < 2:
            continue
        _, ranks, counts = np.unique(direction * values[known], return_inverse=True, return_counts=True)
        # Average the ranks of ties and give missing optional measurements a neutral rank.
        percentiles = (np.cumsum(counts) - (counts + 1) / 2) / (known.sum() - 1)
        ranking = np.full(len(rows), 0.5)
        ranking[known] = percentiles[ranks]
        rankings.append(ranking)
    return np.mean(rankings, axis=0) if rankings else np.full(len(rows), 0.5)


def _pick(candidates, positions, quality, selected):
    indices = np.flatnonzero(candidates)
    if not selected:
        return int(indices[np.argmax(quality[indices])])
    distance = np.linalg.norm(positions[indices, None] - positions[selected], axis=-1).min(axis=1)
    scores = distance * (0.5 + 0.5 * quality[indices])
    order = np.lexsort((-quality[indices], -scores))
    return int(indices[order[0]])


def _coverage(rows):
    if not rows:
        return {"count": 0, "yaw_deg": None, "timestamp_s": None}
    yaws = [row["yaw_deg"] for row in rows]
    times = [row["timestamp_s"] for row in rows]
    return {"count": len(rows), "yaw_deg": [min(yaws), max(yaws)],
            "timestamp_s": [min(times), max(times)]}


def select_head_frames(records, *, train_count=16, holdout_count=4,
                       yaw_limit_deg=30.0, holdout_block_seconds=0.5):
    """Return sorted filename lists, reserved intervals, and reasons for every frame.

    Records need a unique name, timestamp_s, yaw_deg, face_score, and reprojection_px.
    A higher face_score, optional blur_score (sharpness), and optional visibility are better.
    A lower reprojection_px is better. holdout_block_seconds is the full interval width.
    """
    if (type(train_count) is not int or type(holdout_count) is not int
            or train_count < 0 or holdout_count < 0):
        raise ValueError("Frame counts must be nonnegative integers")
    if (not math.isfinite(yaw_limit_deg) or yaw_limit_deg <= 0
            or not math.isfinite(holdout_block_seconds) or holdout_block_seconds <= 0):
        raise ValueError("Yaw limit and holdout interval width must be positive and finite")

    decisions = {}
    for record in records:
        name = record["name"]
        if not isinstance(name, str) or not name or name in decisions:
            raise ValueError(f"A unique, nonempty frame name is required: {name!r}")
        row = {field: _number(record.get(field)) for field in (
            "timestamp_s", "yaw_deg", "face_score", "reprojection_px", "blur_score", "visibility"
        )}
        row.update(name=name, status="rejected")
        if record.get("status") is not None:
            row["observation_status"] = str(record["status"])
        if row["timestamp_s"] is None or row["timestamp_s"] < 0:
            row["reason"] = "invalid_timestamp"
        elif any(row[field] is None for field in ("yaw_deg", "face_score", "reprojection_px")):
            row["reason"] = "missing_face_or_pose_evidence"
        elif abs(row["yaw_deg"]) > yaw_limit_deg:
            row["reason"] = "outside_yaw_limit"
        elif (row["face_score"] <= 0 or row["reprojection_px"] < 0
              or (row["visibility"] is not None and row["visibility"] <= 0)):
            row["reason"] = "invalid_face_evidence"
        else:
            row.update(status="eligible", reason="not_selected_within_budget")
        decisions[name] = row

    rows = sorted((row for row in decisions.values() if row["status"] == "eligible"),
                  key=lambda row: (row["timestamp_s"], row["name"]))
    train = []
    holdout = []
    blocks = []
    if rows:
        times = np.array([row["timestamp_s"] for row in rows])
        yaw = np.array([row["yaw_deg"] for row in rows])
        span = times[-1] - times[0]
        positions = np.column_stack((yaw / yaw_limit_deg, (times - times[0]) / (span or 1)))
        quality = _quality(rows)
        for row, score in zip(rows, quality):
            row["quality_score"] = float(score)

        half_width = holdout_block_seconds / 2
        left = np.searchsorted(times, times - half_width, side="left")
        right = np.searchsorted(times, times + half_width, side="right")
        available = np.ones(len(rows), dtype=bool)
        holdout_candidates = available.copy()
        for _ in range(holdout_count):
            prefix = np.r_[0, np.cumsum(available)]
            unaffected = prefix[-1] - (prefix[right] - prefix[left])
            viable = holdout_candidates & (unaffected >= int(train_count > 0))
            if not viable.any():
                break
            preserves_budget = viable & (unaffected >= train_count)
            if preserves_budget.any():
                viable = preserves_budget
            else:
                viable &= unaffected == unaffected[viable].max()
            index = _pick(viable, positions, quality, holdout)
            start, end = left[index], right[index]
            for row in rows[start:end]:
                row.update(status="reserved", reason="inside_holdout_time_block")
            rows[index].update(status="holdout", reason="view_and_time_coverage_holdout")
            blocks.append({"frame": rows[index]["name"],
                           "start_s": float(times[index] - half_width),
                           "end_s": float(times[index] + half_width),
                           "reserved_frames": [row["name"] for row in rows[start:end]]})
            holdout.append(index)
            available[start:end] = False
            holdout_candidates &= np.abs(times - times[index]) > holdout_block_seconds

        for _ in range(min(train_count, int(available.sum()))):
            index = _pick(available, positions, quality, train)
            rows[index].update(status="train", reason="quality_and_view_time_coverage")
            train.append(index)
            available[index] = False

    trained = [rows[index] for index in sorted(train)]
    held = [rows[index] for index in sorted(holdout)]
    return {
        "settings": {"train_count": train_count, "holdout_count": holdout_count,
                     "yaw_limit_deg": yaw_limit_deg, "holdout_block_seconds": holdout_block_seconds},
        "train": [row["name"] for row in trained],
        "holdout": [row["name"] for row in held],
        "holdout_blocks": sorted(blocks, key=lambda block: block["start_s"]),
        "train_shortfall": train_count - len(trained),
        "holdout_shortfall": holdout_count - len(held),
        "coverage": {"eligible": _coverage(rows), "train": _coverage(trained), "holdout": _coverage(held)},
        "decisions": dict(sorted(decisions.items())),
    }

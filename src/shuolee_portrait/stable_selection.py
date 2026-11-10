"""Deterministic capture-arc roster and reference for one-shot portrait delivery."""
import math
import re

import numpy as np


BODY = np.array([0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 13, 10, 14, 11])
FEET = np.arange(19, 25)
FACE = np.arange(67, 118)
POSE_FIELDS = (
    "global_orient", "body_pose_axis_angle", "camera_translation", "pose_embedding",
    "left_hand_pose", "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(value):
    return type(value) is int and value >= 0


def _source_rows(rows):
    names = [row["frame"] for row in rows]
    indices = [row["source_frame_index"] for row in rows]
    _require(bool(names) and all(isinstance(name, str) and name for name in names)
             and all(_index(index) for index in indices)
             and len(set(names)) == len(set(indices)) == len(names),
             "Capture frame IDs and original source indices must each be unique.")
    return sorted(zip(names, indices), key=lambda pair: pair[1])


def source_layout(layout, rows, video_sha256, arc):
    """Resolve source-video ranges, preserving holdouts even outside either arc."""
    _require(arc in ("half", "full"), "Capture arc must be half or full.")
    _require(isinstance(layout, dict), "Capture layout must be an object.")
    digest = layout.get("video_sha256")
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest)
             and isinstance(video_sha256, str) and digest.lower() == video_sha256.lower(),
             "Capture layout belongs to a different original video.")
    ranges = {}
    for side in ("front", "rear"):
        intervals = layout.get(side + "_source_ranges")
        _require(isinstance(intervals, list)
                 and all(isinstance(pair, list) and len(pair) == 2 and all(map(_index, pair))
                         and pair[0] <= pair[1] for pair in intervals),
                 f"Expected inclusive original-frame ranges for {side}.")
        ranges[side] = intervals
    combined = sorted(ranges["front"] + ranges["rear"])
    _require(all(first[1] < second[0] for first, second in zip(combined, combined[1:])),
             "Capture ranges must be disjoint, including within a side.")
    samples = _source_rows(rows)
    indices = {index for _, index in samples}
    held = layout.get("held_source_indices", [])
    _require(isinstance(held, list) and all(map(_index, held))
             and len(set(held)) == len(held) and set(held) <= indices,
             "Protected source indices must be unique native samples.")
    groups = {side: [name for name, index in samples if any(a <= index <= b for a, b in intervals)]
              for side, intervals in ranges.items()}
    holdouts = [name for name, index in samples if index in held]
    _require(set(groups["front"]) - set(holdouts)
             and (arc == "half" or set(groups["rear"]) - set(holdouts)),
             "Capture ranges must leave usable samples on each requested side.")
    return dict(front_frame_ids=groups["front"], rear_frame_ids=groups["rear"],
                rear_coverage_observed=bool(groups["rear"]), retained_holdouts=holdouts)


def _bins(frames, count):
    return [group.tolist() for group in np.array_split(np.asarray(frames, dtype=object), count)]


def selection_scope(inputs, *, arc="half", capture_layout=None, front_last_source_index=None):
    """Freeze the eligible source bins and exclusions before any body fitting."""
    _require(arc in ("half", "full"), "Capture arc must be half or full.")
    _require(capture_layout is None or front_last_source_index is None,
             "Choose one capture layout or source-index boundary.")
    rows = inputs["rows"]
    samples = _source_rows(rows)
    names = [name for name, _ in samples]
    targets = dict(front=24, rear=12 if arc == "full" else 0)
    if capture_layout is not None:
        layout = source_layout(capture_layout, rows, inputs["source_manifest"].get("video_sha256"), arc)
        front, rear, holdouts = (layout[k] for k in ("front_frame_ids", "rear_frame_ids", "retained_holdouts"))
    elif arc == "full":
        _require(_index(front_last_source_index),
                 "A full capture requires an explicit front/rear layout or source-index boundary.")
        front = [name for name, index in samples if index <= front_last_source_index]
        rear = [name for name, index in samples if index > front_last_source_index]
        holdouts = []
    else:
        _require(front_last_source_index is None, "A front/rear boundary requires full capture.")
        front, rear, holdouts = names, [], []
    _require(bool(front) and (arc == "half" or bool(rear)), "Missing source coverage on a requested side.")
    bins = _bins(front, targets["front"])
    if targets["rear"]:
        bins += _bins(rear, targets["rear"])
    if not holdouts:
        hold_bins = [len(bins) * k // 16 for k in (2, 6, 10, 14)]
        held = {bins[i][len(bins[i]) // 2] for i in hold_bins if len(bins[i]) > 1}
        holdouts = [name for name in names if name in held]
    candidates = set(name for group in bins for name in group) - set(holdouts)
    return dict(arc=arc, view_targets=targets, source_view_bins=bins, retained_holdouts=holdouts,
                fit_frame_ids=[name for group in bins for name in group if name in candidates])


def shape_anchors(inputs, groups):
    """Select the original RTMW-reliability leaders from the protected fit scope."""
    reliability = np.asarray(inputs.get("source_reliability"))
    ids = inputs["ids"]
    _require(reliability.shape == (len(ids),) and np.issubdtype(reliability.dtype, np.floating)
             and np.isfinite(reliability).all(), "Expected one aligned finite RTMW reliability per source frame.")
    by_name = dict(zip(ids, reliability))
    bins, fit = groups["source_view_bins"], set(groups["fit_frame_ids"])
    _require(not fit & set(groups["retained_holdouts"]), "A protected holdout reached the shape pool.")
    anchors = {}
    offset = 0
    for side, target in groups["view_targets"].items():
        pool = [name for group in bins[offset:offset + target] for name in group if name in fit]
        _require(not target or pool, f"No usable {side} source samples for a shared shape.")
        anchors[side] = [max(group, key=by_name.__getitem__) for group in _bins(pool, min(target, len(pool)))] if pool else []
        offset += target
    return anchors


def _float_array(value):
    array = np.asarray(value)
    _require(np.issubdtype(array.dtype, np.floating) and np.isfinite(array).all(), "Nonfinite or nonfloating fitted value.")
    return array


def _region(observed, projected, indices, height):
    target = observed[indices]
    visible = target[:, 2] > 0
    count = int(visible.sum())
    if not count:
        return dict(points=0, mean_confidence=None, rms_over_height=None)
    confidence = target[visible, 2].astype(np.float64)
    delta = (projected[indices][visible].astype(np.float64) - target[visible, :2]) / height
    rms = np.sqrt(np.average(np.sum(delta ** 2, axis=1), weights=confidence ** 2))
    return dict(points=count, mean_confidence=round(float(confidence.mean()), 8),
                rms_over_height=round(float(rms), 8))


def _candidate(row, pose, observed, K, shared, topology, report, screen_pose):
    if report is not None:
        _require(report.get("state") == "complete" and report.get("frame_id") == row["frame"]
                 and report.get("source_frame_index") == row["source_frame_index"]
                 and report.get("source_image_sha256") == row["source_sha256"]
                 and all(report.get(key) is True for key in ("freeze_betas", "translation_applied_once", "full_K_integer_centers")),
                 "Official frame receipt does not match its frozen inputs.")
    if "frame_id" in pose:
        _require(str(np.asarray(pose["frame_id"]).item()) == row["frame"], "Winner frame identity differs.")
    beta = _float_array(pose["betas"])
    _require(beta.shape == (1, 10) and (shared is None or np.array_equal(beta, shared))
             and np.array_equal(pose["K_integer_centers"], K),
             "Winner shape or complete native K differs.")
    for field in POSE_FIELDS:
        value = _float_array(pose[field])
        _require(value.ndim == 2 and value.shape[0] == 1 and value.shape[1] > 0, f"Invalid {field}.")
    vertices, faces = _float_array(pose["vertices_camera"]), np.asarray(pose["faces"])
    _require(vertices.ndim == 3 and vertices.shape[0] == 1 and vertices.shape[2] == 3 and vertices.shape[1] > 0
             and np.all(vertices[..., 2] > .05), "Invalid winner mesh or native camera depth.")
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0 and faces.dtype.kind in "iu"
             and faces.min() >= 0 and faces.max() < vertices.shape[1]
             and (topology is None or (vertices.shape == topology[0] and np.array_equal(faces, topology[1]))),
             "Winner topology differs.")
    projected = _float_array(pose["joints_projected_native"])
    _require(observed.shape == (118, 3) and projected.shape == (1, 118, 2)
             and np.isfinite(observed).all() and np.all((observed[:, 2] >= 0) & (observed[:, 2] <= 1)),
             "Invalid native detector or projected observations.")
    if "observed_native_keypoints" in pose:
        _require(np.array_equal(pose["observed_native_keypoints"], observed[None]),
                 "Winner detector targets differ.")
    box = np.asarray(row.get("initial_box_xyxy", []), dtype=np.float64)
    height = box[3] - box[1] if box.shape == (4,) and np.isfinite(box).all() else 0.
    flags = []
    if screen_pose:
        _require(height > 0, "Missing body box for the original pose screen.")
        visible_body = BODY[observed[BODY, 2] ** 2 > 0]
        _require(len(visible_body) > 0, "No supported body measurements.")
        body_error = np.linalg.norm(projected[0, visible_body] - observed[visible_body, :2], axis=1)
        if float(body_error.max()) / height > .03:
            flags.append("original_supported_body_reprojection_over_existing3percent_height")
        visible_feet = FEET[observed[FEET, 2] > 0]
        foot_error = np.linalg.norm(projected[0, visible_feet] - observed[visible_feet, :2], axis=1)
        if np.any(foot_error / height > .03):
            flags.append("feet_reprojection_over_existing3percent_height")
    height = height if height > 0 else row["wh"][1]
    scores = {name: _region(observed, projected[0], indices, height)
              for name, indices in (("body", BODY), ("feet", FEET), ("face", FACE))}
    return scores, flags, beta, (vertices.shape, faces)


def _body_rank(record):
    body, feet = record["scores"]["body"], record["scores"]["feet"]
    body_rms, feet_rms = body["rms_over_height"], feet["rms_over_height"]
    return (body_rms is None, (body_rms or 0.) + .25 * (feet_rms or 0.),
            -(body["mean_confidence"] or 0.), feet_rms is None,
            -(feet["mean_confidence"] or 0.), record["source_index"])


def select_frames(inputs, winners, *, scope, reports=None, shared_betas=None, screen_pose=True):
    """Rank official technical winners using the original automatic pose screen."""
    rows = {row["frame"]: (index, row) for index, row in enumerate(inputs["rows"])}
    _require(list(rows) == list(inputs["ids"]), "Official inputs and observation order differ.")
    targets, bins = scope["view_targets"], scope["source_view_bins"]
    expected_targets = dict(front=24, rear=12 if scope["arc"] == "full" else 0)
    holdouts = scope["retained_holdouts"]
    fit = set(scope["fit_frame_ids"])
    members = [name for group in bins for name in group]
    _require(scope["arc"] in ("half", "full") and targets == expected_targets and len(bins) == sum(targets.values())
             and len(set(members)) == len(members) and len(set(holdouts)) == len(holdouts)
             and set(members) | set(holdouts) <= set(rows) and fit == set(members) - set(holdouts)
             and set(winners) <= fit and (reports is None or set(reports) == set(winners)),
             "Official winners or protection differ from the frozen source roster.")
    shared = None if shared_betas is None else _float_array(shared_betas).reshape(1, 10)
    topology = None
    selected, reasons, coverage = set(), {}, {}
    offset = 0
    for side in ("front", "rear"):
        target = targets[side]
        interval = range(offset, offset + target)
        eligible, occupied, leaders = [], [], []
        for i in interval:
            group = []
            for frame in bins[i]:
                record = reasons[frame] = dict(side=side, source_bin=i)
                if frame not in fit or frame not in winners:
                    record["status"] = "protected_holdout" if frame not in fit else "no_fit_receipt"
                    continue
                index, row = rows[frame]
                try:
                    scores, flags, shared, topology = _candidate(
                        row, winners[frame], np.asarray(inputs["targets"][index]), inputs["K"][index],
                        shared, topology, None if reports is None else reports[frame], screen_pose)
                except (KeyError, TypeError, ValueError) as error:
                    record.update(status="technical_rejection", detail=str(error))
                    continue
                record.update(status="technically_valid", source_index=row["source_frame_index"],
                              scores=scores, visible_face=scores["face"]["points"] > 0)
                if screen_pose:
                    record["screen_flags"] = flags
                if flags:
                    record["status"] = "pose_screen_rejection"
                    continue
                group.append(frame)
            if group:
                occupied.append(i)
                leaders.append(min(group, key=lambda frame: (_body_rank(reasons[frame]), frame)))
                eligible.extend(group)
        for frame in leaders:
            selected.add(frame)
            reasons[frame]["status"] = "selected_bin_leader"
        fill = sorted((frame for frame in eligible if frame not in selected),
                      key=lambda frame: (_body_rank(reasons[frame]), frame))[:target-len(leaders)]
        for frame in fill:
            selected.add(frame)
            reasons[frame]["status"] = "selected_quota_fill"
        count = len(leaders) + len(fill)
        coverage[side] = dict(target=target, eligible=len(eligible), selected=count, source_bins=target,
                              occupied_source_bins=occupied,
                              missing_source_bins=[index for index in interval if index not in occupied],
                              shortfall=target-count)
        offset += target
    _require(bool(selected), "No technically valid non-holdout poses for automatic training.")
    for frame in holdouts:
        reasons.setdefault(frame, dict(status="protected_holdout"))
    ordered = sorted(selected, key=lambda frame: (reasons[frame]["source_index"], frame))
    visible = [frame for frame in ordered if reasons[frame]["visible_face"]]
    reference = min(visible or ordered, key=lambda frame: (
        reasons[frame]["scores"]["face"]["rms_over_height"] if visible else math.inf,
        -(reasons[frame]["scores"]["face"]["mean_confidence"] or 0.) if visible else 0.,
        _body_rank(reasons[frame]), frame))
    result = dict(arc=scope["arc"], target_frames=sum(targets.values()), selected_frames=ordered,
                  retained_holdouts=holdouts, reference_frame=reference, coverage=coverage, reasons=reasons,
                  full_roster=len(ordered) == sum(targets.values()), quality_accepted=False)
    if screen_pose:
        result["pose_screen"] = "original_body_and_feet_v1"
    return result


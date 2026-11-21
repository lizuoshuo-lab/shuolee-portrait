"""Deterministic training roster from saved official SMPLify-X fit receipts."""

import math
from zipfile import BadZipFile

import numpy as np

from pose_training_gate151 import POSE_FIELDS, require
from review_full_sequence123 import read, relocate, sha
from screen_official_body_contract124 import BODY, IGNORED, NEW_FLAG, measure
from select_quality_frames124 import POLICY


BODY_INDICES = BODY[~np.isin(BODY, IGNORED)]
FEET_INDICES = np.arange(19, 25)
FACE_INDICES = np.arange(67, 118)
ORIGINAL_JOINT_WEIGHTS = np.ones(118)
ORIGINAL_JOINT_WEIGHTS[list(IGNORED)] = 0


def _region_score(observed, projected, indices, height):
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


def _candidate(row, receipt, beta, K, wh, topology, *, screen_pose):
    require(receipt["freeze_betas"] is True and receipt["source_index"] == row["source_frame_index"],
            "fit_receipt_source_or_shared_shape_mismatch")
    pose_path = relocate(receipt["output"])
    require(pose_path.is_file() and sha(pose_path) == receipt["sha256"], "fit_archive_missing_or_changed")
    for key, digest in (("image", "source_sha256"), ("mask", "mask_sha256")):
        path = relocate(row[key])
        require(path.is_file() and sha(path) == row[digest], f"source_{key}_missing_or_changed")

    raster_K = K.copy()
    raster_K[:2, 2] += .5
    require(np.array_equal(row["K_integer_centers"], K) and np.array_equal(row["K"], raster_K)
            and np.array_equal(row["wh"], wh) and np.array_equal(row["w2c"], np.eye(4)),
            "source_camera_mismatch")
    with np.load(pose_path, allow_pickle=False) as pose:
        require(np.array_equal(pose["betas"], beta), "fit_shared_shape_mismatch")
        require(np.array_equal(pose["K_integer_centers"], K), "fit_native_camera_mismatch")
        for name in (*POSE_FIELDS, "pose_embedding"):
            values = pose[name]
            require(values.ndim == 2 and values.shape[0] == 1 and values.shape[1] > 0
                    and np.issubdtype(values.dtype, np.floating) and np.isfinite(values).all(),
                    f"invalid_{name}")
        vertices, faces = pose["vertices_camera"], pose["faces"]
        require(vertices.ndim == 3 and vertices.shape[0] == 1 and vertices.shape[1] > 0
                and vertices.shape[2] == 3 and np.issubdtype(vertices.dtype, np.floating)
                and np.isfinite(vertices).all() and np.all(vertices[..., 2] > .05), "invalid_mesh_or_depth")
        require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
                and np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0
                and faces.max() < vertices.shape[1]
                and (topology is None or (vertices.shape == topology[0] and np.array_equal(faces, topology[1]))),
                "fit_topology_mismatch")
        observed, projected = pose["observed_native_keypoints"], pose["joints_projected_native"]
        require(observed.shape == (1, 118, 3) and projected.shape == (1, 118, 2)
                and np.issubdtype(observed.dtype, np.floating) and np.issubdtype(projected.dtype, np.floating)
                and np.isfinite(observed).all() and np.isfinite(projected).all()
                and np.all((observed[..., 2] >= 0) & (observed[..., 2] <= 1)), "invalid_native_observations")

    box = np.asarray(row.get("initial_box_xyxy", []), dtype=np.float64)
    height = box[3] - box[1] if box.shape == (4,) and np.isfinite(box).all() else 0.
    flags = []
    if screen_pose:
        require(height > 0, "missing_body_box_for_pose_screen")
        body = measure(observed[0], projected[0], height, ORIGINAL_JOINT_WEIGHTS)
        if body["supported_body_gate_failed"]:
            flags.append(NEW_FLAG)
        feet = FEET_INDICES[observed[0, FEET_INDICES, 2] > 0]
        feet_error = np.linalg.norm(projected[0, feet] - observed[0, feet, :2], axis=1)
        if np.any(feet_error / height > POLICY["feet_max_over_height"]):
            flags.append("feet_reprojection_over_existing3percent_height")
    if height <= 0:
        height = wh[1]
    scores = {name: _region_score(observed[0], projected[0], indices, height)
              for name, indices in (("body", BODY_INDICES), ("feet", FEET_INDICES), ("face", FACE_INDICES))}
    return scores, flags, (vertices.shape, faces)


def _body_rank(record):
    body, feet = record["scores"]["body"], record["scores"]["feet"]
    body_rms, feet_rms = body["rms_over_height"], feet["rms_over_height"]
    combined = (body_rms or 0.) + .25 * (feet_rms or 0.)
    return (body_rms is None, combined, -(body["mean_confidence"] or 0.),
            feet_rms is None, -(feet["mean_confidence"] or 0.), record["source_index"])


def select_automatic_fit(fit_dir, arc, *, screen_pose=True):
    """Select up to 24 front and, for a full arc, 12 rear training frames."""
    require(arc in ("half", "full"), "Automatic selection arc must be half or full.")
    fit_dir = relocate(fit_dir).resolve()
    plan_path, fit_path, contacts_path, shape_path = (fit_dir / name for name in
        ("plan.json", "result.json", "contacts.json", "shared-shape.json"))
    plan_sha, fit_sha = sha(plan_path), sha(fit_path)
    plan, result, shape = read(plan_path), read(fit_path), read(shape_path)
    require(result["plan_sha256"] == plan_sha and plan["contacts_sha256"] == sha(contacts_path)
            and sha(shape_path) == (result.get("shared_shape_sha256") or plan.get("neutral_shape_sha256")),
            "Fit plan, contacts or shared-shape receipt changed.")

    beta = np.asarray(shape["betas"], dtype=np.float32)
    K = np.asarray(plan["K_integer_centers"], dtype=np.float64)
    wh = plan["native_wh"]
    require(beta.shape == (1, 10) and np.isfinite(beta).all() and shape["model_unit_scale"] == 1.
            and np.array_equal(result.get("fixed_shape", beta), beta), "Invalid frozen unit-scale body shape.")
    require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0
            and np.array_equal(K[2], [0, 0, 1]) and len(wh) == 2
            and all(type(value) is int and value > 0 for value in wh), "Invalid native camera.")

    bins = plan["source_view_bins"]
    planned_targets = plan["view_targets"]
    front_count = planned_targets["front"]
    require(front_count > 0 and front_count + planned_targets["rear"] == len(bins),
            "Source view bins differ from the front/rear plan.")
    bin_members = [frame for group in bins for frame in group]
    ids, holdouts = plan["source_frames"], plan["retained_holdouts"]
    require(len(ids) == len(set(ids)) and len(holdouts) == len(set(holdouts))
            and len(bin_members) == len(set(bin_members)) and set(ids) <= set(bin_members),
            "Invalid source frame, holdout or view-bin roster.")
    contact_rows, fit_rows = read(contacts_path)["records"], result["frames"]
    contacts = {row["frame"]: row for row in contact_rows}
    receipts = {row["frame"]: row for row in fit_rows}
    require(len(contacts) == len(contact_rows) and len(receipts) == len(fit_rows)
            and contacts.keys() <= set(ids) and receipts.keys() <= set(ids),
            "Contact or fit receipts differ from the candidate plan.")

    face_absent = set(plan.get("face_absence_frames", []))
    targets = {"front": 24, "rear": 12 if arc == "full" else 0}
    ranges = {"front": range(front_count), "rear": range(front_count, len(bins))}
    reasons, chosen, coverage = {}, set(), {}
    topology = None
    for side, bin_range in ranges.items():
        eligible, occupied = [], []
        for bin_index in bin_range:
            candidates = []
            for frame in bins[bin_index]:
                record = dict(side=side, source_bin=bin_index)
                reasons[frame] = record
                if frame in holdouts:
                    record["status"] = "protected_holdout"
                elif targets[side] == 0:
                    record["status"] = "outside_requested_arc"
                elif frame not in contacts or frame not in receipts:
                    record["status"] = "no_fit_or_contact_receipt"
                else:
                    row = contacts[frame]
                    try:
                        source_index = row["source_frame_index"]
                        require(type(source_index) is int and source_index >= 0, "invalid_source_index")
                        scores, flags, topology = _candidate(row, receipts[frame], beta, K, wh, topology,
                                                            screen_pose=screen_pose)
                    except (BadZipFile, KeyError, OSError, TypeError, ValueError) as error:
                        record.update(status="technical_rejection", detail=str(error))
                        continue
                    record.update(status="technically_valid", source_index=source_index, scores=scores,
                                  visible_face=frame not in face_absent and scores["face"]["points"] > 0)
                    if screen_pose:
                        record["screen_flags"] = flags
                    if flags:
                        record["status"] = "pose_screen_rejection"
                        continue
                    candidates.append(frame)
            if candidates:
                occupied.append(bin_index)
                best = min(candidates, key=lambda frame: (_body_rank(reasons[frame]), frame))
                eligible.extend(candidates)
                if sum(reasons[frame]["side"] == side for frame in chosen) < targets[side]:
                    chosen.add(best)
                    reasons[best]["status"] = "selected_bin_leader"
        remaining = targets[side] - sum(reasons[frame]["side"] == side for frame in chosen)
        ranked = sorted((frame for frame in eligible if frame not in chosen),
                        key=lambda frame: (_body_rank(reasons[frame]), frame))
        for frame in ranked[:remaining]:
            chosen.add(frame)
            reasons[frame]["status"] = "selected_quota_fill"
        selected_count = sum(reasons[frame]["side"] == side for frame in chosen)
        coverage[side] = dict(target=targets[side], eligible=len(eligible), selected=selected_count,
                              source_bins=len(bin_range), occupied_source_bins=occupied,
                              missing_source_bins=[index for index in bin_range if index not in occupied] if targets[side] else [],
                              shortfall=targets[side] - selected_count)
    for frame in holdouts:
        reasons.setdefault(frame, dict(status="protected_holdout"))

    if not screen_pose:
        require(chosen, "No technically valid non-holdout poses for automatic training.")
    selected = sorted(chosen, key=lambda frame: (reasons[frame]["source_index"], frame))
    face_frames = [frame for frame in selected if reasons[frame]["visible_face"]]
    reference = min(face_frames or selected, key=lambda frame: (
        reasons[frame]["scores"]["face"]["rms_over_height"] if face_frames else math.inf,
        -(reasons[frame]["scores"]["face"]["mean_confidence"] or 0.) if face_frames else 0.,
        _body_rank(reasons[frame]), frame), default=None)
    result = dict(fit_dir=str(fit_dir), arc=arc, target_frames=sum(targets.values()), selected_frames=selected,
                  retained_holdouts=holdouts, reference_frame=reference, source_fit_sha256=fit_sha,
                  source_plan_sha256=plan_sha, coverage=coverage, reasons=reasons,
                  full_roster=len(selected) == sum(targets.values()), quality_accepted=False)
    if screen_pose:
        result["pose_screen"] = "original_body_and_feet_v1"
    return result

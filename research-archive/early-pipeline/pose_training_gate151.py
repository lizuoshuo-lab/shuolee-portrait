"""Select reviewed fits, then enforce their frame list at the SH0 launch."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from review_full_sequence123 import relocate
from select_quality_frames124 import select
from update_leafwoman129_state import read, sha


TARGETS = {"half": 16, "full": 30}
SCORES = ("body_weighted_rms_px", "face_weighted_rms_px", "face_sharpness", "body_confidence")
FACE_SCORES = ("face_weighted_rms_px", "face_sharpness")
POSE_FIELDS = ("global_orient", "body_pose_axis_angle", "camera_translation",
               "left_hand_pose", "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression")
# Match the existing SMPL-X to Gaussian zero-offset mesh replay tolerance.
MESH_REPLAY_TOLERANCE = 2e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_fit(fit_dir, arc, decisions_path=None, *, balance_reviewed=False):
    fit_dir = fit_dir.resolve()
    plan_path, result_path = (fit_dir / name for name in ("plan.json", "result.json"))
    review_path = fit_dir / "review/result.json"
    plan, result, review = (read(path) for path in (plan_path, result_path, review_path))
    fit_sha = sha(result_path)
    ids = plan["source_frames"]
    require(ids and len(ids) == len(set(ids)), "Missing or duplicate fit candidates.")
    require((result["fitting_complete"] is True or result.get("full_fits_recorded") is True)
            and result["plan_sha256"] == sha(plan_path)
            and [row["frame"] for row in result["frames"]] == ids,
            "Fit result does not match its candidate plan.")
    require(review["source_fit_sha256"] == fit_sha
            and [row["frame"] for row in review["records"]] == ids
            and all(row["source_fit_sha256"] == fit_sha for row in review["records"]),
            "Review does not belong to this exact fit.")

    targets = plan.get("view_targets")
    if balance_reviewed:
        require(targets is not None and decisions_path is not None,
                "Balanced selection requires explicit front/rear targets and visual review.")
    if targets is None:
        target = TARGETS[arc]
    else:
        require(targets == {"front": 24, "rear": 12 if arc == "full" else 0},
                "The view budget must match the recorded arc.")
        target = sum(targets.values())
    bins = plan["source_view_bins"]
    positions = {frame: i for i, frames in enumerate(bins) for frame in frames}
    require(len(bins) == target and len(positions) == sum(map(len, bins))
            and set(ids) <= positions.keys(), "Require unique, preregistered view bins for this arc.")
    require(arc == "half" or plan.get("rear_coverage_observed") is True,
            "Full-orbit selection requires observed rear views.")
    heldouts = set(plan["retained_holdouts"])
    face_absence = set(plan.get("face_absence_frames", ()))
    require(face_absence <= set(ids), "Face support roster does not match the fitted frames.")
    decisions = {}
    if decisions_path is not None:
        document = read(decisions_path)
        decisions = document["records"]
        require(document["source_fit_sha256"] == fit_sha
                and document["fit_review_sha256"] == sha(review_path)
                and decisions.keys() <= set(ids), "Pose decisions belong to a different review.")

    rows, accepted, reasons = [], {}, {}
    for item in review["records"]:
        frame = item["frame"]
        require(isinstance(item["screen_flags"], list), f"Invalid screening record for {frame}.")
        flags = list(item["screen_flags"])
        if result.get("full_fits_recorded") is True:
            depth = item.get("min_camera_mesh_depth")
            require(isinstance(depth, (float, int)), f"Missing mesh depth review for {frame}.")
            if not math.isfinite(depth) or depth <= .05:
                require("invalid_camera_mesh_depth" in flags, f"Unsafe depth was not screened for {frame}.")
        if frame in heldouts:
            flags.append("held_out")
        required_scores = (*SCORES, "time_seconds")
        if frame in face_absence:
            require(all(item.get(k) is None for k in FACE_SCORES), f"Expected missing face observations for {frame}.")
            required_scores = ("body_weighted_rms_px", "body_confidence", "time_seconds")
        if any(not isinstance(item.get(k), (int, float)) or not math.isfinite(item[k]) for k in required_scores):
            flags.append("missing_or_nonfinite_measurement")
        decision = decisions.get(frame, {"status": "unreviewed"})
        status = decision["status"]
        require(status in {"accepted", "rejected", "uncertain", "unreviewed"}, f"Invalid review status for {frame}.")
        row = dict(item, screen_flags=flags)
        # A bad fitted root can flip the yaw. Use the preregistered source
        # positions for diversity. These synthetic degrees only encode bins.
        row["yaw_proxy_deg"] = positions[frame] * (180 if arc == "half" else 360) / target
        if not flags and status == "accepted":
            evidence = [item[key] for key in ("native_image", "side_image")]
            require(decision.get("depth_reviewed") is True
                    and [e["sha256"] for e in evidence] == decision.get("image_sha256")
                    and all(sha(review_path.parent / e["name"]) == e["sha256"] for e in evidence),
                    f"Missing or stale native/side review evidence for {frame}.")
            accepted[frame] = dict(status=status, depth_reviewed=True, evidence_sha256=evidence[1]["sha256"])
        reasons[frame] = dict(screen_flags=flags, review_status=status, review_reason=decision.get("reason"))
        if not flags and frame in accepted:
            rows.append(row)

    # Preserve one winner in every occupied source bin and report empty bins.
    winners, quality = {}, {}
    if rows:
        # Missing face observations are neutral for ranking, while the saved
        # measurements remain unknown and the body/visual gates still apply.
        for key in FACE_SCORES:
            measured = [row[key] for row in rows if row.get(key) is not None]
            neutral = float(np.median(measured)) if measured else 0.
            for row in rows:
                if row.get(key) is None:
                    row[key] = neutral
        for row in select(rows, accepted, target)["records"]:
            frame = row["frame"]
            index = positions[frame]
            quality[frame] = row["quality_score"]
            score = (quality[frame], frame)
            winners[index] = min(winners.get(index, score), score)
    times = {row["frame"]: row["time_seconds"] for row in rows}
    selected_set = {frame for _, frame in winners.values()}
    if balance_reviewed:
        # Source timestamps are recorded in microseconds. Preserve exact spacing ties.
        timestamps = {frame: round(seconds * 1_000_000) for frame, seconds in times.items()}
        front = targets["front"]
        for start, stop in ((0, front), (front, target)):
            eligible = {frame for frame in times if start <= positions[frame] < stop}
            picked = eligible & selected_set
            remaining = eligible - picked
            for _ in range(min(stop - start - len(picked), len(remaining))):
                frame = min(remaining, key=lambda f: (
                    -min(abs(timestamps[f] - timestamps[g]) for g in picked), quality[f], f))
                picked.add(frame)
                remaining.remove(frame)
            selected_set.update(picked)
    selected = sorted(selected_set, key=lambda frame: (times[frame], frame))
    missing_bins = sorted(set(range(target)) - winners.keys())
    for frame, row in reasons.items():
        row.update(source_view_bin=positions[frame], selected=frame in selected_set)
        row["disposition"] = ("selected" if row["selected"] else
                              "screened_out" if row["screen_flags"] else
                              "eligible_not_selected" if frame in accepted else
                              "pose_" + row["review_status"])
    selection = dict(
        schema="shuolee/pose-training-selection/1", arc=arc, fit_dir=str(fit_dir),
        source_plan_sha256=sha(plan_path), source_fit_sha256=fit_sha, fit_review_sha256=sha(review_path),
        decisions_path=str(decisions_path.resolve()) if decisions_path is not None else None,
        decisions_sha256=sha(decisions_path) if decisions_path is not None else None,
        target_frames=target, candidate_count=len(ids), eligible_count=len(rows), selected_frames=selected,
        retained_holdouts=sorted(heldouts), missing_source_view_bins=missing_bins,
        coverage_basis="preregistered source view bins, no measured camera angle",
        full_roster_ready=len(selected) == target, reasons=reasons,
        dispositions=dict(Counter(row["disposition"] for row in reasons.values())),
    )
    if balance_reviewed:
        selection["balance_reviewed"] = True
    return selection


def load_admitted_selection(selection_path):
    frozen = read(selection_path)
    require(frozen.get("schema") == "shuolee/pose-training-selection/1", "Invalid pose selection.")
    decisions_path = frozen["decisions_path"]
    fit_dir = relocate(frozen["fit_dir"])
    current = select_fit(fit_dir, frozen["arc"],
                         relocate(decisions_path) if decisions_path else None,
                         balance_reviewed=frozen.get("balance_reviewed", False))
    # Paths are host-local spellings of the same hashed evidence in Windows/WSL.
    for key in ("fit_dir", "decisions_path"):
        current[key] = frozen[key]
    require(current == frozen, "Pose selection no longer matches the reviewed fit.")
    selected = frozen["selected_frames"]
    require(frozen["full_roster_ready"],
            f"Pose gate selected {len(selected)}/{frozen['target_frames']} views with "
            f"{len(frozen['missing_source_view_bins'])} uncovered bins. Gaussian launch blocked.")
    return fit_dir, frozen


def verify_prepared(selection_path, prepared, *, selection_loader=load_admitted_selection):
    fit_dir, frozen = selection_loader(selection_path)
    selected = frozen["selected_frames"]
    plan = read(prepared / "plan.json")
    require(plan["pose_selection_sha256"] == sha(selection_path), "Prepared run references a different pose selection.")
    roster = read(prepared / "selection.json")["selected_frames"]
    require(len(roster) == plan["training_frames"] == len(selected) and set(roster) == set(selected),
            "Training roster differs from reviewed frames.")
    inputs_path = prepared / "data/training-inputs.json"
    fit_path = prepared / "fit/fitted.npz"
    cache_path = prepared / "fit/surface-cache.npz"
    receipt = read(prepared / "smplify-gaussian-interface-receipt.json")
    require(receipt["frames"] == len(selected)
            and sha(inputs_path) == receipt["training_inputs_sha256"]
            and sha(fit_path) == receipt["fit_sha256"] and sha(cache_path) == receipt["surface_cache_sha256"],
            "Prepared RGB/mask, fit, or mesh cache changed after the handoff.")
    inputs = read(inputs_path)["records"]
    sources = {row["frame"]: row for row in read(fit_dir / "result.json")["frames"]}
    with np.load(fit_path, allow_pickle=False) as fit, np.load(cache_path, allow_pickle=False) as cache:
        ids = fit["frame_ids"].astype(str).tolist()
        require(ids == [row["frame"] for row in inputs] and len(ids) == len(selected) and set(ids) == set(selected),
                "Prepared RGB/mask rows and poses must contain exactly the admitted frames.")
        world = fit["vertices_world"]
        faces = fit["faces"]
        base = cache["base_vertices"]
        beta = fit["betas"]
        require(beta.shape == (1, 10) and np.isfinite(beta).all()
                and ("shared_betas" not in fit or np.array_equal(fit["shared_betas"], beta)),
                "Require one frozen SMPL-X shape for the entire training sequence.")
        require(np.array_equal(cache["frame_ids"], fit["frame_ids"])
                and np.array_equal(cache["faces"], faces)
                and world.ndim == 3 and world.shape[0] == len(ids) and world.shape[2] == 3
                and np.isfinite(world).all() and base.shape == world.shape
                and np.allclose(base, world, rtol=0, atol=MESH_REPLAY_TOLERANCE)
                and not np.any(fit["static_offset"]),
                "Initial Gaussian mesh cache does not match the prepared fitted mesh.")
        poses = {key: fit[key] for key in POSE_FIELDS}
        require(all(len(value) == len(ids) for value in poses.values()), "Incomplete per-frame pose archive.")
        for i, (frame, row) in enumerate(zip(ids, inputs, strict=True)):
            source = sources[frame]
            pose_path = relocate(source["output"])
            require(row["source_frame_index"] == source["source_index"] and sha(pose_path) == source["sha256"],
                    f"Source frame or pose changed for {frame}.")
            require(sha(relocate(row["image"])) == row["source_sha256"] == str(fit["image_sha256"][i])
                    and sha(relocate(row["mask"])) == row["mask_sha256"] == str(fit["mask_sha256"][i]),
                    f"Prepared image or mask changed for {frame}.")
            E = np.asarray(row["w2c"])
            require(E.shape == (4, 4) and np.isfinite(E).all() and np.array_equal(E, fit["w2c"][i]),
                    f"Prepared camera does not match the fitted mesh camera for {frame}.")
            R = E[:3, :3]
            require(np.allclose(E[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
                    and np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-6)
                    and abs(np.linalg.det(R) - 1.) <= 1e-6,
                    f"Prepared camera must be rigid to preserve the shared body scale for {frame}.")
            camera_vertices = world[i:i+1] @ R.T + E[:3, 3]
            with np.load(pose_path, allow_pickle=False) as original:
                require(all(np.array_equal(poses[key][i:i+1], original[key]) for key in POSE_FIELDS)
                        and np.array_equal(beta, original["betas"]),
                        f"Prepared pose or shared shape differs from the reviewed fit for {frame}.")
                reviewed_vertices = original["vertices_camera"]
                require(np.array_equal(faces, original["faces"]) and reviewed_vertices.shape == camera_vertices.shape
                        and np.all(reviewed_vertices[..., 2] > .05)
                        and np.allclose(camera_vertices, reviewed_vertices, rtol=0, atol=MESH_REPLAY_TOLERANCE),
                        f"Prepared mesh differs from the reviewed pose for {frame}.")
    return selected


def verify_training_launch(selection_path, prepared):
    require(selection_path.is_file(),
            f"Missing reviewed pose selection: {selection_path}. Run the pose selector first.")
    selected = verify_prepared(selection_path, prepared)
    review = read(prepared / "validity-review.json")
    require(read(prepared / "validity-cuda-preflight.json")["passed"] is True
            and review["actual_visual_review"] is True and review["training_admitted"] is True,
            "Require the existing Gaussian preflight and selected-set admission.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    screen = modes.add_parser("select")
    screen.add_argument("fit_dir", type=relocate)
    screen.add_argument("--arc", choices=tuple(TARGETS), required=True)
    screen.add_argument("--decisions", type=relocate)
    screen.add_argument("--balance-reviewed", action="store_true",
                        help="Fill the front/rear quotas from reviewed frames while reporting source gaps.")
    for mode in ("verify", "train"):
        launch = modes.add_parser(mode)
        launch.add_argument("selection", type=relocate)
        launch.add_argument("prepared", type=relocate)
    args = parser.parse_args()
    try:
        if args.mode == "select":
            result = select_fit(args.fit_dir, args.arc, args.decisions, balance_reviewed=args.balance_reviewed)
            filename = "pose-training-selection-balanced.json" if args.balance_reviewed else "pose-training-selection.json"
            destination = args.fit_dir / "review" / filename
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            try:
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"{destination} | {len(result['selected_frames'])}/{result['target_frames']} selected")
        elif args.mode == "verify":
            selected = verify_prepared(args.selection, args.prepared)
            print(f"POSE_GATE_READY {len(selected)} views", flush=True)
        else:
            if os.name == "nt":
                verify_training_launch(args.selection, args.prepared)
                raise ValueError("Start the Gaussian trainer from its WSL environment.")
            import cross_subject_sh0124 as gaussian

            require(args.prepared.resolve().parent == gaussian.ROOT.resolve(),
                    "Prepared run must be in the cross-subject training root.")
            gaussian.train(args.prepared.name, pose_selection=args.selection)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

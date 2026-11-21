"""Audit round-ten fit/selection with the original native-pixel screen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


LAB = Path("/opt/gs-slam-lab")
SOURCE = Path("/mnt/d/Projects/Shuolee-smpl-joint-fit-20261111/src")
OUTPUT = Path(__file__).resolve().parents[1] / "outputs/smpl-delivery-round10/input-audit"
BODY_NAMES = {
    0: "nose", 16: "left eye", 15: "right eye", 18: "left ear", 17: "right ear",
    5: "left shoulder", 2: "right shoulder", 6: "left elbow", 3: "right elbow",
    7: "left wrist", 4: "right wrist", 13: "left knee", 10: "right knee",
    14: "left ankle", 11: "right ankle",
}
FEET_NAMES = {19: "left big toe", 20: "left small toe", 21: "left heel",
              22: "right big toe", 23: "right small toe", 24: "right heel"}
BODY_FLAG = "original_supported_body_reprojection_over_existing3percent_height"
FEET_FLAG = "feet_reprojection_over_existing3percent_height"
SELECTED = {"selected_bin_leader", "selected_quota_fill"}
GATE_PASSED = SELECTED | {"technically_valid"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def projection_screen(observed, projected, row, expected):
    """Compute original quality gate and descriptive measurements for each pose."""
    h = float(row["initial_box_xyxy"][3] - row["initial_box_xyxy"][1])
    require(h > 0 and observed.shape == (118, 3) and projected.shape == (118, 2)
            and np.isfinite(observed).all() and np.isfinite(projected).all(),
            f"Invalid native projection observations in {row['frame']}")
    stats, violations = {}, []
    flags = []
    for name, joints, flag in (("body", BODY_NAMES, BODY_FLAG), ("feet", FEET_NAMES, FEET_FLAG)):
        indices = np.asarray([i for i in joints if observed[i, 2] > 0], dtype=int)
        confidence = observed[indices, 2]
        delta = projected[indices] - observed[indices, :2]
        length = np.linalg.norm(delta, axis=1)
        if name == "body":
            fails = len(length) and float(length.max()) / h > .03
        else:
            fails = np.any(length / h > .03)
        if fails:
            flags.append(flag)
        for j, px, weight in zip(indices, length, confidence):
            if float(px) / h > .03:
                violations.append(dict(region=name, joint=joints[j], joint_index=int(j),
                                       residual_px=float(px), residual_over_height=float(px)/h,
                                       fit_confidence=float(weight), observed_xy=observed[j, :2].tolist(),
                                       projected_xy=projected[j].tolist()))
        weighted_delta = (projected[indices].astype(np.float64)-observed[indices, :2])/h
        rms = (float(np.sqrt(np.average(np.sum(weighted_delta**2, axis=1),
                                        weights=confidence.astype(np.float64)**2))) if len(length) else None)
        prior = expected["scores"][name]
        require(prior["points"] == len(indices) and
                prior["rms_over_height"] == (round(rms, 8) if rms is not None else None),
                f"Selection RMS evidence differs in {row['frame']} {name}")
        stats[name] = dict(supported_points=len(indices), weighted_rms_over_height=rms,
                           max_residual_over_height=(float(length.max())/h if len(length) else None),
                           max_residual_px=(float(length.max()) if len(length) else None))
    require(expected["screen_flags"] == flags, f"Selection screen disagrees in {row['frame']}")
    require((expected["status"] in GATE_PASSED) == (not flags),
            f"Selection status disagrees in {row['frame']}")
    return dict(status=expected["status"], screen_flags=flags, statistics=stats, violations=violations)


def coverage_verified(selection):
    result = {}
    for side, report in selection["coverage"].items():
        rows = [r for r in selection["reasons"].values() if r.get("side") == side]
        passed = [r for r in rows if r["status"] in GATE_PASSED]
        picked = [r for r in rows if r["status"] in SELECTED]
        occupied = sorted({r["source_bin"] for r in picked})
        require(occupied == report["occupied_source_bins"] and
                occupied == sorted({r["source_bin"] for r in passed}) and
                len(passed) == report["eligible"] and len(picked) == report["selected"],
                f"Selection coverage is inconsistent in {side}")
        result[side] = report
    selected = {fid for fid, reason in selection["reasons"].items() if reason["status"] in SELECTED}
    require(selected == set(selection["selected_frames"]) and
            selection["reference_frame"] in selected, "Selected frames or reference identity differ")
    return result


def verify_training_fit(run, after, ids, candidates):
    """Verify the named Gaussians input meshes are the selected joint candidates."""
    fitted_path = run / "fit/fitted.npz"
    training_path = run / "data/training-inputs.json"
    receipt_path = run / "fit/official-fit-result.json"
    receipt = read_json(receipt_path)
    require(receipt["fit_sha256"] == sha(fitted_path) and
            receipt["training_inputs_sha256"] == sha(training_path), "Training-fit digests differ")
    expected_ids = [after["reference_frame"]] + [fid for fid in after["selected_frames"]
                                                  if fid != after["reference_frame"]]
    rows = read_json(training_path)["records"]
    require([row["frame"] for row in rows] == expected_ids, "Training rows differ from selected frame identities")
    positions = [ids.index(fid) for fid in expected_ids]
    with np.load(fitted_path, allow_pickle=False) as fitted:
        require(fitted["frame_ids"].tolist() == expected_ids and
                fitted["reference_frame_id"].item() == expected_ids[0], "Training pose identities differ")
        for key in ("vertices_camera", "joints_camera", "joints_projected_native", "K_integer_centers",
                    "source_frames", "image_sha256", "mask_sha256", "observed_native_keypoints"):
            require(np.array_equal(fitted[key], candidates[key][positions]),
                    f"Selected {key} differs from audited joint candidate")
        require(np.array_equal(fitted["betas"], candidates["betas"]), "Training fit beta differs")
    return {str(p): sha(p) for p in (fitted_path, training_path, receipt_path)}


def native_observation_drift(members, rows, old_rows, current_path, previous_path):
    """Identify input drift when comparing new and archived fits."""
    keys = ("keypoints", "scores", "boxes", "flip_disagreement")
    with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as old:
        current_ids = current["frame_ids"].astype(str).tolist()
        old_ids = old["frame_ids"].astype(str).tolist()
        a = [current_ids.index(fid) for fid in members]
        b = [old_ids.index(fid) for fid in members]
        detector = {}
        for key in keys:
            delta = np.abs(current[key][a].astype(np.float64) - old[key][b].astype(np.float64))
            detector[key] = dict(max_absolute_difference=float(delta.max()),
                                 differing_frames=int(np.any(delta.reshape(len(a), -1) != 0, axis=1).sum()))
    masks = []
    for fid in members:
        row, old = rows[fid], old_rows[fid]
        if row["mask_sha256"] == old["mask_sha256"]:
            masks.append(dict(frame=fid, exact_mask_sha256=True, binary_iou=1., differing_binary_pixels=0))
            continue
        with Image.open(row["mask"]) as current, Image.open(old["mask"]) as previous:
            current_fg = np.asarray(current.convert("L")) >= 128
            previous_fg = np.asarray(previous.convert("L")) >= 128
        require(current_fg.shape == previous_fg.shape, f"Mask dimensions differ in {fid}")
        union = np.count_nonzero(current_fg | previous_fg)
        require(union > 0, f"Source masks are empty in {fid}")
        masks.append(dict(frame=fid, exact_mask_sha256=False,
                          binary_iou=float(np.count_nonzero(current_fg & previous_fg)/union),
                          differing_binary_pixels=int(np.count_nonzero(current_fg != previous_fg))))
    return dict(detector=detector, mask_records=masks,
                mask_binary_iou_minimum=min(r["binary_iou"] for r in masks),
                mask_binary_iou_median=float(np.median([r["binary_iou"] for r in masks])))


def audit(subject, run, baseline_path, candidate_project):
    sys.path.insert(0, str(candidate_project / "src"))
    from shuolee_portrait.official_smplifyx import adapt_rtmw_observations
    from shuolee_portrait.stable_selection import BODY, FEET
    require(BODY.tolist() == list(BODY_NAMES) and FEET.tolist() == list(FEET_NAMES),
            "Audit joint index semantics changed")

    baseline = read_json(baseline_path)["subjects"][subject]
    root = run / "fit"
    before_path = root / "selection-before-joint.json"
    after_path = root / "selection.json"
    candidate_path = root / "joint-candidates.npz"
    refinement_path = root / "joint-body-refinement.json"
    before, after, joint = map(read_json, (before_path, after_path, refinement_path))
    require(joint["state"] == "complete", "Joint refinement did not finish")
    before_coverage, after_coverage = map(coverage_verified, (before, after))
    holds = [row["frame"] for row in baseline["retained_holdouts"]]
    require(before["retained_holdouts"] == after["retained_holdouts"] == holds == joint["retained_holdouts"],
            "Original holdout identities differ")
    require(not set(holds) & (set(before["selected_frames"]) | set(after["selected_frames"])),
            "A protected holdout entered selection")
    require(not any((root / "frames" / fid).exists() for fid in holds), "A protected holdout was independently fit")
    manifest_path = run / "data/inputs.json"
    manifest = read_json(manifest_path)
    require(manifest["video_sha256"] == baseline["source_video_sha256"], "Original input video SHA-256 differs")
    original = Path(baseline["original_run"])
    previous_manifest_path = original / "data/inputs.json"
    require(sha(previous_manifest_path) == baseline["manifest_sha256"], "Baseline native manifest changed")
    old_rows = {r["frame"]: r for r in read_json(previous_manifest_path)["records"]}
    rows = {row["frame"]: row for row in manifest["records"]}
    require(len(rows) == len(manifest["records"]) and set(rows) == set(old_rows) ==
            set(before["reasons"]) == set(after["reasons"]), "Original native frame roster differs")
    for fid, row in rows.items():
        fixed = ("source_frame_index", "wh", "K", "K_integer_centers", "w2c")
        require(all(row[k] == old_rows[fid][k] for k in fixed), f"Original frame/camera identity differs in {fid}")
        if row["source_sha256"] != old_rows[fid]["source_sha256"]:
            # Encoding may differ while the native pixels remain identical.
            with Image.open(row["image"]) as current, Image.open(old_rows[fid]["image"]) as previous:
                require(np.array_equal(np.asarray(current.convert("RGB")), np.asarray(previous.convert("RGB"))),
                        f"Native source pixels differ in {fid}")

    members = [fid for fid in rows if fid not in holds]
    require(all(before["reasons"][fid]["source_bin"] == after["reasons"][fid]["source_bin"] for fid in rows),
            "Original source-bin boundaries differ")
    observations_path = run / "data/observations.npz"
    drift = native_observation_drift(members, rows, old_rows, observations_path, original / "data/observations.npz")
    with np.load(observations_path, allow_pickle=False) as archive:
        detector_ids = archive["frame_ids"].astype(str).tolist()
        order = [detector_ids.index(fid) for fid in members]
        observed, _ = adapt_rtmw_observations(*(archive[name][order] for name in
            ("keypoints", "scores", "boxes", "flip_disagreement")))
    targets = dict(zip(members, observed))
    records = {}
    old_placement, fresh_placement = set(), set()
    independent_pose_drift = {}
    max_projection_delta_px = 0.
    with np.load(candidate_path, allow_pickle=False) as archive:
        ids = archive["frame_ids"].astype(str).tolist()
        require(len(ids) == len(members) and set(ids) == set(members) and
                set(ids) == set(joint["candidate_frames"]), "Joint candidate roster differs")
        arrays = {name: archive[name] for name in (
            "source_frames", "image_sha256", "mask_sha256", "K_integer_centers",
            "observed_native_keypoints", "joints_projected_native", "joints_camera", "vertices_camera", "betas",
        )}
    beta = np.asarray(joint["shared_betas"], dtype=arrays["betas"].dtype).reshape(10)
    after_betas = arrays["betas"]
    require(after_betas.shape[-1] == 10 and np.all(after_betas.reshape(-1, 10) == beta[None]),
            "Joint candidate betas are not one shared vector")

    for i, fid in enumerate(ids):
        row = rows[fid]
        require(row["source_frame_index"] == arrays["source_frames"][i] and
                row["source_sha256"] == arrays["image_sha256"][i] and
                row["mask_sha256"] == arrays["mask_sha256"][i],
                f"Joint candidate source image or mask differs in {fid}")
        observed = targets[fid]
        x, y = observed[:, :2].T
        w, h = row["wh"]
        observed[(x < -.5) | (x >= w-.5) | (y < -.5) | (y >= h-.5), 2] = 0
        require(np.array_equal(observed, arrays["observed_native_keypoints"][i].reshape(118, 3)),
                f"Joint candidate detector observations differ in {fid}")
        path = root / "frames" / fid / "winner-camera-space.npz"
        K = np.asarray(row["K_integer_centers"])
        with np.load(path, allow_pickle=False) as winner:
            require(np.array_equal(winner["K_integer_centers"], K) and
                    np.array_equal(arrays["K_integer_centers"][i], K) and
                    np.array_equal(winner["betas"].reshape(10), beta), f"Native K or shared beta differs in {fid}")
            old_projection = winner["joints_projected_native"].reshape(118, 2)
            before_vertices = winner["vertices_camera"]
        previous_frame = original / "fit/frames" / fid
        old_placement.add(read_json(previous_frame / "result.json").get("body_placement_parameters", 0))
        fresh_placement.add(read_json(path.parent / "result.json").get("body_placement_parameters", 0))
        with np.load(previous_frame / "winner-camera-space.npz", allow_pickle=False) as old:
            independent_pose_drift[fid] = dict(
                shared_beta_equal=bool(np.array_equal(old["betas"].reshape(10), beta)),
                max_absolute_projection_delta_px=float(np.abs(old["joints_projected_native"][0]-old_projection).max()),
                max_absolute_camera_vertex_delta_m=float(np.abs(old["vertices_camera"]-before_vertices).max()),
            )
        new_projection = arrays["joints_projected_native"][i].reshape(118, 2)
        camera_joints = arrays["joints_camera"][i].reshape(118, 3)
        projection_h = camera_joints.astype(np.float64) @ K.T
        require(np.all(projection_h[:, 2] > .05), f"Invalid native camera-space joint depth in {fid}")
        projected = projection_h[:, :2] / projection_h[:, 2:3]
        max_projection_delta_px = max(max_projection_delta_px, float(np.max(np.abs(new_projection-projected))))
        changes = joint["frames"][fid]
        records[fid] = dict(
            source_frame_index=row["source_frame_index"], source_sha256=row["source_sha256"],
            mask_sha256=row["mask_sha256"], source_bin=before["reasons"][fid]["source_bin"],
            before=projection_screen(observed, old_projection, row, before["reasons"][fid]),
            after=projection_screen(observed, new_projection, row, after["reasons"][fid]),
            joint_fit_metrics=changes,
        )
    require(max_projection_delta_px < .002, "Joint candidate native projection differs from camera-space geometry")
    require(all(before["reasons"][fid]["status"] == after["reasons"][fid]["status"] == "protected_holdout" for fid in holds),
            "A holdout changed role")
    bins_missing_before = set(before_coverage["front"]["missing_source_bins"])
    bins_missing_after = set(after_coverage["front"]["missing_source_bins"])
    before_passed = {fid for fid, row in records.items() if row["before"]["status"] in GATE_PASSED}
    after_passed = {fid for fid, row in records.items() if row["after"]["status"] in GATE_PASSED}
    skipped_bins = {bin_id: [
        dict(frame=fid, flags=row["after"]["screen_flags"], violating_joints=row["after"]["violations"])
        for fid, row in records.items() if row["source_bin"] == bin_id]
        for bin_id in sorted(bins_missing_after)}
    violations_in_missing_bins = Counter(item["joint"] for group in skipped_bins.values()
                                       for candidate in group for item in candidate["violating_joints"])
    training_fit = verify_training_fit(run, after, ids, arrays)

    return dict(
        state="verified", subject=subject, run=str(run), candidate_frames=len(ids),
        input_files_sha256={str(path): sha(path) for path in
            (baseline_path, before_path, after_path, candidate_path, refinement_path, manifest_path, observations_path)},
        training_fit_sha256=training_fit,
        coverage_basis=after["coverage_basis"],
        gate=after["pose_screen"], original_camera_projection_max_absolute_difference_px=max_projection_delta_px,
        native_fit_input_drift=drift,
        independent_initializer_drift=dict(archived_body_placement_parameter_counts=sorted(old_placement),
                                           fresh_body_placement_parameter_counts=sorted(fresh_placement),
                                           frames=independent_pose_drift),
        round9_front_coverage=baseline["front_coverage"], before=before_coverage, after=after_coverage,
        before_selected=before["selected_frames"], selected=after["selected_frames"],
        reference_frame=after["reference_frame"], retained_holdouts=baseline["retained_holdouts"],
        recovered_bins=sorted(bins_missing_before - bins_missing_after),
        lost_bins=sorted(bins_missing_after - bins_missing_before),
        newly_passing_frames=sorted(after_passed - before_passed),
        newly_failing_frames=sorted(before_passed - after_passed),
        after_missing_bins=skipped_bins, after_missing_bin_joint_violations=dict(violations_in_missing_bins),
        observations_limit="Positive relative RTMW confidence and foreground-mask support do not establish physical visibility.",
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=("016", "006"), required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--baseline", type=Path, default=OUTPUT / "baseline-selection.json")
    parser.add_argument("--project", type=Path, default=SOURCE.parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run or LAB / f"shuolee-smpl-delivery-round10-{args.subject}" / "run"
    result = audit(args.subject, run, args.baseline, args.project)
    target = args.output or OUTPUT / f"{args.subject}-selection-comparison.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(target), sha256=sha(target), recovered_bins=result["recovered_bins"],
                          lost_bins=result["lost_bins"], before=result["before"]["front"]["eligible"],
                          after=result["after"]["front"]["eligible"])))


if __name__ == "__main__":
    main()


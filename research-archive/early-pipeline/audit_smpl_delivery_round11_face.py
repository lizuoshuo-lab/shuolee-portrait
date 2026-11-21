"""Independent native-pixel and fixed-input audit for the round-eleven fit."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_smpl_delivery_round10_selection import (
    GATE_PASSED, coverage_verified, projection_screen, read_json, require, sha, verify_training_fit,
)


LAB = Path("/opt/gs-slam-lab")
WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT = WORKSPACE / "outputs/smpl-delivery-round11/input-audit"
R10_RECEIPT = WORKSPACE / "outputs/smpl-delivery-round10/receipt.json"
R10_RECEIPT_SHA = "359da8f7873f9c6f6a2b0ea3a0f022178da027cfc5693d55b7af55bff0b5c301"
R10_COMMIT = "ed7755c6a293484c2c83fc642fde43981d88b611"
R11_COMMIT = "509657c17692dd3eca50cf3ad139d8b660f04ce1"
BASELINE_SHA = "736bb53d3ba3a00459ecadd3b229da57cdb143dc6cf7780be42e4f257988011c"
SUBJECTS = ("016", "006")
FACE_CONFIDENCE = .5
FACE_TOLERANCE_PX = .25
SCORE_NUMERICAL_TOLERANCE_PX = 5e-5
FACE_REGIONS = {
    "face": list(range(67, 118)),
    "right_eye": list(range(86, 92)),
    "left_eye": list(range(92, 98)),
    "nose": list(range(77, 86)),
    "mouth": list(range(98, 118)),
    "right_ear": [17],
    "left_ear": [18],
}


def pin(path, digest):
    require(sha(path) == digest, f"Unexpected SHA-256 for {path}")
    return path


def read_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def projection_parity(arrays, name):
    xyz = np.asarray(arrays["joints_camera"], dtype=np.float64)
    K = np.asarray(arrays["K_integer_centers"], dtype=np.float64)
    stored = np.asarray(arrays["joints_projected_native"], dtype=np.float64)
    h = xyz @ np.swapaxes(K, -2, -1)
    require(np.isfinite(h).all() and (h[..., 2] > .05).all(), f"Invalid projection depth for {name}")
    require(stored.shape == h[..., :2].shape and np.isfinite(stored).all(), f"Invalid stored pixels for {name}")
    return float(np.max(np.abs(stored - h[..., :2] / h[..., 2:])))


def face_scores(observed, projected, wh, *, dense_positive_only=False):
    xy = observed[:, :2]
    confidence = observed[:, 2]
    width, height = wh
    if dense_positive_only:
        regions = {"face": FACE_REGIONS["face"]}
        supported = confidence > 0
    else:
        regions = FACE_REGIONS
        supported = ((confidence >= FACE_CONFIDENCE) & np.isfinite(observed).all(axis=1)
                     & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                     & (xy[:, 1] >= 0) & (xy[:, 1] < height))
    residual = np.linalg.norm(projected.astype(np.float64) - xy, axis=1)
    scores = {}
    for region, indices in regions.items():
        selected = np.asarray([i for i in indices if supported[i]], dtype=int)
        values = residual[selected]
        weights = confidence[selected].astype(np.float64) ** 2
        scores[region] = dict(points=len(selected),
            rms_px=float(np.sqrt(np.average(values ** 2, weights=weights))) if len(selected) else None,
            max_px=float(values.max()) if len(selected) else None)
    return scores


def face_comparison(observed, before, after, wh):
    initial = face_scores(observed, before, wh)
    current = face_scores(observed, after, wh)
    regions, failed = {}, []
    for name, a in initial.items():
        b = current[name]
        require(a["points"] == b["points"], "Face support changed between paired fits")
        rms = b["rms_px"] - a["rms_px"] if a["points"] else None
        maximum = b["max_px"] - a["max_px"] if a["points"] else None
        for metric, delta in (("rms_px", rms), ("max_px", maximum)):
            if delta is not None and delta > FACE_TOLERANCE_PX + SCORE_NUMERICAL_TOLERANCE_PX:
                failed.append(dict(region=name, metric=metric, delta_px=delta))
        regions[name] = dict(before=a, after=b, rms_delta_px=rms, max_delta_px=maximum)
    baseline_dense = face_scores(observed, before, wh, dense_positive_only=True)["face"]
    current_dense = face_scores(observed, after, wh, dense_positive_only=True)["face"]
    return dict(regions=regions, failures=failed,
                dense_positive_before=baseline_dense, dense_positive_after=current_dense)


def summarize_faces(frames):
    summary = {}
    for region in FACE_REGIONS:
        records = [row["regions"][region] for row in frames.values() if row["regions"][region]["before"]["points"]]
        rms = [r["rms_delta_px"] for r in records]
        maximum = [r["max_delta_px"] for r in records]
        summary[region] = dict(supported_frames=len(records),
            unsupported_frames=len(frames)-len(records),
            rms_worse_frames=sum(delta > SCORE_NUMERICAL_TOLERANCE_PX for delta in rms),
            max_worse_frames=sum(delta > SCORE_NUMERICAL_TOLERANCE_PX for delta in maximum),
            median_rms_before_px=float(np.median([r["before"]["rms_px"] for r in records])) if records else None,
            median_rms_after_px=float(np.median([r["after"]["rms_px"] for r in records])) if records else None,
            worst_rms_delta_px=max(rms, default=None), worst_max_delta_px=max(maximum, default=None))
    dense = [(r["dense_positive_before"]["rms_px"], r["dense_positive_after"]["rms_px"]) for r in frames.values()
             if r["dense_positive_before"]["points"]]
    return dict(regions=summary, failed_frames=[name for name, row in frames.items() if row["failures"]],
        dense_positive=dict(frames=len(dense), worse_frames=sum(b-a > SCORE_NUMERICAL_TOLERANCE_PX for a, b in dense),
                            median_before_px=float(np.median([a for a, _ in dense])) if dense else None,
                            median_after_px=float(np.median([b for _, b in dense])) if dense else None))


def input_rows(run):
    manifest = read_json(run / "data/inputs.json")
    rows = {r["frame"]: r for r in manifest["records"]}
    require(len(rows) == len(manifest["records"]) == 56, "The full native input roster differs")
    return manifest, rows


def verify_native_bytes(rows):
    files = {}
    for fid, row in rows.items():
        for path, field in (("image", "source_sha256"), ("mask", "mask_sha256")):
            actual = Path(row[path])
            pin(actual, row[field])
            files.setdefault(fid, {})[path] = {"path": str(actual), "sha256": row[field]}
    return files


def fixed_fit_names(candidates, manifest, holdouts):
    names = candidates["frame_ids"].astype(str).tolist()
    inputs = {r["frame"] for r in manifest["records"]}
    require(len(names) == len(set(names)) == 52 and len(holdouts) == len(set(holdouts)) == 4
            and set(names).isdisjoint(holdouts) and set(names) | set(holdouts) == inputs,
            "The protected fit/holdout roster differs")
    return names


def input_detector_support(run, candidates, names):
    observed = candidates["observed_native_keypoints"]
    with np.load(run / "data/observations.npz", allow_pickle=False) as inputs:
        ids = inputs["frame_ids"].astype(str).tolist()
        indices = [ids.index(fid) for fid in names]
        flip = inputs["flip_disagreement"][indices, 40:91]
        original_xy = inputs["keypoints"][indices, 40:91].astype(observed.dtype)
        require(np.array_equal(observed[:, 67:, :2], original_xy),
                "Dense face detector map or source coordinates changed")
    dense = observed[:, 67:]
    reliable = dense[..., 2] >= FACE_CONFIDENCE
    spans = [float(np.linalg.norm(np.ptp(row[keep, :2], axis=0))) for row, keep in zip(dense, reliable)
             if np.count_nonzero(keep) >= 2]
    return dict(dense_points_per_frame_minimum=int(reliable.sum(axis=1).min()),
                dense_points_per_frame_maximum=int(reliable.sum(axis=1).max()),
                face_support_diagonal_px_p00_p50_p100=np.quantile(spans, [0., .5, 1.]).tolist(),
                native_flip_disagreement_px_p50_p90_p95=np.quantile(flip[reliable], [.5, .9, .95]).tolist())


def frame_comparisons(names, rows, baseline_run, candidates, selection=None, *, measure_pose_change=False):
    records = {}
    peak = projection_parity(candidates, str(baseline_run))
    for slot, fid in enumerate(names):
        row = rows[fid]
        observed = candidates["observed_native_keypoints"][slot]
        before = read_npz(baseline_run / "fit/frames" / fid / "winner-camera-space.npz")
        K = np.asarray(row["K_integer_centers"])
        require(np.array_equal(before["K_integer_centers"], K)
                and np.array_equal(candidates["K_integer_centers"][slot], K)
                and np.array_equal(before["betas"], candidates["betas"]),
                f"Camera or shared shape differs in {fid}")
        require(int(candidates["source_frames"][slot]) == row["source_frame_index"]
                and candidates["image_sha256"][slot] == row["source_sha256"]
                and candidates["mask_sha256"][slot] == row["mask_sha256"],
                f"Paired source identity differs in {fid}")
        peak = max(peak, projection_parity(before, fid))
        start = before["joints_projected_native"][0]
        finish = candidates["joints_projected_native"][slot]
        comparison = face_comparison(observed, start, finish, row["wh"])
        comparison["source_frame_index"] = row["source_frame_index"]
        if measure_pose_change:
            comparison["pose_parameters_changed"] = any(not np.array_equal(before[field][0], candidates[field][slot])
                for field in ("global_orient", "camera_translation", "body_pose_axis_angle", "pose_embedding"))
            comparison["camera_vertices_changed"] = not np.array_equal(
                before["vertices_camera"][0], candidates["vertices_camera"][slot])
        if selection is not None:
            comparison["original_pose_gate"] = projection_screen(observed, finish, row, selection["reasons"][fid])
        records[fid] = comparison
    require(peak < .002, "Saved native projection differs from independently projected camera joints")
    return records, peak


def capture_baseline():
    receipt = read_json(pin(R10_RECEIPT, R10_RECEIPT_SHA))
    require(receipt["commit"] == R10_COMMIT and set(receipt["subjects"]) == set(SUBJECTS),
            "Unexpected delivered round-ten baseline")
    subjects = {}
    for name in SUBJECTS:
        published = receipt["subjects"][name]
        run = Path(published["result"]).parent
        pin(run / "result.json", published["result_sha256"])
        artifacts = {
            "joint": "fit/joint-body-refinement.json",
            "joint_candidates": "fit/joint-candidates.npz",
            "selection": "fit/selection.json",
            "pre_joint_selection": "fit/selection-before-joint.json",
            "official_fit": "fit/official-fit-result.json",
        }
        pinned = {}
        for key, rel in artifacts.items():
            proof = published["evidence"][key]
            pin(Path(proof["path"]), proof["sha256"])
            pinned[rel] = proof["sha256"]
            pin(run / rel, pinned[rel])
        official = read_json(run / artifacts["official_fit"])
        for rel in ("data/inputs.json", "data/observations.npz"):
            pinned[rel] = official["source"]["input_files"][str(run / rel)]
            pin(run / rel, pinned[rel])
        pinned["fit/shared-body.json"] = official["shared_body_sha256"]
        pin(run / "fit/shared-body.json", pinned["fit/shared-body.json"])
        manifest, rows = input_rows(run)
        native = verify_native_bytes(rows)
        selected = read_json(run / artifacts["selection"])
        candidates = read_npz(run / artifacts["joint_candidates"])
        holds = selected["retained_holdouts"]
        names = fixed_fit_names(candidates, manifest, holds)
        for fid in names:
            base = Path("fit/frames") / fid
            fit_receipt = read_json(run / base / "result.json")
            require(fit_receipt["frame_id"] == fid and fit_receipt["state"] == "complete"
                    and fit_receipt["source_frame_index"] == rows[fid]["source_frame_index"]
                    and fit_receipt["source_image_sha256"] == rows[fid]["source_sha256"],
                    f"Official initializer identity differs in {fid}")
            for rel in (base / "result.json", base / "winner-camera-space.npz"):
                pinned[str(rel)] = sha(run / rel)
        for receipt_entry in official["official_frame_results"]:
            pin(Path(receipt_entry["path"]), receipt_entry["sha256"])
        require(not any((run / "fit/frames" / fid).exists() for fid in holds), "Holdout has an official fit")
        faces, parity = frame_comparisons(names, rows, run, candidates, selected)
        fitted_training = verify_training_fit(run, selected, names, candidates)
        face_report = read_json(run / artifacts["joint"])
        for fid, item in faces.items():
            for phase, metric in (("before", "dense_positive_before"), ("after", "dense_positive_after")):
                recorded = face_report["frames"][fid][phase]["projection"]["face"]["rms_px"]
                require(abs(item[metric]["rms_px"] - recorded) < SCORE_NUMERICAL_TOLERANCE_PX,
                        f"Round-ten dense facial RMS did not reproduce in {fid}")
        subjects[name] = dict(run=str(run), file_sha256=pinned, native_files=native, candidate_frames=names,
            retained_holdouts=holds, reference_frame=selected["reference_frame"],
            selected_frames=selected["selected_frames"], coverage=coverage_verified(selected),
            original_camera_projection_max_abs_px=parity, training_fit_sha256=fitted_training,
            detector_support=input_detector_support(run, candidates, names), face_summary=summarize_faces(faces),
            frames=faces)
    return dict(schema="shuolee-round11-locked-round10-inputs-v1", state="verified",
                round10_receipt=str(R10_RECEIPT), round10_receipt_sha256=R10_RECEIPT_SHA,
                round10_commit=R10_COMMIT, subjects=subjects,
                measurement_boundary="Detector confidence and flip agreement are observation heuristics. "
                "These reprojection comparisons are not physical visibility or pixel ground truth.")


def verify_reuse(subject, run, baseline, fit):
    baseline_run = Path(baseline["run"])
    inventory_path = run.parent / "protected-inputs.json"
    plan = read_json(run.parent / "preregistration.json")
    inventory = read_json(pin(inventory_path, plan["protected_input_inventory_sha256"]))
    require(inventory["schema"] == "shuolee-round11-fixed-inputs-v1"
            and inventory["subject"] == plan["subject"] == subject
            and inventory["baseline_commit"] == plan["baseline_commit"] == R10_COMMIT
            and inventory["baseline_run"] == str(baseline_run)
            and inventory["fit_frames"] == baseline["candidate_frames"]
            and inventory["retained_holdouts"] == plan["retained_holdouts"] == baseline["retained_holdouts"]
            and set(inventory["winners"]) == set(baseline["candidate_frames"]),
            "Preregistered R10 input inventory differs from the independent baseline")
    require(fit["execution_kind"] == "reused_official_initializers_joint_refinement"
            and fit["official_initializers_reused"] is True and fit["new_fitting_started"] is False,
            "The official fit did not reuse the locked initializers")
    prior_files = baseline["file_sha256"]
    bound = fit["source"]["input_files"]
    for rel in ("data/inputs.json", "data/observations.npz", "fit/shared-body.json"):
        pin(run / rel, prior_files[rel])
        actual = baseline_run / rel if rel == "fit/shared-body.json" else run / rel
        require(bound[str(actual.resolve())] == prior_files[rel], f"Fit did not bind the reused {rel}")
    for fid in baseline["candidate_frames"]:
        paths = inventory["winners"][fid]
        for key, filename in (("npz", "winner-camera-space.npz"), ("result", "result.json")):
            rel = f"fit/frames/{fid}/{filename}"
            path = str(baseline_run / rel)
            digest = prior_files[rel]
            require(paths[key] == path and paths[key + "_sha256"] == digest
                    and bound[path] == digest and inventory["files"][path] == digest,
                    f"Fit did not bind the exact official initializer {fid} {filename}")
            pin(Path(path), digest)
    for fid in baseline["retained_holdouts"]:
        require(not (run / "fit/frames" / fid).exists(), "A held-out frame was newly fit")
        require(not any(f"/fit/frames/{fid}/" in path for path in bound), "Holdout fit source in receipt")
    manifest, rows = input_rows(run)
    native_files = verify_native_bytes(rows)
    require(native_files == baseline["native_files"], "A native image or mask differs from the locked source")
    for pair in native_files.values():
        for item in pair.values():
            require(inventory["files"][item["path"]] == item["sha256"], "Native source absent from inventory")
    for rel in ("data/inputs.json", "data/observations.npz", "fit/shared-body.json",
                "fit/selection-before-joint.json"):
        pin(baseline_run / rel, prior_files[rel])
    return manifest, rows, inventory_path


def verify_final_geometry(run, selection, names, candidates, rows):
    hashes = verify_training_fit(run, selection, names, candidates)
    fitted_path = run / "fit/fitted.npz"
    training = read_json(run / "data/training-inputs.json")["records"]
    with np.load(fitted_path, allow_pickle=False) as fitted:
        ids = fitted["frame_ids"].astype(str).tolist()
        slots = [names.index(fid) for fid in ids]
        require(float(fitted["world_scale"]) == 1. and np.array_equal(fitted["faces"], candidates["faces"]),
                "Training altered world scale or candidate topology")
        for field in ("body_pose_axis_angle", "global_orient", "camera_translation", "pose_embedding",
                      "expression", "jaw_pose", "left_hand_pose", "right_hand_pose", "leye_pose", "reye_pose"):
            require(np.array_equal(fitted[field], candidates[field][slots]),
                    f"Selected training geometry parameter differs: {field}")
        for slot, source in enumerate(training):
            original = rows[ids[slot]]
            for key in ("image", "mask", "source_sha256", "mask_sha256", "K", "K_integer_centers", "wh", "w2c"):
                require(source[key] == original[key], f"Training source {key} differs in {ids[slot]}")
            require(np.array_equal(fitted["w2c"][slot], np.asarray(original["w2c"])),
                    f"Saved training camera differs in {ids[slot]}")
    return hashes


def verify_guard_report(joint, records, candidates):
    guard = joint["face_guard"]
    require(guard["reference"] == "official_initializer" and guard["regions"] == FACE_REGIONS
            and guard["threshold_px"] == FACE_TOLERANCE_PX and guard["confidence_threshold"] == FACE_CONFIDENCE
            and guard["exact_reprojection_passed"] is True, "Stored face protection definition differs")
    supported = 0
    for fid, frame in records.items():
        actual = joint["frames"][fid]["face_guard"]
        require(set(actual) == set(FACE_REGIONS), f"Face report omits a region in {fid}")
        supported += any(row["before"]["points"] for row in frame["regions"].values())
        for name, row in frame["regions"].items():
            got = actual[name]
            count = row["before"]["points"]
            require(got["supported_points"] == count and got["status"] == ("pass" if count else "unobserved"),
                    f"Face report support or outcome differs in {fid}, {name}")
            if count:
                require(not got["failures"], f"Delivered face report failure in {fid}, {name}")
                for expected, reported in (("rms_px", "rms_px"), ("max_px", "peak_px")):
                    for phase in ("before", "after"):
                        require(abs(got[phase][reported]-row[phase][expected]) < SCORE_NUMERICAL_TOLERANCE_PX,
                                f"Independent face score differs in {fid}, {name}, {phase}, {reported}")
    require(guard["supported_frame_count"] == supported, "Face support frame count differs")
    config = joint["config"]
    trace = joint["objective_trace"]
    backtrack = joint["face_guard_backtracking_trace"]
    chosen = joint["chosen_checkpoint"]
    require([item["step"] for item in trace] == config["checkpoints"]
            and all(item["fraction"] == 1. for item in trace)
            and [item["fraction"] for item in backtrack] == config["face_guard_backtracking"][:len(backtrack)],
            "Reported optimizer checkpoint schedule changed")
    for item in trace + backtrack:
        require(item["face_guard_passed"] == (not item["face_guard_failures"]), "Face checkpoint reasons omitted")
        for fail in item["face_guard_failures"]:
            require(fail["frame"] in records and fail["region"] in FACE_REGIONS
                    and fail["metric"] in ("rms_px", "peak_px")
                    and fail["after_px"] - fail["before_px"] > FACE_TOLERANCE_PX,
                    "Checkpoint failure does not satisfy the declared guard")
    admitted = lambda item: (item["face_guard_passed"] and item["minimum_depth_m"] > .05
                             and item["objective"] <= trace[0]["objective"])
    rank = lambda item: (-item["pose_gate_roster_capacity"], -item["source_bins_with_pose_gate_eligible"],
                         item["objective"])
    feasible = [item for item in trace[1:] if admitted(item)]
    if guard["fallback"] is None:
        expected = min(feasible, key=rank) if feasible else next((item for item in backtrack if admitted(item)), None)
        require(chosen == expected and joint["objective_final_step"] == chosen["step"]
                and joint["objective_final"] == chosen["objective"] and chosen["step"] > 0,
                "Retained joint checkpoint does not meet its published guard/ranking")
    else:
        require(guard["fallback"] in ("no_feasible_joint_checkpoint", "exact_reprojection_guard")
                and chosen == trace[0] and joint["objective_final_step"] == 0
                and not any(row["pose_parameters_changed"] or row["camera_vertices_changed"] for row in records.values()),
                "Fallback must retain the exact official initializers")
        if guard["fallback"] == "no_feasible_joint_checkpoint":
            require(not feasible and not any(map(admitted, backtrack)), "Fallback discarded an admissible checkpoint")
    shared = np.asarray(joint["shared_latent_final"], dtype=np.float64).reshape(1, 32)
    residual = candidates["pose_embedding"].astype(np.float64) - shared
    mean = float(np.abs(residual.mean(0)).max())
    maximum = float(np.linalg.norm(residual, axis=-1).max())
    require(mean < 1e-5 and abs(maximum-joint["pose_residual_l2_max"]) < 1e-5
            and (guard["fallback"] or maximum <= config["latent_residual_l2_limit"] + 1e-5),
            "Candidate pose latents do not match the reported shared basis and residual bound")
    return dict(reported=guard, chosen_checkpoint=chosen, optimizer_checkpoint_trace=trace,
        optimizer_backtracking_trace=backtrack,
        discarded_raw_nonzero_checkpoints_with_face_failures=sum(not item["face_guard_passed"] for item in trace[1:]),
        changed_pose_parameter_frames=sum(item["pose_parameters_changed"] for item in records.values()),
        changed_camera_vertex_frames=sum(item["camera_vertices_changed"] for item in records.values()),
        independently_measured_residual_l2_max=maximum,
        independently_measured_residual_mean_absolute_max=mean)


def verify_round11(subject, run, baseline_path):
    contract = read_json(pin(baseline_path, BASELINE_SHA))
    require(contract["round10_receipt_sha256"] == R10_RECEIPT_SHA and contract["round10_commit"] == R10_COMMIT,
            "Unexpected baseline contract")
    baseline = contract["subjects"][subject]
    baseline_run = Path(baseline["run"])
    fit = read_json(run / "fit/official-fit-result.json")
    manifest, rows, inventory = verify_reuse(subject, run, baseline, fit)
    proof = fit["joint_body_refinement"]
    joint_path = pin(run / proof["file"], proof["sha256"])
    candidate_path = pin(run / proof["candidate_file"], proof["candidate_sha256"])
    before_path = pin(run / proof["before_selection_file"], baseline["file_sha256"]["fit/selection-before-joint.json"])
    require(sha(before_path) == proof["before_selection_sha256"], "Pre-refinement gate proof does not bind the run")
    selection = read_json(run / "fit/selection.json")
    require(selection == fit["selection"] and fit["fitted_candidate_frames"] == 52,
            "Joint selection does not bind the saved fit")
    before, joint = map(read_json, (before_path, joint_path))
    holdouts = baseline["retained_holdouts"]
    require(before["retained_holdouts"] == selection["retained_holdouts"] == joint["retained_holdouts"]
            == joint["excluded_holdouts"] == holdouts
            and not set(holdouts) & set(selection["selected_frames"])
            and 0 < len(selection["selected_frames"]) == len(set(selection["selected_frames"])) <= 24,
            "Held-out or duplicate frames entered R11 selection")
    require(joint["state"] == "complete" and joint["optimizer_updates"] == 80
            and joint["native_K_frozen"] and joint["shared_betas_frozen"]
            and joint["per_frame_scale_parameters"] == 0, "Joint stage configuration differs")
    require(joint["config"]["face_guard_confidence"] == FACE_CONFIDENCE
            and joint["config"]["face_guard_tolerance_px"] == FACE_TOLERANCE_PX,
            "Unexpected predeclared face guard threshold")
    before_coverage, after_coverage = map(coverage_verified, (before, selection))
    candidates = read_npz(candidate_path)
    names = fixed_fit_names(candidates, manifest, holdouts)
    require(names == joint["fit_frame_ids"] == baseline["candidate_frames"]
            and set(names) == set(joint["frames"]), "A candidate or observation was hidden from verification")
    prior_candidate_rel = "fit/joint-candidates.npz"
    old = read_npz(pin(baseline_run / prior_candidate_rel, baseline["file_sha256"][prior_candidate_rel]))
    for field in ("frame_ids", "faces", "betas", "K_integer_centers", "observed_native_keypoints",
                  "source_frames", "image_sha256", "mask_sha256"):
        require(np.array_equal(candidates[field], old[field]), f"Fixed R10 candidate input changed: {field}")
    records, parity = frame_comparisons(names, rows, baseline_run, candidates, selection, measure_pose_change=True)
    summary = summarize_faces(records)
    require(not summary["failed_frames"], f"Final fit has protected face regressions: {summary['failed_frames']}")
    guard_proof = verify_guard_report(joint, records, candidates)
    cross = {}
    for fid, row in records.items():
        prior = baseline["frames"][fid]
        old_dense = prior["dense_positive_after"]
        row["r10_to_r11_dense_rms_delta_px"] = row["dense_positive_after"]["rms_px"] - old_dense["rms_px"]
        for region, pair in row["regions"].items():
            before_region = prior["regions"][region]["before"]
            require(pair["before"] == before_region, f"Independent initializer changed in {fid}, {region}")
            pair["r10_after"] = prior["regions"][region]["after"]
            if before_region["points"]:
                pair["r10_to_r11_rms_delta_px"] = pair["after"]["rms_px"] - pair["r10_after"]["rms_px"]
        require(before["reasons"][fid]["source_bin"] == selection["reasons"][fid]["source_bin"],
                f"Native time bin changed in {fid}")
    for region in FACE_REGIONS:
        data = [row["regions"][region]["r10_to_r11_rms_delta_px"] for row in records.values()
                if row["regions"][region]["before"]["points"]]
        cross[region] = dict(better_frames=sum(v < -SCORE_NUMERICAL_TOLERANCE_PX for v in data),
            worse_frames=sum(v > SCORE_NUMERICAL_TOLERANCE_PX for v in data),
            median_delta_px=float(np.median(data)) if data else None)
    fit_hashes = verify_final_geometry(run, selection, names, candidates, rows)
    result_descriptors = fit["official_frame_results"]
    require([r["frame_id"] for r in result_descriptors]
            == [selection["reference_frame"]] + [name for name in selection["selected_frames"]
                                               if name != selection["reference_frame"]],
            "The selected training roster does not match the reused official fit descriptors")
    for item in result_descriptors:
        rel = f"fit/frames/{item['frame_id']}/result.json"
        require(item["path"] == str(baseline_run / rel) and item["sha256"] == baseline["file_sha256"][rel],
                f"Wrong reused official descriptor for {item['frame_id']}")
    now = set(after_coverage["front"]["occupied_source_bins"])
    initial = set(before_coverage["front"]["occupied_source_bins"])
    prior = set(baseline["coverage"]["front"]["occupied_source_bins"])
    return dict(state="verified", subject=subject, run=str(run), candidate_frames=len(names),
        source_reuse_verified=True, input_hashes={str(p): sha(p) for p in
            (baseline_path, inventory, joint_path, candidate_path, before_path, run / "fit/selection.json")},
        training_fit_sha256=fit_hashes, original_camera_projection_max_absolute_difference_px=parity,
        retained_holdouts=holdouts, selected_frames=selection["selected_frames"], reference_frame=selection["reference_frame"],
        coverage_basis=selection["coverage_basis"], gate=selection["pose_screen"],
        initial_coverage=before_coverage, r10_coverage=baseline["coverage"], r11_coverage=after_coverage,
        recovered_bins_vs_initializer=sorted(now - initial), lost_bins_vs_initializer=sorted(initial - now),
        recovered_bins_vs_r10=sorted(now - prior), lost_bins_vs_r10=sorted(prior - now),
        missing_bin_rejections={str(i): [dict(frame=fid, **records[fid]["original_pose_gate"])
            for fid in names if selection["reasons"][fid]["source_bin"] == i]
            for i in after_coverage["front"]["missing_source_bins"]},
        face_guard=dict(confidence=FACE_CONFIDENCE, tolerance_px=FACE_TOLERANCE_PX,
            comparison_numerical_tolerance_px=SCORE_NUMERICAL_TOLERANCE_PX, joint_indices=FACE_REGIONS,
            objective_final_step=joint["objective_final_step"], checkpoint_proof=guard_proof,
            final_independent_measurements=summary, r10_to_r11=cross),
        frames=records, measurement_boundary=contract["measurement_boundary"])


def pairing_endpoint(model, terminal_path):
    run = terminal_path.parent
    require(Path(model["run"]) == run, "Pairing references a different training run")
    terminal = read_json(pin(terminal_path, model["terminal_sha256"]))
    replay_path = run / terminal["bound_replay"]
    replay = read_json(pin(replay_path, terminal["bound_replay_sha256"]))
    snapshot = replay_path.parents[1]
    binding_path = snapshot / "bound-replay.json"
    binding = read_json(pin(binding_path, replay["bound_replay_sha256"]))
    require(terminal["state"] == "completed" and terminal["format_passed"] is True and terminal["steps"] == 10000
            and terminal["reference_frame_id"] == model["reference_frame_id"] == binding["reference_frame_id"],
            f"Wrong terminal or reference frame: {run}")
    rows_sha = binding["inputs_sha256"]["training_rows"]
    require(model["training_inputs_sha256"] == rows_sha, f"Training roster hash differs: {run}")
    rows = read_json(pin(run / "data/training-inputs.json", rows_sha))["records"]
    artifacts = dict(
        reference_mesh=pin(snapshot / "reference-mesh.npz", binding["inputs_sha256"]["reference_mesh"]),
        bound_ply=pin(snapshot / "bound-local.ply", binding["bound_ply_sha256"]),
        static_ply=pin(run / terminal["ply"], terminal["sha256"]),
    )
    return {row["frame"]: row for row in rows}, artifacts, binding


def verify_pairing(subject, run):
    from PIL import Image

    out = run.parent / "view-comparison"

    def checked_panel(panel, frame, crop, columns, source=None):
        x0, y0, x1, y1 = crop
        w, h = x1 - x0, y1 - y0
        require(panel["crop_xyxy"] == crop and panel["frame"] == frame and panel["columns"] == columns,
                f"Panel identity differs: {panel['file']}")
        with Image.open(pin(out / panel["file"], panel["sha256"])) as png:
            require(png.size == (len(columns) * w, h), f"Panel dimensions differ: {panel['file']}")
            if source is not None:
                require(np.array_equal(np.asarray(png)[:, :w, :], source[y0:y1, x0:x1]),
                        f"Panel source RGB differs: {panel['file']}")

    result_path = out / "result.json"
    result = read_json(result_path)
    plan_path = pin(out / "plan.json", result["plan_sha256"])
    plan = read_json(plan_path)
    require(result["state"] == "round11_delivery_views_complete" and result["subject"] == plan["subject"] == subject
            and result["commit"] == plan["commit"] == R11_COMMIT, "Pairing belongs to another candidate")
    previous = read_json(pin(R10_RECEIPT, R10_RECEIPT_SHA))["subjects"][subject]
    old_result = Path(previous["result"])
    wrapper = read_json(run.parent / "wrapper-terminal.json")
    require(plan["old"]["terminal_sha256"] == previous["result_sha256"]
            and plan["new"]["terminal_sha256"] == wrapper["result_sha256"]
            and wrapper["state"] == "completed" and wrapper["candidate_commit"] == R11_COMMIT,
            "Terminal identity differs from delivery")
    endpoints = {side: pairing_endpoint(plan[side], path)
                 for side, path in (("old", old_result), ("new", run / "result.json"))}
    old_binding = endpoints["old"][2]
    reference = plan["common_reference"]
    require(result["common_reference"] == reference
            and Path(reference["file"]) == endpoints["old"][1]["reference_mesh"]
            and reference["sha256"] == old_binding["inputs_sha256"]["reference_mesh"]
            and reference["frame"] == old_binding["reference_frame_id"]
            and reference["pose_index"] == old_binding["reference_pose_index"], "Common mesh differs from the R10 terminal")
    mesh_faces = {}
    for side, (_, artifacts, binding) in endpoints.items():
        mesh = read_npz(artifacts["reference_mesh"])
        require(mesh["reference_frame_id"].item() == binding["reference_frame_id"]
                and mesh["reference_pose_index"].item() == binding["reference_pose_index"]
                and mesh["vertices_world"].shape == (reference["vertices"], 3)
                and mesh["faces"].shape == (reference["ordered_faces"], 3),
                f"Reference mesh archive differs from binding: {side}")
        mesh_faces[side] = mesh["faces"]
    require(np.array_equal(mesh_faces["old"], mesh_faces["new"]), "Ordered triangle topology differs")
    old_review_path = pin(Path(previous["review"]), previous["review_sha256"])
    require(Path(plan["baseline_review"]) == old_review_path
            and plan["baseline_review_sha256"] == previous["review_sha256"], "Wrong R10 review pairing input")
    old_review = read_json(old_review_path)
    old_plan = read_json(pin(old_review_path.with_name("plan.json"), old_review["plan_sha256"]))
    old_rois_path = pin(old_review_path.with_name("rois.npz"), old_plan["roi_sha256"])
    rois_path = pin(out / "rois.npz", plan["roi_sha256"])
    expected_common_frames = sorted(set(endpoints["old"][0]) & set(endpoints["new"][0]))
    require(plan["matched_frames"] == expected_common_frames, "Native roster intersection changed")
    require(len(result["old"]["native"]) == len(result["new"]["native"]) == len(plan["pairs"]),
            "Native review pair count differs")
    native, special = {}, {}
    with np.load(old_rois_path, allow_pickle=False) as old_rois, np.load(rois_path, allow_pickle=False) as rois:
        for i, pair in enumerate(plan["pairs"]):
            role, frame, camera = (pair[key] for key in ("role", "frame", "camera"))
            require(role not in native and frame in expected_common_frames, "Invalid native pairing identity")
            mask_digest = pair["score_mask_sha256"]
            source_digest = pair["source_sha256"]
            with Image.open(pin(Path(pair["score_mask"]), mask_digest)) as mask:
                matte = np.asarray(mask.convert("L")) >= 253
            with Image.open(pin(Path(pair["source"]), source_digest)) as source:
                require(tuple(camera["wh"]) == source.size == matte.shape[::-1], "Wrong source pixel grid")
                source_pixels = np.asarray(source.convert("RGB"))
            regions = plan["native_regions"][role]
            previous_role = pair["pinned_r10_role"]
            for name, digest in regions.items():
                region = rois[f"{role}_{name}"]
                require(region.shape == matte.shape and region.dtype == np.bool_ and matte[region].all()
                        and hashlib.sha256(region.tobytes()).hexdigest() == digest, f"Invalid ROI: {role}/{name}")
                if previous_role:
                    require(np.array_equal(region, old_rois[f"{previous_role}_{name}"]),
                            f"Archived ROI changed: {role}/{name}")
            for side, (rows, _, _) in endpoints.items():
                row = rows[frame]
                rendered = result[side]["native"][i]
                require(row["source_sha256"] == rendered["source_sha256"] == source_digest
                        and row["mask_sha256"] == rendered["own_mask_sha256"] == rendered["score_mask_sha256"] == mask_digest
                        and row["source_frame_index"] == rendered["source_frame_index"] == pair["source_frame_index"]
                        and all(row[k] == camera[k] for k in ("K", "K_integer_centers", "w2c", "wh"))
                        and all(rendered["native_camera"][k] == camera[k] for k in ("K", "w2c", "wh"))
                        and rendered["role"] == role and rendered["frame"] == frame
                        and set(rendered["region_metrics"]) == set(regions), f"Native evidence differs: {side}/{frame}")
                panels = result[side]["panels"]
                checked_panel(panels[role], frame, pair["crop_xyxy"],
                              ["source_rgb", "black", "white", "alpha"], source_pixels)
                checked_panel(panels[f"{role}_geometry"], frame, pair["crop_xyxy"],
                              ["source_rgb", "keypoints", "mesh", "same_frame_gaussian"], source_pixels)
                if pair["head_crop_xyxy"]:
                    checked_panel(panels[f"{role}_head"], frame, pair["head_crop_xyxy"],
                                  ["source_rgb", "black", "white", "alpha"], source_pixels)
            native[role] = dict(frame=frame, source_sha256=source_digest, score_mask_sha256=mask_digest,
                                camera=camera, region_mask_sha256=regions, r10_roi_reused=previous_role is not None)
        if plan["common_special"]:
            old_terminal = read_json(old_result)
            replay_path = old_result.parent / old_terminal["bound_replay"]
            replay = read_json(pin(replay_path, old_terminal["bound_replay_sha256"]))
            posed_path = pin(replay_path.parent / replay["mesh_file"], replay["mesh_sha256"])
            archived = {row["frame"]: row for row in replay["records"]}
            prior_pairs = {row["frame"]: row for row in old_plan["pairs"]}
            for item in plan["common_special"]:
                frame, role, slot = (item[key] for key in ("frame", "role", "old_slot"))
                row = endpoints["old"][0][frame]
                original = archived[frame]
                prior = prior_pairs[frame]
                require(Path(item["common_mesh_file"]) == posed_path
                        and item["common_mesh_file_sha256"] == replay["mesh_sha256"]
                        and slot == original["pose_index"]
                        and Path(item["old_archive"]) == replay_path.parent / original["image"]
                        and item["old_archive_sha256"] == original["image_sha256"],
                        f"Special pairing uses a different R10 pose: {frame}")
                pin(Path(item["old_archive"]), item["old_archive_sha256"])
                with Image.open(pin(Path(item["source"]), row["source_sha256"])) as source:
                    source_pixels = np.asarray(source.convert("RGB"))
                with Image.open(pin(Path(item["score_mask"]), row["mask_sha256"])) as mask:
                    matte = np.asarray(mask.convert("L")) >= 253
                require(item["source_sha256"] == row["source_sha256"] and item["score_mask_sha256"] == row["mask_sha256"]
                        and item["crop_xyxy"] == prior["crop_xyxy"] and item["head_crop_xyxy"] == prior["head_crop_xyxy"]
                        and item["R11"] == plan["special_016"][frame]["R11"]
                        and (item["R11"] == "trained") == (frame in endpoints["new"][0]),
                        f"Special pairing input identity differs: {frame}")
                for name, digest in plan["native_regions"][role].items():
                    region = rois[f"{role}_{name}"]
                    require(region.shape == matte.shape and matte[region].all()
                            and hashlib.sha256(region.tobytes()).hexdigest() == digest
                            and np.array_equal(region, old_rois[f"{prior['role']}_{name}"]),
                            f"Special pairing ROI differs: {frame}/{name}")
                a, b = (result[side]["common_special"][frame] for side in ("old", "new"))
                require(a["camera"] == b["camera"]
                        and all(a["camera"][k] == item["camera"][k] == row[k] for k in ("K", "w2c", "wh"))
                        and a["archive_max_abs_8bit"] <= 1 and b["archive_max_abs_8bit"] is None,
                        f"Special camera or archive parity differs: {frame}")
                keys = ("crop_xyxy", "head_crop_xyxy", "common_mesh_file_sha256", "source_sha256", "score_mask_sha256")
                for side, arm in (("old", a), ("new", b)):
                    require(all(arm[key] == item[key] for key in keys) and arm["common_mesh_pose_index"] == slot
                            and arm["R11_fitted_pose_used"] is False
                            and arm["local_bound_ply_sha256"] == endpoints[side][2]["bound_ply_sha256"],
                            f"Special rendering used a different geometry or source: {frame}/{side}")
                    panels = result[side]["panels"]
                    checked_panel(panels[role], frame, item["crop_xyxy"],
                                  ["source_rgb", "black", "white", "alpha"], source_pixels)
                    if item["head_crop_xyxy"]:
                        checked_panel(panels[f"{role}_head"], frame, item["head_crop_xyxy"],
                                      ["source_rgb", "black", "white", "alpha"], source_pixels)
                special[frame] = dict(r11_pose_status=item["R11"], camera=item["camera"], common_pose_index=slot,
                    common_mesh_file_sha256=replay["mesh_sha256"], archive_max_abs_8bit=a["archive_max_abs_8bit"],
                    region_mask_sha256=plan["native_regions"][role], r11_fitted_pose_used=False)
    static = {}
    require(set(plan["static_views"]) == set(old_plan["static_views"]), "Controlled static view list changed")
    for name, view in plan["static_views"].items():
        raw, crop = view["fixed_world_camera"], view["crop_xyxy"]
        old_view = old_plan["static_views"][name]
        old_raw = old_view["fixed_world_camera"]
        require(all(raw[k] == old_raw[k] for k in ("K", "wh")) and raw["frame"] == reference["frame"],
                f"Fixed camera intrinsics differ: {name}")
        if name != "dolly":
            require(raw["w2c"] == old_raw["w2c"] and crop == old_view["crop_xyxy"],
                    f"Fixed camera position or crop differs: {name}")
        else:
            E, old_E = (np.asarray(row["w2c"]) for row in (raw, old_raw))
            front = np.asarray(old_plan["static_views"]["static_reference_front"]["fixed_world_camera"]["w2c"])
            fraction = view["near_movement_fraction"]
            require(fraction in (1., .75, .5, .25, 0.) and np.array_equal(E[:, :3], old_E[:, :3])
                    and np.allclose(E[:3, 3], front[:3, 3] + fraction * (old_E[:3, 3] - front[:3, 3]),
                                    rtol=0., atol=1e-12), "Common dolly changed beyond planned depth safety")
        require(result["old"]["common"][name]["camera"] == result["new"]["common"][name]["camera"],
                f"Common renderer used different cameras: {name}")
        actual_references = {}
        for side, (_, artifacts, binding) in endpoints.items():
            own, common = result[side]["static"][name], result[side]["common"][name]
            own_frame = binding["reference_frame_id"]
            arm = plan["static_arms"][side]
            transform = arm["world_to_viewer"]
            rotation = np.asarray(transform["rotation"])
            E = np.asarray(raw["w2c"])
            exported = np.asarray(own["camera"]["w2c"])
            require(transform["scale"] == 1. and np.allclose(exported[:3, :3] @ rotation, E[:3, :3],
                        rtol=0., atol=1e-10) and np.allclose(exported[:3, 3], E[:3, 3] + E[:3, :3] @ transform["origin_world"],
                        rtol=0., atol=1e-10), f"Static package camera does not map to the common world: {name}/{side}")
            require(common["common_mesh_sha256"] == reference["sha256"]
                    and common["local_bound_ply_sha256"] == arm["bound_ply_sha256"] == binding["bound_ply_sha256"]
                    and common["fixed_world_camera"] == own["fixed_world_camera"] == raw
                    and all(common["camera"][k] == raw[k] for k in ("K", "w2c", "wh"))
                    and all(own["camera"][k] == view["arms"][side]["camera"][k] for k in ("K", "w2c", "wh"))
                    and own["static_ply_sha256"] == arm["ply_sha256"] == sha(artifacts["static_ply"])
                    and common["actual_reference_frame"] == own["actual_reference_frame"] == own_frame
                    and common["source_rgb_compared"] is own["source_rgb_compared"] is False
                    and own["pose_driven"] is False, f"Static artifact binding differs: {name}/{side}")
            for panel, frame in ((common["panel"], reference["frame"]), (own["panel"], own_frame)):
                checked_panel(panel, frame, crop, ["black", "white", "alpha"])
            require(common["panel"]["file"] != own["panel"]["file"], f"Own reference panel aliases replay: {name}/{side}")
            actual_references[side] = own_frame
        static[name] = dict(fixed_world_camera=raw, crop_xyxy=crop, actual_references=actual_references,
                            shared_r10_mesh_sha256=reference["sha256"])
    return dict(state="verified", subject=subject, review_sha256=sha(result_path), plan_sha256=sha(plan_path),
                roi_archive_sha256=sha(rois_path), common_reference=reference, matched_frames=expected_common_frames,
                native=native, static=static, common_special=special, same_ordered_faces=True, static_rgb_ground_truth=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "verify", "pairing"))
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--baseline", type=Path, default=OUTPUT / "baseline-face-and-inputs.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "baseline":
        result = capture_baseline()
        target = args.output or args.baseline
    else:
        if args.subject is None:
            parser.error("--subject is required for verify and pairing")
        run = args.run or LAB / f"shuolee-smpl-delivery-round11-{args.subject}/run"
        if args.action == "verify":
            result = verify_round11(args.subject, run, args.baseline)
            target = args.output or OUTPUT / f"{args.subject}-face-and-coverage.json"
        else:
            result = verify_pairing(args.subject, run)
            target = args.output or OUTPUT / f"{args.subject}-pairing.json"
    write_json(target, result)
    print(json.dumps(dict(path=str(target), sha256=sha(target), state=result["state"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

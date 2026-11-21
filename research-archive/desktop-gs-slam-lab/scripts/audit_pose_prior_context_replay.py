"""CPU artifact audit of two-arm, single-frame frozen-map prior replay.

Never renders, runs ICP, reads GT or optimizes poses. Pose/prior/Adam equations
are independently recomputed from recorded tensors and step histories.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPLAY_SOURCE = "c724e7707d6d7a4ca712c4c8025e9ec3292dd1a14bc00087acfde72356834bda"
CONTEXT_SOURCE = "f1cc7b76a11cd4f4173ad56398bf1310a7dbe70f83b3ac03535e205e9dc6e4eb"
V4_SOURCE = "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f"
PRIOR_SOURCE = "d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b"
MAP_KEYS = ("means", "log_scales", "quats", "color_logits", "opacity_logits")
INPUT_KEYS = ("image", "depth_m", "valid_depth_mask", "K", "initial_w2c",
              "prior_reference_w2c", "icp_prefix_c2w_float64")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_manifest(tensors):
    rows = {}
    for key, value in sorted(tensors.items()):
        require(value.device.type == "cpu" and not value.requires_grad and value.grad is None, "snapshot tensor must be frozen CPU")
        rows[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
                     "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
                     "requires_grad": False}
    return {"tensors": rows, "aggregate_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def se3_numpy(twist):
    rho, omega = np.asarray(twist, dtype=np.float64)[:3], np.asarray(twist, dtype=np.float64)[3:]
    theta = np.linalg.norm(omega)
    x, y, z = omega
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    square = skew@skew
    if theta < 1e-5:
        theta2 = theta**2
        a = 1-theta2/6+theta2**2/120
        b = .5-theta2/24+theta2**2/720
        c = 1/6-theta2/120+theta2**2/5040
    else:
        a, b, c = math.sin(theta)/theta, (1-math.cos(theta))/theta**2, (theta-math.sin(theta))/theta**3
    result = np.eye(4)
    result[:3, :3] = np.eye(3)+a*skew+b*square
    result[:3, 3] = (np.eye(3)+b*skew+c*square)@rho
    return result


def canonical_numpy(camera):
    camera = np.array(camera, dtype=np.float64)
    left, singular, right = np.linalg.svd(camera[:3, :3])
    rotation = left@right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left@right
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, camera[:3, 3]
    return result.astype(np.float32).astype(np.float64)


def rigid_inverse(camera):
    camera = np.asarray(camera, dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = camera[:3, :3].T
    result[:3, 3] = -result[:3, :3]@camera[:3, 3]
    return result


def prior_numpy(candidate, reference, weight, sigma_t, sigma_r):
    if weight == 0:
        return 0.
    candidate, reference = rigid_inverse(candidate), rigid_inverse(reference)
    return float(weight*(np.sum((candidate[:3, 3]-reference[:3, 3])**2)/sigma_t**2+
                         np.sum((candidate[:3, :3]-reference[:3, :3])**2)/(2*sigma_r**2)))


def check_loss(loss, camera, reference, weight, config):
    values = [loss[k] for k in ("objective", "sensor_objective", "prior_objective", "rgb_mse", "depth_mae_m", "alpha_loss")]
    require(all(isinstance(v, (float, int)) and math.isfinite(v) for v in values), "nonfinite completed candidate loss")
    sensor = loss["rgb_mse"]+config["depth_weight"]*loss["depth_mae_m"]+config["alpha_weight"]*loss["alpha_loss"]
    sensor_error = abs(sensor-loss["sensor_objective"])
    total_error = abs(loss["objective"]-loss["sensor_objective"]-loss["prior_objective"])
    prior = prior_numpy(camera, reference, weight, config["pose_prior_translation_sigma_m"], math.radians(config["pose_prior_rotation_sigma_deg"]))
    prior_error = abs(prior-loss["prior_objective"])
    require(sensor_error < 2e-7+2e-6*abs(sensor), "sensor channel arithmetic")
    require(total_error < 2e-7+2e-6*abs(loss["objective"]), "combined arithmetic")
    require(prior_error < 1e-6+1e-5*abs(prior), "independent prior disagreement")
    if weight == 0:
        require(loss["prior_objective"] == 0 and loss["sensor_objective"] == loss["objective"], "weight0 not exact off")
    return np.array([sensor_error, total_error, prior_error])


def adam_numpy(state, gradient, moment, variance, step, learning_rates):
    moment = .9*moment+.1*gradient
    variance = .999*variance+.001*gradient**2
    denominator = np.sqrt(variance)/math.sqrt(1-.999**step)+1e-8
    updated = state-learning_rates/(1-.9**step)*moment/denominator
    return updated, moment, variance


def check_probe(probe):
    require(probe["status"] in ("passed", "failed"), "operation error is not a completed probe")
    require(probe["original_probe_executed_once"] is True and probe["finite_difference_epsilon"] == 1e-4 and probe["min_cosine"] == .8 and probe["max_relative_l2"] == .5, "probe helper/threshold drift")
    analytic = np.array(probe["analytic_six_dof"], dtype=np.float32)
    plus, minus = np.array(probe["positive_losses"], dtype=np.float32), np.array(probe["negative_losses"], dtype=np.float32)
    require(analytic.shape == plus.shape == minus.shape == (6,), "probe missing analytic or 12 finite differences")
    direct_division = (plus-minus)/np.float32(2*probe["finite_difference_epsilon"])
    # PyTorch CUDA scalar division can use a rounded reciprocal multiplication.
    # Preserve both arithmetic paths instead of demanding CPU division bits.
    reciprocal_multiply = (plus-minus)*np.float32(1/(2*probe["finite_difference_epsilon"]))
    recorded = np.array(probe["finite_difference_six_dof"], dtype=np.float32)
    division_exact = np.array_equal(direct_division, recorded)
    reciprocal_exact = np.array_equal(reciprocal_multiply, recorded)
    require(division_exact or reciprocal_exact, "finite differences disagree with both float32 scalar arithmetic paths")
    numeric = recorded
    finite = np.isfinite(analytic).all() and np.isfinite(numeric).all()
    require(bool(finite) == probe["all_finite"], "probe finite claim")
    if finite:
        a, n = analytic.astype(np.float64), numeric.astype(np.float64)
        cosine = float(a@n/(max(np.linalg.norm(a), 1e-8)*max(np.linalg.norm(n), 1e-8)))
        relative = float(np.linalg.norm(a-n)/max(np.linalg.norm(n), 1e-12))
        require(abs(cosine-probe["cosine_similarity"]) < 2e-6, "probe cosine arithmetic")
        require(abs(relative-probe["relative_l2_error"]) < 2e-6+2e-6*relative, "probe relative arithmetic")
        if probe["status"] == "passed":
            require(np.linalg.norm(a) >= 1e-10 and cosine >= .8-2e-6 and relative <= .5+2e-6, "passed probe violated original checks")
    if probe["status"] == "passed":
        original = probe["unchanged_v4_helper_result"]
        require(original["analytic_six_dof"] == probe["analytic_six_dof"] and original["finite_difference_six_dof"] == probe["finite_difference_six_dof"], "hook changed original helper vectors")
    return {"status": probe["status"], "recorded_analytic_calls": 1, "recorded_FD_loss_calls": 12,
            "float32_FD_subtraction_direct_division_exact": bool(division_exact),
            "float32_FD_subtraction_reciprocal_multiply_exact": bool(reciprocal_exact),
            "direct_CPU_division_vs_recorded_max_absolute_difference": float(np.max(np.abs(direct_division-recorded))),
            "all_finite": bool(finite),
            "call_count_scope": "Source helper executes once; saved 1 analytic vector and six positive/six negative losses. Artifact fields alone are not an external call trace."}


def audit_arm(folder, weight, context, manifest):
    result, progress = read_json(folder/"result.json"), read_json(folder/"progress.json")
    checkpoint = torch.load(folder/"final_pose.pt", map_location="cpu", weights_only=True)
    probe = read_json(folder/"pose_gradient_probe.json")
    require(result["weight"] == weight and result["status"] in ("completed_candidate_replay", "completed_candidate_replay_with_failures"), "arm status/weight")
    require(checkpoint["result"] == result and result["pose_gradient_probe"] == probe, "result checkpoint/probe")
    require(checkpoint["source_context_sha256"] == CONTEXT_SOURCE, "arm source context")
    require(result["fixed_tensor_hashes_before"] == result["fixed_tensor_hashes_after"] == progress["fixed_tensor_hashes_before"] == manifest, "arm changed fixed tensor bytes")
    require(progress["status"] == result["status"] and progress["steps_completed"] == result["steps_completed"] and progress["failure_flags"] == result["failure_flags"], "arm progress")
    require(result["candidate_validated_by_probe"] is (probe["status"] == "passed"), "probe failure relabelled")
    if probe["status"] == "failed":
        require("pose_gradient_finite_difference_check_failed" in result["failure_flags"], "probe failure omitted")
    require(bool(result["failure_flags"]) is (result["status"] == "completed_candidate_replay_with_failures"), "flags/success mismatch")
    config = context["configuration"]
    initial, reference = context["initial_w2c"].numpy().astype(np.float64), context["prior_reference_w2c"].numpy().astype(np.float64)
    require(np.array_equal(initial, result["initial_w2c"]) and np.array_equal(reference, result["prior_reference_w2c"]), "arm initial/reference differ")
    rows = [json.loads(line) for line in (folder/"steps.jsonl").read_text().splitlines() if line.strip()]
    normal = [r for r in rows if "optimizer_update" in r]
    exceptional = [r for r in rows if "optimizer_update" not in r]
    require(len(normal) == result["steps_completed"] <= result["steps_requested"] == 100 and len(exceptional) <= 1, "step counts/drop")
    require(not exceptional or bool(result["failure_flags"]), "exceptional row without flags")
    max_errors = {"se3_raw_camera": 0., "canonical_camera": 0., "Adam_update": 0., "gradient_clip": 0., "reported_gradient_norm": 0.,
                  "sensor_arithmetic": 0., "combined_arithmetic": 0., "independent_prior": 0.}
    def loss_check(loss, camera):
        errors = check_loss(loss, camera, reference, weight, config)
        for key, value in zip(("sensor_arithmetic", "combined_arithmetic", "independent_prior"), errors):
            max_errors[key] = max(max_errors[key], float(value))
    loss_check(result["initial_loss"], initial)
    best_loss, best_step, best_camera = result["initial_loss"]["objective"], 0, initial
    selected_raw, selected_raw_loss = initial, result["initial_loss"]
    moment, variance, previous = np.zeros(6), np.zeros(6), np.zeros(6)
    for step, row in enumerate(normal):
        require(row["candidate_step"] == step and row["optimizer_update"] == step+1, "candidate/update indexing")
        before = np.array(row["translation_tangent_before_update"]+row["rotation_tangent_rad_before_update"])
        after = np.array(row["translation_tangent_after_update"]+row["rotation_tangent_rad_after_update"])
        require(np.array_equal(before, previous), "tangent update history discontinuity")
        raw = np.array(row["raw_candidate_w2c"])
        pose_error = float(np.max(np.abs(raw-se3_numpy(before)@initial)))
        max_errors["se3_raw_camera"] = max(max_errors["se3_raw_camera"], pose_error)
        require(pose_error < 3e-6, "recorded raw camera not left-SE3 tangent")
        loss_check(row["loss"], raw)
        considered = row["loss"]["objective"] < best_loss
        canonical_loss = row["canonical_candidate_loss_if_considered"]
        require((canonical_loss is not None) is considered, "canonical consideration mismatch")
        updated = False
        if considered:
            canonical = canonical_numpy(raw)
            loss_check(canonical_loss, canonical)
            if canonical_loss["objective"] < best_loss:
                best_loss, best_step, best_camera = canonical_loss["objective"], step, canonical
                selected_raw, selected_raw_loss, updated = raw, row["loss"], True
        require(row["canonical_best_updated"] is updated and row["best_step"] == best_step and row["best_combined_objective"] == best_loss, "combined canonical best ledger")
        camera_error = float(np.max(np.abs(np.array(row["selected_canonical_w2c"])-best_camera)))
        max_errors["canonical_camera"] = max(max_errors["canonical_camera"], camera_error)
        require(camera_error < 1e-6 and np.array_equal(np.array(row["selected_raw_w2c"]), selected_raw) and row["selected_raw_loss"] == selected_raw_loss, "best candidate pose/raw loss ledger")
        grad_before, grad_after = np.array(row["gradient_six_dof_before_clip"]), np.array(row["gradient_six_dof_after_clip"])
        norm = float(np.linalg.norm(grad_before))
        norm_error = abs(norm-row["gradient_norm_before_clip"])
        max_errors["reported_gradient_norm"] = max(max_errors["reported_gradient_norm"], norm_error)
        require(norm_error < 2e-6+2e-6*norm, "recorded clipping norm")
        expected_clip = grad_before*min(1., 1./(norm+1e-6))
        clip_error = float(np.max(np.abs(expected_clip-grad_after)))
        max_errors["gradient_clip"] = max(max_errors["gradient_clip"], clip_error)
        require(clip_error < 2e-6, "gradient clipping mismatch")
        lrs = np.array(row["learning_rates"])
        expected_lrs = np.array([config["pose_lr_translation"], config["pose_lr_rotation"]])*.99**step
        require(np.max(np.abs(lrs-expected_lrs)) < 2e-14, "scheduler learning rate")
        updated_state, moment, variance = adam_numpy(before, grad_after, moment, variance, step+1, np.repeat(lrs, 3))
        update_error = float(np.max(np.abs(updated_state-after)))
        max_errors["Adam_update"] = max(max_errors["Adam_update"], update_error)
        require(update_error < 2e-6, "Adam update differs from independent moments/bias-correction")
        previous = after
    require(not exceptional, "Current completed audit requires full numeric row fields; early-failure row preserved but not silently accepted")
    require(np.array_equal(checkpoint["translation_tangent_last"].numpy(), previous[:3]) and np.array_equal(checkpoint["rotation_tangent_rad_last"].numpy(), previous[3:]), "last tangent checkpoint")
    last = result["last_candidate_check"]
    require(last["candidate_step"] == result["steps_completed"], "last step mislabel")
    raw_last = np.array(last["raw_candidate_w2c"])
    require(np.max(np.abs(raw_last-se3_numpy(previous)@initial)) < 3e-6, "last tangent/camera")
    canonical_last = np.array(last["canonical_w2c"])
    require(np.max(np.abs(canonical_last-canonical_numpy(raw_last))) < 1e-6, "last canonical camera")
    loss_check(last["canonical_loss"], canonical_last)
    last_best = last["canonical_loss"]["objective"] < best_loss
    require(last["selected_as_best"] is last_best, "last candidate selection")
    if last_best:
        best_camera, best_step, best_loss = canonical_last, result["steps_completed"], last["canonical_loss"]["objective"]
        selected_raw = raw_last
    require(result["best_step_selected_by_tracking_objective_only"] == best_step, "final selected step")
    require(abs(result["selected_loss"]["objective"]-best_loss) < 2e-7+2e-6*abs(best_loss), "final selected rerender combined objective")
    selected = np.array(result["final_selected_w2c"])
    require(np.max(np.abs(selected-best_camera)) < 1e-6 and np.array_equal(checkpoint["final_selected_w2c"].numpy(), selected), "final pose ledger/checkpoint")
    require(np.array_equal(np.array(result["selected_raw_w2c_before_projection"]), selected_raw), "final raw selection ledger")
    loss_check(result["selected_loss"], selected)
    loss_check(result["selected_raw_candidate_loss_before_projection"], selected_raw)
    inverse_error = float(np.max(np.abs(np.array(result["final_selected_c2w"])-rigid_inverse(selected))))
    require(inverse_error < 2e-6, "final analytic inverse")
    correction = float(np.linalg.norm(rigid_inverse(selected)[:3, 3]-rigid_inverse(initial)[:3, 3]))
    require(abs(correction-result["camera_center_correction_m"]) < 2e-7, "recorded center correction")
    csv_rows = list(csv.DictReader((folder/"loss.csv").open(newline="", encoding="utf-8-sig")))
    require(len(csv_rows) == len(rows), "CSV step lost")
    for csv_row, raw_row in zip(csv_rows, rows):
        flattened = {**raw_row, **raw_row["loss"]}
        for key, value in csv_row.items():
            original = flattened.get(key)
            if original is None: require(value == "", "CSV missing-value altered")
            elif isinstance(original, bool): require(value == str(original), "CSV boolean altered")
            elif isinstance(original, (int, float)): require(float(value) == original, "CSV scalar altered")
            else: require(value == str(original), "CSV field altered")
    return {"status": "passed", "weight": weight, "probe": check_probe(probe), "failure_flags": result["failure_flags"],
            "steps_completed": result["steps_completed"], "best_step": best_step, "independent_numeric_max_errors": max_errors,
            "camera_center_correction_m": correction, "initial_loss": result["initial_loss"], "selected_loss": result["selected_loss"],
            "same_fixed_tensor_bytes_and_frozen_map": True, "step_and_CSV_and_checkpoint_consistent": True,
            "input_hashes": {name: sha(folder/name) for name in ("result.json", "progress.json", "pose_gradient_probe.json", "steps.jsonl", "loss.csv", "final_pose.pt")}}


def self_checks():
    require(np.array_equal(se3_numpy(np.zeros(6)), np.eye(4)), "zero SE3")
    translation = se3_numpy([.02, -.03, .01, 0, 0, 0])
    require(np.max(np.abs(translation[:3, 3]-[.02, -.03, .01])) == 0, "SE3 translation")
    rotation = se3_numpy([0, 0, 0, 0, 0, .1])
    require(abs(rotation[0, 1]+math.sin(.1)) < 1e-14, "SE3 rotation")
    require(abs(prior_numpy(translation, np.eye(4), .01, .01, math.radians(1))-.14) < 1e-14, "hand prior")
    update, moment, variance = adam_numpy(np.zeros(6), np.array([1., -1., 2., -2., 3., -3.]), np.zeros(6), np.zeros(6), 1, np.full(6, .003))
    expected = -.003*np.array([1., -1., 2., -2., 3., -3.])/(np.array([1., 1., 2., 2., 3., 3.])+1e-8)
    require(np.max(np.abs(update-expected)) < 1e-14, "Adam first-step hand control")
    wrong_lr = adam_numpy(np.zeros(6), np.array([1., -1., 2., -2., 3., -3.]), np.zeros(6), np.zeros(6), 1, np.full(6, .006))[0]
    require(np.max(np.abs(wrong_lr-update)) > .002, "Adam learning-rate negative control")
    require(not np.allclose(se3_numpy([.02, -.03, .01, .04, -.02, .03]), se3_numpy([.04, -.02, .03, .02, -.03, .01])), "twist-order negative control")
    return {"status": "passed", "hand_SE3_translation_rotation_and_prior": True, "Adam_hand_first_step": True,
            "wrong_LR_and_twist_order_controls_sensitive": True}


def audit_run(run):
    folder = ROOT/"outputs"/run
    metrics, config = read_json(folder/"metrics.json"), read_json(folder/"configuration.json")
    context_path = ROOT/"outputs/tum-online-v4-prior0-transfer100-001/first_tracking_context.pt"
    require(sha(context_path) == CONTEXT_SOURCE, "immutable context changed")
    context = torch.load(context_path, map_location="cpu", weights_only=True)
    tensors = {**{"map."+k: context["parameters"][k] for k in MAP_KEYS}, **{k: context[k] for k in INPUT_KEYS}}
    manifest = tensor_manifest(tensors)
    require(metrics["status"] in ("completed", "completed_with_candidate_failures") and metrics["run"] == config["run"] == run, "partial/incorrect run")
    require(metrics["replay_script_sha256"] == config["replay_script_sha256"] == sha(ROOT/"scripts/replay_pose_prior_context.py") == REPLAY_SOURCE, "source lock")
    require(sha(ROOT/"scripts/train_tum_online_gs_v4.py") == V4_SOURCE and sha(ROOT/"scripts/geometric_pose_prior.py") == PRIOR_SOURCE, "original frozen source changed")
    require(metrics["source_context_sha256"] == config["source_context_sha256"] == CONTEXT_SOURCE, "context source provenance")
    require(metrics["fixed_tensor_hashes_before"] == metrics["fixed_tensor_hashes_after"] == manifest and metrics["same_fixed_input_bytes_for_both_arms"] is True, "both arms not actual identical snapshot bytes")
    require(config["ground_truth_read"] is False and config["dataset_loader_calls"] == config["ICP_calls"] == config["warmup_steps"] == config["mapping_steps"] == 0, "forbidden replay data/training operations")
    require(config["arm_weights"] == [0., .01] and len(metrics["arms"]) == 2, "missing arm")
    arms = []
    for arm, weight, original in zip(("prior0", "prior001"), (0., .01), metrics["arms"]):
        require(read_json(folder/arm/"result.json") == original, "root versus arm result")
        arms.append(audit_arm(folder/arm, weight, context, manifest))
    require(bool(any(a["failure_flags"] for a in arms)) is (metrics["status"] == "completed_with_candidate_failures"), "top-level failure status")
    require(not torch.cuda.is_initialized(), "CPU auditor initialized CUDA")
    return {"status": "passed", "run": run, "artifact_auditor_sha256": sha(Path(__file__)),
            "replay_source_sha256": REPLAY_SOURCE, "immutable_context_sha256": CONTEXT_SOURCE,
            "shared_input_tensor_aggregate_sha256": manifest["aggregate_sha256"], "arms": arms,
            "input_hashes": {"metrics.json": sha(folder/"metrics.json"), "configuration.json": sha(folder/"configuration.json")},
            "CUDA_initialized": False, "limits": ["No sensor rerender or derivative recalculation on the real rasterizer.",
                "Scalar sensor/channel and geometry/prior arithmetic are checked from preserved actual fields.",
                "NumPy float64 equations allow explicit rounding tolerances versus recorded GPU float32 states.",
                "One frozen-map current201 candidate experiment; no accepted trajectory, GT pose accuracy or full SLAM claim.",
                "Initial first-probe status is independent of the final candidate state; failed probes remain failed.",
                "The preserved early-failure step rows are rejected by this completed-run checker rather than silently dropped."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"self_checks": self_checks()} if args.self_test else {}
    if args.run:
        require(Path(args.run).name == args.run and args.run not in ("", ".", ".."), "single run name")
        result.update(audit_run(args.run))
    elif not args.self_test:
        parser.error("provide --run or --self-test")
    output = args.output or (ROOT/"outputs"/args.run/"independent_replay_artifact_audit.json" if args.run else None)
    if output:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

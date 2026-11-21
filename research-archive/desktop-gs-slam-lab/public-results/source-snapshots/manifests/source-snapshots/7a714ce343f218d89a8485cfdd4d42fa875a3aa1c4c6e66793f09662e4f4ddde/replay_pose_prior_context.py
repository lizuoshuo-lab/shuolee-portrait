"""Two-arm single-frame pose replay from one immutable V4 first-probe context.

This reads only prior0's saved warmup map/current sensor/initial camera. It does
not load a dataset, run ICP/warmup/mapping, evaluate GT, or claim online SLAM.
The exact locked V4 render/loss/rigid helpers are extracted without calling main.
GPU execution requires a new unique --run; --check-only never initializes CUDA.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from geometric_pose_prior import geometric_pose_prior, cpu_checks as prior_cpu_checks
from track_fixed_map import se3_exp, cpu_math_check, pose_gradient_probe

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_RUN = "tum-online-v4-prior0-transfer100-001"
V4_HASH = "ee7e28700b489a12208006c7c7cc407dbd9af9ab655d0d310f5dde8e7ef4506f"
PRIOR_HASH = "d61456c32d3556f598a7b1d62d6c60a92e90504870d979d408bdfbdcb617dc5b"
POSE_HASH = "c6f8ca3e80747decc7805aa567db44179f5a6e17b8ee7c36ef8565273463fdab"
CONTEXT_HASH = "f1cc7b76a11cd4f4173ad56398bf1310a7dbe70f83b3ac03535e205e9dc6e4eb"
ARM_WEIGHTS = (0., .01)
MAP_KEYS = ("means", "log_scales", "quats", "color_logits", "opacity_logits")
INPUT_KEYS = ("image", "depth_m", "valid_depth_mask", "K", "initial_w2c",
              "prior_reference_w2c", "icp_prefix_c2w_float64")
TOP_HELPERS = ("combine_tracking_loss", "canonicalize_rigid_camera", "rigid_camera_inverse")
MAIN_HELPERS = ("render", "data_loss", "loss_numbers")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def tensor_manifest(tensors: dict[str, torch.Tensor]) -> dict:
    result = {}
    for key, value in sorted(tensors.items()):
        array = value.detach().cpu().contiguous().numpy()
        result[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
                       "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                       "requires_grad": value.requires_grad}
    aggregate = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"tensors": result, "aggregate_sha256": aggregate}


def locked_helpers() -> tuple[dict, dict]:
    """Execute only selected trusted functions; main/imports/data I/O are absent."""
    source_path = ROOT / "scripts/train_tum_online_gs_v4.py"
    checks = {"v4_source_sha256": sha_file(source_path),
              "prior_helper_sha256": sha_file(ROOT / "scripts/geometric_pose_prior.py"),
              "pose_helper_sha256": sha_file(ROOT / "scripts/track_fixed_map.py")}
    if tuple(checks.values()) != (V4_HASH, PRIOR_HASH, POSE_HASH):
        raise ValueError("Locked V4/prior/pose source hash changed; refusing mixed-source replay")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    top = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    nested = {node.name: node for node in top["main"].body if isinstance(node, ast.FunctionDef)}
    selected = [top[name] for name in TOP_HELPERS]+[nested[name] for name in MAIN_HELPERS]
    functions = {"torch": torch, "F": F, "math": math, "geometric_pose_prior": geometric_pose_prior}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), functions)
    checks["extracted_function_ast_sha256"] = {
        node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in selected}
    checks["main_executed"] = False
    checks["dataset_or_icp_loader_imported_by_replay"] = False
    return functions, checks


def load_context() -> tuple[dict, dict, dict]:
    folder = ROOT / "outputs" / CONTEXT_RUN
    checkpoint, json_path = folder / "first_tracking_context.pt", folder / "first_tracking_context.json"
    actual_hash = sha_file(checkpoint)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if actual_hash != CONTEXT_HASH or metadata["checkpoint_sha256"] != CONTEXT_HASH:
        raise ValueError("The prescribed immutable first context hash does not match")
    context = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {"status", "implementation_version", "configuration", "ground_truth_included",
                "future_sensor_observations_included", "future_maps_or_trajectories_included", "source",
                "parameters", "source_frame_ids_per_gaussian", "map_role", *INPUT_KEYS,
                "sensor_metadata", "initial_tracking_loss", "tracking_objective"}
    if set(context) != expected:
        raise ValueError("Unexpected context fields, including any GT/trajectory/future inputs, are forbidden")
    if context["implementation_version"] != 4 or context["source"]["script_sha256"] != V4_HASH:
        raise ValueError("Context must come from the locked V4 source")
    if context["source"]["pose_prior_helper_sha256"] != PRIOR_HASH or context["source"]["pose_helper_sha256"] != POSE_HASH:
        raise ValueError("Context helper hashes do not match")
    excluded = ("ground_truth_included", "future_sensor_observations_included", "future_maps_or_trajectories_included")
    if any(context[key] is not False or metadata[key] is not False for key in excluded):
        raise ValueError("Only causal GT-free first tracking context is permitted")
    if metadata["saved_before_first_pose_gradient_probe"] is not True or metadata["tracking_optimizer_steps_executed_before_save"] != 0:
        raise ValueError("Context was not saved at the causal first-probe barrier")
    config = context["configuration"]
    locked_config = {"run": CONTEXT_RUN, "width": 160, "height": 120, "tracking_steps": 100,
                     "pose_initializer": "icp-prefix", "tracking_mode": "optimize", "pose_prior_weight": 0.,
                     "pose_prior_translation_sigma_m": .01, "pose_prior_rotation_sigma_deg": 1.,
                     "pose_lr_translation": .003, "pose_lr_rotation": .001, "depth_weight": 1.,
                     "alpha_weight": .01, "alpha_threshold": .5, "max_depth": 5., "seed": 0}
    if any(config[key] != value for key, value in locked_config.items()) or config != metadata["configuration"]:
        raise ValueError("Prescribed configuration changed")
    if context["sensor_metadata"]["frame_id"] != 201 or context["sensor_metadata"]["local_index"] != 1:
        raise ValueError("Replay is restricted to current frame201 at the first probe")
    if set(context["parameters"]) != set(MAP_KEYS) or set(context["source_frame_ids_per_gaussian"]) != {200}:
        raise ValueError("Expected only the first-frame200 warmup map")
    count = len(context["parameters"]["means"])
    shapes = {"means": (count, 3), "log_scales": (count, 3), "quats": (count, 4),
              "color_logits": (count, 3), "opacity_logits": (count,)}
    if count != 12108 or len(context["source_frame_ids_per_gaussian"]) != count:
        raise ValueError("Unexpected warmup point count/source associations")
    for key, value in context["parameters"].items():
        if tuple(value.shape) != shapes[key] or value.dtype != torch.float32:
            raise ValueError("Unexpected Gaussian tensor shape/type: "+key)
    input_shapes = {"image": (120, 160, 3), "depth_m": (120, 160), "valid_depth_mask": (120, 160),
                    "K": (3, 3), "initial_w2c": (4, 4), "prior_reference_w2c": (4, 4),
                    "icp_prefix_c2w_float64": (4, 4)}
    for key, shape in input_shapes.items():
        if tuple(context[key].shape) != shape:
            raise ValueError("Unexpected input tensor shape: "+key)
    tensors = {**{"map."+key: value for key, value in context["parameters"].items()},
               **{key: context[key] for key in INPUT_KEYS}}
    if any(value.requires_grad or value.grad is not None or value.device.type != "cpu" or
           not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("All source map/input tensors must be finite frozen CPU tensors")
    if context["valid_depth_mask"].dtype != torch.bool or int(context["valid_depth_mask"].sum()) != metadata["valid_depth_pixels"]:
        raise ValueError("Invalid valid-depth mask")
    if context["image"].min() < 0 or context["image"].max() > 1 or not context["valid_depth_mask"].any():
        raise ValueError("Invalid sensor ranges or empty valid-depth input")
    if not torch.equal(context["initial_w2c"], context["prior_reference_w2c"]):
        raise ValueError("Prior reference must exactly equal the fixed ICP-prefix initializer")
    if context["sensor_metadata"] != metadata["sensor_metadata"]:
        raise ValueError("JSON/PT sensor metadata mismatch")
    return context, metadata, {"checkpoint": str(checkpoint), "checkpoint_sha256": actual_hash,
                               "context_json_sha256": sha_file(json_path), "input_tensor_manifest": tensor_manifest(tensors),
                               "map_point_count": count, "sensor_metadata": context["sensor_metadata"],
                               "GT_future_fields_absent_except_explicit_false_flags": True}


def cpu_checks(context: dict, functions: dict, provenance: dict) -> dict:
    before = tensor_manifest({**{"map."+key: value for key, value in context["parameters"].items()},
                              **{key: context[key] for key in INPUT_KEYS}})
    reference = context["prior_reference_w2c"].double()
    x = torch.tensor([.02, -.01, .01, .03, -.02, .01], dtype=torch.float64, requires_grad=True)
    candidate = se3_exp(x) @ reference
    sensor_objective = candidate.square().mean()
    off = functions["combine_tracking_loss"]({"objective": sensor_objective}, candidate, reference,
                                             weight=0., sigma_translation_m=.01, sigma_rotation_deg=1.)
    direct_grad = torch.autograd.grad(sensor_objective, x, retain_graph=True)[0]
    off_grad = torch.autograd.grad(off["objective"], x, retain_graph=True)[0]
    if not torch.equal(off["objective"], sensor_objective) or not torch.equal(off_grad, direct_grad):
        raise RuntimeError("Zero-prior objective/gradient is not exactly off")
    enabled = functions["combine_tracking_loss"]({"objective": sensor_objective}, candidate, reference,
                                                  weight=.01, sigma_translation_m=.01, sigma_rotation_deg=1.)
    if not enabled["prior_objective"] > 0 or not torch.equal(enabled["objective"], sensor_objective+enabled["prior_objective"]):
        raise RuntimeError("Positive prior/combined target control failed")
    normalized, correction = functions["canonicalize_rigid_camera"](reference)
    inverse = functions["rigid_camera_inverse"](normalized)
    if not torch.equal(normalized[3], normalized.new_tensor([0., 0., 0., 1.])):
        raise RuntimeError("Rigid bottom row control failed")
    roundtrip = float((inverse @ normalized-torch.eye(4, dtype=normalized.dtype)).abs().max())
    if roundtrip > 2e-6:
        raise RuntimeError("Rigid inverse CPU check failed")
    # A failing six-dimensional probe is retained while V4's candidate loop can
    # still run. No passed status or accepted localization is fabricated.
    failed = retained_probe(lambda twist: twist.sum()*0., device="cpu", dtype=torch.float64)
    if failed["status"] != "failed" or failed["all_finite"] is not True:
        raise RuntimeError("Probe-failure retention check failed")
    after = tensor_manifest({**{"map."+key: value for key, value in context["parameters"].items()},
                             **{key: context[key] for key in INPUT_KEYS}})
    if before != after or torch.cuda.is_initialized():
        raise RuntimeError("CPU checks changed fixed input or initialized CUDA")
    return {"status": "passed", "cuda_initialized": False, "gsplat_imported": "gsplat" in sys.modules,
            "fixed_input_hashes_unchanged": before == after, "locked_provenance": provenance,
            "zero_prior_scalar_and_six_dof_gradient_exact_off": True,
            "positive_prior_and_combined_scalar_passed": True, "probe_failure_not_relabelled_passed": failed,
            "rigid_roundtrip_max_abs_error": roundtrip, "rigid_projection": correction,
            "se3_controls": cpu_math_check(), "prior_helper_controls": prior_cpu_checks()}


def retained_probe(loss_from_twist, *, device: str, dtype=torch.float32) -> dict:
    """Keep numerical evidence even when the unchanged V4 helper rejects it."""
    epsilon, min_cosine, max_relative_l2 = 1e-4, .8, .5
    vector = torch.zeros(6, device=device, dtype=dtype, requires_grad=True)
    analytic = torch.autograd.grad(loss_from_twist(vector), vector)[0]
    plus, minus = [], []
    with torch.no_grad():
        for dimension in range(6):
            offset = torch.zeros(6, device=device, dtype=dtype)
            offset[dimension] = epsilon
            plus.append(float(loss_from_twist(offset)))
            minus.append(float(loss_from_twist(-offset)))
    numeric = torch.tensor([(p-m)/(2*epsilon) for p, m in zip(plus, minus)], device=device, dtype=dtype)
    cosine = float(F.cosine_similarity(analytic[None], numeric[None]))
    relative_l2 = float(torch.linalg.vector_norm(analytic-numeric)/torch.linalg.vector_norm(numeric).clamp_min(1e-12))
    evidence = {"finite_difference_epsilon": epsilon, "min_cosine": min_cosine,
                "max_relative_l2": max_relative_l2, "analytic_six_dof": analytic.detach().cpu().tolist(),
                "finite_difference_six_dof": numeric.detach().cpu().tolist(), "positive_losses": plus,
                "negative_losses": minus, "cosine_similarity": cosine, "relative_l2_error": relative_l2,
                "all_finite": bool(torch.isfinite(analytic).all() and torch.isfinite(numeric).all()),
                "nonzero_analytic_components": int((analytic.abs()>1e-10).sum()),
                "recorded_evidence_does_not_change_thresholds": True}
    try:
        verified = pose_gradient_probe(loss_from_twist, device=device, dtype=dtype, epsilon=epsilon,
                                       min_cosine=min_cosine, max_relative_l2=max_relative_l2)
        return {**evidence, "status": "passed", "unchanged_v4_helper_result": verified}
    except RuntimeError as error:
        return {**evidence, "status": "failed", "reason": str(error),
                "candidate_optimization_does_not_turn_failed_probe_into_validated_localization": True}


def finite_json(value):
    """Preserve nonfinite failures as explicit strings, never fabricated numbers."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def replay_arm(weight: float, folder: Path, context: dict, functions: dict, tensors: dict) -> dict:
    folder.mkdir(exist_ok=False)
    config = context["configuration"]
    initial, reference = tensors["initial_w2c"], tensors["prior_reference_w2c"]
    translation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    rotation = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    sensor = functions["data_loss"]
    numbers = functions["loss_numbers"]
    canonicalize = functions["canonicalize_rigid_camera"]
    inverse = functions["rigid_camera_inverse"]

    def pose(twist=None):
        return se3_exp(torch.cat((translation, rotation)) if twist is None else twist) @ initial

    def loss(camera):
        return functions["combine_tracking_loss"](sensor(camera, 0), camera, reference, weight=weight,
                   sigma_translation_m=config["pose_prior_translation_sigma_m"],
                   sigma_rotation_deg=config["pose_prior_rotation_sigma_deg"])

    before_hashes = tensor_manifest(tensors)
    if any(value.requires_grad or value.grad is not None for value in tensors.values()):
        raise RuntimeError("Map and inputs must be frozen for both arms")
    progress = {"status": "running", "weight": weight, "steps_completed": 0,
                "fixed_tensor_hashes_before": before_hashes, "failure_flags": []}
    atomic_json(folder / "progress.json", progress)
    start = time.perf_counter()
    rows, flags = [], []
    try:
        with torch.no_grad():
            initial_loss = numbers(loss(initial))
        probe = retained_probe(lambda twist: loss(pose(twist))["objective"], device="cuda")
        atomic_json(folder / "pose_gradient_probe.json", finite_json(probe))
        if probe["status"] != "passed":
            flags.append("pose_gradient_finite_difference_check_failed")
        optimizer = torch.optim.Adam([{"params": [translation], "lr": config["pose_lr_translation"]},
                                      {"params": [rotation], "lr": config["pose_lr_rotation"]}], eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)
        best_loss, best_pose, best_step = initial_loss["objective"], initial.clone(), 0
        selected_raw_pose, selected_raw_loss = initial.clone(), initial_loss
        _, selected_projection = canonicalize(initial)
        steps_completed = 0
        with (folder / "steps.jsonl").open("x", encoding="utf-8") as stream:
            for step in range(config["tracking_steps"]):
                candidate = pose()
                losses = loss(candidate)
                objective = losses["objective"]
                raw_numbers = numbers(losses)
                canonical_numbers, candidate_projection = None, None
                best_updated = False
                if not torch.isfinite(objective):
                    flags.append("tracking_nonfinite_objective")
                    rows.append({"candidate_step": step, "status": "nonfinite_objective", "loss": raw_numbers})
                    stream.write(json.dumps(finite_json(rows[-1]))+"\n")
                    stream.flush()
                    break
                if float(objective.detach()) < best_loss:
                    canonical_candidate, candidate_projection = canonicalize(candidate)
                    with torch.no_grad():
                        canonical_numbers = numbers(loss(canonical_candidate))
                    if canonical_numbers["objective"] < best_loss:
                        best_loss, best_pose, best_step = canonical_numbers["objective"], canonical_candidate, step
                        selected_raw_pose = candidate.detach().clone()
                        selected_raw_loss, selected_projection = raw_numbers, candidate_projection
                        best_updated = True
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                if any(value.grad is None or not torch.isfinite(value.grad).all() for value in (translation, rotation)):
                    flags.append("tracking_missing_or_nonfinite_gradient")
                    rows.append({"candidate_step": step, "status": "gradient_failed", "loss": raw_numbers})
                    stream.write(json.dumps(finite_json(rows[-1]))+"\n")
                    stream.flush()
                    break
                grad_before = torch.cat((translation.grad, rotation.grad)).detach().cpu().tolist()
                grad_norm = float(torch.nn.utils.clip_grad_norm_([translation, rotation], max_norm=1.))
                grad_after = torch.cat((translation.grad, rotation.grad)).detach().cpu().tolist()
                lrs = [group["lr"] for group in optimizer.param_groups]
                row = {"candidate_step": step, "optimizer_update": step+1, "loss": raw_numbers,
                       "translation_tangent_before_update": translation.detach().cpu().tolist(),
                       "rotation_tangent_rad_before_update": rotation.detach().cpu().tolist(),
                       "raw_candidate_w2c": candidate.detach().cpu().tolist(),
                       "canonical_candidate_loss_if_considered": canonical_numbers,
                       "canonical_candidate_projection_if_considered": candidate_projection,
                       "canonical_best_updated": best_updated, "best_step": best_step,
                       "best_combined_objective": best_loss, "selected_canonical_w2c": best_pose.cpu().tolist(),
                       "selected_raw_w2c": selected_raw_pose.cpu().tolist(), "selected_raw_loss": selected_raw_loss,
                       "selected_projection": selected_projection, "gradient_six_dof_before_clip": grad_before,
                       "gradient_six_dof_after_clip": grad_after, "gradient_norm_before_clip": grad_norm,
                       "learning_rates": lrs}
                optimizer.step()
                scheduler.step()
                steps_completed = step+1
                row["translation_tangent_after_update"] = translation.detach().cpu().tolist()
                row["rotation_tangent_rad_after_update"] = rotation.detach().cpu().tolist()
                rows.append(row)
                stream.write(json.dumps(finite_json(row), allow_nan=False)+"\n")
                stream.flush()
                progress.update({"steps_completed": steps_completed, "best_step": best_step,
                                 "best_combined_objective": best_loss, "failure_flags": flags})
                atomic_json(folder / "progress.json", progress)
        # V4 explicitly evaluates the post-update step100 candidate, then
        # canonicalizes and recomputes combined loss before any best selection.
        with torch.no_grad():
            last_pose = pose()
            last_check = {"finite_camera": bool(torch.isfinite(last_pose).all()),
                          "raw_candidate_w2c": last_pose.cpu().tolist(), "selected_as_best": False}
            if torch.isfinite(last_pose).all():
                canonical_last, last_projection = canonicalize(last_pose)
                last_loss = numbers(loss(canonical_last))
                last_check.update({"canonical_w2c": canonical_last.cpu().tolist(),
                                   "canonical_loss": last_loss, "projection": last_projection})
                if math.isfinite(last_loss["objective"]) and last_loss["objective"] < best_loss:
                    best_loss, best_pose, best_step = last_loss["objective"], canonical_last, config["tracking_steps"]
                    selected_raw_pose = last_pose.clone()
                    selected_projection, selected_raw_loss = last_projection, numbers(loss(last_pose))
                    last_check["selected_as_best"] = True
            else:
                flags.append("last_candidate_nonfinite_camera")
            selected_loss = numbers(loss(best_pose))
            best_c2w, initial_c2w = inverse(best_pose), inverse(initial)
        after_hashes = tensor_manifest(tensors)
        if before_hashes != after_hashes or any(value.requires_grad or value.grad is not None for value in tensors.values()):
            raise RuntimeError("Frozen map/input hashes or gradient state changed during replay")
        torch.cuda.synchronize()
        result = {"status": "completed_candidate_replay_with_failures" if flags else "completed_candidate_replay",
                  "weight": weight, "initial_loss": initial_loss, "pose_gradient_probe": probe,
                  "failure_flags": flags, "candidate_validated_by_probe": probe["status"] == "passed",
                  "steps_requested": config["tracking_steps"], "steps_completed": steps_completed,
                  "best_step_selected_by_tracking_objective_only": best_step,
                  "selected_loss": selected_loss, "selected_raw_candidate_loss_before_projection": selected_raw_loss,
                  "initial_w2c": initial.cpu().tolist(), "prior_reference_w2c": reference.cpu().tolist(),
                  "final_selected_w2c": best_pose.cpu().tolist(), "final_selected_c2w": best_c2w.cpu().tolist(),
                  "selected_raw_w2c_before_projection": selected_raw_pose.cpu().tolist(),
                  "selected_pose_rigid_projection": selected_projection, "last_candidate_check": last_check,
                  "camera_center_correction_m": float(torch.linalg.vector_norm(best_c2w[:3, 3]-initial_c2w[:3, 3])),
                  "fixed_tensor_hashes_before": before_hashes, "fixed_tensor_hashes_after": after_hashes,
                  "elapsed_seconds": time.perf_counter()-start,
                  "scope": "single-frame frozen-map candidate mechanism, no online acceptance/GT accuracy claim"}
        torch.save({"result": result, "final_selected_w2c": best_pose.cpu(),
                    "translation_tangent_last": translation.detach().cpu(), "rotation_tangent_rad_last": rotation.detach().cpu(),
                    "source_context_sha256": CONTEXT_HASH}, folder / "final_pose.pt")
        fields = ("candidate_step", "optimizer_update", "objective", "sensor_objective", "prior_objective",
                  "rgb_mse", "depth_mae_m", "alpha_loss", "coverage", "covered_valid_pixels", "valid_depth_pixels",
                  "best_step", "best_combined_objective", "canonical_best_updated", "gradient_norm_before_clip")
        with (folder / "loss.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: {**row, **row["loss"]}.get(key) for key in fields} for row in rows)
        atomic_json(folder / "result.json", finite_json(result))
        atomic_json(folder / "progress.json", finite_json({**progress, "status": result["status"],
                    "steps_completed": steps_completed, "failure_flags": flags, "result": "result.json"}))
        return result
    except BaseException as error:
        failure = {"status": "failed", "weight": weight, "steps_completed": progress["steps_completed"],
                   "exception_type": type(error).__name__, "exception_message": str(error),
                   "traceback": traceback.format_exc(), "failure_flags": flags,
                   "fixed_tensor_hashes_before": before_hashes,
                   "fixed_tensor_hashes_after_failure": tensor_manifest(tensors),
                   "partial_candidate_is_a_completed_result": False}
        atomic_json(folder / "failure.json", finite_json(failure))
        atomic_json(folder / "progress.json", finite_json({**progress, **failure}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run", required=True, help="New unique output name; never overwrite experiment outputs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Use only ASCII letters, digits, '-' and '_' in run name")
    output = ROOT / "outputs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    configuration = {"run": args.run, "check_only": args.check_only, "source_context_run": CONTEXT_RUN,
                     "source_context_sha256": CONTEXT_HASH, "arm_weights": list(ARM_WEIGHTS),
                     "replay_script_sha256": sha_file(Path(__file__)), "ground_truth_read": False,
                     "dataset_loader_calls": 0, "ICP_calls": 0, "warmup_steps": 0, "mapping_steps": 0,
                     "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output / "configuration.json", configuration)
    results = []
    fixed_tensors = None
    try:
        functions, sources = locked_helpers()
        context, context_metadata, provenance = load_context()
        checks = cpu_checks(context, functions, provenance)
        atomic_json(output / "cpu_checks.json", checks)
        configuration.update({"locked_sources": sources, "source_context": provenance,
                              "tracking_configuration_from_context": context["configuration"]})
        atomic_json(output / "configuration.json", configuration)
        if args.check_only:
            atomic_json(output / "metrics.json", {"status": "CPU_checks_passed", **configuration,
                        "cuda_initialized": False, "arms_executed": 0})
            print(json.dumps({"status": "CPU_checks_passed", "output": str(output),
                              "replay_script_sha256": configuration["replay_script_sha256"],
                              "source_context_sha256": CONTEXT_HASH}, indent=2), flush=True)
            return
        from gsplat import rasterization
        fixed_tensors = {**{"map."+key: value.detach().clone().to("cuda").contiguous()
                            for key, value in context["parameters"].items()},
                         **{key: context[key].detach().clone().to("cuda").contiguous() for key in INPUT_KEYS}}
        if any(value.requires_grad or value.grad is not None for value in fixed_tensors.values()):
            raise RuntimeError("Frozen shared GPU inputs require no gradients")
        functions.update({"rasterization": rasterization, "args": SimpleNamespace(**context["configuration"]),
                          "map_parameters": {key: fixed_tensors["map."+key] for key in MAP_KEYS},
                          "images_gpu": fixed_tensors["image"][None], "depths_gpu": fixed_tensors["depth_m"][None],
                          "masks_gpu": fixed_tensors["valid_depth_mask"][None], "K_gpu": fixed_tensors["K"][None]})
        before = tensor_manifest(fixed_tensors)
        for weight, arm_name in zip(ARM_WEIGHTS, ("prior0", "prior001")):
            results.append(replay_arm(weight, output / arm_name, context, functions, fixed_tensors))
            print(json.dumps({"arm": arm_name, "status": results[-1]["status"],
                              "probe_status": results[-1]["pose_gradient_probe"]["status"],
                              "selected_loss": results[-1]["selected_loss"],
                              "correction_m": results[-1]["camera_center_correction_m"]}), flush=True)
        after = tensor_manifest(fixed_tensors)
        if before != after or any(result["fixed_tensor_hashes_before"] != before for result in results):
            raise RuntimeError("Two arms did not use exactly the same fixed input bytes")
        atomic_json(output / "metrics.json", finite_json({"status": "completed_with_candidate_failures" if
                    any(result["failure_flags"] for result in results) else "completed",
                    **configuration, "arms": results, "fixed_tensor_hashes_before": before,
                    "fixed_tensor_hashes_after": after, "same_fixed_input_bytes_for_both_arms": True,
                    "GPU": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "limitations": ["One current201 frame with the prior0 warmup map, not complete SLAM or a trajectory benchmark",
                                    "No GT is loaded, so correction/objective changes do not measure pose accuracy",
                                    "Finite difference probe retains original epsilon/thresholds; failed probe remains failed even if candidates optimize",
                                    "GPU reductions/rasterization may be nondeterministic despite identical tensor bytes; deterministic mode is not forced",
                                    "Same sensor objective, clipping, Adam and canonical combined-best selection as locked V4; additional read-only diagnostics may expose backend reduction variation",
                                    "No previous-camera acceptance gate is applied; final states are candidates, not accepted online trajectory poses"]}))
    except BaseException as error:
        failure = {"status": "failed", **configuration, "completed_arm_count": len(results),
                   "exception_type": type(error).__name__, "exception_message": str(error),
                   "traceback": traceback.format_exc(), "completed_arms": results,
                   "partial_run_is_complete": False}
        if fixed_tensors is not None:
            failure["fixed_tensor_hashes_after_failure"] = tensor_manifest(fixed_tensors)
        atomic_json(output / "failure.json", finite_json(failure))
        raise


if __name__ == "__main__":
    main()

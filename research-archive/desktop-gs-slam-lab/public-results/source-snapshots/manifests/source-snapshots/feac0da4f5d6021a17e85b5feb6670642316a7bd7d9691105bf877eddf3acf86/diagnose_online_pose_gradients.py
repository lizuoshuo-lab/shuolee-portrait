"""Read-only first-probe finite-difference diagnostic around locked online V3.

Reuses the reference run's sensor/configuration and original warmup pipeline.
Stops at the first pose probe before any tracking optimizer step or GT evaluation.
This diagnostic is not a completed trajectory run and never relaxes probe criteria.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ONLINE_SHA = "45e67fa78e8382d7fdd134341205ee30d20229542dd12b877a01c18ebe0ac8ff"
EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)


class DiagnosticComplete(BaseException):
    """Bypass V3's RuntimeError gradient-probe handler and end imported main."""


class DiagnosticAbort(BaseException):
    """A hook failure must also bypass V3's RuntimeError handler and end main."""


def abort_safe_probe(callback):
    """No callback error or accidental return may resume the training pipeline."""
    def guarded(*args, **kwargs):
        try:
            callback(*args, **kwargs)
        except (DiagnosticComplete, DiagnosticAbort):
            raise
        except BaseException as error:
            raise DiagnosticAbort(f"First-probe diagnostic aborted: {type(error).__name__}: {error}") from error
        raise DiagnosticAbort("First-probe diagnostic callback returned without a terminal stop")
    return guarded


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_comparison(analytic: torch.Tensor, numeric: torch.Tensor) -> dict:
    cosine = float(torch.nn.functional.cosine_similarity(analytic[None], numeric[None]))
    relative = float(torch.linalg.vector_norm(analytic-numeric)/torch.linalg.vector_norm(numeric).clamp_min(1e-12))
    finite = bool(torch.isfinite(analytic).all() and torch.isfinite(numeric).all())
    return {"finite_difference_six_dof": numeric.detach().cpu().tolist(),
            "cosine_similarity": cosine, "relative_l2_error": relative,
            "all_finite": finite,
            "would_meet_unchanged_original_probe_criteria": finite and cosine >= .8 and relative <= .5
                and float(torch.linalg.vector_norm(analytic)) >= 1e-10}


def public_measure(value: dict, base: dict) -> dict:
    result = {key: item for key, item in value.items() if not key.startswith("_")}
    mask, base_mask = value["_covered_mask"], base["_covered_mask"]
    result.update({"covered_mask_sha256": hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
                   "covered_mask_changed_pixels_vs_zero": int((mask ^ base_mask).sum()),
                   "newly_covered_pixels_vs_zero": int((mask & ~base_mask).sum()),
                   "lost_covered_pixels_vs_zero": int((~mask & base_mask).sum()),
                   "alpha_max_abs_change_vs_zero": float((value["_alpha"]-base["_alpha"]).abs().max())})
    return result


def finite_difference_sweep(loss_from_twist, measure, device, dtype, completed_epsilon=None) -> dict:
    """Match the unchanged helper's centered-FD arithmetic and comparison formula."""
    zero = torch.zeros(6, device=device, dtype=dtype)
    probe = zero.clone().requires_grad_(True)
    analytic_loss = loss_from_twist(probe)
    analytic = torch.autograd.grad(analytic_loss, probe)[0].detach()
    if not torch.isfinite(analytic).all():
        raise RuntimeError("Nonfinite analytic gradient in diagnostic")
    with torch.no_grad():
        base = measure(zero, None)
    result = {"twist_order": ["rho_x_m", "rho_y_m", "rho_z_m", "omega_x_rad", "omega_y_rad", "omega_z_rad"],
              "dtype": str(dtype), "device": str(device),
              "analytic_six_dof": analytic.cpu().tolist(),
              "analytic_zero_objective": float(analytic_loss.detach()),
              "analytic_nonzero_components": int((analytic.abs() > 1e-10).sum()),
              "zero_measurement": public_measure(base, base),
              "original_probe_criteria_unchanged": {"minimum_cosine": .8, "maximum_relative_l2": .5,
                                                    "vanished_gradient_norm_threshold": 1e-10},
              "epsilon_sweeps": []}
    with torch.no_grad():
        for epsilon in EPSILONS:
            numeric, fixed_mask_numeric, axes = [], [], []
            for dimension in range(6):
                offset = zero.clone()
                offset[dimension] = epsilon
                plus, minus = measure(offset, base), measure(-offset, base)
                derivative = float((plus["_objective"]-minus["_objective"])/(2*epsilon))
                fixed_mask_derivative = float((plus["_fixed_mask_objective"]-minus["_fixed_mask_objective"])/(2*epsilon))
                numeric.append(derivative)
                fixed_mask_numeric.append(fixed_mask_derivative)
                axes.append({"dimension": dimension, "coordinate": result["twist_order"][dimension],
                             "epsilon": epsilon, "plus": public_measure(plus, base),
                             "minus": public_measure(minus, base),
                             "plus_minus_covered_mask_changed_pixels": int((plus["_covered_mask"] ^ minus["_covered_mask"]).sum()),
                             "original_centered_derivative": derivative,
                             "fixed_zero_coverage_mask_centered_derivative": fixed_mask_derivative})
            sweep = {"epsilon": epsilon,
                     **vector_comparison(analytic, torch.tensor(numeric, device=device, dtype=dtype)),
                     "fixed_zero_coverage_mask_diagnostic": vector_comparison(
                         analytic, torch.tensor(fixed_mask_numeric, device=device, dtype=dtype)),
                     "axes": axes}
            result["epsilon_sweeps"].append(sweep)
            if completed_epsilon is not None:
                completed_epsilon(result)
    return result


def cpu_check() -> dict:
    # A known full-rank six-dimensional quadratic isolates FD bookkeeping from
    # renderer visibility, mask discontinuities, and SE3 parameterization.
    matrix = torch.tensor([[1., .2, 0., 0., .1, 0.], [0., 1., -.2, 0., 0., .1],
                           [.1, 0., 1., .3, 0., 0.], [0., -.1, 0., 1., 0., .2],
                           [0., .1, 0., 0., 1., -.1], [.2, 0., .1, 0., 0., 1.]], dtype=torch.float64)
    target = torch.tensor([.2, -.3, .4, -.2, .1, -.4], dtype=torch.float64)

    def loss(vector):
        return ((matrix @ vector-target)**2).mean()

    def measurement(vector, base):
        objective = loss(vector)
        mask = torch.ones(2, 3, dtype=torch.bool)
        return {"objective": float(objective), "fixed_zero_coverage_mask_objective": float(objective),
                "coverage": 1., "covered_valid_pixels": 6, "valid_depth_pixels": 6,
                "_objective": objective, "_fixed_mask_objective": objective,
                "_covered_mask": mask, "_alpha": torch.ones(2, 3, dtype=torch.float64)}

    sweep = finite_difference_sweep(loss, measurement, "cpu", torch.float64)
    expected = -2*matrix.T @ target/6
    analytic_error = float(np.max(np.abs(np.asarray(sweep["analytic_six_dof"])-expected.numpy())))
    largest_fd_relative = max(row["relative_l2_error"] for row in sweep["epsilon_sweeps"])
    assert analytic_error < 1e-12 and largest_fd_relative < 1e-9
    # Changed-mask diagnostics must count swaps even when total coverage is equal.
    base_mask = torch.tensor([True, False, True, False])
    changed_mask = torch.tensor([False, True, True, False])
    base = {"_covered_mask": base_mask, "_alpha": torch.zeros(4)}
    value = {"_covered_mask": changed_mask, "_alpha": torch.ones(4)}
    mask_measurement = public_measure(value, base)
    assert mask_measurement["covered_mask_changed_pixels_vs_zero"] == 2
    assert mask_measurement["newly_covered_pixels_vs_zero"] == 1
    assert mask_measurement["lost_covered_pixels_vs_zero"] == 1
    assert not issubclass(DiagnosticComplete, RuntimeError)
    assert not issubclass(DiagnosticAbort, RuntimeError)
    runtime_error_handler_entered = optimizer_continuation_reached = False

    def internally_failing_hook():
        raise RuntimeError("Synthetic internal renderer/torch error")

    try:
        # Simulate the unchanged V3 probe RuntimeError handler. A hook error must
        # leave this entire boundary before any optimizer continuation is reached.
        try:
            abort_safe_probe(internally_failing_hook)()
        except RuntimeError:
            runtime_error_handler_entered = True
        optimizer_continuation_reached = True
    except DiagnosticAbort as error:
        assert isinstance(error.__cause__, RuntimeError)
    assert not runtime_error_handler_entered and not optimizer_continuation_reached

    def successful_hook():
        raise DiagnosticComplete()

    try:
        abort_safe_probe(successful_hook)()
    except DiagnosticComplete:
        successful_stop_preserved = True
    else:
        successful_stop_preserved = False
    assert successful_stop_preserved
    try:
        abort_safe_probe(lambda: None)()
    except DiagnosticAbort:
        unexpected_return_aborted = True
    else:
        unexpected_return_aborted = False
    assert unexpected_return_aborted
    return {"status": "passed", "device": "CPU", "epsilons": EPSILONS,
            "quadratic_analytic_max_abs_error": analytic_error,
            "quadratic_max_fd_relative_l2_error": largest_fd_relative,
            "equal_coverage_mask_swap_detected": True,
            "custom_stop_bypasses_online_RuntimeError_handler": True,
            "internal_RuntimeError_converted_to_BaseException_abort": True,
            "online_RuntimeError_handler_entered_on_hook_failure": runtime_error_handler_entered,
            "optimizer_continuation_reached_on_hook_failure": optimizer_continuation_reached,
            "successful_terminal_stop_preserved": successful_stop_preserved,
            "unexpected_hook_return_aborted": unexpected_return_aborted,
            "cuda_initialized": torch.cuda.is_initialized(),
            "scope": "FD arithmetic, six-dimensional gradient and mask-delta bookkeeping; no renderer or GPU execution"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="New diagnostic output name; required for GPU execution")
    parser.add_argument("--reference-run", default="tum-online-v3-icpprefix-refine20-001")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.reference_run):
        parser.error("Invalid reference run name")
    source_path = ROOT / "scripts/train_tum_online_gs.py"
    if sha256(source_path) != EXPECTED_ONLINE_SHA:
        raise RuntimeError("Online V3 source differs from the locked reviewed implementation")
    checks = cpu_check()
    if args.check_only:
        print(json.dumps({"status": "cpu_checks_passed", "cpu_checks": checks,
                          "diagnostic_script_sha256": sha256(Path(__file__)),
                          "online_source_sha256": EXPECTED_ONLINE_SHA}, indent=2))
        return
    if not args.run or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("A unique ASCII --run is required")
    output = ROOT / "outputs" / args.run
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    reference_path = ROOT / "outputs" / args.reference_run / "configuration.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference["script_sha256"] != EXPECTED_ONLINE_SHA:
        raise RuntimeError("Reference run was produced by a different source")
    configuration = reference["configuration"].copy()
    if configuration["tracking_mode"] != "optimize" or configuration["pose_initializer"] != "icp-prefix":
        raise ValueError("This diagnostic requires the configured ICP-prefix refinement reference")
    configuration["run"], configuration["check_only"] = args.run, False
    import train_tum_online_gs as online
    original_probe, original_argv = online.pose_gradient_probe, sys.argv
    hook_calls, started = 0, time.perf_counter()

    def diagnostic_probe_impl(loss_from_twist, device="cuda", dtype=torch.float32, **unused_original_probe_options):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls != 1:
            raise RuntimeError("Diagnostic must stop after the first probe")
        closure = inspect.getclosurevars(loss_from_twist).nonlocals
        data_loss, pose, view = closure["data_loss"], closure["pose"], int(closure["view"])
        if view != 1:
            raise RuntimeError("Expected the first adjacent sensor pair")
        loss_context = inspect.getclosurevars(data_loss).nonlocals
        render = loss_context["render"]
        render_context = inspect.getclosurevars(render).nonlocals
        pose_context = inspect.getclosurevars(pose).nonlocals
        runtime_args = loss_context["args"]
        parameters = render_context["map_parameters"]
        if any(value.requires_grad for value in parameters.values()):
            raise RuntimeError("Diagnostic expects the warmup map to be frozen")
        frozen_parameters = {key: value.detach().cpu().clone() for key, value in parameters.items()}
        saved_checkpoint = output / "last_checkpoint.pt"
        warmup = torch.load(saved_checkpoint, map_location="cpu", weights_only=True)
        warmup_progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        assert warmup["ground_truth_included"] is False and warmup_progress["completed_frame_count"] == 1
        assert len(warmup["estimated_w2c"]) == 1
        assert all(torch.equal(warmup["parameters"][key], value) for key, value in frozen_parameters.items())
        shutil.copyfile(saved_checkpoint, output / "diagnostic_warmup_checkpoint.pt")
        shutil.copyfile(output / "progress.json", output / "diagnostic_warmup_progress.json")
        masks, depths, images = loss_context["masks_gpu"], loss_context["depths_gpu"], loss_context["images_gpu"]
        initial_w2c = pose_context["initial_w2c"].detach().cpu().clone()
        writer_context = inspect.getclosurevars(online._FAILURE_WRITER).nonlocals
        initializer = writer_context["icp_initializer"]
        np.savez(output / "diagnostic_sensor_bundle.npz",
                 image=images[view].detach().cpu().numpy(), depth_m=depths[view].detach().cpu().numpy(),
                 valid_depth_mask=masks[view].cpu().numpy(), K=render_context["K_gpu"].cpu().numpy(),
                 initial_w2c=initial_w2c.numpy(), icp_prefix_c2w_float64=initializer.prefix.copy())
        initial_pose_parameters = {key: pose_context[key].detach().cpu().clone() for key in ("translation", "rotation")}

        def measurement(twist, base):
            w2c = pose(twist)
            values = data_loss(w2c, view)
            _, rendered_depth, alpha = render(w2c)
            valid = masks[view]
            covered = valid & (alpha.detach() > runtime_args.alpha_threshold) & torch.isfinite(rendered_depth) & (rendered_depth.detach() > 0)
            fixed_mask = covered if base is None else base["_covered_mask"]
            fixed_depth = (rendered_depth[fixed_mask]-depths[view][fixed_mask]).abs().mean() if int(fixed_mask.sum()) else rendered_depth.sum()*0.
            fixed_objective = values["rgb_mse"]+runtime_args.depth_weight*fixed_depth+runtime_args.alpha_weight*values["alpha_loss"]
            numeric_values = {key: float(value.detach()) if isinstance(value, torch.Tensor) else value
                              for key, value in values.items()}
            near_threshold = {str(tolerance): int((valid & ((alpha-runtime_args.alpha_threshold).abs() <= tolerance)).sum())
                              for tolerance in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)}
            numeric_values.update({"reconstructed_covered_valid_pixels": int(covered.sum()),
                                   "second_render_coverage_count_matches_data_loss": int(covered.sum()) == values["covered_valid_pixels"],
                                   "alpha_near_threshold_valid_pixels": near_threshold,
                                   "fixed_zero_coverage_mask_depth_mae_m": float(fixed_depth),
                                   "fixed_zero_coverage_mask_objective": float(fixed_objective),
                                   "_objective": values["objective"], "_fixed_mask_objective": fixed_objective,
                                   "_covered_mask": covered, "_alpha": alpha.detach()})
            return numeric_values

        base_metadata = {"status": "diagnostic_running", "online_run_completed": False,
                         "online_source_sha256": EXPECTED_ONLINE_SHA, "diagnostic_script_sha256": sha256(Path(__file__)),
                         "reference_run": args.reference_run, "reference_configuration_sha256": sha256(reference_path),
                         "configuration": configuration, "first_probe_local_index": view,
                         "icp_prefix_source_frame_id": initializer.records[-1]["source_current_frame_id"],
                         "icp_prefix_target_frame_id": initializer.records[-1]["target_previous_frame_id"],
                         "icp_registration": initializer.records[-1], "cpu_checks": checks,
                         "warmup_checkpoint_copy_sha256": sha256(output / "diagnostic_warmup_checkpoint.pt"),
                         "warmup_parameters_match_live_probe_map_exactly": True,
                         "probe_criteria_modified": False, "ground_truth_used": False,
                         "fixed_mask_control_scope": "read-only centered FD with the zero-pose covered mask held constant; no optimizer or candidate selection uses this alternate diagnostic value",
                         "limitations": ["Warmup is recomputed from locked source/config; CUDA may differ from the historical run, so historical first-probe failure is not promised to reproduce bit for bit",
                                         "Hard-mask changes and raster visibility can coexist; a mask-fixed FD discrepancy still includes renderer/visibility and float32 numerical effects",
                                         "No completed trajectory, GT evaluation, loss-threshold relaxation or tuning result is produced"]}
        write_json(output / "diagnostic_configuration.json", base_metadata)

        def partial(sweep):
            write_json(output / "diagnostic_progress.json", {**base_metadata, "sweep": sweep})

        sweep = finite_difference_sweep(loss_from_twist, measurement, device, dtype, partial)
        unchanged = all(torch.equal(value.detach().cpu(), frozen_parameters[key]) for key, value in parameters.items())
        pose_unchanged = all(torch.equal(pose_context[key].detach().cpu(), value) for key, value in initial_pose_parameters.items())
        if not unchanged or not pose_unchanged:
            raise RuntimeError("Read-only diagnostic unexpectedly changed map or pose parameters")
        result = {**base_metadata, "status": "diagnostic_completed", "sweep": sweep,
                  "map_parameters_unchanged": unchanged, "pose_optimizer_parameters_unchanged": pose_unchanged,
                  "first_tracking_optimizer_step_executed": False, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "seconds": time.perf_counter()-started}
        write_json(output / "pose_gradient_diagnostic.json", result)
        write_json(output / "diagnostic_completion.json", {key: result[key] for key in
                   ("status", "online_run_completed", "online_source_sha256", "diagnostic_script_sha256",
                    "map_parameters_unchanged", "pose_optimizer_parameters_unchanged", "first_tracking_optimizer_step_executed", "ground_truth_used", "seconds")})
        raise DiagnosticComplete()

    online.pose_gradient_probe = abort_safe_probe(diagnostic_probe_impl)
    argv = [str(source_path)]
    for key, value in configuration.items():
        if isinstance(value, bool):
            if value:
                argv.append("--"+key.replace("_", "-"))
        elif value is not None:
            argv.extend(["--"+key.replace("_", "-"), str(value)])
    sys.argv = argv
    try:
        online.main()
        raise RuntimeError("Online main reached trajectory completion instead of first-probe stop")
    except DiagnosticComplete:
        assert hook_calls == 1 and not (output / "metrics.json").exists()
        assert not (output / "trajectory_evaluation.json").exists()
        print(json.dumps({"status": "diagnostic_completed", "output": str(output),
                          "online_run_completed": False, "tracking_optimizer_steps_executed": 0,
                          "ground_truth_used": False, "epsilon_count": len(EPSILONS)}, indent=2))
    except BaseException as error:
        if online._FAILURE_WRITER is not None:
            online._FAILURE_WRITER("failed", error)
        if output.exists():
            write_json(output / "diagnostic_failure.json", {"status": "diagnostic_failed", "online_run_completed": False,
                       "exception_type": type(error).__name__, "exception_message": str(error),
                       "original_hook_exception_type": type(error.__cause__).__name__ if error.__cause__ is not None else None,
                       "hook_calls": hook_calls})
        raise
    finally:
        online.pose_gradient_probe, online._FAILURE_WRITER = original_probe, None
        sys.argv = original_argv


if __name__ == "__main__":
    main()

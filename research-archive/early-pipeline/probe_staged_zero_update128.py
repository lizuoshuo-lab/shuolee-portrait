"""Zero-update native reread of the synthetic step8 or step9 checkpoint.

``audit`` is CPU/static only. ``run`` is for the root-owned bounded GPU probe.
No existing source/result is edited, no training step is executed, and strict
forward/state comparisons are reported without changing a tolerance.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import traceback


H = Path(__file__).resolve().parent
ORIGINAL = H / "probe_staged_lifecycle128.py"
HELPER = H / "staged_training_lifecycle128.py"
NONZERO = H / "probe_staged_nonzero_lifecycle128.py"
ORIGINAL_SHA = "0dbab3f629fbd20708195f026fb384f03beecf6d2e7671d225750f12d62f2ed1"
HELPER_SHA = "d809fa52b92c126534e4d605194359d7e641b097ebf840a512cefa550e9ce224"
NONZERO_SHA = "e0cbf3fc59504ddac6973c992e6c7d89f64ada64c73d0f8e1e78393cbd9ec348"
LAB = Path("/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab")
BASE_CONTROL = LAB / "outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/checkpoint-lifecycle128"
BOUNDARY = 8
CONTROL = BASE_CONTROL
OUTPUT = CONTROL / "zero-update-restore8"


def configure(boundary):
    global BOUNDARY, CONTROL, OUTPUT
    assert boundary in (8, 9)
    BOUNDARY = boundary
    CONTROL = BASE_CONTROL if boundary == 8 else BASE_CONTROL / "nonzero-boundary9"
    OUTPUT = CONTROL / f"zero-update-restore{boundary}"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ZeroUpdateComplete(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__("zero_update_native_restore_probe_complete")


def adapted_run_node():
    if BOUNDARY == 8:
        source = ORIGINAL.read_text(encoding="utf-8")
        fixture_adaptation = dict(boundary=8, original_fixture_sha256=ORIGINAL_SHA)
    else:
        wrapper = load_module(NONZERO, "frozen_nonzero_source128")
        source, fixture_adaptation = wrapper.patched_run_source()
    module = ast.parse(source)
    original = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    function = copy.deepcopy(original)
    positions = [i for i, n in enumerate(function.body)
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pipe" for t in n.targets)]
    assert len(positions) == 1
    insertion = ast.parse("_install_zero_update_probe(life, namespace, cams, pipe, render, dest)").body[0]
    function.body.insert(positions[0] + 1, insertion)
    ast.fix_missing_locations(function)
    diff = "".join(difflib.unified_diff(ast.unparse(original).splitlines(True),
                                      ast.unparse(function).splitlines(True),
                                      fromfile="frozen_original_run", tofile="zero_update_in_memory_run"))
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removed = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert len(added) == 1 and not removed
    assert "_install_zero_update_probe" in added[0]
    # The upstream training AST is extracted/executed unchanged by the original fixture.
    compile(ast.Module(body=[function], type_ignores=[]), str(ORIGINAL), "exec")
    return function, diff, fixture_adaptation


def verify_frozen_sources():
    assert sha(ORIGINAL) == ORIGINAL_SHA, "Frozen control fixture changed"
    assert sha(HELPER) == HELPER_SHA, "Frozen lifecycle helper changed"
    if BOUNDARY == 9:
        assert sha(NONZERO) == NONZERO_SHA, "Frozen nonzero-moment fixture changed"


def _inspect_and_render(life, cams, pipe, render, destination, diff):
    import torch
    import staged_training_lifecycle128 as state

    checkpoint = destination / f"chkpnt{BOUNDARY}.pth"
    sidecar_path = Path(str(checkpoint) + ".lifecycle.pt")
    manifest_path = Path(str(checkpoint) + ".lifecycle.json")
    expected_path = destination / f"actual-state-{BOUNDARY:02d}.pt"
    expected_render_path = destination / f"actual-render-{BOUNDARY:02d}.pt"
    assert destination == CONTROL / "staged"
    assert life._pending is None and life.last_resume_receipt["next_iteration"] == BOUNDARY + 1
    expected = torch.load(expected_path, map_location="cpu", weights_only=False)
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    assert sidecar["completed_iteration"] == BOUNDARY
    before_rng = state.capture_rng_state()
    before_capture = state._snapshot_tree(life.model.capture())
    before_offset = life.model.static_offset.detach().cpu().clone()
    before_loader = state._snapshot_tree(life.loader.state_dict())
    before_counters = json.loads(json.dumps(life.counters_getter(), allow_nan=False))
    before_paired = life.paired.generator.get_state().cpu().clone()
    before_calls = int(life.paired.calls)
    old_timestep = int(life.model.timestep)
    checks = dict(
        all14_capture_fields_including_7Adam_exact=state._equal(before_capture, expected["capture"]),
        offset_values_exact=state._equal(before_offset, expected["static_offset"]),
        adapter_counters_exact=state._equal(before_counters, expected["counters"]),
        all_global_RNG_exact_sidecar=state._equal(before_rng, sidecar["rng"]),
        Python_RNG_exact=state._equal(before_rng["python"], expected["python_rng"]),
        NumPy_RNG_exact=state._equal(before_rng["numpy"], expected["numpy_rng"]),
        torch_CPU_RNG_exact=state._equal(before_rng["torch_cpu"], expected["cpu_rng"]),
        torch_CUDA_RNG_exact=(len(before_rng["torch_cuda"]) == 1 and
                            state._equal(before_rng["torch_cuda"][0], expected["cuda_rng"])),
        paired_generator_exact=state._equal(before_paired, expected["paired_rng"]),
        paired_calls_exact=(before_calls == expected["paired_calls"] == BOUNDARY),
        camera_order_cursor_exact=state._equal(before_loader, sidecar["camera_loader"]),
        no_next_camera_draw=(before_loader["sampled_count"] == BOUNDARY),
        post_update_zero_grad=all(p.grad is None for g in life.model.optimizer.param_groups for p in g["params"]),
        shared_offset_actually_nonzero=bool(before_offset.abs().max() > 0),
    )
    groups = []
    for group in life.model.optimizer.param_groups:
        parameter = group["params"][0]
        saved = life.model.optimizer.state.get(parameter, {})
        groups.append(dict(name=group["name"], parameter_elements=parameter.numel(),
            Adam_step=int(saved["step"]) if "step" in saved else None,
            exp_avg_nonzero=int(torch.count_nonzero(saved["exp_avg"])) if "exp_avg" in saved else 0,
            exp_avg_sq_nonzero=int(torch.count_nonzero(saved["exp_avg_sq"])) if "exp_avg_sq" in saved else 0))
    if BOUNDARY == 9:
        checks["nonzero_Gaussian_Adam_moments_actually_covered"] = any(
            g["name"] != "static_offset" and g["exp_avg_nonzero"] > 0 and g["exp_avg_sq_nonzero"] > 0 for g in groups)
    assert all(checks.values()), checks
    expected_render = torch.load(expected_render_path, map_location="cpu", weights_only=True)
    try:
        with torch.no_grad():
            life.model.select_mesh_by_timestep(0)
            actual_render = render(cams[0], life.model, pipe, torch.zeros(3, device="cuda"))["render"].detach().cpu()
    finally:
        with torch.no_grad():
            life.model.select_mesh_by_timestep(old_timestep)
    after_rng = state.capture_rng_state()
    checks.update(
        native_render_shape_dtype_exact=(actual_render.shape == expected_render.shape and actual_render.dtype == expected_render.dtype),
        native_render_finite=bool(torch.isfinite(actual_render).all() and torch.isfinite(expected_render).all()),
        native_render_bitwise_exact=torch.equal(actual_render, expected_render),
        native_forward_did_not_change_model=state._equal(before_capture, life.model.capture()),
        native_forward_did_not_change_offset=state._equal(before_offset, life.model.static_offset),
        native_forward_did_not_change_any_RNG=state._equal(before_rng, after_rng),
        native_forward_did_not_sample_background=(before_calls == life.paired.calls and
                       state._equal(before_paired, life.paired.generator.get_state())),
        native_forward_did_not_advance_camera=state._equal(before_loader, life.loader.state_dict()),
        native_forward_did_not_change_counters=state._equal(before_counters, life.counters_getter()),
    )
    assert actual_render.shape == expected_render.shape
    difference = (actual_render - expected_render).abs()
    result = dict(schema="zero-update-native-restore128-v1", passed=all(checks.values()),
        synthetic_only=True, person_data_read=False, person_model_trained=False,
        original_next_training_step_executed=False, optimizer_updates_in_probe=0,
        next_camera_or_paired_draws_in_probe=0, original_total_clock=18000,
        completed_checkpoint_iteration=BOUNDARY, next_original_iteration=BOUNDARY + 1,
        all_state_exact_before_native_forward=True,
        native_forward_max_abs=float(difference.max()), native_forward_mean_abs=float(difference.mean()),
        strict_bitwise_threshold_unchanged=True, quality_accepted=False,
        SH0_rest_tensor_elements=int(life.model._features_rest.numel()),
        empty_SH0_rest_group_not_a_nonzero_gradient_claim=True,
        static_offset_max_abs=float(before_offset.abs().max()),
        restored_timestep=old_timestep, reviewed_render_timestep=0,
        optimizer_group_names=[g["name"] for g in life.model.optimizer.param_groups],
        optimizer_moment_coverage=groups,
        actual_resume_receipt=life.last_resume_receipt, checks=checks,
        input_hashes={str(p): sha(p) for p in ([ORIGINAL, HELPER, Path(__file__), checkpoint,
                        sidecar_path, manifest_path, expected_path, expected_render_path] + ([NONZERO] if BOUNDARY == 9 else []))},
        in_memory_adapter_diff=diff)
    torch.save(actual_render, OUTPUT / f"restored-native-render-{BOUNDARY:02d}.pt")
    torch.save(dict(capture=before_capture, static_offset=before_offset, RNG=before_rng,
                    paired_generator=before_paired, paired_calls=before_calls,
                    camera_loader=before_loader, counters=before_counters), OUTPUT / "restored-state-before-forward.pt")
    (OUTPUT / "executed-run-function.diff").write_text(diff, encoding="utf-8")
    result["output_hashes"] = {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    write_json(OUTPUT / "result.json", result)
    raise ZeroUpdateComplete(result)


def install_factory(diff):
    def install(life, namespace, cams, pipe, render, destination):
        real_factory = namespace["DataLoader"]

        class StopAfterRestore:
            def __init__(self, real):
                self.real = real

            def __len__(self):
                return len(self.real)

            def __iter__(self):
                # The real helper completes original model.restore validation and all
                # state restoration here; no next(iterator) is called in this probe.
                iter(self.real)
                _inspect_and_render(life, cams, pipe, render, destination, diff)
                raise AssertionError("The zero-update probe must terminate before the next camera")

        def factory(dataset, **kwargs):
            return StopAfterRestore(real_factory(dataset, **kwargs))

        namespace["DataLoader"] = factory
    return install


def audit():
    verify_frozen_sources()
    _, diff, fixture_adaptation = adapted_run_node()
    assert not OUTPUT.exists(), "Probe was already run; do not overwrite its evidence"
    result = dict(passed=True, prepared_only=True, GPU_executed=False,
                  person_data_read=False, training_executed=False,
                  frozen_control_unchanged=True, frozen_helper_unchanged=True,
                  strict_comparison_thresholds_unchanged=True,
                  original_fixture_sha256=ORIGINAL_SHA, helper_sha256=HELPER_SHA,
                  wrapper_sha256=sha(__file__), in_memory_adapter_diff=diff,
                  fixture_adaptation=fixture_adaptation, boundary=BOUNDARY,
                  run_command=f"python -B probe_staged_zero_update128.py run --boundary {BOUNDARY}",
                  output_directory=str(OUTPUT), quality_accepted=False)
    path = H / f"probe_staged_zero_update128_boundary{BOUNDARY}_cpu_audit.json"
    write_json(path, result)
    print(json.dumps(dict(passed=True, audit=str(path), wrapper_sha256=result["wrapper_sha256"])))


def run():
    verify_frozen_sources()
    function, diff, fixture_adaptation = adapted_run_node()
    assert not OUTPUT.exists(), "Never overwrite an earlier zero-update result"
    assert (CONTROL / f"staged/chkpnt{BOUNDARY}.pth.lifecycle.json").is_file()
    OUTPUT.mkdir()
    if BOUNDARY == 8:
        namespace = load_module(ORIGINAL, "frozen_lifecycle_fixture128").__dict__
    else:
        _, namespace, build_receipt = load_module(NONZERO, "frozen_nonzero_fixture128").build_run()
        assert build_receipt == fixture_adaptation
    assert namespace["OUT"] == CONTROL
    namespace["_install_zero_update_probe"] = install_factory(diff)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORIGINAL), "exec"), namespace)
    write_json(OUTPUT / "fixture-adaptation.json", fixture_adaptation)
    try:
        namespace["run"]("resume")
    except ZeroUpdateComplete as completed:
        print(json.dumps(dict(passed=completed.result["passed"],
              native_forward_max_abs=completed.result["native_forward_max_abs"],
              result=str(OUTPUT / "result.json"))), flush=True)
        return 0 if completed.result["passed"] else 1
    except Exception:
        write_json(OUTPUT / "failure.json", dict(error=traceback.format_exc(),
                    optimizer_updates_in_probe=0, quality_accepted=False))
        raise
    finally:
        signal.alarm(0)
    raise AssertionError("The original training loop unexpectedly returned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "run"])
    parser.add_argument("--boundary", type=int, choices=[8, 9], default=8)
    args = parser.parse_args()
    configure(args.boundary)
    audit() if args.mode == "audit" else sys.exit(run())

"""CPU synthetic formula/gate checks plus a non-aggregating input snapshot.

This checker exercises the new collector against constructed controls. Expected
metric answers use closed-form rigid transformations and separate direct NumPy
calculations. It does not establish independent registration accuracy.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

import summarize_v4_full_sequence_controls as collector


def rejected(function, expected_text=None):
    try:
        function()
    except (ValueError, KeyError, FileNotFoundError) as error:
        if expected_text is not None:
            assert expected_text in str(error), (expected_text, str(error))
        return {"rejected": True, "reason": str(error)}
    raise AssertionError("Invalid synthetic input unexpectedly passed")


def time_record(value):
    return {"utc": value.isoformat(), "beijing": value.astimezone(timezone(timedelta(hours=8))).isoformat()}


def synthetic_ledger(protocol):
    start = datetime(2026, 10, 8, 17, 0, tzinfo=timezone.utc)
    rows = []
    for i, item in enumerate(protocol["new_runs"]):
        ready = start + timedelta(minutes=i*3+1)
        rows.append({"order": i+1, "run": item["run"], "configuration": copy.deepcopy(item["configuration"]),
            "returncode": 0, "status": "completed", "frames": item["configuration"]["frames"], "log": item["log_relative_path"],
            "ready_at": time_record(ready), "spawn_guard_at": time_record(ready+timedelta(seconds=1)),
            "Popen_returned_at": time_record(ready+timedelta(seconds=2)), "ended_at": time_record(ready+timedelta(minutes=1)),
            "metrics_sha256": "a"*64, "log_sha256": "b"*64})
    return {"status": collector.TERMINAL, "protocol_sha256": collector.PROTOCOL_SHA,
        "driver_sha256": protocol["driver_sha256"], "source_sha256": copy.deepcopy(protocol["source_sha256"]),
        "deadline_beijing": protocol["deadline_beijing"], "GPU_runs_started": 4, "current_run": None,
        "run_records": rows, "started_at": time_record(start), "ended_at": time_record(start+timedelta(minutes=12))}


def gate_checks(protocol):
    ledger = synthetic_ledger(protocol)
    durations = collector.validate_ledger(protocol, ledger)
    assert len(durations) == 4 and set(durations.values()) == {59.}
    results = {"valid_four_run_fixture": "passed"}
    cases = {}
    x = copy.deepcopy(ledger); x["status"] = "running"; cases["nonterminal"] = x
    x = copy.deepcopy(ledger); x["run_records"].pop(); cases["missing_fourth"] = x
    x = copy.deepcopy(ledger); x["run_records"][1]["returncode"] = 1; cases["nonzero_exit"] = x
    x = copy.deepcopy(ledger); x["run_records"][1]["configuration"]["pose_prior_weight"] = .02; cases["weight_changed"] = x
    x = copy.deepcopy(ledger); x["run_records"][1]["frames"] -= 1; cases["missing_frame"] = x
    x = copy.deepcopy(ledger); x["run_records"][1]["metrics_sha256"] = "unknown"; cases["missing_metric_hash"] = x
    x = copy.deepcopy(ledger); x["run_records"][0],x["run_records"][1] = x["run_records"][1],x["run_records"][0]; cases["wrong_order"] = x
    x = copy.deepcopy(ledger); x["run_records"][1]["ready_at"] = x["run_records"][0]["ready_at"]; cases["serial_overlap"] = x
    x = copy.deepcopy(ledger); x["source_sha256"]["scripts/tum_rgbd.py"] = "0"*64; cases["source_changed"] = x
    x = copy.deepcopy(ledger)
    deadline = datetime.fromisoformat(protocol["deadline_beijing"])
    for key, offset in (("ready_at", -1), ("spawn_guard_at", 0), ("Popen_returned_at", 1), ("ended_at", 60)):
        x["run_records"][0][key] = time_record(deadline+timedelta(seconds=offset))
    cases["start_exactly_at_deadline"] = x
    for name, changed in cases.items():
        results[name] = rejected(lambda changed=changed: collector.validate_ledger(protocol, changed))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); (root/"evidence.json").write_text("actual bytes")
        results["artifact_hash_tamper"] = rejected(lambda: collector.verify_hash(root, "evidence.json", "0"*64, {}), "SHA mismatch")
        results["path_escape"] = rejected(lambda: collector.safe_path(root, "../outside.json"), "escapes")
    results["oversized_JSON"] = rejected(lambda: collector.bounded({"failure_flags": ["X"*collector.MAX_BYTES]}), "exceeds 1 MiB")
    records = [{"frame_id": i, "accepted": i%5 != 0, "failure_flags": ["synthetic_rejected"] if i%5 == 0 else []} for i in range(748)]
    table = collector.column_table(records)
    assert [dict(zip(table["columns"], row)) for row in table["rows"]] == records
    assert len(table["rows"]) == 748 and sum(not r[1] for r in table["rows"]) == 150
    results["complete_column_table_retains_748_rows_and_150_rejections"] = "passed"
    return results


def pose(rotation_vector, translation):
    result = np.eye(4); result[:3,:3] = Rotation.from_rotvec(rotation_vector).as_matrix(); result[:3,3] = translation
    return result


def formula_checks():
    truth = np.array([pose([.03*i,-.04*i,.02*i], [np.sin(i*.4), .1*i, .15*np.cos(i*.7)]) for i in range(8)])
    offset = pose([.23,-.19,.31], [.4,-.25,.17])
    estimated = np.linalg.inv(offset) @ truth
    stats, arrays = collector.trajectory(estimated, truth)
    expected_raw = np.linalg.norm(estimated[:,:3,3]-truth[:,:3,3],axis=1)
    assert abs(stats["raw_anchored_translation_rmse_m"]-np.sqrt(np.sum(expected_raw**2)/8)) < 1e-14
    assert stats["se3_aligned_ate_translation_rmse_m"] < 1e-14
    assert stats["adjacent_rpe_translation_rmse_m"] < 1e-14
    assert stats["adjacent_rpe_rotation_rmse_degrees"] < 1e-12
    assert np.max(np.abs(np.asarray(stats["alignment_estimated_to_truth"])-offset)) < 1e-14
    assert arrays["rpe_translation_m"][0] is None and len(arrays["raw_error_m"]) == 8
    scaled = truth.copy(); scaled[:,:3,3] *= 2
    scale_stats,_ = collector.trajectory(scaled,truth)
    assert scale_stats["se3_aligned_ate_translation_rmse_m"] > .2
    nonrigid = truth.copy(); nonrigid[2,0,0] += .1
    invalid_rotation = rejected(lambda: collector.trajectory(nonrigid,truth), "Nonrigid")
    dg = pose([.4,-.2,.1], [.3,-.1,.2]); de = pose([-.2,.3,.4], [-.2,.15,.1])
    correct = np.linalg.inv(dg) @ de
    inverse = np.linalg.inv(de) @ dg
    wrong_side = de @ np.linalg.inv(dg)
    # E and inverse(E) have invariant translation norm / rotation angle. Treat
    # that negative control as insensitive, never claim it validates direction.
    correct_t = float(np.linalg.norm(correct[:3,3])); inverse_t = float(np.linalg.norm(inverse[:3,3]))
    wrong_t = float(np.linalg.norm(wrong_side[:3,3]))
    assert abs(correct_t-inverse_t) < 1e-14 and abs(correct_t-wrong_t) > .001
    assert abs(collector.angle_degrees(correct[:3,:3])-collector.angle_degrees(inverse[:3,:3])) < 1e-12
    world = pose([.2,.1,-.3],[.4,-.2,.1])
    step = pose([-.1,.4,.2],[.03,-.02,.01])
    correct_prefix = world @ step
    wrong_prefix = world @ np.linalg.inv(step)
    assert np.max(np.abs(correct_prefix-wrong_prefix)) > .1
    quaternion = Rotation.from_rotvec([.43,-.21,.15]).as_quat()
    wrong_quaternion = np.roll(quaternion, 1)
    assert np.max(np.abs(Rotation.from_quat(quaternion).as_matrix()-Rotation.from_quat(wrong_quaternion).as_matrix())) > .1
    return {"known_global_rigid_alignment_matrix_max_abs_error": float(np.max(np.abs(np.asarray(stats["alignment_estimated_to_truth"])-offset))),
        "raw_RMSE_closed_form_absolute_error_m": abs(stats["raw_anchored_translation_rmse_m"]-np.sqrt(np.sum(expected_raw**2)/8)),
        "global_rigid_case_ATE_m": stats["se3_aligned_ate_translation_rmse_m"],
        "global_rigid_case_RPE_translation_m": stats["adjacent_rpe_translation_rmse_m"],
        "scale2_negative_control_ATE_m": scale_stats["se3_aligned_ate_translation_rmse_m"],
        "nonrigid_input": invalid_rotation, "wrong_xyzw_order_matrix_max_abs": float(np.max(np.abs(Rotation.from_quat(quaternion).as_matrix()-Rotation.from_quat(wrong_quaternion).as_matrix()))),
        "wrong_relative_prefix_direction_max_abs": float(np.max(np.abs(correct_prefix-wrong_prefix))),
        "RPE_inverse_whole_residual_is_insensitive": {"correct_translation_norm_m": correct_t, "inverse_norm_m": inverse_t,
            "not_claimed_as_direction_validation": True},
        "RPE_wrong_side_translation_norm_m": wrong_t, "RPE_correct_translation_norm_m": correct_t,
        "all_closed_form_controls_passed": True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="v4-full-sequence-collector-cpu-check-001")
    args=parser.parse_args()
    if not args.run.startswith("v4-full-sequence-collector-cpu-check-") or "/" in args.run or "\\" in args.run:
        parser.error("Unique CPU-check output name required")
    output=collector.ROOT/"outputs"/args.run
    output.mkdir(exist_ok=False)
    try:
        protocol=collector.read_json(collector.ROOT/collector.PROTOCOL)
        checks={"status":"passed_CPU_synthetic_formula_and_gate_controls", "gate_checks":gate_checks(protocol),
            "formula_checks":formula_checks(), "collector_sha256":collector.sha(Path(collector.__file__)),
            "checker_sha256":collector.sha(Path(__file__)), "real_trajectory_aggregation_runs":0, "GPU_calls":0,
            "scope":"Synthetic metric/gate controls and actual readiness only, not independent registration validation"}
        checks["actual_input_preflight"]=collector.check_inputs()
        assert "torch" not in sys.modules and "gsplat" not in sys.modules
        checks["torch_and_gsplat_not_imported"]=True
        collector.exclusive_json(output/"checks.json",checks)
        for source in (Path(collector.__file__),Path(__file__)):
            with (output/source.name).open("xb") as stream:stream.write(source.read_bytes())
        print(json.dumps({"status":checks["status"], "output":str(output),
            "actual_ready":checks["actual_input_preflight"]["ready"],"collector_sha256":checks["collector_sha256"]},indent=2))
    except BaseException as error:
        collector.exclusive_json(output/"failure.json",{"status":"failed_CPU_check", "error":str(error),"GPU_calls":0})
        raise


if __name__=="__main__":
    main()

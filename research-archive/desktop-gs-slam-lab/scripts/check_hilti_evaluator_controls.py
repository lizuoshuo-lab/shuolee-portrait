"""Known geometric and temporal counterexamples for the new Hilti evaluator."""
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

LAB = Path(__file__).resolve().parents[1]
source = LAB / "scripts/evaluate_fast_livo2_hilti.py"
spec = importlib.util.spec_from_file_location("hilti_eval_under_review", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
OUT = LAB / "outputs/fast-livo2-hilti-gt-001/evaluation-controls"
OUT.mkdir(parents=True, exist_ok=True)

n = 12
stamps = 10_000_000_000 + np.arange(n, dtype=np.int64) * 100_000_000
ref = np.repeat(np.eye(4)[None], n, axis=0)
step = np.arange(n)
ref[:, :3, 3] = np.column_stack((np.sin(step * .4), np.cos(step * .3), step ** 2 * .025))
transform = np.eye(4)
transform[:3, :3] = Rotation.from_euler("xyz", [20, -35, 65], degrees=True).as_matrix()
transform[:3, 3] = [7., -2., 4.]
prediction = np.linalg.inv(transform)[None] @ ref
metrics, _, aligned, anchored = m.errors(prediction, ref, stamps, np.arange(n))
checks = {
    "known_rigid_gauge_has_zero_ATE": aligned.max() < 1e-12,
    "first_matched_pose_gauge_removes_known_rigid_transform": anchored.max() < 1e-12,
    "known_global_rigid_transform_preserves_translation_RPE": metrics["common_reference_adjacent_rpe_translation"]["rmse"] < 1e-12,
    "known_global_rigid_transform_preserves_rotation_RPE": metrics["common_reference_adjacent_rpe_rotation"]["rmse"] < 1e-10,
    "SE3_alignment_rotation_det_positive": abs(np.linalg.det(np.array(metrics["SE3_alignment_estimated_to_reference"])[:3, :3]) - 1) < 1e-12,
}
scaled = ref.copy()
scaled[:, :3, 3] *= 1.7
scale_metric, *_ = m.errors(scaled, ref, stamps, np.arange(n))
checks["scale_error_is_not_absorbed"] = scale_metric["ate_position_se3"]["rmse"] > .1 and scale_metric["scale"] == 1.
rotated = ref.copy()
rotated[:, :3, :3] = Rotation.from_euler("z", np.linspace(0, 30, n), degrees=True).as_matrix()
rotation_metric, *_ = m.errors(rotated, ref, stamps, np.arange(n))
checks["known_rotation_increment_uses_degrees"] = abs(rotation_metric["common_reference_adjacent_rpe_rotation"]["rmse"] - 30 / (n - 1)) < 1e-12

two = np.repeat(np.eye(4)[None], 2, axis=0)
two[1, :3, 3] = [1., 2., 4.]
two[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
times = np.array([10_000_000_000, 10_100_000_000], dtype=np.int64)
queries = np.array([times[0] - 1, times[0], times[0] + 25_000_000, times[1], times[1] + 1], dtype=np.int64)
interpolated, valid, rejected = m.interpolate(times, two, queries)
checks["inclusive_actual_endpoints_no_extrapolation"] = valid.tolist() == [False, True, True, True, False] and rejected["outside_estimate_span"] == 2
checks["known_linear_position_quarter"] = np.allclose(interpolated[2, :3, 3], [.25, .5, 1.], atol=1e-14)
checks["known_SLerp_orientation_quarter"] = abs(Rotation.from_matrix(interpolated[2, :3, :3]).magnitude() * 180 / np.pi - 22.5) < 1e-12
_, valid_gap, rejection_gap = m.interpolate(np.array([times[0], times[0] + 500_000_000]), two, np.array([times[0] + 250_000_000]))
checks["large_estimate_gap_is_rejected"] = not valid_gap.any() and rejection_gap["estimate_bracket_gap_exceeded"] == 1
skipped_indices = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
missing_metric, *_ = m.errors(ref[skipped_indices], ref[skipped_indices], stamps[skipped_indices], skipped_indices)
checks["no_RPE_bridge_across_rejected_original_query"] = missing_metric["common_reference_adjacent_rpe_translation"]["count"] == 9 and missing_metric["reference_adjacent_pairs_excluded"] == 1
long_gap_stamps = stamps.copy()
long_gap_stamps[6:] += 500_000_000
long_gap_metric, *_ = m.errors(ref, ref, long_gap_stamps, np.arange(n))
checks["no_RPE_bridge_across_long_original_reference_gap"] = long_gap_metric["common_reference_adjacent_rpe_translation"]["count"] == 10

with tempfile.TemporaryDirectory(prefix="reader-controls-", dir=OUT) as folder:
    for name, text in {
        "duplicates": "10 0 0 0 0 0 0 1\n10 1 0 0 0 0 0 1\n11 0 1 0 0 0 0 1\n",
        "placeholder_orientation": "10 0 0 0 0 0 0 0\n11 1 0 0 0 0 0 0\n12 0 1 0 0 0 0 0\n",
        "nonfinite": "10 nan 0 0 0 0 0 1\n11 1 0 0 0 0 0 1\n12 0 1 0 0 0 0 1\n",
    }.items():
        path = Path(folder) / (name + ".txt")
        path.write_text(text, encoding="utf-8")
        try:
            m.read_poses(path)
        except ValueError:
            checks["rejects_" + name] = True
        else:
            checks["rejects_" + name] = False
    precise = Path(folder) / "precise.txt"
    precise.write_text("1649764528.123456789 0 0 0 0 0 0 1\n1649764528.123456790 1 0 0 0 0 0 1\n1649764528.123456791 0 1 0 0 0 0 1\n", encoding="utf-8")
    exact_times, _ = m.read_poses(precise)
    checks["Decimal_preserves_nanosecond_large_timestamps"] = exact_times.tolist() == [1649764528123456789, 1649764528123456790, 1649764528123456791]

checks = {name: bool(value) for name, value in checks.items()}
result = {"status": "passed" if all(checks.values()) else "failed", "check_count": len(checks), "checks": checks,
          "evaluator_source_sha256": m.sha(source), "controls_source_sha256": m.sha(Path(__file__)),
          "scope": "Known geometry, units, exact decimal timestamps, interpolation and explicit rejection examples. No real replay, GT score, source mutation, GPU or ROS node."}
(OUT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result))
raise SystemExit(0 if all(checks.values()) else 1)

"""Reconcile the current accepted ablation snapshot without replacing older work."""
import datetime
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
summary = json.loads((LAB / "outputs/splatam-ablation-repeat-001/metrics.json").read_text(encoding="utf-8"))
path = LAB / "manifests/progress.json"
progress = json.loads(path.read_text(encoding="utf-8"))
entry = next(r for r in progress["results"] if r["id"] == "splatam-ablation-repeat-001")
entry["status"] = summary["status"]
entry["metrics"] = {
    "new_full_runs_independently_verified": len(summary["verified_runs"]) - 1,
    "frozen_full_baseline_reused": 1,
    "preregistered_new_full_runs": 9,
    "jobs": [{k: job.get(k) for k in ("index", "sequence", "tracking_loss", "seed", "status", "rejection_reason")} for job in summary["jobs"]],
    "verified_full_runs": [{k: r[k] for k in ("sequence", "profile", "seed", "frames", "ate_se3_rmse_m", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg", "gaussian_count")} for r in summary["verified_runs"]],
    "three_seed_statistics": summary["three_seed_statistics"],
}
complete = summary["status"] == "full_ablation_repeats_and_room_independently_verified"
entry["scope"] = "只改变 tracking 的 RGB 或深度残差权重，保留 RGB-D 初始化和相同 mapping 配置；不同估计轨迹仍产生各自地图。只有实际完整运行与独立封存才纳入统计。N=3 是探索性重复；room增加同一办公室的覆盖与闭环输入，不证明跨环境泛化或算法回环纠偏。"
progress["updated"] = datetime.datetime.now().date().isoformat()
progress["current_stage"] = ("SplaTAM_tracking_ablation_three_seeds_and_room_independently_verified" if complete
                             else "SplaTAM_tracking_ablation_repeats_and_room_in_progress")
progress["next_steps"] = (["SplaTAM 完整验收后进行 FAST-LIVO2 Hilti exp14 官方 IMU 参考两模式回放与对照"] if complete
                           else ["按预登记队列完成剩余完整训练并独立验收", "各配置三个种子全部通过后统计实际分布与 room 轨迹", "SplaTAM 队列后进行 FAST-LIVO2 Hilti exp14 官方 IMU 参考两模式对照"])
temp = path.with_suffix(".json.tmp")
temp.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
temp.replace(path)
print(json.dumps({"results_preserved": len(progress["results"]), "stage": progress["current_stage"],
                  "verified_new_full_runs": entry["metrics"]["new_full_runs_independently_verified"]}))

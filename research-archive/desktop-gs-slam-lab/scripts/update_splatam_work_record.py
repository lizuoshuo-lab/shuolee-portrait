"""Expose verified SplaTAM evidence in the existing local knowledge library."""
from pathlib import Path
import datetime as dt
import json

LAB = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((LAB / path).read_text(encoding="utf-8-sig"))


def main():
    progress = read("manifests/progress.json")
    acceptance = read("outputs/splatam-linux-bootstrap-001/metrics.json")
    if acceptance["status"] != "isolated_official_renderer_cuda_acceptance_passed":
        raise SystemExit("Original GPU environment acceptance is incomplete")
    smoke = read("outputs/splatam-independent-eval-smoke-001/metrics.json")
    if smoke["status"] != "independently_verified" or not all(smoke["checks"].values()):
        raise SystemExit("Smoke evidence has not passed independent evaluation")
    items = [{
        "id": "splatam-official-local-smoke-001",
        "title": "官方 SplaTAM 本机 GPU 与真实 20 帧验收",
        "status": "official_renderer_actual_smoke_independently_verified",
        "metrics": {"frames": smoke["frames"], "gaussians": smoke["gaussian_count"],
                    "ate_se3_rmse_m": smoke["ate_se3_translation"]["rmse"],
                    "rpe_1frame_translation_rmse_m": smoke["rpe_1frame_translation"]["rmse"],
                    "rpe_1frame_rotation_rmse_deg": smoke["rpe_1frame_rotation"]["rmse"],
                    "actual_child_wall_seconds": smoke["child_wall_seconds"],
                    "independent_checks": smoke["check_count"]},
        "artifacts": ["experiments/28_splatam_official_local_project.md", "knowledge/24_splatam_official_rgbd_system.md",
                      "outputs/splatam-linux-bootstrap-001/metrics.json", "outputs/splatam-linux-bootstrap-001/report.md",
                      "outputs/splatam-linux-bootstrap-001/renderer-smoke.json", "outputs/splatam-independent-eval-smoke-001/metrics.json",
                      "outputs/splatam-independent-eval-smoke-001/report.md", "outputs/splatam-independent-eval-smoke-001/trajectory-errors.png",
                      "configs/splatam/LOCAL_PROJECT.md", "scripts/splatam_local/run_wsl.py", "scripts/splatam_local/run_official.py"],
        "scope": "Real original-renderer GPU run; 20/592 official associated frames and 0.636 seconds of input. SE3 fixed scale1. Functional smoke only, not full-sequence robustness or paper-resolution accuracy.",
    }]
    full_path = LAB / "outputs/splatam-project-001/metrics.json"
    if full_path.is_file():
        project = read("outputs/splatam-project-001/metrics.json")
        if project["status"] != "official_smoke_and_full_sequence_independently_verified":
            raise SystemExit("Full project summary has not passed actual independent review")
        full = project["runs"]["full"]
        items.insert(0, {
            "id": "splatam-official-local-full-001", "title": "官方 SplaTAM 完整 desk 序列与独立评测",
            "status": project["status"],
            "metrics": {"frames": full["frames"], "gaussians": full["gaussian_count"],
                        "ate_se3_rmse_m": full["ate_se3_rmse_m"],
                        "rpe_1frame_translation_rmse_m": full["rpe_1frame_translation_rmse_m"],
                        "rpe_1frame_rotation_rmse_deg": full["rpe_1frame_rotation_rmse_deg"],
                        "actual_child_wall_seconds": full["actual_child_wall_seconds"],
                        "rendered_training_views": full["upstream_rendering"]["sampled_frame_count"]},
            "artifacts": ["experiments/28_splatam_official_local_project.md", "outputs/splatam-project-001/metrics.json",
                          "outputs/splatam-project-001/report.md", "outputs/splatam-project-001/rendered-view-comparison.png",
                          "outputs/splatam-project-001/lowest-psnr-view-comparison.png",
                          "outputs/splatam-independent-eval-full-001/metrics.json", "outputs/splatam-independent-eval-full-001/report.md",
                          "outputs/splatam-independent-eval-full-001/trajectory-errors.png", "outputs/splatam-project-independent-review-001/report.md",
                          "manifests/splatam-local-project.json", "scripts/splatam_local/evaluate_run.py", "scripts/summarize_splatam_project.py"],
            "scope": "All official associated desk frames, one seed, 320x240. Independent C2W SE3 scale1 ATE/RPE. Final-map image metrics cover 30 seen training views; depth MAE, no held-out evaluation. No loop closure, physical recalibration, hardware sync or LiDAR/IMU fusion established.",
        })
        progress["current_stage"] = "SplaTAM_official_local_full_sequence_independently_verified"
        progress["next_steps"] = ["另一个公开序列验证适用范围", "同协议下 640x480 分辨率对照", "颜色/深度或关键帧窗口消融与漂移分析"]
    else:
        progress["current_stage"] = "SplaTAM_official_local_smoke_verified_full_sequence_pending"
        progress["next_steps"] = ["完成官方 desk 592 帧本机运行", "独立轨迹与真实输出审核", "把完整结果和复现入口同步到知识网站"]
    owned_ids = {"splatam-official-local-smoke-001", "splatam-official-local-full-001"}
    progress["results"] = items + [result for result in progress["results"] if result["id"] not in owned_ids]
    for result in progress["results"]:
        if result["id"] == "splatam-source-and-data-001":
            result["status"] = "official_source_data_and_evaluator_controls_verified"
    progress["updated"] = dt.datetime.now().date().isoformat()
    (LAB / "manifests/progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"current_stage": progress["current_stage"], "results": len(progress["results"])}))


if __name__ == "__main__":
    main()

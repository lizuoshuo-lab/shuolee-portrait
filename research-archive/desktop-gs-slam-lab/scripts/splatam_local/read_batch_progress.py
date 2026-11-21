"""Read compact live evidence without waiting for or changing GPU processes."""
import json
from pathlib import Path
import re

LAB = Path(__file__).resolve().parents[2]
NATIVE = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\splatam\outputs\runs")
record = json.loads((LAB / "outputs/splatam-ablation-repeat-001/batch-progress.json").read_text(encoding="utf-8"))
jobs = record["jobs"]
result = {
    "batch_status": record["status"], "jobs_launched": len(jobs),
    "jobs_training_finished": sum(j["status"] == "completed_pending_independent_evaluation" for j in jobs),
    "jobs_failed": sum(j["status"].startswith("failed") for j in jobs),
}
if jobs:
    active = jobs[-1]
    result["latest_job"] = {k: active.get(k) for k in ("index", "sequence", "tracking_loss", "seed", "status", "run_name", "native_status")}
    result["elapsed_wrapper_seconds"] = round(active.get("elapsed_wrapper_seconds", active.get("wrapper_wall_seconds", 0)))
    if active.get("run_name"):
        path = NATIVE / active["run_name"] / "stdout.log"
        if path.is_file():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                tail = stream.read().decode("utf-8", errors="replace")
            steps = re.findall(r"(Tracking|Mapping) Time Step:\s*(\d+)", tail)
            result["latest_logged_step"] = {"phase": steps[-1][0], "frame_index": int(steps[-1][1])} if steps else None
            # Pose Error is a normal trajectory diagnostic, not a Python failure.
            result["stdout_tail_error_lines"] = [line.strip()[:400] for line in tail.splitlines()
                if "Traceback (most recent call last)" in line
                or re.search(r"\b(?:RuntimeError|ValueError|AssertionError|ImportError|OSError|CUDAError):", line)
                or "out of memory" in line.lower()][-3:]
print(json.dumps(result, ensure_ascii=True))

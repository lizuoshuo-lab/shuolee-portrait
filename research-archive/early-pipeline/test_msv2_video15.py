"""Local Windows SfM to isolated Linux CUDA pipeline, without reusing fitted state."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

RECORD = Path("D:/00/记录/2026-11-05_MSV2_video15")
PREP = Path("D:/GS-SLAM-Data/msv2-video15-20261105")
RUN = Path(r"\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\msv2-video15-20261105\runs\video15")
STATUS = RECORD / "测试状态.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(state):
    state["updated"] = now()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main():
    RECORD.mkdir(parents=True, exist_ok=True)
    assert not RUN.exists(), "Fresh run name already exists"
    state = dict(state="running", phase="video_and_colmap", created=now(), pid=os.getpid(),
        input="D:/00/工作/数据集/video15.mp4", source_sha256="965befa346fdb0b5e5140745ab5651ef6ed7f8a87ae32777e1d88017bc45e547",
        prep=str(PREP), run=str(RUN), scales=[1,2,4], iterations=18000, sample_fps=3,
        fresh_gaussians=True, fresh_optimizer=True, mask_used=False, crop_used=False,
        training_completed=False, quality_accepted=False,
        local_execution="Windows FFmpeg/COLMAP preparation; public MSV2 CLI COLMAP import and isolated CUDA training in WSL",
        root="/opt/gs-slam-lab/msv2-video15-20261105", public_commit="79fb532b7e4dbf936683083b30c798ebce783efe")
    save(state)
    try:
        while True:
            if (PREP / "state.json").exists():
                prepared = read(PREP / "state.json")
                if prepared["state"] == "failed":
                    raise RuntimeError("Video preparation failed: " + str(prepared.get("error")))
                if prepared["state"] == "completed":
                    break
            time.sleep(5)
        assert prepared["frames_registered"] == prepared["frames_extracted"]
        assert prepared["video_sha256"] == state["source_sha256"]
        state.update(phase="training_pipeline", frames=prepared["frames_registered"], sparse_points=prepared["sparse_points"])
        save(state)
        for name in ("state.json", "pixel-lineage.json", "camera-inputs.json"):
            shutil.copy2(PREP / name, RECORD / ("视频准备_" + name))
        logs = RECORD / "logs"
        logs.mkdir(exist_ok=True)
        command = ["wsl.exe", "-d", "Ubuntu-20.04", "--", "bash",
            "/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/run_msv2_video15.sh",
            "train", "/mnt/d/GS-SLAM-Data/msv2-video15-20261105/dataset", "--name", "video15"]
        with (logs / "训练入口.log").open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state["launcher_pid"] = process.pid
            save(state)
            exit_code = process.wait()
        pipeline_state = read(RUN / "state.json")
        state.update(pipeline_state=pipeline_state, pipeline_exit_code=exit_code, phase="copying_results")
        save(state)
        if (RUN / "export").exists():
            shutil.copytree(RUN / "export", RECORD, dirs_exist_ok=True)
            if (RECORD / "comparison.html").exists():
                shutil.copy2(RECORD / "comparison.html", RECORD / "查看完整图倍率对照.html")
        for filename in ("state.json", "plan.json", "job.json"):
            if (RUN / filename).exists():
                shutil.copy2(RUN / filename, RECORD / ("训练_" + filename))
        state.update(state=pipeline_state["state"], phase="awaiting_visual_review", finished=now(),
            training_completed=pipeline_state["training_completed"], quality_accepted=False)
        save(state)
        print("VIDEO15_PIPELINE_FINISHED", json.dumps(state, ensure_ascii=False), flush=True)
        return exit_code
    except BaseException as error:
        state.update(state="failed", error=repr(error), finished=now())
        save(state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

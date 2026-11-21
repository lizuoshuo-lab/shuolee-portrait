"""Unexecuted SplaTAM TUM smoke configs, derived from the pinned upstream config."""
import copy
import os
from pathlib import Path
import runpy


def make_config(frame_count):
    lab = Path(os.environ.get("GS_SLAM_LAB", Path(__file__).resolve().parents[2])).resolve()
    repo = lab / "repos" / "SplaTAM"
    config = copy.deepcopy(runpy.run_path(str(repo / "configs/tum/splatam.py"))["config"])
    config.update(
        workdir=str(lab / "outputs/splatam"),
        run_name=f"fr1_desk_{frame_count}f_320x240_seed0",
        use_wandb=False,
        eval_every=5,
        report_global_progress_every=1000,
        save_checkpoints=True,
        checkpoint_interval=20 if frame_count == 20 else 100,
    )
    config["data"].update(
        basedir=os.environ.get("SPLATAM_TUM_ROOT", str(lab / "data/tum")),
        gradslam_data_cfg=str(repo / "configs/data/TUM/freiburg1_desk.yaml"),
        sequence="rgbd_dataset_freiburg1_desk",
        desired_image_height=240,
        desired_image_width=320,
        start=0,
        end=frame_count,
        stride=1,
        num_frames=-1,
    )
    # This remains pose-estimating RGB-D SLAM. GT is used by the benchmark
    # loader for relative frame convention and evaluation, not per-frame tracking.
    config["tracking"]["use_gt_poses"] = False
    config["tracking"]["visualize_tracking_loss"] = False
    config["viz"]["enter_interactive_post_online"] = False
    config["wandb"]["save_qual"] = False
    config["wandb"]["eval_save_qual"] = False
    return config

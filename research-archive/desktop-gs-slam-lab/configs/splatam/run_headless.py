"""Run the pinned upstream SplaTAM CLI without changing upstream source.

This is not the locally validated gsplat experiment. Requires a separately
installed SplaTAM-compatible environment and its exact custom rasterizer.
"""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

PIN = "da6bbcd24c248dc884ac7f49d62e91b841b26ccc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", choices=(20, 200), type=int)
    parser.add_argument("--check-only", action="store_true", help="Validate config/data paths without loading CUDA.")
    args = parser.parse_args()
    lab = Path(os.environ.get("GS_SLAM_LAB", Path(__file__).resolve().parents[2])).resolve()
    repo = lab / "repos/SplaTAM"
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise SystemExit(f"SplaTAM commit mismatch: expected {PIN}, found {actual}")
    config_path = lab / f"configs/splatam/tum_desk_{args.frames}.py"
    config = runpy.run_path(str(config_path))["config"]
    data = Path(config["data"]["basedir"]) / config["data"]["sequence"]
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "rgb", "depth"):
        if not (data / name).exists():
            raise SystemExit(f"Missing TUM data: {data / name}")
    assert config["tracking"]["use_gt_poses"] is False
    assert config["use_wandb"] is False
    print(f"Pinned upstream: {PIN}\nConfig: {config_path}\nDataset: {data}")
    print("320x240; matched-frame prefix; upstream tracking=200, mapping=30 iterations.")
    if args.check_only:
        print("Configuration/path checks passed; no CUDA pipeline was executed.")
        return
    output = Path(config["workdir"]) / config["run_name"]
    if output.exists():
        raise SystemExit(f"Output already exists; preserve it and choose a new run_name: {output}")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.chdir(repo)
    sys.argv = [str(repo / "scripts/splatam.py"), str(config_path)]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

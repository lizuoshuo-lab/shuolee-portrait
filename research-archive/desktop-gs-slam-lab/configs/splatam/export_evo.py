"""Export pinned SplaTAM params.npz to standard c2w TUM trajectories.

No CUDA dependency; numpy and scipy only. Reconstructs timestamps using the
same association and 32 Hz filter as the pinned upstream TUM loader. Ground
truth comes from saved gt_w2c_all_frames, not an independently resampled file.
"""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from scipy.spatial.transform import Rotation


def matched_rgb_timestamps(sequence):
    rgb = np.loadtxt(sequence / "rgb.txt", dtype=str, comments="#", ndmin=2)
    depth = np.loadtxt(sequence / "depth.txt", dtype=str, comments="#", ndmin=2)
    pose = np.loadtxt(sequence / "groundtruth.txt", dtype=str, comments="#", skiprows=1, ndmin=2)
    tr, td, tp = rgb[:, 0].astype(float), depth[:, 0].astype(float), pose[:, 0].astype(float)
    associated = []
    for stamp in tr:
        if np.min(np.abs(td - stamp)) < 0.08 and np.min(np.abs(tp - stamp)) < 0.08:
            associated.append(stamp)
    if not associated:
        raise ValueError("No RGB/depth/GT associations found")
    retained = [associated[0]]
    for stamp in associated[1:]:
        if stamp - retained[-1] > 1.0 / 32:
            retained.append(stamp)
    return np.asarray(retained)


def camera_matrices(params):
    # Upstream stores quaternions as (1,4,N), scalar first, and t as (1,3,N).
    q_wxyz = np.asarray(params["cam_unnorm_rots"])[0].T
    translation = np.asarray(params["cam_trans"])[0].T
    if not np.all(np.isfinite(q_wxyz)) or np.any(np.linalg.norm(q_wxyz, axis=1) < 1e-12):
        raise ValueError("Invalid camera quaternion")
    w2c = np.repeat(np.eye(4)[None], len(q_wxyz), axis=0)
    w2c[:, :3, :3] = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    w2c[:, :3, 3] = translation
    gt_w2c = np.asarray(params["gt_w2c_all_frames"])
    if gt_w2c.shape != w2c.shape:
        raise ValueError(f"GT/estimated frame mismatch: {gt_w2c.shape} vs {w2c.shape}")
    if not np.all(np.isfinite(w2c)) or not np.all(np.isfinite(gt_w2c)):
        raise ValueError("Non-finite trajectory; inspect tracking before evaluation")
    return np.linalg.inv(w2c), np.linalg.inv(gt_w2c)


def tum_rows(stamps, c2w):
    return np.column_stack((stamps, c2w[:, :3, 3], Rotation.from_matrix(c2w[:, :3, :3]).as_quat()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--params", type=Path)
    args = parser.parse_args()
    config = runpy.run_path(str(args.config.resolve()))["config"]
    run_dir = Path(config["workdir"]) / config["run_name"]
    params_path = args.params or run_dir / "params.npz"
    data = config["data"]
    sequence = Path(data["basedir"]) / data["sequence"]
    stamps = matched_rgb_timestamps(sequence)
    end = None if data["end"] == -1 else data["end"]
    stamps = stamps[data["start"]:end:data["stride"]]
    if data["num_frames"] != -1:
        stamps = stamps[:data["num_frames"]]
    with np.load(params_path, allow_pickle=False) as params:
        estimated, reference = camera_matrices(params)
    if len(stamps) != len(estimated):
        raise ValueError(f"Timestamp/pose count mismatch: {len(stamps)} vs {len(estimated)}")
    target = params_path.parent / "evo"
    target.mkdir(exist_ok=True)
    for name, trajectory in (("estimated.tum", estimated), ("reference.tum", reference)):
        np.savetxt(target / name, tum_rows(stamps, trajectory), fmt="%.9f", header="timestamp tx ty tz qx qy qz qw (camera-to-world)")
    metadata = {
        "frames": len(stamps), "frame": "c2w relative to first input camera", "translation_unit": "m",
        "source_params": str(params_path.resolve()), "tracking_uses_gt_poses": config["tracking"]["use_gt_poses"],
        "association": "upstream nearest RGB-depth-GT <0.08s then >1/32s RGB interval",
        "metric_warning": "upstream evaluate_ate returns mean translation error, despite RMSE label",
        "ape_command": "evo_ape tum reference.tum estimated.tum -a --save_results ape_se3.zip",
        "rpe_command": "evo_rpe tum reference.tum estimated.tum -r trans_part --delta 1 --delta_unit f --save_results rpe_1frame.zip",
        "scale_alignment": False,
    }
    (target / "export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {len(stamps)} poses into {target}; run the recorded evo commands there.")


if __name__ == "__main__":
    main()

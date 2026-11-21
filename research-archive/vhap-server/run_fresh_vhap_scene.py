"""Train a new VHAP and MSv2 scene from one video in an empty run directory."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from registered_portrait_pipeline import run
from run_video import PROJECT, input_hash, write_json


def read(path):
    return json.loads(path.read_bytes())


def build_split(initialization, selection):
    """Use registered angular coverage to reserve separated Gaussian RGB views."""
    records = read(initialization)["frames"]
    choice = read(selection)
    reference = max(choice["train"], key=lambda name: (
        choice["decisions"][name]["quality_score"] - abs(choice["decisions"][name]["yaw_deg"]) / 120., name))
    held = []
    for target in (-70., 0., 70., 145.):
        candidates = [row for row in records if row["name"] != reference
                      and all(abs(row["timestamp_s"] - other["timestamp_s"]) >= .55 for other in held)]
        if not candidates:
            raise ValueError("Insufficient separated frames for a held RGB split")
        held.append(min(candidates, key=lambda row: (abs((row["baseline_yaw_deg"] - target + 180.) % 360. - 180.),
                                                     row["timestamp_s"])))
    return {"reference": reference, "held_frames": [row["name"] for row in held],
            "held_yaw_deg": [row["baseline_yaw_deg"] for row in held],
            "split_scope": "Gaussian RGB and shared-neck observations. VHAP uses all registered frames."}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--sample-fps", type=float, default=3.)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--split", type=Path, help="Optional frozen evaluation split matching this source and frame order.")
    parser.add_argument("--torso-neck-grid", action="store_true",
                        help="Fit the VHAP neck base in tracked garment coordinates and finish with cross-camera grids.")
    parser.add_argument("--initial-neck-posture", choices=("template-neutral","original"), default="template-neutral",
                        help="Initial FLAME neck inclination relative to the observed upper torso.")
    args = parser.parse_args()
    args.video = args.video.resolve(strict=True)
    if args.video.suffix.lower() != ".mp4" or not args.video.is_file() or args.steps < 1:
        parser.error("A readable MP4 and positive training budget are required")
    root = args.output.resolve()
    config = {"video": str(args.video), "video_sha256": input_hash(args.video),
              "width": args.width, "sample_fps": args.sample_fps, "gaussian_steps": args.steps,
              "split_sha256": input_hash(args.split) if args.split else None,
              "previous_run_inputs": [], "workflow": "registered-portrait",
              "torso_neck_grid": args.torso_neck_grid,
              "initial_neck_posture": args.initial_neck_posture,
              "source_sha256": {p.name: input_hash(p) for p in sorted(PROJECT.glob("*.py"))}}
    root.mkdir(parents=True, exist_ok=False)
    state = {"configuration": config, "started_utc": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), "status": "running", "stages": {}}
    write_json(root/"run.json", state)
    try:
        run(args, root, state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root/"run.json", state)


if __name__ == "__main__":
    main()

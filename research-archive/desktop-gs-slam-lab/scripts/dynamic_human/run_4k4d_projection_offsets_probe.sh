#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
cd "$LAB/repos/4K4D"
python "$LAB/scripts/dynamic_human/probe_4k4d_projection_offsets.py" -c "$LAB/outputs/dynamic-human-4k4d-official-035/pilot.yaml" > "$LAB/outputs/dynamic-human-enerf-static-calibration-037/projection-adapter.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
cd "$LAB/repos/4K4D"
python "$LAB/scripts/dynamic_human/probe_4k4d_projection_cpu.py" -c "$LAB/outputs/dynamic-human-4k4d-official-035/pilot.yaml" > "$LAB/outputs/dynamic-human-4k4d-official-035/projection-cpu-diagnostic.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
OUT="$LAB/outputs/dynamic-human-4k4d-official-035/continuation-10000"
cd "$LAB/repos/4K4D"
python "$LAB/scripts/dynamic_human/prepare_4k4d_continuation.py"
python "$LAB/scripts/dynamic_human/train_4k4d_continuation.py" -c "$OUT/pilot.yaml" > "$OUT/training.log" 2>&1

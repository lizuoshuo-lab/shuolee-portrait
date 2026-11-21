#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
export FOURKD_RUN_DIR="$LAB/outputs/dynamic-human-4k4d-official-035/continuation-10000"
cd "$LAB/repos/4K4D"
python "$LAB/scripts/dynamic_human/audit_4k4d_continuation_state.py"
python "$LAB/scripts/dynamic_human/evaluate_4k4d_pilot.py" -c "$FOURKD_RUN_DIR/pilot.yaml" > "$FOURKD_RUN_DIR/native-evaluation.log" 2>&1

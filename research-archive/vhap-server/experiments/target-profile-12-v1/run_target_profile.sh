#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
    echo "Usage: bash run_target_profile.sh VIDEO.mp4 NEW_OUTPUT SPLIT.json" >&2
    exit 2
fi
project=$(cd "$(dirname "$0")" && pwd)
run=$(realpath -m "$2")
python3 -Bu "$project/run_fresh_vhap_scene.py" "$1" --output "$run" --width 960 \
    --sample-fps 3 --split "$3" --initial-neck-posture template-neutral --monocular-neck-only
cd "$project/upstream/VHAP-6e1e770"
export PYTHONPATH="$PWD:$project" PYTHONNOUSERSITE=1
"$HOME/venvs/vhap/bin/python" -Bu "$project/fit_target_profile.py" "$run" 2>&1 | tee "$run/logs/target-profile.log"

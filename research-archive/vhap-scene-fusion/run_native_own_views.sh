#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: bash run_native_own_views.sh VIDEO.mp4 NEW_OUTPUT SPLIT.json FRAME [FRAME ...]" >&2
    exit 2
fi
project=$(cd "$(dirname "$0")" && pwd)
video=$(realpath "$1")
run=$(realpath -m "$2")
split=$(realpath "$3")
shift 3
python3 -Bu "$project/run_fresh_vhap_scene.py" "$video" --output "$run" --width 960 \
    --sample-fps 3 --split "$split" --initial-neck-posture template-neutral --native-only
cd "$project/upstream/VHAP-6e1e770"
PYTHONPATH="$PWD:$project" PYTHONNOUSERSITE=1 "$HOME/venvs/vhap/bin/python" -Bu \
    "$project/review_native_own_views.py" "$run" --output "$run/own-view-grid" \
    --flame "$PWD/asset/flame" --frames "$@"

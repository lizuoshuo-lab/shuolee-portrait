#!/usr/bin/env bash
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=fusion
if [[ ${1:-} == --train ]]; then
    mode=joint
    shift
fi
out=${1:-"$project/runs/demo2-$mode-$(date -u +%Y%m%dT%H%M%SZ)"}
if (( $# > 0 )); then shift; fi
scene="$HOME/data/demo2_scene"
tracker="$HOME/VHAP/output/monocular/demo2_s30crop17_cam/2026-11-16_17-47-37/tracked_flame_params_30.npz"
landmarks="$project/inputs/demo2/scene-landmarks.npz"

mkdir -p "$out" "$(dirname -- "$landmarks")"
out=$(realpath -- "$out")
if [[ ! -s "$landmarks" ]]; then
    cd "$HOME/VHAP"
    PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" "$HOME/venvs/vhap/bin/python" -B "$project/detect_scene_landmarks.py" \
        --scene "$scene" --selected "$HOME/VHAP/data/monocular/demo2_s30c17/images" \
        --tracker "$tracker" --output "$landmarks"
fi

cd "$HOME/GaussianAvatars"
entry=fuse_scene.py
options=(--color-steps 156)
if [[ $mode == joint ]]; then
    entry=train_joint.py
    options=(--holdout 00428.jpg 00439.jpg 00549.jpg 00560.jpg)
fi
exec env PYTHONPATH="$HOME/ga_pkgs:$PWD" "$HOME/miniconda3/envs/fastavatar/bin/python" -u -B "$project/$entry" \
    --scene "$scene" --background "$scene/scene_out/bg_body2" \
    --bound "$HOME/output/ga_demo2_v8_600k/point_cloud/iteration_600000/point_cloud.ply" \
    --tracker "$tracker" --landmarks "$landmarks" --output "$out" --export-frame 00494.jpg \
    --flame-boundary-mask "$PWD/flame_model/assets/flame/FLAME_masks.pkl" "${options[@]}" "$@"

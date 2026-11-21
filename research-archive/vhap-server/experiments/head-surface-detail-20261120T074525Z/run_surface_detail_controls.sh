#!/usr/bin/env bash
# From video: bash run_surface_detail_controls.sh VIDEO.mp4 NEW_RUN SPLIT.json
# A completed input stage from this experiment can also be passed as the only argument.
set -euo pipefail
project=$(cd "$(dirname "$0")" && pwd)
if [[ $# == 3 ]]; then
    source_run=$(realpath -m "$2")
    bash "$project/run_target_profile.sh" "$1" "$source_run" "$3"
elif [[ $# == 1 ]]; then
    source_run=$(realpath "$1")
else
    echo "Pass VIDEO.mp4 NEW_RUN SPLIT.json or the fresh source stage directory" >&2
    exit 2
fi
out="$source_run/surface-detail-controls"
mkdir "$out"
cd "$project"
train() {
    local name=$1
    shift
    python3 -Bu "$project/train_bilateral_scene.py" "$source_run" "$out/$name" \
        --steps 10000 --surface-samples 4 --fit-footprints "$@"
}
audit() (
    cd "$HOME/GaussianAvatars"
    export PYTHONPATH="$HOME/ga_pkgs:$PWD"
    "$HOME/miniconda3/envs/fastavatar/bin/python" -Bu "$project/audit_surface_fit.py" \
        "$out/$1" "$out/$1/surface-fit"
)
(
    cd "$HOME/GaussianAvatars"
    export PYTHONPATH="$HOME/ga_pkgs:$PWD"
    "$HOME/miniconda3/envs/fastavatar/bin/python" -Bu "$project/prepare_surface_contours.py" "$source_run"
)
train baseline
audit baseline
train matte --prepared-seeds "$out/baseline" --soft-matte
audit matte
train matte-centers --prepared-seeds "$out/baseline" --soft-matte --contour-offset centers
audit matte-centers
train matte-mesh --prepared-seeds "$out/baseline" --soft-matte --contour-offset mesh
audit matte-mesh
train combined --prepared-seeds "$out/baseline" --soft-matte --contour-offset both
audit combined
(
    cd "$HOME/GaussianAvatars"
    export PYTHONPATH="$HOME/ga_pkgs:$PWD"
    "$HOME/miniconda3/envs/fastavatar/bin/python" -Bu "$project/compare_fixed_head_detail.py" \
        "$out"/{baseline,matte,matte-centers,matte-mesh,combined} \
        --labels Baseline Matte Matte+Centers Matte+Mesh Combined --output "$out/fixed-review"
    "$HOME/miniconda3/envs/fastavatar/bin/python" -Bu "$project/summarize_surface_controls.py" "$out"
)

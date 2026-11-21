#!/usr/bin/env bash
# OPTIONAL / NOT EXECUTED. Use only after a separately validated Linux install.
set -euo pipefail
LAB_ROOT="${GS_SLAM_LAB:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FRAMES="${1:-20}"
[[ "$FRAMES" == 20 || "$FRAMES" == 200 ]] || { echo 'Usage: bash run_splatam_optional.sh 20|200'; exit 1; }
source "$LAB_ROOT/.venvs/splatam-cu121/bin/activate"
export GS_SLAM_LAB="$LAB_ROOT" MPLBACKEND=Agg WANDB_MODE=disabled WANDB_DISABLED=true
python "$LAB_ROOT/configs/splatam/run_headless.py" "$FRAMES" --check-only
python "$LAB_ROOT/configs/splatam/run_headless.py" "$FRAMES"
python "$LAB_ROOT/configs/splatam/export_evo.py" "$LAB_ROOT/configs/splatam/tum_desk_${FRAMES}.py"

#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/msv2-clean-runtime-20261105/runtime.env
export MSV2_ROOT=/opt/gs-slam-lab/msv2-video15-20261105
export CUDA_MODULE_LOADING=LAZY
export PATH="/opt/gs-slam-lab/msv2-pipeline-validation-20261105/tools:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export MAX_JOBS=1
mkdir -p "$MSV2_ROOT"
if [[ ! -f "$MSV2_ROOT/config.json" ]]; then
  cp /mnt/d/Projects/msv2/configs/default.json "$MSV2_ROOT/config.json"
fi
exec bash /mnt/d/Projects/msv2/msv2.sh "$@"

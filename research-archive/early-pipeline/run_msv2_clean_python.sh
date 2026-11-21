#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/msv2-clean-runtime-20261105/runtime.env
export CUDA_MODULE_LOADING=LAZY
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -n "${MSV2_EXTENSIONS_DIR:-}" ]]; then
  export PYTHONPATH="${MSV2_EXTENSIONS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$MSV2_PYTHON" -s "$@"

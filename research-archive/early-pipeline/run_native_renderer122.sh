#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc"
test -f "$CUDA_HOME/include/cuda_runtime.h"
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 PYTHONNOUSERSITE=1
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
SHUOLEE_PROBE_ATTEMPT="${1:-cuda-path-fixed}"
case "$SHUOLEE_PROBE_ATTEMPT" in cuda-path-fixed|projection-observability-fix) ;; *) exit 2;; esac
SHUOLEE_PROBE_ROOT="/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/$SHUOLEE_PROBE_ATTEMPT"
mkdir -p "$SHUOLEE_PROBE_ROOT"
test ! -e "$SHUOLEE_PROBE_ROOT/native-renderer-progress.json"
"$CUDA_HOME/bin/nvcc" --version
/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_native_renderer122.py' --attempt "$SHUOLEE_PROBE_ATTEMPT" 2>&1 | tee "$SHUOLEE_PROBE_ROOT/run.log"

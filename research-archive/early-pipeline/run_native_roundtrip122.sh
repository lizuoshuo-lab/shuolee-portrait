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
SHUOLEE_R1B_ROOT='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/covariance-sh-ply'
test -s "$SHUOLEE_R1B_ROOT/plan.json"
test ! -e "$SHUOLEE_R1B_ROOT/progress.json"
/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_native_roundtrip122.py' 2>&1 | tee "$SHUOLEE_R1B_ROOT/run.log"

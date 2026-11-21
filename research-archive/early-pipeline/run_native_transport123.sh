#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc"
test -f "$CUDA_HOME/include/cuda_runtime.h"
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/dynamic-human/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=0
exec timeout --signal=TERM --kill-after=10s 240s /opt/gs-slam-lab/dynamic-human/.venv/bin/python -B '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_clothing_transport123.py' --output-dir "$1"

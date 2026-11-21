#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat
unset SHUOLEE_EXTRA_PYTHONPATH
exec 9>/tmp/shuolee-native122-gpu.lock
flock -n 9
SHUOLEE_GPU_TASKS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
test -z "$SHUOLEE_GPU_TASKS"
export SHUOLEE_R3_LOCK=held-by-flock
exec /opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python -s '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_residual_capacity122.py'

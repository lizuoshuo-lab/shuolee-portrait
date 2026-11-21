#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
exec 9>/tmp/shuolee-native122-gpu.lock
flock -n 9
export SHUOLEE_T7_SHARED_LOCK=held-by-flock
python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_native_capacity122.py' 2>&1 | tee '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-capacity/run.log'

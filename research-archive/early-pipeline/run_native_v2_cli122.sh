#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
exec 9>/tmp/shuolee-native122-gpu.lock
flock -n 9
/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/check_native_v2_cli122.py'

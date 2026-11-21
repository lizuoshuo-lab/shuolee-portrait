#!/usr/bin/env bash
# Local native CLI: reviewed gsplat source, not the old human training launcher.
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 PYTHONNOUSERSITE=1
# gsplat is a source checkout in this existing environment, not a pip package.
# The Python loader verifies its version and source hashes against the manifest.
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat
unset SHUOLEE_EXTRA_PYTHONPATH
exec /opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python -s '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/native_scene_package122.py' "$@"

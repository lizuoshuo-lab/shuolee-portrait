#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8
test -x "$CUDA_HOME/bin/nvcc"
test -f "$CUDA_HOME/include/cuda_runtime.h"
export PATH="$CUDA_HOME/bin:/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 VERBOSE=1
export TORCH_EXTENSIONS_DIR=/opt/gs-slam-lab/transport123-native-cache
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
case "$1" in
  load) exec timeout --signal=TERM --kill-after=10s 60s /opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python -B '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/check_transport_runtime123.py' "$2" ;;
  probe) exec timeout --signal=TERM --kill-after=10s 240s /opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python -B '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/probe_clothing_transport123_v2.py' --output-dir "$2" ;;
  *) exit 2 ;;
esac

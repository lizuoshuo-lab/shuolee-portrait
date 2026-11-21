#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
exec /opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/check_native_admission122.py' "$@"

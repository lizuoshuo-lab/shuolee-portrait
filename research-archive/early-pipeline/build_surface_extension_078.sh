#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_root=/opt/gs-slam-lab/shuolee-surface-moments
task_source=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee/submodules/diff-plane-moment-rasterization
test ! -e "$task_root"
mkdir -p "$task_root"
cp -a "$task_source" "$task_root/source"
cd "$task_root/source"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -m pip install . --no-deps --no-build-isolation --target "$task_root/target" > "$task_root/build.log" 2>&1
tail -n 8 "$task_root/build.log"

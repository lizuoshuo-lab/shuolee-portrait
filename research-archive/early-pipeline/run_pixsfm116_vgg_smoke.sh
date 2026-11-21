#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
ROOT=/opt/gs-slam-lab/shuolee-pixsfm116
OUT=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/pixsfm
SCRIPT='/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/run_pixsfm116_vgg_smoke.py'
export LD_LIBRARY_PATH="$ROOT/prefix/lib:$ROOT/prefix/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_HOME=/root/.cache/torch
test ! -e "$OUT/ka-vgg-smoke"
timeout --signal=TERM --kill-after=10 1820s "$ROOT/.venv/bin/python" -u "$SCRIPT" > "$OUT/ka-vgg-smoke.log" 2> "$OUT/ka-vgg-smoke-stderr.log"

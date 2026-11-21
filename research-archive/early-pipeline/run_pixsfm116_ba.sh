#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export LD_LIBRARY_PATH="/opt/gs-slam-lab/shuolee-pixsfm116/prefix/lib:/opt/gs-slam-lab/shuolee-pixsfm116/prefix/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_HOME=/root/.cache/torch
out='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/pixsfm'
exec timeout --signal=TERM --kill-after=10 12600 /opt/gs-slam-lab/shuolee-pixsfm116/.venv/bin/python -u \
 '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/run_pixsfm116_ba.py' \
 > "$out/ba-vgg-candidate.log" 2> "$out/ba-vgg-candidate.stderr.log"

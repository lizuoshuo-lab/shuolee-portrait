#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
out='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/pixsfm'
exec timeout --signal=TERM --kill-after=10 1800 /opt/gs-slam-lab/dynamic-human/.venv/bin/python -u \
 '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/evaluate_pixsfm116_candidate.py' \
 > "$out/ba-evaluation.log" 2> "$out/ba-evaluation.stderr.log"

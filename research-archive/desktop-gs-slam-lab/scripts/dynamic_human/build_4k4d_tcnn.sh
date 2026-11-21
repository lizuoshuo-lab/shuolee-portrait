#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/4k4d/env.sh
OUT=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-4k4d-official-035
cd /opt/gs-slam-lab/4k4d/tiny-cuda-nn/bindings/torch
python -m pip install . --no-build-isolation --no-deps -v > "$OUT/tcnn-build-recovery.log" 2>&1

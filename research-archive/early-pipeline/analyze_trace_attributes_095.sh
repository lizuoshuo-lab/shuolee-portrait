#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u analyze_trace_attributes_095.py > /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/whole-body-early-trace-fresh-095/attributes.log 2>&1
tail -n 6 /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/whole-body-early-trace-fresh-095/attributes.log

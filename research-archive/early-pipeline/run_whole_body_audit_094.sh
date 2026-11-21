#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
task_lab=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
task_out="$task_lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/whole-body-alignment-audit-094"
mkdir -p "$task_out"
cd "$task_lab/repos/PGSR-shuolee"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u audit_whole_body_alignment_094.py > "$task_out/audit.log" 2>&1

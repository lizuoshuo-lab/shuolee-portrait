#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee"
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u shuolee_static.py --run "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098" --steps 18000 --height 1536 --mask-foreground-only --observed-correspondence "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/fullscene-person-constraint-audit-076/observed-bank/pairs.npz" --observed-weight 0.002 --pixel-centers-consistent --ncc-factor 1 --shared-face-world "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098/shared-targets/shared-face-points.npz" --shared-face-weight 0.002 > "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098/gaussian.log" 2>&1
tail -n 2 "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-shared-face-fresh-098/gaussian.log"

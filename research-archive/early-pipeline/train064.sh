#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u shuolee_static.py --run /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-pgsr-fresh-064 "$@" > /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-pgsr-fresh-064/gaussian.log 2>&1

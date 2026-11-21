#!/usr/bin/env bash
set -euo pipefail
export SHUOLEE_EXTRA_PYTHONPATH='/opt/gs-slam-lab/shuolee-ga-raster/target:/opt/gs-slam-lab/mini-splatting2-native-001/deps:/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/novel-view-face/head-structure/private-runtime/python-deps:/opt/gs-slam-lab/shuolee-hsfm116/python'
exec /opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python '/mnt/d/00/工作/人体重建/run.py' "$@"

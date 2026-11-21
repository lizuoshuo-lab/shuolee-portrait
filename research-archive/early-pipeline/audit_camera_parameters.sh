#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee
/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python -u audit_camera_parameters_073.py > /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/person-static-registered-camera-fresh-072/camera-parameters.log 2>&1

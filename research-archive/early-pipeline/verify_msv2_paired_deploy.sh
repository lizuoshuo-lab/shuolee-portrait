#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/msv2-clean-runtime-20261105/runtime.env
export MSV2_ROOT=/opt/gs-slam-lab/msv2-paired-deploy-validation-20261106
export MSV2_ROI_BACKEND_DIR=/opt/gs-slam-lab/msv2-roi-backend-20261106
export MSV2_HEAD_PARSER_SOURCE=/mnt/d/GS-SLAM-Data/msv2-head-parser-assets
export MSV2_HEAD_PARSER_WEIGHT=/mnt/d/GS-SLAM-Data/msv2-head-parser-assets/resnet18.pt
export MSV2_HEAD_DETECTOR_WEIGHT=/mnt/d/GS-SLAM-Data/msv2-face-assets/yunet/face_detection_yunet_2023mar.onnx
export CUDA_MODULE_LOADING=LAZY
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash /mnt/d/Projects/msv2-paired/scripts/bootstrap.sh
# Verify saved runtime pointers on a second deployment, without caller overrides.
unset MSV2_PYTHON MSV2_SOURCE_DIR MSV2_SWINIR_SOURCE MSV2_SWINIR_WEIGHT MSV2_HEAD_PARSER_SOURCE MSV2_HEAD_PARSER_WEIGHT MSV2_HEAD_DETECTOR_WEIGHT MSV2_ROI_BACKEND_DIR CUDA_HOME
bash /mnt/d/Projects/msv2-paired/scripts/bootstrap.sh

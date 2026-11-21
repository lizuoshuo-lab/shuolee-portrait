#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/msv2-clean-runtime-20261105/runtime.env
export MSV2_ROOT=/opt/gs-slam-lab/msv2-video15-20261105
export MSV2_HEAD_PARSER_SOURCE=/mnt/d/GS-SLAM-Data/msv2-head-parser-assets
export MSV2_HEAD_PARSER_WEIGHT=/mnt/d/GS-SLAM-Data/msv2-head-parser-assets/resnet18.pt
export MSV2_HEAD_DETECTOR_WEIGHT=/mnt/d/GS-SLAM-Data/msv2-face-assets/yunet/face_detection_yunet_2023mar.onnx
export CUDA_MODULE_LOADING=LAZY
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec bash /mnt/d/Projects/msv2-head/msv2.sh train \
  /mnt/d/GS-SLAM-Data/msv2-video15-20261105/dataset \
  --config '/mnt/d/00/记录/2026-11-05_MSV2_video15_头部框超分/头部训练配置.json' \
  --scales 4 --name video15-head4x

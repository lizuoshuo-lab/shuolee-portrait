#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first: sudo apt-get install ffmpeg'; exit 1; }
command -v nvcc >/dev/null || { echo 'CUDA development toolkit (nvcc) is required to build the renderer.'; exit 1; }
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./src/avatar_core/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./src/avatar_core/submodules/simple-knn
python -m pip install --no-build-isolation ./third_party/nvdiffrast
python -m pip install --no-build-isolation ./third_party/pytorch3d
echo 'Code environment installed. Configure model assets outside the Git/WPS tree, then run:'
echo 'python run.py doctor --assets /data/shuolee-models'

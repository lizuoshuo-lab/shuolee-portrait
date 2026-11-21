#!/usr/bin/env bash
# OPTIONAL / NOT EXECUTED. User's current path is the local gsplat experiment.
# For a future existing Linux RTX3090/4090 machine only. Does not provision,
# purchase, upload data, install a system driver, or modify a global environment.
set -euo pipefail
LAB_ROOT="${GS_SLAM_LAB:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPO="$LAB_ROOT/repos/SplaTAM"
EXPECTED=da6bbcd24c248dc884ac7f49d62e91b841b26ccc
[[ "$(uname -s)" == Linux ]] || { echo 'Requires Linux'; exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED" ]] || { echo 'Repository commit mismatch'; exit 1; }
command -v python3.10 >/dev/null || { echo 'Provide Python 3.10 first'; exit 1; }
command -v nvcc >/dev/null || { echo 'Provide CUDA Toolkit 12.1 (nvcc), not just an NVIDIA driver'; exit 1; }
nvcc --version | grep -q 'release 12.1' || { echo 'This proposed recipe is specifically CUDA Toolkit 12.1'; exit 1; }
ENV_DIR="$LAB_ROOT/.venvs/splatam-cu121"
[[ ! -e "$ENV_DIR" ]] || { echo "Preserving existing environment: $ENV_DIR"; exit 1; }
python3.10 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0 ninja==1.11.1.1
python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$LAB_ROOT/scripts/cloud/splatam-offline-requirements.txt"
# The exact upstream depth/silhouette rasterizer is required, not gsplat.
python -m pip install --no-build-isolation 'git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110'
python -m pip check
python -m pip freeze > "$LAB_ROOT/manifests/repos/splatam_optional_linux_freeze.txt"
python "$LAB_ROOT/scripts/cloud/verify_splatam_cuda.py"
echo 'CUDA smoke completed. Full SplaTAM/TUM execution is a separate check.'

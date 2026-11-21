#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/gs-slam-lab/hamer
BASE=/opt/gs-slam-lab/dynamic-human/.venv
mkdir -p "$ROOT"
if [ ! -x "$ROOT/.venv/bin/python" ]; then "$BASE/bin/python" -m venv "$ROOT/.venv"; fi
printf '%s\n' "$BASE/lib/python3.10/site-packages" > "$ROOT/.venv/lib/python3.10/site-packages/gs_existing_runtime.pth"
"$ROOT/.venv/bin/python" -m pip install 'pytorch-lightning==2.5.1' 'einops==0.8.1' 'pyrender==0.1.45' 'webdataset==0.2.111' 'hydra-core==1.3.2' 'rich==13.9.4'
"$ROOT/.venv/bin/python" -m pip install --no-deps -e /opt/gs-slam-lab/dynamic-human/repos/drivestudio/third_party/smplx
# pyrender pins3.1.0, whose OSMesa binding lacks CreateContextAttribs.
# Keep this isolated preview override, validated against libosmesa6 on Ubuntu20.04.
"$ROOT/.venv/bin/python" -m pip install --no-deps 'PyOpenGL==3.1.7' 'chumpy==0.70'
printf '%s\n' 'export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/hamer:${PYTHONPATH:-}' > "$ROOT/env.sh"
printf '%s\n' 'source /opt/gs-slam-lab/hamer/.venv/bin/activate' >> "$ROOT/env.sh"
printf '%s\n' 'export PYOPENGL_PLATFORM=egl' >> "$ROOT/env.sh"
source "$ROOT/env.sh"
python /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/check_hamer_import.py

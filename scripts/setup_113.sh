#!/usr/bin/env bash
# Add the original 113 plane rasterizer to the pinned, isolated server environment.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHUOLEE_ENV_DIR="${SHUOLEE_ENV_DIR:-${HOME}/.local/share/shuolee-portrait/env-cu128}"
PY="$SHUOLEE_ENV_DIR/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
[[ -x "$PY" ]] || { echo 'Run bash scripts/setup_server.sh first.' >&2; exit 1; }
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'Set CUDA_HOME to the CUDA 12.8 toolkit.' >&2; exit 1; }
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8,'
export PATH="$SHUOLEE_ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$PY" -s -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')}"
export CPATH="$("$PY" -s -c 'import pathlib,sysconfig; r=pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia"; print(":".join(str(p) for p in sorted(r.glob("*/include"))))')"
export LIBRARY_PATH="$("$PY" -s -c 'import pathlib,sysconfig; r=pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia"; print(":".join(str(p) for p in sorted(r.glob("*/lib"))))')"
mkdir -p "$SHUOLEE_ENV_DIR/install"
SOURCE="$("$PY" -s - "$PROJECT_ROOT" "$SHUOLEE_ENV_DIR" <<'PY'
from pathlib import Path
import hashlib,sys,shutil
p,e=map(lambda x:Path(x).resolve(),sys.argv[1:]);s=p/'third_party/pgsr113-rasterizer'
assert p not in e.parents and p!=e, 'Environment must be outside project.'
h=hashlib.sha256()
for f in sorted(s.rglob('*')):
    if f.is_file(): h.update(f.relative_to(s).as_posix().encode());h.update(f.read_bytes())
d=e/'install'/('pgsr113-'+h.hexdigest()[:16])
if not d.exists(): shutil.copytree(s,d)
print(d)
PY
)"
"$PY" -s -m pip install --no-deps --no-build-isolation "$SOURCE" 2>&1 | tee "$SHUOLEE_ENV_DIR/install/setup113.log"
"$PY" -s -m pip check
"$PY" -s -c 'import torch,diff_plane_rasterization._C; assert torch.cuda.is_available(); print("113 rasterizer imported; run route113.py doctor next.")'

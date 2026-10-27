#!/usr/bin/env bash
# Install code dependencies only. Model weights and reconstruction outputs are separate.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SHUOLEE_ENV_DIR="${SHUOLEE_ENV_DIR:-${HOME}/.local/share/shuolee-portrait/env-cu128}"
export MAX_JOBS="${MAX_JOBS:-2}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH

fail() { echo "ERROR: $*" >&2; exit 1; }
mode=install
ASSETS_DIR=""
RUNS_DIR="${HOME}/shuolee/runs"
while [[ $# -gt 0 ]]; do
  case "$1" in
    install|--preflight-only|--resume-install|--verify-existing) mode="$1"; shift ;;
    --assets|--runs)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing path after $1"
      if [[ "$1" == --assets ]]; then ASSETS_DIR="$2"; else RUNS_DIR="$2"; fi
      shift 2 ;;
    --help|-h)
      echo 'Usage: bash scripts/setup_server.sh [--preflight-only|--resume-install|--verify-existing] [--assets /path/to/models] [--runs /path/to/runs]'
      echo 'Install/verify Python+CUDA reconstruction only; --assets validates models and configures train119.sh.'
      exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ -f "$PROJECT_ROOT/constraints-linux-cu128.txt" ]] || fail 'Missing pinned dependency constraints.'
export PIP_CONSTRAINT="$PROJECT_ROOT/constraints-linux-cu128.txt"
configure_119() {
  if [[ -n "$ASSETS_DIR" ]]; then
    "$ENV_PYTHON" -s "$PROJECT_ROOT/run119.py" configure --assets "$ASSETS_DIR" --runs "$RUNS_DIR"
    echo '119 deployment ready. Train: bash scripts/train119.sh /path/to/video.mp4'
  else
    echo 'Dependencies checked. To bind models: bash scripts/setup_server.sh --verify-existing --assets /path/to/models'
  fi
}
if [[ "$mode" == --verify-existing ]]; then
  ENV_PYTHON="$SHUOLEE_ENV_DIR/bin/python"
  [[ -x "$ENV_PYTHON" && -f "$SHUOLEE_ENV_DIR/pyvenv.cfg" ]] || fail 'Configured isolated Python environment is missing.'
  "$ENV_PYTHON" -s -m pip check
  "$ENV_PYTHON" -s "$PROJECT_ROOT/scripts/check_environment.py" --output "$SHUOLEE_ENV_DIR/install/environment-check.json"
  configure_119
  exit 0
fi
for exe in ffmpeg ffprobe g++ cmake; do
  command -v "$exe" >/dev/null || fail "Missing $exe. Install FFmpeg and C++ build tools first."
done
command -v "$PYTHON_BIN" >/dev/null || fail 'Set PYTHON_BIN to a Python 3.10 executable.'
"$PYTHON_BIN" -s -c 'import platform,sys; assert sys.version_info[:2]==(3,10), "Python 3.10 required"; assert sys.platform=="linux" and platform.machine()=="x86_64", "Linux x86_64 required"'
if [[ -z "${CUDA_HOME:-}" ]]; then
  nvcc_path="$(command -v nvcc || true)"
  [[ -n "$nvcc_path" ]] || fail 'Set CUDA_HOME to the CUDA 12.8 development toolkit directory (containing bin/nvcc).'
  CUDA_HOME="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
fi
export CUDA_HOME
[[ -x "$CUDA_HOME/bin/nvcc" ]] || fail "nvcc not found in $CUDA_HOME/bin"
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8,' || fail 'This pinned environment requires CUDA toolkit 12.8.'
export PATH="$CUDA_HOME/bin:$PATH"
"$PYTHON_BIN" -s - "$PROJECT_ROOT" "$SHUOLEE_ENV_DIR" <<'PY'
from pathlib import Path
import sys
project,env=map(lambda p:Path(p).expanduser().resolve(),sys.argv[1:])
if env==project or project in env.parents:
    raise SystemExit('Environment must be outside the Git/WPS project. Set SHUOLEE_ENV_DIR.')
if str(env).lower().startswith('/mnt/d/00/'):
    raise SystemExit('D:/00 is WPS-synchronized; place the environment elsewhere.')
print(f'Project: {project}\nEnvironment: {env}')
PY
echo "Python: $(command -v "$PYTHON_BIN"); CUDA toolkit: $CUDA_HOME; build workers: $MAX_JOBS"
[[ "$mode" != --preflight-only ]] || exit 0

if [[ -e "$SHUOLEE_ENV_DIR" ]]; then
  [[ "$mode" == --resume-install ]] || fail 'Environment already exists. For an interrupted dependency install only, inspect its log and use --resume-install. This never resumes training.'
  [[ -f "$SHUOLEE_ENV_DIR/pyvenv.cfg" ]] || fail 'Existing destination is not a Python venv.'
else
  "$PYTHON_BIN" -s -m venv "$SHUOLEE_ENV_DIR"
fi
ENV_PYTHON="$SHUOLEE_ENV_DIR/bin/python"
export PATH="$SHUOLEE_ENV_DIR/bin:$PATH"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SHUOLEE_ENV_DIR/install/cache}"
export TORCH_EXTENSIONS_DIR="$SHUOLEE_ENV_DIR/install/torch-extensions"
mkdir -p "$SHUOLEE_ENV_DIR/install"
exec > >(tee -a "$SHUOLEE_ENV_DIR/install/setup.log") 2>&1
trap 'echo "Install failed at line $LINENO. Logs retained; no training was started."' ERR
"$ENV_PYTHON" -s - "$SHUOLEE_ENV_DIR" <<'PY'
from pathlib import Path
import site,sys
env=Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve()==env
assert not site.ENABLE_USER_SITE
foreign=[p for p in sys.path if 'site-packages' in p and env not in Path(p).resolve().parents]
assert not foreign, f'Foreign site-packages: {foreign}'
PY
"$ENV_PYTHON" -s -m pip install pip==25.1.1 setuptools==79.0.1 wheel==0.45.1
if [[ -n "${CUDA_WHEELHOUSE:-}" ]]; then
  # Install reviewed cached wheels first: an online index may otherwise win over find-links.
  "$ENV_PYTHON" -s -m pip install --no-index --find-links "$CUDA_WHEELHOUSE" -r "$PROJECT_ROOT/scripts/cuda128-requirements.txt"
fi
"$ENV_PYTHON" -s -m pip install torch==2.7.1 torchvision==0.22.1 -c "$PROJECT_ROOT/requirements.txt" --index-url https://download.pytorch.org/whl/cu128
# Chumpy's legacy build imports pip/NumPy; install the prerequisites before its source build.
"$ENV_PYTHON" -s -m pip install numpy==1.26.4 scipy==1.14.1 six==1.17.0
"$ENV_PYTHON" -s -m pip install --no-build-isolation chumpy==0.70
"$ENV_PYTHON" -s -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$ENV_PYTHON" -s -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU/driver unavailable"; assert torch.version.cuda=="12.8"; print(torch.__version__, torch.cuda.get_device_name())'
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$("$ENV_PYTHON" -s -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
fi
export TORCH_CUDA_ARCH_LIST
# Split CUDA toolkit installations may omit library headers. Use the matching
# NVIDIA wheels installed in this venv, never headers from another experiment.
NVIDIA_PACKAGE_ROOT="$("$ENV_PYTHON" -s -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/nvidia")')"
export CPATH="$("$ENV_PYTHON" -s - "$NVIDIA_PACKAGE_ROOT" <<'PY'
from pathlib import Path
import sys
print(':'.join(str(p) for p in sorted(Path(sys.argv[1]).glob('*/include')) if p.is_dir()))
PY
)"
export LIBRARY_PATH="$("$ENV_PYTHON" -s - "$NVIDIA_PACKAGE_ROOT" <<'PY'
from pathlib import Path
import sys
print(':'.join(str(p) for p in sorted(Path(sys.argv[1]).glob('*/lib')) if p.is_dir()))
PY
)"
# Stage native sources outside the synchronized checkout; no build artifacts in D:/00.
NATIVE_ROOT="$("$ENV_PYTHON" -s "$PROJECT_ROOT/scripts/stage_native_sources.py" "$SHUOLEE_ENV_DIR/install")"
for package in diff-gaussian-rasterization simple-knn nvdiffrast pytorch3d; do
  "$ENV_PYTHON" -s -m pip install --no-deps --no-build-isolation "$NATIVE_ROOT/$package"
done
"$ENV_PYTHON" -s -m pip check
"$ENV_PYTHON" -s -m pip freeze > "$SHUOLEE_ENV_DIR/install/installed.txt"
"$ENV_PYTHON" -s "$PROJECT_ROOT/scripts/check_environment.py" --output "$SHUOLEE_ENV_DIR/install/environment-check.json"
echo "Dependency installation passed. Activate: source '$SHUOLEE_ENV_DIR/bin/activate'"
configure_119
echo 'No model weights were downloaded; no reconstruction quality is implied.'

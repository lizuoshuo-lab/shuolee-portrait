#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
ROOT=/opt/gs-slam-lab/shuolee-pixsfm116
OUT='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/pixsfm'
mkdir -p "$ROOT/src" "$ROOT/prefix" "$ROOT/build" "$OUT"
exec 9>"$ROOT/setup.lock"
flock -n 9 || exit 73
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=2
export DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0
BASEPY=/opt/gs-slam-lab/dynamic-human/.venv/bin/python
STAGE=preflight
mark() {
  "$BASEPY" - "$OUT/setup-state.json" "$1" "$STAGE" "$$" <<'PY'
import json,sys,datetime,os
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp')
state={'state':sys.argv[2],'stage':sys.argv[3],'pid':int(sys.argv[4]),'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'isolated native dependency build; no model optimization orGaussian training'}
t.write_text(json.dumps(state,indent=2)+'\n');os.replace(t,p)
PY
}
trap 'code=$?; if [ "$code" -ne 0 ]; then mark failed; printf "Setup failed at %s (exit %s)\n" "$STAGE" "$code"; fi' EXIT
date -Is
mark running
STAGE=apt_development_dependencies;mark running
apt-get -y --no-install-recommends install libgoogle-glog-dev libgflags-dev libsuitesparse-dev libfreeimage-dev libcgal-dev ninja-build
clone_tag() {
  local name="$1" url="$2" tag="$3"
  if [ ! -d "$ROOT/src/$name/.git" ]; then git clone --depth 1 --branch "$tag" "$url" "$ROOT/src/$name"; fi
  git -C "$ROOT/src/$name" rev-parse HEAD
}
STAGE=source_checkout;mark running
clone_tag ceres https://github.com/ceres-solver/ceres-solver.git 2.1.0
clone_tag colmap https://github.com/colmap/colmap.git 3.8
clone_tag pycolmap https://github.com/colmap/pycolmap.git v0.4.0
clone_tag pyceres https://github.com/cvg/pyceres.git v1.0
if [ ! -d "$ROOT/src/pixel-perfect-sfm/.git" ]; then
  git clone --depth 1 https://github.com/cvg/pixel-perfect-sfm.git "$ROOT/src/pixel-perfect-sfm"
fi
test "$(git -C "$ROOT/src/pixel-perfect-sfm" rev-parse HEAD)" = 40f7c1339328b2a0c7cf71f76623fb848e0c0357
git -C "$ROOT/src/pixel-perfect-sfm" submodule update --init --recursive --depth 1
git -C "$ROOT/src/pycolmap" submodule update --init --recursive --depth 1
git -C "$ROOT/src/pyceres" submodule update --init --recursive --depth 1
STAGE=isolated_python;mark running
if [ ! -x "$ROOT/.venv/bin/python" ]; then "$BASEPY" -m venv "$ROOT/.venv"; fi
PY="$ROOT/.venv/bin/python"
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
# Shared existing Torch is read-only; local site-packages takes precedence for compiled modules.
printf '%s\n' /opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages > "$SITE/zz_existing_torch_readonly.pth"
"$PY" -m pip install 'setuptools<70' wheel 'numpy<2' 'cmake>=3.22,<4' 'pybind11<2.12' ninja
export PATH="$ROOT/.venv/bin:$PATH"
export CMAKE_PREFIX_PATH="$ROOT/prefix"
export LD_LIBRARY_PATH="$ROOT/prefix/lib:$ROOT/prefix/lib64:${LD_LIBRARY_PATH:-}"
STAGE=build_ceres;mark running
cmake -S "$ROOT/src/ceres" -B "$ROOT/build/ceres" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/prefix" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$ROOT/build/ceres" -j2
cmake --install "$ROOT/build/ceres"
STAGE=build_colmap;mark running
cmake -S "$ROOT/src/colmap" -B "$ROOT/build/colmap" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/prefix" -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DCGAL_ENABLED=OFF -DTESTS_ENABLED=OFF
cmake --build "$ROOT/build/colmap" -j2
cmake --install "$ROOT/build/colmap"
STAGE=stage1_complete;mark ready_for_python_bindings
date -Is

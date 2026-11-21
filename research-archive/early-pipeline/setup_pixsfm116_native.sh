#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
ROOT=/opt/gs-slam-lab/shuolee-pixsfm116
OUT=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/pixsfm
exec 9>"$ROOT/setup.lock"
flock -n 9 || exit 73
PY="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:$PATH"
export CMAKE_PREFIX_PATH="$ROOT/prefix"
export LD_LIBRARY_PATH="$ROOT/prefix/lib:$ROOT/prefix/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=2
STAGE=native_build_recovery
mark() {
  "$PY" - "$OUT/setup-state.json" "$1" "$STAGE" "$$" <<'PY'
import json,sys,datetime,os
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix('.tmp')
s={'state':sys.argv[2],'stage':sys.argv[3],'pid':int(sys.argv[4]),'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline':'2026-10-25T08:16:20+00:00','scope':'isolated native andPython runtime setup; no subject optimization orGaussian training'}
t.write_text(json.dumps(s,indent=2)+'\n');os.replace(t,p)
PY
}
trap 'code=$?; if [ "$code" -ne 0 ]; then mark failed; printf "Setup failed at %s (exit %s)\n" "$STAGE" "$code"; fi' EXIT
date -Is
mark running
test "$(git -C "$ROOT/src/ceres" rev-parse HEAD)" = f68321e7de8929fbcdb95dd42877531e64f72f66
test "$(git -C "$ROOT/src/colmap" rev-parse HEAD)" = 43de802cfb3ed2bd155150e7e5e3e8c8dd5aaa3e
test "$(git -C "$ROOT/src/pixel-perfect-sfm" rev-parse HEAD)" = 40f7c1339328b2a0c7cf71f76623fb848e0c0357
STAGE=build_ceres_cpu_only;mark running
cmake -S "$ROOT/src/ceres" -B "$ROOT/build/ceres" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/prefix" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=ON -DCUDA=OFF
cmake --build "$ROOT/build/ceres" -j2
cmake --install "$ROOT/build/ceres"
STAGE=build_colmap;mark running
cmake -S "$ROOT/src/colmap" -B "$ROOT/build/colmap" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/prefix" -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DCGAL_ENABLED=OFF -DTESTS_ENABLED=OFF
cmake --build "$ROOT/build/colmap" -j2
cmake --install "$ROOT/build/colmap"
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
STAGE=build_pycolmap;mark running
cmake -S "$ROOT/src/pycolmap" -B "$ROOT/build/pycolmap" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$SITE" -DPYTHON_EXECUTABLE="$PY" -DPYBIND11_LTO_CXX_FLAGS= -DPYBIND11_LTO_LINKER_FLAGS=
cmake --build "$ROOT/build/pycolmap" -j1
STAGE=build_pyceres;mark running
cmake -S "$ROOT/src/pyceres" -B "$ROOT/build/pyceres" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$SITE" -DPYTHON_EXECUTABLE="$PY" -DPYBIND11_LTO_CXX_FLAGS= -DPYBIND11_LTO_LINKER_FLAGS=
cmake --build "$ROOT/build/pyceres" -j1
STAGE=python_requirements;mark running
"$PY" -m pip install --retries 1 --timeout 20 omegaconf tqdm matplotlib plotly scipy h5py packaging
STAGE=build_pixsfm;mark running
cmake -S "$ROOT/src/pixel-perfect-sfm" -B "$ROOT/build/pixsfm" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$ROOT/src/pixel-perfect-sfm/pixsfm" -DPYTHON_EXECUTABLE="$PY" -DTESTS_ENABLED=OFF -DHIGHFIVE_UNIT_TESTS=OFF -DPYBIND11_LTO_CXX_FLAGS= -DPYBIND11_LTO_LINKER_FLAGS=
cmake --build "$ROOT/build/pixsfm" -j1
printf '%s\n' "$ROOT/src/pixel-perfect-sfm" > "$SITE/shuolee_pixsfm_source.pth"
STAGE=import_check;mark running
"$PY" - "$OUT/runtime-check.json" <<'PY'
import datetime,json,sys,sysconfig
from pathlib import Path
import torch,torchvision,pycolmap,pyceres,pixsfm
from pixsfm.refine_colmap import PixSfM
root='/opt/gs-slam-lab/shuolee-pixsfm116'
assert str(pycolmap.__file__).startswith(root)
assert str(pyceres.__file__).startswith(root)
assert str(pixsfm.__file__).startswith(root)
assert torch.__version__.startswith('2.7.1')
r={'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'imports_passed':True,'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'module_paths':{m.__name__:str(m.__file__) for m in (torch,pycolmap,pyceres,pixsfm)},'native_modules':'locally compiled againstCeres2.1/COLMAP3.8','training_started':False,'weights_loaded':False,'feature_optimization_tested':False,'model_quality_accepted':False}
Path(sys.argv[1]).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
PY
STAGE=runtime_imports_complete;mark ready_for_feature_weight_check
date -Is

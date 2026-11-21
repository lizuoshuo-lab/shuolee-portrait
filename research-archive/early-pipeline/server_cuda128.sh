#!/usr/bin/env bash
# Companion to the 113 deployment guide; does not alter the frozen Git source.
# Default: discover and check. --install: use existing Conda only if none found.
set -euo pipefail
mode=check
explicit_root=
case "${1:-}" in
  '') ;;
  --diagnose) [[ $# -le 2 ]] || { echo 'Usage: bash server_cuda128.sh --diagnose [/path/to/cuda]'; exit 2; }; mode=diagnose; explicit_root=${2:-} ;;
  --install) mode=install ;;
  --toolkit) [[ $# == 2 ]] || { echo 'Usage: bash server_cuda128.sh --toolkit /path/to/cuda'; exit 2; }; explicit_root=$2 ;;
  *) echo 'Usage: bash server_cuda128.sh [--install | --toolkit /path/to/cuda | --diagnose [/path/to/cuda]]'; exit 2 ;;
esac
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'ERROR: Linux x86_64 server required.'; exit 2; }
task_dir="${SHUOLEE_CUDA_TASK_DIR:-$HOME/shuolee}"
install_root="$HOME/.local/share/shuolee-portrait/cuda-toolkit-12.8"
mkdir -p "$task_dir/logs"
env_file="$task_dir/cuda128.env"
log_file="$task_dir/logs/cuda128-check.log"
if [[ "$mode" == diagnose ]]; then log_file="$task_dir/logs/cuda128-diagnostic.txt"; fi
command -v flock >/dev/null || { echo 'ERROR: flock (util-linux) is required to prevent duplicate installation.'; exit 9; }
exec 9>"$task_dir/cuda128.lock"
if ! flock -n 9; then
  echo 'CUDA_INSTALL_IN_PROGRESS: another CUDA helper holds the lock. Wait; do not start another installation.'
  [[ "$mode" == diagnose ]] || exit 8
fi
# The earlier helper had no lock. Detect its Conda child by exact user/prefix,
# before writing the shared log or inspecting a partly installed toolkit.
guard_python="$(command -v python3.10 || command -v python3 || true)"
[[ -n "$guard_python" ]] || { echo 'ERROR: Python 3 is required for the installation process check.'; exit 9; }
"$guard_python" - "$install_root" "$mode" <<'PY'
from pathlib import Path
import os
import sys
target = os.path.abspath(sys.argv[1])
busy = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        args = [p.decode(errors='replace') for p in (proc / 'cmdline').read_bytes().split(b'\0') if p]
    except (OSError, ProcessLookupError):
        continue
    if not any(Path(arg).name in ('conda', 'mamba', 'micromamba') for arg in args):
        continue
    if not any(arg in ('create', 'install') for arg in args):
        continue
    prefixes = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg in ('--prefix', '-p')]
    prefixes += [arg.split('=', 1)[1] for arg in args if arg.startswith('--prefix=')]
    if any(os.path.abspath(p) == target for p in prefixes):
        busy.append(int(proc.name))
if busy:
    print('CUDA_INSTALL_IN_PROGRESS: existing Conda installation PIDs:', ', '.join(map(str, sorted(busy))))
    print('Wait for that installation. Existing log was not modified.')
    if sys.argv[2] != 'diagnose':
        raise SystemExit(8)
PY
exec > >(tee -a "$log_file" 9>&-) 2>&1
printf '\n=== CUDA helper revision=2026-10-26-install-steps %s mode=%s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode"
echo "CUDA 12.8 check. Log: $log_file"
echo "nvidia-smi driver capability is not proof of an installed nvcc compiler."

roots=()
if [[ -n "$explicit_root" ]]; then
  roots+=("$explicit_root")
else
  roots+=("${CUDA_HOME:-}" "${CUDA_PATH:-}" "$install_root")
  nvcc_cmd="$(command -v nvcc || true)"
  if [[ -n "$nvcc_cmd" ]]; then
    nvcc_cmd="$(readlink -f -- "$nvcc_cmd")"
    roots+=("$(dirname "$(dirname "$nvcc_cmd")")")
  fi
  shopt -s nullglob
  roots+=(/usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*
          /opt/nvidia/cuda* /usr/lib/cuda /usr "${CONDA_PREFIX:-}")
  roots+=("$HOME"/.conda/envs/*)
  for conda_root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                    "$HOME/mambaforge" /opt/conda /opt/miniconda3 /opt/anaconda3; do
    roots+=("$conda_root" "$conda_root"/envs/*)
  done
  shopt -u nullglob
  if [[ -f "$HOME/.conda/environments.txt" ]]; then
    registered_count=0
    while IFS= read -r registered_root; do
      [[ -z "$registered_root" ]] || roots+=("$registered_root")
      registered_count=$((registered_count+1))
      [[ "$registered_count" -lt 128 ]] || break
    done < "$HOME/.conda/environments.txt"
  fi
fi

selected=
found_compilers=0
wrong_versions=0
missing_headers=0
declare -A seen
for candidate in "${roots[@]}"; do
  [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
  candidate="$(cd "$candidate" && pwd -P)"
  [[ -z "${seen[$candidate]:-}" ]] || continue
  seen[$candidate]=1
  found_compilers=$((found_compilers+1))
  if ! version="$("$candidate/bin/nvcc" --version 2>&1)"; then
    echo "Skip broken nvcc: $candidate"; continue
  fi
  echo "Found nvcc: $candidate"
  echo "$version"
  if ! grep -q 'release 12\.8,' <<< "$version"; then
    wrong_versions=$((wrong_versions+1))
    echo 'Skip: version is not 12.8.'; continue
  fi
  header_root=
  for inc in "$candidate/include" "$candidate/targets/x86_64-linux/include"; do
    if [[ -f "$inc/cuda_runtime.h" && -f "$inc/cuda.h" ]]; then header_root=$inc; break; fi
  done
  if [[ -z "$header_root" ]]; then
    missing_headers=$((missing_headers+1))
    echo 'CUDA_12_8_HEADERS_MISSING: nvcc exists, but cuda.h/cuda_runtime.h were not found in include or targets/x86_64-linux/include.'; continue
  fi
  echo "Development headers: $header_root"
  selected=$candidate
  [[ "$mode" == diagnose ]] || break
done

echo "CHECK_SUMMARY: compilers=$found_compilers wrong_version=$wrong_versions missing_headers=$missing_headers"
if [[ "$mode" == diagnose ]]; then
  echo "Dedicated installation prefix: $install_root"
  if [[ -d "$install_root/conda-meta" ]]; then
    echo 'Conda CUDA package records:'
    find "$install_root/conda-meta" -maxdepth 1 -type f -name '*cuda*.json' -printf '%f\n' | sort
  elif [[ -e "$install_root" ]]; then
    echo 'INSTALL_PREFIX_WITHOUT_CONDA_RECORDS: directory exists, installation completion is not established.'
  else
    echo 'INSTALL_PREFIX_ABSENT: this dedicated installation directory does not exist.'
  fi
  if [[ -s "$env_file" ]]; then echo "Environment file exists: $env_file"; else echo "ENV_FILE_ABSENT: $env_file"; fi
  if [[ -f "$task_dir/logs/cuda128-check.log" ]]; then
    echo 'Last 100 lines of the installation/check log:'
    tail -n 100 "$task_dir/logs/cuda128-check.log"
  fi
  echo "DIAGNOSTIC_ONLY: no installation, compile or environment-file change was performed. Report: $log_file"
  echo 'A historical READY line in the copied log is not a new successful check.'
  exit 0
fi

if [[ -z "$selected" && "$mode" == install ]]; then
  conda_exe=
  for possible in "${CONDA_EXE:-}" "$(type -P conda || true)" \
                  "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
                  "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" \
                  /opt/conda/bin/conda /opt/miniconda3/bin/conda /opt/anaconda3/bin/conda; do
    if [[ -n "$possible" && -x "$possible" ]]; then conda_exe=$possible; break; fi
  done
  if [[ -z "$conda_exe" ]]; then
    echo 'CONDA_NOT_FOUND: ask the server administrator for an existing CUDA 12.8 toolkit path or a CUDA 12.8 toolkit installation.'
    echo 'No driver or system packages changed.'
    exit 4
  fi
  if [[ -e "$install_root" ]]; then
    echo "ERROR: destination already exists but did not pass the check: $install_root"
    echo 'Keep this directory and log; do not overwrite or delete another environment.'
    exit 5
  fi
  echo "Installing official NVIDIA toolkit 12.8.1 into: $install_root"
  echo 'Network download required. Dedicated toolkit prefix; existing Python environments and drivers are unchanged.'
  "$conda_exe" create --yes --prefix "$install_root" --override-channels \
    --channel https://conda.anaconda.org/nvidia/label/cuda-12.8.1 \
    --channel https://conda.anaconda.org/conda-forge \
    cuda-toolkit=12.8.1
  selected=$install_root
fi

if [[ -z "$selected" ]]; then
  echo 'CUDA_12_8_NOT_FOUND: no usable toolkit found in the checked common locations.'
  echo 'CHECK_ONLY_NO_INSTALL: this check did not install CUDA. Repeating check mode will not install it.'
  echo 'NEXT_INSTALL: in an interactive SSH/PuTTY terminal, run: bash "$HOME/shuolee/server_cuda128.sh" --install'
  echo 'Follow guide step 4A. If that installation fails, save the complete log; do not repeat installation blindly.'
  echo 'Use --diagnose to record versions, headers and the installation log.'
  echo 'If the administrator supplies another path: bash server_cuda128.sh --toolkit /actual/path'
  echo 'Do not run setup_server.sh until CUDA_12_8_READY and CUDA_ENV_READY appear.'
  exit 3
fi

[[ -x "$selected/bin/nvcc" ]] || { echo 'ERROR: selected toolkit has no nvcc.'; exit 6; }
"$selected/bin/nvcc" --version | grep -q 'release 12\.8,' || { echo 'ERROR: selected compiler is not 12.8.'; exit 6; }
command -v g++ >/dev/null || { echo 'ERROR: g++ is missing.'; exit 6; }
# Compile only: no GPU execution, model download, or training.
probe_dir="$(mktemp -d "$task_dir/logs/cuda128-probe.XXXXXX")"
cat > "$probe_dir/probe.cu" <<'CU'
#include <cuda_runtime.h>
__global__ void shuolee_probe(float *values) { values[threadIdx.x] += 1.0f; }
CU
if ! "$selected/bin/nvcc" -c --cudart none -arch=sm_80 "$probe_dir/probe.cu" -o "$probe_dir/probe.o"; then
  echo "CUDA_12_8_COMPILE_FAILED: see $log_file and $probe_dir"
  echo 'Stop here. The toolkit/host compiler needs repair before installing Python dependencies.'
  exit 7
fi

# Atomically replace the task-local environment only after version and compile pass.
env_tmp="$(mktemp "$task_dir/.cuda128.env.XXXXXX")"
{
  echo '# Generated by server_cuda128.sh after a successful CUDA 12.8 compile check.'
  printf 'export CUDA_HOME=%q\n' "$selected"
  printf 'export PATH=%q/bin:"$PATH"\n' "$selected"
} > "$env_tmp"
chmod 600 "$env_tmp"
mv -- "$env_tmp" "$env_file"
echo "CUDA_12_8_READY=$selected"
echo "Environment saved: $env_file"
echo 'Continue with guide step 4B in the same interactive SSH terminal.'

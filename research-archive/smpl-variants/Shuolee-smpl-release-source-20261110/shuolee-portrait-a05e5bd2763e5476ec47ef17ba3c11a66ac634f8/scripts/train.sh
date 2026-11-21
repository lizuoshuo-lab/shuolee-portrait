#!/usr/bin/env bash
# Forward one video and camera configuration to the unified reconstruction CLI.
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/train.sh VIDEO --camera-config CONFIG [--reference-frame frame_000001] [--output RUN]'
  echo 'Uses the environment/assets/runs deployment profile created by bash deploy.sh.'
  if [[ $# -eq 0 ]]; then exit 2; fi
  exit 0
fi
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMPLX_ROOT="${SMPLX_ROOT:-${SHUOLEE_ROOT:-${HOME}/smpl-x}}"
DEPLOYMENT_FILE="$SMPLX_ROOT/queue/deployment.json"
USER_ARGS=("$@")
for ((i=0; i<${#USER_ARGS[@]}; i++)); do
  case "${USER_ARGS[i]}" in
    --deployment)
      [[ $((i+1)) -lt ${#USER_ARGS[@]} && -n "${USER_ARGS[i+1]}" && "${USER_ARGS[i+1]}" != --* ]] || { echo 'Missing path after --deployment.' >&2; exit 2; }
      DEPLOYMENT_FILE="${USER_ARGS[i+1]}" ;;
    --deployment=*)
      DEPLOYMENT_FILE="${USER_ARGS[i]#--deployment=}"
      [[ -n "$DEPLOYMENT_FILE" ]] || { echo 'Missing path after --deployment.' >&2; exit 2; } ;;
  esac
done
SHUOLEE_ENV_DIR="${SMPLX_ENV_DIR:-${SHUOLEE_ENV_DIR:-}}"
if [[ -z "$SHUOLEE_ENV_DIR" && -f "$DEPLOYMENT_FILE" ]]; then
  SHUOLEE_ENV_DIR="$(python3 - "$DEPLOYMENT_FILE" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['environment'])
PY
)"
fi
if [[ -z "$SHUOLEE_ENV_DIR" ]]; then
  SHUOLEE_ENV_DIR="${HOME}/.local/share/smpl-x/env-cu128"
  if [[ ! -x "$SHUOLEE_ENV_DIR/bin/python" && -x "${HOME}/.local/share/shuolee-portrait/env-cu128/bin/python" ]]; then
    SHUOLEE_ENV_DIR="${HOME}/.local/share/shuolee-portrait/env-cu128"
  fi
fi
if [[ -f "$SMPLX_ROOT/cuda128.env" ]]; then
  source "$SMPLX_ROOT/cuda128.env"
elif [[ -f "$HOME/shuolee/cuda128.env" ]]; then
  source "$HOME/shuolee/cuda128.env"
fi
ENV_PYTHON="$SHUOLEE_ENV_DIR/bin/python"
[[ -x "$ENV_PYTHON" ]] || { echo 'Run bash deploy.sh first.' >&2; exit 2; }
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "$ENV_PYTHON" -s "$PROJECT_ROOT/run_smplx.py" train --deployment "$DEPLOYMENT_FILE" "$@"

#!/usr/bin/env bash
# Source and dependencies come from Git/setup; the only daily input is a video.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHUOLEE_ENV_DIR="${SHUOLEE_ENV_DIR:-${HOME}/.local/share/shuolee-portrait/env-cu128}"
ENV_PYTHON="$SHUOLEE_ENV_DIR/bin/python"
[[ -x "$ENV_PYTHON" ]] || { echo 'Run bash scripts/setup_server.sh --assets /path/to/models first.' >&2; exit 2; }
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "$ENV_PYTHON" -s "$PROJECT_ROOT/run119.py" train "$@"

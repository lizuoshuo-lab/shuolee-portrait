#!/usr/bin/env bash
# One-time dependency and deployment-profile setup for the unified entrypoint.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash deploy.sh'
  echo 'Prepares dependencies, models and a deployment profile, then prints the reconstruction command.'
  echo 'Training input: VIDEO --camera-config CONFIG [--reference-frame FRAME] [--output RUN].'
  echo 'Optional environment paths: SMPLX_ROOT, SMPLX_ENV_DIR, SMPLX_ASSETS.'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'Usage: bash deploy.sh' >&2; exit 1; }
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SHUOLEE_ROOT="${SMPLX_ROOT:-${SHUOLEE_ROOT:-${HOME}/smpl-x}}"
SHUOLEE_ENV_DIR="${SMPLX_ENV_DIR:-${SHUOLEE_ENV_DIR:-}}"
if [[ -z "$SHUOLEE_ENV_DIR" ]]; then
  SHUOLEE_ENV_DIR="${HOME}/.local/share/smpl-x/env-cu128"
  if [[ ! -x "$SHUOLEE_ENV_DIR/bin/python" && -x "${HOME}/.local/share/shuolee-portrait/env-cu128/bin/python" ]]; then
    SHUOLEE_ENV_DIR="${HOME}/.local/share/shuolee-portrait/env-cu128"
  fi
fi
export SHUOLEE_ENV_DIR
[[ "$(uname -s)" == Linux ]] || { echo 'A Linux host is required.' >&2; exit 1; }
command -v git >/dev/null
command -v python3 >/dev/null
if [[ "$(git -C "$PROJECT_ROOT" config --get core.sparseCheckout || true)" == true ]]; then
  # Older Git defaults to non-cone patterns, which would remove root entrypoints
  # and dependency locks when selecting only directories below.
  git -C "$PROJECT_ROOT" sparse-checkout init --cone
  git -C "$PROJECT_ROOT" sparse-checkout set configs scripts src third_party docs tests provenance .github
fi
# Existing 113 servers already have their compiler/driver paths here.
if [[ -f "$SHUOLEE_ROOT/cuda128.env" ]]; then
  source "$SHUOLEE_ROOT/cuda128.env"
elif [[ -f "$HOME/shuolee/cuda128.env" ]]; then
  source "$HOME/shuolee/cuda128.env"
fi
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
ENV_PYTHON="$SHUOLEE_ENV_DIR/bin/python"
ASSETS_DIR="${SMPLX_ASSETS:-${SHUOLEE_ASSETS:-}}"
ASSET_ARGS=(--root "$SHUOLEE_ROOT" --environment "$SHUOLEE_ENV_DIR")
if [[ -n "$ASSETS_DIR" ]]; then
  ASSET_ARGS+=(--assets "$ASSETS_DIR")
fi
ASSETS_DIR="$(python3 -s "$PROJECT_ROOT/scripts/prepare_assets.py" "${ASSET_ARGS[@]}")"
if [[ -x "$ENV_PYTHON" && -f "$SHUOLEE_ENV_DIR/pyvenv.cfg" ]]; then
  MODE=--verify-existing
else
  MODE=install
fi
DEPLOYMENT_FILE="$SHUOLEE_ROOT/queue/deployment.json"
bash "$PROJECT_ROOT/scripts/setup_server.sh" "$MODE" --assets "$ASSETS_DIR" --runs "$SHUOLEE_ROOT/runs" --deployment "$DEPLOYMENT_FILE"
[[ -f "$DEPLOYMENT_FILE" ]] || { echo "Deployment profile was not created: $DEPLOYMENT_FILE" >&2; exit 2; }
printf '\nDeployment configuration ready. Run reconstruction with:\n'
printf '  %q -s %q train /path/to/person.mp4 --camera-config /path/to/camera.json --deployment %q\n' "$ENV_PYTHON" "$PROJECT_ROOT/run_smplx.py" "$DEPLOYMENT_FILE"
printf 'Optional arguments: --reference-frame frame_000001 --output /path/to/new-run\n'

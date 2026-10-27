#!/usr/bin/env bash
# One-time Git deployment. Thereafter, uploading a video is the only action.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash deploy.sh'
  echo 'Creates ~/smpl-x/videos and installs the automatic smpl-x reconstruction service.'
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
SHUOLEE_SERVICE_NAME="${SMPLX_SERVICE_NAME:-${SHUOLEE_SERVICE_NAME:-smpl-x}}"
[[ "$SHUOLEE_SERVICE_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid service name' >&2; exit 1; }
[[ -d /run/systemd/system ]] || { echo 'A Linux host with systemd is required.' >&2; exit 1; }
command -v git >/dev/null
command -v python3 >/dev/null
if [[ "$(git -C "$PROJECT_ROOT" config --get core.sparseCheckout || true)" == true ]]; then
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
if [[ -z "$ASSETS_DIR" ]]; then
  ASSETS_DIR="$(python3 - "$SHUOLEE_ROOT" "$SHUOLEE_ENV_DIR" <<'PY'
import json, sys
from pathlib import Path
root, env = map(Path, sys.argv[1:])
candidates = []
for profile in (root / 'queue/deployment.json', Path.home() / 'shuolee/queue/deployment.json', env / 'install/deployment119.json'):
    if profile.is_file():
        candidates.append(Path(json.loads(profile.read_text())['assets']))
candidates += [root/'models', root/'assets', Path.home()/'shuolee/models/113', Path('/data/shuolee-models')]
for candidate in candidates:
    if (candidate/'smplx/SMPLX_NEUTRAL.npz').is_file():
        print(candidate.resolve())
        break
else:
    raise SystemExit('Server model assets missing. Set SMPLX_ASSETS to the existing model directory (docs/模型配置.md).')
PY
)"
fi
if [[ -x "$ENV_PYTHON" && -f "$SHUOLEE_ENV_DIR/pyvenv.cfg" ]]; then
  MODE=--verify-existing
else
  MODE=install
fi
bash "$PROJECT_ROOT/scripts/setup_server.sh" "$MODE" --assets "$ASSETS_DIR" --runs "$SHUOLEE_ROOT/runs" --deployment "$SHUOLEE_ROOT/queue/deployment.json"
"$ENV_PYTHON" -s "$PROJECT_ROOT/scripts/install_service.py" --name "$SHUOLEE_SERVICE_NAME" --deployment "$SHUOLEE_ROOT/queue/deployment.json"

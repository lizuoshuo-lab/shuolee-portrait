#!/usr/bin/env bash
# One-time Git deployment. Thereafter, uploading a video is the only action.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash deploy119.sh'
  echo 'Creates ~/shuolee/videos and installs the automatic 119 reconstruction service.'
  echo 'Optional environment paths: SHUOLEE_ROOT, SHUOLEE_ENV_DIR, SHUOLEE_ASSETS.'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'Usage: bash deploy119.sh' >&2; exit 1; }
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SHUOLEE_ROOT="${SHUOLEE_ROOT:-${HOME}/shuolee}"
export SHUOLEE_ENV_DIR="${SHUOLEE_ENV_DIR:-${HOME}/.local/share/shuolee-portrait/env-cu128}"
SHUOLEE_SERVICE_NAME="${SHUOLEE_SERVICE_NAME:-shuolee119}"
[[ "$SHUOLEE_SERVICE_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid service name' >&2; exit 1; }
[[ -d /run/systemd/system ]] || { echo 'A Linux host with systemd is required.' >&2; exit 1; }
command -v git >/dev/null
command -v python3 >/dev/null
if [[ "$(git -C "$PROJECT_ROOT" config --get core.sparseCheckout || true)" == true ]]; then
  git -C "$PROJECT_ROOT" sparse-checkout set configs scripts src third_party docs tests provenance .github
fi
# Existing 113 servers already have their compiler/driver paths here.
if [[ -f "$SHUOLEE_ROOT/cuda128.env" ]]; then source "$SHUOLEE_ROOT/cuda128.env"; fi
unset PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
ENV_PYTHON="$SHUOLEE_ENV_DIR/bin/python"
ASSETS_DIR="${SHUOLEE_ASSETS:-}"
if [[ -z "$ASSETS_DIR" ]]; then
  ASSETS_DIR="$(python3 - "$SHUOLEE_ROOT" "$SHUOLEE_ENV_DIR" <<'PY'
import json, sys
from pathlib import Path
root, env = map(Path, sys.argv[1:])
candidates = []
for profile in (root / 'queue/deployment.json', env / 'install/deployment119.json'):
    if profile.is_file():
        candidates.append(Path(json.loads(profile.read_text())['assets']))
candidates += [root/'models/113', root/'assets', Path('/data/shuolee-models')]
for candidate in candidates:
    if (candidate/'smplx/SMPLX_NEUTRAL.npz').is_file():
        print(candidate.resolve())
        break
else:
    raise SystemExit('Server model assets missing. Set SHUOLEE_ASSETS to the existing model directory (docs/模型配置.md).')
PY
)"
fi
if [[ -x "$ENV_PYTHON" && -f "$SHUOLEE_ENV_DIR/pyvenv.cfg" ]]; then
  MODE=--verify-existing
else
  MODE=install
fi
bash "$PROJECT_ROOT/scripts/setup_server.sh" "$MODE" --assets "$ASSETS_DIR" --runs "$SHUOLEE_ROOT/runs" --deployment "$SHUOLEE_ROOT/queue/deployment.json"
"$ENV_PYTHON" -s "$PROJECT_ROOT/scripts/install_service119.py" --name "$SHUOLEE_SERVICE_NAME" --deployment "$SHUOLEE_ROOT/queue/deployment.json"

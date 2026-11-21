#!/usr/bin/env bash
set -euo pipefail
printf 'NATIVE_LINUX_PID=%s\n' "$$"
exec bash '/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/render_native_scene122.sh' "$@"

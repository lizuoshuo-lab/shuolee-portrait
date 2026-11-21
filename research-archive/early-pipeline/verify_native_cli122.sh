#!/usr/bin/env bash
# CPU rejection only; never renders another GPU fixture or starts training.
set -euo pipefail
SHUOLEE_NATIVE_R1C='/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-package-loader'
SHUOLEE_NATIVE_LAUNCHER='/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/render_native_scene122.sh'
test ! -e "$SHUOLEE_NATIVE_R1C/cli-missing-camera.log"
set +e
bash "$SHUOLEE_NATIVE_LAUNCHER" --scene "$SHUOLEE_NATIVE_R1C/fixture-package/scene.json" --camera missing --projection pinhole --output "$SHUOLEE_NATIVE_R1C/must-not-render" > "$SHUOLEE_NATIVE_R1C/cli-missing-camera.log" 2>&1
SHUOLEE_NATIVE_EXIT=$?
set -e
test "$SHUOLEE_NATIVE_EXIT" -ne 0
grep -q 'ValueError: Recorded camera ID not found: missing' "$SHUOLEE_NATIVE_R1C/cli-missing-camera.log"
test ! -e "$SHUOLEE_NATIVE_R1C/must-not-render"
printf '%s\n' 'CLI_PARSE_RUNTIME_AND_PRE_GPU_REJECTION_PASSED'

#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/dynamic-human/env.sh
PY=/opt/gs-slam-lab/shuolee-gvhmr/.venv/bin/python
GA=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee
BASE=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee
RUN="$BASE/current/person-static-fullscene-camera-fresh-075"
# The exact already-running training job; do not start or restart training here.
while kill -0 386 2>/dev/null; do sleep 5; done
"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["steps"]==18000 and r["from_scratch"] and r["frames_used"]==175; print("18000-step fresh training complete",flush=True)' "$RUN/gaussian-run/result.json"
cd "$GA"
"$PY" -u audit_pgsr_static_delivery.py --run "$RUN" --reference "$BASE/current/person-static-registered-camera-fresh-072/gaussian-run/point_cloud.ply" --reference-label '072 prior cameras' --label '075 fullscene cameras and fresh person' > "$RUN/export-review.log" 2>&1
echo 'Native, standard PLY and upright export checks complete'
"$PY" -u review_fullscene_person_075.py > "$RUN/same-camera-review.log" 2>&1
echo '175-camera evaluation and continuous comparison complete'

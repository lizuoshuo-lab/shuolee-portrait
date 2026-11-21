#!/usr/bin/env bash
set -euo pipefail
while [[ -r /proc/997526/stat ]] && [[ "$(cut -d' ' -f3 /proc/997526/stat)" != Z ]]; do sleep 5; done
python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="complete"' '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-control-20261120T064127Z/run.json'
python3 -B -u '/home/zhouhaoyi/projects/vhap-scene-fusion/experiments/head-opacity-prune-20261120T064127Z/train_bilateral_scene.py' '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-bilateral-20261119T150523Z' '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-prune-20261120T064127Z' --steps 10000 --prepared-seeds '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-bilateral-mesh-offset-20261120T021120Z' --surface-samples 4 --fit-footprints --late-prune
cd "$HOME/GaussianAvatars"
export PYTHONPATH="$HOME/ga_pkgs:$HOME/GaussianAvatars"
"$HOME/miniconda3/envs/fastavatar/bin/python" -B -u '/home/zhouhaoyi/projects/vhap-scene-fusion/experiments/head-opacity-prune-20261120T064127Z/compare_fixed_head_detail.py' '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-control-20261120T064127Z' '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-prune-20261120T064127Z' --labels 'Full head' 'Pruned at 8000' --output '/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-prune-20261120T064127Z/quality-comparison'

"$HOME/miniconda3/envs/fastavatar/bin/python" -B "/home/zhouhaoyi/projects/vhap-scene-fusion/experiments/head-opacity-prune-20261120T064127Z/review-paired-result.py" "/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-control-20261120T064127Z" "/home/zhouhaoyi/projects/vhap-scene-fusion/runs/demo2-00560-opacity-prune-20261120T064127Z" "/home/zhouhaoyi/projects/vhap-scene-fusion/experiments/head-opacity-prune-20261120T064127Z"

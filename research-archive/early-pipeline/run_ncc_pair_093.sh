#!/bin/bash
set -euo pipefail
task_scripts='/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update'
bash "$task_scripts/train_ncc_pair_093.sh" control
bash "$task_scripts/train_ncc_pair_093.sh" no_ncc

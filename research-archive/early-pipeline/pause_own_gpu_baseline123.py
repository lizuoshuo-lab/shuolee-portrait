"""Yield only this chat's known baseline when an independent GPU train is active."""
from pathlib import Path
import os
import signal
import time
from probe_gaussian_surface_guidance123 import process,OUT
import run_official_gaussians123 as io

own=process(331);other=process(14904)
assert own['start_identity']=='9229'
assert 'train_demo_body_upstream_offset123.py train' in own['command']
assert other['start_identity']=='4134191'
assert '/static-person-upsampling-20261102/sofa/experiment.py train --subject sofa --arm upsampled2x' in other['command']
assert other['state'] not in ('T','Z')
os.kill(331,signal.SIGSTOP)
for _ in range(40):
    if process(331)['state']=='T':break
    time.sleep(.05)
assert process(331)['state']=='T'
io.save(OUT/'gpu-resource-conflict-20261102.json',dict(updated=io.now(),own_before=own,other_observed=other,
    own_after=process(331),action='suspend_only_own_baseline_to_avoid_two_active_GPU_training_jobs',
    other_process_modified=False,own_goal_paused=False,
    next='Before any further CUDA work inspect all actual GPU PIDs and command identities. Resume331 only after foreign training is no longer active and the start_identity remains9229; do not restart it or create another trainer.',
    timing_caveat='The second surface-control30s wall time overlapped an unrelated GPU training start. It is not an isolated GPU performance measurement; numerical outputs and original input hashes remain recorded.'))
print('OWN_BASELINE_331_SUSPENDED_FOREIGN_14904_UNCHANGED',flush=True)

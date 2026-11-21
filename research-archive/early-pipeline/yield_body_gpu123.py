"""Temporarily yield only the identified original control to another live job."""
import os
import signal
import time
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import OUT,process
from fit_gaussian_teacher_visible123 import active_gpu_pids

p=process(331)
assert p['start_identity']=='9229' and 'train_demo_body_upstream_offset123.py train' in p['command']
foreign=active_gpu_pids()-{331}
assert foreign,'Do not suspend without the observed resource conflict'
records=[process(pid) for pid in sorted(foreign)]
os.kill(331,signal.SIGSTOP)
for _ in range(50):
    if process(331)['state']=='T':break
    time.sleep(.05)
assert process(331)['state']=='T'
value=dict(updated=io.now(),state='own_baseline_suspended_for_observed_foreign_GPU',baseline_before=p,
    baseline_after=process(331),foreign=records,goal_paused=False,new_pilot_launched=False,
    scope='Temporary resource handoff only. Other jobs untouched. Resume the same331 only after foreign CUDA jobs end and identity9229 still matches.')
io.save(OUT/'original-import-teacher-contract/gpu-resource-handoff.json',value)
io.save(OUT/'teacher-import-surface-1000/launch-preflight.json',dict(updated=io.now(),
    state='not_launched_foreign_GPU_active',optimizer_updates=0,foreign=records,baseline_suspended=True))
print('OWN_BASELINE_YIELDED',value,flush=True)

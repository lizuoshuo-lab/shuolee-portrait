"""Resume the same temporary-yielded control only after other CUDA work ends."""
import os
import signal
import time
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import OUT,process
from fit_gaussian_teacher_visible123 import active_gpu_pids

before=process(331)
assert before['start_identity']=='9229'
assert 'train_demo_body_upstream_offset123.py train' in before['command']
assert before['state']=='T'
assert not active_gpu_pids()-{331}, 'Other GPU process remains active; nothing resumed'
os.kill(331,signal.SIGCONT)
for _ in range(30):
    if process(331)['state']!='T':break
    time.sleep(.02)
after=process(331)
assert after['start_identity']==before['start_identity'] and after['state'] not in ('T','Z')
io.save(OUT/'original-import-teacher-contract/gpu-resource-resume.json',
        dict(updated=io.now(),state='resumed_same_original_control',before=before,after=after,
             other_tasks_modified=False,new_training_process=False,goal_paused=False))
print('RESUMED_SAME_BASELINE',after,flush=True)

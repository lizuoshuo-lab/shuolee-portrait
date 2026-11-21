"""Local validation coordination only; never changes deployed training code."""
from pathlib import Path
import fcntl,json,os,signal,time

pid,owner=555,319
run=Path('/opt/gs-slam-lab/smpl-x-orientations-20261028/runs/portrait-32be30636adabfc4')
coord=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
receipt=Path('/mnt/d/Projects/Shuolee-validation-20261028/gpu-coordination.json')
assert 'run_smplx.py' in Path(f'/proc/{pid}/cmdline').read_text()
assert 'validate-orientations.py' in Path(f'/proc/{owner}/cmdline').read_text()
owner_identity=Path(f'/proc/{owner}/stat').read_text().split()[21]
paused=False
def state(value):
    data=dict(state=value,training_pid=pid,validation_pid=owner,updated=time.time(),lock='/tmp/shuolee-native122-gpu.lock')
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    current=json.loads(coord.read_text())
    current['current_local_validation']['gpu_coordination']=data
    coord.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    print(value,flush=True)
state('CPU_CAMERA_PREPARATION_GPU_WAITS_FOR_EXISTING_SEGMENTATION')
with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lock:
    try:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not Path(f'/proc/{pid}').exists():
                    raise RuntimeError('Training process ended before GPU coordination.')
                if (run/'data/inputs.json').exists() and not paused:
                    os.kill(pid,signal.SIGSTOP)
                    paused=True
                    state('PAUSED_AFTER_CAMERA_PREPARATION_WAITING_FOR_SEGMENTATION')
            time.sleep(.25)
        if paused:
            os.kill(pid,signal.SIGCONT)
            paused=False
        state('SMPLX_OWNS_LOCAL_GPU_LOCK')
        while Path(f'/proc/{owner}/stat').exists() and Path(f'/proc/{owner}/stat').read_text().split()[21]==owner_identity:
            time.sleep(2)
    finally:
        if paused and Path(f'/proc/{pid}').exists():os.kill(pid,signal.SIGCONT)
        fcntl.flock(lock,fcntl.LOCK_UN)
        state('LOCAL_GPU_LOCK_RELEASED')

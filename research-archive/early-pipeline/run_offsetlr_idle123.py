"""Finite launch after WSL restarted; no training/objective/source changes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import train_smplify_eight_offsetlr123 as candidate
from run_official_gaussians123 import read, save, sha, now

OUT = candidate.OUT


def gpu_pids():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       check=True, text=True, capture_output=True)
    return [int(x.strip()) for x in p.stdout.splitlines() if x.strip().isdigit()]


def guard():
    lease = read(OUT/'idle-gpu-lease.json')
    assert lease['state'] == 'running'
    assert os.getppid() == lease['supervisor_pid']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == lease['boot_id']
    assert not set(gpu_pids()) - {os.getpid()}, 'Unexpected GPU task; stop control'


def run():
    assert not (OUT/'gaussian-run').exists(), 'Never resume or repeat'
    plan = read(OUT/'plan.json')
    for path, expected in plan['source_hashes'].items():
        assert sha(path) == expected, path
    locks = [open(p, 'a+') for p in ('/tmp/shuolee-native122-gpu.lock', '/tmp/shuolee-body-export123.lock')]
    for handle in locks:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert gpu_pids() == [], 'GPU must be idle after restart'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError):
            continue
        assert not ('python' in cmd and (' train ' in cmd or cmd.rstrip().endswith(' train'))), cmd
    lease = dict(updated=now(), state='running', supervisor_pid=os.getpid(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 old_baseline_absent=True, baseline_resumed=False, timeout_seconds=360,
                 training_function_unchanged=True, wrapper_sha256=sha(__file__))
    save(OUT/'idle-gpu-lease.json', lease)
    save(OUT/'launch-amendment.json', dict(**lease,
        reason='Live WSL boot/process audit found historical PID331 absent. Run on verified idle GPU; do not recreate or signal stale PID. Only orchestration changes.',
        candidate_plan_sha256=sha(OUT/'plan.json')))
    start = time.monotonic()
    try:
        with open(OUT/'train.log', 'w', buffering=1) as log:
            subprocess.run([sys.executable, '-B', __file__, 'child'], check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=360)
        lease['outcome'] = 'completed'
    except BaseException as error:
        lease.update(outcome='failed_preserved', error=repr(error))
        raise
    finally:
        lease.update(updated=now(), state='released', seconds=time.monotonic()-start, gpu_pids_after=gpu_pids())
        save(OUT/'idle-gpu-lease.json', lease)
        print(json.dumps(lease), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'child':
        fn, _, _ = candidate.source()
        fn.__globals__['suspended_baseline'] = guard
        fn()
    else:
        run()

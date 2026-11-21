"""Resume the existing demo job after the explicitly completed GPU handoff."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import signal
import subprocess
import time

STATUS = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
RUN = Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo/upsampled3x')
OWN_PID, OWN_START = 24271, '5641422'
BASE_PID, BASE_START = 331, '9229'
EARLIER_PID, EARLIER_START = 24188, '5632383'


def now():
    return datetime.now(timezone.utc).isoformat()


def process(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return None
    try:
        text = (p / 'stat').read_text(); values = text[text.rfind(')') + 2:].split()
        return dict(pid=pid, state=values[0], start=values[19],
                    command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def update(**changes):
    data = json.loads(STATUS.read_text(encoding='utf-8-sig'))
    parent = data['static_person_upsampling']
    parent['demo_3x'].update(updated=now(), **changes)
    parent['own_gpu_jobs_running'] = changes.get('own_gpu_jobs_running', parent['own_gpu_jobs_running'])
    tmp = STATUS.with_name(STATUS.name + '.demo3x-runtime.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(STATUS)


def wait_stopped(pid, identity):
    for _ in range(200):
        p = process(pid)
        assert p and p['start'] == identity
        if p['state'] == 'T':
            return
        time.sleep(.05)
    raise RuntimeError('Process did not enter the stopped state.')


def main(release_confirmed):
    assert release_confirmed, 'Run only after the owner has released the previous GPU work including rendering.'
    earlier = process(EARLIER_PID)
    assert not earlier or earlier['start'] != EARLIER_START or earlier['state'] == 'Z', earlier
    current = process(OWN_PID)
    assert current and current['start'] == OWN_START and current['state'] == 'T'
    assert 'demo_static_upsample3x.py train' in current['command']
    # A resumed baseline is the only process this handoff is permitted to stop.
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    active = []
    for text in gpu.splitlines():
        if not text.strip().isdigit():
            continue
        p = process(int(text.strip()))
        if p and p['state'] not in ('T', 'Z') and p['pid'] not in (BASE_PID, OWN_PID):
            active.append(p)
    assert not active, active
    baseline = process(BASE_PID)
    assert baseline and baseline['start'] == BASE_START and 'train_demo_body_upstream_offset123.py train' in baseline['command']
    paused_baseline = False
    lease = dict(created=now(), owner_pid=OWN_PID, owner_start=OWN_START, baseline_before=baseline,
                 previous_candidate_release_confirmed=True, state='acquiring')
    evaluation = None
    try:
        if baseline['state'] != 'T':
            os.kill(BASE_PID, signal.SIGSTOP)
            paused_baseline = True
            wait_stopped(BASE_PID, BASE_START)
        lease.update(state='active', baseline_paused_by_this_handoff=paused_baseline)
        (RUN / 'gpu-lease.json').write_text(json.dumps(lease, indent=2) + '\n')
        os.kill(OWN_PID, signal.SIGCONT)
        update(state='training_resumed_after_gpu_handoff', process_state='R', own_gpu_jobs_running=True,
               resumed_at=now(), baseline_lease=lease, resource_monitor_pid=os.getpid())
        print('RESUMED_SAME_TRAINING_PROCESS', OWN_PID, OWN_START, flush=True)
        while True:
            current = process(OWN_PID)
            if not current or current['state'] == 'Z':
                break
            assert current['start'] == OWN_START, current
            progress = json.loads((RUN / 'progress.json').read_text())
            update(state='training', process_state=current['state'], current_step=progress['step'])
            print('TRAIN_STEP', progress['step'], flush=True)
            time.sleep(30)
        execution = json.loads((RUN / 'execution.json').read_text())
        result = json.loads((RUN / 'gaussian-run/result.json').read_text())
        assert execution['state'] == 'completed_pending_visual_comparison' and result['steps'] == 18000
        script = Path(__file__).with_name('review_demo_static_3x.py')
        python = '/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python'
        update(state='evaluating_three_resolutions', own_gpu_jobs_running=True, current_step=18000)
        with (RUN / 'evaluation.log').open('w') as log:
            evaluation = subprocess.Popen([python, '-B', str(script)], stdout=log, stderr=subprocess.STDOUT)
            update(evaluation_pid=evaluation.pid)
            ret = evaluation.wait()
        assert ret == 0, f'Evaluation failed with exit code {ret}; see evaluation.log'
        update(state='gpu_work_complete_pending_publication', own_gpu_jobs_running=False, own_live_process=False,
               evaluation_completed=True, current_step=18000)
        lease['outcome'] = 'training_and_comparison_completed'
    finally:
        if lease.get('outcome') != 'training_and_comparison_completed':
            if evaluation is not None and evaluation.poll() is None:
                evaluation.terminate()
                try:
                    evaluation.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    evaluation.kill(); evaluation.wait()
            current = process(OWN_PID)
            if current and current['start'] == OWN_START and current['state'] not in ('T', 'Z'):
                os.kill(OWN_PID, signal.SIGSTOP)
                wait_stopped(OWN_PID, OWN_START)
            update(state='interrupted_requires_local_recovery', own_gpu_jobs_running=False)
        # Never restore an unrelated replacement PID or override another live process state.
        baseline = process(BASE_PID)
        if paused_baseline and baseline and baseline['start'] == BASE_START and baseline['state'] == 'T':
            os.kill(BASE_PID, signal.SIGCONT)
            lease['baseline_restored_by_this_handoff'] = True
        else:
            lease['baseline_restored_by_this_handoff'] = False
        lease.update(state='released', released_at=now())
        (RUN / 'gpu-lease.json').write_text(json.dumps(lease, indent=2) + '\n')
        update(baseline_lease=lease, resource_monitor_finished=True)
        print('GPU_LEASE_RELEASED', json.dumps(lease), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-confirmed', action='store_true')
    args = parser.parse_args()
    main(args.release_confirmed)

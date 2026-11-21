"""Sequential pipeline: existing baseline -> SMPL full-scene check -> 30k fit."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB / 'outputs/dynamic-human-street-smpl-joint-011'
BASELINE = LAB / 'outputs/dynamic-human-street-joint-009/resume-10000-expandable'
LOGS = Path('/opt/gs-slam-lab/dynamic-human/logs')
BASELINE_PID = 345
BASELINE_SCRIPT = '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/resume_street_deformable.py'
SCRIPT = LAB / 'scripts/dynamic_human/train_street_smpl.py'


def status(state, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'pipeline.json'
    value = dict(status=state, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pipeline_pid=os.getpid(), **fields)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(target)
    print(json.dumps(value), flush=True)


def main():
    assert not (OUT / 'pipeline.json').exists(), 'Existing pipeline: inspect instead of duplicating it'
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True,garbage_collection_threshold:0.8'
    handle = Path(f'/proc/{BASELINE_PID}/cmdline')
    if handle.exists():
        assert BASELINE_SCRIPT.encode() in handle.read_bytes().split(b'\0'), 'PID belongs to a different command'
    status('waiting_for_existing_baseline_process_to_exit', baseline_pid=BASELINE_PID, timeout_seconds=7200)
    deadline = time.monotonic()+7200
    while handle.exists() and BASELINE_SCRIPT.encode() in handle.read_bytes().split(b'\0'):
        if time.monotonic() > deadline:
            raise TimeoutError('Baseline process has not exited; no duplicate GPU work started')
        time.sleep(15)
    completion = json.loads((BASELINE / 'completion.json').read_text())
    assert completion['optimizer_updates'] == 30000 and len(completion['views']) == 105
    assert len(completion['person_box_regions']) == 435
    # Stages use fresh processes so the smoke model and optimizer never seed training.
    for label, arguments in [('preflight', ['--preflight']), ('articulated', [])]:
        status(f'{label}_launching')
        log_path = LOGS / f'street-smpl-{label}.log'
        assert not log_path.exists(), 'Preserve previous attempts and logs'
        with log_path.open('w') as stream:
            child = subprocess.Popen([sys.executable, str(SCRIPT), *arguments], stdout=stream, stderr=subprocess.STDOUT)
            status(f'{label}_running', child_pid=child.pid, log=str(log_path))
            code = child.wait()
        if code != 0:
            raise RuntimeError(f'{label} exited {code}; inspect {log_path}')
        receipt = json.loads((OUT / label / 'completion.json').read_text())
        if label == 'preflight':
            assert receipt['status'] == 'full_scene_SMPL_joint_gradients_and_heldout_render_passed'
        else:
            assert receipt['optimizer_updates'] == 30000 and len(receipt['views']) == 105
    status('training_and_all105_heldout_renders_finished_independent_visual_audit_required')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_needs_inspection', error=repr(error))
        raise

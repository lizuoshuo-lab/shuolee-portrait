"""Isolated control-flow test. No real CUDA installation or training is performed."""
from pathlib import Path
import hashlib, json, os, subprocess, sys, tempfile, time

SOURCE = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/server113_train.py')
base = Path(tempfile.mkdtemp(prefix='shuolee-train-launch-test-'))
checks = []


def fixture(name):
    home = base/name
    paths = [home/'shuolee/logs', home/'shuolee/portrait113/scripts', home/'bin',
             home/'shuolee/datasets/hover175', home/'shuolee/models/113/smplx',
             home/'.local/share/shuolee-portrait/env-cu128/bin']
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
    (home/'shuolee/server113_train.py').write_bytes(SOURCE.read_bytes())
    (home/'shuolee/logs/server113-install-state.json').write_text('{"state":"completed"}')
    (home/'shuolee/cuda128.env').write_text('export CUDA_HOME=/fixture/toolkit\n')
    (home/'revision').write_text('5fda749')
    for rel in ['shuolee/datasets/hover175/training-inputs.json', 'shuolee/models/113/smplx/SMPLX_NEUTRAL.npz']:
        (home/rel).write_text('test fixture')
    (home/'bin/git').write_text('#!/bin/sh\ncat "$HOME/revision"\n')
    (home/'bin/nvidia-smi').write_text('#!/bin/sh\nif [ -f "$HOME/gpu-busy" ]; then echo "999, other-job"; fi\nexit 0\n')
    for p in (home/'bin').iterdir():
        p.chmod(0o755)
    python = home/'.local/share/shuolee-portrait/env-cu128/bin/python'
    python.symlink_to(sys.executable)
    fake = '''from pathlib import Path
import json,os,sys,time
h=Path.home(); stage=sys.argv[1]
with (h/'calls').open('a') as f:f.write(stage+'\\n')
assert os.environ['CUDA_HOME']=='/fixture/toolkit'
assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
if stage=='doctor' and (h/'fail-doctor').exists():raise SystemExit(7)
if stage=='all':
    root=Path(sys.argv[sys.argv.index('--run')+1]);root.mkdir()
    (root/'gaussian-run').mkdir();(root/'stages').mkdir();(root/'logs').mkdir()
    (root/'gaussian-run/progress.json').write_text(json.dumps(dict(fixture_run=root.name)))
    (root/'logs/train.log').write_text('FAKE_TRAIN_LOG_'+root.name+'\\n')
    time.sleep(2)
    (root/'gaussian-run/result.json').write_text(json.dumps(dict(steps=18000,from_scratch=True)))
    (root/'gaussian-run/delivery.json').write_text(json.dumps(dict(compatibility_passed=True)))
    (root/'stages/export.json').write_text(json.dumps(dict(state='completed')))
'''
    (home/'shuolee/portrait113/scripts/route113.py').write_text(fake)
    (home/'shuolee/runs').mkdir()
    env = dict(os.environ, HOME=str(home), PATH=str(home/'bin')+':'+os.environ['PATH'])
    return home, env


def call(home, env, mode, *extra):
    return subprocess.run([sys.executable, str(home/'shuolee/server113_train.py'), mode, *extra],
                          env=env, text=True, capture_output=True, timeout=15)


def finish(home, deadline=20):
    end=time.time()+deadline
    while time.time()<end:
        state=json.loads((home/'shuolee/logs/server113-train-state.json').read_text())
        if state['state'] not in ('starting','running'):
            return state
        time.sleep(.15)
    raise AssertionError('Worker timed out')


def fingerprint(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


h,e=fixture('success')
r=call(h,e,'--start');assert r.returncode==0,r.stderr
assert 'PIPELINE_STARTED' in r.stdout
r2=call(h,e,'--start');assert r2.returncode!=0 and 'ALREADY_RUNNING' in r2.stderr,r2
r3=call(h,e,'--start','--run-name','113_002')
assert r3.returncode!=0 and 'ALREADY_RUNNING' in r3.stderr,r3
assert not (h/'shuolee/runs/113_002').exists()
s=finish(h);assert s['state']=='completed' and not s['quality_accepted'],s
assert Path(s['run']).name=='113_001',s
assert (h/'calls').read_text().splitlines()==['audit','doctor','all']
assert call(h,e,'--status').returncode==0
assert 'FRESH_RUN_REQUIRED' in call(h,e,'--start').stderr
checks += ['noninteractive_detach', 'duplicate_rejected', 'explicit_python_no_activate',
           'audit_doctor_before_training', 'completed_markers', 'existing_run_not_overwritten',
           'default_run_001', 'cross_run_concurrent_rejected']
original=h/'shuolee/runs/113_001'
original_files=fingerprint(original)
original_state=dict(s)
r=call(h,e,'--start','--run-name','113_002');assert r.returncode==0,r.stderr
s=finish(h);assert s['state']=='completed' and Path(s['run']).name=='113_002',s
assert not s['quality_accepted']
assert fingerprint(original)==original_files,'Starting 002 changed preserved 001'
assert (h/'calls').read_text().splitlines()==['audit','doctor','all']*2
r=call(h,e,'--status');assert r.returncode==0,r.stderr
assert 'FAKE_TRAIN_LOG_113_002' in r.stdout and '"fixture_run": "113_002"' in r.stdout,r.stdout
assert 'FAKE_TRAIN_LOG_113_001' not in r.stdout and '"fixture_run": "113_001"' not in r.stdout,r.stdout
history=h/'shuolee/logs/server113-state-history'
assert any(json.loads(p.read_text())==original_state for p in history.glob('*.json')),'No exact old state archive'
assert 'FRESH_RUN_REQUIRED' in call(h,e,'--start','--run-name','113_002').stderr
checks += ['explicit_run_002_forwarded_to_worker', 'new_run_preserves_001_files',
           'status_uses_saved_run_002', 'previous_state_archived_exactly', 'explicit_existing_run_rejected']
h,e=fixture('invalid-run-name')
invalid_names=['../113_002','/tmp/113_002','113_000','113_1000','113_0001',
               '113_2','113_002/../113_003','113_002\\..\\113_003','113_002 ','112_002']
for name in invalid_names:
    r=call(h,e,'--start','--run-name',name)
    assert r.returncode!=0,(name,r.stdout)
    assert not (h/'calls').exists(),name
    assert not (h/'shuolee/logs/server113-train-state.json').exists(),name
    assert not list((h/'shuolee/runs').iterdir()),name
assert call(h,e,'--status','--run-name','113_999').returncode==0
checks += ['run_name_strict_range_001_to_999', 'run_name_path_traversal_rejected_before_launch']
h,e=fixture('doctor-failure');(h/'fail-doctor').touch()
assert call(h,e,'--start').returncode==0
s=finish(h);assert s['state']=='failed' and s['stage']=='doctor',s
assert (h/'calls').read_text().splitlines()==['audit','doctor']
assert not (h/'shuolee/runs/113_001').exists()
checks += ['doctor_failure_blocks_training']
h,e=fixture('incomplete-install');(h/'shuolee/logs/server113-install-state.json').write_text('{"state":"running"}')
assert 'INSTALL_NOT_COMPLETED' in call(h,e,'--start').stderr
h,e=fixture('wrong-revision');(h/'revision').write_text('bad0000')
assert 'Wrong Git revision' in call(h,e,'--start').stderr
h,e=fixture('busy-gpu');(h/'gpu-busy').touch()
assert 'GPU_0_BUSY' in call(h,e,'--start').stderr
checks += ['incomplete_install_rejected','wrong_revision_rejected','busy_gpu_rejected']
report=dict(passed=True,checks=checks,fixture=str(base),
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            python_version=sys.version,python_executable=sys.executable,
            target_server_python='3.10',target_server_runtime_tested=False,
            real_cuda_or_training_tested=False,server_commands_executed=False)
Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap/training-launcher-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

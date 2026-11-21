"""Isolated Linux scheduling tests; CUDA/compiler installation is deliberately simulated."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,time
PACKAGE=Path('/mnt/d/00/工作/SMPL-X/server113_pip_repair.pyz')
root=Path(tempfile.mkdtemp(prefix='shuolee-install-background-test-'))
home=root/'home';home.mkdir();target=home/'.local/share/shuolee-portrait/env-cu128'
subprocess.run([sys.executable,'-m','venv','--without-pip',str(target)],check=True)
(target/'bin/activate').unlink() # Explicitly reproduce the user condition, only in this temporary fixture.
project=home/'shuolee/portrait113';(project/'scripts').mkdir(parents=True)
(home/'shuolee/cuda128.env').write_text('export CUDA_HOME=/test-only-not-a-real-toolkit\n')
(project/'scripts/setup_server.sh').write_text('set -eu\nsleep 3\necho SIMULATED_SETUP_NO_CUDA_COMPILE\n')
(project/'scripts/setup_113.sh').write_text('set -eu\necho SIMULATED_113_IMPORT_NO_GPU\n')
binpath=root/'bin';binpath.mkdir();git=binpath/'git';git.write_text('#!/bin/sh\necho 5fda749\n');git.chmod(0o755)
env=dict(os.environ,HOME=str(home),PATH=str(binpath)+':'+os.environ['PATH'])
logs=home/'shuolee/logs'
base=[sys.executable,str(PACKAGE)]
def run(*args):return subprocess.run(base+list(args),env=env,stdin=subprocess.DEVNULL,text=True,capture_output=True)
def wait_state(expected):
    for _ in range(120):
        p=logs/'server113-install-state.json'
        state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('state')==expected:return state
        if state.get('state')=='failed' and expected!='failed':raise AssertionError(state)
        time.sleep(.25)
    raise AssertionError(('timeout',state))
r=run('--install-background');assert r.returncode==0 and 'INSTALL_STARTED PID=' in r.stdout,(r.stdout,r.stderr)
r2=run('--install-background');assert r2.returncode==0 and 'INSTALL_ALREADY_RUNNING' in r2.stdout,(r2.stdout,r2.stderr)
done=wait_state('completed');assert done['training_started'] is False
assert not (target/'bin/activate').exists()
check=subprocess.run([str(target/'bin/python'),'-I','-m','pip','--version'],text=True,capture_output=True,check=True)
assert '25.1.1' in check.stdout
status=run('--status');assert 'completed' in status.stdout and 'DEPENDENCY_INSTALL_FINISHED_CONTINUE_GUIDE_STEP_5' in status.stdout
# The completion marker is not retained as current success after a later failing run.
(project/'scripts/setup_server.sh').write_text('echo SIMULATED_FAILURE\nexit 7\n')
r=run('--install-background');assert r.returncode==0
failed=wait_state('failed');assert 'returned non-zero exit status 7' in failed['error'],failed
git.write_text('#!/bin/sh\necho abcdef0\n')
r=run('--install-background');assert r.returncode==0
for _ in range(120):
    state=json.loads((logs/'server113-install-state.json').read_text())
    if state.get('state')=='failed' and 'Expected frozen Git' in state.get('error',''):break
    time.sleep(.25)
else:raise AssertionError(state)
report=dict(passed=True,fixture_root=str(root),checks=['offline_pip_without_activate','noninteractive_detached_launch','duplicate_installer_lock','explicit_python_without_activate','completed_state','installer_failure_propagation','wrong_git_revision_rejected'],
            limitations='Dependency scripts and git revision output are controlled test fixtures; no CUDA build, company-server execution or training was tested.')
out=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap/background-validation.json')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

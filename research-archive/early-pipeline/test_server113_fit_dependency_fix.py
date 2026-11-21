"""Fake-HOME Linux tests plus a real CPU-only sole-index control.

Reads the licensed fixed template; never calls fit, CUDA, training or a server.
"""
from pathlib import Path
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().with_name('server113_fit_dependency_fix.py')
PEER = SOURCE.with_name('server113_head_path_fix.py')
REPO = Path('/mnt/d/00/工作/人体重建')
MODEL = Path('/mnt/d/GS-SLAM-Data/shuolee-113-assets/smplx/SMPLX_NEUTRAL.npz')
RECEIPT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap/fit-dependency-validation.json')
base = Path(tempfile.mkdtemp(prefix='shuolee-fit-dependency-test-'))
spec = importlib.util.spec_from_file_location('fit_fix_under_test', SOURCE)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
spec = importlib.util.spec_from_file_location('head_fix_for_fixture', PEER)
peer = importlib.util.module_from_spec(spec); spec.loader.exec_module(peer)
assert hashlib.sha256(PEER.read_bytes()).hexdigest() == helper.PEER_SHA256
payload = base64.b64decode(helper.PAYLOAD_B64, validate=True)
assert len(payload) == 2860 and hashlib.sha256(payload).hexdigest() == helper.DEPENDENCY_SHA256
real_git = shutil.which('git')
blobs = {}
for relative in [*peer.SOURCES, helper.FIT_PATH, 'src/avatar_core/shuolee_common.py']:
    data = subprocess.check_output([real_git, '-c', 'safe.directory=*', '-C', str(REPO),
                                   'show', peer.COMMIT+':'+relative])
    assert b'\r\n' not in data, relative
    blobs[relative] = data
assert hashlib.sha256(blobs[helper.FIT_PATH]).hexdigest() == helper.FIT_SHA256
for relative, expected in peer.SOURCES.items():
    assert hashlib.sha256(blobs[relative]).hexdigest() == expected
checks = ['peer_identity_and_git_lf_fixtures_verified', 'original_2860_byte_payload_sha_verified']


def fixture(name):
    home = base/name
    project = home/'shuolee/portrait113'
    for relative, data in blobs.items():
        path = project/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (project/peer.ALIAS).symlink_to(peer.RELATIVE_TARGET, target_is_directory=True)
    (home/'bin').mkdir()
    (home/'revision').write_text(peer.COMMIT)
    (home/'git-top').write_text(str(project))
    git = home/'bin/git'
    git.write_text('''#!/bin/sh
case "$*" in
  *"--show-toplevel"*) cat "$HOME/git-top" ;;
  *"rev-parse HEAD"*) cat "$HOME/revision" ;;
  *) echo 'unexpected fake git call' >&2; exit 7 ;;
esac
''')
    git.chmod(0o755)
    for run_name in ['113_001', '113_002']:
        run = home/'shuolee/runs'/run_name; run.mkdir(parents=True)
        (run/'preserve-original.log').write_text('Preserve failed '+run_name+'\n')
    env_python = home/'.local/share/shuolee-portrait/env-cu128/bin/python'
    env_python.parent.mkdir(parents=True); env_python.symlink_to(sys.executable)
    # Preserve the existing venv's package lookup when invoking the symlink -I.
    # These fixture files only reference the already installed local environment.
    fake_env = env_python.parent.parent
    (fake_env/'pyvenv.cfg').write_bytes((Path(sys.prefix)/'pyvenv.cfg').read_bytes())
    fake_lib = fake_env/'lib/python3.10'; fake_lib.mkdir(parents=True)
    (fake_lib/'site-packages').symlink_to(Path(sys.prefix)/'lib/python3.10/site-packages', target_is_directory=True)
    model = home/'shuolee/models/113/smplx/SMPLX_NEUTRAL.npz'
    model.parent.mkdir(parents=True); model.symlink_to(MODEL)
    (home/'shuolee/server113_head_path_fix.py').write_bytes(PEER.read_bytes())
    (home/'shuolee/server113_fit_dependency_fix.py').write_bytes(SOURCE.read_bytes())
    env = dict(os.environ, HOME=str(home), PATH=str(home/'bin')+':'+os.environ['PATH'])
    # Exercise the production isolation and environment scrubbing.
    env['PYTHONPATH'] = str(home/'hostile-pythonpath')
    env['SHUOLEE_EXTRA_PYTHONPATH'] = str(home/'hostile-extra-pythonpath')
    return home, project, env


def call(home, env, *extra):
    return subprocess.run([sys.executable, str(home/'shuolee/server113_fit_dependency_fix.py'), *extra],
                          env=env, text=True, capture_output=True, timeout=60)


def snapshot(root):
    result = {}
    for path in root.rglob('*'):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            result[rel] = {'link':os.readlink(path)}
        elif path.is_file():
            result[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            result[rel] = {'directory':True}
    return result


def preserved(home, project):
    assert {rel:hashlib.sha256((project/rel).read_bytes()).hexdigest() for rel in blobs} == {
        rel:hashlib.sha256(data).hexdigest() for rel, data in blobs.items()}
    for run_name in ['113_001','113_002']:
        assert (home/'shuolee/runs'/run_name/'preserve-original.log').read_text() == 'Preserve failed '+run_name+'\n'


def rejected(home, env, message=None):
    r = call(home, env, '--apply')
    assert r.returncode == 2 and 'FIT_DEPENDENCY_FIX_STOPPED' in r.stderr, r
    if message:
        assert message in r.stderr, r.stderr
    return r


h, p, e = fixture('first-apply-idempotent')
before = snapshot(h)
r = call(h, e)
assert r.returncode == 0 and 'FIT_DEPENDENCY_CHECK_ONLY' in r.stdout, r
assert snapshot(h) == before, 'Default check wrote files in fake HOME'
checks += ['default_check_is_completely_read_only']
r = call(h, e, '--apply')
assert r.returncode == 0 and 'FIT_DEPENDENCY_FIX_READY' in r.stdout, r
target = p/helper.DEPENDENCY
assert target.read_bytes() == payload
assert len(target.read_bytes()) == 2860 and hashlib.sha256(target.read_bytes()).hexdigest() == helper.DEPENDENCY_SHA256
assert not target.is_symlink()
report_path = h/'shuolee/logs/server113-fit-dependency-fix.json'
report = json.loads(report_path.read_text())
assert report['state'] == 'completed' and report['dependency_created'] is True, report
assert report['dependency_in_frozen_git'] is False and report['historical113_dependency_hash_recorded'] is False
assert report['training_started'] is False and report['existing_runs_modified'] is False
assert report['tracked_sources_changed'] is False
control = report['cpu_probe']
assert control['state'] == 'FIT_DEPENDENCY_CPU_PROBE_PASSED'
assert control['sole_indices']['left'] == dict(vertices=99, indices_sha256='e641e25e68f77a0d196244b14426f5af61044af9b6c52f3653abcc77dc4f7cba')
assert control['sole_indices']['right'] == dict(vertices=99, indices_sha256='4ffd62b8ab3bc58de0c22909b25c18c9bdf0f7cc92fc7d010fe0a9ab413de04a')
assert control['cuda_executed'] is False and control['optimizer_updates'] == 0
assert 'assert \'torch\' not in sys.modules' in helper.CPU_PROBE
preserved(h, p)
assert not list(p.rglob('__pycache__')), 'CPU probe wrote bytecode in project'
checks += ['first_apply_restores_original_bytes_and_sha', 'real_cpu_original_sole_indices_left_99_right_99',
           'both_sole_index_hashes_match', 'cpu_probe_does_not_import_torch_or_cuda',
           'tracked_sources_and_runs_001_002_preserved', 'cpu_probe_creates_no_project_bytecode',
           'receipt_labels_untracked_supplement_and_no_historical_hash_claim']
inode = target.stat().st_ino
r = call(h, e, '--apply'); assert r.returncode == 0, r
assert target.stat().st_ino == inode and target.read_bytes() == payload
assert json.loads(report_path.read_text())['dependency_created'] is False
preserved(h, p)
checks += ['second_apply_idempotent_with_cpu_recheck']

for kind in ['missing-peer', 'modified-peer', 'wrong-git', 'wrong-git-top', 'missing-head-alias']:
    h, p, e = fixture(kind)
    peer_path = h/'shuolee/server113_head_path_fix.py'
    if kind == 'missing-peer': peer_path.unlink()
    elif kind == 'modified-peer': peer_path.write_bytes(peer_path.read_bytes()+b'\n# modified\n')
    elif kind == 'wrong-git': (h/'revision').write_text('0'*40)
    elif kind == 'wrong-git-top': (h/'git-top').write_text(str(h/'shuolee'))
    else: (p/peer.ALIAS).unlink()
    rejected(h, e)
    assert not (p/helper.DEPENDENCY).exists()
    preserved(h, p)
    checks.append(kind.replace('-', '_')+'_rejected')

for relative in [helper.FIT_PATH, *peer.SOURCES]:
    h, p, e = fixture('modified-'+Path(relative).name)
    path = p/relative; path.write_bytes(path.read_bytes()+b'\n# modified fixture\n')
    before = path.read_bytes()
    rejected(h, e)
    assert path.read_bytes() == before
    assert not (p/helper.DEPENDENCY).exists()
    checks.append('modified_'+Path(relative).name+'_rejected')

for kind in ['unknown-file', 'directory', 'same-byte-symlink', 'broken-symlink']:
    h, p, e = fixture(kind); target = p/helper.DEPENDENCY
    if kind == 'unknown-file': target.write_text('Preserve unknown file\n')
    elif kind == 'directory': target.mkdir(); (target/'preserve.txt').write_text('keep')
    elif kind == 'same-byte-symlink':
        outside = h/'original-helper.py'; outside.write_bytes(payload); target.symlink_to(outside)
    else: target.symlink_to(h/'missing-helper.py')
    inode = target.lstat().st_ino
    before = snapshot(p)
    rejected(h, e)
    assert inode == target.lstat().st_ino and snapshot(p) == before
    preserved(h, p)
    checks.append(kind.replace('-', '_')+'_never_overwritten')

for lock_name in ['server113-install.lock', 'server113-train.lock']:
    h, p, e = fixture('held-'+lock_name)
    logs = h/'shuolee/logs'; logs.mkdir()
    with (logs/lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        rejected(h, e, 'INSTALL_OR_TRAIN_ALREADY_RUNNING')
    assert not (p/helper.DEPENDENCY).exists()
    preserved(h, p)
    checks.append('held_'+lock_name+'_rejects_apply')

h, p, e = fixture('running-project-process')
sleeping = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', str(p/'scripts/route113.py')], env=e)
try:
    rejected(h, e, 'Project processes are still running')
    assert not (p/helper.DEPENDENCY).exists()
finally:
    sleeping.terminate(); sleeping.wait(timeout=5)
preserved(h, p)
checks += ['running_project_process_rejects_apply']

h, p, e = fixture('wrong-model-sha')
model = h/'shuolee/models/113/smplx/SMPLX_NEUTRAL.npz'
model.unlink(); model.write_bytes(b'wrong fixture model')
rejected(h, e, 'CPU dependency probe failed')
assert (p/helper.DEPENDENCY).read_bytes() == payload
report = json.loads((h/'shuolee/logs/server113-fit-dependency-fix.json').read_text())
assert report['state'] == 'cpu_probe_failed' and report['probe_returncode'] != 0
assert 'Unexpected SMPL-X template SHA256' in Path(report['probe_log']).read_text()
assert report['training_started'] is False
preserved(h, p)
checks += ['wrong_model_sha_stops_cpu_probe_and_preserves_added_source_and_error_log']

receipt = dict(passed=True, checks=checks, fixture=str(base),
               source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), peer_sha256=helper.PEER_SHA256,
               python_version=sys.version, python_executable=sys.executable,
               target_server_python='3.10', target_server_runtime_tested=False,
               baseline_git=peer.COMMIT, fixed_fitter_sha256=helper.FIT_SHA256,
               dependency_sha256=helper.DEPENDENCY_SHA256, dependency_bytes=len(payload),
               cpu_template_read_only=str(MODEL), cpu_original_sole_indices=control['sole_indices'],
               real_fit_tested=False, real_cuda_or_training_tested=False,
               real_project_modified=False, server_commands_executed=False)
RECEIPT.parent.mkdir(parents=True, exist_ok=True)
RECEIPT.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))

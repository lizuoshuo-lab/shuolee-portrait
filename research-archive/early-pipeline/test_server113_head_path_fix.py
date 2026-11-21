"""Isolated Linux source-path repair tests, with Git-LF fixtures and fake Git.

No CUDA, dependency install, real project mutation, model import or training.
"""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().with_name('server113_head_path_fix.py')
REPO = Path('/mnt/d/00/工作/人体重建')
RECEIPT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap/head-source-path-validation.json')
base = Path(tempfile.mkdtemp(prefix='shuolee-head-path-test-'))
spec = importlib.util.spec_from_file_location('head_path_fix_under_test', SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
real_git = shutil.which('git')
blobs = {}
for relative, expected in helper.SOURCES.items():
    data = subprocess.check_output([real_git, '-c', 'safe.directory=*', '-C', str(REPO),
                                    'show', helper.COMMIT+':'+relative])
    assert hashlib.sha256(data).hexdigest() == expected, relative
    assert b'\r\n' not in data, relative
    blobs[relative] = data
checks = ['source_fixtures_from_frozen_git_blobs_with_lf_sha256']


def fixture(name):
    home = base/name
    project = home/'shuolee/portrait113'
    for relative, data in blobs.items():
        path = project/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (home/'bin').mkdir()
    (home/'revision').write_text(helper.COMMIT)
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
    run = home/'shuolee/runs/113_001'
    run.mkdir(parents=True)
    (run/'preserve-original.log').write_text('Original failed run must be preserved.\n')
    (home/'shuolee/server113_head_path_fix.py').write_bytes(SOURCE.read_bytes())
    env = dict(os.environ, HOME=str(home), PATH=str(home/'bin')+':'+os.environ['PATH'])
    return home, project, env


def call(home, env, *extra):
    return subprocess.run([sys.executable, str(home/'shuolee/server113_head_path_fix.py'), *extra],
                          env=env, text=True, capture_output=True, timeout=15)


def source_hashes(project):
    return {rel: hashlib.sha256((project/rel).read_bytes()).hexdigest() for rel in blobs}


def rejected(home, project, env, message=None):
    r = call(home, env, '--apply')
    assert r.returncode == 2 and 'HEAD_SOURCE_PATH_FIX_STOPPED' in r.stderr, r
    if message:
        assert message in r.stderr, r.stderr
    assert not (home/'shuolee/logs/server113-head-path-fix.json').exists()
    assert (home/'shuolee/runs/113_001/preserve-original.log').read_text() == 'Original failed run must be preserved.\n'
    return r


h, p, e = fixture('check-apply-idempotent')
before = source_hashes(p)
alias = p/helper.ALIAS
r = call(h, e)
assert r.returncode == 0 and 'HEAD_SOURCE_PATH_FIX_NEEDED' in r.stdout, r
assert not alias.is_symlink() and not alias.exists()
assert not (h/'shuolee/logs').exists(), 'Default check wrote logs/locks'
assert source_hashes(p) == before
checks += ['default_check_is_read_only_and_reports_missing_alias']
r = call(h, e, '--apply')
assert r.returncode == 0 and 'HEAD_SOURCE_PATH_FIX_READY' in r.stdout, r
assert alias.is_symlink() and os.readlink(alias) == '../../avatar_core/flame_model'
assert alias.resolve() == (p/helper.CANONICAL).resolve()
assert source_hashes(p) == before
report_path = h/'shuolee/logs/server113-head-path-fix.json'
report = json.loads(report_path.read_text())
assert report['alias_created'] is True and report['alias_exists'] is True, report
assert report['source_sha256'] == before
assert report['tracked_sources_changed'] is False and report['existing_runs_modified'] is False
assert report['training_started'] is False
assert (h/'shuolee/runs/113_001/preserve-original.log').read_text() == 'Original failed run must be preserved.\n'
for filename in ['flame.py', 'lbs.py']:
    assert (alias/filename).read_bytes() == blobs[helper.CANONICAL+'/'+filename]
checks += ['apply_adds_one_relative_alias', 'alias_resolves_to_canonical_sources',
           'source_sha256_and_failed_run_unchanged', 'receipt_records_no_training_or_source_change']
before_inode = alias.lstat().st_ino
before_receipt = report_path.read_bytes()
r = call(h, e); assert r.returncode == 0 and 'HEAD_SOURCE_PATH_FIX_READY' in r.stdout, r
assert report_path.read_bytes() == before_receipt
r = call(h, e, '--apply'); assert r.returncode == 0, r
assert alias.lstat().st_ino == before_inode
assert json.loads(report_path.read_text())['alias_created'] is False
checks += ['correct_existing_alias_check_read_only', 'apply_is_idempotent']

for kind in ['regular-file', 'directory', 'wrong-link', 'broken-link']:
    h, p, e = fixture(kind)
    alias = p/helper.ALIAS
    if kind == 'regular-file':
        alias.write_text('preserve existing file')
    elif kind == 'directory':
        alias.mkdir(); (alias/'preserve.txt').write_text('preserve existing directory')
    elif kind == 'wrong-link':
        (h/'elsewhere').mkdir(); alias.symlink_to(h/'elsewhere', target_is_directory=True)
    else:
        alias.symlink_to(h/'missing', target_is_directory=True)
    inode = alias.lstat().st_ino
    before = source_hashes(p)
    rejected(h, p, e)
    assert alias.lstat().st_ino == inode
    assert source_hashes(p) == before
    checks.append(kind.replace('-', '_')+'_rejected_without_overwrite')

h, p, e = fixture('wrong-commit'); (h/'revision').write_text('0'*40)
rejected(h, p, e, 'exact frozen 113 project and commit')
assert not (p/helper.ALIAS).exists()
h, p, e = fixture('short-commit'); (h/'revision').write_text('5fda749')
rejected(h, p, e, 'exact frozen 113 project and commit')
h, p, e = fixture('wrong-git-root'); (h/'git-top').write_text(str(h/'shuolee'))
rejected(h, p, e, 'exact frozen 113 project and commit')
checks += ['wrong_full_commit_rejected', 'short_commit_not_accepted_as_full_identity', 'wrong_git_toplevel_rejected']

for relative in blobs:
    h, p, e = fixture('modified-'+Path(relative).name)
    changed = p/relative
    changed.write_bytes(changed.read_bytes()+b'\n# changed fixture\n')
    before = changed.read_bytes()
    rejected(h, p, e, 'Source identity differs')
    assert changed.read_bytes() == before
    assert not (p/helper.ALIAS).exists()
    checks.append('modified_'+Path(relative).name+'_rejected')
h, p, e = fixture('working-tree-crlf')
flame = p/(helper.CANONICAL+'/flame.py')
flame.write_bytes(flame.read_bytes().replace(b'\n', b'\r\n'))
rejected(h, p, e, 'Source identity differs')
checks += ['crlf_working_tree_is_rejected_instead_of_silently_normalized']

h, p, e = fixture('source-symlink')
relative = helper.CANONICAL+'/flame.py'
outside = h/'source-copy.py'; outside.write_bytes(blobs[relative])
(p/relative).unlink(); (p/relative).symlink_to(outside)
rejected(h, p, e, 'Unexpected source location')
assert (p/relative).is_symlink()
checks += ['symlink_source_rejected']
h, p, e = fixture('project-symlink')
real = p.with_name('portrait113-real'); p.rename(real); p.symlink_to(real, target_is_directory=True)
rejected(h, p, e, 'Project root is a symbolic link')
assert p.is_symlink()
checks += ['symlink_project_root_rejected']

for lock_name in ['server113-install.lock', 'server113-train.lock']:
    h, p, e = fixture('held-'+lock_name)
    logs = h/'shuolee/logs'; logs.mkdir()
    with (logs/lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        rejected(h, p, e, 'INSTALL_OR_TRAIN_ALREADY_RUNNING')
    assert not (p/helper.ALIAS).exists()
    checks.append('held_'+lock_name+'_rejects_apply')

h, p, e = fixture('running-project-process')
# The sleeping Python only carries a fake argv path; route113 is never executed.
sleeping = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)',
                             str(p/'scripts/route113.py')], env=e)
try:
    rejected(h, p, e, 'Project processes are still running')
    assert not (p/helper.ALIAS).exists()
finally:
    sleeping.terminate(); sleeping.wait(timeout=5)
checks += ['running_route_process_guard_rejects_apply']

receipt = dict(passed=True, checks=checks, fixture=str(base),
               source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               python_version=sys.version, python_executable=sys.executable,
               target_server_python='3.10', target_server_runtime_tested=False,
               baseline_git=helper.COMMIT, git_lf_fixture_sha256=helper.SOURCES,
               real_cuda_or_training_tested=False, real_model_import_tested=False,
               real_project_modified=False, server_commands_executed=False)
RECEIPT.parent.mkdir(parents=True, exist_ok=True)
RECEIPT.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))

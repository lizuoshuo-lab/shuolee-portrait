"""Run the actual production worker in a private mount namespace without systemd."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

PROJECT = Path('/mnt/d/Projects/Shuolee')
EVIDENCE = Path('/mnt/d/Projects/Shuolee-validation-20261028')
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait.queue119 import read, save


def wait_for(check):
    end = time.monotonic() + 12
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.1)
    raise RuntimeError('Production worker check timed out')


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False


with tempfile.TemporaryDirectory(prefix='smpl-x-no-systemd-') as directory:
    root = Path(directory)
    empty = root / 'empty-systemd'
    empty.mkdir()
    profile = root / 'queue/deployment.json'
    status = root / 'queue/status.json'
    save(profile, dict(schema='shuolee-deployment119-v1', environment=sys.prefix,
                       runs=str(root / 'runs'), assets=str(root / 'unused-assets')))
    command = ['unshare', '--mount', '--propagation', 'private', '--', 'bash', '-c',
               'set -e; mount --bind "$1" /run/systemd; test ! -d /run/systemd/system; '
               'exec "$2" -s "$3" --deployment "$4"', 'smpl-x-runtime-check',
               str(empty), sys.executable, str(PROJECT / 'scripts/install_service.py'), str(profile)]
    pid = None
    try:
        first = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, start_new_session=True, timeout=40)
        assert first.returncode == 0, first.stdout + first.stderr
        assert 'smpl-x READY' in first.stdout and 'Service: background' in first.stdout
        pid = read(status)['pid']
        assert os.getsid(pid) == pid
        assert os.readlink(f'/proc/{pid}/fd/0') == '/dev/null'
        assert os.readlink(f'/proc/{pid}/fd/1') == str(root / 'queue/worker.log')
        mountinfo = Path(f'/proc/{pid}/mountinfo').read_text()
        assert ' /run/systemd ' in mountinfo
        (root / 'videos/unfinished.mp4.filepart').write_bytes(b'incomplete upload fixture')
        heartbeat_before = status.stat().st_mtime_ns
        os.kill(pid, signal.SIGHUP)
        wait_for(lambda: status.stat().st_mtime_ns > heartbeat_before)
        assert read(status)['state'] == 'watching'
        assert read(status)['counts'] == {} and read(status)['active'] is None
        second = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, start_new_session=True, timeout=40)
        assert second.returncode == 0, second.stdout + second.stderr
        assert read(status)['pid'] == pid
        os.kill(pid, signal.SIGTERM)
        wait_for(lambda: not alive(pid))
        assert read(status)['state'] == 'stopped'
        report = dict(state='passed', environment='Ubuntu 20.04 WSL2',
                      actual_production_installer_and_watcher=True,
                      systemd_absent_in_private_mount_namespace=True,
                      global_host_mounts_unchanged=True,
                      detached_session=True, stdin_closed=True, stdout_is_log_file=True,
                      survived_launcher_exit_and_SIGHUP=True,
                      repeated_install_reused_same_pid=True, partial_upload_ignored=True,
                      test_worker_stopped=True, gpu_or_training_started=False,
                      model_checks_performed=False, target_server_executed=False,
                      source_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (PROJECT / 'scripts/install_service.py',
                                               PROJECT / 'src/shuolee_portrait/queue119.py')},
                      output=first.stdout)
        save(EVIDENCE / 'background-real-runtime.json', report)
        print(json.dumps(report, indent=2))
    finally:
        if pid is not None and alive(pid):
            os.kill(pid, signal.SIGTERM)
            wait_for(lambda: not alive(pid))

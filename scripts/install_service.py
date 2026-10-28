"""Start video intake using systemd when available, otherwise a detached process."""
import argparse
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait.queue119 import VideoQueue, read, worker_lock


def quoted(value):
    value = str(value)
    if any(c in value for c in '\r\n\0'):
        raise ValueError('Invalid newline in service setting.')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def service_text(profile):
    user = pwd.getpwuid(os.getuid())
    runtime_environment = ['Environment=' + quoted(key + '=' + os.environ[key])
                           for key in ('PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'TORCH_EXTENSIONS_DIR')
                           if os.environ.get(key)]
    return '\n'.join([
        '[Unit]', 'Description=smpl-x video reconstruction',
        'After=network-online.target', 'Wants=network-online.target', '',
        '[Service]', 'Type=simple', 'User=' + user.pw_name,
        'Environment=' + quoted('HOME=' + user.pw_dir),
        'Environment=PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4',
        *runtime_environment,
        'UnsetEnvironment=PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH',
        'ExecStart=' + ' '.join(map(quoted, [sys.executable, '-s', PROJECT/'watch.py', '--deployment', profile])),
        'Restart=on-failure', 'RestartSec=5', 'KillMode=mixed', 'TimeoutStopSec=90',
        'UMask=0027', '', '[Install]', 'WantedBy=multi-user.target', '',
    ])


def systemd_available():
    if not Path('/run/systemd/system').is_dir() or not shutil.which('systemctl'):
        return False
    try:
        return subprocess.run(['systemctl', 'show', '--property=Version', '--value'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def worker_status(queue):
    try:
        return read(queue.root / 'status.json')
    except (OSError, ValueError):
        return {}


def ready(queue, status):
    return (status.get('state') == 'watching'
            and time.time() - (queue.root / 'status.json').stat().st_mtime < 15)


def matching_worker(pid, profile):
    # A stale status file must never identify an unrelated, reused PID as our worker.
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
        expected = ['-s', str(PROJECT / 'watch.py'), '--deployment', str(profile)]
        return args[1:] == [os.fsencode(value) for value in expected]
    except OSError:
        return False


def report_ready(queue, backend):
    print(f'smpl-x READY\nUpload videos: {queue.inbox}\nResults: {queue.runs}\n'
          f'Status: {queue.root / "status.json"}\nService: {backend}', flush=True)


def start_background(queue, profile):
    log_path = queue.root / 'worker.log'
    with worker_lock(queue.root / 'install.lock'):
        status = worker_status(queue)
        if matching_worker(status.get('pid'), profile):
            if ready(queue, status):
                report_ready(queue, f'background PID={status["pid"]}; log={log_path}')
                return
            raise RuntimeError(f'Existing worker has not reported ready; preserved. Read: {log_path}')
        # Also detects a live worker before its first heartbeat, or a surviving training child.
        with worker_lock(queue.root / 'worker.lock'):
            pass
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'SHUOLEE_EXTRA_PYTHONPATH'):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with log_path.open('ab', buffering=0) as log:
            child = subprocess.Popen(
                [sys.executable, '-s', str(PROJECT / 'watch.py'), '--deployment', str(profile)],
                cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, umask=0o027)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f'Background worker exited with status {child.returncode}. Read: {log_path}')
            status = worker_status(queue)
            if status.get('pid') == child.pid and ready(queue, status):
                report_ready(queue, f'background PID={child.pid}; log={log_path}')
                return
            time.sleep(0.1)
        raise RuntimeError(f'Background worker has not reported ready. Read: {log_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='smpl-x')
    parser.add_argument('--deployment', type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.name):
        parser.error('Invalid service name')
    profile = args.deployment.resolve()
    queue = VideoQueue(profile)
    if not systemd_available():
        start_background(queue, profile)
        return
    target = Path('/etc/systemd/system') / (args.name + '.service')
    unit = service_text(profile)
    admin = [] if os.getuid() == 0 else ['sudo']
    if target.exists() and target.read_text() != unit:
        old = target.read_text()
        expected = next(line for line in unit.splitlines() if line.startswith('ExecStart='))
        active = subprocess.run(['systemctl', 'is-active', '--quiet', target.name]).returncode == 0
        if expected not in old.splitlines() or active:
            raise RuntimeError(f'Existing service uses different settings or is active: {target}. Existing training was preserved.')
    with tempfile.TemporaryDirectory(prefix='shuolee-service-') as temp:
        source = Path(temp) / target.name
        source.write_text(unit, encoding='utf-8')
        subprocess.run(['systemd-analyze', 'verify', str(source)], check=True)
        subprocess.run([*admin, 'install', '-m', '644', str(source), str(target)], check=True)
    subprocess.run([*admin, 'systemctl', 'daemon-reload'], check=True)
    subprocess.run([*admin, 'systemctl', 'enable', '--now', target.name], check=True)
    for _ in range(30):
        active = subprocess.run(['systemctl', 'is-active', '--quiet', target.name]).returncode == 0
        status_path = queue.root / 'status.json'
        if active and status_path.is_file():
            status = read(status_path)
            if status.get('state') == 'watching' and time.time() - status_path.stat().st_mtime < 15:
                report_ready(queue, target.name)
                return
        time.sleep(1)
    raise RuntimeError(f'Service did not become ready. Read: journalctl -u {target.name} -n 80')


if __name__ == '__main__':
    main()

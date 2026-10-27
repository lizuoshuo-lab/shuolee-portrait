"""Install a systemd service under the invoking user's identity."""
import argparse
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait.queue119 import VideoQueue, read


def quoted(value):
    value = str(value)
    if any(c in value for c in '\r\n\0'):
        raise ValueError('Invalid newline in service setting.')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def service_text(profile):
    user = pwd.getpwuid(os.getuid())
    return '\n'.join([
        '[Unit]', 'Description=Shuolee 119 video reconstruction',
        'After=network-online.target', 'Wants=network-online.target', '',
        '[Service]', 'Type=simple', 'User=' + user.pw_name,
        'WorkingDirectory=' + quoted(PROJECT),
        'Environment=' + quoted('HOME=' + user.pw_dir),
        'Environment=PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4',
        'UnsetEnvironment=PYTHONPATH SHUOLEE_EXTRA_PYTHONPATH',
        'ExecStart=' + ' '.join(map(quoted, [sys.executable, '-s', PROJECT/'watch119.py', '--deployment', profile])),
        'Restart=on-failure', 'RestartSec=5', 'KillMode=mixed', 'TimeoutStopSec=90',
        'UMask=0027', '', '[Install]', 'WantedBy=multi-user.target', '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='shuolee119')
    parser.add_argument('--deployment', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.name):
        parser.error('Invalid service name')
    profile = args.deployment.resolve()
    queue = VideoQueue(profile)
    target = Path('/etc/systemd/system') / (args.name + '.service')
    unit = service_text(profile)
    admin = [] if os.getuid() == 0 else ['sudo']
    if target.exists() and target.read_text() != unit:
        raise RuntimeError(f'Existing service uses different paths: {target}. Existing training was preserved.')
    with tempfile.TemporaryDirectory(prefix='shuolee-service-') as temp:
        source = Path(temp) / target.name
        source.write_text(unit, encoding='utf-8')
        subprocess.run([*admin, 'install', '-m', '644', str(source), str(target)], check=True)
    subprocess.run([*admin, 'systemctl', 'daemon-reload'], check=True)
    subprocess.run([*admin, 'systemctl', 'enable', '--now', target.name], check=True)
    for _ in range(30):
        active = subprocess.run(['systemctl', 'is-active', '--quiet', target.name]).returncode == 0
        status_path = queue.root / 'status.json'
        if active and status_path.is_file():
            status = read(status_path)
            if status.get('state') == 'watching' and time.time() - status_path.stat().st_mtime < 15:
                print(f'119 READY\nUpload videos: {queue.inbox}\nResults: {queue.runs}\nStatus: {status_path}\nService: {target.name}', flush=True)
                return
        time.sleep(1)
    raise RuntimeError(f'Service did not become ready. Read: journalctl -u {target.name} -n 80')


if __name__ == '__main__':
    main()

"""Temporarily test a WSL configuration; always restore exact original bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

args = argparse.ArgumentParser()
args.add_argument('--name', required=True)
args.add_argument('--setting', action='append', required=True)
args = args.parse_args()
if not re.fullmatch(r'[a-z0-9-]+', args.name):
    raise ValueError('Invalid probe name')
root = Path(__file__).resolve().parent / 'wsl-service-repair'
config = Path.home() / '.wslconfig'
state_path = root / (args.name + '-status.json')
wsl = str(Path(os.environ['SystemRoot']) / 'System32' / 'wsl.exe')
state = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
             phase='preflight', reboot=False, config_restored=False, settings=args.setting)
def save():
    state['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
def run(command):
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=subprocess.CREATE_NO_WINDOW)
def decode(data):
    return data.decode('utf-16le' if b'\x00' in data else 'utf-8', errors='replace').lstrip('\ufeff')
save()
original = None
candidate = None
try:
    running = run([wsl, '--list', '--running', '--quiet'])
    if running.returncode != 0 or decode(running.stdout).strip():
        raise RuntimeError('A running distribution or unknown list state exists; aborting.')
    original = config.read_bytes()
    state['original_sha256'] = hashlib.sha256(original).hexdigest()
    backup = root / (args.name + '-wslconfig-original.bin')
    with backup.open('xb') as handle:
        handle.write(original)
    raw = original.decode('utf-8-sig')
    if len(re.findall(r'^\[', raw, flags=re.M)) != 1 or '[wsl2]' not in raw:
        raise RuntimeError('Unexpected configuration sections; inspect before editing.')
    for setting in args.setting:
        key, value = setting.split('=', 1)
        if key not in {'safeMode', 'guiApplications', 'nestedVirtualization', 'kernelBootTimeout'}:
            raise RuntimeError('Unsupported diagnostic setting')
        if re.search(r'^\s*'+re.escape(key)+r'\s*=', raw, re.M):
            raise RuntimeError('Setting already exists; inspect rather than override.')
        raw = raw.rstrip() + '\r\n' + key + '=' + value + '\r\n'
    candidate = raw.encode('utf-8')
    shutdown = run([wsl, '--shutdown'])
    if shutdown.returncode != 0:
        raise RuntimeError('WSL shutdown failed before probe')
    config.write_bytes(candidate)
    state['phase'] = 'probing'
    save()
    started = time.monotonic()
    probe = run([wsl, '-d', 'Ubuntu-20.04', '--exec', '/bin/sh', '-lc',
                 'printf "WSL_READY\\n"; uname -r; cat /proc/uptime; dmesg | tail -n 60'])
    state['probe_seconds'] = time.monotonic() - started
    state['probe_exit_code'] = probe.returncode
    state['probe_ready'] = probe.returncode == 0 and 'WSL_READY' in decode(probe.stdout)
    (root / (args.name + '-stdout.bin')).write_bytes(probe.stdout)
    (root / (args.name + '-stderr.bin')).write_bytes(probe.stderr)
    (root / (args.name + '-output.txt')).write_text(decode(probe.stdout), encoding='utf-8')
    (root / (args.name + '-stderr.txt')).write_text(decode(probe.stderr), encoding='utf-8')
    state['phase'] = 'probe_complete'
    save()
except Exception as exc:
    state['phase'] = 'failed'
    state['error'] = repr(exc)
finally:
    if original is not None and candidate is not None:
        current = config.read_bytes()
        if current == candidate:
            config.write_bytes(original)
        elif current != original:
            state['restore_error'] = 'Configuration changed externally; not overwriting.'
        state['config_restored'] = config.read_bytes() == original
        if state['config_restored']:
            shutdown = run([wsl, '--shutdown'])
            state['final_shutdown_exit_code'] = shutdown.returncode
    state['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()

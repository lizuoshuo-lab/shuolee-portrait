"""Resume the authorized local environment after a manual Windows restart."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import winreg

LAB = Path(__file__).resolve().parent.parent
TARGET = Path(r'D:\WSL\Ubuntu-20.04')
DISTRO = 'Ubuntu-20.04'
OUT = LAB / 'outputs/fast-livo2-wsl-setup'


def timestamp(value):
    # Windows emits seven fractional digits; Python3.10 accepts microseconds.
    value = re.sub(r'(\.\d{6})\d+(?=Z|[+-])', r'\1', value)
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def decode(raw):
    if b'\x00' in raw[:256]:
        return raw.decode('utf-16-le', errors='replace').lstrip('\ufeff')
    for encoding in ('utf-8', 'gb18030'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode('utf-8', errors='replace')


def distributions():
    found = []
    try:
        parent = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r'Software\Microsoft\Windows\CurrentVersion\Lxss')
    except FileNotFoundError:
        return found
    with parent:
        for index in range(winreg.QueryInfoKey(parent)[0]):
            with winreg.OpenKey(parent, winreg.EnumKey(parent, index)) as key:
                values = {}
                for name in ('DistributionName', 'BasePath', 'Version'):
                    try:
                        values[name] = winreg.QueryValueEx(key, name)[0]
                    except FileNotFoundError:
                        pass
                found.append(values)
    return found


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc)
    run = OUT / ('continue-' + started.strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True, exist_ok=False)
    state = {'recorded_utc': started.isoformat(), 'status': 'preflight',
             'target': str(TARGET), 'distribution': DISTRO, 'commands': [],
             'automatic_restart_performed': False, 'fusion_started': False}

    def save():
        (run / 'receipt.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def execute(argv, timeout):
        print('Running: ' + subprocess.list2cmdline(argv), flush=True)
        index = len(state['commands'])
        command = {'argv': argv, 'status': 'running', 'timeout_seconds': timeout}
        state['commands'].append(command)
        save()
        try:
            result = subprocess.run(argv, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            partial = decode(exc.stdout or b'') + '\n' + decode(exc.stderr or b'')
            (run / ('command-%02d.txt' % index)).write_text(partial, encoding='utf-8')
            command['status'] = 'timed_out_guest_process_state_requires_inspection_before_retry'
            save()
            raise
        stdout, stderr = decode(result.stdout), decode(result.stderr)
        (run / ('command-%02d.txt' % index)).write_text(stdout + '\n' + stderr, encoding='utf-8')
        command.update(status='exited', exit_code=result.returncode)
        save()
        if stdout.strip():
            print(stdout, flush=True)
        if stderr.strip():
            print(stderr, flush=True)
        if result.returncode:
            raise RuntimeError('Native command failed with exit %s' % result.returncode)
        return stdout

    try:
        boot = execute(['powershell.exe', '-NoProfile', '-Command',
                        "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')"], 30).strip()
        state['windows_boot_utc'] = boot
        stage = json.loads((OUT / 'windows-platform-status.json').read_text(encoding='utf-8-sig'))
        if stage['restart_required'] and timestamp(boot) <= timestamp(stage['recorded_utc']):
            state['status'] = 'manual_windows_restart_required'
            save()
            print('Windows platform changes are pending. Restart Windows manually, then run this entry again.')
            return 0 if args.check_only else 2

        known = [item for item in distributions() if item.get('DistributionName') == DISTRO]
        if len(known) > 1:
            raise RuntimeError('Multiple distributions match; review registration before continuing.')
        state['registered_distribution_before'] = known
        if args.check_only:
            state['status'] = 'read_only_preflight_completed_no_installation'
            save()
            return 0
        if not known:
            TARGET.parent.mkdir(parents=True, exist_ok=True)
            if TARGET.exists() and any(TARGET.iterdir()):
                raise RuntimeError('Target directory contains existing files; no overwrite performed.')
            state['status'] = 'installing_ubuntu_with_official_wsl_installer'
            save()
            execute(['wsl.exe', '--install', '--distribution', DISTRO, '--location', str(TARGET),
                     '--version', '2', '--no-launch', '--web-download'], 1800)
            known = [item for item in distributions() if item.get('DistributionName') == DISTRO]
        if len(known) != 1 or known[0].get('Version') != 2:
            raise RuntimeError('Ubuntu WSL2 registration not verified; installer exit alone is insufficient.')
        base = Path(os.path.expandvars(known[0]['BasePath'].removeprefix('\\\\?\\'))).resolve()
        if not base.is_relative_to(TARGET.resolve()):
            raise RuntimeError('Ubuntu registered outside the agreed D: target; no package changes performed.')
        state['registered_distribution_verified'] = known[0]
        probe = execute(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', '/bin/bash', '-lc',
                         'source /etc/os-release; printf "ID=%s\\nVERSION_ID=%s\\n" "$ID" "$VERSION_ID"; uname -r'], 120)
        probe_lines = probe.splitlines()
        if 'ID=ubuntu' not in probe_lines or 'VERSION_ID=20.04' not in probe_lines or not any('microsoft-standard-WSL2' in line for line in probe_lines):
            raise RuntimeError('Actual Ubuntu20.04/WSL2 boot probe failed; bootstrap not executed.')
        runtime_path = LAB / 'manifests/fast-livo2-local-runtime.json'
        runtime = json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.is_file() else {}
        if runtime.get('compiled_environment_accepted'):
            state['status'] = 'accepted_existing_environment_verifying_without_rematerializing_sources'
            state['existing_runtime_evidence'] = runtime
            save()
            execute(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', '/bin/bash',
                     '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/fast_livo2_linux/bootstrap.sh',
                     '--jobs', '4', '--stage', 'verify'], 180)
            state['status'] = 'existing_accepted_environment_verified_no_reinstallation'
            save()
            print('Existing Ubuntu/ROS environment verified. Saved local source changes were preserved; no packages or sources were reinstalled.')
            return 0
        state['status'] = 'ubuntu_boot_verified_installing_and_compiling_ros_environment'
        save()
        execute(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', '/bin/bash', '-lc',
                 "if pgrep -af '(^|/)bootstrap[.]sh([ ]|$)'; then echo 'An earlier guest bootstrap is still active; do not run concurrently.' >&2; exit 75; fi"], 30)
        execute(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', '/bin/bash',
                 '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/fast_livo2_linux/bootstrap.sh', '--jobs', '4'], 7200)
        state['status'] = 'linux_bootstrap_exited_zero_requires_acceptance_receipt_readback'
        save()
        print('Environment command completed. Read the Linux acceptance receipt before running fusion. Bright Screen replay is the next experiment.')
        return 0
    except Exception as exc:
        state['status'] = 'failed_preserved_logs'
        state['error'] = str(exc)
        save()
        print('Stopped: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

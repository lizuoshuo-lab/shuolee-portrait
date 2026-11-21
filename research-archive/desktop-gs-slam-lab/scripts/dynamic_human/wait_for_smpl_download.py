"""Bounded local download dependency: install and check once final ZIP appears."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / 'outputs/dynamic-human-smpl-010'
DOWNLOADS = Path.home() / 'Downloads'


def status(state, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(status=state, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    target = OUT / 'download-watch.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(value), flush=True)


def main():
    deadline = time.monotonic() + 3600
    status('waiting_for_browser_completed_zip', timeout_seconds=3600, directory=str(DOWNLOADS))
    while time.monotonic() < deadline:
        archives = sorted(DOWNLOADS.glob('SMPL_python_v.1.1.0*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        if archives:
            archive = archives[0]
            before = archive.stat()
            time.sleep(5)
            after = archive.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                continue
            status('validating_completed_zip', archive=str(archive))
            subprocess.run([sys.executable, str(LAB / 'scripts/dynamic_human/install_smpl_template.py'), str(archive)], check=True)
            status('installed_runtime_mesh_audit_running', archive=str(archive))
            command = ('source /opt/gs-slam-lab/dynamic-human/env.sh; '
                '/opt/gs-slam-lab/dynamic-human/.venv/bin/python '
                '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/audit_street_smpl_mesh.py')
            subprocess.run(['wsl.exe', '-d', 'Ubuntu-20.04', '--', 'bash', '-lc', command], check=True)
            status('installed_and_cpu_mesh_checks_finished_visual_review_pending', archive=str(archive))
            return
        time.sleep(30)
    status('wait_timeout_no_completed_zip', timeout_seconds=3600)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_needs_inspection', error=repr(error))
        raise

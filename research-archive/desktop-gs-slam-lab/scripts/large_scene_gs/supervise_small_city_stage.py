"""Durable local supervisor for unchanged official SmallCity stage wrappers.

Launch with Start-Process -WindowStyle Hidden. This module is CPU-only; CUDA
is initialized only by the unchanged stage command. It records actual process
snapshots and return codes, preserves failed runs, and never resumes without a
real checkpoint. Process observations are evidence at their timestamp, not a
promise that a later process remains alive.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'
ENVIRONMENT = ROOT / 'scripts/large_scene_gs/run_local.py'
STAGE = ROOT / 'scripts/large_scene_gs/run_small_city_stage.py'
POWERSHELL = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, target)


def current_job_membership():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    member = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(member)):
        return {'query_succeeded': False, 'winerror': ctypes.get_last_error()}
    return {'query_succeeded': True, 'is_in_windows_job': bool(member.value),
            'scope': 'Membership only. Does not prove behavior when the Codex app exits or a job is closed.'}


def python_processes():
    # Fixed read-only command; no interpolated shell content or secrets.
    command = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run([str(POWERSHELL), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f'Process observation failed: {result.returncode}: {result.stderr}')
    value = json.loads(result.stdout.strip() or '[]')
    return value if isinstance(value, list) else [value]


def tail_progress(path):
    if not path.is_file():
        return {'exists': False}
    stat = path.stat()
    with path.open('rb') as stream:
        stream.seek(max(0, stat.st_size - 65536))
        text = stream.read().decode('utf-8', errors='replace')
    values = re.findall(r'(\d+)/(\d+)\s*\[', text)
    return {'exists': True, 'bytes': stat.st_size, 'modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            'last_logged_iterations': int(values[-1][0]) if values else None,
            'last_logged_budget': int(values[-1][1]) if values else None,
            'scope': 'TQDM log observation only; not an optimizer instrumentation count'}


def audit_interrupted(run):
    directory = BASE / 'official-training' / run
    files = list(sorted(p for p in directory.rglob('*') if p.is_file()))
    outcome = json.loads((directory / 'outcome.json').read_text(encoding='utf-8'))
    processes = python_processes()
    relevant = [p for p in processes if 'train_coarse.py' in (p.get('CommandLine') or '') or
                (run in (p.get('CommandLine') or '') and 'run_small_city_stage.py' in (p.get('CommandLine') or ''))]
    if relevant:
        raise RuntimeError('Refusing interruption audit while a relevant Python process is live')
    checkpoint = [p for p in files if p.suffix.lower() in ('.pth', '.pt', '.ckpt') or 'chkpnt' in p.name.lower()]
    final_models = [p for p in files if p.name in ('point_cloud.ply', 'point_cloud.bin')]
    if checkpoint or final_models:
        raise RuntimeError('Saved training state exists; inspect before deciding recovery route')
    target = BASE / 'recovery-audit' / f'{run}-interruption-001'
    target.mkdir(parents=True, exist_ok=False)
    inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    report = {'status': 'interrupted_no_live_python_no_saved_training_state', 'observed_utc': utc(),
              'run': run, 'original_outcome_preserved': outcome, 'original_status_is_stale': outcome.get('status') == 'running',
              'actual_processes_at_observation': processes, 'matching_training_processes': relevant,
              'progress': tail_progress(directory / 'training.log'), 'optimizer_checkpoint_files': [], 'final_model_files': [],
              'all_original_files_preserved': True, 'original_file_inventory': inventory,
              'cause': None, 'exit_code': None, 'limitations': ['No authoritative exit code or termination cause was captured for the lost process.',
              'A stream disconnect was reported in the parent task, but temporal association does not establish process termination causality.'],
              'recovery': 'New full run from scratch with unchanged official stage and independent hidden supervisor; not checkpoint resume',
              'supervisor_source_sha256': sha(__file__)}
    save(target / 'interruption-audit.json', report)
    print(json.dumps({'audit': str(target / 'interruption-audit.json'), 'status': report['status'],
                      'last_logged_iterations': report['progress']['last_logged_iterations']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-interrupted-run')
    parser.add_argument('--stage', choices=['coarse', 'chunk'])
    parser.add_argument('--run')
    parser.add_argument('--input', default='small-city-training-002')
    parser.add_argument('--chunk', choices=['0_0', '0_1', '1_0', '1_1'])
    parser.add_argument('--coarse-run')
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--resolution', type=int, default=4)
    parser.add_argument('--exposure', choices=['official', 'disabled'], default='official')
    parser.add_argument('--port', type=int, default=6022)
    args = parser.parse_args()
    for name in (args.run, args.input, args.coarse_run, args.audit_interrupted_run):
        if name is not None and (not name.replace('-', '').isalnum() or len(name) > 100):
            raise ValueError('Invalid run/input identifier')
    if args.audit_interrupted_run:
        audit_interrupted(args.audit_interrupted_run)
        return
    if args.stage is None or args.run is None:
        parser.error('--stage and --run required for supervised execution')
    if (BASE / 'official-training' / args.run).exists():
        raise ValueError('Existing training run; refusing overwrite')
    free = shutil.disk_usage(ROOT).free
    if free < 80 * 1024**3:
        raise ValueError('System-drive reserve below 80 GiB')
    existing = python_processes()
    other_trainers = [p for p in existing if any(marker in (p.get('CommandLine') or '')
                     for marker in ('train_coarse.py', 'train_single.py', 'train_post.py'))]
    if other_trainers:
        raise ValueError(f'Another official GPU trainer is live: {other_trainers}')
    observation_dir = BASE / 'official-training-supervision' / args.run
    observation_dir.mkdir(parents=True, exist_ok=False)
    for source, filename in ((Path(__file__), 'supervisor-source.py'), (ENVIRONMENT, 'environment-launcher-source.py'), (STAGE, 'stage-launcher-source.py')):
        shutil.copyfile(source, observation_dir / filename)
    command = [str(PYTHON), '-u', str(ENVIRONMENT), str(STAGE), '--stage', args.stage, '--run', args.run,
               '--input', args.input, '--iterations', str(args.iterations), '--resolution', str(args.resolution),
               '--exposure', args.exposure, '--port', str(args.port), '--execute']
    if args.stage == 'chunk':
        if not args.chunk or not args.coarse_run:
            raise ValueError('Chunk and completed local coarse dependency required')
        command.extend(['--chunk', args.chunk, '--coarse-run', args.coarse_run])
    manifest_path = BASE / 'official-inputs' / args.input / 'manifest.json'
    receipt = {'status': 'supervisor_started_child_not_started', 'started_utc': utc(), 'supervisor_pid': os.getpid(),
               'supervisor_parent_pid': os.getppid(), 'child_pid': None, 'command_argv': command, 'cwd': str(ROOT),
               'run': args.run, 'stage': args.stage, 'requested_iterations': args.iterations,
               'resolution': args.resolution, 'exposure': args.exposure,
               'input_manifest': str(manifest_path), 'input_manifest_sha256': sha(manifest_path),
               'supervisor_source_sha256': sha(__file__), 'stage_launcher_sha256': sha(STAGE),
               'environment_launcher_sha256': sha(ENVIRONMENT), 'system_drive_free_bytes': free,
               'official_algorithm_modified': False, 'restart_policy': 'Never automatically restart; preserve real terminal evidence',
               'windows_job_membership': current_job_membership(),
               'recovery_kind': 'from_scratch', 'checkpoint_resume': False,
               'process_snapshot_log': str(observation_dir / 'process-observations.jsonl'),
               'stage_stdout': str(observation_dir / 'stage-wrapper.log'),
               'training_outcome': str(BASE / 'official-training' / args.run / 'outcome.json')}
    save(observation_dir / 'supervisor-outcome.json', receipt)
    start = time.perf_counter()
    child = None
    try:
        with (observation_dir / 'stage-wrapper.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            receipt.update(status='child_started', child_pid=child.pid, child_started_utc=utc())
            save(observation_dir / 'supervisor-outcome.json', receipt)
            while True:
                code = child.poll()
                observation = {'observed_utc': utc(), 'supervisor_pid': os.getpid(), 'child_pid': child.pid,
                               'child_poll_exit_code': code,
                               'progress': tail_progress(BASE / 'official-training' / args.run / 'training.log')}
                try:
                    processes = python_processes()
                    observed_pids = {os.getpid(), child.pid}
                    while True:
                        added = {p['ProcessId'] for p in processes if p['ParentProcessId'] in observed_pids}
                        if added <= observed_pids:
                            break
                        observed_pids |= added
                    observation['actual_processes'] = [p for p in processes if p['ProcessId'] in observed_pids]
                    observation['trainer_pids'] = [p['ProcessId'] for p in observation['actual_processes']
                                                   if ('train_coarse.py' if args.stage == 'coarse' else 'train_single.py') in (p.get('CommandLine') or '')]
                except Exception as error:
                    observation['process_snapshot_error'] = f'{type(error).__name__}: {error}'
                with (observation_dir / 'process-observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
                save(observation_dir / 'latest-process-observation.json', observation)
                receipt.update(last_observed_utc=observation['observed_utc'], trainer_pids=observation.get('trainer_pids', []))
                save(observation_dir / 'supervisor-outcome.json', receipt)
                if code is not None:
                    break
                time.sleep(15)
        receipt.update(child_exit_code=code, status='child_terminal')
        stage_outcome = Path(receipt['training_outcome'])
        if stage_outcome.is_file():
            stage = json.loads(stage_outcome.read_text(encoding='utf-8'))
            receipt.update(stage_outcome_status=stage['status'], stage_outcome_sha256=sha(stage_outcome))
        receipt['status'] = 'completed' if code == 0 and receipt.get('stage_outcome_status') == 'completed' else 'failed'
    except Exception as error:
        receipt.update(status='supervisor_failed', error_type=type(error).__name__, error=str(error),
                       child_poll_exit_code=child.poll() if child is not None else None)
    finally:
        receipt.update(completed_utc=utc(), elapsed_seconds=time.perf_counter() - start)
        save(observation_dir / 'supervisor-outcome.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

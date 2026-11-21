"""Start the frozen 113 reproduction from WinSCP without an interactive terminal."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REVISION = '2026-10-27-winscp-training-2-fresh-run-name'
BASE = Path.home() / 'shuolee'
LOGS = BASE / 'logs'
STATE = LOGS / 'server113-train-state.json'
LOG = LOGS / 'server113-train.log'
PROJECT = BASE / 'portrait113'
ENV = Path.home() / '.local/share/shuolee-portrait/env-cu128'
RUN = BASE / 'runs/113_001'


def run_path(name):
    if not re.fullmatch(r'113_[0-9]{3}', name) or name == '113_000':
        raise ValueError('Use a run name from 113_001 to 113_999; no paths or existing run reuse.')
    run = BASE / 'runs' / name
    if (BASE/'runs').is_symlink() or run.is_symlink():
        raise ValueError('Run paths must not be symbolic links.')
    return run


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(STATE)


def status():
    if not STATE.exists():
        print('TRAIN_NOT_STARTED')
        return
    state = json.loads(STATE.read_text())
    saved_run = Path(state.get('run', str(RUN)))
    if saved_run != BASE/'runs'/saved_run.name:
        raise RuntimeError('Unexpected run path in saved state; inspect the state file.')
    observed_run = run_path(saved_run.name)
    if state['state'] in ('starting', 'running'):
        proc = Path('/proc') / str(state['pid']) / 'cmdline'
        alive = proc.exists() and b'--worker' in proc.read_bytes()
        state['process_alive'] = alive
        if not alive:
            state['state'] = 'interrupted_without_completion'
    print(json.dumps(state, indent=2))
    for path in [LOG, observed_run/'fit/progress.json', observed_run/'gaussian-run/progress.json', observed_run/'logs/train.log']:
        if path.exists():
            print(f'LAST_LOG_LINES {path}:')
            print('\n'.join(path.read_text(errors='replace').splitlines()[-20:]))


def prerequisites():
    install = LOGS / 'server113-install-state.json'
    if not install.exists() or json.loads(install.read_text())['state'] != 'completed':
        raise RuntimeError('INSTALL_NOT_COMPLETED: run server113_pip_repair.pyz --status first. No training started.')
    if RUN.exists():
        raise RuntimeError(f'FRESH_RUN_REQUIRED: {RUN} already exists. Inspect it; do not delete or resume it.')
    revision = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', '--short=7', 'HEAD'], text=True).strip()
    if revision != '5fda749':
        raise RuntimeError(f'Wrong Git revision: {revision}; expected frozen 113 5fda749.')
    for p in [ENV/'bin/python', BASE/'cuda128.env', PROJECT/'scripts/route113.py',
              BASE/'datasets/hover175/training-inputs.json', BASE/'models/113/smplx/SMPLX_NEUTRAL.npz']:
        if not p.is_file():
            raise RuntimeError(f'Missing required file: {p}')
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError(f'GPU_0_BUSY: wait for the existing compute job; no training started.\n{gpu}')


def launch():
    handles = []
    for name in ['server113-install.lock', 'server113-train.lock']:
        handle = (LOGS/name).open('a')
        handles.append(handle)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('INSTALL_OR_TRAIN_ALREADY_RUNNING: check status; do not start a duplicate.')
    prerequisites()
    if STATE.exists():
        previous = STATE.read_bytes()
        history = LOGS/'server113-state-history'
        history.mkdir(exist_ok=True)
        archived = history/(hashlib.sha256(previous).hexdigest()+'.json')
        if not archived.exists():
            archived.write_bytes(previous)
    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    snapdir = LOGS/'server113-launcher'
    snapdir.mkdir(exist_ok=True)
    snapshot = snapdir/(digest+'.py')
    if not snapshot.exists():
        snapshot.write_bytes(source.read_bytes())
    readfd, writefd = os.pipe()
    command = [sys.executable, str(snapshot), '--worker', '--run-name', RUN.name, '--start-fd', str(readfd),
               '--lock-fds', ','.join(str(x.fileno()) for x in handles)]
    with LOG.open('a', buffering=1) as output:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=tuple([readfd]+[x.fileno() for x in handles]))
    save(dict(revision=REVISION, state='starting', stage='precheck', pid=child.pid,
              started=now(), run=str(RUN), log=str(LOG), launcher_sha256=digest,
              training_started=False, quality_accepted=False))
    os.close(readfd)
    os.write(writefd, b'1')
    os.close(writefd)
    for handle in handles:
        handle.close()
    print(f'PIPELINE_STARTED PID={child.pid}')
    print(f'Run: {RUN}')
    print('Audit and CUDA doctor run before fitting/training. You may close WinSCP.')
    print('Check: python3.10 ~/shuolee/server113_train.py --status')


def worker(args):
    if args.start_fd is None or not args.lock_fds:
        raise RuntimeError('Use --start, not --worker directly.')
    if os.read(args.start_fd, 1) != b'1':
        raise RuntimeError('Launcher state handshake failed.')
    os.close(args.start_fd)
    for fd in args.lock_fds.split(','):
        fcntl.flock(int(fd), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(STATE.read_text())
    try:
        prerequisites()
        env = os.environ.copy()
        for key in ['PYTHONPATH', 'PYTHONHOME', 'SHUOLEE_EXTRA_PYTHONPATH']:
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
                   PATH=str(ENV/'bin')+os.pathsep+env.get('PATH', ''))
        # Source the validated CUDA environment in Bash; never depend on bin/activate.
        shell = 'set -euo pipefail\nsource "$HOME/shuolee/cuda128.env"\nexec "$@"\n'
        for stage in ['audit', 'doctor', 'all']:
            state.update(state='running', stage=stage, updated=now(), pipeline_execution_started=stage=='all')
            save(state)
            print(f'{now()} STAGE={stage}', flush=True)
            command = [str(ENV/'bin/python'), '-s', str(PROJECT/'scripts/route113.py'), stage,
                       '--dataset', str(BASE/'datasets/hover175'), '--assets', str(BASE/'models/113')]
            if stage == 'all':
                command += ['--run', str(RUN)]
            subprocess.run(['bash', '-c', shell, 'shuolee113', *command], cwd=PROJECT, env=env, check=True)
        result = json.loads((RUN/'gaussian-run/result.json').read_text())
        delivery = json.loads((RUN/'gaussian-run/delivery.json').read_text())
        exported = json.loads((RUN/'stages/export.json').read_text())
        if not (result['steps'] == 18000 and result['from_scratch'] and
                delivery['compatibility_passed'] and exported['state'] == 'completed'):
            raise RuntimeError('Pipeline returned, but final result/export markers did not pass.')
        state.update(state='completed', stage='export', finished=now(), training_started=True,
                     steps=18000, compatibility_passed=True, quality_accepted=False)
        save(state)
        print('TRAINING_AND_EXPORT_COMPLETED_REVIEW_PLY_QUALITY', flush=True)
    except BaseException as exc:
        state.update(state='failed', finished=now(), error=str(exc), quality_accepted=False)
        save(state)
        traceback.print_exc()
        raise


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-name', default='113_001',
                        help='Fresh output name, 113_001..113_999. Existing outputs are never reused.')
    parser.add_argument('--start-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fds', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        RUN = run_path(args.run_name)
    except ValueError as exc:
        parser.error(str(exc))
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.status:
        status()
    elif args.start:
        launch()
    else:
        worker(args)


if __name__ == '__main__':
    main()

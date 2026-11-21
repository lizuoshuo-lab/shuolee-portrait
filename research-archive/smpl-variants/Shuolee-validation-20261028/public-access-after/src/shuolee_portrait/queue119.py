"""Persistent, single-worker video intake. Only the child imports CUDA code."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def worker_lock(path):
    """An OS lock is released even if the worker is killed or the host reboots."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('This video queue already has a worker.') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('This video queue already has a worker.') from exc
        try:
            yield stream
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def process_group_running(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Container PID 1 may leave exited children as zombies. They hold no locks
    # or GPU state; count live group members, including descendants of a dead leader.
    uncertain = False
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pid and fields[0] not in ('Z', 'X'):
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            uncertain = True
    return uncertain


class VideoQueue:
    def __init__(self, deployment, *, settle_seconds=30, command=None):
        self.deployment = Path(deployment).resolve()
        profile = read(self.deployment)
        if profile.get('schema') != 'shuolee-deployment119-v1':
            raise ValueError('Invalid deployment profile. Run deploy.sh.')
        if Path(profile['environment']).resolve() != Path(sys.prefix).resolve():
            raise RuntimeError('The worker must use the configured Python environment.')
        self.runs = Path(profile['runs']).resolve()
        self.inbox = self.runs.parent / 'videos'
        self.root = self.runs.parent / 'queue'
        for path in (self.runs, self.inbox, self.root):
            if path == PROJECT or PROJECT in path.parents:
                raise ValueError('Videos, queue and runs must be outside the Git checkout.')
            path.mkdir(parents=True, exist_ok=True)
        for name in ('jobs', 'inputs', 'logs'):
            (self.root / name).mkdir(exist_ok=True)
        self.settle_seconds = settle_seconds
        self.command = command or [sys.executable, '-s', str(PROJECT / 'run_smplx.py'), 'train']
        self.observed = {}
        self.active = None
        self.child = None
        self.log = None
        self.stopping = False
        self.lock_stream = None

    def jobs(self):
        return sorted((read(p) for p in (self.root / 'jobs').glob('*.json')),
                      key=lambda job: (job['created'], job['id']))

    def record(self, job):
        job['updated'] = now()
        save(self.root / 'jobs' / (job['id'] + '.json'), job)

    def snapshot(self, source, expected):
        """Keep the upload untouched and train an immutable copy, identified by content."""
        if source.is_symlink():
            return None
        fd, filename = tempfile.mkstemp(prefix='upload-', suffix='.part', dir=self.root / 'inputs')
        temp = Path(filename)
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, 'wb') as dest, source.open('rb') as src:
                for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                    dest.write(block)
                    digest.update(block)
                dest.flush()
                os.fsync(dest.fileno())
            if signature(source) != expected or temp.stat().st_size != expected[0]:
                return None
            job_id = digest.hexdigest()
            job_file = self.root / 'jobs' / (job_id + '.json')
            if job_file.exists():
                return read(job_file)
            stored = self.root / 'inputs' / (job_id + source.suffix.lower())
            temp.replace(stored)
            label = re.sub(r'[^\w.-]+', '_', source.stem).strip('._')[:48] or 'video'
            job = dict(id=job_id, created=now(), state='queued', video=str(source),
                       input=str(stored), run=str(self.runs / (label + '-' + job_id[:16])),
                       log=str(self.root / 'logs' / (job_id + '.log')))
            self.record(job)
            print(json.dumps(dict(state='queued', video=source.name, run=job['run']), ensure_ascii=False), flush=True)
            return job
        finally:
            temp.unlink(missing_ok=True)

    def scan(self, monotonic=None):
        clock = time.monotonic() if monotonic is None else monotonic
        present = set()
        for source in sorted(self.inbox.iterdir()):
            if source.name.startswith('.') or source.suffix.lower() not in VIDEO_SUFFIXES or source.is_symlink():
                continue
            try:
                if not source.is_file():
                    continue
                sig = signature(source)
                if sig[0] == 0:
                    continue
                present.add(source)
                previous = self.observed.get(source)
                if previous is None or previous['signature'] != sig:
                    self.observed[source] = dict(signature=sig, since=clock, processed=False)
                    continue
                if previous['processed'] or clock - previous['since'] < self.settle_seconds:
                    continue
                job = self.snapshot(source, sig)
                previous['processed'] = job is not None
            except FileNotFoundError:
                continue  # An uploader may rename or replace a file while scanning.
        self.observed = {p: value for p, value in self.observed.items() if p in present}

    @staticmethod
    def finished_result(job):
        try:
            result = read(Path(job['run']) / 'result.json')
            ply = Path(job['run']) / 'export/point_cloud.ply'
            if (result.get('state') == 'completed' and result.get('steps') == 18000
                    and result.get('format_passed') is True and ply.is_file()):
                return result
        except (OSError, ValueError):
            pass
        return None

    def recover(self):
        # A detached worker can die while a training process group is still alive.
        # Do not recover its job or admit another one until that group has exited.
        for job in self.jobs():
            if job['state'] == 'running':
                if (os.name != 'nt' and isinstance(job.get('pid'), int) and job['pid'] > 0
                        and process_group_running(job['pid'])):
                    raise RuntimeError(f'Previous training process group {job["pid"]} is still active; '
                                       'wait for it to exit before restarting this queue.')
                result = self.finished_result(job)
                job.update(state='completed' if result else 'interrupted', finished=now(),
                           error=None if result else 'Worker stopped before training completed; output retained.')
                self.record(job)

    def start_next(self):
        if self.active is not None or self.stopping:
            return
        pending = [job for job in self.jobs() if job['state'] == 'queued']
        if not pending:
            return
        job = pending[0]
        job.update(state='running', started=now())
        self.record(job)  # A crash between admission and spawn never causes an implicit retrain.
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('SHUOLEE_EXTRA_PYTHONPATH', None)
        env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        self.log = Path(job['log']).open('ab', buffering=0)
        inherited = {}
        if os.name != 'nt' and self.lock_stream is not None and not self.lock_stream.closed:
            inherited['pass_fds'] = (self.lock_stream.fileno(),)
        try:
            self.child = subprocess.Popen(
                [*self.command, job['input'], '--deployment', str(self.deployment), '--output', job['run']],
                cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=os.name != 'nt', **inherited)
        except OSError as exc:
            self.log.close()
            self.log = None
            job.update(state='failed', finished=now(), error=str(exc))
            self.record(job)
            return
        self.active = job
        job['pid'] = self.child.pid
        self.record(job)
        print(json.dumps(dict(state='running', run=job['run'], pid=self.child.pid)), flush=True)

    def collect(self):
        if self.child is None or self.child.poll() is None:
            return
        code = self.child.returncode
        result = self.finished_result(self.active)
        state = 'completed' if code == 0 and result else ('interrupted' if self.stopping else 'failed')
        self.active.update(state=state, finished=now(), exit_code=code,
                           error=None if state == 'completed' else 'See job log and run/status.json.')
        self.record(self.active)
        print(json.dumps(dict(state=state, run=self.active['run'], exit_code=code)), flush=True)
        self.log.close()
        self.active = self.child = self.log = None

    def heartbeat(self):
        counts = {}
        for job in self.jobs():
            counts[job['state']] = counts.get(job['state'], 0) + 1
        save(self.root / 'status.json', dict(state='stopping' if self.stopping else 'watching',
             updated=now(), pid=os.getpid(), inbox=str(self.inbox), runs=str(self.runs),
             active=self.active, counts=counts))

    def stop(self, *_):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            if os.name != 'nt':
                os.killpg(self.child.pid, signal.SIGINT)
            else:
                self.child.terminate()

    def serve(self, poll_seconds=3):
        with worker_lock(self.root / 'worker.lock') as stream:
            self.lock_stream = stream
            self.recover()
            signal.signal(signal.SIGTERM, self.stop)
            signal.signal(signal.SIGINT, self.stop)
            if os.name != 'nt':
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
            try:
                while not self.stopping or self.child is not None:
                    self.collect()
                    if not self.stopping:
                        self.scan()
                        self.start_next()
                    self.heartbeat()
                    time.sleep(poll_seconds)
            finally:
                self.stop()
                if self.child is not None:
                    try:
                        self.child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        if os.name != 'nt':
                            os.killpg(self.child.pid, signal.SIGKILL)
                        else:
                            self.child.kill()
                        self.child.wait()
                    self.collect()
                save(self.root / 'status.json', dict(state='stopped', updated=now(), pid=os.getpid()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, default=Path.home() / 'smpl-x/queue/deployment.json')
    args = parser.parse_args(argv)
    VideoQueue(args.deployment).serve()

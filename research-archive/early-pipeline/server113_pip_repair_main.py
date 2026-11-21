"""Offline repair of the existing portrait113 Python environment, never training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone

REVISION = "2026-10-27-offline-pip-3-cuda-headers"
VERSIONS = {"pip": "25.1.1", "setuptools": "79.0.1", "wheel": "0.45.1"}


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def background_status(logdir):
    path = logdir/'server113-install-state.json'
    if not path.is_file():
        print('INSTALL_NOT_STARTED: run --install-background after uploading revision2.')
        return 0
    state = json.loads(path.read_text())
    if state.get('state') in ('starting', 'running'):
        proc = Path('/proc')/str(state.get('pid', 0))/'cmdline'
        alive = proc.is_file() and b'--background-worker' in proc.read_bytes()
        state['process_alive'] = alive
        if not alive:
            state['state'] = 'interrupted_without_completion'
    print(json.dumps(state, indent=2))
    path = logdir/'server113-install.log'
    if path.exists():
        print('LAST_LOG_LINES:')
        print('\n'.join(path.read_text(errors='replace').splitlines()[-30:]))
    return 0


def start_background(args):
    """A detached install with a kernel-held lock, usable from WinSCP's command window."""
    import fcntl
    lock = (args.log_dir/'server113-install.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('INSTALL_ALREADY_RUNNING: do not start another installer.')
        return background_status(args.log_dir)
    # Guard other installations launched using the earlier guide as well.
    conflicts = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if any(Path(os.fsdecode(a)).name in ('setup_server.sh', 'setup_113.sh') for a in argv if a):
                conflicts.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if conflicts:
        lock.close()
        raise RuntimeError(f'An installation is already running: PIDs {conflicts}. Inspect its log first.')
    path = Path(sys.argv[0]).resolve()
    # Snapshot the small installer so re-uploading a new package cannot mutate executing code.
    run_dir = args.log_dir/'server113-installer'
    run_dir.mkdir(exist_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot = run_dir/(digest+'.pyz')
    if not snapshot.exists():
        snapshot.write_bytes(path.read_bytes())
    ready_read, ready_write = os.pipe()
    command = [sys.executable, str(snapshot), '--resume-install', '--background-worker',
               '--lock-fd', str(lock.fileno()), '--start-fd', str(ready_read), '--env-dir', str(args.env_dir), '--log-dir', str(args.log_dir)]
    state_path = args.log_dir/'server113-install-state.json'
    with (args.log_dir/'server113-install.log').open('a', buffering=1) as output:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True, pass_fds=(lock.fileno(),ready_read))
        save_json(state_path, dict(revision=REVISION, state='starting', pid=child.pid,
                  started=datetime.now(timezone.utc).isoformat(), package_sha256=digest,
                  command=command, log=str(args.log_dir/'server113-install.log'), training_started=False))
    os.close(ready_read)
    os.write(ready_write,b'1')
    os.close(ready_write)
    lock.close()
    print(f'INSTALL_STARTED PID={child.pid}')
    print(f'Log: {args.log_dir / "server113-install.log"}')
    print('Check: python3.10 ~/shuolee/server113_pip_repair.pyz --status')
    print('You may close the WinSCP command window. No training was started.')
    return 0


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, value):
        self.console.write(value)
        self.logfile.write(value)
        self.flush()

    def flush(self):
        self.console.flush()
        self.logfile.flush()


def run(command, env):
    print("RUN:", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", bufsize=1, env=env)
    for line in process.stdout:
        print(line, end="", flush=True)
    rc = process.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)


def apply_install_patch(project, logdir):
    """Patch only reviewed frozen-113 installer files; reject unknown local edits."""
    with zipfile.ZipFile(Path(sys.argv[0]).resolve()) as archive:
        patch = json.loads(archive.read('install-patch.json'))
        changes = []
        for item in patch['files']:
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'scripts':
                raise RuntimeError('Invalid installer patch target')
            destination = project/relative
            if destination.is_symlink():
                raise RuntimeError(f'Refusing to replace a symbolic-link installer: {destination}')
            payload = archive.read('install-patch/'+relative.as_posix())
            digest = hashlib.sha256(payload).hexdigest()
            if digest != item['after_sha256']:
                raise RuntimeError('Installer patch checksum mismatch')
            existing = destination.read_bytes() if destination.exists() else None
            before = hashlib.sha256(existing.replace(b'\r\n', b'\n')).hexdigest() if existing is not None else None
            if before == digest:
                continue
            if before != item['before_sha256']:
                raise RuntimeError(f'Unknown local edit in {relative}; kept unchanged. '
                                   'Send the current script instead of overwriting it.')
            changes.append((destination, payload, existing, item))
        backup = logdir/'installer-before-cuda-header-patch'
        backup.mkdir(exist_ok=True)
        receipt = []
        for destination, payload, existing, item in changes:
            if existing is not None:
                saved = backup/(destination.name+'.'+hashlib.sha256(existing).hexdigest()[:16])
                if not saved.exists():
                    saved.write_bytes(existing)
            temporary = destination.with_name(destination.name+'.cuda-header-patch.tmp')
            temporary.write_bytes(payload)
            if destination.exists():
                temporary.chmod(destination.stat().st_mode)
            temporary.replace(destination)
            receipt.append(item)
        save_json(logdir/'cuda-header-installer-patch.json', dict(revision=REVISION, changed=receipt,
                  baseline_git='5fda749', training_code_changed=False, training_started=False))
        print('CUDA_HEADER_INSTALL_PATCH_READY: development include and library paths precede pip runtime headers.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-install", action="store_true",
                        help="After pip repair, continue dependency installation in an interactive SSH terminal.")
    parser.add_argument('--install-background', action='store_true', help='Continue installation detached, with status/logs; works from WinSCP.')
    parser.add_argument('--status', action='store_true', help='Read the detached installation state and last log lines.')
    parser.add_argument('--background-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--start-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument("--env-dir", type=Path,
                        default=Path.home()/".local/share/shuolee-portrait/env-cu128")
    parser.add_argument("--log-dir", type=Path, default=Path.home()/"shuolee/logs")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.status:
        return background_status(args.log_dir)
    if args.install_background:
        if sys.platform != 'linux':
            raise RuntimeError('Run on the Linux server.')
        return start_background(args)
    if args.background_worker:
        if args.lock_fd is None or args.start_fd is None:
            raise RuntimeError('Missing inherited install lock; use --install-background.')
        if os.read(args.start_fd,1)!=b'1':
            raise RuntimeError('Launcher did not publish installation state.')
        os.close(args.start_fd)
        import fcntl
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.log_dir/'server113-install-state.json'
    log_path = args.log_dir/"pip-repair.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as logfile:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, logfile), Tee(stderr, logfile)
        try:
            print(f"\n=== Pip repair {REVISION} {datetime.now(timezone.utc).isoformat()} ===")
            print(f"Complete log: {log_path}")
            print("Pip/setuptools/wheel are installed from this package without network access.")
            print("No CUDA installation, environment deletion, or training is performed.")
            if sys.platform != "linux":
                raise RuntimeError("Run this package on the Linux server, not in Windows Python.")
            target = args.env_dir.expanduser().resolve()
            python = target/"bin/python"
            if not (target/"pyvenv.cfg").is_file() or not python.is_file():
                raise RuntimeError(f"Existing Python virtual environment not found: {target}")
            env = os.environ.copy()
            for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_CONSTRAINT"):
                env.pop(key, None)
            env["PYTHONNOUSERSITE"] = "1"
            env["PIP_CONFIG_FILE"] = os.devnull
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            check = subprocess.run([str(python), "-I", "-c",
                "import sys,json;print(json.dumps(dict(prefix=sys.prefix,base=sys.base_prefix,version=list(sys.version_info[:2]))))"],
                env=env, text=True, capture_output=True, check=True)
            identity = json.loads(check.stdout)
            print("Python identity:", json.dumps(identity))
            if (Path(identity["prefix"]).resolve() != target or
                    identity["prefix"] == identity["base"] or identity["version"] != [3, 10]):
                raise RuntimeError("Expected the selected Python 3.10 virtual environment; refusing to alter system Python.")
            install = target/"install"
            install.mkdir(exist_ok=True)
            import fcntl
            with (install/"pip-repair.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("Another copy of this repair is running. Do not start a second installer.")
                wheelhouse = install/"pip-bootstrap-wheels"
                wheelhouse.mkdir(exist_ok=True)
                with zipfile.ZipFile(Path(sys.argv[0]).resolve()) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    if manifest["versions"] != VERSIONS or len(manifest["wheels"]) != 3:
                        raise RuntimeError("Unexpected bootstrap package contents.")
                    for item in manifest["wheels"]:
                        name = item["filename"]
                        if Path(name).name != name or not name.endswith("-none-any.whl"):
                            raise RuntimeError("Unexpected wheel filename.")
                        data = archive.read("wheels/"+name)
                        if hashlib.sha256(data).hexdigest() != item["sha256"]:
                            raise RuntimeError("Wheel hash mismatch: "+name)
                        destination = wheelhouse/name
                        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                            temporary = destination.with_suffix(".partial")
                            temporary.write_bytes(data)
                            temporary.replace(destination)
                        print("WHEEL_HASH_OK", name)
                pip_wheel = wheelhouse/"pip-25.1.1-py3-none-any.whl"
                # Run pip directly from its verified wheel; this needs neither pip nor ensurepip in the venv.
                bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
                run([str(python), "-I", "-c", bootstrap, str(pip_wheel), "--isolated",
                     "--disable-pip-version-check", "--no-input", "install", "--no-index", "--no-deps",
                     "--find-links", str(wheelhouse)] + [f"{k}=={v}" for k, v in VERSIONS.items()], env)
                run([str(python), "-I", "-m", "pip", "--version"], env)
                verify = (
                    "from pathlib import Path;from importlib import metadata;import json,sys;"
                    "root=Path(sys.prefix).resolve();expected=json.loads(sys.argv[1]);"
                    "actual={k:metadata.version(k) for k in expected};assert actual==expected,(actual,expected);"
                    "locations={k:str(Path(metadata.distribution(k).locate_file('')).resolve()) for k in expected};"
                    "assert all(Path(p)==root or root in Path(p).parents for p in locations.values()),locations;"
                    "print('VERIFIED_PACKAGES',json.dumps(actual));print('VERIFIED_LOCATIONS',json.dumps(locations))")
                run([str(python), "-I", "-c", verify, json.dumps(VERSIONS)], env)
                print("PIP_REPAIR_OK")
                if not args.resume_install:
                    print("Pip repair completed only; dependency installation and training have not started.")
                    return 0
                if not sys.stdin.isatty() and not args.background_worker:
                    raise RuntimeError("Pip is repaired. In WinSCP run --install-background; in interactive SSH use --resume-install. Do not skip to guide step5.")
                project = Path.home()/"shuolee/portrait113"
                cuda_env = Path.home()/"shuolee/cuda128.env"
                if (not cuda_env.is_file() or not (project/"scripts/setup_server.sh").is_file()
                        or not (project/"scripts/setup_113.sh").is_file()):
                    raise RuntimeError("cuda128.env, setup_server.sh, or setup_113.sh is missing. Pip repair was completed.")
                commit = subprocess.run(["git", "-C", str(project), "rev-parse", "--short=7", "HEAD"],
                                        text=True, capture_output=True, check=True).stdout.strip()
                if commit != "5fda749":
                    raise RuntimeError(f"Expected frozen Git 113 version 5fda749; found {commit}. No dependency install started.")
                apply_install_patch(project, args.log_dir)
                print('ACTIVATE_NOT_REQUIRED: setup uses the explicit environment Python path.')
                if args.background_worker:
                    save_json(state_path, dict(revision=REVISION,state='running',pid=os.getpid(),
                              updated=datetime.now(timezone.utc).isoformat(),stage='install_dependencies',
                              log=str(args.log_dir/'server113-install.log'),training_started=False))
                script = ('set -euo pipefail; source "$1"; cd "$2"; '
                          'export PYTHON_BIN=python3.10 SHUOLEE_ENV_DIR="$3" MAX_JOBS=2; '
                          'bash scripts/setup_server.sh --resume-install; '
                          'bash scripts/setup_113.sh')
                run(["bash", "-c", script, "shuolee-113-install", str(cuda_env), str(project), str(target)], env)
                print("DEPENDENCY_INSTALL_FINISHED_CONTINUE_GUIDE_STEP_5")
                print("Dependencies installed. No training was started by this repair package.")
                if args.background_worker:
                    save_json(state_path, dict(revision=REVISION,state='completed',pid=os.getpid(),
                              updated=datetime.now(timezone.utc).isoformat(),training_started=False,
                              next_action='guide_step5_audit_and_doctor',log=str(args.log_dir/'server113-install.log')))
            return 0
        except Exception as exc:
            print(f"REPAIR_STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            if args.background_worker:
                save_json(state_path, dict(revision=REVISION,state='failed',pid=os.getpid(),
                          updated=datetime.now(timezone.utc).isoformat(),error=str(exc),training_started=False,
                          log=str(args.log_dir/'server113-install.log')))
            print(f"Keep the existing environment. Send this complete log: {log_path}", file=sys.stderr)
            return 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    raise SystemExit(main())

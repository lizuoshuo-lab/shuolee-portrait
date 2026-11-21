"""Check or add one relative source-path alias for the frozen 113 head stage.

No package installation, source replacement, camera update, fitting or training.
The default only checks. --apply adds the alias after identity and idle checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REVISION = '2026-10-27-head-source-path-1'
COMMIT = '5fda749498cca899fb63011dc7b1c71511c14050'
# Git blobs use LF. The Windows working tree may still contain older CRLF bytes.
SOURCES = {
    'src/route113/preparation/fit_head_surface_082.py':
        '797207cbf299684813e5d2f8f79373bd41edf4cc6e4143019a4354ac56f87bf3',
    'src/avatar_core/flame_model/flame.py':
        '09f4052a3df4251539b20af907475dd58ab9a2048d1761703c3d55480de278b5',
    'src/avatar_core/flame_model/lbs.py':
        '132c9266bd79d7c68a39e225910ca251bcb9664ae3a8409eb5454b0b9c02afb8',
}
ALIAS = 'src/route113/preparation/flame_model'
CANONICAL = 'src/avatar_core/flame_model'
RELATIVE_TARGET = '../../avatar_core/flame_model'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_project(project):
    project = Path(project)
    root = project.resolve(strict=True)
    if project.is_symlink():
        raise RuntimeError('Project root is a symbolic link; kept unchanged.')
    top = Path(subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True).strip()).resolve()
    commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if top != root or commit != COMMIT:
        raise RuntimeError(f'Expected the exact frozen 113 project and commit {COMMIT}; kept unchanged.')
    checked = {}
    for relative, expected in SOURCES.items():
        source = root/relative
        resolved = source.resolve(strict=True)
        if not source.is_file() or source.is_symlink() or root not in resolved.parents:
            raise RuntimeError(f'Unexpected source location: {relative}; kept unchanged.')
        actual = sha(source)
        if actual != expected:
            raise RuntimeError(f'Source identity differs: {relative}; expected Git LF SHA256 {expected}. Kept unchanged.')
        checked[relative] = actual
    canonical = root/CANONICAL
    if canonical.is_symlink() or not canonical.is_dir():
        raise RuntimeError('Canonical FLAME source directory has an unexpected type.')
    alias = root/ALIAS
    if alias.parent.resolve(strict=True) != alias.parent:
        raise RuntimeError('Alias parent resolves through a symbolic link; kept unchanged.')
    if alias.is_symlink():
        try:
            correct = alias.resolve(strict=True) == canonical.resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            correct = False
        if not correct:
            raise RuntimeError('Existing alias points elsewhere or is broken; kept unchanged.')
        alias_exists = True
    elif alias.exists():
        raise RuntimeError('Alias location already contains a regular file/directory; kept unchanged.')
    else:
        alias_exists = False
    return dict(revision=REVISION, baseline_git=commit, project=str(root), source_sha256=checked,
                alias=str(alias), relative_target=RELATIVE_TARGET, canonical=str(canonical),
                alias_exists=alias_exists, tracked_sources_changed=False,
                existing_runs_modified=False, training_started=False)


def check_processes():
    conflicts = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            names = {Path(os.fsdecode(a)).name for a in argv if a}
            command = b' '.join(argv)
            direct = names.intersection({'route113.py', 'setup_server.sh', 'setup_113.sh',
                                         'fit_head_surface_082.py', 'initialize_ncc_pair_093.py'})
            detached = ((b'server113-launcher/' in command and b'--worker' in argv)
                        or (b'server113-installer/' in command and b'--background-worker' in argv))
            if direct or detached:
                conflicts.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if conflicts:
        raise RuntimeError(f'Project processes are still running: {conflicts}. Wait; no change applied.')


def apply_alias(project, logdir):
    if sys.platform != 'linux':
        raise RuntimeError('--apply must run on the Linux server.')
    import fcntl
    handles = []
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        for name in ['server113-install.lock', 'server113-train.lock']:
            handle = (logdir/name).open('a')
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('INSTALL_OR_TRAIN_ALREADY_RUNNING: no change applied.')
        check_processes()
        report = inspect_project(project)
        alias = Path(report['alias'])
        created = not report['alias_exists']
        if created:
            alias.symlink_to(RELATIVE_TARGET, target_is_directory=True)
        canonical = Path(report['canonical']).resolve(strict=True)
        if alias.resolve(strict=True) != canonical:
            raise RuntimeError('Alias verification failed; preserve the current filesystem for inspection.')
        for name in ['flame.py', 'lbs.py']:
            if sha(alias/name) != SOURCES[CANONICAL+'/'+name]:
                raise RuntimeError(f'Alias source SHA mismatch: {name}.')
        report.update(alias_exists=True, alias_created=created,
                      resolved_target=str(canonical), applied_at=datetime.now(timezone.utc).isoformat())
        path = logdir/'server113-head-path-fix.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)
        return report
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Add the verified alias; never starts training.')
    args = parser.parse_args()
    project = Path.home()/'shuolee/portrait113'
    logs = Path.home()/'shuolee/logs'
    report = apply_alias(project, logs) if args.apply else inspect_project(project)
    print(json.dumps(report, indent=2))
    print('HEAD_SOURCE_PATH_FIX_READY' if report['alias_exists'] else 'HEAD_SOURCE_PATH_FIX_NEEDED: re-run with --apply.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'HEAD_SOURCE_PATH_FIX_STOPPED: {exc}', file=sys.stderr)
        raise SystemExit(2)

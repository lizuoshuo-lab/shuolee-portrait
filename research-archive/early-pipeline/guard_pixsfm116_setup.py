"""Enforce the preregistered setup deadline on one verified WSL process tree."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time

p = argparse.ArgumentParser()
p.add_argument('--pid', type=int, required=True)
p.add_argument('--needle', required=True)
p.add_argument('--deadline', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
deadline = dt.datetime.fromisoformat(a.deadline).timestamp()

def process(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19]), 'command': command}
    except (OSError, ValueError):
        return None

def save(state, **extra):
    value = {'state': state, 'updated': dt.datetime.now(dt.timezone.utc).isoformat(),
             'guard_pid': os.getpid(), 'target_pid': a.pid, 'deadline': a.deadline,
             'target_identity': original, **extra}
    temp = a.output.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(a.output)

original = process(a.pid)
if not original or a.needle not in original['command']:
    raise SystemExit('Target identity does not match; no signal sent.')
save('watching')
while True:
    current = process(a.pid)
    if not current or current['start_ticks'] != original['start_ticks']:
        save('target_exited_before_deadline')
        break
    if time.time() < deadline:
        time.sleep(min(20, max(0.1, deadline-time.time())))
        continue
    all_procs = {int(f.name): process(int(f.name)) for f in Path('/proc').iterdir() if f.name.isdigit()}
    family = {a.pid: current}
    while True:
        children = {pid: x for pid, x in all_procs.items() if x and x['ppid'] in family and pid not in family}
        if not children:
            break
        family.update(children)
    save('deadline_reached_terminating_verified_tree', affected=list(family.values()))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, known in reversed(list(family.items())):
            live = process(pid)
            if live and live['start_ticks'] == known['start_ticks']:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(5)
    save('deadline_enforced', affected=list(family.values()))
    break

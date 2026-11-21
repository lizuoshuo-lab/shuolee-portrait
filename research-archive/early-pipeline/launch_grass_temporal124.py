import os, subprocess, sys
from pathlib import Path
from run_official_gaussians123 import save,now,sha
import run_grass_temporal124 as c
log=c.OUT/'console.log'
assert not log.exists()
cmd=[sys.executable,'-B',str(Path(c.__file__)),'run']
with log.open('x',encoding='utf-8') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
    save(c.OUT/'launch.json',dict(updated=now(),pid=child.pid,command=cmd,log=str(log),launcher_sha256=sha(__file__),training_started=False))
    print('GRASS_TEMPORAL_STARTED',child.pid,flush=True)
    rc=child.wait()
save(c.OUT/'process-exit.json',dict(updated=now(),pid=child.pid,exit_code=rc))
raise SystemExit(rc)
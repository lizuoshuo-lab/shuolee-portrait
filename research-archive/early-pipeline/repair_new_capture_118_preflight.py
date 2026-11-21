"""Resume this run's immutable decode after a logging-only pre-SfM code error."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,signal,subprocess,time
P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
R=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
python='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python'
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
status=json.loads((R/'status.json').read_text())
assert status['state']=='failed' and status['stage']=='sfm' and status['completed']==['extract']
assert "multiple values for keyword argument 'raw_fisheye_coefficients_applied'" in (R/'logs/sfm.log').read_text()
assert not (P/'.gpu-task.lock').exists()
assert not list((R/'sfm').rglob('*.*')) and not (R/'sfm/database.db').exists()
lock=open('/tmp/shuolee-new-capture-118.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(R/'sfm').rename(R/'sfm-before-logging-fix')
(R/'logs/sfm.log').rename(R/'logs/sfm-before-logging-fix.log')
fix=dict(updated=now(),reason='Duplicate dict keyword in camera initialization receipt; feature extraction and geometry had not started.',
    changed='dict(**initial, ...) to dict(initial, ...) to explicitly replace duplicated metadata only.',
    decoded_sha256=hashlib.sha256((R/'decoded.json').read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((P/'src/shuolee_portrait/video_sfm.py').read_bytes()).hexdigest(),
    thresholds_changed=False,old_subject_or_camera_or_gaussian_loaded=False,same_run_raw_decode_reused=True)
save(O/'preflight-code-repair.json',fix)
completed=['extract'];start=time.monotonic();code=0
stages=['sfm','observations','material-tracks','alignment-diagnostics']
for stage in stages:
 command=[python,str(P/'run.py'),'_stage','--stage',stage,'--output',str(R),'--assets','/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets']
 save(R/'status.json',dict(state='running',stage=stage,completed=completed,updated=now(),quality_accepted=False))
 with (R/'logs'/f'{stage}.log').open('x') as log:
  proc=subprocess.Popen(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
      env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
  while proc.poll() is None:
   save(O/'queue-state.json',dict(state='running',stage=stage,pid=proc.pid,supervisor_pid=os.getpid(),updated=now(),
       seconds=time.monotonic()-start,completed=completed,command=command,training_started=False,quality_accepted=False,run_root=str(R)))
   if time.monotonic()-start>1800:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    break
   time.sleep(5)
  code=proc.wait()
 if code:break
 completed.append(stage)
state='preflight_complete_pending_review' if not code else 'preflight_failed'
save(R/'status.json',dict(state=state,stage=stage,completed=completed,updated=now(),exit_code=code,training_started=False,quality_accepted=False))
save(O/'queue-state.json',dict(state=state,stage=stage,completed=completed,updated=now(),exit_code=code,pid=None,
    run_root=str(R),training_started=False,quality_accepted=False))
print(json.dumps(dict(state=state,completed=completed,exit_code=code)),flush=True)
raise SystemExit(code)

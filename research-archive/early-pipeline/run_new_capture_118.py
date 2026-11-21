"""Bounded one-shot supervisor for the registered 118 preprocessing run."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,json,os,signal,subprocess,time

project=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
record=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
root=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
python='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python'
command=[python,str(project/'run.py'),'reconstruct','--video','/mnt/d/00/工作/数据集/df776c03-91a4_1792572380952.mp4',
 '--assets','/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets','--config',str(project/'configs/new-capture-118.json'),
 '--output',str(root),'--stop-after','alignment-diagnostics']
def now():return datetime.now(timezone.utc).isoformat()
def save(x):
 p=record/'queue-state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)

assert not root.exists(),'Fresh output required; do not resume or duplicate.'
lock=open('/tmp/shuolee-new-capture-118.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=time.monotonic()
with (record/'supervisor.log').open('x') as log:
 env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 proc=subprocess.Popen(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 print(json.dumps(dict(supervisor_pid=os.getpid(),pid=proc.pid,command=command,log=str(record/'supervisor.log'))),flush=True)
 timed_out=False
 while proc.poll() is None:
  status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'environment_check'}
  save(dict(updated=now(),state='running',pid=proc.pid,supervisor_pid=os.getpid(),command=command,
      stage=status.get('stage'),completed=status.get('completed',[]),seconds=time.monotonic()-start,
      training_started=False,quality_accepted=False,run_root=str(root)))
  if time.monotonic()-start>1800:
   timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
   break
  time.sleep(5)
 code=proc.wait()
 status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
 save(dict(updated=now(),state='timeout' if timed_out else ('preflight_complete_pending_review' if code==0 else 'preflight_failed'),
    pid=None,supervisor_pid=os.getpid(),completed=status.get('completed',[]),stage=status.get('stage'),
    exit_code=code,command=command,seconds=time.monotonic()-start,training_started=False,
    quality_accepted=False,run_root=str(root),next_action='Inspect evidence and images before starting Gaussian training.'))
 print(json.dumps(dict(exit_code=code,timed_out=timed_out,status=status)),flush=True)
 raise SystemExit(code)

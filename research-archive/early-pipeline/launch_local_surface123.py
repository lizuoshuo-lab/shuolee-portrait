from pathlib import Path
import fcntl,json,os,subprocess,sys
from local_surface123 import ROOT,REC,HERE,read,save,now

if (REC/'worker-launch.json').exists():
    raise RuntimeError('A launch was already recorded. Inspect PID and log; never launch a duplicate.')
guard=open(ROOT/'worker.lock','a+')
fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
if any('local_surface123.py --run' in s for s in processes.splitlines()):
    raise RuntimeError('Existing surface123 worker command found.')
env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
         CUDA_HOME='/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8',
         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2')
env.pop('PYTHONPATH',None);env.pop('SHUOLEE_EXTRA_PYTHONPATH',None)
env['PATH']=env['CUDA_HOME']+'/bin:'+str(Path(sys.executable).parent)+':'+env.get('PATH','')
env['LD_LIBRARY_PATH']=env['CUDA_HOME']+'/lib64:'+env.get('LD_LIBRARY_PATH','')
log=ROOT/'worker.log'
cmd=[sys.executable,str(HERE/'local_surface123.py'),'--run']
with log.open('x') as output:
    p=subprocess.Popen(cmd,env=env,cwd=HERE,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                       start_new_session=True,close_fds=True)
    save(REC/'worker-launch.json',dict(created=now(),pid=p.pid,command=cmd,log=str(log),
        note='Sequential local worker; actual fit/surface/1000step review receipts required; no timer.'))
fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
print(json.dumps(read(REC/'worker-launch.json')),flush=True)

"""Foreground worker for a detached, durable local WSL training job."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,traceback

ROOT=Path('/opt/gs-slam-lab/msv2-demo-fullframe-20261104')
SCRIPTS=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
def stamp():return datetime.now(timezone.utc).isoformat()
def write(p,j):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2));t.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['original','interpolation2x','swinir4x','stress4','publish']);a=p.parse_args()
    env=os.environ.copy();env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    ex=ROOT/'experiment.py';review=SCRIPTS/'review_fullframe_msv2_demo.py'
    commands=[]
    if a.job in ['original','interpolation2x','swinir4x']:
        scale={'original':1,'interpolation2x':2,'swinir4x':4}[a.job]
        if scale==4:commands.extend([(ex,'superresolve','--scale','4'),(ex,'cache','--scale','4')])
        commands.extend([(ex,'preflight','--scale',str(scale)),(ex,'train','--scale',str(scale)),
                         (review,'export','--scale',str(scale)),(review,'evaluate','--scale',str(scale))])
    elif a.job=='stress4':commands=[(ex,'stress4','--scale','4')]
    else:commands=[(review,'publish')]
    state=ROOT/(a.job+'-job.json');j=dict(state='running',job=a.job,pid=os.getpid(),started=stamp(),completed=[]);write(state,j)
    try:
        for i,args in enumerate(commands):
            j.update(active=[str(x) for x in args],updated=stamp());write(state,j)
            log=ROOT/(a.job+f'-job-{i+1:02d}.log')
            with log.open('w') as f:subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
            j['completed'].append([str(x) for x in args]);write(state,j)
        j.update(state='completed',finished=stamp());write(state,j)
    except BaseException:
        j.update(state='failed',finished=stamp(),error=traceback.format_exc());write(state,j);raise

if __name__=='__main__':main()

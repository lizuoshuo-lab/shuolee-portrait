"""Sequential, logged GPU jobs; saved live PID for interruption-safe recovery."""
from pathlib import Path
import sys,json,time,subprocess,os
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029';scripts=LAB/'scripts/dynamic_human';root='/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor1_4';statefile=out/'pipeline-state.json'
assert json.loads((out/'same-index/first-frame/completion.json').read_text())['step']==6000
jobs=[('offset-first','train_enerf_first_frame.py',['--out',str(out/'offset-index'),'--steps','6000'],out/'offset-index/first-frame/completion.json')]
for group in ['same-index','offset-index']:
    jobs.append((group+'-motion','train_enerf_motion.py',['--out',str(out/group),'--manifest',str(out/group/'appearance-manifest.json'),'--root',root,'--iterations','600'],out/group/'motion/completion.json'))
jobs.append(('offset-static','train_enerf_multitime_static.py',['--out',str(out/'offset-index'),'--manifest',str(out/'offset-index/appearance-manifest.json'),'--root',root,'--steps','14400'],out/'offset-index/static-multitime/completion.json'))
history=[]
def save(x):statefile.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
for name,script,args,done in jobs:
    if done.exists():history.append(dict(stage=name,status='already_completed',marker=str(done)));continue
    command=[sys.executable,'-u',str(scripts/script)]+args;log=out/f'{name}.log';began=time.time()
    with log.open('a') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        save(dict(status='running',parent_pid=os.getpid(),stage=name,child_pid=process.pid,command=command,log=str(log),started_unix=began,history=history));print('START',name,'pid',process.pid,flush=True)
        code=process.wait()
    history.append(dict(stage=name,exit_code=code,elapsed_seconds=time.time()-began,marker_exists=done.exists()))
    if code!=0 or not done.exists():save(dict(status='failed',parent_pid=os.getpid(),stage=name,history=history));raise SystemExit(code or 1)
    print('DONE',name,round(time.time()-began,1),flush=True)
save(dict(status='all_training_complete_evaluation_pending',parent_pid=os.getpid(),history=history));print('ALL_TRAINING_COMPLETE',flush=True)

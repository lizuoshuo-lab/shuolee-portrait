"""Bounded dependency queue for this already-running training job only."""
from pathlib import Path
import os,json,time,subprocess,sys,traceback,datetime
LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-4k4d-official-035/continuation-10000'
start=time.time();lock=OUT/'evaluation-queue.lock'
with lock.open('x') as f:f.write(str(os.getpid()))
state=dict(status='waiting_for_training_completion',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=[])
def save():
    state['elapsed_seconds']=time.time()-start
    temp=OUT/'evaluation-queue-status.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(OUT/'evaluation-queue-status.json')
def run(label,args,env=None):
    state['status']=label;save()
    with (OUT/f'evaluation-queue-{label}.log').open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW,timeout=1800)
    state['steps'].append(dict(step=label,returncode=result.returncode));save()
    assert result.returncode==0,(label,result.returncode)
try:
    save()
    while not (OUT/'training-completion.json').is_file():
        assert time.time()-start<3*3600,'Bounded wait expired; training was not terminated.'
        assert not (OUT/'training-failure.json').is_file(),'Training failure recorded; evaluation not started.'
        try:
            live=json.loads((OUT/'training-live.json').read_text())
            state['last_training_update']=live.get('history_updates')
            assert live['status']!='failed','Live training failure'
        except json.JSONDecodeError:pass
        save();time.sleep(20)
    time.sleep(10)
    run('linux_state_audit_and_native_render',['wsl','-d','Ubuntu-20.04','--','bash','/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/run_4k4d_continuation_eval.sh'])
    env=dict(os.environ,FOURKD_RUN_DIR=str(OUT))
    run('cpu_image_audit',[sys.executable,str(LAB/'scripts/dynamic_human/audit_4k4d_native_images.py')],env)
    run('matched_native_comparison',[sys.executable,str(LAB/'scripts/dynamic_human/compare_4k4d_continuation.py')])
    state['status']='render_and_audits_complete_visual_review_pending'
    state['goal_complete']=False
    save()
except Exception:
    state.update(status='failed',error=traceback.format_exc(),goal_complete=False);save();raise
finally:
    # Retain lock and status as evidence: accidental duplicate queues must fail.
    save()

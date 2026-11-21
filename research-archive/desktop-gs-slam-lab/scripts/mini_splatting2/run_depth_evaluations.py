"""Run the frozen follow-up plan only after the serial GPU training queue ends."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def main():
    state_path=ROOT/'native-depth-evaluation-queue.json'
    assert not state_path.exists(), 'Inspect the existing queue and process before restarting'
    plan_path=ROOT/'native-depth-runtime/evaluation-plan.json'
    plan=json.loads(plan_path.read_text())
    scripts=Path(__file__).parent
    state=dict(status='waiting_for_training',started=time.time(),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        script_sha256={s['script']:hashlib.sha256((scripts/s['script']).read_bytes()).hexdigest() for s in plan['scripts']},completed=[])
    def save():state_path.write_text(json.dumps(state,indent=2))
    save()
    try:
        while True:
            queue=json.loads((ROOT/plan['after_queue']).read_text())
            if queue['status']=='complete':break
            if queue['status']=='failed':raise RuntimeError('Upstream training failed; no dependent evaluation started')
            time.sleep(15)
        state.update(status='running',training_completed_at=queue.get('finished'));save()
        for step in plan['scripts']:
            script=step['script']
            assert hashlib.sha256((scripts/script).read_bytes()).hexdigest()==state['script_sha256'][script], 'Evaluation source changed after plan was frozen'
            state['current']=script;save()
            log=ROOT/'native-depth-runtime'/f'{Path(script).stem}.log'
            assert not log.exists(), f'Preserve existing evaluation log: {log}'
            start=time.time()
            with log.open('w') as f:
                result=subprocess.run([sys.executable,str(scripts/script),*step['arguments']],stdout=f,stderr=subprocess.STDOUT)
            entry=dict(script=script,returncode=result.returncode,seconds=time.time()-start,log=str(log))
            state['completed'].append(entry);save();print(json.dumps(entry),flush=True)
            if result.returncode:raise RuntimeError(f'{script} failed; inspect preserved log')
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as ex:
        state.update(status='failed',error=str(ex),finished=time.time());save();raise


if __name__=='__main__':main()

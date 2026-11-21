"""Run one bounded depth-reinitialization ablation after the Mip queue."""
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS=Path(__file__).resolve().parent;ROOT=SCRIPTS.parents[1]/'outputs/mini-splatting2-mip-depth-001'

def main():
    name='native-motion-mip-depthmask-18000-s0';mp=ROOT/'native-depthmask-queue.json'
    state=dict(status='waiting_for_native_mip',started=time.time(),current=name)
    def save():mp.write_text(json.dumps(state,indent=2))
    save()
    while True:
        q=json.loads((ROOT/'native-mip-queue.json').read_text())
        if q['status']=='complete':break
        if 'failed' in q['status']:state['status']='upstream_failed';save();return 1
        time.sleep(15)
    state['status']='running';save()
    for script,args in [('run_native_depthmask.py',[]),('evaluate_native_mip.py',[name])]:
        with (ROOT/f'{name}-{script}.log').open('w') as log:
            p=subprocess.run([sys.executable,str(SCRIPTS/script),*args],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:state.update(status='failed',script=script,returncode=p.returncode);save();return p.returncode
    control=json.loads((ROOT/'native-motion-mip-18000-s0/evaluation-summary.json').read_text())
    candidate=json.loads((ROOT/name/'evaluation-summary.json').read_text())
    result=dict(status='one_seed_evaluated',control=control,candidate=candidate,
        differences={k:candidate['native_mean'][k]-control['native_mean'][k] for k in control['native_mean']},
        intervention='training motion mask additionally excludes depth-reinitialization samples',
        monocular_depth_prior_added=False,visual_review='pending')
    (ROOT/'native-depthmask-summary.json').write_text(json.dumps(result,indent=2))
    state.update(status='complete',finished=time.time(),visual_review='pending');save()

if __name__=='__main__':raise SystemExit(main())

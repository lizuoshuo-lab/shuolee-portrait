"""Sequential full Mip experiments after the existing repeat queue releases GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
SCRIPTS=Path(__file__).resolve().parent

def main():
    path=ROOT/'native-mip-queue.json'
    state=dict(status='waiting_for_native_repeats',started=time.time(),jobs=[])
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    while True:
        queue=json.loads((ROOT/'native-repeat-queue.json').read_text())
        if queue['status']=='complete':break
        if 'failed' in queue['status']:
            state.update(status='blocked_by_repeat_failure',repeat_status=queue);save();return 1
        time.sleep(15)
    state['status']='running';save()
    for masked in [True,False]:
        name=f'native-{"motion-" if masked else ""}mip-18000-s0'
        state['current']=name;save()
        if not (ROOT/name).exists():
            cmd=[sys.executable,str(SCRIPTS/'run_native_mip.py')]+(['--masked'] if masked else [])
            with (ROOT/f'{name}-runner.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:state.update(status='training_failed',returncode=r.returncode);save();return r.returncode
        assert json.loads((ROOT/name/'manifest.json').read_text())['status']=='complete'
        with (ROOT/name/'evaluate.log').open('w') as log:
            r=subprocess.run([sys.executable,str(SCRIPTS/'evaluate_native_mip.py'),name],stdout=log,stderr=subprocess.STDOUT)
        state['jobs'].append(dict(name=name,evaluation_returncode=r.returncode));save()
        if r.returncode:state.update(status='evaluation_failed');save();return r.returncode
    state.update(status='complete',finished=time.time());save()

if __name__=='__main__':raise SystemExit(main())

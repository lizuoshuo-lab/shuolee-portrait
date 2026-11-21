"""CPU report preparation after the serial GPU acceptance queue has finished."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS=Path(__file__).parent
ROOT=SCRIPTS.parents[1]/'outputs/mini-splatting2-mip-depth-001'


def main():
    target=ROOT/'native-spread-publication-queue.json';assert not target.exists()
    files=['evaluate_spread_view_groups.py','build_native_spread_report.py','native_spread_report.html']
    frozen={n:hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest() for n in files}
    state=dict(status='waiting_for_gpu_audits',started=time.time(),script_sha256=frozen,
        completed=[],browser_verified=False,remote_delivered=False)
    def save():target.write_text(json.dumps(state,indent=2))
    save();logs=ROOT/'native-spread-publication-logs';logs.mkdir(exist_ok=False)
    try:
        while True:
            upstream=json.loads((ROOT/'native-spread-acceptance-queue.json').read_text())
            if upstream['status']=='complete':break
            if upstream['status']=='failed':raise RuntimeError('Upstream evaluation failed; preserve and inspect it')
            time.sleep(15)
        assert all(hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest()==sha for n,sha in frozen.items())
        for script in files[:2]:
            state.update(status='running',current=script);save();started=time.time()
            with (logs/(script+'.log')).open('x',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,str(SCRIPTS/script)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(script+' failed; see retained log')
            state['completed'].append(dict(script=script,seconds=time.time()-started));save()
            print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete_pending_browser_review',finished=time.time());state.pop('current',None);save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise


if __name__=='__main__':main()

"""Queue one diagnostic after the primary native and secondary GPU evaluations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

S=Path(__file__).parent;ROOT=S.parents[1]/'outputs/mini-splatting2-mip-depth-001'

def main():
    path=ROOT/'native-training-fit-queue.json';assert not path.exists()
    script=S/'evaluate_native_training_fit.py'
    state=dict(status='waiting_for_primary_evaluation',started=time.time(),
        question='Does late densification mainly change fitting of training photos or also improve primary heldout photos?',
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),training_updated=False,accepted=False)
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
    save()
    try:
        while True:
            q=json.loads((ROOT/'native-late-publication-queue.json').read_text())
            if q['status']=='failed':raise RuntimeError('Primary publication failed; do not start secondary GPU work')
            if q['status']=='complete_pending_browser_and_visual_review':break
            time.sleep(20)
        assert hashlib.sha256(script.read_bytes()).hexdigest()==state['script_sha256']
        state.update(status='running');save()
        with (ROOT/'native-training-fit.log').open('x') as log:
            subprocess.run([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='complete_pending_review',finished=time.time());save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise

if __name__=='__main__':main()

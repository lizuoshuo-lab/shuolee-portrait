"""Wait for both native resolution evaluations; then create visual comparison artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001';SCRIPTS=Path(__file__).parent


def main():
    target=ROOT/'native-resolution-publication-queue.json';assert not target.exists()
    state=dict(status='waiting_for_training',started=time.time(),completed=[],accepted=False)
    def save():target.write_text(json.dumps(state,indent=2))
    save()
    try:
        source=(SCRIPTS/'render_native_spread_paths.py').read_text()
        start=source.index('NAMES=');end=source.index('\n\n\n',start)
        source=source[:start]+"NAMES={'Low-res training':'native-resolution-low-18000-s0','Full-res training':'native-resolution-full-18000-s0'}"+source[end:]
        assert "out=ROOT/'native-spread-paths-001'" in source and 'width=(3*w+1)' in source
        source=source.replace("out=ROOT/'native-spread-paths-001'","out=ROOT/'native-resolution-paths-001'")
        source=source.replace("round(reference['width']/2)","reference['width']").replace("round(reference['height']/2)","reference['height']")
        source=source.replace('width=(3*w+1)','width=(2*w+1)').replace('render(model,c,divisor=2,mode=','render(model,c,divisor=1,mode=')
        script=SCRIPTS/'render_native_resolution_paths.py';assert not script.exists();script.write_text(source)
        state['script_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [script,SCRIPTS/'build_resolution_report.py',SCRIPTS/'native_resolution_report.html']};save()
        while True:
            q=json.loads((ROOT/'native-resolution-queue.json').read_text())
            if q['status']=='failed':raise RuntimeError('Training/evaluation queue failed: '+q.get('error','inspect log'))
            if q['status']=='complete':break
            time.sleep(20)
        for path in [script,SCRIPTS/'build_resolution_report.py']:
            state.update(status='running',current=path.name);save()
            with (ROOT/('resolution-'+path.stem+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,str(path)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(path.name+' failed; inspect preserved log')
            state['completed'].append(path.name);save()
        state.update(status='complete_pending_browser_and_visual_review',finished=time.time());state.pop('current',None);save()
        print(json.dumps(state),flush=True)
    except Exception as e:
        state.update(status='failed',error=str(e),finished=time.time());save();raise


if __name__=='__main__':main()

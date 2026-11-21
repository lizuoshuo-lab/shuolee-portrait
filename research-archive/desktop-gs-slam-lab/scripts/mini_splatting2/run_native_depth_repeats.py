"""Matched guided-model repeats after baseline seed variance became material."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from run_native_depth_stage import ROOT,run


def main():
    target=ROOT/'native-depth-repeat-queue.json'
    assert not target.exists(),'Do not restart or overwrite an existing repeat queue'
    jobs=[('native-depthguide003-18000-s1',1,.003),('native-depthguide003-18000-s2',2,.003),
          ('native-depthguide010-18000-s1',1,.01),('native-depthguide010-18000-s2',2,.01)]
    state=dict(status='waiting_for_evaluations',started=time.time(),planned=jobs,completed=[],
        rationale='Baseline matched seed1 exceeds both guided seed0 full-image PSNR values; repeat both pre-existing weights without selecting a winner',
        weights_and_seeds_frozen_before_new_results=True,
        runner_sha256=hashlib.sha256(Path(__file__).with_name('run_native_depth_stage.py').read_bytes()).hexdigest())
    def save():target.write_text(json.dumps(state,indent=2))
    save()
    try:
        while True:
            evaluation=json.loads((ROOT/'native-depth-evaluation-queue.json').read_text())
            if evaluation['status']=='complete':break
            if evaluation['status']=='failed':raise RuntimeError('Upstream evaluation failed; inspect before starting repeats')
            time.sleep(15)
        state.update(status='rendering_paths',current='render_native_depth_paths.py');save()
        with (ROOT/'native-depth-runtime/render_native_depth_paths.log').open('x') as f:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('render_native_depth_paths.py'))],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Shared path render failed; inspect preserved log')
        state.update(status='running');save()
        for name,seed,coefficient in jobs:
            state['current']=name;save()
            state['completed'].append(run(name,seed,coefficient));save()
            print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as ex:
        state.update(status='failed',error=str(ex),finished=time.time());save();raise


if __name__=='__main__':main()

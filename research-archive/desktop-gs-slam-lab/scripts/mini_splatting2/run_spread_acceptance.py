"""Wait for all predeclared variance training jobs, then serialize GPU audits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS=Path(__file__).parent
ROOT=SCRIPTS.parents[1]/'outputs/mini-splatting2-mip-depth-001'
NAMES={'reference0':'native-depthguide003-18000-s0',
    'control0':'native-depthspread000-18000-s0',
    'spread0030':'native-depthspread003-18000-s0',
    'spread0100':'native-depthspread010-18000-s0'}


def main():
    target=ROOT/'native-spread-acceptance-queue.json';assert not target.exists()
    jobs=[('additional24',['evaluate_additional_frames.py','--output-name','additional-frame-spread-004']+
              [item for key,name in NAMES.items() for item in ['--variant',key+'='+name]]),
        ('sparse-geometry',['evaluate_native_geometry.py','--output','native-spread-geometry-001']+list(NAMES.values())),
        ('common-depth-spread',['evaluate_depth_spread_candidates.py','--output','depth-spread-candidates-002']),
        ('head-and-aggregate',['analyze_spread_candidates.py']),
        ('distance-pitch-paths',['render_native_spread_paths.py'])]
    helpers=['local_refinement.py','depth_ray_moments.py','evaluate_head_detail.py']
    hashes={name:hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest()
        for name in set([a[0] for _,a in jobs]+helpers)}
    state=dict(status='waiting_for_training',started=time.time(),jobs=jobs,completed=[],
        script_sha256=hashes,models=NAMES,seed=0,
        scope='Predeclared development comparison; no independent surface or wide-pitch ground truth')
    def save():target.write_text(json.dumps(state,indent=2))
    save();logs=ROOT/'native-spread-acceptance-logs';logs.mkdir(exist_ok=False)
    try:
        while True:
            q=json.loads((ROOT/'native-spread-queue.json').read_text())
            if q['status']=='complete':break
            if q['status']=='failed':raise RuntimeError('Upstream variance training failed; inspect preserved logs')
            time.sleep(15)
        assert all(hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'Evaluation scripts changed after freeze'
        state.update(status='running');save()
        for stage,args in jobs:
            state['current']=stage;save();started=time.time()
            with (logs/(stage+'.log')).open('x') as log:
                r=subprocess.run([sys.executable,str(SCRIPTS/args[0])]+args[1:],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(stage+' failed; inspect preserved log')
            state['completed'].append(dict(stage=stage,elapsed=time.time()-started));save()
            print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as e:
        state.update(status='failed',error=str(e),finished=time.time());save();raise


if __name__=='__main__':
    main()

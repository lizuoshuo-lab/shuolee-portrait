"""Serialize the repeat evaluation and read-only mechanism audit after training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
SCRIPTS=Path(__file__).parent


def main():
    target=ROOT/'depth-repeat-acceptance-queue.json';assert not target.exists()
    new=[f'native-depthguide{weight}-18000-s{seed}' for weight in ['003','010'] for seed in [1,2]]
    jobs=[('matched-native', ['analyze_native_depth_repeats.py']),
        ('moment-validation',['depth_ray_moments.py','--gpu','--output',str(ROOT/'depth-spread-research-001/gpu-validation.json')]),
        ('additional24',['evaluate_additional_frames.py','--output-name','additional-frame-depth-repeats-003']+
            [item for name in new for item in ['--variant',name+'='+name]]),
        ('geometry',['evaluate_native_geometry.py','--output','native-depth-geometry-repeats-002']+new),
        ('ray-depth-spread',['evaluate_depth_spread.py','--output','depth-spread-evaluation-001'])]
    helpers=['depth_ray_moments.py','local_refinement.py']
    hashes={name:hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest() for name in set([args[0] for _,args in jobs]+helpers)}
    state=dict(status='waiting_for_training',started=time.time(),jobs=jobs,completed=[],script_sha256=hashes,
        reason='Repeat acceptance is evaluated only after all4 predeclared guided runs have finished; GPU stages are serial')
    def save():target.write_text(json.dumps(state,indent=2))
    save();logdir=ROOT/'depth-repeat-acceptance-logs';logdir.mkdir(exist_ok=False)
    try:
        while True:
            training=json.loads((ROOT/'native-depth-repeat-queue.json').read_text())
            if training['status']=='complete':break
            if training['status']=='failed':raise RuntimeError('Training failed; inspect before evaluation')
            time.sleep(15)
        assert all(hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items()),'Frozen scripts changed'
        state.update(status='running');save()
        for key,args in jobs:
            state['current']=key;save();started=time.time()
            with (logdir/(key+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,str(SCRIPTS/args[0])]+args[1:],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{key} failed; see preserved log')
            state['completed'].append(dict(stage=key,elapsed=time.time()-started));save()
            print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise


if __name__=='__main__':main()

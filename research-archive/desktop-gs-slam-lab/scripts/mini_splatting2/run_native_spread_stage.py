"""Smoke-test then serialize a zero-weight control and two full variance runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');REPO=RUNTIME/'repo-depthspread'
BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001');SCRIPTS=Path(__file__).parent


def run(name,coefficient,steps=18000):
    out=ROOT/name;out.mkdir(exist_ok=False)
    config=(REPO/'config/fast').read_text()
    if steps!=18000:config=config.replace('iterations=18_000',f'iterations={steps}')
    (out/'config').write_text(config)
    start,ramp=(4000,1000) if steps==18000 else (100,200)
    env=os.environ.copy();env.update(PYTHONPATH=f'{RUNTIME}/deps:{REPO}:{BASE}/deps',OMP_NUM_THREADS='4',
        MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'),MINIGS_DEPTH_PRIORS=str(ROOT/'depth-priors'),
        MINIGS_DEPTH_WEIGHT='.003',MINIGS_DEPTH_SEED='0',MINIGS_SPREAD_WEIGHT=str(coefficient),
        MINIGS_SPREAD_START=str(start),MINIGS_SPREAD_RAMP=str(ramp))
    script=REPO/'msv2/train_depth_spread.py'
    command=[sys.executable,str(script),'-s',str(BASE/'dataset'),'-m',str(out/'model'),'-r','1','--eval',
        '--imp_metric','indoor','--config_path',str(out/'config'),'--test_iterations','-1','--port','6121']
    manifest=dict(status='running',started=time.time(),seed=0,depth_weight=.003,spread_weight=coefficient,
        steps=steps,spread_start=start,spread_ramp=ramp,command=command,
        code_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((REPO/'native_depth_spread.py').read_bytes()).hexdigest(),
        intervention='robust relative center-depth variance regularization; not2DGSsurface loss',
        compared_to='native-depthspread000-18000-s0')
    target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('x') as log:result=subprocess.run(command,cwd=REPO/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,elapsed_seconds=time.time()-manifest['started'])
    if result.returncode==0:
        raw=out/f'model/point_cloud/iteration_{steps}/point_cloud.ply';baked=raw.with_name('point_cloud_mip3d_baked.ply')
        manifest.update(ply=str(raw),baked_ply=str(baked),ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest())
        usage=json.loads((out/'model/spread-guidance-usage.json').read_text());manifest['spread_usage']=usage
        if coefficient>0:
            assert usage['active_calls']>0 and all(int(n)%8!=0 for n in usage['training_frames_used'])
            assert usage['last_records'] and all(r['supervised_fraction']>0 for r in usage['last_records'])
        else:assert usage['active_calls']==0
    target.write_text(json.dumps(manifest,indent=2))
    if result.returncode:raise RuntimeError(name+' failed; inspect preserved train.log')
    if steps==18000:
        with (out/'evaluate.log').open('x') as log:
            result=subprocess.run([sys.executable,str(SCRIPTS/'evaluate_native_mip.py'),name],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' evaluation failed')
    return dict(name=name,training='complete',evaluation='complete' if steps==18000 else 'smoke_only',elapsed=manifest['elapsed_seconds'])


def main():
    path=ROOT/'native-spread-queue.json';assert not path.exists()
    assert json.loads((ROOT/'native-spread-runtime/validation.json').read_text())['status']=='passed'
    jobs=[('native-depthspread000-18000-s0',0.),('native-depthspread003-18000-s0',.003),('native-depthspread010-18000-s0',.01)]
    state=dict(status='smoke',started=time.time(),planned=jobs,completed=[],seed=0,coefficients_frozen_before_results=True,
        rationale='Mean-depth constraints alone increased head center-depth spread across3seeds without improving head fidelity')
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    try:
        state['current']='native-depthspread-smoke-600-s0';save();state['smoke']=run(state['current'],.01,600);save()
        state['status']='running';save()
        for name,weight in jobs:
            state['current']=name;save();state['completed'].append(run(name,weight));save();print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise


if __name__=='__main__':main()

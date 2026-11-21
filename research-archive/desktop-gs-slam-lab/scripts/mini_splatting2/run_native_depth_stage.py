"""Smoke-test then run two depth strengths and matched baseline repeats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');REPO=RUNTIME/'repo-depthguide'
BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001')

def run(name,seed,coefficient,steps=18000):
    out=ROOT/name;out.mkdir(exist_ok=False)
    config=out/'config';text=(REPO/'config/fast').read_text()
    if steps!=18000:text=text.replace('iterations=18_000',f'iterations={steps}')
    config.write_text(text)
    env=os.environ.copy();env.update(PYTHONPATH=f'{RUNTIME}/deps:{REPO}:{BASE}/deps',OMP_NUM_THREADS='4',
        MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'),MINIGS_DEPTH_PRIORS=str(ROOT/'depth-priors'),
        MINIGS_DEPTH_WEIGHT=str(coefficient),MINIGS_DEPTH_SEED=str(seed))
    script=REPO/'msv2/train_depth_guided.py'
    cmd=[sys.executable,str(script),'-s',str(BASE/'dataset'),'-m',str(out/'model'),'-r','1','--eval',
        '--imp_metric','indoor','--config_path',str(config),'--test_iterations','-1','--port','6121']
    mp=out/'manifest.json';manifest=dict(status='running',seed=seed,depth_weight=coefficient,steps=steps,command=cmd,
        started=time.time(),intervention='training-only confidence-weighted inverse expected z; native Mip plus RGB/depth-reinitialization motion masks',
        compared_to=f'native-motion-mip-depthmask-18000-s{seed}',
        code_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((REPO/'native_depth_guidance.py').read_bytes()).hexdigest())
    mp.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('w') as f:result=subprocess.run(cmd,cwd=REPO/'msv2',env=env,stdout=f,stderr=subprocess.STDOUT)
    manifest.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,elapsed_seconds=time.time()-manifest['started'])
    if result.returncode==0:
        raw=out/f'model/point_cloud/iteration_{steps}/point_cloud.ply';baked=raw.with_name('point_cloud_mip3d_baked.ply')
        manifest.update(ply=str(raw),baked_ply=str(baked),ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest())
        usage=json.loads((out/'model/depth-guidance-usage.json').read_text());manifest['depth_usage']=usage
        if coefficient>0:
            assert usage['active_calls']>0
            assert all(int(n)%8!=0 for n in usage['training_frames_used'])
            assert usage['last_records'] and all(r['supervised_fraction']>0 and r['weighted_loss']>=0 for r in usage['last_records'])
    mp.write_text(json.dumps(manifest,indent=2))
    if result.returncode:raise RuntimeError(f'{name}: training failed, inspect train.log')
    if steps==18000:
        with (out/'evaluate.log').open('w') as f:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_native_mip.py')),name],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name}: evaluation failed')
    return dict(name=name,training='complete',evaluation='complete' if steps==18000 else 'not_applicable',elapsed=manifest['elapsed_seconds'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    assert json.loads((ROOT/'native-depth-runtime/validation.json').read_text())['status']=='passed'
    if a.smoke:
        print(json.dumps(run('native-depth-smoke-600-s0',0,.01,600)),flush=True);return
    assert json.loads((ROOT/'native-depth-smoke-600-s0/manifest.json').read_text())['status']=='complete'
    path=ROOT/'native-depth-stage-queue.json'
    jobs=[('native-depthguide003-18000-s0',0,.003),('native-depthguide010-18000-s0',0,.01),
          ('native-motion-mip-depthmask-18000-s1',1,0.),('native-motion-mip-depthmask-18000-s2',2,0.)]
    assert not path.exists(),'Do not overwrite a queue; inspect existing processes and job state'
    q=dict(status='running',started=time.time(),planned=jobs,completed=[])
    def save():path.write_text(json.dumps(q,indent=2))
    save()
    try:
        for name,seed,coefficient in jobs:
            q['current']=name;save();q['completed'].append(run(name,seed,coefficient));save()
            print(json.dumps(q['completed'][-1]),flush=True)
        q.update(status='complete',finished=time.time());save()
    except Exception as ex:
        q.update(status='failed',error=str(ex),finished=time.time());save();raise

if __name__=='__main__':main()

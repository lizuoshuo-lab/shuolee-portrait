"""Run an isolated native Mini-Splatting2 smoke/full baseline; no source overwrite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-001')

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,choices=[32,18000],required=True);args=p.parse_args()
    name=f'native-baseline-{args.steps:05d}-s0';out=ROOT/name;out.mkdir(exist_ok=False)
    runtime=json.loads((ROOT/'native-runtime/manifest.json').read_text());assert runtime['status']=='complete'
    repo=RUNTIME/'repo';dataset=RUNTIME/'dataset'
    if not dataset.exists():shutil.copytree(LAB/'outputs/mini-splatting2-computer00-001/hover-sfm-001/dataset',dataset)
    config=out/'config';config.write_text((repo/'config/fast').read_text().replace('iterations=18_000',f'iterations={args.steps}'))
    target=out/'model'
    env=os.environ.copy();env.update(PYTHONPATH=str(RUNTIME/'deps')+os.pathsep+str(repo),OMP_NUM_THREADS='4',TORCH_CUDA_ARCH_LIST='12.0')
    command=[sys.executable,str(repo/'msv2/train.py'),'-s',str(dataset),'-m',str(target),'-r','1','--eval',
        '--imp_metric','indoor','--config_path',str(config),'--test_iterations','-1','--port','6119']
    manifest=dict(status='running',command=command,steps=args.steps,seed=0,
        source_commit=runtime['source_commit'],compatibility_patches=runtime['compatibility_patches'],
        method='unmodified native Mini-Splatting2 training algorithm',
        data_split='official LLFF holdout every 8 frames; same 51/8 as frozen prior baseline',
        execution_stack='Torch2.7.1/CUDA12.8 on RTX5080; differs from original RTX3090 reproduction',
        train_evaluation_disabled=True,started=time.time())
    mp=out/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('w') as log:
        result=subprocess.run(command,cwd=repo/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(returncode=result.returncode,elapsed_seconds=time.time()-manifest['started'],status='complete' if result.returncode==0 else 'failed')
    ply=target/f'point_cloud/iteration_{args.steps}/point_cloud.ply'
    if result.returncode==0:
        if not ply.exists():raise RuntimeError('Successful exit but expected model missing')
        manifest['ply']=str(ply);manifest['ply_sha256']=hashlib.sha256(ply.read_bytes()).hexdigest()
    mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())

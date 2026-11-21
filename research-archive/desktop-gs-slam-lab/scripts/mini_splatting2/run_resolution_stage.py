"""Matched lossless low/full-resolution native training; no new loss module."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001');REPO=RUNTIME/'repo-depthguide'
BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001');SCRIPTS=Path(__file__).parent
PAIR=ROOT/'source-resolution-rectified-002'


def run(name,arm,steps):
    out=ROOT/name;out.mkdir(exist_ok=False)
    config=(REPO/'config/fast').read_text().replace('iterations=18_000',f'iterations={steps}')
    (out/'config').write_text(config)
    dataset=PAIR/('dataset' if arm=='full' else 'matched-low-resolution-control')
    env=os.environ.copy();env.update(PYTHONPATH=f'{RUNTIME}/deps:{REPO}:{BASE}/deps',OMP_NUM_THREADS='4',
        MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'),MINIGS_DEPTH_PRIORS=str(ROOT/'depth-priors'),
        MINIGS_DEPTH_WEIGHT='.003',MINIGS_DEPTH_SEED='0')
    script=REPO/'msv2/train_depth_guided.py'
    command=[sys.executable,str(script),'-s',str(dataset),'-m',str(out/'model'),'-r','1','--eval',
        '--data_device','cpu','--imp_metric','indoor','--config_path',str(out/'config'),
        '--test_iterations','-1','--port','6121']
    manifest=dict(status='running',started=time.time(),arm=arm,seed=0,steps=steps,dataset=str(dataset),
        depth_weight=.003,spread_weight=0.,head_attention=False,diffusion=False,command=command,
        code_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        resolution=[1514,2694] if arm=='full' else [757,1347],
        image_residency='cpu in both arms; existing trainer transfers RGB to CUDA',
        source_protocol=str(ROOT/'native-resolution-protocol-001.json'),
        compared_to='native-resolution-low-18000-s0')
    target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2))
    with (out/'train.log').open('x') as log:
        result=subprocess.run(command,cwd=REPO/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,
        elapsed_seconds=time.time()-manifest['started'])
    if result.returncode==0:
        raw=out/f'model/point_cloud/iteration_{steps}/point_cloud.ply';baked=raw.with_name('point_cloud_mip3d_baked.ply')
        manifest.update(ply=str(raw),baked_ply=str(baked),ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest())
        usage=json.loads((out/'model/depth-guidance-usage.json').read_text())
        assert usage['active_calls']>0 and all(int(n)%8!=0 for n in usage['training_frames_used'])
        depth_samples=json.loads((out/'model/depth-mask-usage.json').read_text())
        assert len(depth_samples)==51 and all(x['sampled_masked_pixels']==0 for x in depth_samples)
        manifest.update(depth_usage=usage,depth_reinitialization_validated=True)
    target.write_text(json.dumps(manifest,indent=2))
    if result.returncode:raise RuntimeError(name+' failed; preserved train.log')
    if steps==18000:
        with (out/'evaluate-resolution.log').open('x') as log:
            result=subprocess.run([sys.executable,str(SCRIPTS/'evaluate_resolution_stage.py'),name],
                cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' evaluation failed')
    return dict(name=name,training='complete',evaluation='complete' if steps==18000 else 'smoke_only',
        elapsed_seconds=manifest['elapsed_seconds'])


def main():
    assert json.loads((ROOT/'native-resolution-probe-001/summary.json').read_text())['status']=='passed'
    assert (SCRIPTS/'evaluate_resolution_stage.py').exists()
    path=ROOT/'native-resolution-queue.json';assert not path.exists()
    protocol=dict(created=time.time(),status='frozen_before_training',
        question='Does retaining original video sampling improve human detail with the same native method?',
        intervention='Lossless rectified1514x2694 versus matched area-downsampled757x1347',
        shared=dict(frames=59,train=51,development_holdout=8,seed=0,iterations=18000,depth_weight=.003,
            motion_masks='same masks scaled nearest-neighbor',depth_priors='same existing training-only aligned predictions; confidence-weighted resampling',
            camera_poses='unchanged',sfm_points='unchanged',image_cache='cpu',mip='same3D/2Dmethod, sampling filter naturally depends on training resolution'),
        not_added=['variance_loss','head_attention','diffusion','new_poses','new_geometry'],
        evaluation='Both models rendered at BOTH1514x2694and757x1347against identical lossless real references',
        primary='same frozen head/shoulder rectangle; PSNR plus normalized reference gradient error',
        secondary='full image, person rectangle, all8view consistency, fixed camera paths',
        acceptance='A single seed cannot establish a stable improvement. Repeat a useful result and inspect artifacts.',
        caveat='Input resolution also affects Mip3D sampling and native topology; this is the complete resolution intervention, not isolated texture information.',
        raw_input_001_rejected=True,rectified_input_002_verified=True,
        training_source_hashes=hashlib.sha256((PAIR/'resolution-ablation-inputs.json').read_bytes()).hexdigest(),
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SCRIPTS/'evaluate_resolution_stage.py']})
    (ROOT/'native-resolution-protocol-001.json').write_text(json.dumps(protocol,indent=2))
    jobs=[('native-resolution-low-18000-s0','low',18000),('native-resolution-full-18000-s0','full',18000)]
    state=dict(status='smoke',started=time.time(),planned=jobs,completed=[])
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    try:
        state['current']='native-resolution-full-smoke-2200-s0';save()
        state['smoke']=run(state['current'],'full',2200);state['status']='running';save()
        for name,arm,steps in jobs:
            state['current']=name;save();state['completed'].append(run(name,arm,steps));save()
            print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as e:
        state.update(status='failed',error=str(e),finished=time.time());save();raise


if __name__=='__main__':main()

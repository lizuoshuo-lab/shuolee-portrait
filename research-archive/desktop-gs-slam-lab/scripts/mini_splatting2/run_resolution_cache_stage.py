"""Matched resolution intervention with bounded reference caches and joint backward."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]; ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001'); REPO=RUNTIME/'repo-depth-cache'
BASE=Path('/opt/gs-slam-lab/mini-splatting2-native-001'); SCRIPTS=Path(__file__).parent
PAIR=ROOT/'source-resolution-rectified-002'
PROTOCOL=ROOT/'native-resolution-cache-protocol-001.json'


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)); temporary.replace(path)


def library_audit(pid):
    paths=set()
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        if '/diff_gaussian_rasterization' in line and '.so' in line:
            paths.add(line.split()[-1])
    if not paths:
        return None
    assert all(str(RUNTIME/'deps') in x for x in paths),paths
    return dict(pid=pid,created=time.time(),libraries=[dict(path=x,sha256=hashlib.sha256(Path(x).read_bytes()).hexdigest()) for x in sorted(paths)])


def run(name,arm,steps):
    out=ROOT/name; out.mkdir(exist_ok=False)
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
        guidance_sha256=hashlib.sha256((REPO/'native_depth_guidance.py').read_bytes()).hexdigest(),
        resolution=[1514,2694] if arm=='full' else [757,1347],source_protocol=str(PROTOCOL),
        memory_runtime=str(ROOT/'native-resolution-cache-runtime/manifest.json'),
        reference_cache_validation=str(ROOT/'native-resolution-cache-runtime/validation.json'),
        image_residency='CPU',reference_gpu_cache_bound=4,joint_backward_unchanged=True,
        compared_to='native-resolution-cache-low-18000-s0')
    target=out/'manifest.json';dump(target,manifest)
    with (out/'train.log').open('x') as log:
        process=subprocess.Popen(command,cwd=REPO/'msv2',env=env,stdout=log,stderr=subprocess.STDOUT)
        manifest['pid']=process.pid;dump(target,manifest)
        audited=False
        while process.poll() is None:
            if not audited:
                try:
                    audit=library_audit(process.pid)
                    if audit:
                        dump(out/'training-runtime.json',audit);audited=True
                except FileNotFoundError:
                    pass
                except Exception:
                    process.terminate();process.wait();raise
            time.sleep(10)
        code=process.returncode
    manifest.update(status='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.time()-manifest['started'])
    dump(target,manifest)
    if code:
        raise RuntimeError(name+' failed; preserved train.log')
    assert audited
    raw=out/f'model/point_cloud/iteration_{steps}/point_cloud.ply';baked=raw.with_name('point_cloud_mip3d_baked.ply')
    usage=json.loads((out/'model/depth-guidance-usage.json').read_text())
    assert usage['active_calls']>0 and all(int(n)%8!=0 for n in usage['training_frames_used'])
    depth_samples=json.loads((out/'model/depth-mask-usage.json').read_text())
    assert len(depth_samples)==51 and all(x['sampled_masked_pixels']==0 for x in depth_samples)
    memory=[json.loads(x) for x in (out/'model/memory-usage.jsonl').read_text().splitlines()]
    assert memory and all(x['prior_cache_entries']<=4 and x['motion_cache_entries']<=4 for x in memory)
    manifest.update(ply=str(raw),baked_ply=str(baked),ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest(),depth_usage=usage,
        depth_reinitialization_validated=True,
        memory_summary=dict(max_allocated_mib=max(x['allocated_mib'] for x in memory),
            max_reserved_mib=max(x['reserved_mib'] for x in memory),
            peak_allocated_mib=max(x['peak_allocated_mib'] for x in memory),last=memory[-1]))
    dump(target,manifest)
    if steps==3300:
        transition=[x for x in memory if x['iteration']>=3100]
        assert len(transition)>=3
        assert max(x['allocated_mib'] for x in transition)<13000,'Stop before long training: live tensors exceed conservative VRAM gate'
        assert max(x['reserved_mib'] for x in transition)<14000,'Stop before long training: reservation still approaches16GB'
        assert manifest['elapsed_seconds']<900,'Stop before long training: smoke unexpectedly slow'
    if steps==18000:
        with (out/'evaluate-resolution-v2.log').open('x') as log:
            result=subprocess.run([sys.executable,str(SCRIPTS/'evaluate_resolution_stage.py'),name],
                cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(name+' evaluation failed')
    return dict(name=name,training='complete',evaluation='complete' if steps==18000 else 'transition_smoke_passed',
        elapsed_seconds=manifest['elapsed_seconds'],memory_summary=manifest['memory_summary'])


def main():
    validation=json.loads((ROOT/'native-resolution-cache-runtime/validation.json').read_text())
    assert validation['status']=='passed' and validation['checks']==106
    runtime=json.loads((ROOT/'native-resolution-cache-runtime/manifest.json').read_text())
    for relative,digest in runtime['files'].items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest()==digest
    path=ROOT/'native-resolution-cache-queue.json';assert not path.exists() and not PROTOCOL.exists()
    protocol=dict(created=time.time(),status='frozen_before_training',
        question='Does preserving original video resolution improve human detail under the same native method?',
        intervention='Lossless rectified1514x2694 versus matched area-downsampled757x1347',
        shared=dict(frames=59,train=51,development_holdout=8,seed=0,iterations=18000,depth_weight=.003,
            camera_poses='unchanged',sfm_points='unchanged',motion_masks='same nearest-neighbor resampling',
            depth_priors='same existing training-only aligned predictions and confidence resampling',
            image_cache='CPU',gpu_reference_cache_bound=4,backward='original joint RGB+depth',
            mip='same3D/2Dmethod; sampling filter naturally depends on training resolution'),
        not_added=['variance_loss','head_attention','diffusion','new_poses','new_geometry'],
        evaluation='Both models rendered at BOTH1514x2694and757x1347with validated native Mip kernels against identical real references',
        primary='frozen head/shoulder rectangle PSNR and normalized reference gradient error',
        secondary='full image, person rectangle, consistency across8views, fixed camera paths',
        acceptance='Development result only; repeat useful results and inspect real images before acceptance',
        caveat='Resolution affects information, Mip sampling and native topology; not isolated texture information',
        replaces_interrupted_queue='native-resolution-queue.json',legacy_evaluator_v1_excluded=True,
        sequential_backward_rejected=True,cache_validation=str(ROOT/'native-resolution-cache-runtime/validation.json'),
        training_source_hashes=hashlib.sha256((PAIR/'resolution-ablation-inputs.json').read_bytes()).hexdigest(),
        runtime_hashes=runtime['files'],
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SCRIPTS/'evaluate_resolution_stage.py']})
    dump(PROTOCOL,protocol)
    jobs=[('native-resolution-cache-full-18000-s0','full',18000),('native-resolution-cache-low-18000-s0','low',18000)]
    state=dict(status='smoke',started=time.time(),planned=jobs,completed=[],protocol=str(PROTOCOL))
    def save():dump(path,state)
    save()
    try:
        state['current']='native-resolution-cache-full-smoke-3300-s0';save()
        state['smoke']=run(state['current'],'full',3300);state['status']='running';save()
        for name,arm,steps in jobs:
            state['current']=name;save();state['completed'].append(run(name,arm,steps));save()
            print(json.dumps(state['completed'][-1]),flush=True)
        state.update(status='complete',finished=time.time());state.pop('current',None);save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise


if __name__=='__main__':main()

"""Original schedule, isolated normalized head RGB-L1 weighting, gated before18k."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from run_resolution_cache_stage import dump, library_audit, ROOT, RUNTIME, BASE, SCRIPTS, PAIR

REPO = RUNTIME/'repo-depth-head'
PROTOCOL = ROOT/'native-head-protocol-001.json'


def run(name, steps):
    out = ROOT/name; out.mkdir(exist_ok=False)
    config = (REPO/'config/fast').read_text().replace('iterations=18_000', f'iterations={steps}')
    (out/'config').write_text(config)
    dataset = PAIR/'dataset'
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{RUNTIME}/deps:{REPO}:{BASE}/deps', OMP_NUM_THREADS='4',
               MINIGS_MOTION_MASKS=str(ROOT/'motion-masks-001'), MINIGS_DEPTH_PRIORS=str(ROOT/'depth-priors'),
               MINIGS_DEPTH_WEIGHT='.003', MINIGS_DEPTH_SEED='0',
               MINIGS_HEAD_REGIONS=str(ROOT/'training-head-regions-003'), MINIGS_HEAD_WEIGHT='2')
    script = REPO/'msv2/train_depth_guided.py'
    command = [sys.executable, str(script), '-s', str(dataset), '-m', str(out/'model'), '-r', '1', '--eval',
               '--data_device', 'cpu', '--imp_metric', 'indoor', '--config_path', str(out/'config'),
               '--test_iterations', '-1', '--port', '6121']
    manifest = dict(status='running', started=time.time(), arm='full', seed=0, steps=steps, dataset=str(dataset),
                    depth_weight=.003, spread_weight=0., head_attention=True, head_weight=2., diffusion=False,
                    late_densification=False, original_topology_schedule=True, command=command,
                    code_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), resolution=[1514,2694],
                    source_protocol=str(PROTOCOL), runtime='native-head-runtime-001/manifest.json',
                    image_residency='CPU', reference_gpu_cache_bound=4, joint_backward_unchanged=True,
                    compared_to='native-resolution-cache-full-18000-s0', accepted=False)
    target = out/'manifest.json'; dump(target, manifest)
    with (out/'train.log').open('x') as log:
        process = subprocess.Popen(command, cwd=REPO/'msv2', env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest['pid'] = process.pid; dump(target, manifest); audited = False
        while process.poll() is None:
            if not audited:
                try:
                    audit = library_audit(process.pid)
                    if audit: dump(out/'training-runtime.json', audit); audited = True
                except FileNotFoundError: pass
                except Exception:
                    process.terminate(); process.wait(); raise
            time.sleep(10)
        code = process.returncode
    manifest.update(status='complete' if code==0 else 'failed', returncode=code, elapsed_seconds=time.time()-manifest['started'])
    dump(target, manifest)
    if code: raise RuntimeError(name+' training failed; inspect preserved train.log')
    assert audited
    raw = out/f'model/point_cloud/iteration_{steps}/point_cloud.ply'
    baked = raw.with_name('point_cloud_mip3d_baked.ply')
    usage = json.loads((out/'model/depth-guidance-usage.json').read_text())
    assert usage['active_calls']>0 and all(int(n)%8!=0 for n in usage['training_frames_used'])
    samples = json.loads((out/'model/depth-mask-usage.json').read_text())
    assert len(samples)==51 and all(row['sampled_masked_pixels']==0 for row in samples)
    head = json.loads((out/'model/head-guidance-usage.json').read_text())
    assert head['coefficient']==2 and len(head['training_frames_used'])==51
    assert all(int(n)%8!=0 for n in head['training_frames_used'])
    audits = [json.loads(line) for line in (out/'model/head-training-audit.jsonl').read_text().splitlines()]
    assert len(audits)==steps//100 and all(row['finite'] and row['cache_entries']<=4 for row in audits)
    memory = [json.loads(line) for line in (out/'model/memory-usage.jsonl').read_text().splitlines()]
    assert memory and all(row['prior_cache_entries']<=4 and row['motion_cache_entries']<=4 for row in memory)
    manifest.update(ply=str(raw), baked_ply=str(baked), ply_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                    baked_ply_sha256=hashlib.sha256(baked.read_bytes()).hexdigest(), depth_usage=usage, head_usage=head,
                    depth_reinitialization_validated=True,
                    memory_summary=dict(max_allocated_mib=max(row['allocated_mib'] for row in memory),
                                        max_reserved_mib=max(row['reserved_mib'] for row in memory),
                                        peak_allocated_mib=max(row['peak_allocated_mib'] for row in memory), last=memory[-1]))
    dump(target, manifest)
    if steps==3300:
        transition = [row for row in memory if row['iteration']>=3100]
        assert len(transition)==3
        assert max(row['allocated_mib'] for row in transition)<13000
        assert max(row['reserved_mib'] for row in transition)<14000
        assert manifest['elapsed_seconds']<900
        assert all([row['width'],row['height']]==[1514,2694] for row in audits if row['iteration']>=3100)
    if steps==18000:
        with (out/'evaluate-resolution-v2.log').open('x') as log:
            result = subprocess.run([sys.executable, str(SCRIPTS/'evaluate_resolution_stage.py'), name], cwd=REPO,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(name+' native evaluation failed')
    return dict(name=name, training='complete', evaluation='complete' if steps==18000 else 'transition_smoke_passed',
                elapsed_seconds=manifest['elapsed_seconds'], memory_summary=manifest['memory_summary'])


def main():
    runtime_path = ROOT/'native-head-runtime-001/manifest.json'
    runtime = json.loads(runtime_path.read_text())
    assert runtime['neutral_ast_equivalent'] and runtime['ssim_depth_joint_backward_byte_identical']
    for part in ['unchanged_files','changed_files']:
        for path, digest in runtime[part].items(): assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest
    check = json.loads((ROOT/runtime['interface_validation']).read_text()); assert check['status']=='interface_validated'
    assert not PROTOCOL.exists()
    protocol = dict(status='frozen_before_training', created=time.time(), baseline='native-resolution-cache-full-18000-s0',
                    intervention=runtime['intervention'], coefficient=2., head_regions='training-head-regions-003',
                    formula='sum(valid*(1+H)*abs(I-G))/(3*sum(valid*(1+H)))',
                    shared=dict(images='source-resolution-rectified-002/dataset', resolution=[1514,2694],
                                seed=0, iterations=18000, depth_weight=.003, ssim_weight=.2, mip=True,
                                camera_poses='unchanged', motion_masks='unchanged', backward='joint RGB+depth',
                                topology='Original MiniGS2 fast schedule; no late densification'),
                    smoke_steps=3300, source_hashes=runtime, native_parameter_repeatability_unresolved=True,
                    preserved_failed_gates=runtime['rejected_gates_preserved'],
                    evaluation='Frozen8 development views at both target resolutions, existing24-frame supplement, fixed distance/pitch paths',
                    primary_metrics=['head.psnr','head.normalized_gradient_error'],
                    promotion_rule='Require visual improvement without new path artifacts and consistent primary plus24frame gains before repetition. Never accept by training loss alone.',
                    caveat='Training rectangles include neck/background. Weight normalization redistributes relative contribution. One same-height video lacks large-pitch or metric-depth GT.',
                    additional_methods=[], accepted=False)
    dump(PROTOCOL, protocol)
    queue = ROOT/'native-head-queue.json'; assert not queue.exists()
    state = dict(status='smoke', started=time.time(), completed=[], accepted=False, protocol=str(PROTOCOL))
    def save(): dump(queue,state)
    save()
    try:
        state['current']='native-head-full-smoke-3300-s0'; save()
        state['smoke']=run(state['current'],3300)
        dump(ROOT/'native-head-runtime-001/smoke.json', dict(status='passed', result=state['smoke']))
        runtime.update(status='smoke_passed', smoke='native-head-runtime-001/smoke.json'); dump(runtime_path,runtime)
        state.update(status='training', current='native-head-full-18000-s0'); save()
        state['completed'].append(run(state['current'],18000))
        state.update(status='complete_pending_review', finished=time.time()); state.pop('current',None); save()
        print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time()); save(); raise


if __name__=='__main__': main()

"""Gate a single isolated native topology candidate through a full transition smoke."""
import hashlib
import json
import os
from pathlib import Path
import time
import run_resolution_cache_stage as runner

ROOT=runner.ROOT;REPO=runner.RUNTIME/'repo-depth-late';PROTOCOL=ROOT/'native-late-protocol-001.json'

def main():
    runtime=json.loads((ROOT/'native-late-runtime-001/manifest.json').read_text())
    assert runtime['neutral_ast_equivalent'] and runtime['joint_loss_and_backward_byte_identical']
    for part in ['unchanged_files','changed_files']:
        for path,digest in runtime[part].items():assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest
    assert not PROTOCOL.exists()
    protocol=dict(status='frozen_before_training',created=time.time(),baseline='native-resolution-cache-full-18000-s0',
        baseline_protocol='native-resolution-cache-protocol-001.json',intervention=runtime['intervention'],
        shared=dict(images='source-resolution-rectified-002/dataset',resolution=[1514,2694],seed=0,iterations=18000,
                    camera_poses='unchanged',depth_weight=.003,mip=True,motion_masks='unchanged',backward='joint RGB+depth'),
        no_head_attention=True,no_diffusion=True,no_variance_loss=True,late_until=5000,smoke_steps=5100,
        caveat=runtime['caveat'],acceptance='Report all8 development views at both target resolutions plus previously inspected24frames. Candidate remains unaccepted until visual review and repetitions.',
        source_hashes=runtime,primary_metrics=['head.psnr','head.normalized_gradient_error'],
        decision='No claim of geometric accuracy without GT. Compare to identical full-resolution baseline.',
        expected_topology_events=list(range(3100,5000,100)))
    runner.dump(PROTOCOL,protocol)
    runner.REPO=REPO;runner.PROTOCOL=PROTOCOL
    os.environ['MINIGS_LATE_DENSIFY_UNTIL']='5000'
    queue=ROOT/'native-late-queue.json';assert not queue.exists()
    state=dict(status='smoke',started=time.time(),completed=[],accepted=False,protocol=str(PROTOCOL))
    def save():runner.dump(queue,state)
    save()
    try:
        name='native-late-full-smoke-5100-s0';state['current']=name;save()
        result=runner.run(name,'full',5100)
        rows=[json.loads(x) for x in (ROOT/name/'model/late-topology-audit.jsonl').read_text().splitlines()]
        assert [r['iteration'] for r in rows]==list(range(3000,5200,100))
        assert all(r['status']=='passed' for r in rows)
        assert all([r['width'],r['height']]==[1514,2694] for r in rows if r['iteration']>=3100)
        assert any(rows[i]['gaussians']!=rows[i-1]['gaussians'] for i in range(1,len(rows)))
        assert max(r['allocated_mib'] for r in rows)<13000
        assert result['memory_summary']['max_reserved_mib']<14000
        assert result['elapsed_seconds']<1800
        smoke=dict(result=result,topology_rows=rows,status='passed',checks=['nonfinite tensors','topology lengths','optimizer moments','full-resolution camera','memory bound','runtime bound'])
        runner.dump(ROOT/'native-late-runtime-001/smoke.json',smoke)
        runtime.update(status='smoke_passed',smoke=str(ROOT/'native-late-runtime-001/smoke.json'));runner.dump(ROOT/'native-late-runtime-001/manifest.json',runtime)
        state.update(smoke=result,status='training',current='native-late-full-18000-s0');save()
        result=runner.run(state['current'],'full',18000)
        p=ROOT/state['current']/'manifest.json';m=json.loads(p.read_text())
        m.update(compared_to=protocol['baseline'],late_densification_until=5000,topology_smoke='native-late-runtime-001/smoke.json');runner.dump(p,m)
        state['completed'].append(result);state.update(status='complete_pending_review',finished=time.time());state.pop('current',None);save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished=time.time());save();raise

if __name__=='__main__':main()

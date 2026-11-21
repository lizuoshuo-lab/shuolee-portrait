"""Remove a duplicate outer GPU lock; retain the original sequence's lock."""
import argparse
import inspect
from pathlib import Path
import run_sofa_temporal_warmstart124 as base

FAILED=base.OUT
base.OUT=FAILED.with_name('sofa-temporal-warmstart-v2')

def corrected_run():
    source=inspect.getsource(base.run)
    line="    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
    assert source.count(line)==1
    original=(Path(base.core.__file__).parent/'run_official_sequence123.py').read_text()
    assert "open('/tmp/shuolee-native122-gpu.lock'" in original and 'fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)' in original
    scope=dict(base.run.__globals__)
    exec(compile(source.replace(line,''),'<remove-duplicate-outer-lock-retain-upstream-lock>','exec'),scope)
    return scope['run']

def prepare():
    failure=base.read(FAILED/'failure.json')
    assert failure['computed_frames']==0 and 'BlockingIOError' in failure['error']
    assert not (FAILED/'official-fit').exists()
    corrected_run();base.prepare()
    p=base.read(base.OUT/'plan.json')
    p['sources'][str(Path(__file__).resolve())]=base.sha(__file__)
    p['implementation_repair']=dict(previous_output=str(FAILED),previous_fit_calls=0,
        cause='Outer wrapper and original sequence took the same advisory lock through different file descriptors in one process.',
        change='Remove duplicate outer acquisition. The original sequence GPU lock remains enforced.',
        algorithm_parameters_changed=False,failed_plan_sha256=base.sha(FAILED/'plan.json'))
    base.save(base.OUT/'plan.json',p)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args();(prepare if args.mode=='prepare' else corrected_run())()

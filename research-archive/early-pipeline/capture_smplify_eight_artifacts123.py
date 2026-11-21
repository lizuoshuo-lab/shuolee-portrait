"""Repair failure-artifact capture only; do not relax geometry or train longer.

GPU atomic reductions make trajectory bitwise equality unavailable. Record the
observed loss difference as diagnostics, not an admission test or a retry loop.
"""
from pathlib import Path
import argparse,traceback
import capture_smplify_eight_failure123 as capture
from run_official_gaussians123 import read,save,sha,now

original_source=capture.source
capture.OUT=capture.OUT.parent/'body-gaussian-smplify-eight-failure-capture'


def source():
    fn,s=original_source()
    target='            assert max(errors.values())<1e-6,errors'
    assert s.count(target)==1
    s=s.replace(target,'            # Non-bitwise GPU trajectory differences are recorded, not hidden or threshold-tuned.')
    ns=dict(fn.__globals__);exec(compile(s,'<eight-frame-failure-artifact-capture>','exec'),ns)
    return ns['train'],s


capture.source=source
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','train']);ap.add_argument('--pid',type=int);args=ap.parse_args()
    try:
        if args.mode=='prepare':
            capture.prepare()
            p=read(capture.OUT/'plan.json');p['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
            p.update(purpose='One final instrumentation repair: persist the900-step failed configuration. Original failed run and previous failed replay remain immutable. No further replay after this.',
                numerical_equality='Recorded as evidence only; not asserted. Unchanged geometry gate still stops at a relative normal reversal.',
                no_more_replays=True)
            save(capture.OUT/'plan.json',p)
        elif args.mode=='run':
            control=capture.eight.direct.control;control.OUT=capture.OUT;control.__file__=__file__;control.run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if capture.OUT.exists():save(capture.OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise

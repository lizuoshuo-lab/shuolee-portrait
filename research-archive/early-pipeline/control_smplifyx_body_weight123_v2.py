"""Repair snapshot globals in the diagnostic adapter, retaining failed evidence."""
import argparse
import inspect
from pathlib import Path
import control_smplifyx_body_weight123 as control

b=control.b
FAILED=b.OUT
FAILED_PLAN=b.PLAN
FAILED_HANDOFF=b.HANDOFF
b.OUT=b.BODY/'worst-region-body-weight-control-v2'
b.PLAN=b.BODY/'worst-region-body-weight-control-v2-plan.json'
b.HANDOFF=b.BODY/'worst-region-body-weight-control-v2-gpu-handoff.json'


def live_config_fit():
    original=b.entry.fit_one
    source=inspect.getsource(original)
    old="jw = torch.ones((1,118), dtype=dtype, device=dev); jw[:,[1,9,12]] = 0"
    assert source.count(old)==1
    source=source.replace(old,"jw = torch.full((1,118), 2.0, dtype=dtype, device=dev); jw[:,[1,9,12]] = 0")
    compiled=compile(source,'<body-weight-only-live-entry-configuration>','exec')
    def fit(**kwargs):
        # Resolve config/trace/seed hooks after the original worker installs them.
        scope=dict(original.__globals__)
        exec(compiled,scope)
        return scope['fit_one'](**kwargs)
    return fit


control.patched_fit=live_config_fit


def prepare():
    failure=b.read(FAILED/'failure.json')
    assert failure['error']=='AssertionError()' and b.read(FAILED_HANDOFF)['state']=='resumed'
    control.prepare();p=b.read(b.PLAN)
    p['source_hashes'][str(Path(__file__).resolve())]=b.sha(__file__)
    for path in [FAILED_PLAN,FAILED/'failure.json',FAILED/'official-result.pkl',FAILED_HANDOFF]:
        p['input_hashes'][str(path)]=b.sha(path)
    p['implementation_repair']=dict(previous_output=str(FAILED),previous_fit_executed_but_invalid=True,
        cause='Copied function globals resolved the original five-stage config before worker hooks installed the registered two-stage config/seed/trace.',
        correction='Copy the live module globals at fit invocation after config/seed/trace hooks are installed.',
        previous_result_not_used=True,numeric_weight_change_from_registered_plan=False,failed_case_preserved=True)
    b.save(b.PLAN,p)


def run(pid):
    scope=dict(b.run.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(b.run),'<existing-exclusive-corrected-body-weight-lease>','exec'),scope)
    scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:control.worker()

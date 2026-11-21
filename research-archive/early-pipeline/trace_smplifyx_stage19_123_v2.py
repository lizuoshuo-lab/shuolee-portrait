"""Repair the diagnostic's missing-YAML-default assertion; solver unchanged."""
import argparse
import ast
import inspect
from pathlib import Path
import trace_smplifyx_stage19_123 as base

FAILED_OUT, FAILED_PLAN, FAILED_HANDOFF = base.OUT, base.PLAN, base.HANDOFF
base.OUT = base.BODY/'stage19-original-trajectory-v2'
base.PLAN = base.BODY/'stage19-original-trajectory-v2-plan.json'
base.HANDOFF = base.BODY/'stage19-original-trajectory-v2-gpu-handoff.json'


def corrected_worker():
    source=inspect.getsource(base.worker)
    old="c['side_view_thsh']==25"
    assert source.count(old)==1
    source=source.replace(old,"c.get('side_view_thsh',25)==25")
    scope=dict(base.worker.__globals__)
    exec(compile(source,'<diagnostic-default-assertion-only-repair>','exec'),scope)
    return scope['worker']


def prepare():
    failure=base.read(FAILED_OUT/'failure.json')
    assert failure['error']=="KeyError('side_view_thsh')"
    assert base.read(FAILED_HANDOFF)['state']=='resumed'
    official=Path(base.arguments().official_root)/'smplifyx/fit_single_frame.py'
    function=next(n for n in ast.parse(official.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fit_single_frame')
    defaults=dict(zip([v.arg for v in function.args.args][-len(function.args.defaults):],function.args.defaults))
    assert ast.literal_eval(defaults['side_view_thsh'])==25
    corrected_worker()
    base.prepare()
    plan=base.read(base.PLAN)
    plan['source_hashes'][str(Path(__file__).resolve())]=base.sha(__file__)
    plan['input_hashes'].update({str(p):base.sha(p) for p in [FAILED_PLAN,FAILED_OUT/'failure.json',FAILED_HANDOFF]})
    plan['implementation_repair']=dict(previous_failure='Diagnostic asserted a config key omitted by upstream YAML; original function default is25.',
        previous_solve_started=False, previous_original_fitter_calls=0, solver_parameters_changed=False,
        change="Use c.get('side_view_thsh',25) in diagnostic assertion only; no extra config field.")
    base.save(base.PLAN,plan)
    print('PREPARED_STAGE19_DEFAULT_ASSERTION_REPAIR',flush=True)


def run(pid):
    source=inspect.getsource(base.run)
    scope=dict(base.run.__globals__,__file__=__file__)
    exec(compile(source,'<existing-gpu-handoff-corrected-diagnostic-entry>','exec'),scope)
    scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:corrected_worker()()

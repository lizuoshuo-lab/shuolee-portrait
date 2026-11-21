"""Match the upstream default data-weight list to the registered tail stages."""
import argparse
import inspect
from pathlib import Path
import control_smplifyx_warmtail19_123 as control

b=control.b
FAILED=b.OUT
FAILED_HANDOFF=b.HANDOFF
b.OUT=b.BODY/'stage19-warmstart-original-tail-v2'
b.PLAN=b.BODY/'stage19-warmstart-original-tail-v2-plan.json'
b.HANDOFF=b.BODY/'stage19-warmstart-original-tail-v2-gpu-handoff.json'
original_config=b.entry.load_official_config


def explicit_original_default(root):
    cfg=original_config(root)
    assert cfg.get('data_weights') is None
    source=(Path(root)/'smplifyx/fit_single_frame.py').read_text()
    assert "if data_weights is None:\n        data_weights = [1, ] * 5" in source
    cfg['data_weights']=[1]*5
    return cfg


b.entry.load_official_config=explicit_original_default
control.FIELDS=control.FIELDS+['data_weights']


def prepare():
    assert 'number of data term weights 5' in b.read(FAILED/'failure.json')['error']
    assert b.read(FAILED_HANDOFF)['state']=='resumed'
    control.prepare();p=b.read(b.PLAN)
    p['source_hashes'][str(Path(__file__).resolve())]=b.sha(__file__)
    p['implementation_repair']=dict(previous_output=str(FAILED),optimizer_updates_before_failure=0,
        description='Slice the original implicit data_weights=[1]*5 along with the other original stage lists. No numeric weight changed.')
    for path in [FAILED/'failure.json',FAILED_HANDOFF]:p['input_hashes'][str(path)]=b.sha(path)
    b.save(b.PLAN,p)


def run(pid):
    scope=dict(b.run.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(b.run),'<existing-handoff-tail-data-list-repair>','exec'),scope);scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:control.worker_function()()

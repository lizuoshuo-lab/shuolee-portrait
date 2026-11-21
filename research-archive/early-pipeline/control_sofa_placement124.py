"""One matched sofa16 control reusing the existing three-scalar placement adapter."""
import argparse,inspect,os,sys,time,subprocess,difflib
from pathlib import Path
import numpy as np
import control_unique_wrist124 as old
import run_grass_translation124 as placement

base=old.base
a,trace,entry=old.a,old.trace,old.entry
read,save,sha,now=old.read,old.save,old.sha,old.now
OUT=old.a.RUN.parent/'sofa-body-placement124'


def verify():
    p=read(OUT/'plan.json')
    for path,digest in p['frozen'].items():assert sha(path)==digest,path
    return p


def source(root):
    # Reuse the already checked placement adapter after the same original
    # body/root-seed hook. Do not alter either archived source file.
    previous=placement.BASE_SOURCE
    placement.BASE_SOURCE=trace.probe.adapted_source
    try:return placement.source(root)
    finally:placement.BASE_SOURCE=previous


def worker_source():
    original=inspect.getsource(old.worker)
    start=original.index('        fit_targets,counts=inputs.unique_wrist_observations')
    end=original.index('        loader=entry.module',start)
    code=original[:start]+'''        fit_targets=targets.copy()
        assert np.array_equal(fit_targets,targets)
        save(OUT/'input-delta.json',dict(updated=now(),changed_indices=[],all118_targets_and_confidences_unchanged=True))
'''+original[end:]
    replacements=[
        ('entry.adapted_entry_source,entry.load_official_config,entry.load_official=trace.probe.adapted_source,config,load',
         'entry.adapted_entry_source,entry.load_official_config,entry.load_official=placement_source,config,load'),
        ("for f in ('K_integer_centers','camera_translation','betas','faces')", "for f in ('K_integer_centers','betas','faces')"),
        ("checks['original_all_parameter_stages']=True", "checks['original_all_parameter_stages_plus_three_placement_scalars']=True\n        for stage in stages:\n            if stage['kind']=='body':\n                c=stage['_context']\n                assert sum(v is c['camera'].translation for v in c['optimizer'].param_groups[0]['params'])==1\n                assert c['camera'].translation.numel()==3"),
        ("unique_wrist_endpoint=newmetrics", "placement_endpoint=newmetrics"),
        ("original_report=report,diagnostic_only=True", "original_report=report,placement_delta=(arrays['camera_translation']-baseline['camera_translation']).tolist(),original_targets_unchanged=True,diagnostic_only=True"),
        ("UNIQUE_WRIST_CONTROL_COMPLETE", "SOFA_PLACEMENT_CONTROL_COMPLETE")]
    for before,after in replacements:
        assert code.count(before)==1,before
        code=code.replace(before,after)
    compile(code,'<same-sofa-replay-with-existing-placement-adapter>','exec')
    return code,''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='existing-matched-control',tofile='same-input-original-body-plus-placement'))


def prepare():
    assert not OUT.exists()
    assert read(a.RUN/'quality-stop-decision.json')['pid'] is None
    # Preserve the failed unique-wrist result as evidence, not as an initial state.
    assert read(old.OUT/'conclusion.json')['quality_accepted'] is False
    row,_=base.target()
    paths=[Path(__file__),Path(old.__file__),Path(placement.__file__),Path(base.__file__),Path(trace.__file__),
           Path(trace.probe.__file__),Path(a.__file__),Path(entry.__file__),Path(inspect.getsourcefile(base.stage_config)),
           Path(inspect.getsourcefile(base.install_loss_factory)),Path(trace.sequence.__file__),
           Path(inspect.getsourcefile(trace.probe.cache_static_topology)),
           base.OUT/'trajectory.npz',base.OUT/'result.json',base.OUT/'conclusion.json',
           a.RUN/'plan.json',a.frame(a.RUN,16),a.frame(a.RUN,17),Path(row['image']),Path(row['mask']),
           a.SRC/'data/inputs.json',a.SRC/'data/observations.npz',old.OUT/'conclusion.json',
           a.RUN.parent/'official-body-screen124/sofa/result.json']
    paths += [a.OFFICIAL/name for name in entry.PINNED]
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered',frame='frame_000016',diagnostic_only=True,
        hypothesis='Torso-only placement remains absent from the sofa body optimizer while pose must absorb all subsequent body/face/hand disagreement. Test the existing3placement scalars in the original optimizer, keeping K and observations fixed.',
        prior_evidence='Read-only47frame analysis: maxima at left knee21frames,right knee5,right elbow19,left shoulder2. Earlier unique-wrist and parameter-partition controls failed. The current grass selected-view adapter already optimizes placement; the older sofa sequence does not.',
        intervention='Reuse run_grass_translation124.source after the original seed hook: append camera.translation once to body_params and reset it between original orientation branches. This is foreground body placement, not background-camera calibration.',
        fixed=['Same recorded17 seed only for matched diagnosis, not new sequence initialization.','All118 original targets/confidences; no unique-wrist suppression.','Identical original torso initialization, fixed sharedbeta, nativeK, body/root temporal terms,stages3/4,maxiters30,losses,LBFGS.'],
        budget=dict(fitter_calls=1,frames=1,max_seconds=300),
        comparative_gate=read(old.OUT/'plan.json')['comparative_gate'],
        risks=['Placement may absorb geometry error or worsen depth/face/feet.','A lower loss or smaller2Dresidual cannot establish true3Dpose.'],
        stop='One matched fit and native body/face/hands/feet review only. No full-sequence resume, automatic GS, thresholds/weights sweep, or mixing winners.',
        frozen={str(p):sha(p) for p in paths},quality_accepted=False,gaussian_admitted=False))
    previous=getattr(entry,'_root_seed_original_source',None)
    entry._root_seed_original_source=entry.adapted_entry_source
    try:code,delta=source(a.OFFICIAL)
    finally:
        if previous is not None:entry._root_seed_original_source=previous
        else:delattr(entry,'_root_seed_original_source')
    (OUT/'placement-interface.diff').write_text(delta)
    worker,delta=worker_source();(OUT/'matched-replay-interface.diff').write_text(delta)
    # Original optimizer membership and gradient controls, with zero real fit steps.
    import torch
    _,_,_,factory=entry.load_official(a.OFFICIAL)
    body=torch.nn.Parameter(torch.zeros(2));t=torch.nn.Parameter(torch.tensor([[0.,0.,2.]]))
    optimizer,_=factory.create_optimizer([body,t],optim_type='lbfgsls',lr=1.,maxiters=30)
    assert sum(v is t for v in optimizer.param_groups[0]['params'])==1
    (t.square().sum()).backward();assert torch.isfinite(t.grad).all() and float(t.grad.abs().max())>0
    save(OUT/'preflight.json',dict(updated=now(),source_compiles=True,existing_placement_adapter_reused=True,
        one3scalar_parameter_membership=True,finite_nonzero_gradient=True,optimizer_steps=0,
        original_inputs_K_and_loss_unchanged=True,quality_accepted=False))
    print('SOFA_PLACEMENT_MATCHED_CONTROL_REGISTERED',flush=True)


def worker():
    scope=old.__dict__.copy();scope.update(OUT=OUT,verify=verify,placement_source=source)
    code,_=worker_source();exec(compile(code,'<same-sofa-replay-with-existing-placement-adapter>','exec'),scope)
    scope['worker']()


def run():
    import fcntl
    p=verify();assert (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker']
    with (OUT/'fit.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        save(OUT/'launch.json',dict(updated=now(),pid=child.pid,command=cmd,proc_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]))
        print('SOFA_PLACEMENT_CONTROL_STARTED',child.pid,flush=True)
        try:rc=child.wait(timeout=p['budget']['max_seconds'])
        except subprocess.TimeoutExpired:
            child.terminate();child.wait(timeout=10);save(OUT/'timeout.json',dict(updated=now(),pid=child.pid,budget_seconds=p['budget']['max_seconds']));raise
    save(OUT/'process-exit.json',dict(updated=now(),pid=child.pid,exit_code=rc))
    if rc:raise subprocess.CalledProcessError(rc,cmd)
    print('SOFA_PLACEMENT_CONTROL_FINISHED',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','worker']);args=ap.parse_args()
    dict(prepare=prepare,run=run,worker=worker)[args.mode]()

"""One controlled extension: optimize the existing root-placement translation.

Original SMPLify-X optimizer, objectives and five stages are retained. The
camera K and world extrinsics do not change. Frozen runs are never modified.
"""
import argparse
import difflib
import inspect
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import run_grass_temporal124 as grass
import official_smplifyx_entry123 as entry
from run_official_gaussians123 import read,save,sha,now

OUT=grass.ROOT/'grass-temporal-roottranslation'
PREVIOUS=grass.OUT
OFFICIAL=grass.REC/'smplifyx-reference-control/official'
BASE_SOURCE=entry.adapted_entry_source


def source(root):
    original,diff=BASE_SOURCE(root)
    changes=[
        ("        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):",
         "        _body_camera_start = camera.translation.detach().clone()\n        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):\n            with torch.no_grad():\n                camera.translation.copy_(_body_camera_start)"),
        ('                body_params = list(body_model.parameters())',
         '                body_params = list(body_model.parameters())\n                assert camera.translation.requires_grad and camera.translation.numel() == 3\n                assert all(p is not camera.translation for p in body_params)\n                body_params.append(camera.translation)'),
        ('            # Get the result of the fitting process',
         "            __import__('official_smplifyx_entry123')._translation_refine_trace = dict(\n                parameter_added='camera.translation', scalars=3,\n                initial=_body_camera_start.detach().cpu().numpy().copy(),\n                final=camera.translation.detach().cpu().numpy().copy(),\n                last_gradient=camera.translation.grad.detach().cpu().numpy().copy())\n            # Get the result of the fitting process")]
    changed=original
    for old,new in changes:
        assert changed.count(old)==1,old
        changed=changed.replace(old,new)
    compile(changed,'<smplifyx-original-plus-root-translation>','exec')
    return changed,diff+'\n'+''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile='original-frozen-body-placement',tofile='existing-three-placement-scalars-in-original-optimizer'))


def adapted_runner():
    original=inspect.getsource(grass.run)
    old="        assert np.array_equal(arrays['camera_translation'], np.asarray(trace['camera_translation']))"
    new="        verify_translation(frame, arrays, receipt, trace)"
    assert original.count(old)==1
    changed=original.replace(old,new)
    compile(changed,'<grass-temporal-new-placement-receipt>','exec')
    return changed,''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile='frozen-grass-runner',tofile='new-placement-parameter-receipt'))


def verify_translation(frame,arrays,receipt,trace):
    with np.load(grass.BASE/'frames'/frame/'winner-camera-space.npz',allow_pickle=False) as z:
        assert np.array_equal(trace['camera_translation'],z['camera_translation']), 'Original camera initialization changed.'
        assert np.array_equal(arrays['betas'],z['betas']) and np.array_equal(arrays['K_integer_centers'],z['K_integer_centers'])
    meta=entry._translation_refine_trace
    assert np.array_equal(meta['initial'],trace['camera_translation'])
    # The original selected-winner exporter is used; the trace is the last branch.
    assert np.isfinite(meta['last_gradient']).all()
    receipt['root_translation_refinement']=dict(parameter='existing camera.translation (body placement)',scalars=3,
        included_in_all_original_body_stages=True,background_world_extrinsics_unchanged=True,
        initial=meta['initial'].tolist(),final=arrays['camera_translation'].tolist(),
        final_minus_initial=(arrays['camera_translation']-meta['initial']).tolist(),
        last_branch_gradient=meta['last_gradient'].tolist(),winner_export_unchanged=True,
        camera_initialization_exact_baseline_parity=True)


def prepare():
    assert read(PREVIOUS/'temporal-progress.json')['pid'] is None
    if OUT.exists():
        assert not (OUT/'interface-check.json').exists() and not (OUT/'official-fit').exists()
        assert not (OUT/'plan-before-preflight-hook-repair.json').exists()
        (OUT/'plan-before-preflight-hook-repair.json').write_bytes((OUT/'plan.json').read_bytes())
        save(OUT/'preflight-repair.json',dict(updated=now(),error='Exact source hook omitted upstream tqdm wrapper.',
            fitted_frames=0,fix='Match pinned upstream orientation loop including tqdm; intended candidate unchanged.'))
    audit=read(PREVIOUS/'objective-audit/result.json')
    assert audit['optimizer_steps']==0 and not audit['quality_accepted']
    p=read(PREVIOUS/'plan.json')
    p.update(updated=now(),state='registered',
        hypothesis='Freezing body placement after torso-only initialization while freezing shared beta can force joint rotation to absorb placement mismatch. Test only adding the existing three translation scalars to the original five body optimizers.',
        causal_change='camera.translation joins original body_params; reset it to the identical torso initialization for each original orientation branch. K and background E remain fixed.',
        comparison='Previous grass temporal13 candidate, plus historical independent poses; same input hashes, reference32 beta, frame order, root transport, stages, objectives and temporal weights. Sequential downstream effects of the changed placement are part of this candidate.',
        risks=['More placement freedom can worsen depth or move the whole body.', 'No independent physical camera or3D pose truth; lower loss is not acceptance.'],
        stopping='One new candidate,13-frame body/face/feet and common-side checkpoint. Stop on repeated wrong raised-arm depth, new anatomy or projection regression. Continue to27 and94 only after actual quality passes. No weight sweep, old-pose initialization, selected-frame stitching or GS without full-roster review.',
        root_translation_parameter_free=True,old_root_translation_run=str(PREVIOUS),
        objective_audit=str(PREVIOUS/'objective-audit/result.json'),quality_accepted=False,gaussian_admitted=False)
    extra=[Path(__file__),PREVIOUS/'objective-audit/plan.json',PREVIOUS/'objective-audit/result.json']
    p['sources'].update({str(v):sha(v) for v in extra})
    save(OUT/'plan.json',p)
    code,delta=source(OFFICIAL);(OUT/'placement-adapter.diff').write_text(delta)
    code2,delta2=adapted_runner();(OUT/'runner-receipt.diff').write_text(delta2)
    # Preflight the same optimizer interface without fitting any real pose.
    import torch
    _,_,_,factory=entry.load_official(OFFICIAL)
    original=[torch.nn.Parameter(torch.zeros(2))];translation=torch.nn.Parameter(torch.tensor([[.1,.2,2.]]))
    optimizer,_=factory.create_optimizer(original+[translation],optim_type='lbfgsls',lr=1.,maxiters=30)
    params=optimizer.param_groups[0]['params']
    assert len(params)==2 and params[-1] is translation and translation.numel()==3
    assert [id(p) for p in params].count(id(translation))==1
    ((translation-torch.tensor([[0.,0.,1.]])).square().sum()).backward()
    assert torch.isfinite(translation.grad).all() and float(translation.grad.abs().max())>0
    save(OUT/'interface-check.json',dict(updated=now(),transport=grass.check_transport(),
        source_compiles=True,one_existing_three_scalar_parameter_added=True,actual_original_optimizer_membership=True,
        finite_nonzero_translation_gradient=True,optimizer_steps=0,
        original_stage_loss_weights_unchanged=True,world_camera_and_K_unchanged=True,
        quality_accepted=False,adapter_diff_sha256=sha(OUT/'placement-adapter.diff'),runner_diff_sha256=sha(OUT/'runner-receipt.diff')))
    print('GRASS_TRANSLATION_REGISTERED',flush=True)


def run():
    assert not (OUT/'official-fit').exists()
    p=read(OUT/'plan.json')
    for path,digest in p['sources'].items():assert sha(path)==digest,path
    grass.OUT=OUT
    grass.verify_translation=verify_translation
    code,_=adapted_runner()
    exec(compile(code,'<grass-temporal-new-placement-receipt>','exec'),grass.__dict__)
    entry.adapted_entry_source=source
    grass.run()


def launch():
    # The original sequence runner owns this lock; the parent must not lock
    # the same file and then prevent its child from acquiring it.
    assert not (OUT/'console.log').exists()
    cmd=[sys.executable,'-B',str(Path(__file__)),'run']
    with (OUT/'console.log').open('x') as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        save(OUT/'launch.json',dict(updated=now(),pid=child.pid,command=cmd,training_started=False,lock='/tmp/shuolee-native122-gpu.lock'))
        print('GRASS_TRANSLATION_STARTED',child.pid,flush=True)
        rc=child.wait()
    save(OUT/'process-exit.json',dict(updated=now(),pid=child.pid,exit_code=rc))
    raise SystemExit(rc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','launch']);args=p.parse_args()
    {'prepare':prepare,'run':run,'launch':launch}[args.mode]()

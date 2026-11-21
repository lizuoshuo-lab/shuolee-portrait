"""Separately authorized same-candidate131 continuation, never a fresh trial.

The original1000-only plan/helper/checkpoint stay immutable. Keep its exact
lifecycle contract, and bind this added engineering glue in a separate plan
and actual-review decision. Only rebind the unchanged DC hook after original
restore's second training_setup, and recover its already-completed step count.
"""
import argparse
import copy
import dis
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
import types

import numpy as np
import train_leafwoman_dcbox_safe131 as original

H=Path(__file__).resolve().parent
gs=original.gs
OUT=original.OUT
RESUME_PLAN=H/'resume_leafwoman_dcbox_safe131.plan.json'
AUDIT=H/'resume_leafwoman_dcbox_safe131.audit.json'
ORIGINAL_HELPER_SHA='5e8db46276792c86a7498afa58cc41848d2c5832575cba44d92433e238562608'
LIFECYCLE_SHA=original.previous.LIFECYCLE_SHA
CHECKPOINT=OUT/'gaussian-run/chkpnt1000.pth'
SIDECAR=Path(str(CHECKPOINT)+'.lifecycle.pt')
MANIFEST=Path(str(CHECKPOINT)+'.lifecycle.json')
RUNTIME={}
assert gs.sha(original.__file__)==ORIGINAL_HELPER_SHA


def _install_resume_dc_projection(model):
    """Original restore constructs a second Adam; discard only the old hook."""
    old=getattr(model,'_dcbox_safe131_step_hook',None)
    if old is not None:
        old.remove()
        model._dcbox_safe131_step_hook=None
    bindings=int(getattr(model,'_dcbox_safe131_resume_bindings',0))+1
    assert bindings<=2,'Only first setup plus the original restore setup is expected.'
    original._install_dc_projection(model)
    model._dcbox_safe131_projection_steps=1000
    model._dcbox_safe131_resume_bindings=bindings


def verify_external_plan():
    plan=gs.read(RESUME_PLAN)
    assert plan['resume_driver_sha256']==gs.sha(__file__)
    assert plan['maximum_schedule_step']==18000 and plan['expected_final_optimizer_updates']==17999
    assert plan['fresh_initialization_allowed'] is False and plan['original1000_only_plan_modified'] is False
    for path,digest in plan['immutable_sources'].items():assert gs.sha(path)==digest,path
    return plan


def verify_additional_decision():
    plan=verify_external_plan()
    request,decision=original._verify_decision()
    assert decision['continuation_authorization']=='same_131_candidate_maximum18000_once'
    assert decision['resume_driver_sha256']==gs.sha(__file__)
    assert decision['resume_plan_sha256']==gs.sha(RESUME_PLAN)
    assert decision['maximum_schedule_step']==18000
    assert decision['scientific_training_constraints_unchanged'] is True
    assert decision['original1000_only_plan_remains_unchanged'] is True
    assert decision.get('quality_accepted',False) is False
    return plan,request,decision


def _begin_phase(mode):
    assert mode=='resume'
    verify_additional_decision()
    return original._begin_phase(mode)


def _make_lifecycle(local,phase):
    # Do not append new driver hashes to the old source contract: the extra
    # driver is independently SHA-bound in the new plan and actual decision.
    life=original._make_lifecycle(local,phase)
    RUNTIME.update(model=local['model'],life=life)
    return life


def _adapter_namespace(phase,started):
    assert phase['mode']=='resume'
    namespace=original._adapter_namespace(phase,started)
    original_save=namespace['save']
    def save_with_resume_binding(path,value):
        if Path(path)==OUT/'gaussian-run/resume-state-verification.json':
            model=RUNTIME['model'];life=RUNTIME['life']
            assert model._dcbox_safe131_resume_bindings==2
            assert model._dcbox_safe131_projection_steps==1000
            assert life.last_resume_receipt['next_iteration']==1001
            assert value['actual_first_resumed_iteration']==1001
            value=dict(value,resume_driver_sha256=gs.sha(__file__),
                additional_resume_plan_sha256=gs.sha(RESUME_PLAN),
                original_lifecycle_contract_sha256=gs.read(MANIFEST)['contract_sha256'],
                DC_hook_bound_to_current_restored_Adam=True,DC_projection_counter_before1001_update=1000,
                old_hook_removed_before_second_setup=True,
                original1000_only_plan_not_rewritten=True,quality_accepted=False)
        return original_save(path,value)
    namespace.update(_install_dc_projection=_install_resume_dc_projection,
        _make_lifecycle=_make_lifecycle,save=save_with_resume_binding)
    return namespace


def run():
    verify_additional_decision()
    assert not (OUT/'resume-launch.json').exists() and not (OUT/'compute-phase-resume.json').exists()
    # Original131 scope retains fcntl/math/signal and its frozen scientific IDs.
    scope=dict(original._scope)
    scope.update(_begin_phase=_begin_phase,_adapter_namespace=_adapter_namespace)
    fn=original.previous.train
    resumed=types.FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__)
    resumed('resume')
    assert gs.read(OUT/'progress.json')['state']=='bounded_control_completed'
    result=gs.read(OUT/'gaussian-run/result.json')
    assert result['snapshot_step']==18000 and result['optimizer_updates']==17999
    model=RUNTIME['model'];assert model._dcbox_safe131_projection_steps==17999
    dc=model._features_dc.detach()
    assert bool((dc>=original.DC_MIN).all()) and bool((dc<=original.DC_MAX).all())
    verify_external_plan()
    gs.save(OUT/'dc-projection-resume-receipt.json',dict(updated=gs.now(),
        state='same_candidate_continuation_completed',configured_schedule_steps=18000,
        actual_total_optimizer_updates=17999,DC_projection_steps=17999,
        original1000_native_checkpoint_sha256=gs.sha(CHECKPOINT),
        original_lifecycle_contract_sha256=gs.read(MANIFEST)['contract_sha256'],
        resume_driver_sha256=gs.sha(__file__),additional_resume_plan_sha256=gs.sha(RESUME_PLAN),
        original_policy=original.projection_policy(),stored_DC_min=float(dc.min()),stored_DC_max=float(dc.max()),
        same_fit_cameras_masks_loss_rates_density_reset_and_original_clock=True,
        checkpoint1000_preserved=True,original1000_only_plan_preserved=True,
        final18000_snapshot_is_pre_terminal_optimizer_branch=True,quality_accepted=False))
    path=OUT/'resume-process-exit.json';assert not path.exists()
    gs.save(path,dict(updated=gs.now(),exit_code=0,state=result['state'],step=18000,
        optimizer_updates=17999,resume_driver_sha256=gs.sha(__file__),quality_accepted=False))


def _cpu_rebind_test():
    import torch
    from types import SimpleNamespace
    before_rng=torch.get_rng_state().clone()
    p=torch.nn.Parameter(torch.tensor([[[.1,.2,.3]]],dtype=torch.float32))
    model=SimpleNamespace(_features_dc=p,active_sh_degree=0,max_sh_degree=0)
    first=torch.optim.Adam([dict(params=[p],name='f_dc')],lr=.0025)
    model.optimizer=first;_install_resume_dc_projection(model)
    assert model._dcbox_safe131_projection_steps==1000
    p.grad=torch.ones_like(p);first.step();first.zero_grad(set_to_none=True)
    assert model._dcbox_safe131_projection_steps==1001
    old_state=copy.deepcopy(first.state_dict())
    q=torch.nn.Parameter(p.detach().clone())
    model._features_dc=q
    model.optimizer=torch.optim.Adam([dict(params=[q],name='f_dc')],lr=.0025)
    _install_resume_dc_projection(model)
    model.optimizer.load_state_dict(old_state)
    assert model._dcbox_safe131_resume_bindings==2 and model._dcbox_safe131_projection_steps==1000
    from staged_training_lifecycle128 import _equal
    assert _equal(model.optimizer.state_dict(),old_state)
    # The old optimizer no longer owns a live callback on this model.
    p.grad=torch.ones_like(p);first.step();assert model._dcbox_safe131_projection_steps==1000
    q.grad=torch.ones_like(q);model.optimizer.step()
    assert model._dcbox_safe131_projection_steps==1001
    assert _equal(before_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    return dict(second_original_training_setup_rebound=True,old_hook_removed=True,
        projection_counter_restored1000=True,Adam_state_load_exact=True,
        first_resumed_step_counter1001=True,RNG_unchanged=True,CUDA_initialized=False)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not RESUME_PLAN.exists() and not AUDIT.exists()
    assert not (OUT/'resume-launch.json').exists() and not (OUT/'compute-phase-resume.json').exists()
    assert gs.read(OUT/'process-exit.json')['exit_code']==0
    accounting=gs.read(OUT/'compute-phase-train.json')
    assert accounting['successful'] is True and accounting['state']=='checkpoint_ready_for_offline_review'
    manifest=gs.read(MANIFEST);request=gs.read(OUT/'review-1000-request.json')
    assert manifest['completed_iteration']==request['step']==request['optimizer_updates']==1000
    assert manifest['checkpoint_phase']=='after_density_adam_zero_grad' and manifest['total_iterations']==18000
    assert manifest['native_checkpoint_sha256']==gs.sha(CHECKPOINT)==request['native_checkpoint_sha256']
    assert manifest['sidecar_sha256']==gs.sha(SIDECAR)==request['lifecycle_sidecar_sha256']
    assert gs.sha(MANIFEST)==request['lifecycle_manifest_sha256']
    import torch
    from staged_training_lifecycle128 import GROUPS,_equal
    native=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    sidecar=torch.load(SIDECAR,map_location='cpu',weights_only=False)
    assert type(native) is tuple and len(native)==2 and native[1]==1000 and len(native[0])==14
    capture=native[0];adam=capture[12]
    assert [group['name'] for group in adam['param_groups']]==list(GROUPS)
    assert len(adam['param_groups'])==7 and all(len(group['params'])==1 for group in adam['param_groups'])
    assert sidecar['contract_sha256']==manifest['contract_sha256']
    assert sidecar['contract']['identity']['plan_sha256']==gs.sha(OUT/'plan.json')
    assert sidecar['contract']['total_iterations']==18000
    assert sidecar['contract']['identity']['dc_parameter_projection']==original.projection_policy()
    assert sidecar['paired']['calls']==sidecar['camera_loader']['sampled_count']==1000
    assert len(sidecar['counters']['counts'])==12 and sum(sidecar['counters']['counts'])==1000
    assert set(sidecar['counters'])=={'counts'}
    for item in capture:
        if torch.is_tensor(item) and item.dtype.is_floating_point:assert bool(torch.isfinite(item).all())
    assert bool((capture[2]>=original.DC_MIN).all()) and bool((capture[2]<=original.DC_MAX).all())
    assert set(sidecar['extra_parameters'])=={'static_offset'}
    assert tuple(sidecar['extra_parameters']['static_offset'].shape)==(10475,3)
    for path,digest in sidecar['contract']['source_hashes'].items():assert gs.sha(path)==digest,path
    toy=_cpu_rebind_test()
    fn=original.previous.train
    scope=dict(original._scope);scope.update(_begin_phase=_begin_phase,_adapter_namespace=_adapter_namespace)
    wanted={i.argval for i in dis.get_instructions(fn) if i.opname=='LOAD_GLOBAL'}
    import builtins
    assert all(name in scope or hasattr(builtins,name) for name in wanted)
    assert all(name in scope for name in ('fcntl','math','signal'))
    immutable={Path(__file__),Path(original.__file__),original.PREREG,original.PATCH,
        H/'staged_training_lifecycle128.py',OUT/'plan.json',OUT/'data/training-inputs.json',
        OUT/'fit/fitted.npz',OUT/'fit24-to-GS12-subset-receipt.json',OUT/'process-exit.json',
        OUT/'compute-phase-train.json',OUT/'review-1000-request.json',
        OUT/'dc-projection-1000-receipt.json',CHECKPOINT,SIDECAR,MANIFEST,
        OUT/'gaussian-run/step-001000/result.json',OUT/'gaussian-run/step-001000/world.ply',
        OUT/'adapter.py',OUT/'training-color-boundary-cpu.json'}
    hashes={str(path):gs.sha(path) for path in sorted(immutable,key=str)}
    plan=dict(updated=gs.now(),schema='additional-same-candidate131-continuation-v1',
        resume_driver_sha256=gs.sha(__file__),original1000_helper_sha256=ORIGINAL_HELPER_SHA,
        immutable_sources=hashes,original1000_only_plan_modified=False,
        fresh_initialization_allowed=False,additional_authorization_required=True,
        same_candidate_only=True,source_1000_completed_optimizer_updates=1000,
        first_resumed_iteration=1001,maximum_schedule_step=18000,expected_final_optimizer_updates=17999,
        original_18k_clock_unchanged=True,projection_policy=original.projection_policy(),
        original_lifecycle_contract_sha256=manifest['contract_sha256'],
        old_lifecycle_contract_and_source_identity_unchanged=True,
        added_resume_driver_identity='This independent plan plus actual review decision; not silently appended to the old checkpoint contract.',
        prior_accounted_compute_seconds=accounting['cumulative_compute_seconds'],
        total_cumulative_compute_budget_seconds=2400,offline_review_walltime_excluded=True,
        first_report_requires_original_state_restore_and_hook_binding=True,
        later_18000_terminal_snapshot_phase='Original Scene.save before terminal optimizer branch;17999 completed updates.',
        required_extra_decision_fields=dict(continuation_authorization='same_131_candidate_maximum18000_once',
            resume_driver_sha256=gs.sha(__file__),resume_plan_sha256='SHA256 of this plan afterfreeze',
            maximum_schedule_step=18000,scientific_training_constraints_unchanged=True,
            original1000_only_plan_remains_unchanged=True,quality_accepted=False),
        CPU_rebind_test=toy,known_failures='Persistent holes/lower skirt fragments/frame18 face ghosting; extended training is exploratory, not acceptance.',
        no_new_loss_solver_renderer_raster_pass_or_scientific_parameters=True,
        no_GPU_executed=True,quality_accepted=False)
    gs.save(RESUME_PLAN,plan)
    receipt=dict(updated=gs.now(),state='CPU_checkpoint_and_rebind_audit_passed',
        driver_sha256=gs.sha(__file__),resume_plan_sha256=gs.sha(RESUME_PLAN),
        original_native_checkpoint_sha256=gs.sha(CHECKPOINT),sidecar_sha256=gs.sha(SIDECAR),
        original_lifecycle_contract_sha256=manifest['contract_sha256'],points=int(capture[1].shape[0]),
        optimizer_groups=[group['name'] for group in adam['param_groups']],
        captured_counters_do_not_include_DC_projection_counter=True,counter_recovered_from_actual_step1000=True,
        Adam_RNG_camera_paired_restore_delegated_to_exact_original_lifecycle=True,
        CPU_rebind_test=toy,actual_resume=False,GPU_executed=False,quality_accepted=False)
    gs.save(AUDIT,receipt)
    assert not torch.cuda.is_initialized()
    print('SAFE131_ADDITIONAL_RESUME_CPU_AUDIT_PASSED',gs.sha(__file__),gs.sha(RESUME_PLAN),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','run'])
    mode=parser.parse_args().mode
    try:globals()[mode]()
    except BaseException as error:
        if mode=='run':
            path=OUT/'additional-resume-failure.json'
            if not path.exists():gs.save(path,dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False,quality_accepted=False))
        raise

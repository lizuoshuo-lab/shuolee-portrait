"""One fresh1000 SH0 native-boundary-corrected DC-box causal trial, on the immutable129 fit24->GS12.

Only scientific change: after original Adam.step, project the current DC
Parameter to the independently adopted fixed native float32 interval. Original gradients, moments,
rates, loss, renderer, paired backgrounds, density and18k clock stay intact.
There is no resume/18k CLI;1000 exits with the complete offline checkpoint.
"""
import argparse
import ast
import copy
import difflib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
import types
from types import SimpleNamespace

import numpy as np
import train_leafwoman_half12_sh0129 as previous

H = Path(__file__).resolve().parent
gs = previous.gs
BASE129_SHA = '017414dabb4692293a14c005f84cc1279d35d0c69547ae2a13f402485da6ee6d'
assert gs.sha(previous.__file__) == BASE129_SHA
SUBJECT = 'leafwoman-half12-dcbox-safe131'
OUT = gs.ROOT/SUBJECT
FIT, SELECTION = previous.FIT, previous.SELECTION
HANDOFF = gs.ROOT/(SUBJECT+'-native-handoff')
BASELINE = previous.OUT
PREREG = H/'train_leafwoman_dcbox_safe131.plan.json'
PATCH = H/'train_leafwoman_dcbox_safe131.adapter.diff'
SH_UTILS = gs.UP/'utils/sh_utils.py'
_sh_tree = ast.parse(SH_UTILS.read_text())
C0 = ast.literal_eval(next(n.value for n in _sh_tree.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='C0' for t in n.targets)))
assert C0 == 0.28209479177387814
DC_BOUND = .5/C0 # historical real-valued reference, not the implemented interval
DC_MIN = float(np.asarray(0xbfe2dfc3,dtype=np.uint32).view(np.float32))
DC_MAX = float(np.asarray(0x3fe2dfc5,dtype=np.uint32).view(np.float32))
NUMERIC_ROOT = gs.ROOT/'leafwoman-half12-dcbox1000-130'
NATIVE_RESULT = NUMERIC_ROOT/'numerical-boundary-probe/result.json'
NATIVE_SUPPLEMENT = NUMERIC_ROOT/'numerical-boundary-reduction-supplement.json'
NATIVE_RESULT_SHA = 'e7bc3857462bd31dc02672a8af68b3f3c4747031de14b614d7ef183b66ebf8d7'
NATIVE_SUPPLEMENT_SHA = 'f6e2aaba7ee1624fa2419368dafddf6e433cc55616b74208b10ffc00f5e2c1ed'
N, FIT_N, HOLD_N = previous.N, previous.FIT_N, previous.HOLD_N
TOTAL, COMPUTE_BUDGET = previous.TOTAL, previous.COMPUTE_BUDGET


def projection_policy():
    return dict(kind='post_original_Adam_step_parameter_projection',SH_degree=0,C0=C0,
        dc_min=DC_MIN,dc_max=DC_MAX,dc_min_float32_bits='0xbfe2dfc3',dc_max_float32_bits='0x3fe2dfc5',decoded_RGB_min=0.,decoded_RGB_max=1.,
        expression='features_dc.clamp_(DC_MIN,DC_MAX) under torch.no_grad(); fixed native-tested float32 bits',
        scope='Standard source-SDR training RGB values; not an HDR or calibrated physical-radiance model.',
        forward_renderer_changed=False,gradient_replaced_or_clipped=False,
        Adam_moments_zeroed_or_reset=False,all_original_rates_and_losses_unchanged=True,
        risks=['Valid bright/highlight radiance may be clipped by the SDR bound.',
            'Unchanged Adam moments may keep pushing into a boundary (lagged momentum).',
            'Color/opacity ambiguity, wrong geometry or low coverage may remain; quality can drop.'],
        native_adoption=dict(original_registered_bit_equal_passed=False,
            independent_engineering_adoption=True,
            evidence='Fixed native lower has positive RGB2.9802322387695312e-8 and nonzero gradients; upper has RGB1 and nonzero gradients. Original bit-equal criterion remains false; atomic reduction supplement explains measured1-2ULP differences without changing the old record.',
            native_result_sha256=NATIVE_RESULT_SHA,reduction_supplement_sha256=NATIVE_SUPPLEMENT_SHA,
            scope='This tested original native binary only; no broad hardware/quality guarantee.'),
        proven_fix=False)


def _install_dc_projection(model):
    """Adam owns all state; the post-hook changes only current DC values."""
    import torch
    assert getattr(model,'_dcbox_safe131_step_hook',None) is None
    assert model.active_sh_degree == model.max_sh_degree == 0
    optimizer = model.optimizer
    groups = [group for group in optimizer.param_groups if group.get('name')=='f_dc']
    assert len(groups)==1 and len(groups[0]['params'])==1
    assert groups[0]['params'][0] is model._features_dc
    model._dcbox_safe131_projection_steps = 0
    def project_after_original_step(actual_optimizer,args,kwargs):
        assert actual_optimizer is optimizer
        # Densification replaces Parameters; always use the current model field.
        assert next(group for group in optimizer.param_groups if group.get('name')=='f_dc')['params'][0] is model._features_dc
        next_step=model._dcbox_safe131_projection_steps+1
        boundary=next_step==1000
        if boundary:
            state=optimizer.state[model._features_dc]
            before_moments={key:value.detach().clone() for key,value in state.items() if torch.is_tensor(value)}
            assert {'exp_avg','exp_avg_sq','step'}<=set(before_moments)
        def boundary_RGB():
            rgb=.5+C0*model._features_dc.detach()
            return dict(min=float(rgb.min()),max=float(rgb.max()),
                points_with_any_RGB_below0=int((rgb<0).flatten(1).any(1).sum()),
                points_with_any_RGB_above1=int((rgb>1).flatten(1).any(1).sum()))
        if boundary: before_RGB=boundary_RGB()
        with torch.no_grad():
            model._features_dc.clamp_(DC_MIN,DC_MAX)
        model._dcbox_safe131_projection_steps += 1
        if boundary:
            assert all(torch.equal(value,state[key]) for key,value in before_moments.items())
            model._dcbox_safe131_boundary_receipt=dict(before_projection_RGB=before_RGB,
                after_projection_RGB=boundary_RGB(),actual_Adam_tensor_state_unchanged_by_projection=True,
                Adam_state_keys_checked=sorted(before_moments),
                measurement='Immediate boundary1000 post-Adam values, before and after the one projection; not all-training extrema.')
    model._dcbox_safe131_step_hook = optimizer.register_step_post_hook(project_after_original_step)


_scope = dict(previous.train.__globals__)
_scope.update(previous.__dict__)
_scope.update(globals())
_CLONE_NAMES = ('subset_view','reader_adapters','selection_context','adapted_prepare',
    'adapter_from_prepare','_compute_elapsed','_save_frozen_metadata','_write_frozen_text',
    '_verify_decision','_begin_phase','preflight_function')
for _name in _CLONE_NAMES:
    _fn = getattr(previous,_name)
    globals()[_name] = types.FunctionType(_fn.__code__,_scope,_fn.__name__,_fn.__defaults__,_fn.__closure__)


def lifecycle_adapter(original):
    unchanged, receipt = previous.lifecycle_adapter(original)
    tree = ast.parse(unchanged)
    train_fn = tree.body[0]
    body = next(n for n in train_fn.body if isinstance(n,ast.ClassDef) and n.name=='Body')
    setup = next(n for n in body.body if isinstance(n,ast.FunctionDef) and n.name=='training_setup')
    assert len(setup.body)==2
    assert ast.unparse(setup.body[0])=='super().training_setup(training_args)'
    assert ast.unparse(setup.body[1]).startswith('self.optimizer.add_param_group(')
    setup.body.append(ast.parse('_install_dc_projection(self)').body[0])
    changed = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    # Removing the one hook call must return the entire inherited adapter AST.
    setup.body.pop()
    assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(unchanged),include_attributes=False)
    compile(changed,'<dcbox_safe131-one-post-Adam-hook>','exec')
    return changed,dict(receipt,dc_projection_hook_added=1,
        original_train_function_AST_not_edited=True,one_scientific_constraint=projection_policy())


def _actual_training_sources(local):
    fn = types.FunctionType(previous._actual_training_sources.__code__,_scope,
        previous._actual_training_sources.__name__,previous._actual_training_sources.__defaults__,
        previous._actual_training_sources.__closure__)
    return sorted(set(fn(local))|{Path(previous.__file__).resolve(),SH_UTILS.resolve()},key=str)


_identity = previous._identity_after
assert _identity.count("schema='leafwoman129-staged-original-18k-v1'")==1
assert _identity.count('scientific_training_profile_changed=False')==1
_identity = _identity.replace("schema='leafwoman129-staged-original-18k-v1'",
    "schema='leafwoman131-dcbox-staged-original-18k-v1'").replace(
    'scientific_training_profile_changed=False',
    'scientific_training_profile_changed=True,dc_parameter_projection=projection_policy()')
exec(compile(_identity,str(__file__)+'::same_lifecycle_new_constraint_identity','exec'),_scope)
_make_lifecycle = _scope['_make_lifecycle']


def _stage_preview_after_checkpoint(boundary,scene,life,local,phase):
    assert phase['mode']=='train' and local['model']._dcbox_safe131_projection_steps==1000
    fn = types.FunctionType(previous._stage_preview_after_checkpoint.__code__,_scope,
        previous._stage_preview_after_checkpoint.__name__,previous._stage_preview_after_checkpoint.__defaults__,
        previous._stage_preview_after_checkpoint.__closure__)
    fn(boundary,scene,life,local,phase)
    dc = local['model']._features_dc.detach()
    assert bool((dc>=DC_MIN).all()) and bool((dc<=DC_MAX).all())
    gs.save(OUT/'dc-projection-1000-receipt.json',dict(updated=gs.now(),step=1000,
        original_Adam_steps=1000,DC_projection_steps=1000,policy=projection_policy(),
        stored_dc_min=float(dc.min()),stored_dc_max=float(dc.max()),
        actual_boundary_measurements=local['model']._dcbox_safe131_boundary_receipt,
        complete_checkpoint_after_projection=True,Adam_moments_preserved=True,
        no_extra_training_raster_pass=True,quality_accepted=False))


def _adapter_namespace(phase,started):
    fn = types.FunctionType(previous._adapter_namespace.__code__,_scope,
        previous._adapter_namespace.__name__,previous._adapter_namespace.__defaults__,
        previous._adapter_namespace.__closure__)
    return dict(fn(phase,started),_install_dc_projection=_install_dc_projection)


def prepare():
    """Explicit CPU-only future preparation; not called by audit."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and not HANDOFF.exists()
    prereg = gs.read(PREREG)
    assert prereg['helper_sha256']==gs.sha(__file__) and prereg['adapter_patch_sha256']==gs.sha(PATCH)
    for path,digest in prereg['immutable_source_hashes'].items(): assert gs.sha(path)==digest,path
    fn = types.FunctionType(previous.prepare.__code__,_scope,previous.prepare.__name__,
        previous.prepare.__defaults__,previous.prepare.__closure__)
    fn()
    plan = gs.read(OUT/'plan.json')
    plan.update(subject=SUBJECT,label='持叶新拍摄 · SH0 DC RGB界限1000步对照',
        only_parameter_change='One scientific constraint: post-original-Adam DC parameter projection to source-SDR RGB[0,1].',
        dc_parameter_projection=projection_policy(),maximum_authorized_updates=1000,
        automatic_18k_continuation=False,resume_authorized=False,
        preregistered_plan=str(PREREG),preregistered_plan_sha256=gs.sha(PREREG),
        baseline1000_comparison=prereg['comparison'],quality_accepted=False,proven_fix=False)
    plan['source_hashes'].update(prereg['immutable_source_hashes'])
    plan['source_hashes'].update({str(PREREG):gs.sha(PREREG),str(PATCH):gs.sha(PATCH),
        str(Path(previous.__file__)):BASE129_SHA,str(SH_UTILS):gs.sha(SH_UTILS)})
    gs.save(OUT/'plan.json',plan)
    # The inherited plan lists actual scientific options; the projection adds no option rewrite.
    baseline = gs.read(BASELINE/'plan.json')
    assert all(plan[key]==baseline[key] for key in previous.PROFILE_KEYS)
    (OUT/'dc-projection-interface.diff').write_text(PATCH.read_text())
    print('LEAFWOMAN_DCBOX_SAFE131_CPU_PREPARED',gs.sha(OUT/'plan.json'),flush=True)


def preflight():
    function,diff = preflight_function()
    (OUT/'preflight-interface.diff').write_text(diff)
    function('leafwoman-dcbox_safe131-zero-update-preflight')


def train():
    fn = types.FunctionType(previous.train.__code__,_scope,previous.train.__name__,
        previous.train.__defaults__,previous.train.__closure__)
    fn('train')
    assert gs.read(OUT/'progress.json')['state']=='checkpoint_ready_for_offline_review'
    assert gs.read(OUT/'dc-projection-1000-receipt.json')['DC_projection_steps']==1000


def _cpu_projection_test():
    import torch
    p = torch.nn.Parameter(torch.tensor([[[-DC_BOUND+.01,0.,DC_BOUND-.01]]],dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    model = SimpleNamespace(_features_dc=p,active_sh_degree=0,max_sh_degree=0)
    model.optimizer = torch.optim.Adam([dict(params=[p],name='f_dc')],lr=.1)
    control = torch.optim.Adam([dict(params=[q],name='f_dc')],lr=.1)
    _install_dc_projection(model)
    grad = torch.tensor([[[1.,.5,-1.]]],dtype=torch.float64)
    p.grad=grad.clone();q.grad=grad.clone()
    model.optimizer.step();control.step()
    assert torch.equal(p,torch.clamp(q,DC_MIN,DC_MAX))
    assert float(q.min()) < DC_MIN and float(q.max()) > DC_MAX
    for key,value in model.optimizer.state[p].items():
        assert torch.equal(value,control.state[q][key]),key
    assert p.grad is not None and torch.equal(p.grad,grad)
    model.optimizer.zero_grad(set_to_none=True)
    assert p.grad is None and model._dcbox_safe131_projection_steps==1
    # Simulate the original densifier replacing its current DC Parameter.
    replacement = torch.nn.Parameter(torch.tensor([[[0.,DC_BOUND-.01,0.]]],dtype=torch.float64))
    model._features_dc=replacement;model.optimizer.param_groups[0]['params']=[replacement]
    replacement.grad=torch.tensor([[[0.,-1.,0.]]],dtype=torch.float64)
    old = p.detach().clone();model.optimizer.step()
    assert torch.equal(p,old) and bool(((replacement>=DC_MIN)&(replacement<=DC_MAX)).all())
    assert model._dcbox_safe131_projection_steps==2
    fixed=torch.tensor([DC_MIN,DC_MAX],dtype=torch.float32)
    assert fixed.numpy().view(np.uint32).tolist()==[0xbfe2dfc3,0x3fe2dfc5]
    assert float(fixed[0])==DC_MIN and float(fixed[1])==DC_MAX
    assert not torch.cuda.is_initialized()
    return dict(original_Adam_called_once_per_hook=True,Adam_states_exact_to_unprojected_control=True,
        original_gradient_unchanged_until_zero_grad=True,current_Parameter_after_replacement_used=True,
        no_cuda_initialized=True,synthetic_only=True)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and not HANDOFF.exists()
    assert not PREREG.exists() and not PATCH.exists(), 'Freeze once; retain prior audits rather than overwrite.'
    frames=[f'frame_{i:06d}' for i in range(1,N+1)];reference=frames[5]
    _,_,changed,_ = adapted_prepare(reference,frames)
    inherited = previous.lifecycle_adapter(adapter_from_prepare(changed,reference))[0]
    candidate, receipt = lifecycle_adapter(adapter_from_prepare(changed,reference))
    diff=''.join(difflib.unified_diff(inherited.splitlines(True),candidate.splitlines(True),
        fromfile='same129-profile-and-complete-checkpoint',tofile='one-post-Adam-DC-projection-hook'))
    assert diff.count('+            _install_dc_projection(self)')==1
    preflight_function()
    assert gs.sha(NATIVE_RESULT)==NATIVE_RESULT_SHA and gs.sha(NATIVE_SUPPLEMENT)==NATIVE_SUPPLEMENT_SHA
    native = gs.read(NATIVE_RESULT); supplement = gs.read(NATIVE_SUPPLEMENT)
    assert native['inward_lower_and_existing_upper_native_checks_passed'] is False
    assert supplement['original_registered_checks_passed_flag'] is False
    assert supplement['independent_conclusion']['safe_fixed_lower_supported_for_this_native_binary'] is True
    assert DC_MIN == -1.7724536657333374 and DC_MAX == 1.7724539041519165
    cpu = _cpu_projection_test()
    assert gs.sha(previous.__file__)==BASE129_SHA
    assert gs.sha(gs.UP/'train.py')==previous.UPSTREAM_TRAIN_SHA
    assert gs.sha(H/'staged_training_lifecycle128.py')==previous.LIFECYCLE_SHA
    # CPU metadata/hashes only. No previous optimized Gaussian is loaded.
    source = selection_context()
    baseline_plan = gs.read(BASELINE/'plan.json')
    baseline_rows = gs.read(BASELINE/'data/training-inputs.json')['records']
    assert set(source['frames'])=={row['frame'] for row in baseline_rows}
    assert source['reference']==baseline_plan['reference_frame']=='frame_000004'
    assert baseline_plan['model_sh_degree']==0
    original130 = gs.ROOT/'leafwoman-half12-dcbox1000-130'
    source130 = gs.read(original130/'data/training-inputs.json')['records']
    assert len(source130)==len(baseline_rows)==12
    for a,b in zip(baseline_rows,source130):
        for key in ('frame','wh','K','K_integer_centers','w2c','source_sha256','mask_sha256','validity_sha256'):
            assert a[key]==b[key],(a['frame'],key)
    with np.load(BASELINE/'fit/fitted.npz',allow_pickle=False) as a, np.load(original130/'fit/fitted.npz',allow_pickle=False) as b:
        assert set(a)==set(b) and all(np.array_equal(a[key],b[key]) for key in a)
    paths={Path(__file__),Path(previous.__file__),SH_UTILS,gs.UP/'train.py',H/'staged_training_lifecycle128.py',
        NATIVE_RESULT,NATIVE_SUPPLEMENT,original130/'plan.json',original130/'data/training-inputs.json',original130/'fit/fitted.npz',
        BASELINE/'plan.json',BASELINE/'data/training-inputs.json',BASELINE/'pixel-mapping.json',
        BASELINE/'gaussian-run/step-001000/result.json',*map(Path,source['immutable_hashes'])}
    for row in baseline_rows:
        paths.update(Path(row[key]) for key in ('image','mask','validity'))
        paths.update(BASELINE/'gaussian-run/step-001000'/(row['frame']+suffix)
            for suffix in ('_gaussian.png','_cumulative-alpha.npy'))
    immutable={str(path):gs.sha(path) for path in sorted(paths,key=str)}
    PATCH.write_text(diff)
    plan=dict(updated=gs.now(),schema='leafwoman131-one-dcbox1000-causal-trial-v1',
        helper_sha256=gs.sha(__file__),adapter_patch_sha256=gs.sha(PATCH),subject=SUBJECT,
        planned_output=str(OUT),planned_fresh_handoff=str(HANDOFF),fit=str(FIT),
        frames=source['frames'],reference_frame=source['reference'],fit_candidate_count=24,GS_count=12,
        same_input_fit_masks_cameras_and_seed_as129=True,original_schedule_clock=TOTAL,
        hard_stop_actual_optimizer_updates=1000,complete_post_update_checkpoint=True,
        automatic_18k_continuation=False,resume_authorized=False,old_Gaussians_or_optimizer_loaded=False,
        policy=projection_policy(),immutable_source_hashes=immutable,
        comparison=dict(baseline=str(BASELINE/'gaussian-run/step-001000'),
            required='All12 exact same-camera RGB/source-mask/cumulative-alpha and native mesh, plus known-failure frame18 and novel inspection.',
            known_failure_frame='frame_000018',no_automatic_frame_reselection=True,
            alpha_statistic_is_not_quality_acceptance=True,
            accept='Actual visual comparison required; lower floaters, geometry and cloth coverage may remain or worsen.'),
        no_renderer_or_extra_training_raster_call=True,no_alpha_loss=True,
        rates_loss_density_reset_clock_seed_unchanged=True,CPU_projection_test=cpu,
        lifecycle=receipt,actual_prepare=False,GPU_executed=False,quality_accepted=False,proven_fix=False,
        original_native_registered_bit_equal_criterion_passed=False,
        independent_fixed_boundary_engineering_adoption=projection_policy()['native_adoption'],
        prepared130_input_arrays_equal129=True,
        resume_implementation='Not enabled in this1000-only stage; preserve original130 lifecycle and hard stop. Any future resume needs separate129-lifecycle integration before use.')
    gs.save(PREREG,plan)
    gs.save(H/'train_leafwoman_dcbox_safe131.audit.json',dict(updated=gs.now(),state='CPU_audit_passed',
        helper_sha256=gs.sha(__file__),preregistered_plan_sha256=gs.sha(PREREG),patch_sha256=gs.sha(PATCH),
        CPU_projection_test=cpu,one_hook_AST_delta_verified=True,original_train_AST_unchanged=True,
        source_fit_archive_read_CPU_only=True,old_Gaussian_values_loaded=False,
        output_training_directory_created=False,GPU_executed=False,quality_accepted=False))
    assert not OUT.exists() and not HANDOFF.exists()
    print('LEAFWOMAN_DCBOX_SAFE131_CPU_AUDIT_PASSED',gs.sha(__file__),gs.sha(PREREG),gs.sha(PATCH),flush=True)


_scope.update(globals())
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['audit','prepare','preflight','train'])
    mode=parser.parse_args().mode
    try:
        globals()[mode]()
        if mode=='train':
            path=OUT/'process-exit.json';assert not path.exists()
            gs.save(path,dict(updated=gs.now(),exit_code=0,state=gs.read(OUT/'progress.json')['state']))
    except BaseException as error:
        if OUT.exists():
            path=OUT/(mode+'-failure.json')
            if not path.exists():gs.save(path,dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise

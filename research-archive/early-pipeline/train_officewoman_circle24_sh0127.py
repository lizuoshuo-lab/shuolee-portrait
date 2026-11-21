"""Input/count/reference adapter for one new24-view SH0 bound-Gaussian run.

Reuses the reviewed football interface and unchanged original training profile.
No fitting, new loss, opacity adjustment, old individual geometry or GPU task
is performed by the CPU audit. Prepare requires actual fresh-fit admission.
"""
import ast
import difflib
import inspect
import os
import signal
import sys
import traceback
from pathlib import Path

import numpy as np
import train_football_half12_sh0126 as base
from train_indoor_circle16_sh0124 import reused
import cross_subject_sh0124 as gs

SUBJECT='officewoman-circle24-paired18k127'
OUT=gs.ROOT/SUBJECT
FIT=gs.ROOT/'officewoman-circle24-pose127'
HANDOFF=FIT/'gaussian-native-handoff'
N=24
NATIVE_WH=(1520,2704)
PROFILE_KEYS=('options','upstream_options','schedule_overrides','shared_laplacian_weight',
    'goal_points_approx','goal_point_band','max_points_safety_stop')


def adapted_prepare(reference,frames):
    # Exact interface fragments only. Do not replace substrings of identifiers
    # (the earlier eight->twelve transformation once corrupted weight).
    edits=[
        ('        ("row[\'wh\']==[1520,2704]", "row[\'wh\']==[2704,1520]", 1),\n',''),
        ("code = code.replace('all8', 'all12')","code = code.replace('all8', 'all'+str(N))"),
        ("code = re.sub(r'\\beight\\b', 'twelve', code)","code = re.sub(r'\\beight\\b', 'twentyfour', code)"),
        ("code = re.sub(r'\\bEight\\b', 'Twelve', code)","code = re.sub(r'\\bEight\\b', 'Twentyfour', code)"),
        ('("record_id=\'001\'", "record_id=\'011\'", 2)', '("record_id=\'001\'", "record_id=\'019\'", 2)'),
        ("code = code.replace('winterhat', 'football').replace('Winterhat', 'Football').replace('冬帽', '足球场新')",
         "code = code.replace('winterhat', 'officewoman').replace('Winterhat', 'Officewoman').replace('冬帽', '办公室新')"),
    ]
    fn,diff=reused(base.adapted_prepare,edits,dict(N=N,OUT=OUT,FIT=FIT,HANDOFF=HANDOFF,
        SUBJECT=SUBJECT,__file__=__file__),'officewoman24-input-connectors')
    prepared,before,changed=fn(reference,frames)
    tree=ast.parse(changed)
    key_loops=[node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.iter,ast.Tuple)
        and any(isinstance(x,ast.Constant) and x.value=='shared_laplacian_weight' for x in node.iter.elts)]
    assert len(key_loops)==1
    assert tuple(ast.literal_eval(x) for x in key_loops[0].iter.elts)==PROFILE_KEYS
    assert "row['wh']==[1520,2704]" in changed
    assert 'shared_laplacian_wtwelve' not in changed and 'shared_laplacian_wtwentyfour' not in changed
    return prepared,before,changed,diff


def adapter_from_prepare(changed,reference):
    """Evaluate only the same read-only adapter-string generation as prepare."""
    function=ast.parse(changed).body[0]
    start=next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='before' for t in node.targets)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and node.value.func.attr=='read_text')
    end=next(i for i,node in enumerate(function.body[start:],start) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='ast'
        and node.value.func.attr=='parse')
    scope=dict(base.previous.prepare.__globals__,reference=reference)
    exec(compile(ast.Module(body=function.body[start:end],type_ignores=[]),'<same-actual-prepare-adapter-path>','exec'),scope)
    code=base.add_alpha_export(scope['code'])
    check_adapter(code,reference)
    return code


def check_adapter(code,reference):
    tree=ast.parse(code)
    for marker in [f"receipt['frames'] == {N}",f'self.num_timesteps = {N}',
        f'assert len(training_slots)=={N}',f'counts = np.zeros({N}, dtype=int)',
        f'frames_rendered={N}',f'training_frames={N}, review_frames={N}']:
        assert code.count(marker)==1,marker
    assert not any(isinstance(node,ast.Constant) and type(node.value) is int and node.value in (8,12)
        for node in ast.walk(tree)), 'No stale8/12 numeric computation in generated adapter.'
    assert 'frame_000071' not in code and 'frame_000105' not in code
    assert repr(reference) in code and 'shared_laplacian_weight' in code
    assert 'KnownPixelLoss' not in code and 'photometric.use_camera' not in code
    assert "deadline=time.monotonic()+600" in code and "'lr': 1.e-6" in code
    assert 'override_color=torch.zeros_like(model.get_xyz)' in code


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not OUT.exists() and not HANDOFF.exists()
    p=gs.read(FIT/'plan.json');review=gs.read(FIT/'conclusion.json')
    assert review['actual_visual_review'] and review['exploratory_gaussian_admitted']
    rows=p['rows'];frames=[r['frame'] for r in rows];reference=p['reference_frame']
    assert len(rows)==len(set(frames))==N and reference in frames and reference!=frames[0]
    assert all(r['wh']==list(NATIVE_WH) and np.array_equal(r['w2c'],np.eye(4)) for r in rows)
    with np.load(FIT/'selected-fit.npz',allow_pickle=False) as z:
        assert z['frame_ids'].astype(str).tolist()==frames
        assert z['betas'].shape==(1,10) and z['body_pose_axis_angle'].shape==(1,63) and z['pose_embedding'].shape==(1,32)
        assert z['K_integer_centers'].shape==(3,3)
        assert all(np.array_equal(z['K_integer_centers'],np.asarray(r['K_integer_centers'])) for r in rows)
    baseline=gs.read(base.previous.BASELINE/'plan.json')
    assert all(key in baseline for key in PROFILE_KEYS)
    fn,before,changed,connector_diff=adapted_prepare(reference,frames)
    expected_adapter=adapter_from_prepare(changed,reference)
    fn()
    adapter=OUT/'adapter.py'
    original=adapter.read_text();adapter.write_text(base.add_alpha_export(original))
    assert adapter.read_text()==expected_adapter
    check_adapter(adapter.read_text(),reference)
    current=gs.read(OUT/'plan.json')
    assert all(current[key]==baseline[key] for key in PROFILE_KEYS)
    (OUT/'twentyfour-connector.diff').write_text(connector_diff)
    (OUT/'twentyfour-prepare-interface.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),changed.splitlines(True),fromfile='existing-reviewed-interface',tofile='officewoman24-native-inputs')))
    (OUT/'read-only-alpha-export.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),adapter.read_text().splitlines(True),fromfile='same-training-profile',tofile='same-snapshot-alpha-diagnostic')))
    current.update(updated=gs.now(),label='办公室新人物 · 环绕24帧 SH0',record_id='019',
        reference_frame=reference,initial_admission_max_step=1000,automatic_18k_continuation=False,
        user_authorization='做新的人物；半圈12，一圈24；沿已跑通SH0表面绑定路线。',
        hypothesis='Frozen SH0/ROI2x/paired-background/18k training profile applied to a new person and24 reviewed source views.',
        only_parameter_change='New subject, source/count/vertical-pixel/reference connectors and fresh official fit; no Gaussian loss, LR amplitude or opacity change.',
        causal_scope='New-person pipeline test. The predeclared face-visibility confidence layer belongs to fitting only; not a view-count-only ablation or a new Gaussian loss.',
        new_face_visibility_confidence_fit_only=True,
        fixed='SH0;ROI2x;paired backgrounds;original losses/LR amplitudes;18k clock;20908one-per-triangle initialization;empty optimizer;clothing offset1e-6;protected face/head/hands/feet.',
        criteria='Actual all24source and unknown-view review at1000 before a hash-bound continue/stop; at most18000, one candidate, no automatic retry.',
        known_limitation='Source-contact coverage qualitatively spans both sides and rear; no measured360-degree/physical-camera/occluded-surface truth. FullK uses the existing conditional vertical H0 transform.',
        budget_note='Unchanged2400seconds total including review; unchanged600seconds1000review wait.',
        alpha_diagnostic=dict(enabled=True,snapshot_only=True,raw_float32_saved=True,display_only_clamp=True,
            erosion_radius_derived_pixels=5,loss_or_optimizer_changed=False,surface_opacity_acceptance_claim=False),
        copied_profile_keys=list(PROFILE_KEYS),copied_profile_values_exact=True,quality_accepted=False)
    for path in list(current['source_hashes']):
        if Path(path).is_relative_to(OUT):
            current['source_hashes'][path]=gs.sha(path)
    for path in [Path(__file__),Path(base.__file__),gs.HERE/'body_root_sh_colors123.py']:
        current['source_hashes'][str(path)]=gs.sha(path)
    gs.save(OUT/'plan.json',current)
    assert all(current[key]==baseline[key] for key in PROFILE_KEYS)
    print('OFFICEWOMAN24_PROFILE_PREPARED',reference,flush=True)


def preflight():
    edits=[('paired.calls==8',f'paired.calls=={N}'),("fn('eight-preflight')","fn('twentyfour-preflight')")]
    fn,diff=reused(base.previous.preflight,edits,dict(OUT=OUT),'officewoman24-paired-preflight')
    (OUT/'twentyfour-preflight-interface.diff').write_text(diff)
    fn()


def train():
    assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
    assert gs.read(OUT/'validity-cuda-preflight.json')['passed'] and gs.read(OUT/'validity-review.json')['training_admitted']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Registered2400seconds')))
    signal.alarm(2400)
    try:
        gs.train(SUBJECT)
    finally:
        signal.alarm(0)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and not HANDOFF.exists()
    frames=[f'frame_{i:06d}' for i in range(1,N+1)];reference=frames[N//2]
    _,_,changed,_=adapted_prepare(reference,frames)
    adapter=adapter_from_prepare(changed,reference)
    baseline=gs.read(base.previous.BASELINE/'plan.json')
    assert all(key in baseline for key in PROFILE_KEYS)
    ast.parse(inspect.getsource(prepare));ast.parse(inspect.getsource(preflight));ast.parse(adapter)
    from review_officewoman_circle24_gaussian127 import compiled_review
    compiled_review()
    review_path=Path(__file__).with_name('review_officewoman_circle24_gaussian127.py')
    result=dict(updated=gs.now(),state='static_full_prepare_interface_audit_passed',helper_sha256=gs.sha(__file__),
        parent_helper_sha256=gs.sha(base.__file__),frames=N,native_wh=list(NATIVE_WH),reference_from_future_fit_plan=True,
        review_helper_sha256=gs.sha(review_path),complete_CPU_review_compiled_without_execution=True,
        shared_source_K_shape=[3,3],all_profile_copy_keys_checked=list(PROFILE_KEYS),actual_prepare_pre_and_post_profile_value_guards=True,
        complete_generated_adapter_AST_checked=True,no_stale8_or12_numeric_computation=True,
        original_alpha_export_reused_unchanged=True,optimizer_or_loss_changed=False,
        review_timeout_seconds=600,total_budget_seconds=2400,GPU_started=False,fit_read=False,experiment_created=False,
        quality_accepted=False)
    gs.save(Path(__file__).with_suffix('.audit.json'),result)
    print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':
    try:
        mode=sys.argv[1]
        {'audit':audit,'prepare':prepare,'preflight':preflight,'train':train}[mode]()
        if mode=='train':
            gs.save(OUT/'process-exit.json',dict(updated=gs.now(),exit_code=0))
    except BaseException as error:
        if OUT.exists():
            gs.save(OUT/(sys.argv[1]+'-failure.json'),dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise

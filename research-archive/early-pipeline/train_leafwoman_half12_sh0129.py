"""Video22/leafwoman: one immutable fit24 -> selected GS12 input view.

Keep the128 SH0/ROI2x/paired-background profile and complete post-update
checkpoint/offline-review lifecycle. Only subject/pixel/record connectors and
the explicit frame-ID subset before the original mesh bridge are new here.
No source-fit file, confidence statistic, pose, solver or training loss changes.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
from types import SimpleNamespace

import numpy as np
import train_graywoman_half12_sh0128 as lifecycle_base
import train_football_half12_sh0126 as base
import train_indoor_person_zoom124 as zoom
from train_indoor_circle16_sh0124 import reused
import cross_subject_sh0124 as gs

SUBJECT = 'leafwoman-half12-paired18k129'
OUT = gs.ROOT/SUBJECT
FIT = gs.ROOT/'leafwoman-half12-pose129'
HANDOFF = FIT/'gaussian-native-handoff'
SELECTION = FIT/'gs-selection/selection.json'
N, FIT_N, HOLD_N = 12, 24, 4
NATIVE_WH = [2704, 1520]
TOTAL = lifecycle_base.TOTAL
COMPUTE_BUDGET = lifecycle_base.COMPUTE_BUDGET
UPSTREAM_TRAIN_SHA = lifecycle_base.UPSTREAM_TRAIN_SHA
LIFECYCLE_SHA = lifecycle_base.LIFECYCLE_SHA
PROFILE_KEYS = lifecycle_base.PROFILE_KEYS
BASE128_SHA = 'a352cc852c17fbaa0b7ffabec3d6affd33915e4771a411189336f7defc7459a8'
assert gs.sha(lifecycle_base.__file__) == BASE128_SHA

# Shapes are the existing official selected-fit archive schema, not new data.
# A field is never classified solely by its first dimension.
PER_FRAME_TAILS = dict(global_orient=(3,), left_hand_pose=(12,), right_hand_pose=(12,),
    jaw_pose=(3,), leye_pose=(3,), reye_pose=(3,), expression=(10,), camera_translation=(3,),
    vertices_camera=(10475,3), joints_camera=(118,3), joints_projected_native=(118,2),
    observed_native_keypoints=(118,3), frame_ids=(), image_sha256=(), mask_sha256=(), source_frames=())
SHARED_SHAPES = dict(betas=(1,10), body_pose_axis_angle=(1,63), pose_embedding=(1,32),
    faces=(20908,3), K_integer_centers=(3,3), K_compute_float32=(3,3))


def subset_view(source_plan, archive, selected_frames, reference):
    """Pure CPU input view; no IO, resampling, refit or shared-value indexing."""
    assert set(archive) == set(PER_FRAME_TAILS) | set(SHARED_SHAPES), 'Unexpected fit schema; classify it explicitly.'
    ids = archive['frame_ids'].astype(str).tolist()
    assert len(ids) == len(set(ids)) == FIT_N
    assert ids == [row['frame'] for row in source_plan['rows']]
    assert len(selected_frames) == len(set(selected_frames)) == N
    assert set(selected_frames) <= set(ids) and reference in selected_frames
    assert not set(selected_frames) & set(source_plan['review_only_excluded'])
    indices = [ids.index(frame) for frame in selected_frames]
    for key, tail in PER_FRAME_TAILS.items():
        assert np.asarray(archive[key]).shape == (FIT_N,) + tail, (key, np.asarray(archive[key]).shape)
    for key, shape in SHARED_SHAPES.items():
        assert np.asarray(archive[key]).shape == shape, (key, np.asarray(archive[key]).shape)
    arrays = {key: np.asarray(archive[key])[indices].copy() for key in PER_FRAME_TAILS}
    arrays.update({key: np.asarray(archive[key]).copy() for key in SHARED_SHAPES})
    rows = [copy.deepcopy(source_plan['rows'][index]) for index in indices]
    for row, index in zip(rows, range(N)):
        assert row['frame'] == str(arrays['frame_ids'][index])
        assert row['source_sha256'] == str(arrays['image_sha256'][index])
        assert row['mask_sha256'] == str(arrays['mask_sha256'][index])
        assert row['source_frame_index'] == int(arrays['source_frames'][index])
        assert row['wh'] == NATIVE_WH and np.array_equal(row['w2c'], np.eye(4))
        assert np.array_equal(row['K_integer_centers'], arrays['K_integer_centers'])
    result_plan = copy.deepcopy(source_plan)
    result_plan.update(rows=rows, frames=list(selected_frames), reference_frame=reference,
        source_fit_reference_frame=source_plan['reference_frame'],
        source_fit_candidate_count=FIT_N, gaussian_selected_count=N,
        source_fit_files_modified=False, frame_ID_subset_only=True)
    assert all(np.array_equal(arrays[key], archive[key]) for key in SHARED_SHAPES)
    assert arrays['body_pose_axis_angle'].shape == (1,63) and arrays['pose_embedding'].shape == (1,32)
    return result_plan, arrays


class MemoryArchive(dict):
    def __enter__(self): return self
    def __exit__(self, *args): return False


def reader_adapters(plan_view, archive_view):
    """Intercept only originalFIT plan/archive reads inside existing prepare."""
    def read(path):
        if Path(path) == FIT/'plan.json':
            return copy.deepcopy(plan_view)
        return gs.read(path)
    def load(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == FIT/'selected-fit.npz':
            assert not args and kwargs == dict(allow_pickle=False)
            return MemoryArchive({key:value.copy() for key,value in archive_view.items()})
        return np.load(path, *args, **kwargs)
    return (SimpleNamespace(**dict(gs.__dict__, read=read)),
            SimpleNamespace(**dict(np.__dict__, load=load)))


def selection_context():
    """Called only by actual prepare after a frozen, actually reviewed fit."""
    plan, result, review = (gs.read(FIT/name) for name in ('plan.json','result.json','conclusion.json'))
    selection = gs.read(SELECTION)
    assert gs.read(FIT/'process-exit.json')['exit_code'] == 0
    assert review['actual_visual_review'] is True and review['exploratory_gaussian_admitted'] is True
    assert len(plan['rows']) == FIT_N and len(plan['review_only_excluded']) == HOLD_N
    assert plan['old_individual_shape_used'] is False and plan['old_optimized_pose_used'] is False
    assert selection['fit_sha256'] == result['fit_sha256'] == gs.sha(FIT/'selected-fit.npz')
    assert selection['source_plan_sha256'] == gs.sha(FIT/'plan.json')
    source_input = Path(plan['source_selection'])
    assert selection['input_source_plan_sha256'] == gs.sha(source_input)
    assert selection['heldout_frames'] == plan['review_only_excluded']
    frames, reference = selection['training_frames'], selection['reference_frame']
    assert reference in frames
    source_reference = next(row for row in gs.read(source_input)['rows'] if row['frame'] == reference)
    assert source_reference['face_region_is_observed'] is True
    assert source_reference['visible_eyes_open'] is True
    assert result['frames_count'] == FIT_N and result['shared_beta_parameter_count'] == 10
    assert result['old_individual_shape_used'] is False and result['old_optimized_pose_used'] is False
    assert result['hand_data_control']['candidate'] == [0.] * 5
    assert result['hand_data_control']['same_for_all_frames'] is True
    strategy = gs.read(FIT/'confidence-input-strategy.json')
    assert strategy['frozen_statistics_sha256'] == gs.sha(FIT/'train-confidence-normalizer.npz')
    with np.load(FIT/'selected-fit.npz', allow_pickle=False) as z:
        arrays = dict(z)
    plan_view, archive_view = subset_view(plan, arrays, frames, reference)
    for row in plan_view['rows']:
        assert gs.sha(row['image']) == row['source_sha256']
        assert gs.sha(row['mask']) == row['mask_sha256']
    immutable = [FIT/'plan.json', FIT/'result.json', FIT/'conclusion.json', FIT/'process-exit.json', FIT/'selected-fit.npz',
        FIT/'train-confidence-normalizer.npz', FIT/'confidence-input-strategy.json',
        FIT/'source-observation-visibility.json', source_input, SELECTION]
    return dict(selection=selection, frames=list(frames), reference=reference, source_plan=plan,
        plan_view=plan_view, archive_view=archive_view,
        immutable_hashes={str(path):gs.sha(path) for path in immutable})


def adapted_prepare(reference, frames, views=None):
    """Compile the unchanged horizontal12 handoff with the input reader view."""
    edits = [
        ('("record_id=\'001\'", "record_id=\'011\'", 2)', '("record_id=\'001\'", "record_id=\'022\'", 2)'),
        ("code = code.replace('winterhat', 'football').replace('Winterhat', 'Football').replace('冬帽', '足球场新')",
         "code = code.replace('winterhat', 'leafwoman').replace('Winterhat', 'Leafwoman').replace('冬帽', '持叶新拍摄')"),
    ]
    function, diff = reused(base.adapted_prepare, edits, dict(N=N,OUT=OUT,FIT=FIT,
        HANDOFF=HANDOFF,SUBJECT=SUBJECT,__file__=__file__), 'leafwoman129-horizontal12-input-only')
    prepared, before, changed = function(reference, frames)
    assert "row['wh']==[2704,1520]" in changed
    if views is not None:
        gs_view, np_view = reader_adapters(views['plan_view'], views['archive_view'])
        def input_reused(fn, changes, scope, label):
            # The source-only proxies affect just the two originalFIT reads.
            # Native handoff, ROI and all numerical bridge code remain original.
            return reused(fn, changes, dict(scope, gs=gs_view, np=np_view), label)
        prepared.__globals__.update(gs=gs_view, np=np_view, reused=input_reused)
    return prepared, before, changed, diff


# Preserve lifecycle implementation code objects. New globals point solely to
# this experiment; no128 fit, Gaussian values, optimizer or RNG are imported.
_life_ns = dict(lifecycle_base.__dict__)
_life_ns.update(globals())
_LIFE_NAMES = ('adapter_from_prepare','lifecycle_adapter','_compute_elapsed',
    '_save_frozen_metadata','_write_frozen_text','_verify_decision','_begin_phase',
    '_stage_preview_after_checkpoint','_adapter_namespace','preflight_function','train')
for _name in _LIFE_NAMES:
    _fn = getattr(lifecycle_base, _name)
    globals()[_name] = types.FunctionType(_fn.__code__, _life_ns, _fn.__name__, _fn.__defaults__, _fn.__closure__)
    assert globals()[_name].__code__ is _fn.__code__


def _actual_training_sources(local):
    fn = lifecycle_base._actual_training_sources
    bound = types.FunctionType(fn.__code__, _life_ns, fn.__name__, fn.__defaults__, fn.__closure__)
    return sorted(set(bound(local)) | {Path(lifecycle_base.__file__).resolve()}, key=str)


# Only the lifecycle identity's subject/version label changes.
_identity_before = inspect.getsource(lifecycle_base._make_lifecycle)
assert _identity_before.count("schema='graywoman128-staged-original-18k-v1'") == 1
_identity_after = _identity_before.replace("schema='graywoman128-staged-original-18k-v1'",
                                         "schema='leafwoman129-staged-original-18k-v1'")
exec(compile(_identity_after, str(__file__)+'::unchanged128_lifecycle_new_identity', 'exec'), _life_ns)
_make_lifecycle = _life_ns['_make_lifecycle']


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists() and not HANDOFF.exists()
    context = selection_context()
    frames, reference = context['frames'], context['reference']
    fn, before, changed, connector_diff = adapted_prepare(reference, frames, context)
    expected = lifecycle_adapter(adapter_from_prepare(changed, reference))[0]
    baseline = gs.read(base.previous.BASELINE/'plan.json')
    fn()
    adapter = OUT/'adapter.py'
    original = base.add_alpha_export(adapter.read_text())
    code, lifecycle_receipt = lifecycle_adapter(original)
    assert code == expected
    adapter.write_text(code)
    (OUT/'leafwoman-input-connectors.diff').write_text(connector_diff)
    (OUT/'staged-lifecycle-interface.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), code.splitlines(True), fromfile='unchanged-scientific-profile',
        tofile='original128-complete-checkpoint-offline-review')))
    with np.load(HANDOFF/'fit/fitted.npz', allow_pickle=False) as z:
        final = dict(z)
    final_ids = final['frame_ids'].astype(str).tolist()
    expected_ids = [reference] + [frame for frame in frames if frame != reference]
    assert final_ids == expected_ids
    assert final['betas'].shape == final['shared_betas'].shape == (1,10)
    for key in SHARED_SHAPES:
        if key != 'K_integer_centers':
            # The original bridge repeats only the shared local pose and latent.
            value = context['archive_view'][key]
            wanted = np.repeat(value, N, axis=0) if key in ('body_pose_axis_angle','pose_embedding') else value
            assert np.array_equal(final[key], wanted), key
    for key in PER_FRAME_TAILS:
        order = [context['frames'].index(frame) for frame in expected_ids]
        assert np.array_equal(final[key], context['archive_view'][key][order]), key
    assert final['K_integer_centers'].shape == (N,3,3)
    assert all(np.array_equal(k,context['archive_view']['K_integer_centers']) for k in final['K_integer_centers'])
    assert context['immutable_hashes'] == {path:gs.sha(path) for path in context['immutable_hashes']}
    # These new output descriptions must identify the actual post-fit subset,
    # rather than inherit the old "all fitted views" labels from the template.
    for folder in (HANDOFF, OUT):
        for name in ('selection.json','selection-rule.json'):
            description = gs.read(folder/name)
            key = 'selected_frames' if name == 'selection.json' else 'frames'
            assert description[key] == expected_ids
            description.update(original_fit_candidate_count=FIT_N, selected_gaussian_count=N,
                postfit_selection_receipt=str(SELECTION), postfit_selection_sha256=gs.sha(SELECTION),
                source_fit_files_modified=False, shared_betas_and_pose_not_refit=True,
                policy='Frozen deterministic quality/distribution subset of the same24-view fit; no refit or heldout transfer.',
                rule='Use the frozen post-fit frame-ID subset only; no post-training selection.')
            gs.save(folder/name, description)
    handoff_plan = gs.read(HANDOFF/'plan.json')
    for path in list(handoff_plan['source_hashes']):
        if Path(path).is_relative_to(HANDOFF): handoff_plan['source_hashes'][path] = gs.sha(path)
    gs.save(HANDOFF/'plan.json', handoff_plan)
    subset_receipt = dict(updated=gs.now(), original_fit_candidate_count=FIT_N,
        selected_gaussian_count=N, selected_frames=frames, handoff_reference_first_frames=expected_ids,
        reference_frame=reference, original_fit_reference_frame=context['source_plan']['reference_frame'],
        selection_path=str(SELECTION), selection_sha256=gs.sha(SELECTION),
        immutable_source_hashes=context['immutable_hashes'],
        per_frame_whitelist=list(PER_FRAME_TAILS), shared_whitelist=list(SHARED_SHAPES),
        original_fit_and_train24_q90_unchanged=True, shared_betas_and_pose_not_refit=True,
        handoff_full12_original_forward_oracle_preserved=True, quality_accepted=False)
    gs.save(OUT/'fit24-to-GS12-subset-receipt.json', subset_receipt)
    current = gs.read(OUT/'plan.json')
    assert all(current[key] == baseline[key] for key in PROFILE_KEYS)
    current.update(updated=gs.now(), subject=SUBJECT, label='持叶新拍摄 · 半圈12帧 SH0', record_id='022',
        reference_frame=reference, source_fit_candidate_count=FIT_N, selected_gaussian_count=N,
        selection_receipt=str(SELECTION), selection_receipt_sha256=gs.sha(SELECTION),
        subset_receipt=str(OUT/'fit24-to-GS12-subset-receipt.json'),
        initial_admission_max_step=1000, automatic_18k_continuation=False,
        fixed='Original SH0/ROI2x/paired Uniform/18k clock/20908 one per triangle/offset1e-6 and protected face-head-hands-feet.',
        only_parameter_change='New capture and immutable fit24 frame-ID subset12; original128 complete checkpoint/offline review, training science unchanged.',
        known_limitation='Source-only stable chest-arm group. Cropped/occluded lower body and fingers remain unknown; conditionalK is not independently verified calibration.',
        budget_note='2400 seconds cumulative computation; offline review wall time separate. No live600-second wait.',
        lifecycle=dict(lifecycle_receipt, helper_sha256=LIFECYCLE_SHA, review_offline=True,
            first_checkpoint_optimizer_updates=1000, first_resume_iteration=1001,
            original_terminal18000_optimizer_updates=17999, total_compute_budget_seconds=COMPUTE_BUDGET),
        quality_accepted=False, model_promoted=False, end_to_end_15min_accepted=False)
    for path in list(current['source_hashes']):
        if Path(path).is_relative_to(OUT) or Path(path).is_relative_to(HANDOFF):
            current['source_hashes'][path] = gs.sha(path)
    extra = [Path(__file__), Path(lifecycle_base.__file__), Path(base.__file__),
        gs.HERE/'staged_training_lifecycle128.py', gs.HERE/'body_root_sh_colors123.py',
        OUT/'fit24-to-GS12-subset-receipt.json', *map(Path,context['immutable_hashes'])]
    current['source_hashes'].update({str(path):gs.sha(path) for path in extra})
    gs.save(OUT/'plan.json', current)
    assert all(current[key] == baseline[key] for key in PROFILE_KEYS)
    print('LEAFWOMAN12_SUBSET_LIFECYCLE_PREPARED', reference, flush=True)


def preflight():
    function, diff = preflight_function()
    (OUT/'preflight-interface.diff').write_text(diff)
    function('leafwoman-preflight')


def synthetic_subset_checks():
    ids = [f'frame_{i:06d}' for i in range(1,FIT_N+1)]
    arrays = {key:np.zeros((FIT_N,)+tail,np.float32) for key,tail in PER_FRAME_TAILS.items()}
    for key, tail in PER_FRAME_TAILS.items():
        if key not in ('frame_ids','image_sha256','mask_sha256'):
            arrays[key][...] = np.arange(FIT_N).reshape((FIT_N,)+(1,)*len(tail))
    arrays['frame_ids'] = np.asarray(ids)
    arrays['image_sha256'] = np.asarray([f'image-{i}' for i in range(FIT_N)])
    arrays['mask_sha256'] = np.asarray([f'mask-{i}' for i in range(FIT_N)])
    arrays.update({key:np.arange(np.prod(shape)).reshape(shape) for key,shape in SHARED_SHAPES.items()})
    K = np.array([[1120.,0,1353.],[0,1119.,764.],[0,0,1.]],np.float64)
    arrays['K_integer_centers'] = K
    arrays['K_compute_float32'] = K.astype(np.float32)
    rows = [dict(frame=frame, source_sha256=str(arrays['image_sha256'][i]),
        mask_sha256=str(arrays['mask_sha256'][i]), source_frame_index=int(arrays['source_frames'][i]),
        wh=NATIVE_WH, w2c=np.eye(4).tolist(), K_integer_centers=K.tolist()) for i,frame in enumerate(ids)]
    plan = dict(rows=rows, frames=ids, reference_frame=ids[5],
        review_only_excluded=['held-a','held-b','held-c','held-d'])
    before = {key:value.copy() for key,value in arrays.items()}
    selected = ids[1::2];reference = selected[3]
    view, subset = subset_view(plan, arrays, selected, reference)
    assert len(view['rows']) == N and subset['frame_ids'].astype(str).tolist() == selected
    assert all(np.array_equal(arrays[key],before[key]) for key in arrays)
    assert all(np.array_equal(subset[key],arrays[key]) for key in SHARED_SHAPES)
    assert all(np.array_equal(subset[key],arrays[key][1::2]) for key in PER_FRAME_TAILS)
    # Exercise the original native.prepare reference-first ordering and shared
    # expansion using its exact AST statements before any model construction.
    source = inspect.getsource(base.previous.native.prepare)
    tree = ast.parse(source)
    first = next(i for i,node in enumerate(tree.body[0].body) if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='order' for t in node.targets))
    last = next(i for i,node in enumerate(tree.body[0].body[first:],first) if isinstance(node,ast.For))
    namespace = dict(np=np, fit=copy.deepcopy(subset), rows=copy.deepcopy(view['rows']),
                     ids=list(selected), n=N, REFERENCE=reference)
    exec(compile(ast.Module(body=tree.body[0].body[first:last],type_ignores=[]),'<original-native-reference-order-no-model>', 'exec'),namespace)
    native = namespace['fit'];ordered = [reference] + [f for f in selected if f != reference]
    indices = [ids.index(frame) for frame in ordered]
    assert native['frame_ids'].astype(str).tolist() == ordered
    assert native['betas'].shape == native['shared_betas'].shape == (1,10)
    assert native['body_pose_axis_angle'].shape == (N,63) and native['pose_embedding'].shape == (N,32)
    assert np.array_equal(native['body_pose_axis_angle'],np.repeat(arrays['body_pose_axis_angle'],N,axis=0))
    assert native['K_integer_centers'].shape == (N,3,3)
    assert all(np.array_equal(native[key],arrays[key][indices]) for key in PER_FRAME_TAILS)
    gs_view,np_view = reader_adapters(view,subset)
    readback = gs_view.read(FIT/'plan.json')
    with np_view.load(FIT/'selected-fit.npz',allow_pickle=False) as z:
        assert np.array_equal(z['frame_ids'],subset['frame_ids'])
        z['betas'][0,0] = -100
    assert np.array_equal(subset['betas'],arrays['betas'])
    assert readback['rows'] == view['rows']
    prepared,_,_,_ = adapted_prepare(reference,selected,dict(plan_view=view,archive_view=subset))
    deep,_ = prepared.__globals__['reused'](base.previous.native.prepare,
        [('assert n == 7','assert n == 12')],
        dict(OUT=HANDOFF,SOURCE=FIT,SUBJECT=SUBJECT,REFERENCE=reference,__file__=__file__),
        'synthetic-original-native-input-namespace-only')
    assert deep.__globals__['gs'].read(FIT/'plan.json')['frames'] == selected
    with deep.__globals__['np'].load(FIT/'selected-fit.npz',allow_pickle=False) as z:
        assert z['frame_ids'].astype(str).tolist() == selected
    # Execute only the original native input/order block, stopping before its
    # source-image IO, torch/SMPL-X model construction or any mkdir/save.
    deep_tree = ast.parse(source.replace('assert n == 7','assert n == 12'))
    body = deep_tree.body[0].body
    stop = next(i for i,node in enumerate(body) if isinstance(node,ast.For))
    deep_tree.body[0].body = body[:stop] + [ast.parse('return locals()').body[0]]
    deep_scope = dict(deep.__globals__)
    exec(compile(ast.fix_missing_locations(deep_tree),'<deep-original-input-read-no-model-no-output>','exec'),deep_scope)
    replay = deep_scope['prepare']()
    assert replay['ids'] == selected and replay['fit']['frame_ids'].astype(str).tolist() == ordered
    assert all(np.array_equal(replay['fit'][key],native[key]) for key in native)
    rejected = []
    cases = dict(duplicate=selected[:-1]+[selected[0]], outside=selected[:-1]+['not-fit'],
                 heldout=selected[:-1]+['held-a'])
    for name, bad in cases.items():
        try: subset_view(plan,arrays,bad,reference)
        except AssertionError: rejected.append(name)
        else: raise AssertionError('Invalid selection accepted: '+name)
    bad = dict(arrays, unexpected=np.zeros(24))
    try: subset_view(plan,bad,selected,reference)
    except AssertionError: rejected.append('unclassified_key')
    else: raise AssertionError('Unknown array key accepted.')
    return dict(synthetic_only=True, fit_rows=FIT_N, GS_rows=N, per_frame_keys=len(PER_FRAME_TAILS),
        shared_keys=len(SHARED_SHAPES), original_arrays_unchanged=True,
        original_native_order_and_repeat_executed_without_model=True, reference_first=True,
        nested_reused_native_reader_namespace_and_original_input_replay_verified=True,
        shared_parameters_and_fullK_exact=True, invalid_selections_rejected=rejected,
        reader_intercepts_only_original_plan_and_archive=True, real_person_values_loaded=False)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists() and not HANDOFF.exists()
    synthetic = synthetic_subset_checks()
    frames = [f'frame_{i:06d}' for i in range(1,N+1)];reference = frames[5]
    _,_,changed,_ = adapted_prepare(reference,frames)
    original = adapter_from_prepare(changed,reference)
    adapter, lifecycle = lifecycle_adapter(original)
    assert "receipt['frames'] == 12" in adapter and 'counts = np.zeros(12, dtype=int)' in adapter
    assert "row['wh']==[2704,1520]" in changed
    assert 'shared_laplacian_weight' in changed and 'shared_laplacian_wtwelve' not in changed
    assert 'Review timeout' not in adapter and 'deadline = ' not in adapter
    preflight_function()
    assert gs.sha(gs.UP/'train.py') == UPSTREAM_TRAIN_SHA
    assert gs.sha(gs.HERE/'staged_training_lifecycle128.py') == LIFECYCLE_SHA
    assert gs.sha(lifecycle_base.__file__) == BASE128_SHA
    import torch
    assert not torch.cuda.is_initialized()
    assert not OUT.exists() and not HANDOFF.exists()
    receipt = dict(updated=gs.now(), state='CPU_static_synthetic_subset129_passed',
        helper_sha256=gs.sha(__file__), inherited128_helper_sha256=BASE128_SHA,
        synthetic_subset=synthetic, lifecycle=lifecycle,
        lifecycle_functions_code_object_identical={name:globals()[name].__code__ is getattr(lifecycle_base,name).__code__ for name in _LIFE_NAMES},
        new_identity_label_only_diff=''.join(difflib.unified_diff(_identity_before.splitlines(True),_identity_after.splitlines(True))),
        original_profile_AST_preserved=True, fit_candidate_count=FIT_N, GS_count=N,
        native_wh=NATIVE_WH, record_id='022', sourcefit24_or_selection_read=False,
        train24_q90_refitted=False, original_fit_files_modified=False,
        original18k_terminal_updates=17999, post_update1000_offline_exit=True,
        automatic18k_continuation=False, cumulative_compute_budget_seconds=COMPUTE_BUDGET,
        actual_prepare=False, GPU_executed=False, experiment_created=False, quality_accepted=False)
    gs.save(Path(__file__).with_suffix('.audit.json'),receipt)
    print(json.dumps(receipt),flush=True)


# Rebound lifecycle calls resolve all mutual helpers only after definitions.
_life_ns.update(globals())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit','prepare','preflight','train','resume'])
    mode = parser.parse_args().mode
    try:
        if mode in ('train','resume'):
            train(mode)
            path = OUT/('process-exit.json' if mode=='train' else 'resume-process-exit.json')
            assert not path.exists()
            gs.save(path,dict(updated=gs.now(),exit_code=0,state=gs.read(OUT/'progress.json').get('state')))
        else:
            globals()[mode]()
    except BaseException as error:
        if OUT.exists():
            path = OUT/(mode+'-failure.json')
            if not path.exists(): gs.save(path,dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise

"""DRAFT video22/leafwoman selected-fit interface; source roster not consumed.

Reuse128 official5stage/max30/shared neutral beta and body pose, original
loss/prior/optimizer, source-confidence COPY gates and null empty metrics.
BODY25 wrists remain enabled wherever this source explicitly supports them;
no128 wrist-all-false values are copied. Cropped/hidden knee/feet are unknown,
not a reason to synthesize coordinates or add losses. The eventual source plan
must explicitly admit one stable arm-pose group before any shared-pose fit.
The first-YAML native horizontalK remains conditional, not calibration proof.
"""
import ast
import copy
import difflib
import hashlib
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import run_officewoman_circle24_fit127 as base

original, winterhat, prior = base.original, base.winterhat, base.prior
read, save, sha, now = base.read, base.save, base.sha, base.now
HAND_CONTROL, hand_control = base.HAND_CONTROL, base.hand_control
OUT = original.ROOT / 'leafwoman-half12-pose129'
AUDIT = original.ROOT / 'leafwoman-half12-input129'
SOURCE = Path('/opt/gs-slam-lab/shuolee-cross-subject-129/leafwoman')
CPU_AUDIT = Path(__file__).with_name('run_leafwoman_half12_fit129_cpu_audit.json')
STATIC_AUDIT = Path(__file__).with_name('run_leafwoman_half12_fit129_static_audit.json')
BASE_SHA = '0dd3be080d32705bc4f6e8315eb0cacaeea66a4599e478676409abf7156bcac3'
# Prospective fit-candidate count only; final Gaussian roster target is12.
# Bind/freeze N against the eventual source plan before any actual audit/prepare.
N, HOLD_COUNT, SUBJECT = 24, 4, 'leafwoman'
GAUSSIAN_VIEW_TARGET = 12
NATIVE_WH = [2704, 1520]
FACE_SOURCE, FACE_TARGET = base.FACE_SOURCE.copy(), base.FACE_TARGET.copy()
GROUP_SOURCE = dict(left_knee=[13], right_knee=[14], left_foot=[15,17,18,19], right_foot=[16,20,21,22],
                    left_wrist=[9], right_wrist=[10], left_elbow=[7], right_elbow=[8])
GROUP_TARGET = dict(left_knee=[13], right_knee=[10], left_foot=[14,19,20,21], right_foot=[11,22,23,24],
                    left_wrist=[7], right_wrist=[4], left_elbow=[6], right_elbow=[3])
EXPECTED_GROUP_SOURCE, EXPECTED_GROUP_TARGET = copy.deepcopy(GROUP_SOURCE), copy.deepcopy(GROUP_TARGET)
MASKABLE_SOURCE = np.unique(np.concatenate([FACE_SOURCE] + [np.asarray(x) for x in GROUP_SOURCE.values()]))
MASKABLE_TARGET = np.unique(np.concatenate([FACE_TARGET] + [np.asarray(x) for x in GROUP_TARGET.values()]))
config_reads = []


def supported_metrics(errors, weights, name):
    """Post-fit reporting only; exactly the old formula for nonempty support."""
    errors, weights = np.asarray(errors), np.asarray(weights)
    assert errors.shape == weights.shape and errors.ndim == 1
    assert np.isfinite(errors).all() and np.isfinite(weights).all() and (weights >= 0).all()
    count = int(np.count_nonzero(weights > 0))
    return {name + '_supported_count': count,
            name + '_weighted_rms_px': float(np.sqrt(np.sum(weights * errors ** 2) / weights.sum())) if count else None,
            name + '_supported_max_px': float(errors[weights > 0].max()) if count else None}


def source_pose_group_guard(p):
    """Source-only admission for one shared-pose fit, before any optimizer."""
    assert N == 24 and HOLD_COUNT == 4
    assert GAUSSIAN_VIEW_TARGET == 12 and GAUSSIAN_VIEW_TARGET != N
    assert len(p['training_frames']) == N and len(p['heldout_frames']) == HOLD_COUNT
    grouping = p['source_pose_grouping']
    assert grouping['actual_source_only_review'] is True
    assert grouping['selected_training_pose_group'] == 'early_chest_leaf'
    assert grouping['selected_training_arms_stable'] is True
    by_frame = grouping['training_group_by_frame']
    assert set(by_frame) == set(p['training_frames'])
    assert all(by_frame[frame] == 'early_chest_leaf' for frame in p['training_frames'])
    # Source grouping receipts are bound by the existing source_hashes checks.
    # The one24-view fit and the later quality/distribution-based GS12 subset
    # are distinct rosters; this guard never picks a fit winner or moves heldout.
    assert isinstance(p['postfit_selection_protocol'], dict) and p['postfit_selection_protocol']


def visibility_record(p):
    frames = p['training_frames'] + p['heldout_frames']
    flags, groups = p['face_visibility'], p['joint_group_visibility']
    assert len(frames) == N + HOLD_COUNT and set(flags) == set(groups) == set(frames)
    assert all(type(flags[f]) is bool for f in frames)
    assert all(set(groups[f]) == set(GROUP_SOURCE) and all(type(groups[f][k]) is bool for k in GROUP_SOURCE) for f in frames)
    assert all(row['face_region_is_observed'] is flags[row['frame']] for row in p['rows'])
    assert all(row['source_joint_group_supported'] == groups[row['frame']] for row in p['rows'])
    reference = next(row for row in p['rows'] if row['frame'] == p['reference_frame'])
    assert flags[p['reference_frame']] is True and reference['visible_eyes_open'] is True
    return dict(subject=SUBJECT, training_frames=p['training_frames'], heldout_frames=p['heldout_frames'],
        face_visibility={f:flags[f] for f in frames}, joint_group_visibility={f:groups[f] for f in frames},
        joint_group_source_indices=GROUP_SOURCE, joint_group_target_indices=GROUP_TARGET,
        source_score_indices=FACE_SOURCE.tolist(), mapped_target_indices=FACE_TARGET.tolist(),
        source_only_actual_review=True, unknown_policy='halt', raw_XY_and_archives_unchanged=True,
        prior_amplitudes_and_hand_schedule_unchanged=True,
        policy='On a scores COPY, source-reviewed unsupported head/face, knee/foot, wrist/elbow groups become0 BEFORE unchanged trainN q90. All training+heldout source flags must be bool; unknown stops. Zero rows remain in original quantile formula; weights are heuristic, not physical truth.')


# Exact string constants only: these are receipt labels, not solver expressions.
LABELS = {
    '<same-algorithm-trainN-statistics-officewoman127-mapping>': '<same-algorithm-trainN-statistics-leafwoman129-mapping>',
    'Unmodified relative_confidence; select24 before fitting its q90 statistic.': 'Unmodified relative_confidence; selectN before fitting its q90 statistic.',
    'OFFICEWOMAN_CIRCLE24_NEUTRAL_FIT127_REGISTERED': 'LEAFWOMAN_SELECTED_NEUTRAL_FIT129_REGISTERED',
    'OFFICEWOMAN_CIRCLE24_FIT127_CPU_PREPARATION_PASSED': 'LEAFWOMAN_SELECTED_FIT129_CPU_PREPARATION_PASSED',
    'Same relative_confidence formula and parameters, now select24 before estimating q90. Holdout coordinates and score statistics do not enter fitting. Future validation applies saved trainN q90 without refitting.':
        'Same relative_confidence formula and parameters, now selectN before estimating q90. Heldout4 coordinates and scores never enter fitting; validation uses saved trainN q90.',
    'Source-reviewed unobserved head/face and knee/foot-group confidence zero on scores copy before unchanged training-only q90; XY/rawNPZ/priors unchanged. Not a frame-count-only ablation.':
        'Source-reviewed unsupported head/face, knee/foot and wrist/elbow confidence zero on scores COPY before unchanged trainN q90; XY/rawNPZ/priors unchanged. Not a frame-count-only ablation.',
    'Necessary new train/validation split boundary: q90 estimated on selected24 rather than full_sourceN; source-reviewed invisible head/face and knee/foot-group score copy set0 before unchanged q90; no solver/loss/threshold change.':
        'Necessary new-capture N/4 input boundary: trainN-only q90; reviewed invisible face/head and unsupported knee/foot/wrist/elbow scores zeroed on COPY. No optimizer, loss, prior, schedule or threshold change.',
    'Apply the same neutral shared-shape selected-view and current Gaussian pipeline to the new officewoman person, using a source-only24+heldout split.':
        'Apply the same neutral shared-shape selected-view/current Gaussian pipeline to the leafwoman capture instance, using a source-only N+4 split.',
    'Official neutral zero10D beta, zero VPoser, standard front body roots in all24. The designated source/reference frame is not a beta/pose initializer. No old officewoman optimized values are loaded.':
        'Official neutral zero10D beta, zero VPoser, standard front body roots in all selected views. Reference labels output identity only; no old individual optimized shape/pose/geometry is loaded.',
    'All24/both hands use original hand_joints_weights=[0,0,0,0,0]; BODY25 wrists and hand pose priors remain unchanged.':
        'All selected views/both hands use original hand_joints_weights=[0,0,0,0,0]. BODY25 wrist/elbow data use explicit source-support COPY gates; mapping and pose-prior amplitudes remain unchanged.',
    'Combined new full-circle roster, source-only face and knee/foot-group visibility gating, fresh neutral shared shape and current pipeline; no isolated frame-count claim or reuse of historical validation as independent evidence.':
        'Combined new half-circle source roster, source-only face/knee/foot/wrist/elbow visibility input, fresh neutral shared shape and current pipeline; no isolated view-count claim or historical validation reuse.',
    'Standard front rotation for all24; source/reference frame names output identity only; front/back orientation can remain ambiguous until actual full-circle fit review.':
        'Standard front rotation for all selected views; source/reference labels are not initial geometry. Actual native/side fit review is required for orientation and hidden-limb plausibility.',
}


def adapted_function(name, replacements=()):
    source = inspect.getsource(getattr(base, name))
    for old, new in replacements:
        assert source.count(old) == 1, (name, old, source.count(old))
        source = source.replace(old, new)
    tree = ast.parse(source)
    class MetadataOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is str and node.value in LABELS:
                return ast.copy_location(ast.Constant(LABELS[node.value]), node)
            return node
        def visit_keyword(self, node):
            self.generic_visit(node)
            if name == 'prepare' and node.arg == 'round':
                assert isinstance(node.value, ast.Constant) and node.value.value == 127
                node.value = ast.copy_location(ast.Constant(129), node.value)
            if name == 'prepare' and node.arg == 'record_id':
                assert isinstance(node.value, ast.Constant) and node.value.value == '019'
                node.value = ast.copy_location(ast.Constant('022'), node.value)
            return node
    tree = MetadataOnly().visit(tree)
    ast.fix_missing_locations(tree)
    return tree, ast.unparse(tree) + '\n', inspect.getsource(getattr(base, name))


ADAPTER128 = Path(__file__).with_name('run_graywoman_half12_fit128.py')
ADAPTER128_SHA = 'a524289b0a80868b922492edf4652282fa1dbe97d92afa2d52e19c97ead3821e'
assert sha(ADAPTER128) == ADAPTER128_SHA
assert sha(base.__file__) == BASE_SHA
_ns = dict(base.__dict__)
_ns.update(globals())
FUNCTION_DIFFS = {}
_function_changes = {
    'source_plan': [("assert p['source_root'] == '/opt/gs-slam-lab/shuolee-cross-subject-127/officewoman'", "assert p['source_root'] == str(SOURCE)"),
                    ('len(hold) > 0', 'len(hold) == HOLD_COUNT'),
                    ('    return p\n', '    source_pose_group_guard(p)\n    return p\n')],
    'data': [('assert np.array_equal(target[:,[4,7]], unmasked_target[:,[4,7]])', 'assert np.array_equal(target[:,[9,12]], unmasked_target[:,[9,12]])')],
    'synthetic_visibility_checks': [('count = N + 8', 'count = N + HOLD_COUNT'),
        ('np.random.default_rng(127)', 'np.random.default_rng(128)'), ('heldout_rows=8,seed=127', 'heldout_rows=HOLD_COUNT,seed=128'),
        ('all4_group_target_confidences_zero_where_unsupported', 'all8_group_target_confidences_zero_where_unsupported'),
        ('body_hips9_12_and_wrists4_7_exact=np.array_equal(masked[:,[4,7,9,12]],baseline[:,[4,7,9,12]])', 'body_hips9_12_exact=np.array_equal(masked[:,[9,12]],baseline[:,[9,12]])'),
        ('GROUP_SOURCE==dict(left_knee=[13],right_knee=[14],left_foot=[15,17,18,19],right_foot=[16,20,21,22]) and GROUP_TARGET==dict(left_knee=[13],right_knee=[10],left_foot=[14,19,20,21],right_foot=[11,22,23,24])', 'GROUP_SOURCE==EXPECTED_GROUP_SOURCE and GROUP_TARGET==EXPECTED_GROUP_TARGET')],
    'prepare': [('paths = [Path(__file__), Path(winterhat.__file__)', 'paths = [Path(__file__), ADAPTER128, Path(base.__file__), Path(winterhat.__file__)')],
}
for _name in ('local_path','visibility_score_copy','visibility_sha256','source_plan','data','frozen_confidence',
              'synthetic_visibility_checks','confidence_audit','hand_config','make_entry','prepare','preflight','run','launch','audit'):
    _tree, _executed, _before = adapted_function(_name, _function_changes.get(_name, ()))
    exec(compile(_tree, str(__file__) + '::existing_' + _name, 'exec'), _ns)
    globals()[_name] = _ns[_name]
    FUNCTION_DIFFS[_name] = ''.join(difflib.unified_diff(_before.splitlines(True), _executed.splitlines(True),
                                         fromfile='frozen127.' + _name, tofile='leafwoman129.' + _name))


runner = winterhat.runner
_bad = "                w=target[i,ix,2]**2;metric[name+'_weighted_rms_px']=float(np.sqrt(np.sum(w*error[ix]**2)/w.sum()))\n                metric[name+'_supported_max_px']=float(error[ix][w>0].max())"
_fixed = "                w=target[i,ix,2]**2\n                metric.update(supported_metrics(error[ix],w,name))"
assert runner.count(_bad) == 1
runner = runner.replace(_bad, _fixed)
assert runner.split('        metrics=[]')[0] == winterhat.runner.split('        metrics=[]')[0]
RUNNER_DIFF = ''.join(difflib.unified_diff(winterhat.runner.splitlines(True), runner.splitlines(True),
                          fromfile='frozen_original_shared_runner', tofile='leafwoman129_empty_metric_guard_only'))
scope = prior.shared.scope.copy()
scope.update(OUT=OUT, data=data, supported_metrics=supported_metrics,
    adapter=SimpleNamespace(**dict(original.adapter.__dict__, make_entry=make_entry)))
exec(compile(runner, str(__file__) + '::same_original_neutral_shared_runner', 'exec'), scope)
_ns.update(scope=scope, runner=runner)


def static_audit():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not OUT.exists(), 'Do not prepare or overwrite any fit in this CPU draft audit.'
    errors = np.asarray([1., 3., 5.], dtype=np.float32)
    weights = np.asarray([.1, 0., .4], dtype=np.float32)
    zero = supported_metrics(errors, weights * 0, 'face')
    nonzero = supported_metrics(errors, weights, 'body')
    assert zero == dict(face_supported_count=0, face_weighted_rms_px=None, face_supported_max_px=None)
    assert nonzero['body_supported_count'] == 2
    assert nonzero['body_weighted_rms_px'] == float(np.sqrt(np.sum(weights * errors ** 2) / weights.sum()))
    assert nonzero['body_supported_max_px'] == float(errors[weights > 0].max())
    synthetic = synthetic_visibility_checks()
    assert synthetic['train_rows'] == N and synthetic['heldout_rows'] == HOLD_COUNT and all(synthetic['checks'].values())
    assert 'reference-fit/winner-camera-space.npz' not in runner
    assert 'beta=np.zeros((1,10),dtype=np.float32)' in runner and 'freeze_betas=False' in runner
    assert 'make_models(len(p["frames"]))' in runner
    assert not torch.cuda.is_initialized() and not OUT.exists()
    result = dict(updated=now(), passed=True, CPU_only=True, source_plan_frozen_or_read=False,
        helper_sha256=sha(__file__), frozen_base_sha256=BASE_SHA,
        N=N, prospective_holdout_count=HOLD_COUNT, subject=SUBJECT, source_root=str(SOURCE),
        fit_output=str(OUT), input_plan=str(AUDIT/'plan.json'), record_id='022', native_wh=NATIVE_WH,
        official_solver_new_code=False, original5stages_max30_LBFGS_priors_preserved=True,
        neutral_shared10D_beta_zero32D_latent_standard_front_roots=True,
        hand_data_control_uniform_zero=True, source_visibility_groups=GROUP_SOURCE,
        target_visibility_groups=GROUP_TARGET, unknown_source_visibility_policy='halt',
        trainN_only_q90_raw_XY_and_NPZ_preserved=True,
        empty_support_metrics=zero, nonempty_metric_formula_exact=True,
        synthetic_visibility_contract=synthetic, runner_diff=RUNNER_DIFF,
        adapter_function_diffs=FUNCTION_DIFFS, no_fit_directory_created=True,
        no_prepare_no_GPU_no_training=True, quality_accepted=False)
    save(STATIC_AUDIT, result)
    print('LEAFWOMAN_SELECTED_FIT129_STATIC_ONLY_PASSED', sha(__file__), flush=True)


if __name__ == '__main__':
    {'static-audit': static_audit, 'audit': audit, 'prepare': prepare,
     'preflight': preflight, 'run': run, 'launch': launch}[sys.argv[1]]()

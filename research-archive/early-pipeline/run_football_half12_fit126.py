"""Fresh football12 shared fit; reuse the reviewed neutral winterhat interfaces.

The new12+4 source-only split must be frozen and reviewed before prepare. No old
football shape, fitted pose, geometry or Gaussian is read as initialization. The
fitter retains the existing uniform zero hand2D-data schedule, as in winterhat;
wrists/priors remain. The new split fits confidence statistics on training12
only, with the existing formula and constants, before any new fitting.
"""
import copy
import difflib
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import run_winterhat_half8_fit124 as winterhat
from run_official_gaussians123 import read, save, sha, now

prior = winterhat.prior
original = winterhat.original
OUT = original.ROOT / 'football-half12-pose126'
AUDIT = original.ROOT / 'football-half12-input126'
CPU_AUDIT = Path(__file__).with_name('run_football_half12_fit126_cpu_audit.json')
HAND_CONTROL = winterhat.HAND_CONTROL
hand_control = winterhat.hand_control
config_reads = []


def local_path(value):
    """The source audit is Windows-authored; only paths change for WSL IO."""
    value = str(value).replace('\\', '/')
    if len(value) >= 3 and value[1:3] == ':/':
        value = '/mnt/' + value[0].lower() + value[2:]
    return Path(value)


def source_plan():
    p = read(AUDIT / 'plan.json')
    for key in ('source_root', 'source_inputs', 'source_observations'):
        p[key] = str(local_path(p[key]))
    p['source_hashes'] = {str(local_path(path)): digest for path, digest in p['source_hashes'].items()}
    assert p['subject'] == 'football'
    assert p['source_root'] == '/opt/gs-slam-lab/shuolee-cross-subject-126/football'
    ids = p['training_frames']
    hold = p['heldout_frames']
    assert len(ids) == len(set(ids)) == 12 and ids == sorted(ids)
    assert len(hold) == len(set(hold)) == 4 and not set(ids) & set(hold)
    reserved = {frame for slot in p['holdout_reservation'] for frame in slot['candidate_order']}
    assert len(reserved) == 8 and set(hold) <= reserved and not set(ids) & reserved
    assert p['reference_frame'] in ids
    assert p['training_roster_ready'] is True and p['actual_source_visual_review'] is True
    assert p['static_gesture_plausible'] is True
    rejected = set(p['source_rejected_frames'])
    assert not rejected & (set(ids) | set(hold))
    assert p['frames'] == ids and p['review_only_frames'] == hold
    assert set(p['holdout_roles']) == set(hold)
    assert all(r['fit_excluded'] and r['training_statistics_excluded'] and not r['physical_joint_ground_truth']
               for r in p['holdout_roles'].values())
    assert [r['frame'] for r in p['rows']] == ids
    assert Path(p['source_inputs']) == Path(p['source_root']) / 'data/inputs.json'
    assert Path(p['source_observations']) == Path(p['source_root']) / 'data/observations.npz'
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, path
    for r in p['rows']:
        assert r['wh'] == [2704, 1520]
        assert r['K_integer_centers'] == p['rows'][0]['K_integer_centers']
        assert np.array_equal(r['w2c'], np.eye(4))
        assert sha(r['image']) == r['source_sha256'] and sha(r['mask']) == r['mask_sha256']
    return p


def data():
    p = source_plan()
    # Execute the existing118-joint/score/raster mapping. Change only which
    # registered input IDs it indexes; use source MP4 pixels at full resolution.
    source = inspect.getsource(original.data)
    old = "selected=read(SRC/'selection.json')['training_frames']"
    new = "selected=read(AUDIT/'plan.json')['training_frames']"
    assert source.count(old) == 1
    source = source.replace(old, new)
    # Split before estimating q90: heldout scores and coordinates never enter
    # the fitting confidence statistics. The existing function is unchanged.
    identity = "    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]"
    insertion = identity + "\n    rows=[rows[i] for i in indices]\n    obs={k:(v[indices] if k in ('scores','boxes','flip_disagreement','keypoints') else v) for k,v in obs.items()}\n    indices=list(range(len(rows)))"
    assert source.count(identity) == 1
    source = source.replace(identity, insertion)
    def registered_confidence(scores, boxes, flip):
        assert len(scores) == 12
        confidence, q90 = original.relative_confidence(scores, boxes, flip)
        frozen = OUT / 'train-confidence-normalizer.npz'
        if frozen.exists():
            with np.load(frozen, allow_pickle=False) as archive:
                assert np.array_equal(archive['q90'], q90)
                assert archive['training_frame_ids'].astype(str).tolist() == p['training_frames']
                assert str(archive['source_observations_sha256']) == sha(p['source_observations'])
        return confidence, q90
    ns = dict(original.__dict__, SRC=Path(p['source_root']), AUDIT=AUDIT,
        relative_confidence=registered_confidence)
    exec(compile(source, '<same-algorithm-train12-statistics-football126-mapping>', 'exec'), ns)
    native_rows, target = ns['data']()
    assert target.shape == (12, 118, 3) and np.isfinite(target).all()
    assert [r['frame'] for r in native_rows] == p['training_frames']
    for native, row in zip(native_rows, p['rows']):
        for key in ('frame', 'image', 'mask', 'wh', 'K_integer_centers', 'source_sha256', 'mask_sha256', 'source_frame_index'):
            assert native[key] == row[key], (row['frame'], key)
        # Any original camera is provenance only in this body-view gauge;
        # body root/placement are fit afresh, never initialized from it.
        if 'source_camera_w2c_unused' in row:
            assert np.array_equal(row['source_camera_w2c_unused'], native['w2c'])
    return copy.deepcopy(p['rows']), target


def frozen_confidence(scores, boxes, flip_disagreement, q90):
    """The unchanged formula with previously fitted training-only statistics.

    This function never computes a quantile. Validation may use this function
    with the saved train12 q90, and must never refit statistics on its own rows.
    """
    scores = np.asarray(scores, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    flip = np.asarray(flip_disagreement, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    assert scores.ndim == 2 and boxes.shape == (len(scores), 4) and flip.shape == scores.shape
    assert q90.shape == (scores.shape[1],) and np.isfinite(q90).all() and (q90 >= 1.e-6).all()
    assert all(np.isfinite(x).all() for x in (scores, boxes, flip))
    height = boxes[:, 3] - boxes[:, 1]
    assert (height > 0).all() and (scores >= 0).all() and (flip >= 0).all()
    return (np.clip(scores / q90, 0, 1) * np.exp(-.25 * (flip / (.012 * height[:, None])) ** 2)).astype(np.float32)


def confidence_audit(p):
    with np.load(p['source_observations'], allow_pickle=False) as archive:
        obs = dict(archive)
    ids = obs['frame_ids'].astype(str).tolist()
    assert len(ids) == 59
    train = [ids.index(f) for f in p['training_frames']]
    hold = [ids.index(f) for f in p['heldout_frames']]
    assert len(train) == 12 and len(hold) == 4 and not set(train) & set(hold)
    conf, q90 = original.relative_confidence(obs['scores'][train], obs['boxes'][train], obs['flip_disagreement'][train])
    replay = frozen_confidence(obs['scores'][train], obs['boxes'][train], obs['flip_disagreement'][train], q90)
    assert np.array_equal(conf, replay)
    # A read-only equivalence check for the full_source59-row implementation;
    # this historical statistic is never supplied to the new fitter.
    old_conf, old_q90 = original.relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    assert np.array_equal(old_conf, frozen_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'], old_q90))
    frozen_before = q90.copy()
    hold_conf = frozen_confidence(obs['scores'][hold], obs['boxes'][hold], obs['flip_disagreement'][hold], q90)
    assert np.array_equal(q90, frozen_before) and hold_conf.shape == (4, 133)
    assert q90.shape == (133,) and conf.shape == (12, 133)
    return q90, dict(algorithm='Unmodified relative_confidence; select12 before fitting its q90 statistic.',
        quantile=.9, score_epsilon=1.e-6, flip_coefficient=.25, relative_height_scale=.012,
        train_rows=12, heldout_rows=4, scores_per_row=133,
        train_frozen_formula_exact=True, full_source59_frozen_formula_exact=True,
        heldout_evaluation_uses_frozen_train12_q90=True, heldout_q90_refit=False,
        heldout_statistics_or_coordinates_used_for_fitting=False,
        full59_vs_train12_q90_max_abs=float(np.max(np.abs(old_q90 - q90))),
        full59_vs_train12_training_confidence_max_abs=float(np.max(np.abs(old_conf[train] - conf))),
        full59_q90_diagnostic_only=True, formula_or_threshold_changed=False,
        source_observations_sha256=sha(p['source_observations']))


def hand_config(root):
    baseline = original.entry.load_official_config(root)
    candidate = hand_control.candidate_config(baseline)
    assert candidate['use_hands'] is True
    assert [k for k in baseline if baseline[k] != candidate[k]] == ['hand_joints_weights']
    config_reads.append(dict(baseline=baseline['hand_joints_weights'], candidate=candidate['hand_joints_weights']))
    return candidate


def make_entry(n, roots):
    method, delta = original.adapter.make_entry(n, roots)
    method.__globals__['load_official_config'] = hand_config
    return method, delta


runner = winterhat.runner
assert 'beta=np.zeros((1,10),dtype=np.float32)' in runner
assert 'reference-fit/winner-camera-space.npz' not in runner
assert 'freeze_betas=False' in runner
assert 'make_models(len(p["frames"]))' in runner
scope = prior.shared.scope.copy()
scope.update(OUT=OUT, data=data,
    adapter=SimpleNamespace(**dict(original.adapter.__dict__, make_entry=make_entry)))
exec(compile(runner, '<same-official-neutral-shared-football126-runner>', 'exec'), scope)


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists()
    p = source_plan()
    rows, target = data()
    roots = original.root_initialization(rows, p['reference_frame'])
    assert np.allclose(roots, np.tile([0., np.pi, 0.], (12, 1)), rtol=0, atol=1.e-6)
    baseline = original.entry.load_official_config(original.OFFICIAL)
    candidate = hand_control.candidate_config(baseline)
    assert baseline['maxiters'] == 30 and candidate['hand_joints_weights'] == [0.] * 5
    paths = [Path(__file__), Path(winterhat.__file__), Path(prior.__file__), Path(prior.shared.__file__),
        Path(original.__file__), Path(original.adapter.__file__), Path(original.entry.__file__), HAND_CONTROL,
        original.H / 'smplify_score_contract123.py', original.H / 'smplify_observation_validity124.py',
        AUDIT / 'plan.json', Path(p['source_inputs']), Path(p['source_observations']),
        original.ASSETS / 'smplx/SMPLX_NEUTRAL.npz']
    paths += [original.OFFICIAL / path for path in original.entry.PINNED]
    paths += [Path(path) for path in p['source_hashes']]
    for row in rows:
        paths.extend([Path(row['image']), Path(row['mask'])])
    import smplx
    paths.extend([Path(smplx.__file__).parent / 'body_models.py', Path(smplx.__file__).parent / 'lbs.py'])
    OUT.mkdir()
    q90, confidence_receipt = confidence_audit(p)
    np.savez_compressed(OUT / 'train-confidence-normalizer.npz', q90=q90,
        training_frame_ids=np.asarray(p['training_frames']),
        source_observations_sha256=np.asarray(sha(p['source_observations'])))
    confidence_receipt.update(updated=now(), training_frames=p['training_frames'], heldout_frames=p['heldout_frames'],
        frozen_statistics_sha256=sha(OUT / 'train-confidence-normalizer.npz'))
    save(OUT / 'confidence-input-strategy.json', confidence_receipt)
    selection = dict(updated=now(), subject='football', training_frames=p['training_frames'],
        reference_frame=p['reference_frame'], review_only_frames=p['heldout_frames'],
        source_rejected_frames=list(p['source_rejected_frames']), input_audit=str(AUDIT / 'plan.json'),
        user_rule=dict(half_circle_target=12, full_circle_target=24),
        selection_before_fit=True, historical_validation_reused=False,
        independently_unseen_video_validation=False, quality_accepted=False)
    save(OUT / 'selection.json', selection)
    review = dict(updated=now(), actual_visual_review=p['actual_source_visual_review'],
        static_gesture_plausible=p['static_gesture_plausible'], source_audit=str(AUDIT / 'plan.json'),
        source_audit_sha256=sha(AUDIT / 'plan.json'), retained_input_frames=p['training_frames'],
        rejected_input_frames=list(p['source_rejected_frames']), source_review_reused=True,
        newly_generated_fit_or_render_review=False, physical_landmark_truth=False, quality_accepted=False)
    save(OUT / 'source-review.json', review)
    paths += [OUT / 'selection.json', OUT / 'source-review.json', OUT / 'train-confidence-normalizer.npz', OUT / 'confidence-input-strategy.json']
    plan = dict(updated=now(), state='registered_source_reviewed', subject='football', round=126, record_id='011',
        frames=p['training_frames'], rows=rows, reference_frame=p['reference_frame'], reference_for_root_seed=p['reference_frame'],
        review_only_excluded=p['heldout_frames'], source_rejected_frames=list(p['source_rejected_frames']),
        holdout_roles=copy.deepcopy(p['holdout_roles']),
        source_selection=str(AUDIT / 'plan.json'), source_split_newly_authorized=True,
        historical_validation_reused=False, new_four_are_prospective_fit_holdout=True,
        heldout_coordinates_used_in_fit=False, heldout_statistics_used_in_fit=False,
        strict_unseen_material_or_video_validation=False,
        confidence_normalization='Same relative_confidence formula and parameters, now select12 before estimating q90. Holdout4 coordinates and score statistics do not enter fitting. Future validation applies saved train12 q90 without refitting.',
        input_strategy_change_registered_before_fit='Necessary new train/validation split boundary: q90 estimated on selected12 rather than full_source59; no solver/loss/threshold change.',
        confidence_input_strategy=str(OUT / 'confidence-input-strategy.json'),
        independently_unseen_video_validation=False,
        hypothesis='Apply the same neutral shared-shape selected-view and current Gaussian pipeline to the new football person, using a source-only12+4 split.',
        user_basis='New people only; half circle12/full circle24. Freeze the new-person source-only split before its first fit.',
        algorithm='Existing shared10D beta and32D VPoser; per-view root/placement/face/fingers; original5stages/max30/LBFGS and priors, uniform original hand2D-data zero option.',
        initialization='Official neutral zero10D beta, zero VPoser, standard front body roots in all12. The designated source/reference frame is not a beta/pose initializer. No old football optimized values are loaded.',
        reference_frame_role='Names the fixed image/reference pose for packaging; all body-view roots and placements remain learnable.',
        identity_camera_warning='Identity w2c is a body-view fitting gauge, not physical scene calibration. Original camera provenance is retained but not used as pose initialization.',
        source_input_root=p['source_root'], original_inputs_K_and_priors_preserved=True,
        shared_shape_can_update=True, old_optimized_pose_used=False, old_individual_shape_used=False,
        original_zero_body_latent=True, standard_front_root_all_views=True,
        hand_observation_policy='All12/both hands use original hand_joints_weights=[0,0,0,0,0]; BODY25 wrists and hand pose priors remain unchanged.',
        hand_data_control=dict(baseline=baseline['hand_joints_weights'], candidate=candidate['hand_joints_weights'], same_for_all_frames=True),
        static_gesture_assumption=True, physical_camera_calibration=False,
        causal_scope='Combined new roster, fresh neutral shared shape and current pipeline; no isolated frame-count claim or reuse of historical validation as independent evidence.',
        budget=dict(candidates=1, selected_views=12, GPU_seconds=600, new_gaussian_steps=0),
        stop='One fit and actual native/side review. No roster/weight/threshold sweep; no Gaussian launch before review.',
        quality_accepted=False, gaussian_admitted=False,
        source_hashes={str(path): sha(path) for path in paths})
    save(OUT / 'plan.json', plan)
    _, delta = original.adapter.source(original.OFFICIAL)
    (OUT / 'upstream-interface.diff').write_text(delta)
    _, delta = make_entry(12, roots)
    (OUT / 'entry-interface.diff').write_text(delta)
    (OUT / 'executed-runner.py.txt').write_text(runner)
    print('FOOTBALL_HALF12_NEUTRAL_FIT126_REGISTERED', p['training_frames'], p['reference_frame'], flush=True)


def preflight():
    source_plan()
    assert read(OUT / 'source-review.json')['actual_visual_review']
    proxy = SimpleNamespace(**dict(original.__dict__, SRC=Path(source_plan()['source_root'])))
    code = inspect.getsource(prior.shared.preflight)
    old = "    with np.load(original.SRC/'reference-fit/winner-camera-space.npz') as z:seed=z['betas'].copy()"
    assert code.count(old) == 1
    code = code.replace(old, "    seed=np.zeros((1,10),dtype=np.float32)")
    assert code.count('for n in (1,8):') == code.count('if n==8:') == 1
    code = code.replace('for n in (1,8):', 'for n in (1,len(plan["frames"])):')
    code = code.replace('if n==8:', 'if n==len(plan["frames"]):')
    code = code.replace('original_shape_prior_not_divided_by8', 'original_shape_prior_not_divided_byN')
    old = "        gradient=torch.autograd.grad(objective,body.betas,retain_graph=True)[0]"
    assert code.count(old) == 1
    code = code.replace(old, old + "\n        root_gradient,latent_gradient=torch.autograd.grad(objective,(body.global_orient,latent),retain_graph=True)\n        assert root_gradient.shape==(n,3) and latent_gradient.shape==(1,32)\n        assert torch.isfinite(root_gradient).all() and torch.isfinite(latent_gradient).all()\n        assert (root_gradient.abs().sum(1)>0).all() and latent_gradient.abs().max()>0")
    old = '    torch.set_num_threads(4)'
    assert code.count(old) == 1
    code = code.replace(old, old + "\n    oracle=original.adapter.verify(original.OFFICIAL)\n    assert all(oracle['checks'].values())")
    code = code.replace("    print('SHARED_SHAPE_GPU_PREFLIGHT_PASSED',checks,flush=True)",
        "    saved=read(OUT/'preflight.json');saved['original_N1_loss_gradient_oracle']=oracle\n    save(OUT/'preflight.json',saved)\n    print('FOOTBALL_HALF12_GPU_PREFLIGHT_PASSED',checks,flush=True)")
    assert 'reference-fit/winner-camera-space.npz' not in code
    ns = dict(prior.shared.__dict__, OUT=OUT, original=proxy)
    exec(compile(code, '<existing-neutral-shared-preflight-football126>', 'exec'), ns)
    ns['preflight']()
    baseline = original.entry.load_official_config(original.OFFICIAL)
    candidate = hand_control.candidate_config(baseline)
    source = (original.OFFICIAL / 'smplifyx/fit_single_frame.py').read_text()
    assert "joint_weights[:, 25:67] = curr_weights['hand_weight']" in source
    result = read(OUT / 'preflight.json')
    result.update(original_hand_weight_option_only=True, hand_data_schedule=candidate['hand_joints_weights'],
        body_wrist_indices_unchanged=[4, 7], mapping_length_unchanged=118, hand_pose_prior_retained=True,
        configuration_verified_no_additional_optimizer_updates=True,
        old_individual_shape_used=False, old_optimized_pose_used=False, source_reference_role='label_only',
        optimizer_steps=0, heldout_excluded=source_plan()['heldout_frames'])
    save(OUT / 'preflight.json', result)


def run():
    source_plan()
    assert not config_reads
    scope['run']()
    assert len(config_reads) == 1 and config_reads[0]['candidate'] == [0.] * 5
    result = read(OUT / 'result.json')
    with np.load(OUT / 'selected-fit.npz', allow_pickle=False) as archive:
        beta = archive['betas'].copy()
        assert archive['frame_ids'].astype(str).tolist() == source_plan()['training_frames']
        assert archive['body_pose_axis_angle'].shape == (1, 63)
        assert archive['pose_embedding'].shape == (1, 32)
        assert archive['vertices_camera'].shape == (12, 10475, 3)
        assert all(archive[k].shape[0] == 12 for k in ('global_orient', 'expression', 'left_hand_pose', 'right_hand_pose'))
    assert beta.shape == (1, 10) and np.isfinite(beta).all()
    result.update(beta_before=np.zeros((1, 10)).tolist(), beta_after=beta.tolist(),
        beta_delta_l2=float(np.linalg.norm(beta)), shared_beta_parameter_count=10,
        shared_shape_updated=True, freeze_betas=False, old_individual_shape_used=False,
        old_optimized_pose_used=False, source_camera_identity_placeholder=True,
        frames_count=12, reference_frame=source_plan()['reference_frame'],
        hand_data_control=dict(config_reads[0], same_for_all_frames=True,
            body_wrists_and_hand_pose_prior_retained=True, existing_control=str(HAND_CONTROL), source_sha256=sha(HAND_CONTROL)),
        actual_root_seed='Standard front rotation for all12; source/reference frame names output identity only.',
        historical_validation_reused=False, heldout_excluded=source_plan()['heldout_frames'],
        quality_accepted=False, gaussian_admitted=False)
    save(OUT / 'result.json', result)


def launch():
    source_plan()
    assert read(OUT / 'source-review.json')['actual_visual_review']
    code = inspect.getsource(prior.launch)
    ns = dict(prior.__dict__, OUT=OUT, __file__=__file__)
    exec(compile(code, '<same-single-process-football126-fit-launch>', 'exec'), ns)
    ns['launch']()


def audit():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not OUT.exists()
    p = source_plan()
    rows, target = data()
    q90, confidence_receipt = confidence_audit(p)
    old_mapping = inspect.getsource(original.data).replace(
        "selected=read(SRC/'selection.json')['training_frames']",
        "selected=read(AUDIT/'plan.json')['training_frames']")
    old_ns = dict(original.__dict__, SRC=Path(p['source_root']), AUDIT=AUDIT)
    exec(compile(old_mapping, '<read-only-full59-normalizer-mapping-oracle>', 'exec'), old_ns)
    old_rows, old_target = old_ns['data']()
    assert [r['frame'] for r in old_rows] == p['training_frames']
    assert np.array_equal(target[:, :, :2], old_target[:, :, :2])
    roots = original.root_initialization(rows, p['reference_frame'])
    assert np.allclose(roots, np.tile([0., np.pi, 0.], (12, 1)), rtol=0, atol=1.e-6)
    baseline = original.entry.load_official_config(original.OFFICIAL)
    candidate = hand_control.candidate_config(baseline)
    assert baseline['maxiters'] == 30 and candidate['hand_joints_weights'] == [0.] * 5
    assert [k for k in baseline if baseline[k] != candidate[k]] == ['hand_joints_weights']
    method, entry_delta = make_entry(12, roots)
    _, upstream_delta = original.adapter.source(original.OFFICIAL)
    compile(runner, '<football126-runner-cpu-static-check>', 'exec')
    compile(Path(__file__).read_text(), str(Path(__file__)), 'exec')
    assert not torch.cuda.is_initialized() and not OUT.exists()
    result = dict(updated=now(), mode='CPU_only_preparation', passed=True,
        helper_sha256=sha(__file__), input_plan_sha256=sha(AUDIT / 'plan.json'),
        dynamic_training_frames=p['training_frames'], heldout_frames=p['heldout_frames'],
        holdout_roles=copy.deepcopy(p['holdout_roles']),
        reference_frame=p['reference_frame'], original_mapping_shape=list(target.shape),
        original_native_xy_mapping_exact=True,
        native_K_and_pixels_preserved=True, official_five_stages_max30=True,
        shared_shape_seed=np.zeros((1, 10)).tolist(), shared_shape_trainable=True,
        zero_vposer_preserved=True, all_front_root_seed=roots.tolist(),
        old_beta_pose_geometry_initialization_read=False,
        source_only_split_frozen_before_fit=True, historical_validation_reused=False,
        heldout_coordinates_used_in_fit=False, heldout_statistics_used_in_fit=False,
        train12_frozen_q90=q90.tolist(), confidence_input_strategy=confidence_receipt,
        original_hand_option_only=True, baseline_hand_weights=baseline['hand_joints_weights'],
        candidate_hand_weights=candidate['hand_joints_weights'], body_wrists_and_hand_prior_retained=True,
        runner_sha256=__import__('hashlib').sha256(runner.encode()).hexdigest(),
        upstream_interface_diff=upstream_delta, entry_interface_diff=entry_delta,
        cuda_initialized=False, fit_directory_created=False, fitting_started=False,
        GPU_preflight_executed=False, quality_accepted=False)
    save(CPU_AUDIT, result)
    print('FOOTBALL_HALF12_FIT126_CPU_PREPARATION_PASSED', p['training_frames'], p['reference_frame'], flush=True)


if __name__ == '__main__':
    {'audit': audit, 'prepare': prepare, 'preflight': preflight, 'run': run, 'launch': launch}[sys.argv[1]]()

"""Fresh officewoman24 shared fit; reuse the reviewed neutral winterhat interfaces.

The new24+heldout source-only split must be frozen and reviewed before prepare. No old
officewoman shape, fitted pose, geometry or Gaussian is read as initialization. The
fitter retains the existing uniform zero hand2D-data schedule, as in winterhat;
wrists/priors remain. The new split fits confidence statistics on training24
only, with the existing formula and constants, before any new fitting. Confirmed
unobserved face/head and knee/foot-group scores are zeroed on a COPY before this unchanged q90;
unknown source visibility halts. Original coordinates and raw archives are kept.
"""
import copy
import difflib
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import run_winterhat_half8_fit124 as winterhat
from run_official_gaussians123 import read, save, sha, now

prior = winterhat.prior
original = winterhat.original
OUT = original.ROOT / 'officewoman-circle24-pose127'
AUDIT = original.ROOT / 'officewoman-circle24-input127'
CPU_AUDIT = Path(__file__).with_name('run_officewoman_circle24_fit127_cpu_audit.json')
HAND_CONTROL = winterhat.HAND_CONTROL
hand_control = winterhat.hand_control
config_reads = []
N = 24
SUBJECT = 'officewoman'
NATIVE_WH = [1520, 2704]
FACE_SOURCE = np.r_[np.arange(5), np.arange(40, 91)]
FACE_TARGET = np.r_[0, np.arange(15, 19), np.arange(67, 118)]


GROUP_SOURCE = dict(left_knee=[13],right_knee=[14],left_foot=[15,17,18,19],right_foot=[16,20,21,22])
GROUP_TARGET = dict(left_knee=[13],right_knee=[10],left_foot=[14,19,20,21],right_foot=[11,22,23,24])
MASKABLE_SOURCE = np.unique(np.concatenate([FACE_SOURCE]+[np.asarray(x) for x in GROUP_SOURCE.values()]))
MASKABLE_TARGET = np.unique(np.concatenate([FACE_TARGET]+[np.asarray(x) for x in GROUP_TARGET.values()]))


def visibility_score_copy(scores, face_observed, joint_groups):
    """Source visibility only; never move XY or edit the raw NPZ."""
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != 133 or len(face_observed) != len(values) or len(joint_groups) != len(values):
        raise ValueError('Expected source133 scores and reviewed visibility per row.')
    if any(type(flag) is not bool for flag in face_observed):
        raise ValueError('Unknown source face visibility; review is required before fitting.')
    if any(set(groups) != set(GROUP_SOURCE) or any(type(groups[k]) is not bool for k in GROUP_SOURCE) for groups in joint_groups):
        raise ValueError('Unknown source joint-group support; review is required before fitting.')
    result = values.copy()
    for i,(face,groups) in enumerate(zip(face_observed,joint_groups)):
        if not face: result[i,FACE_SOURCE] = 0
        for name,indices in GROUP_SOURCE.items():
            if not groups[name]: result[i,indices] = 0
    return result


def visibility_record(p):
    frames = p['training_frames'] + p['heldout_frames']
    flags = p['face_visibility']
    assert set(flags) == set(frames) and all(type(flags[f]) is bool for f in frames)
    groups = p['joint_group_visibility']
    assert set(groups) == set(frames)
    assert all(set(groups[f]) == set(GROUP_SOURCE) and all(type(groups[f][k]) is bool for k in GROUP_SOURCE) for f in frames)
    assert all(row['face_region_is_observed'] is flags[row['frame']] for row in p['rows'])
    assert all(row['source_joint_group_supported'] == groups[row['frame']] for row in p['rows'])
    reference = next(row for row in p['rows'] if row['frame'] == p['reference_frame'])
    assert flags[p['reference_frame']] is True and reference['visible_eyes_open'] is True
    return dict(subject=SUBJECT,training_frames=p['training_frames'],heldout_frames=p['heldout_frames'],
        face_visibility={f:flags[f] for f in frames},joint_group_visibility={f:groups[f] for f in frames},
        joint_group_source_indices=GROUP_SOURCE,joint_group_target_indices=GROUP_TARGET,source_score_indices=FACE_SOURCE.tolist(),
        mapped_target_indices=FACE_TARGET.tolist(),source_only_actual_review=True,
        unknown_policy='halt', raw_XY_and_archives_unchanged=True,
        prior_amplitudes_and_hand_schedule_unchanged=True,
        policy='On a scores COPY, confirmed source-unobserved head/face or knee/foot-group entries become zero BEFORE unchanged q90. Zero rows remain in its unchanged quantile formula; these are heuristic weights, not calibrated probabilities.')


def visibility_sha256(p):
    return hashlib.sha256(json.dumps(visibility_record(p),sort_keys=True,separators=(',',':')).encode()).hexdigest()


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
    assert p['subject'] == SUBJECT
    assert p['source_root'] == '/opt/gs-slam-lab/shuolee-cross-subject-127/officewoman'
    ids = p['training_frames']
    hold = p['heldout_frames']
    assert len(ids) == len(set(ids)) == N and ids == sorted(ids)
    assert len(hold) == len(set(hold)) and len(hold) > 0 and not set(ids) & set(hold)
    slots = p['holdout_reservation']
    assert slots and all(slot['candidate_order'] and len(set(slot['candidate_order'])) == len(slot['candidate_order']) for slot in slots)
    reserved = {frame for slot in slots for frame in slot['candidate_order']}
    assert len(reserved) >= len(hold) and set(hold) <= reserved and not set(ids) & reserved
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
    visibility_record(p)
    assert Path(p['source_inputs']) == Path(p['source_root']) / 'data/inputs.json'
    assert Path(p['source_observations']) == Path(p['source_root']) / 'data/observations.npz'
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, path
    for r in p['rows']:
        assert r['wh'] == NATIVE_WH
        assert np.asarray(r['K_integer_centers']).shape == (3, 3)
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
        assert len(scores) == N
        observed = [p['face_visibility'][f] for f in p['training_frames']]
        confidence, q90 = original.relative_confidence(visibility_score_copy(scores, observed, [p['joint_group_visibility'][f] for f in p['training_frames']]), boxes, flip)
        frozen = OUT / 'train-confidence-normalizer.npz'
        if frozen.exists():
            with np.load(frozen, allow_pickle=False) as archive:
                assert np.array_equal(archive['q90'], q90)
                assert archive['training_frame_ids'].astype(str).tolist() == p['training_frames']
                assert str(archive['source_observations_sha256']) == sha(p['source_observations'])
                assert str(archive['source_observation_visibility_sha256']) == visibility_sha256(p)
        return confidence, q90
    ns = dict(original.__dict__, SRC=Path(p['source_root']), AUDIT=AUDIT,
        relative_confidence=registered_confidence)
    exec(compile(source, '<same-algorithm-trainN-statistics-officewoman127-mapping>', 'exec'), ns)
    native_rows, target = ns['data']()
    assert target.shape == (N, 118, 3) and np.isfinite(target).all()
    # Compare against the same training-only mapping without visibility gating;
    # this is an oracle only, never an alternative fit input or frame selector.
    unmasked_ns = dict(ns, relative_confidence=original.relative_confidence)
    exec(compile(source, '<read-only-unmasked-trainN-mapping-oracle>', 'exec'), unmasked_ns)
    _, unmasked_target = unmasked_ns['data']()
    assert np.array_equal(target[:,:,:2], unmasked_target[:,:,:2])
    nonface = np.setdiff1d(np.arange(118), MASKABLE_TARGET)
    assert np.array_equal(target[:,nonface,2], unmasked_target[:,nonface,2])
    assert np.array_equal(target[:,[4,7]], unmasked_target[:,[4,7]])
    hidden = [i for i,f in enumerate(p['training_frames']) if not p['face_visibility'][f]]
    assert np.all(target[np.ix_(hidden, FACE_TARGET, [2])] == 0)
    for i,f in enumerate(p['training_frames']):
        for name,indices in GROUP_TARGET.items():
            if not p['joint_group_visibility'][f][name]: assert np.all(target[i,indices,2] == 0)

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
    with the saved trainN q90, and must never refit statistics on its own rows.
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



def synthetic_visibility_checks():
    """Bounded CPU contract, with the unchanged actual118-point mapper."""
    generator = np.random.default_rng(127)
    count = N + 8
    frames = [f'frame_{i:06d}' for i in range(count)]
    rows = [dict(frame=f, wh=NATIVE_WH) for f in frames]
    raw = dict(frame_ids=np.asarray(frames), scores=generator.uniform(.2, 2., (count,133)).astype(np.float32),
        boxes=np.tile([0.,0.,1000.,2000.],(count,1)),
        flip_disagreement=generator.uniform(0.,.2,(count,133)).astype(np.float32),
        keypoints=generator.uniform([10.,10.],[1400.,2500.],(count,133,2)).astype(np.float32))
    untouched = {k:v.copy() for k,v in raw.items()}
    observed = [i%3 != 0 for i in range(N)]
    source = inspect.getsource(original.data)
    identity = "    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]"
    assert source.count(identity) == 1
    source = source.replace(identity, identity + "\n    rows=[rows[i] for i in indices]\n    obs={k:(v[indices] if k in ('scores','boxes','flip_disagreement','keypoints') else v) for k,v in obs.items()}\n    indices=list(range(len(rows)))")
    class Archive(dict):
        def __enter__(self): return self
        def __exit__(self,*args): return False
    def mapped(flags, groups, archive):
        captured = {}
        def confidence(scores, boxes, flip):
            assert len(scores) == N
            result, q90 = original.relative_confidence(visibility_score_copy(scores, flags, groups), boxes, flip)
            captured['q90'] = q90.copy()
            return result, q90
        fake_np = SimpleNamespace(**dict(np.__dict__,load=lambda *args,**kwargs:Archive(archive)))
        namespace = dict(original.__dict__,np=fake_np,SRC=Path('/synthetic-no-io'),
            read=lambda path:dict(records=rows) if path.name=='inputs.json' else dict(training_frames=frames[:N]),
            relative_confidence=confidence)
        exec(compile(source,'<synthetic-actual-original118-mapping-no-io>','exec'),namespace)
        result_rows,target = namespace['data']()
        assert [r['frame'] for r in result_rows] == frames[:N]
        return target,captured['q90']
    supported = [dict.fromkeys(GROUP_SOURCE,True) for _ in range(N)]
    groups = [{name:(i+slot)%4 != 0 for slot,name in enumerate(GROUP_SOURCE)} for i in range(N)]
    baseline,baseline_q90 = mapped([True]*N, supported, raw)
    masked,masked_q90 = mapped(observed, groups, raw)
    hidden = np.flatnonzero(np.logical_not(observed))
    nonface = np.setdiff1d(np.arange(118), MASKABLE_TARGET)
    perturbed = {k:v.copy() for k,v in raw.items()}
    perturbed['scores'][N:] *= 10000
    perturbed['keypoints'][N:] += 100000
    heldout_changed,heldout_q90 = mapped(observed,groups,perturbed)
    unknown_rejected = False
    try: visibility_score_copy(raw['scores'][:N], [None]+observed[1:], groups)
    except ValueError: unknown_rejected = True
    unknown_group_rejected = False
    missing_group = [dict(g) for g in groups];missing_group[0]['left_foot']=None
    try: visibility_score_copy(raw['scores'][:N], observed, missing_group)
    except ValueError: unknown_group_rejected = True
    all_visible_copy = visibility_score_copy(raw['scores'][:N], [True]*N, supported)
    groups_zero = all(bool(np.all(masked[i,GROUP_TARGET[name],2] == 0)) for i in range(N) for name in GROUP_SOURCE if not groups[i][name])
    checks = dict(raw_scores_boxes_flip_XY_frameids_unchanged=all(np.array_equal(raw[k],untouched[k]) for k in raw),
        copy_has_independent_memory=not np.shares_memory(all_visible_copy,raw['scores']),
        all_visible_scores_exact=np.array_equal(all_visible_copy,raw['scores'][:N]),
        unknown_face_visibility_rejected=unknown_rejected,
        unknown_limb_group_visibility_rejected=unknown_group_rejected,
        all4_group_target_confidences_zero_where_unsupported=groups_zero,
        mapped_all_XY_exact=np.array_equal(masked[:,:,:2],baseline[:,:,:2]),
        mapped_unobserved56_headface_weights_zero=bool(np.all(masked[np.ix_(hidden,FACE_TARGET,[2])]==0)),
        mapped_nonmasked_weights_exact=np.array_equal(masked[:,nonface,2],baseline[:,nonface,2]),
        body_hips9_12_and_wrists4_7_exact=np.array_equal(masked[:,[4,7,9,12]],baseline[:,[4,7,9,12]]),
        unmasked_masked_nonmasked_q90_exact=np.array_equal(masked_q90[np.setdiff1d(np.arange(133),MASKABLE_SOURCE)],baseline_q90[np.setdiff1d(np.arange(133),MASKABLE_SOURCE)]),
        heldout_scores_and_XY_cannot_change_train_target_or_q90=np.array_equal(masked,heldout_changed) and np.array_equal(masked_q90,heldout_q90),
        group_mapping_dictionary_exact=GROUP_SOURCE==dict(left_knee=[13],right_knee=[14],left_foot=[15,17,18,19],right_foot=[16,20,21,22]) and GROUP_TARGET==dict(left_knee=[13],right_knee=[10],left_foot=[14,19,20,21],right_foot=[11,22,23,24]),
        target118_mapping_shapes=masked.shape==(N,118,3))
    assert all(checks.values()),checks
    return dict(CPU_only=True,synthetic_data_only=True,actual_original_mapping_executed=True,
        source_rows=count,train_rows=N,heldout_rows=8,seed=127,checks=checks,
        real_source_data_audited=False,GPU_used=False,optimizer_updates=0)


def confidence_audit(p):
    with np.load(p['source_observations'], allow_pickle=False) as archive:
        obs = dict(archive)
    ids = obs['frame_ids'].astype(str).tolist()
    assert ids == [r['frame'] for r in read(p['source_inputs'])['records']]
    train = [ids.index(f) for f in p['training_frames']]
    hold = [ids.index(f) for f in p['heldout_frames']]
    assert len(train) == N and len(hold) == len(p['heldout_frames']) and not set(train) & set(hold)
    scores = visibility_score_copy(obs['scores'][train], [p['face_visibility'][f] for f in p['training_frames']], [p['joint_group_visibility'][f] for f in p['training_frames']])
    conf, q90 = original.relative_confidence(scores, obs['boxes'][train], obs['flip_disagreement'][train])
    unmasked_conf, unmasked_q90 = original.relative_confidence(obs['scores'][train], obs['boxes'][train], obs['flip_disagreement'][train])
    replay = frozen_confidence(scores, obs['boxes'][train], obs['flip_disagreement'][train], q90)
    assert np.array_equal(conf, replay)
    # A read-only equivalence check for the full_sourceN-row implementation;
    # this historical statistic is never supplied to the new fitter.
    old_conf, old_q90 = original.relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    assert np.array_equal(old_conf, frozen_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'], old_q90))
    frozen_before = q90.copy()
    hold_scores = visibility_score_copy(obs['scores'][hold], [p['face_visibility'][f] for f in p['heldout_frames']], [p['joint_group_visibility'][f] for f in p['heldout_frames']])
    hold_conf = frozen_confidence(hold_scores, obs['boxes'][hold], obs['flip_disagreement'][hold], q90)
    assert np.array_equal(q90, frozen_before) and hold_conf.shape == (len(hold), 133)
    assert q90.shape == (133,) and conf.shape == (N, 133)
    return q90, dict(algorithm='Unmodified relative_confidence; select24 before fitting its q90 statistic.',
        quantile=.9, score_epsilon=1.e-6, flip_coefficient=.25, relative_height_scale=.012,
        train_rows=N, heldout_rows=len(hold), scores_per_row=133,
        train_frozen_formula_exact=True, full_sourceN_frozen_formula_exact=True,
        heldout_evaluation_uses_frozen_trainN_q90=True, heldout_q90_refit=False,
        heldout_statistics_or_coordinates_used_for_fitting=False,
        fullsource_vs_trainN_q90_max_abs=float(np.max(np.abs(old_q90 - q90))),
        fullsource_vs_trainN_training_confidence_max_abs=float(np.max(np.abs(old_conf[train] - conf))),
        fullsource_q90_diagnostic_only=True, formula_or_threshold_changed=False,
        source_observations_sha256=sha(p['source_observations']),
        source_observation_visibility_sha256=visibility_sha256(p),
        source_observation_visibility_policy=visibility_record(p),
        masked_train_q90=q90.tolist(), unmasked_train_q90_diagnostic_only=unmasked_q90.tolist(),
        unmasked_masked_q90_max_abs=float(np.max(np.abs(unmasked_q90-q90))),
        q90_zero_rows_retained_in_unchanged_formula=True, calibrated_probabilities=False,
        source_visibility_not_frame_count_only_change=True)


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
exec(compile(runner, '<same-official-neutral-shared-officewoman127-runner>', 'exec'), scope)


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists()
    p = source_plan()
    rows, target = data()
    roots = original.root_initialization(rows, p['reference_frame'])
    assert np.allclose(roots, np.tile([0., np.pi, 0.], (N, 1)), rtol=0, atol=1.e-6)
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
    save(OUT / 'source-observation-visibility.json', dict(updated=now(), source_input_plan_sha256=sha(AUDIT / 'plan.json'),
        policy_sha256=visibility_sha256(p), **visibility_record(p)))
    q90, confidence_receipt = confidence_audit(p)
    np.savez_compressed(OUT / 'train-confidence-normalizer.npz', q90=q90,
        training_frame_ids=np.asarray(p['training_frames']),
        source_observations_sha256=np.asarray(sha(p['source_observations'])),
        source_observation_visibility_sha256=np.asarray(visibility_sha256(p)))
    confidence_receipt.update(updated=now(), training_frames=p['training_frames'], heldout_frames=p['heldout_frames'],
        frozen_statistics_sha256=sha(OUT / 'train-confidence-normalizer.npz'))
    save(OUT / 'confidence-input-strategy.json', confidence_receipt)
    selection = dict(updated=now(), subject=SUBJECT, training_frames=p['training_frames'],
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
    paths += [OUT / 'selection.json', OUT / 'source-review.json', OUT / 'train-confidence-normalizer.npz', OUT / 'confidence-input-strategy.json', OUT / 'source-observation-visibility.json']
    plan = dict(updated=now(), state='registered_source_reviewed', subject=SUBJECT, round=127, record_id='019',
        frames=p['training_frames'], rows=rows, reference_frame=p['reference_frame'], reference_for_root_seed=p['reference_frame'],
        review_only_excluded=p['heldout_frames'], source_rejected_frames=list(p['source_rejected_frames']),
        holdout_roles=copy.deepcopy(p['holdout_roles']),
        source_selection=str(AUDIT / 'plan.json'), source_split_newly_authorized=True,
        source_observation_visibility=str(OUT / 'source-observation-visibility.json'),
        source_observation_visibility_policy_sha256=visibility_sha256(p),
        observation_adapter_change='Source-reviewed unobserved head/face and knee/foot-group confidence zero on scores copy before unchanged training-only q90; XY/rawNPZ/priors unchanged. Not a frame-count-only ablation.',
        historical_validation_reused=False, new_holdouts_are_prospective_fit_holdout=True,
        heldout_coordinates_used_in_fit=False, heldout_statistics_used_in_fit=False,
        strict_unseen_material_or_video_validation=False,
        confidence_normalization='Same relative_confidence formula and parameters, now select24 before estimating q90. Holdout coordinates and score statistics do not enter fitting. Future validation applies saved trainN q90 without refitting.',
        input_strategy_change_registered_before_fit='Necessary new train/validation split boundary: q90 estimated on selected24 rather than full_sourceN; source-reviewed invisible head/face and knee/foot-group score copy set0 before unchanged q90; no solver/loss/threshold change.',
        confidence_input_strategy=str(OUT / 'confidence-input-strategy.json'),
        independently_unseen_video_validation=False,
        hypothesis='Apply the same neutral shared-shape selected-view and current Gaussian pipeline to the new officewoman person, using a source-only24+heldout split.',
        user_basis='New people only; half circle12/full circle24. Freeze the new-person source-only split before its first fit.',
        algorithm='Existing shared10D beta and32D VPoser; per-view root/placement/face/fingers; original5stages/max30/LBFGS and priors, uniform original hand2D-data zero option.',
        initialization='Official neutral zero10D beta, zero VPoser, standard front body roots in all24. The designated source/reference frame is not a beta/pose initializer. No old officewoman optimized values are loaded.',
        reference_frame_role='Names the fixed image/reference pose for packaging; all body-view roots and placements remain learnable.',
        identity_camera_warning='Identity w2c is a body-view fitting gauge, not physical scene calibration. Original camera provenance is retained but not used as pose initialization.',
        source_input_root=p['source_root'], original_inputs_K_and_priors_preserved=True,
        shared_shape_can_update=True, old_optimized_pose_used=False, old_individual_shape_used=False,
        original_zero_body_latent=True, standard_front_root_all_views=True,
        hand_observation_policy='All24/both hands use original hand_joints_weights=[0,0,0,0,0]; BODY25 wrists and hand pose priors remain unchanged.',
        hand_data_control=dict(baseline=baseline['hand_joints_weights'], candidate=candidate['hand_joints_weights'], same_for_all_frames=True),
        static_gesture_assumption=True, physical_camera_calibration=False,
        causal_scope='Combined new full-circle roster, source-only face and knee/foot-group visibility gating, fresh neutral shared shape and current pipeline; no isolated frame-count claim or reuse of historical validation as independent evidence.',
        budget=dict(candidates=1, selected_views=N, GPU_seconds=600, new_gaussian_steps=0),
        stop='One fit and actual native/side review. No roster/weight/threshold sweep; no Gaussian launch before review.',
        quality_accepted=False, gaussian_admitted=False,
        source_hashes={str(path): sha(path) for path in paths})
    save(OUT / 'plan.json', plan)
    _, delta = original.adapter.source(original.OFFICIAL)
    (OUT / 'upstream-interface.diff').write_text(delta)
    _, delta = make_entry(N, roots)
    (OUT / 'entry-interface.diff').write_text(delta)
    (OUT / 'executed-runner.py.txt').write_text(runner)
    print('OFFICEWOMAN_CIRCLE24_NEUTRAL_FIT127_REGISTERED', p['training_frames'], p['reference_frame'], flush=True)


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
        "    saved=read(OUT/'preflight.json');saved['original_N1_loss_gradient_oracle']=oracle\n    save(OUT/'preflight.json',saved)\n    print('OFFICEWOMAN_CIRCLE24_GPU_PREFLIGHT_PASSED',checks,flush=True)")
    assert 'reference-fit/winner-camera-space.npz' not in code
    ns = dict(prior.shared.__dict__, OUT=OUT, original=proxy)
    exec(compile(code, '<existing-neutral-shared-preflight-officewoman127>', 'exec'), ns)
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
        assert archive['K_integer_centers'].shape == (3, 3)
        assert np.array_equal(archive['K_integer_centers'], np.asarray(source_plan()['rows'][0]['K_integer_centers']))
        assert archive['vertices_camera'].shape == (N, 10475, 3)
        assert all(archive[k].shape[0] == N for k in ('global_orient', 'expression', 'left_hand_pose', 'right_hand_pose'))
    assert beta.shape == (1, 10) and np.isfinite(beta).all()
    result.update(beta_before=np.zeros((1, 10)).tolist(), beta_after=beta.tolist(),
        beta_delta_l2=float(np.linalg.norm(beta)), shared_beta_parameter_count=10,
        shared_shape_updated=True, freeze_betas=False, old_individual_shape_used=False,
        old_optimized_pose_used=False, source_camera_identity_placeholder=True,
        frames_count=N, reference_frame=source_plan()['reference_frame'],
        hand_data_control=dict(config_reads[0], same_for_all_frames=True,
            body_wrists_and_hand_pose_prior_retained=True, existing_control=str(HAND_CONTROL), source_sha256=sha(HAND_CONTROL)),
        actual_root_seed='Standard front rotation for all24; source/reference frame names output identity only; front/back orientation can remain ambiguous until actual full-circle fit review.',
        historical_validation_reused=False, heldout_excluded=source_plan()['heldout_frames'],
        quality_accepted=False, gaussian_admitted=False)
    save(OUT / 'result.json', result)


def launch():
    source_plan()
    assert read(OUT / 'source-review.json')['actual_visual_review']
    code = inspect.getsource(prior.launch)
    ns = dict(prior.__dict__, OUT=OUT, __file__=__file__)
    exec(compile(code, '<same-single-process-officewoman127-fit-launch>', 'exec'), ns)
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
    exec(compile(old_mapping, '<read-only-fullsource-normalizer-mapping-oracle>', 'exec'), old_ns)
    old_rows, old_target = old_ns['data']()
    assert [r['frame'] for r in old_rows] == p['training_frames']
    assert np.array_equal(target[:, :, :2], old_target[:, :, :2])
    roots = original.root_initialization(rows, p['reference_frame'])
    assert np.allclose(roots, np.tile([0., np.pi, 0.], (N, 1)), rtol=0, atol=1.e-6)
    baseline = original.entry.load_official_config(original.OFFICIAL)
    candidate = hand_control.candidate_config(baseline)
    assert baseline['maxiters'] == 30 and candidate['hand_joints_weights'] == [0.] * 5
    assert [k for k in baseline if baseline[k] != candidate[k]] == ['hand_joints_weights']
    method, entry_delta = make_entry(N, roots)
    _, upstream_delta = original.adapter.source(original.OFFICIAL)
    compile(runner, '<officewoman127-runner-cpu-static-check>', 'exec')
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
        trainN_frozen_q90=q90.tolist(), confidence_input_strategy=confidence_receipt,
        source_observation_visibility_policy_sha256=visibility_sha256(p),
        source_observation_visibility=visibility_record(p),
        synthetic_visibility_contract=synthetic_visibility_checks(),
        original_hand_option_only=True, baseline_hand_weights=baseline['hand_joints_weights'],
        candidate_hand_weights=candidate['hand_joints_weights'], body_wrists_and_hand_prior_retained=True,
        runner_sha256=__import__('hashlib').sha256(runner.encode()).hexdigest(),
        upstream_interface_diff=upstream_delta, entry_interface_diff=entry_delta,
        cuda_initialized=False, fit_directory_created=False, fitting_started=False,
        GPU_preflight_executed=False, quality_accepted=False)
    save(CPU_AUDIT, result)
    print('OFFICEWOMAN_CIRCLE24_FIT127_CPU_PREPARATION_PASSED', p['training_frames'], p['reference_frame'], flush=True)


if __name__ == '__main__':
    {'audit': audit, 'prepare': prepare, 'preflight': preflight, 'run': run, 'launch': launch}[sys.argv[1]]()

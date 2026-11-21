"""New full-sequence CPU control: frozen8D primary, cached adjacent ICP fallback.

Preparation never registers a real pair. A GT-free witness contains ONLY the
original adjacent relative candidate and sensor/decision evidence. Extraction
honestly reads an original JSON with GT/absolute-trajectory fields but selects
none of those fields. Estimated ICP absolute poses never initialize this model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
import numpy as np
import track_rgbd_affine_full as full

ROOT = Path(__file__).resolve().parents[1]
affine, frozen = full.affine, full.frozen
require, sha, read, write, update = full.require, full.sha, full.read, full.write, full.update
DEADLINE = full.DEADLINE
FULL_PREP = ROOT / 'outputs/rgbd-affine-full-cpu-preparation-001'
FULL_PROTOCOL_SHA = 'ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a'
INPUT_REVIEW_SHA = 'd815c85825decc9d7330594033416fbfdf1dfc09d472c081176a4329265dfdeb'
ICP_AUDIT = ROOT / 'outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json'
ICP_AUDIT_SHA = '70f65dfced84eabf1ec313e93a908062d565a19401b1ae4c49b0d4d71801417e'
CORE_ICP = {'track_rgbd_icp.py': '21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a',
            'tum_rgbd.py': '471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11',
            'audit_tum_trajectory.py': 'e7acfaea60297dbc31eccc62b6ea9f50f577fb317209ac50e3342630424a42d7'}
SENSOR_KEYS = ('frame_id', 'rgb_index', 'depth_index', 'rgb_timestamp', 'depth_timestamp', 'rgb_depth_abs_dt_s', 'rgb_path', 'depth_path')
ROUNDS = [{'order': i + 1, 'round_id': scene + '-affine-adjacent-icp-fallback-full', 'scene': scene,
           'run': f'tum-{scene}-affine-icp-fallback-full{full.SCENES[scene]["frames"]}-001'} for i, scene in enumerate(('desk', 'xyz'))]
OLD_AFFINE = {'desk': '52bf33985013dda1f654348aae2814aa1d5c797c2e0c2abf3ba782dac31a4262',
              'xyz': '312259b7fd1545b2ff30ba05146a8a92456cf0b8470616a70bea1924f9539f9f'}


def sensor(metadata): return {k: metadata[k] for k in SENSOR_KEYS}
def canonical_hash(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
before_deadline = full.before_deadline


def source_checks():
    require(sha(FULL_PREP / 'protocol.json') == FULL_PROTOCOL_SHA and sha(FULL_PREP / 'root-independent-full-preflight-review.json') == INPUT_REVIEW_SHA, 'Frozen full source/input independent preflight')
    previous = read(FULL_PREP / 'protocol.json')['source_sha256']; require(full.source_checks() == previous, 'All18 original sources unchanged')
    require(sha(ICP_AUDIT) == ICP_AUDIT_SHA, 'Locked prior independent full ICP trajectory audit')
    return {**previous, **{name: sha(ROOT / 'scripts' / name) for name in ('track_rgbd_affine_icp_fallback.py', 'run_rgbd_affine_icp_fallback.py', 'check_rgbd_affine_icp_fallback.py')}}


def rigid_reason(matrix):
    if matrix is None: return 'candidate_unknown'
    try: a = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError): return 'candidate_invalid_shape_or_type'
    if a.shape != (4, 4): return 'candidate_invalid_shape_or_type'
    if not np.isfinite(a).all(): return 'candidate_nonfinite'
    if np.max(np.abs(a[3] - [0, 0, 0, 1])) >= 1e-12: return 'candidate_homogeneous_row_invalid'
    if np.max(np.abs(a[:3, :3].T @ a[:3, :3] - np.eye(3))) >= 1e-8 or abs(np.linalg.det(a[:3, :3]) - 1.) >= 1e-8: return 'candidate_not_proper_rigid'
    return None


def fallback_checks(row, index, current, previous):
    checks = {'original_ICP_accepted': row['original_accepted'] is True,
              'original_failure_flags_empty': row['original_failure_flags'] == [],
              'current_and_previous_IDs_are_adjacent': row['current_frame_id'] == index and row['previous_frame_id'] == index - 1,
              'current_and_previous_sensor_metadata_exact': row['current_sensor'] == sensor(current) and row['previous_sensor'] == sensor(previous),
              'original_current_timestamp_exact': row['original_current_rgb_timestamp'] == current['rgb_timestamp'],
              'frozen_ICP_core_source_exact': row['source_core_sha256'] == CORE_ICP,
              'current_to_previous_direction': row['candidate_direction'] == 'current -> immediately previous SENSOR frame',
              'candidate_proper_rigid': rigid_reason(row['candidate_previous_from_current']) is None}
    reasons = [name for name, passed in checks.items() if not passed]
    return not reasons, checks, reasons


def integrate(index, reference, poses, primary, fallback):
    """Pure causal decision. No loss, GT, sensor image or optimizer is accessed."""
    require(index == len(poses) and 0 <= reference < index, 'Causal pose/current/reference ledger')
    require(np.array_equal(poses[reference], poses[-1]), 'Last final accepted prefix equals propagated previous pose')
    require(primary['accepted'] == (not primary['failure_flags']), 'Frozen primary acceptance/flags agree')
    if primary['accepted']:
        require(rigid_reason(primary['candidate_reference_from_current']) is None, 'Accepted primary rigid')
        pose = np.asarray(poses[reference]) @ np.asarray(primary['candidate_reference_from_current'], dtype=np.float64)
        selection = 'primary_affine'; accepted = True; flags = []
    elif fallback['eligible']:
        require(fallback['current_frame_id'] == index and fallback['previous_frame_id'] == index - 1 and
                fallback['original_accepted'] is True and fallback['original_failure_flags'] == [] and
                fallback['candidate_direction'] == 'current -> immediately previous SENSOR frame' and
                rigid_reason(fallback['candidate_previous_from_current']) is None, 'Eligible adjacent original ICP candidate')
        pose = np.asarray(poses[index - 1]) @ np.asarray(fallback['candidate_previous_from_current'], dtype=np.float64)
        selection = 'adjacent_depth_ICP_fallback'; accepted = True; flags = []
    else:
        pose = np.asarray(poses[-1]).copy(); selection = 'hold_both_rejected'; accepted = False
        flags = ['primary/' + f for f in primary['failure_flags']] + ['fallback/' + f for f in fallback['ineligible_reasons']]
    next_reference = index if accepted else reference
    return pose, next_reference, {'selection': selection, 'accepted': accepted, 'failure_flags': flags,
            'primary_rejected': not primary['accepted'], 'fallback_selected': selection == 'adjacent_depth_ICP_fallback',
            'fallback_was_eligible': fallback['eligible'], 'primary_failure_flags': primary['failure_flags'],
            'fallback_ineligible_reasons': fallback['ineligible_reasons']}


def configuration_for(scene):
    return {**full.configuration_for(scene, 8),
            'fusion_policy': 'primary8D to last FINAL accepted SENSOR reference; otherwise accepted original adjacent ICP relative at preceding final pose; both reject hold',
            'primary_reference_is_last_final_accepted_SENSOR': True,
            'fallback_source': 'locked cached GT-free adjacent current->previous relative sensor candidates; no absolute ICP poses',
            'primary_priority_over_eligible_fallback': True, 'promote_current_sensor_on_primary_or_fallback_accept': True,
            'fallback_recomputes_ICP_or_changes_solver_parameters': False}


def extract_fallback(scene, witness):
    spec = full.SCENES[scene]; folder = ROOT / 'outputs' / spec['baseline']; path = folder / 'metrics.json'
    require(sha(path) == spec['metrics_sha256'] and sha(folder / 'validation.json') == spec['validation_sha256'], 'Exact completed original full ICP')
    m = read(path); validate = read(folder / 'validation.json'); metadata_path = folder / 'selected_sensor_metadata.json'; metadata = read(metadata_path)['selected_frames']
    require(m['status'] == 'completed' and m['frames_processed'] == spec['frames'] and len(m['frame_metrics']) == spec['frames'] and
            m['configuration']['start_frame'] == 0 and m['configuration']['source_sha256'] == CORE_ICP and validate['status'] == 'passed', 'Original frozen full ICP source/status')
    require([{k: r[k] for k in witness['selected_sensor_metadata'][0]} for r in metadata] == witness['selected_sensor_metadata'], 'Full source metadata matches reviewed input witness')
    prior_audit = read(ICP_AUDIT); require(prior_audit['status'] == 'passed_independent_lineage_full_trajectory_chunk_and_tensor_audit' and
        any(r['run'] == spec['baseline'] and r['full_frames'] == spec['frames'] for r in prior_audit['six_complete_trajectory_checks']), 'Original full ICP covered by independent numerical audit')
    rows = []
    for index, original in enumerate(m['frame_metrics'][1:], 1):
        require(original['frame_id'] == original['local_index'] == index, 'Every original adjacent pair chronological')
        row = {'current_frame_id': index, 'previous_frame_id': index - 1, 'original_accepted': original['accepted'],
               'original_failure_flags': copy.deepcopy(original['failure_flags']),
               'candidate_previous_from_current': copy.deepcopy(original['candidate_previous_from_current']),
               'candidate_direction': 'current -> immediately previous SENSOR frame',
               'current_sensor': sensor(witness['selected_sensor_metadata'][index]), 'previous_sensor': sensor(witness['selected_sensor_metadata'][index - 1]),
               'original_current_rgb_timestamp': original['rgb_timestamp'], 'source_core_sha256': CORE_ICP,
               'original_full_metric_row_sha256': canonical_hash(original), 'original_metric_row_index': index,
               'original_final_counts': copy.deepcopy(original['final_counts']), 'original_final_statistics': copy.deepcopy(original['final_statistics'])}
        eligible, checks, reasons = fallback_checks(row, index, witness['selected_sensor_metadata'][index], witness['selected_sensor_metadata'][index - 1])
        row.update(eligible=eligible, eligibility_checks=checks, ineligible_reasons=reasons); rows.append(row)
    return {'status': 'extracted_locked_adjacent_relative_sensor_candidates_only', 'scene': scene, 'frames': spec['frames'], 'start_frame': 0,
            'source_run': spec['baseline'], 'source_metrics_sha256': spec['metrics_sha256'], 'source_validation_sha256': spec['validation_sha256'],
            'source_metadata_sha256': sha(metadata_path), 'source_core_sha256': CORE_ICP, 'independent_original_full_trajectory_audit_sha256': ICP_AUDIT_SHA,
            'rows': rows, 'eligible_pair_count': sum(r['eligible'] for r in rows), 'requested_adjacent_pair_count': spec['frames'] - 1,
            'contains_GT_pose_or_absolute_ICP_pose': False,
            'scope': 'Extraction reads original JSON containing GT evaluation and estimated absolute pose fields; strict whitelist selects only relative sensor candidate/decision/RGB-depth metadata. No GT or absolute trajectory is used in fallback selection.'}


def synthetic_controls():
    exp = frozen.recovery.matrix_exp
    t01 = exp([.03, -.02, .015, .1, -.06, .04]); t12 = exp([-.01, .025, .01, -.05, .08, .02]); t02 = t01 @ t12
    def primary(accepted, transform=None): return {'accepted': accepted, 'failure_flags': [] if accepted else ['synthetic_primary_reject'], 'candidate_reference_from_current': (transform.tolist() if transform is not None else None)}
    def fallback(index, accepted, transform):
        return {'eligible': accepted, 'current_frame_id': index, 'previous_frame_id': index - 1, 'original_accepted': accepted,
                'original_failure_flags': [] if accepted else ['synthetic_ICP_reject'], 'candidate_previous_from_current': transform.tolist(),
                'candidate_direction': 'current -> immediately previous SENSOR frame', 'ineligible_reasons': [] if accepted else ['original_ICP_accepted', 'original_failure_flags_empty']}
    pose, ref, decision = integrate(1, 0, [np.eye(4)], primary(True, t01), fallback(1, True, t12))
    require(np.array_equal(pose, t01) and decision['selection'] == 'primary_affine', 'Primary always wins over eligible different ICP')
    pose2, ref2, decision2 = integrate(2, ref, [np.eye(4), pose], primary(False), fallback(2, True, t12))
    wrong_inverse = pose @ np.linalg.inv(t12); wrong_order = t12 @ pose
    require(np.max(np.abs(pose2 - t02)) < 1e-15 and ref2 == 2 and decision2['fallback_selected'] and
            np.max(np.abs(wrong_inverse - pose2)) > .01 and np.max(np.abs(wrong_order - pose2)) > .001, 'Fallback current->prev noncommuting direction/product negative controls')
    held, href, hdecision = integrate(2, ref, [np.eye(4), pose], primary(False), fallback(2, False, t12))
    require(np.array_equal(held, pose) and href == ref and len(hdecision['failure_flags']) == 3, 'Both rejects hold and retain all reason families')
    recovered, rref, rdecision = integrate(3, href, [np.eye(4), pose, held], primary(True, t02), fallback(3, False, t12))
    require(np.array_equal(recovered, pose @ t02) and rref == 3, 'Primary held-reference transform uses matching reference pose')
    stale_denied = False
    try: integrate(2, 0, [np.eye(4), pose], primary(False), fallback(2, True, t12))
    except ValueError: stale_denied = True
    require(stale_denied, 'Stale reference with changed previous pose is invalid ledger')
    patterns = {'all_primary': [(True, True)] * 5, 'all_fallback': [(False, True)] * 5,
                'all_hold': [(False, False)] * 5, 'hold_then_fallback_then_primary': [(False, False), (False, False), (False, True), (True, True), (False, False)]}
    ledgers = {}
    for name, choices in patterns.items():
        poses = [np.eye(4)]; reference = 0; rows = []
        for index, (pflag, fflag) in enumerate(choices, 1):
            before = reference; transform = exp([.01 * index, -.004, .002, .008, -.009 * index, .003])
            p, reference, d = integrate(index, reference, poses, primary(pflag, transform), fallback(index, fflag, transform))
            expected = poses[before] @ transform if pflag else poses[index - 1] @ transform if fflag else poses[-1].copy()
            require(np.array_equal(p, expected) and reference == (index if pflag or fflag else before), 'Synthetic complete selection/reference ledger')
            poses.append(p); rows.append({'current': index, 'primary_reference': before, 'fallback_reference': index - 1,
                                        'reference_after': reference, 'reference_gap': index - before, 'estimated_c2w': p.tolist(), **d})
        ledgers[name] = rows
    meta = [{k: (i if 'index' in k or k == 'frame_id' else float(i) if 'timestamp' in k else .0 if k == 'rgb_depth_abs_dt_s' else f'{k}/{i}.png') for k in SENSOR_KEYS} for i in range(2)]
    row = {**fallback(1, True, t01), 'current_sensor': meta[1], 'previous_sensor': meta[0], 'original_current_rgb_timestamp': meta[1]['rgb_timestamp'], 'source_core_sha256': CORE_ICP}
    require(fallback_checks(row, 1, meta[1], meta[0])[0], 'Valid synthetic sensor qualification')
    negatives = {}
    for name, change in {'wrong_previous_ID': {'previous_frame_id': 1}, 'wrong_source_metadata': {'previous_sensor': meta[1]},
                         'original_rejection': {'original_accepted': False}, 'source_drift': {'source_core_sha256': {}},
                         'wrong_direction_label': {'candidate_direction': 'previous -> current'}, 'unknown_candidate': {'candidate_previous_from_current': None}}.items():
        eligible, _, reasons = fallback_checks({**row, **change}, 1, meta[1], meta[0]); require(not eligible and reasons, 'Qualification negative ' + name); negatives[name] = reasons
    return {'status': 'passed_synthetic_fusion_controls', 'noncommuting_fallback_vs_inverse_max_difference': float(np.max(np.abs(wrong_inverse - pose2))),
            'noncommuting_fallback_vs_wrong_order_max_difference': float(np.max(np.abs(wrong_order - pose2))),
            'primary_priority': True, 'all_rejection_reasons_retained': hdecision['failure_flags'], 'stale_reference_invalid_ledger_rejected': True,
            'sensor_source_direction_qualification_negative_controls': negatives, 'complete_ledgers': ledgers,
            'important_boundary': 'In a valid promoted-or-held ledger Cref equals Cprev numerically. Sensor reference IDs/directions still differ after holds. Adjacent fallback cannot reconstruct unobserved movement from previously rejected pairs.',
            'real_registration_or_normal_equation_calls': 0, 'GPU_calls': 0}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__); modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--execute', action='store_true')
    p.add_argument('--protocol', type=Path); p.add_argument('--protocol-sha'); p.add_argument('--round-id', choices=[r['round_id'] for r in ROUNDS]); p.add_argument('--run', required=True)
    a = p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', a.run): p.error('Unique simple run')
    if any(a.execute != (v is not None) for v in (a.protocol, a.protocol_sha, a.round_id)): p.error('Execute requires protocol/hash/round; prepare accepts none')
    if a.protocol_sha and not re.fullmatch(r'[a-f0-9]{64}', a.protocol_sha): p.error('Exact lowercase SHA')
    return a


def read_protocol(path, digest):
    raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == digest, 'Exact reviewed fusion protocol'); p = json.loads(raw)
    require(p['status'] == 'ready_for_explicit_root_review_no_real_registration' and p['source_sha256'] == source_checks() and
            p['round_order'] == [r['round_id'] for r in ROUNDS] and len(p['rounds']) == 2 and p['deadline_beijing'] == DEADLINE.isoformat(), 'Locked2 rounds/source/deadline')
    for row, spec in zip(p['rounds'], ROUNDS): require({k: row[k] for k in spec} == spec and row['configuration'] == configuration_for(row['scene']), 'Exact complete2 configs')
    return raw, p


def prepare(output):
    sources = source_checks(); original = read(FULL_PREP / 'protocol.json'); independent_review = read(FULL_PREP / 'root-independent-full-preflight-review.json')
    require(independent_review['status'] == 'passed_independent_full_preflight_review_no_actual_execution' and independent_review['actual_associated_sensor_frames'] == 1297, 'Explicit reviewed full input reuse')
    witnesses = {}; references = {}
    for scene in ('desk', 'xyz'):
        spec = original['datasets'][scene]; path = ROOT / spec['path']; require(sha(path) == spec['sha256'], 'Frozen full input witness')
        fallback = extract_fallback(scene, read(path)); target = output / (scene + '-adjacent-ICP-relative-candidates.json'); write(target, fallback)
        witnesses[scene] = {'path': target.relative_to(ROOT).as_posix(), 'sha256': sha(target), 'frames': fallback['frames'],
                            'eligible_pair_count': fallback['eligible_pair_count'], 'candidate_row_count': len(fallback['rows']), 'contains_GT_pose_or_absolute_ICP_pose': False}
        ref = ROOT / 'outputs' / f'tum-{scene}-lastaccepted-8d-full{full.SCENES[scene]["frames"]}-001' / 'metrics.json'
        require(sha(ref) == OLD_AFFINE[scene], 'Original full8D reference bytes retained; no reexecution')
        references[scene] = {'path': ref.relative_to(ROOT).as_posix(), 'sha256': OLD_AFFINE[scene], 'comparison_metrics_requires_independent_actual_audit_before_publication': True}
    controls = synthetic_controls(); write(output / 'synthetic_fusion_controls.json', controls)
    rounds = []
    for spec in ROUNDS:
        require(not (ROOT / 'outputs' / spec['run']).exists(), 'Actual fusion output absent'); rounds.append({**spec, 'configuration': configuration_for(spec['scene'])})
    p = {'status': 'ready_for_explicit_root_review_no_real_registration', 'source_sha256': sources, 'rounds': rounds,
         'round_order': [r['round_id'] for r in ROUNDS], 'deadline_beijing': DEADLINE.isoformat(), 'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
         'datasets': original['datasets'], 'full_input_preflight_reuse': {'protocol_path': (FULL_PREP / 'protocol.json').relative_to(ROOT).as_posix(), 'protocol_sha256': FULL_PROTOCOL_SHA,
               'independent_review_path': (FULL_PREP / 'root-independent-full-preflight-review.json').relative_to(ROOT).as_posix(), 'independent_review_sha256': INPUT_REVIEW_SHA,
               'scope': 'Reuse already independently checked1297 raw sensor metadata/2594PNG/30full arrays/30short parity; preparation rehashes immutable witness, not a second array audit.'},
         'fallback_candidate_witnesses': witnesses, 'independent_original_ICP_audit': {'path': ICP_AUDIT.relative_to(ROOT).as_posix(), 'sha256': ICP_AUDIT_SHA},
         'old_full8D_references': references, 'synthetic_controls_sha256': sha(output / 'synthetic_fusion_controls.json'), 'CPU_registration_runs': 0, 'GPU_runs': 0,
         'limits': ['Full549/748 fixed associated frames, identity first camera and original15c8D identity/a0/b0 pair initialization; all weights/scales/gates unchanged.',
                    'Original fullICP was previously completed with frozen21b sensor solver. Cached relative candidate witness is consumed only at current frame; no ICP absolute pose/GT pose used to initialize or select.',
                    'Extractor honestly reads original JSON containing GT/absolute pose fields; strict witness whitelist excludes them. Loader reads GT/timestamps for association, GT matrices consumed only after all final poses exported.',
                    'Primary accept uses matching final accepted SENSORreference c2w@Tprimary; otherwise qualified adjacent ICP uses preceding FINAL c2w@TICP; either accept promotes current sensor; both reject hold.',
                    'Primary rejection counts remain separate from final rejection counts. All primary candidates/trials/flags, all fallback evidence/decisions and all final frames remain.',
                    'Adjacent fallback cannot restore unobserved motion from earlier dual rejections; this is a whole-pipeline/path control, not a proof of illumination cause or a complete SLAM implementation.',
                    'Prepare forbids register/normal systems; actual2 serial CPU requires root review, strict cutoff before Popen and first pair; started child collected, no retry.',
                    'No cached-ICP isolated speed/FPS or cross-sequence average claim. Old full8D metrics comparison waits for independent actual audit.']}
    write(output / 'protocol.json', p); digest = sha(output / 'protocol.json'); parsed = []
    read_protocol(output / 'protocol.json', digest)
    for row in rounds:
        argv = ['--execute', '--protocol', str(output / 'protocol.json'), '--protocol-sha', digest, '--round-id', row['round_id'], '--run', row['run']]
        a = arguments(argv); require(a.run == row['run'] and a.round_id == row['round_id'], 'Actual parsed2 child CLI'); parsed.append({'argv': argv, 'configuration': row['configuration']})
    write(output / 'cli_checks.json', {'status': 'passed_no_execution', 'protocol_sha256': digest, 'rounds': parsed, 'registration_and_normal_systems_forbidden': True, 'GPU_calls': 0})
    snap = output / 'source_snapshot'; snap.mkdir()
    for name in sources: (snap / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    write(output / 'source_manifest.json', {'source_sha256': sources, 'protocol_sha256': digest})
    require(source_checks() == sources and not any(n in sys.modules for n in ('torch', 'gsplat')), 'Source/CPU guard')
    return {'status': p['status'], 'protocol_sha256': digest, 'actual_registration_runs': 0, 'GPU_calls': 0}


def execute(args, output):
    raw, p = read_protocol(args.protocol, args.protocol_sha); row = next(r for r in p['rounds'] if r['round_id'] == args.round_id); require(row['run'] == args.run, 'Exact output')
    scene = row['scene']; ws = p['fallback_candidate_witnesses'][scene]; require(sha(ROOT / ws['path']) == ws['sha256'], 'Frozen causal relative witness')
    witness = read(ROOT / ws['path']); count = row['configuration']['frames']; require(len(witness['rows']) == count - 1 and witness['contains_GT_pose_or_absolute_ICP_pose'] is False, 'Complete GT-free candidate witness')
    input_spec = p['datasets'][scene]; require(sha(ROOT / input_spec['path']) == input_spec['sha256'], 'Input witness exact')
    record, data, frames = full.dataset_preflight(scene); require(record == read(ROOT / input_spec['path']), 'Actual sensor input equality')
    started_utc = before_deadline(); output.mkdir(exist_ok=False); (output / 'pairs').mkdir(); (output / 'candidates').mkdir(); (output / 'protocol_used.json').write_bytes(raw)
    state = {'status': 'running', 'configuration': row['configuration'], 'source_sha256': p['source_sha256'], 'protocol_sha256': args.protocol_sha,
             'started_utc': started_utc, 'frame_metrics': [], 'frames_processed': 0, 'GT_pose_used_in_optimizer': False, 'GT_pose_used_in_selection': False,
             'fallback_uses_absolute_ICP_pose': False, 'ground_truth_scope': record['ground_truth_scope'], 'GPU_runs': 0}
    write(output / 'configuration.json', {**row['configuration'], 'source_sha256': p['source_sha256'], 'protocol_sha256': args.protocol_sha, 'CPU_only': True})
    config = frozen.config_for(); poses = [np.eye(4)]; reference = 0; rows = state['frame_metrics']; start = time.perf_counter()
    def geometry(index): return [affine.frozen.make_frame(d['depth'][index], d['valid'][index], d['K'], d['rgb'][index], config.base) for d in data]
    try:
        targets = geometry(0); anchor = {'frame_id': 0, 'local_index': 0, 'accepted': True, 'failure_flags': [], 'selection': 'fixed_first_camera_identity',
             'primary_rejected': False, 'fallback_selected': False, 'reference_frame_id': 0, 'reference_gap_frames': 0, 'last_accepted_frame_id_after': 0, 'estimated_c2w': poses[0].tolist()}
        rows.append(anchor); write(output / 'pairs/0000.json', anchor); state['frames_processed'] = 1
        with (output / 'iterations.jsonl').open('x', encoding='utf-8') as stream:
            for index in range(1, count):
                before = reference; sources = geometry(index)
                if index == 1: state['first_pair_registration_start_utc'] = before_deadline()
                try: primary = affine.register_pair_affine(sources, targets, config)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as error:
                    primary = {'accepted': False, 'failure_flags': ['pair_numeric_exception'], 'candidate_reference_from_current': None,
                       'candidate_log_relative_gain': None, 'candidate_relative_gain': None, 'candidate_bias_intensity': None, 'final_counts': {}, 'final_statistics': {},
                       'iteration_history': [], 'levels': [], 'exception_type': type(error).__name__, 'exception_message': str(error), 'traceback': traceback.format_exc(), 'candidate_unknown_due_to_exception': True}
                fallback = copy.deepcopy(witness['rows'][index - 1]); eligible, checks, reasons = fallback_checks(fallback, index, record['selected_sensor_metadata'][index], record['selected_sensor_metadata'][index - 1])
                require((eligible, checks, reasons) == (fallback['eligible'], fallback['eligibility_checks'], fallback['ineligible_reasons']), 'Current causal fallback eligibility unchanged')
                candidate = {'frame_id': index, 'primary_reference_frame_id': before, 'primary': primary, 'fallback': fallback,
                    'primary_current_sensor': sensor(record['selected_sensor_metadata'][index]), 'primary_reference_sensor': sensor(record['selected_sensor_metadata'][before]),
                    'pair_initial_transform': np.eye(4).tolist(), 'pair_initial_log_relative_gain': 0., 'pair_initial_bias_intensity': 0., 'brightness_accumulated_across_pairs': False}
                write(output / 'candidates' / f'{index:04d}.json', candidate)
                pose, reference, decision = integrate(index, before, poses, primary, fallback); poses.append(pose)
                pair = {**candidate, **decision, 'local_index': index, 'current_frame_id': index, 'reference_frame_id': before, 'reference_gap_frames': index - before,
                        'fallback_reference_frame_id': index - 1, 'last_accepted_frame_id_before': before, 'last_accepted_frame_id_after': reference,
                        'estimated_c2w': pose.tolist(), 'maximum_sensor_frame_used': index, 'pair_initialized_from_GT': False}
                write(output / 'pairs' / f'{index:04d}.json', pair); rows.append(pair)
                for item in primary['iteration_history']: stream.write(json.dumps({'frame_id': index, 'primary_reference_frame_id': before, **item}, allow_nan=False) + '\n')
                stream.flush()
                if decision['accepted']: targets = sources
                state.update(frames_processed=len(rows), last_completed_frame_id=index, last_final_reference_id=reference, last_selection=decision['selection'])
                update(output / 'progress.json', {k: v for k, v in state.items() if k != 'frame_metrics'}); full.original.pose_text(output / 'trajectory_estimated_partial.txt', frames[:len(poses)], poses)
                print(json.dumps({'round': row['round_id'], 'frame': index, 'primary_reference': before, 'selection': decision['selection'], 'primary_flags': primary['failure_flags'], 'final_flags': decision['failure_flags']}), flush=True)
        full.original.pose_text(output / 'trajectory_estimated.txt', frames, poses)
        # GT matrix consumption after the entire finalized accepted/rejected path export.
        absolute = np.stack([f.c2w for f in frames]); anchored = np.linalg.inv(absolute[0])[None] @ absolute
        from audit_tum_trajectory import audit_pose_arrays
        trajectory = audit_pose_arrays(np.asarray(poses), anchored); full.original.pose_text(output / 'trajectory_groundtruth_anchored.txt', frames, anchored)
        np.savez_compressed(output / 'trajectory_arrays.npz', estimated_c2w=np.asarray(poses), evaluation_only_anchored_GT_c2w=anchored, rgb_timestamps=np.asarray([f.rgb_timestamp for f in frames]))
        failures = sum(bool(r['failure_flags']) for r in rows); primary_failures = sum(r['primary_rejected'] for r in rows); fallback_selections = sum(r['fallback_selected'] for r in rows)
        state.update(status='completed' if not failures else 'completed_with_final_pair_rejections', frames_processed=count, failure_frame_count=failures, final_failure_frame_count=failures,
              primary_failure_frame_count=primary_failures, fallback_selected_frame_count=fallback_selections, tracked_nonanchor_frame_denominator=count - 1, all_frame_denominator=count,
              final_failure_flag_counts=dict(Counter(f for r in rows for f in r['failure_flags'])), primary_failure_flag_counts=dict(Counter(f for r in rows[1:] for f in r['primary']['failure_flags'])),
              selection_counts=dict(Counter(r['selection'] for r in rows)), trajectory=trajectory, selected_sensor_metadata=record['selected_sensor_metadata'],
              GT_pose_evaluation_after_all_estimates_saved=True, observed_wall_seconds=time.perf_counter() - start,
              timing_scope='Cached-ICP candidate input; CPU includes primary solver/geometry/I/O. No isolated speed/FPS comparison.', limits=p['limits'])
        require(source_checks() == p['source_sha256'] and sha(ROOT / ws['path']) == ws['sha256'] and not any(k in sys.modules for k in ('torch', 'gsplat')), 'Source/witness/CPU postguard')
        write(output / 'metrics.json', state); update(output / 'progress.json', {k: v for k, v in state.items() if k != 'frame_metrics'})
        write(output / 'pair_manifest.json', {str(q.relative_to(output)).replace('\\', '/'): {'bytes': q.stat().st_size, 'sha256': sha(q)} for sub in ('pairs', 'candidates') for q in sorted((output / sub).glob('*.json'))})
        return {'status': state['status'], 'frames_processed': count, 'primary_rejection_count': primary_failures, 'fallback_selection_count': fallback_selections, 'final_rejection_count': failures}
    except BaseException as error:
        state.update(status='failed_incomplete_original_frames_retained', exception_type=type(error).__name__, exception_message=str(error), traceback=traceback.format_exc(), frames_processed=len(rows), GT_final_eval_completed=False)
        write(output / 'failure.json', state); update(output / 'progress.json', {k: v for k, v in state.items() if k != 'frame_metrics'}); raise


def main():
    args = arguments(); output = ROOT / 'outputs' / args.run; require(not output.exists(), 'Unique output')
    if args.prepare:
        output.mkdir(); (output / 'preparation_source.py').write_bytes(Path(__file__).read_bytes())
        targets = [(affine, 'register_pair_affine'), (affine, 'normalized_system'), (affine.frozen, 'register_pair'), (affine.frozen, 'weighted_system'), (affine.frozen.icp, 'register_pair')]
        saved = [getattr(m, k) for m, k in targets]
        def forbidden(*a, **k): raise AssertionError('No real registration/normal equations in fusion preparation')
        for m, k in targets: setattr(m, k, forbidden)
        try: result = prepare(output)
        except BaseException as error:
            write(output / 'failure.json', {'status': 'fusion_preparation_failed_preserved', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'GPU_calls': 0}); raise
        finally:
            for (m, k), value in zip(targets, saved): setattr(m, k, value)
    else: result = execute(args, output)
    print(json.dumps(result))


if __name__ == '__main__': main()

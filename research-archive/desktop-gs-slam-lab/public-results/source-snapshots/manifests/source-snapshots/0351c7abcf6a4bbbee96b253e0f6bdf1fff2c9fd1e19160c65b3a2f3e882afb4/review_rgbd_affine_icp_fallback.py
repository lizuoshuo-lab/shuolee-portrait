"""Independent SOURCE/CPU review; no actual fusion, pair registration or training."""
from __future__ import annotations
import ast
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True
import track_rgbd_affine_icp_fallback as subject

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/rgbd-affine-icp-fallback-cpu-preparation-001'
PROTOCOL_SHA = '358a418359ef4f2a78c9ad73ac78a7fa69582457843d23cb28cd142e018bf077'
NEW_SOURCES = {
    'track_rgbd_affine_icp_fallback.py': 'fb25204d6c9780378cf2f58eec6bc4a27d6c32297ea2c7d283e22f4129cb6922',
    'run_rgbd_affine_icp_fallback.py': '98892ca2ff4da1248aa093b3b4d17145ac3f831c607917c466190d9debfa49ba',
    'check_rgbd_affine_icp_fallback.py': 'cefe45a8a715f04a91598d613e8127a4d033446a8dcfd9453dfe8e4660350b1c',
}
SENSOR_KEYS = {'frame_id', 'rgb_index', 'depth_index', 'rgb_timestamp', 'depth_timestamp', 'rgb_depth_abs_dt_s', 'rgb_path', 'depth_path'}
ROW_KEYS = {'current_frame_id', 'previous_frame_id', 'original_accepted', 'original_failure_flags', 'candidate_previous_from_current',
            'candidate_direction', 'current_sensor', 'previous_sensor', 'original_current_rgb_timestamp', 'source_core_sha256',
            'original_full_metric_row_sha256', 'original_metric_row_index', 'original_final_counts', 'original_final_statistics',
            'eligible', 'eligibility_checks', 'ineligible_reasons'}
COUNT_KEYS = {'valid_source_normals', 'positive_projectable', 'projected_inside', 'valid_target_normals', 'normal_consistent', 'distance_consistent', 'unique_correspondences'}
STAT_KEYS = {'rank', 'hessian_eigenvalues_ascending', 'column_normalized_hessian_eigenvalues_ascending',
             'column_normalized_hessian_condition', 'jacobian_column_norms', 'plane_rmse_m', 'plane_median_absolute_m', 'robust_objective'}
CORE = {
    'track_rgbd_icp.py': '21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a',
    'tum_rgbd.py': '471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11',
    'audit_tum_trajectory.py': 'e7acfaea60297dbc31eccc62b6ea9f50f577fb317209ac50e3342630424a42d7',
}


def require(condition, message):
    if not condition: raise AssertionError(message)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def rigid(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    return a.shape == (4, 4) and np.isfinite(a).all() and np.max(np.abs(a[3] - [0, 0, 0, 1])) < 1e-12 and np.max(np.abs(a[:3, :3].T @ a[:3, :3] - np.eye(3))) < 1e-8 and abs(np.linalg.det(a[:3, :3]) - 1) < 1e-8


def strict_row(row):
    require(set(row) == ROW_KEYS, 'Relative witness exact top-level field whitelist')
    require(set(row['current_sensor']) == set(row['previous_sensor']) == SENSOR_KEYS, 'Sensor-only metadata whitelist')
    require(set(row['original_final_counts']) == COUNT_KEYS and set(row['original_final_statistics']) == STAT_KEYS, 'ICP statistics/counts contain no pose or GT fields')


def original_candidates(protocol, witnesses):
    results = []
    for scene, count in [('desk', 549), ('xyz', 748)]:
        entry = protocol['fallback_candidate_witnesses'][scene]
        inp = protocol['datasets'][scene]
        require(digest(ROOT / entry['path']) == entry['sha256'] and digest(ROOT / inp['path']) == inp['sha256'], 'Actual witness/input hashes')
        witness = read(ROOT / entry['path']); input_record = read(ROOT / inp['path'])
        spec = subject.full.SCENES[scene]; folder = ROOT / 'outputs' / spec['baseline']
        mpath = folder / 'metrics.json'; vpath = folder / 'validation.json'; meta_path = folder / 'selected_sensor_metadata.json'
        require(digest(mpath) == spec['metrics_sha256'] and digest(vpath) == spec['validation_sha256'], 'Frozen original ICP metrics/validation bytes')
        original = read(mpath); original_metadata = read(meta_path)['selected_frames']
        require(original['status'] == 'completed' and read(vpath)['status'] == 'passed' and original['frames_processed'] == count and original['configuration']['source_sha256'] == CORE, 'Original completed ICP source/config')
        require(len(witness['rows']) == count - 1 and len(original['frame_metrics']) == count and len(input_record['selected_sensor_metadata']) == count, 'All non-anchor rows retained')
        require(witness['contains_GT_pose_or_absolute_ICP_pose'] is False and witness['source_metadata_sha256'] == digest(meta_path), 'GT-free declared payload and metadata provenance')
        selected = input_record['selected_sensor_metadata']
        require([{k: row[k] for k in selected[0]} for row in original_metadata] == selected, 'All original source association metadata equals reused independent input')
        max_rgram = max_bottom = 0.
        for index, (old, row) in enumerate(zip(original['frame_metrics'][1:], witness['rows']), 1):
            strict_row(row)
            require(old['frame_id'] == old['local_index'] == index, 'Original adjacent index chronology')
            c, prev = ({k: selected[j][k] for k in SENSOR_KEYS} for j in (index, index - 1))
            a = np.asarray(old['candidate_previous_from_current'], dtype=np.float64)
            independent_checks = {
                'original_ICP_accepted': old['accepted'] is True,
                'original_failure_flags_empty': old['failure_flags'] == [],
                'current_and_previous_IDs_are_adjacent': row['current_frame_id'] == index and row['previous_frame_id'] == index - 1,
                'current_and_previous_sensor_metadata_exact': row['current_sensor'] == c and row['previous_sensor'] == prev,
                'original_current_timestamp_exact': old['rgb_timestamp'] == selected[index]['rgb_timestamp'],
                'frozen_ICP_core_source_exact': original['configuration']['source_sha256'] == CORE,
                'current_to_previous_direction': row['candidate_direction'] == 'current -> immediately previous SENSOR frame',
                'candidate_proper_rigid': rigid(a),
            }
            reasons = [name for name, ok in independent_checks.items() if not ok]
            rebuilt = {
                'current_frame_id': index, 'previous_frame_id': index - 1, 'original_accepted': old['accepted'],
                'original_failure_flags': old['failure_flags'], 'candidate_previous_from_current': old['candidate_previous_from_current'],
                'candidate_direction': 'current -> immediately previous SENSOR frame', 'current_sensor': c, 'previous_sensor': prev,
                'original_current_rgb_timestamp': old['rgb_timestamp'], 'source_core_sha256': CORE,
                'original_full_metric_row_sha256': canonical(old), 'original_metric_row_index': index,
                'original_final_counts': old['final_counts'], 'original_final_statistics': old['final_statistics'],
                'eligible': not reasons, 'eligibility_checks': independent_checks, 'ineligible_reasons': reasons,
            }
            require(row == rebuilt, 'Every cached candidate/decision/statistic/opaque original-row hash independently reconstructed')
            require(all(independent_checks.values()), 'All predeclared cached candidates independently eligible')
            max_rgram = max(max_rgram, float(np.max(np.abs(a[:3, :3].T @ a[:3, :3] - np.eye(3)))))
            max_bottom = max(max_bottom, float(np.max(np.abs(a[3] - [0, 0, 0, 1]))))
        require(witness['eligible_pair_count'] == witness['requested_adjacent_pair_count'] == count - 1, 'Exact eligible/requested counts')
        witnesses[scene] = witness
        results.append({'scene': scene, 'frames': count, 'relative_rows_reconstructed': count - 1, 'eligible_rows': count - 1,
                        'maximum_Rgram_element_error': max_rgram, 'maximum_bottom_row_error': max_bottom,
                        'candidate_witness_sha256': entry['sha256'], 'original_metrics_sha256': digest(mpath), 'input_witness_sha256': inp['sha256']})
    return results


def synthetic(witnesses):
    def transform(rot, trans):
        a = np.eye(4); a[:3, :3] = Rotation.from_rotvec(rot).as_matrix(); a[:3, 3] = trans; return a
    c0 = transform([.1, -.2, .3], [1., -.4, .6])
    primary_t = transform([-.15, .04, .08], [.03, .01, -.02])
    icp_t = transform([.12, .02, -.09], [-.04, .02, .015])
    primary = {'accepted': True, 'failure_flags': [], 'candidate_reference_from_current': primary_t.tolist()}
    fallback = copy.deepcopy(witnesses['desk']['rows'][0]); fallback['candidate_previous_from_current'] = icp_t.tolist()
    out, ref, decision = subject.integrate(1, 0, [c0], primary, fallback)
    require(np.array_equal(out, c0 @ primary_t) and ref == 1 and decision['selection'] == 'primary_affine', 'Primary has priority even when eligible fallback differs')
    rejected = {'accepted': False, 'failure_flags': ['insufficient_correspondences', 'ill_conditioned_system'], 'candidate_reference_from_current': primary_t.tolist()}
    out, ref, decision = subject.integrate(1, 0, [c0], rejected, fallback)
    expected = c0 @ icp_t
    require(np.array_equal(out, expected) and ref == 1 and decision['selection'] == 'adjacent_depth_ICP_fallback' and decision['failure_flags'] == [] and decision['primary_failure_flags'] == rejected['failure_flags'], 'Adjacent fallback selection, promotion and primary rejection evidence preserved separately')
    point = np.array([.2, .1, 1., 1.])
    inverse_error = float(np.linalg.norm((c0 @ np.linalg.inv(icp_t) @ point - expected @ point)[:3]))
    left_error = float(np.linalg.norm((icp_t @ c0 @ point - expected @ point)[:3]))
    require(inverse_error > .01 and left_error > .01, 'Independent world-point negatives distinguish inverse and left composition')
    bad = copy.deepcopy(fallback); bad.update(eligible=False, ineligible_reasons=['original_ICP_accepted', 'candidate_proper_rigid'])
    out, ref, decision = subject.integrate(1, 0, [c0], rejected, bad)
    require(np.array_equal(out, c0) and ref == 0 and decision['failure_flags'] == ['primary/' + f for f in rejected['failure_flags']] + ['fallback/' + f for f in bad['ineligible_reasons']], 'Both reject holds exact pose and retains both failure families')
    held = [c0, c0.copy(), c0.copy()]; following = copy.deepcopy(fallback); following.update(current_frame_id=3, previous_frame_id=2)
    out, ref, d = subject.integrate(3, 0, held, rejected, following)
    require(np.array_equal(out, c0 @ icp_t) and ref == 3, 'Fallback after two holds uses physical adjacent2->3 candidate; valid reference0 and previous2 poses are numerically equal')
    stale = held.copy(); stale[-1] = transform([0, 0, 0], [2, 0, 0])
    try: subject.integrate(3, 0, stale, rejected, following)
    except ValueError: pass
    else: raise AssertionError('Invalid unequal held ledger must be rejected')
    row = copy.deepcopy(witnesses['desk']['rows'][0]); current, previous = row['current_sensor'], row['previous_sensor']
    patches = {
        'wrong_current_id': {'current_frame_id': 2}, 'wrong_previous_id': {'previous_frame_id': 9},
        'original_rejected': {'original_accepted': False}, 'original_flag_nonempty': {'original_failure_flags': ['rank_deficient']},
        'source_drift': {'source_core_sha256': {}}, 'direction_reversed': {'candidate_direction': 'previous -> current'},
        'timestamp_changed': {'original_current_rgb_timestamp': current['rgb_timestamp'] + .1},
        'sensor_mismatch': {'previous_sensor': {**previous, 'rgb_path': 'different.png'}},
        'candidate_unknown': {'candidate_previous_from_current': None},
        'candidate_shape': {'candidate_previous_from_current': np.eye(3).tolist()},
        'candidate_NaN': {'candidate_previous_from_current': np.full((4, 4), np.nan).tolist()},
        'candidate_bad_bottom': {'candidate_previous_from_current': (np.eye(4) + np.diag([0, 0, 0, 1e-6])).tolist()},
        'candidate_reflection': {'candidate_previous_from_current': np.diag([-1, 1, 1, 1]).tolist()},
    }
    negatives = {}
    for name, patch in patches.items():
        edited = {**copy.deepcopy(row), **patch}
        eligible, checks, reasons = subject.fallback_checks(edited, 1, current, previous)
        require(not eligible and reasons, 'Invalid eligibility must be rejected: ' + name)
        negatives[name] = {'status': 'rejected_as_required', 'failed_checks': reasons}
    for name, patch in [('GT_pose_injection', {'GT_c2w': np.eye(4).tolist()}), ('absolute_ICP_pose_injection', {'estimated_c2w': np.eye(4).tolist()})]:
        try: strict_row({**row, **patch})
        except AssertionError: negatives[name] = {'status': 'rejected_by_independent_exact_whitelist'}
        else: raise AssertionError(name)
    inverted = copy.deepcopy(row); inverted['candidate_previous_from_current'] = np.linalg.inv(np.asarray(row['candidate_previous_from_current'])).tolist()
    require(canonical(inverted) != canonical(row), 'Rigid inverse with unchanged label is caught by frozen witness/original-row reconstruction, not falsely by mere rigid eligibility')
    return {'status': 'passed_synthetic_direction_ledger_flags_and_eligibility_controls', 'inverse_world_point_error_m': inverse_error,
            'left_composition_world_point_error_m': left_error, 'negative_controls': negatives,
            'semantic_boundary': 'A rigid inverted transform can pass structural eligibility if labelled identically; exact witness SHA and independent original-row reconstruction bind its physical meaning. Cref==Cprev in valid holds, so changing only those equal matrices is not a meaningful mathematical negative.'}


def source_structure(protocol):
    sources = {name: (ROOT / 'scripts' / name).read_text(encoding='utf-8') for name in NEW_SOURCES}
    compiled = []
    for name, text in sources.items():
        compile(text, name, 'exec'); tree = ast.parse(text)
        require(not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(x in ast.unparse(n) for x in ['torch', 'gsplat']) for n in ast.walk(tree)), 'New scripts do not import CUDA renderer/trainer')
        compiled.append(name)
    tree = ast.parse(sources['track_rgbd_affine_icp_fallback.py'])
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    execute = ast.unparse(funcs['execute'])
    require(execute.index("pose_text(output / 'trajectory_estimated.txt'") < execute.index('absolute = np.stack([f.c2w for f in frames])'), 'GT matrix consumption occurs only after final all-estimate export')
    require("if index == 1:" in execute and execute.index("before_deadline()", execute.index("if index == 1:")) < execute.index('affine.register_pair_affine('), 'First actual registration has late deadline guard')
    promotion = [n for n in ast.walk(funcs['execute']) if isinstance(n, ast.If) and ast.unparse(n.test) == "decision['accepted']"]
    require(len(promotion) == 1 and [ast.unparse(n) for n in promotion[0].body] == ['targets = sources'], 'Sensor target promoted on either final acceptance only')
    require("'pair_initial_transform': np.eye(4).tolist()" in execute and "'pair_initial_log_relative_gain': 0.0" in execute and "'pair_initial_bias_intensity': 0.0" in execute, 'Identity/loggain0/bias0 primary initialization preserved')
    driver = ast.parse(sources['run_rgbd_affine_icp_fallback.py'])
    guard_adjacent = False
    for node in ast.walk(driver):
        if hasattr(node, 'body') and isinstance(node.body, list):
            for left, right in zip(node.body, node.body[1:]):
                if isinstance(left, ast.Assign) and isinstance(right, ast.Assign):
                    if ast.unparse(left.value) == 'pipeline.before_deadline()' and isinstance(right.value, ast.Call) and ast.unparse(right.value.func) == 'subprocess.Popen': guard_adjacent = True
    require(guard_adjacent, 'No hash/openlog/progress I/O between deadline guard and Popen')
    old = (ROOT / 'scripts/track_rgbd_icp.py').read_text(encoding='utf-8')
    require('register_pair(source_pyramid, target_pyramid, config)' in old and 'target_pyramid = source_pyramid' in old, 'Frozen original ICP physical source=current,target=immediately previous sensor')
    mock = read(ROOT / 'outputs/rgbd-affine-icp-fallback-driver-cpu-controls-001/metrics.json')
    require(mock['status'] == 'passed_all_mock_fusion_driver_controls' and len(mock['fixtures']) == 11 and mock['real_children_started'] == mock['real_pair_registration_runs'] == mock['GPU_calls'] == 0 and mock['protocol_sha256'] == PROTOCOL_SHA and mock['source_sha256'] == protocol['source_sha256'] and mock['checker_sha256'] == NEW_SOURCES['check_rgbd_affine_icp_fallback.py'], 'Existing actual 11 mock outcomes source/protocol bound; reused without actual children')
    for fixture in mock['fixtures'].values(): require(fixture['status'] == 'passed_mock_control', 'All actual mock cases passed')
    return {'compiled_without_execution': compiled, 'deadline_immediately_before_Popen_AST': guard_adjacent,
            'existing_mock_controls': 11, 'existing_mock_metrics_sha256': digest(ROOT / 'outputs/rgbd-affine-icp-fallback-driver-cpu-controls-001/metrics.json'),
            'manual_full_source_review': ['Primary15c acceptance gates/solver unchanged; only arbitration and physical sensor reference policy added.',
                'Complete primary candidates, levels, iterations and numeric-exception unknown state preserved; fallback originals and final flags retained separately.',
                'Child exit1 stops; started child is waited and collected on interruption; no automatic retry or later child.',
                'Final GT evaluation is postponed; loader/association honestly reads GT fields earlier for association, with no GT matrix pose initialization or decision use.']}


def main():
    output = ROOT / 'outputs/rgbd-affine-icp-fallback-independent-cpu-review-002'; output.mkdir(exist_ok=False)
    (output / 'review_source.py').write_bytes(Path(__file__).read_bytes())
    consumed = {}
    registrations = {'count': 0}
    forbidden_targets = [(subject.affine, 'register_pair_affine'), (subject.affine, 'normalized_system'),
                         (subject.affine.frozen, 'register_pair'), (subject.affine.frozen, 'weighted_system'), (subject.affine.frozen.icp, 'register_pair')]
    saved = [getattr(module, name) for module, name in forbidden_targets]
    def forbidden(*args, **kwargs):
        registrations['count'] += 1; raise AssertionError('No actual registration/normal system allowed in independent preparation review')
    for module, name in forbidden_targets: setattr(module, name, forbidden)
    try:
        require(digest(PREP / 'protocol.json') == PROTOCOL_SHA, 'Exact locked protocol')
        protocol = read(PREP / 'protocol.json')
        require(len(protocol['source_sha256']) == 21 and all(protocol['source_sha256'][k] == v for k, v in NEW_SOURCES.items()), 'Exact21 source lock')
        for name, expected in protocol['source_sha256'].items():
            for p in [ROOT / 'scripts' / name, PREP / 'source_snapshot' / name]:
                require(digest(p) == expected, 'Actual/snapshot frozen source hash'); consumed[p.relative_to(ROOT).as_posix()] = expected
        for p in [PREP / 'protocol.json', subject.FULL_PREP / 'protocol.json', subject.FULL_PREP / 'root-independent-full-preflight-review.json', subject.ICP_AUDIT]: consumed[p.relative_to(ROOT).as_posix()] = digest(p)
        require(consumed[subject.FULL_PREP.relative_to(ROOT).as_posix() + '/protocol.json'] == subject.FULL_PROTOCOL_SHA and digest(subject.FULL_PREP / 'root-independent-full-preflight-review.json') == subject.INPUT_REVIEW_SHA and digest(subject.ICP_AUDIT) == subject.ICP_AUDIT_SHA, 'Reused independent input/full ICP audit locks exact')
        require(subject.read_protocol(PREP / 'protocol.json', PROTOCOL_SHA)[1] == protocol, 'Actual protocol parser source/config/deadline gate')
        witnesses = {}; rows = original_candidates(protocol, witnesses)
        structure = source_structure(protocol); controls = synthetic(witnesses)
        cli = []
        for row in protocol['rounds']:
            argv = ['--execute', '--protocol', str(PREP / 'protocol.json'), '--protocol-sha', PROTOCOL_SHA, '--round-id', row['round_id'], '--run', row['run']]
            parsed = subject.arguments(argv)
            require(parsed.execute and parsed.run == row['run'] and parsed.round_id == row['round_id'], 'Actual two child CLI parse only')
            require(not (ROOT / 'outputs' / row['run']).exists(), 'Review precedes actual output creation')
            cli.append({'scene': row['scene'], 'run': parsed.run, 'frames': row['configuration']['frames'], 'start_frame': row['configuration']['start_frame']})
        for key, expected in consumed.items(): require(digest(ROOT / key) == expected, 'Consumed source/input guards unchanged after review')
        require(registrations['count'] == 0 and not any(name in sys.modules for name in ['torch', 'gsplat']), 'Zero registration/normal-system/GPU imports')
        report = {'status': 'passed_independent_fallback_source_and_CPU_review_no_actual_execution',
            'review_source_sha256': digest(Path(__file__)), 'protocol_sha256': PROTOCOL_SHA, 'source_sha256': protocol['source_sha256'],
            'scene_candidate_reconstruction': rows, 'all_relative_rows_reconstructed': sum(r['relative_rows_reconstructed'] for r in rows),
            'source_structure_review': structure, 'synthetic_controls': controls, 'parsed_actual_child_commands_only': cli,
            'consumed_source_and_reused_input_hashes': consumed, 'real_registration_runs': 0, 'real_children_started': 0, 'GPU_calls': 0,
            'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'limitations': ['This review does not execute new full registration, verify its numerical trajectory, or grant execution permission.',
                'Full raw PNG/array association audit is explicitly reused from frozen d815 preflight, not repeated.',
                'Original metrics JSON/association contain GT fields; strictly copied relative payload and sensor-only decisions exclude their values. Opaque original-row hash may cover original evaluation/absolute fields.',
                'Valid held poses make last-accepted and previous-final camera matrices equal; sensor reference IDs, candidate physical direction and target-promotion policy remain essential.',
                'Cached adjacent ICP cannot recover motion already lost during earlier dual rejections; no isolated ICP runtime/FPS or illumination-causality claim.']}
        write(output / 'metrics.json', report)
        print(json.dumps({'status': report['status'], 'review_source_sha256': report['review_source_sha256'], 'metrics_sha256': digest(output / 'metrics.json'), 'relative_rows': report['all_relative_rows_reconstructed'], 'real_registration_runs': 0, 'GPU_calls': 0}))
    except BaseException as error:
        write(output / 'failure.json', {'status': 'independent_review_failure_preserved', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'real_registration_runs': registrations['count'], 'GPU_calls': 0}); raise
    finally:
        for (module, name), value in zip(forbidden_targets, saved): setattr(module, name, value)


if __name__ == '__main__': main()

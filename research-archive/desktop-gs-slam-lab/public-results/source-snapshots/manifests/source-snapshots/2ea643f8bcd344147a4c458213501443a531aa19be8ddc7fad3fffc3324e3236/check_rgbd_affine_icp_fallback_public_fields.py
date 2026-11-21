"""Stdlib-only public field preservation check, never collector/solver import.

--prepare writes SOURCE/negative controls and shallow readonly preflight only.
--check requires a reviewed package and compares existing terminal JSON fields.
No trajectory formulas, registration, GT evaluation or metric fitting occur.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/rgbd-affine-icp-fallback-comparison-001'
PREP = ROOT / 'outputs/rgbd-affine-icp-fallback-public-field-check-preparation-001'
OUTPUT = ROOT / 'outputs/rgbd-affine-icp-fallback-public-field-check-001'
SUMMARY_SHA = '9701897445669ddbd2fa9b11d6140e2d253c9abc4cba4f33dc10619e3e1cbb3f'
COMPLETION_SHA = '75642e95965a5e5c870b9edbadfe7e83b3418483b771bf80a171fd34151038ec'
COLLECTOR_SHA = 'b5f7dba359b65f08c5363bac8f477a5da9e57c1a0c1d6de2acafd8d2787c08fd'
AUDIT_SHA = '69b3856ca466d079c3dc485a6327d668aa88067d161b534b8f4d9c899da87715'
BATCH_SHA = '51f51ab169853f26384e8678e3deae436924c10f142e5d0129c2904a0d39fa07'
BATCH = ROOT / 'outputs/rgbd-affine-icp-fallback-validation-001'
ARMS = [('desk', 549, 'tum-desk-affine-icp-fallback-full549-001', 'f4b6cef8240f96054163dc3ac4f354cc00bdf98a5f484569254b3b6b5bb35b3d'),
        ('xyz', 748, 'tum-xyz-affine-icp-fallback-full748-001', 'c60f3b5827bccbee82996cc54fcafa09be6c8c91107c8af133ebe44ec933569c')]
ERRORS = {'raw_translation_error_m': ('per_frame_raw_translation_error_m', False),
          'SE3_ATE_translation_error_m': ('per_frame_se3_aligned_ate_translation_error_m', False),
          'adjacent_RPE_translation_m': ('per_interval_rpe_translation_error_m', True),
          'adjacent_RPE_rotation_deg': ('per_interval_rpe_rotation_error_deg', True)}
SCALARS = ('raw_anchored_translation_rmse_m', 'se3_aligned_ate_translation_rmse_m', 'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_deg', 'raw_rotation_rmse_deg')
CANDIDATE_KEYS = ('frame_id', 'primary_reference_frame_id', 'primary', 'fallback', 'primary_current_sensor', 'primary_reference_sensor', 'pair_initial_transform', 'pair_initial_log_relative_gain', 'pair_initial_bias_intensity', 'brightness_accumulated_across_pairs')
HISTORY_ENCODING = 'Python json.dumps ensure_ascii=False sort_keys=True separators=(comma,colon) allow_nan=False UTF8; no trailing newline'
HISTORY_SCOPE = 'The full primary iteration_history, including trials, stays in the original local pair/candidate and JSONL; not copied into this page.'


def require(condition, label):
    if not condition: raise ValueError(label)


def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
def equal(actual, expected, label): require(canonical(actual) == canonical(expected), 'Typed persisted JSON mismatch: ' + label)
def bytes_sha(raw): return hashlib.sha256(raw).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''): h.update(piece)
    return h.hexdigest()


def unique_pairs(items):
    d = {}
    for k, v in items:
        require(k not in d, 'Duplicate JSON key'); d[k] = v
    return d


def decode(raw):
    def bad(value): raise ValueError('Nonfinite JSON constant ' + value)
    def finite(value):
        x = float(value); require(math.isfinite(x), 'Finite JSON float'); return x
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=bad, parse_float=finite)


def write(path, value):
    raw = canonical(value) + b'\n'; require(len(raw) < 1024 * 1024, 'New checker artifact strictly below1MiB')
    with Path(path).open('xb') as stream: stream.write(raw)


class Files:
    def __init__(self): self.bindings = {}
    def path(self, relative):
        require(not Path(relative).is_absolute(), 'Original relative file path required')
        path = (ROOT / relative).resolve(); require(path.is_relative_to(ROOT) and path.is_file(), 'Existing original file in lab'); return path
    def watch(self, path, expected=None, size=None):
        path = Path(path).resolve(); require(path.is_relative_to(ROOT), 'File remains in lab')
        binding = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        require(expected is None or binding['sha256'] == expected, 'Frozen file SHA ' + path.name)
        require(size is None or binding['bytes'] == size, 'Frozen byte count ' + path.name)
        key = path.relative_to(ROOT).as_posix()
        if key in self.bindings: equal(binding, self.bindings[key], 'Repeated original file read')
        self.bindings[key] = binding; return binding
    def read(self, path, expected=None, size=None):
        self.watch(path, expected, size); return decode(Path(path).read_bytes())
    def recheck(self):
        for rel, b in list(self.bindings.items()): self.watch(self.path(rel), b['sha256'], b['bytes'])


def expected_row(frame, sensor, trajectory, index, run, pair_binding, candidate_binding):
    copied = dict(frame); history_binding = None
    if 'primary' in frame:
        primary = dict(frame['primary']); history = primary.pop('iteration_history'); copied['primary'] = primary
        history_binding = {'rows': len(history), 'line_search_trials': sum(len(step.get('line_search_trials', [])) for step in history),
                           'sha256': bytes_sha(canonical(history)), 'encoding': HISTORY_ENCODING, 'scope': HISTORY_SCOPE}
    else: require(index == 0, 'Only anchor lacks primary attempt')
    return {'frame': copied, 'sensor_metadata': sensor,
            'stored_errors': {name: None if interval and index == 0 else trajectory[key][index - int(interval)] for name, (key, interval) in ERRORS.items()},
            'primary_history_local_evidence': history_binding,
            'immutable_local_evidence': {'pair_path': f'outputs/{run}/pairs/{index:04d}.json', 'pair_binding': pair_binding,
                                       'candidate_path': f'outputs/{run}/candidates/{index:04d}.json' if index else None, 'candidate_binding': candidate_binding}}


def validate_page(page, entry, scene, page_index, next_id):
    require(page['scene'] == scene and page['page_index'] == page_index and page['status'] == 'all_final_candidate_frame_fields_retained_history_bound_locally', 'Page identity and actual final-candidate scope')
    equal(page['stored_error_mapping'], {k: list(v) for k, v in ERRORS.items()}, 'Four stored arrays and interval offsets')
    require(page['RPE_row_convention'] == 'row i contains original interval i-1 to i; anchor null', 'Explicit anchor RPE null')
    n = len(page['rows']); require(0 < n <= 200 and n == entry['rows'] and page['first_frame_id'] == entry['first_frame_id'] == next_id and page['last_frame_id'] == entry['last_frame_id'] == next_id + n - 1, 'Contiguous complete page boundaries')
    require([r['frame']['frame_id'] for r in page['rows']] == list(range(next_id, next_id + n)), 'No dropped, repeated or reordered page frame')
    return n


def controls():
    negatives = {}
    def reject(name, fn):
        try: fn()
        except (ValueError, KeyError): negatives[name] = 'rejected_as_required'
        else: raise AssertionError('Negative control accepted ' + name)
    for name, a, b in [('int_vs_float', 1, 1.), ('bool_vs_integer', True, 1), ('null_vs_zero', None, 0), ('negative_zero', -0., 0.), ('missing_field', {'a': 1}, {}), ('lost_fallback_flag', {'flags': ['a']}, {'flags': []})]: reject(name, lambda a=a, b=b: equal(a, b, 'Synthetic typed negative'))
    for name, raw in [('duplicate_JSON_key', b'{"a":1,"a":2}'), ('nonfinite_JSON', b'{"a":NaN}'), ('overflow_JSON_float', b'{"a":1e400}')]: reject(name, lambda raw=raw: decode(raw))
    trajectory = {key: ([.2] if interval else [.1, .2]) for key, interval in ERRORS.values()}
    source = {'frame_id': 0, 'accepted': True}; expected = expected_row(source, {'frame_id': 0}, trajectory, 0, 'synthetic', {'bytes': 1, 'sha256': '0' * 64}, None)
    equal(expected['stored_errors']['adjacent_RPE_translation_m'], None, 'Synthetic anchor null positive')
    altered = dict(expected); altered['stored_errors'] = {**expected['stored_errors'], 'adjacent_RPE_translation_m': 0.}
    reject('anchor_RPE_invented_zero', lambda: equal(altered, expected, 'Synthetic anchor exact subset'))
    page = {'status': 'all_final_candidate_frame_fields_retained_history_bound_locally', 'scene': 'fixture', 'page_index': 0, 'rows': [expected], 'first_frame_id': 0, 'last_frame_id': 0, 'stored_error_mapping': {k: list(v) for k, v in ERRORS.items()}, 'RPE_row_convention': 'row i contains original interval i-1 to i; anchor null'}
    entry = {'rows': 1, 'first_frame_id': 0, 'last_frame_id': 0}; validate_page(page, entry, 'fixture', 0, 0)
    reject('page_gap', lambda: validate_page(page, entry, 'fixture', 0, 1))
    reject('page_mislabel', lambda: validate_page(page, entry, 'other', 0, 0))
    reject('page_missing_row', lambda: validate_page({**page, 'rows': []}, entry, 'fixture', 0, 0))
    return {'status': 'passed_typed_JSON_and_page_negative_controls', 'negative_controls': negatives, 'actual_full_field_checks': 0, 'new_metric_calculations': 0, 'registration_calls': 0, 'GPU_calls': 0}


def prepare():
    PREP.mkdir(exist_ok=False); (PREP / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        files = Files(); summary = files.read(PUBLIC / 'summary.json', SUMMARY_SHA); completion = files.read(PUBLIC / 'completion.json', COMPLETION_SHA)
        files.watch(PUBLIC / 'source_script.py', COLLECTOR_SHA); files.watch(ROOT / 'scripts/summarize_rgbd_affine_icp_fallback_results.py', COLLECTOR_SHA)
        require(summary['status'] == 'completed_after_actual_independent_hybrid_result_audit' and completion['status'] == 'completed' and summary['actual_independent_audit_sha256'] == AUDIT_SHA, 'Shallow terminal public artifact readiness')
        files.watch(BATCH / 'independent_affine_icp_fallback_result_audit.json', AUDIT_SHA); files.watch(BATCH / 'metrics.json', BATCH_SHA)
        for _, _, run, digest in ARMS: files.watch(ROOT / 'outputs' / run / 'metrics.json', digest)
        require(len(completion['files']) == 21 and len(summary['input_hash_pages']) == 6, '21 public result manifests plus one SOURCE snapshot; six input-hash pages')
        for entry in completion['files']: files.watch(PUBLIC / entry['path'], entry['sha256'], entry['bytes']); require(entry['bytes'] < 1024 * 1024, 'Every predeclared public JSON below1MiB')
        checked = controls(); write(PREP / 'CPU_controls.json', checked)
        compile(Path(__file__).read_text(encoding='utf-8'), str(Path(__file__)), 'exec')
        imports = [ast.unparse(n) for n in ast.walk(ast.parse(Path(__file__).read_text(encoding='utf-8'))) if isinstance(n, (ast.Import, ast.ImportFrom))]
        require(not any(any(bad in n for bad in ['collector', 'summarize', 'numpy', 'scipy', 'torch', 'gsplat', 'track_', 'tum_rgbd']) for n in imports), 'Stdlib-only independent field checker imports')
        package = {'status': 'prepared_stdlib_public_field_checker_no_actual_execution', 'checker_source_sha256': sha(Path(__file__)), 'CPU_controls_sha256': sha(PREP / 'CPU_controls.json'), 'readonly_shallow_preflight_file_locks': files.bindings,
            'root_source_review_expected_status': 'passed_root_hybrid_public_field_checker_source_review', 'root_source_review_relative_path': (PREP / 'root_source_review.json').relative_to(ROOT).as_posix(),
            'output_relative_path': OUTPUT.relative_to(ROOT).as_posix(), 'expected_frames': 1297, 'expected_primary_steps': 33900, 'expected_primary_trials': 35746,
            'scope': 'Exact persisted JSON field/hash preservation only; no new source experiment audit, registration, trajectory formula or GT evaluation.', 'actual_full_field_checks': 0, 'GPU_calls': 0}
        write(PREP / 'protocol.json', package); files.recheck()
        print(json.dumps({'status': package['status'], 'checker_source_sha256': package['checker_source_sha256'], 'protocol_sha256': sha(PREP / 'protocol.json'), 'CPU_controls_sha256': package['CPU_controls_sha256'], 'shallow_preflight_files': len(files.bindings), 'actual_full_field_checks': 0}))
    except BaseException as error:
        write(PREP / 'failure.json', {'status': 'preparation_failure_retained', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'actual_full_field_checks': 0}); raise


def check(protocol_sha):
    require(sha(PREP / 'protocol.json') == protocol_sha, 'Exact reviewed checker package')
    package = decode((PREP / 'protocol.json').read_bytes()); require(package['checker_source_sha256'] == sha(Path(__file__)), 'Frozen checker SOURCE')
    review = decode((ROOT / package['root_source_review_relative_path']).read_bytes())
    require(review['status'] == package['root_source_review_expected_status'] and review['protocol_sha256'] == protocol_sha and review['checker_source_sha256'] == package['checker_source_sha256'], 'Explicit root actual-field-check source review')
    OUTPUT.mkdir(exist_ok=False); (OUTPUT / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    files = Files()
    try:
        for rel, binding in package['readonly_shallow_preflight_file_locks'].items(): files.watch(files.path(rel), binding['sha256'], binding['bytes'])
        files.watch(Path(__file__), package['checker_source_sha256']); files.watch(PREP / 'protocol.json', protocol_sha); files.watch(ROOT / package['root_source_review_relative_path'])
        summary = files.read(PUBLIC / 'summary.json', SUMMARY_SHA); completed = files.read(PUBLIC / 'completion.json', COMPLETION_SHA)
        batch = files.read(BATCH / 'metrics.json', BATCH_SHA); audit = files.read(BATCH / 'independent_affine_icp_fallback_result_audit.json', AUDIT_SHA)
        require(batch['status'] == 'all_two_predeclared_CPU_fusion_rounds_completed' and [x['returncode'] for x in batch['run_records']] == [0, 0] and audit['status'] == 'passed_independent_terminal_affine_ICP_fallback_result_audit', 'Existing actual terminal experiment audit reused')
        equal(summary['CPU_child_returncodes'], [0, 0], 'Public child returncodes')
        equal(summary['source_sha256_original_hybrid'], batch['source_sha256'], 'Frozen experiment source dictionary')
        require(summary['source_sha256'] == completed['source_sha256'] == COLLECTOR_SHA and summary['batch_metrics_sha256'] == BATCH_SHA and summary['actual_independent_audit_sha256'] == AUDIT_SHA and summary['registration_calls'] == summary['new_metric_calculations'] == summary['GPU_calls'] == 0, 'Source/audit/no-new-calculation public scope')
        completion_entries = completed['files'] + [completed['source_snapshot']]
        require(len(completion_entries) == 22 and len({e['path'] for e in completion_entries}) == 22, 'All22 manifested result/source files unique')
        equal(completed['source_snapshot'], {'path': 'source_script.py', 'sha256': COLLECTOR_SHA, 'bytes': (PUBLIC / 'source_script.py').stat().st_size}, 'Collector source snapshot identity')
        actual_names = {p.name for p in PUBLIC.iterdir() if p.is_file()}
        require(actual_names == {e['path'] for e in completion_entries} | {'completion.json'}, 'No hidden missing/extra completion file')
        for entry in completion_entries: files.watch(PUBLIC / entry['path'], entry['sha256'], entry['bytes']); require(entry['bytes'] < 1024 * 1024, 'Strict exclusive public size bound')
        public_manifest = {e['path']: e for e in completed['files']}
        input_bindings = {}; input_order = []
        for index, entry in enumerate(summary['input_hash_pages']):
            require(entry['path'] == f'input-hashes-{index:02d}.json', 'All six input pages contiguous')
            equal(entry, public_manifest[entry['path']], 'Input page completion binding')
            rows = files.read(PUBLIC / entry['path'], entry['sha256'], entry['bytes']); require(0 < len(rows) <= 500, 'Input page declared split boundary')
            for rel, binding in rows.items():
                require(rel not in input_bindings, 'Original input path occurs exactly once'); require(set(binding) == {'bytes', 'sha256'}, 'Input binding schema')
                input_bindings[rel] = binding; input_order.append(rel); files.watch(files.path(rel), binding['sha256'], binding['bytes'])
        require(input_order == sorted(input_order) and len(input_bindings) == summary['consumed_files_rehashed_unchanged'] == 2672, 'Every original2672 input paths fresh-hashed without omission')
        old_full = files.read(ROOT / 'outputs/rgbd-affine-full-comparison-001/summary.json', summary['old_full8D_summary_sha256'])
        old_icp = files.read(ROOT / 'outputs/tum-online-v4-full-sequence-recovery-summary-001/summary.json', summary['old_fullICP_summary_sha256'])
        scene_results = []; all_rows = all_steps = all_trials = 0; seen_pages = []
        for arm_index, (scene, n, run, digest) in enumerate(ARMS):
            folder = ROOT / 'outputs' / run; m = files.read(folder / 'metrics.json', digest); original_rows = m['frame_metrics']; metadata = m['selected_sensor_metadata']
            require(len(original_rows) == len(metadata) == n and [r['frame_id'] for r in original_rows] == list(range(n)), 'Every original frame retained for comparison')
            for key, interval in ERRORS.values(): require(len(m['trajectory'][key]) == n - int(interval), 'Exact complete source error-array denominator')
            scene_summary = files.read(PUBLIC / (scene + '-summary.json')); equal(scene_summary, summary['scenes'][scene], 'Per-scene and root-summary full fields')
            public = scene_summary['new_hybrid']; manifest = files.read(folder / 'pair_manifest.json')
            next_id = steps = trials = 0; primary_flags = Counter(); final_flags = Counter(); selections = Counter(); reject_primary = []; use_fallback = []; reject_final = []
            jsonl_path = folder / 'iterations.jsonl'; files.watch(jsonl_path, public['local_iterations_jsonl']['sha256'], public['local_iterations_jsonl']['bytes'])
            equal(public['local_iterations_jsonl']['path'], jsonl_path.relative_to(ROOT).as_posix(), 'Original JSONL path binding')
            with jsonl_path.open(encoding='utf-8') as stream:
                for page_index, entry in enumerate(public['public_frame_pages']):
                    name = entry['path']; require(name == f'{scene}-frames-{page_index:02d}.json', 'Scene page names exact consecutive order')
                    equal(entry, public_manifest[name], 'Every scene page completion metadata')
                    page = files.read(PUBLIC / name, entry['sha256'], entry['bytes']); count = validate_page(page, entry, scene, page_index, next_id); seen_pages.append(name)
                    for offset, exported in enumerate(page['rows']):
                        index = next_id + offset; frame = original_rows[index]; pb = manifest[f'pairs/{index:04d}.json']; cb = manifest[f'candidates/{index:04d}.json'] if index else None
                        pair_path = folder / f'pairs/{index:04d}.json'; equal(files.read(pair_path, pb['sha256'], pb['bytes']), frame, 'Every full original immutable pair equals original metric frame')
                        if index:
                            candidate_path = folder / f'candidates/{index:04d}.json'; equal(files.read(candidate_path, cb['sha256'], cb['bytes']), {k: frame[k] for k in CANDIDATE_KEYS}, 'All immutable original primary/fallback candidate fields')
                        expected = expected_row(frame, metadata[index], m['trajectory'], index, run, pb, cb); equal(exported, expected, 'All frame fields/full fallback/primary except explicitly omitted history/sensor/fourerrors')
                        selections[frame['selection']] += 1; final_flags.update(frame['failure_flags'])
                        if frame['primary_rejected']: reject_primary.append(index)
                        if frame['fallback_selected']: use_fallback.append(index)
                        if frame['failure_flags']: reject_final.append(index)
                        if index:
                            history = frame['primary']['iteration_history']; steps += len(history); trials += sum(len(s.get('line_search_trials', [])) for s in history); primary_flags.update(frame['primary']['failure_flags'])
                            for item in history:
                                line = stream.readline(); require(bool(line), 'Original JSONL includes every bound local history row')
                                equal(decode(line), {'frame_id': index, 'primary_reference_frame_id': frame['primary_reference_frame_id'], **item}, 'Original JSONL actual rows equal original pair/candidate history')
                    next_id += count
                require(stream.read().strip() == '', 'No extra original JSONL history rows')
            require(next_id == n and [p['first_frame_id'] for p in public['public_frame_pages']][0] == 0 and public['public_frame_pages'][-1]['last_frame_id'] == n - 1, 'No page omissions across full sequence')
            terminal = batch['run_records'][arm_index]
            expected_current = {'run': run, 'metrics_sha256': digest, 'frames': n, 'tracked_pairs': n - 1, 'configuration': m['configuration'], 'source_sha256': m['source_sha256'], 'status': m['status'], 'selection_counts': dict(selections), 'primary_rejected_frame_ids': reject_primary, 'fallback_selected_frame_ids': use_fallback, 'final_rejected_frame_ids': reject_final, 'primary_failure_frame_count': len(reject_primary), 'fallback_selected_frame_count': len(use_fallback), 'final_failure_frame_count': len(reject_final), 'primary_failure_flag_counts': dict(primary_flags), 'final_failure_flag_counts': dict(final_flags), 'trajectory_scalars': {k: m['trajectory'][k] for k in SCALARS}, 'endpoint_raw_translation_m': m['trajectory']['per_frame_raw_translation_error_m'][-1], 'alignment_scale': m['trajectory']['alignment_scale'], 'last_final_reference_id': m['last_final_reference_id'], 'last_frame': {k: v for k, v in original_rows[-1].items() if k not in ('primary', 'fallback')}, 'last_sensor_metadata': metadata[-1], 'observed_wall_seconds_including_cached_ICP_input_IO': m['observed_wall_seconds'], 'timing_scope': m['timing_scope'], 'started_utc': terminal['started_utc'], 'ended_utc': terminal['ended_utc'], 'returncode': terminal['returncode'], 'local_primary_history_rows': steps, 'local_primary_trial_count': trials, 'local_iterations_jsonl': {'path': jsonl_path.relative_to(ROOT).as_posix(), **files.bindings[jsonl_path.relative_to(ROOT).as_posix()]}, 'public_frame_pages': public['public_frame_pages']}
            equal(public, expected_current, 'Every stored scene summary scalar/final SENSOR/decision/timing field exact')
            arm = next(a for a in old_full['scene_results'][scene]['arms'] if a['model_dimension'] == 8)
            for key in ('run', 'metrics_sha256', 'frames', 'trajectory_scalars', 'endpoint_raw_translation_m', 'failure_frame_count', 'last_camera', 'configuration', 'observed_wall_seconds_including_IO'): equal(scene_summary['old_full8D'][key], arm[key], 'Reused original8D field ' + key)
            baseline = old_icp['scenes'][scene]['baseline']; old_metric = baseline['metrics']; mapping = scene_summary['old_fullICP']['scalar_key_mapping_from_audited_summary']
            equal(scene_summary['old_fullICP']['trajectory_scalars'], {new: old_metric[old] for new, old in mapping.items()}, 'Old ICP audited-summary scalar mapping, no recomputation')
            for key, value in [('run', baseline['run']), ('frames', n), ('failure_frame_count', baseline['failure_frame_count']), ('stored_original_timing_s', baseline['timing_s']), ('endpoint_raw_translation_m', old_metric['raw_endpoint_translation_error_m'])]: equal(scene_summary['old_fullICP'][key], value, 'Reused original ICP field ' + key)
            all_rows += n; all_steps += steps; all_trials += trials
            scene_results.append({'scene': scene, 'frame_rows_exactly_preserved': n, 'public_frame_pages': len(public['public_frame_pages']), 'original_history_rows_bound': steps, 'original_history_trials_bound': trials, 'primary_rejection_count': len(reject_primary), 'fallback_selection_count': len(use_fallback), 'final_rejection_count': len(reject_final)})
        require((all_rows, all_steps, all_trials) == (1297, 33900, 35746) and summary['all_new_frame_rows'] == all_rows and summary['all_new_nonanchor_pairs'] == all_rows - 2 and summary['all_local_primary_history_rows'] == all_steps and summary['all_local_primary_trial_count'] == all_trials, 'All original full-frame/history/trial counts preserved')
        require({e['path'] for e in completed['files']} == set(seen_pages) | {p['path'] for p in summary['input_hash_pages']} | {'desk-summary.json', 'xyz-summary.json', 'summary.json'}, 'Every completion public object classified and checked')
        files.recheck()
        result = {'status': 'passed_independent_stdlib_public_hybrid_field_preservation', 'checker_source_sha256': sha(Path(__file__)), 'checker_package_protocol_sha256': protocol_sha, 'collector_source_sha256': COLLECTOR_SHA, 'public_summary_sha256': SUMMARY_SHA, 'public_completion_sha256': COMPLETION_SHA, 'reused_actual_experiment_audit_sha256': AUDIT_SHA, 'frame_rows_exactly_preserved': all_rows, 'nonanchor_pairs': all_rows - 2, 'original_history_rows_bound': all_steps, 'original_history_trials_bound': all_trials, 'completion_result_file_count': 21, 'completion_source_snapshot_file_count': 1, 'manifested_total_files_excluding_completion_self': 22, 'input_hash_pages': 6, 'original_input_bindings_fresh_hashed': len(input_bindings), 'all_consumed_files_unchanged': len(files.bindings), 'scene_results': scene_results, 'registration_calls': 0, 'new_GT_or_trajectory_metric_calculations': 0, 'GPU_calls': 0, 'scope': 'Typed canonical persisted JSON equality, complete original pair/candidate/history/JSONL bindings and fresh file hashes only; no new solver/GT formula or real experiment.'}
        write(OUTPUT / 'metrics.json', result)
        print(json.dumps({'status': result['status'], 'metrics_sha256': sha(OUTPUT / 'metrics.json'), 'frames': all_rows, 'steps': all_steps, 'trials': all_trials, 'consumed_files': len(files.bindings), 'GPU_calls': 0}))
    except BaseException as error:
        write(OUTPUT / 'failure.json', {'status': 'readonly_public_field_check_failure_retained', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'registration_calls': 0, 'GPU_calls': 0}); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--check', action='store_true'); parser.add_argument('--protocol-sha')
    args = parser.parse_args(); require(args.check == (args.protocol_sha is not None), 'Actual check needs reviewed package SHA; prepare must omit it')
    if args.prepare: prepare()
    else: check(args.protocol_sha)


if __name__ == '__main__': main()

"""CPU posthoc validation of the complete V4 seed-study public statistics.

Reads immutable artifacts, never executes the trainer or the statistics author.
Uses Python math/statistics for aggregate arithmetic and explicit tensor byte
hashes for within-seed context comparisons. Existing audits are not rewritten.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from audit_tum_v4_seed_replication import summary_audit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'manifests/tum-online-v4-seed-replication-protocol-001.json'
PROTOCOL_SHA = 'b86496b7731352457e5d86becdd4d06e4709efee794c1ae3d81546cdccc75eb9'
PUBLIC_AUDITOR_SHA = '06aa58f623f1bcf9e6c9bdcd75ff08938c71ac0a60dd4d4c69e41c317c5f9c9e'
KEYS = ('raw_anchored_translation_rmse_m', 'ate_rmse_m_se3_no_scale',
        'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_degrees',
        'raw_anchored_rotation_rmse_degrees')
TABLE_TRANSLATION_KEYS = KEYS[:3]
ICP_KEYS = dict(zip(KEYS, ('raw_anchored_translation_rmse_m',
    'se3_aligned_ate_translation_rmse_m', 'adjacent_rpe_translation_rmse_m',
    'adjacent_rpe_rotation_rmse_deg', 'raw_rotation_rmse_deg')))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def close(value, expected, label, abs_tol=2e-16, rel_tol=2e-13):
    require(math.isfinite(value) and math.isfinite(expected), label + ' finite')
    require(math.isclose(value, expected, abs_tol=abs_tol, rel_tol=rel_tol), label)
    return abs(value - expected)


def tensor_record(tensor):
    # Hash raw contiguous tensor payload, including IEEE signed-zero bits.
    # Shape/dtype are separate compared fields rather than inferred from bytes.
    a = tensor.detach().cpu().contiguous().numpy()
    return {'dtype': str(a.dtype), 'shape': list(a.shape),
            'raw_contiguous_bytes': a.nbytes,
            'raw_contiguous_payload_sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def inventory(p):
    records = []
    for scene, rows in p['existing_seed0_records'].items():
        records.extend({'scene': scene, 'seed': 0, 'weight': row['weight'],
                        'run': row['run'], 'configuration': row['configuration'],
                        'seed0_metrics_sha256': row['metrics_sha256']} for row in rows)
    records.extend({'scene': r['scene'], 'seed': r['configuration']['seed'],
                    'weight': r['configuration']['pose_prior_weight'], 'run': r['run'],
                    'configuration': r['configuration']} for r in p['new_runs'])
    require(len(records) == 12 and len({r['run'] for r in records}) == 12, 'complete inventory')
    require({(r['scene'], r['seed'], r['weight']) for r in records} ==
            {(scene, seed, weight) for scene in ('desk', 'xyz') for seed in (0, 1, 2)
             for weight in (0., .01)}, 'complete factorial grid')
    return records


def raw_context_audit(rows):
    plus = torch.tensor([0.], dtype=torch.float32)
    minus = torch.tensor([-0.], dtype=torch.float32)
    control = {'torch_equal_values': torch.equal(plus, minus),
               'positive_zero': tensor_record(plus), 'negative_zero': tensor_record(minus)}
    require(control['torch_equal_values'] and tensor_record(plus) != tensor_record(minus),
            'signed-zero negative control detects numerical versus byte equality')
    pairs = []
    for scene in ('desk', 'xyz'):
        for seed in (0, 1, 2):
            members = sorted((r for r in rows if r['scene'] == scene and r['seed'] == seed),
                             key=lambda r: r['weight'])
            contexts, hashes = [], []
            for row in members:
                folder = ROOT / 'outputs' / row['run']
                context_path = folder / 'first_tracking_context.pt'
                metadata = read(folder / 'first_tracking_context.json')
                require(metadata['checkpoint_sha256'] == sha(context_path), 'actual context file SHA')
                require(metadata['tracking_optimizer_steps_executed_before_save'] == 0 and
                        metadata['saved_before_first_pose_gradient_probe'], 'context before tracking')
                for field in ('ground_truth_included', 'future_sensor_observations_included',
                              'future_maps_or_trajectories_included'):
                    require(metadata[field] is False, 'context causal field')
                contexts.append(torch.load(context_path, map_location='cpu', weights_only=True))
                hashes.append({'run': row['run'], 'context_pt_sha256': sha(context_path),
                               'context_json_sha256': sha(folder / 'first_tracking_context.json')})
            require(set(contexts[0]['parameters']) == set(contexts[1]['parameters']), 'parameter keys')
            maps = {}
            for key in contexts[0]['parameters']:
                a, b = (c['parameters'][key] for c in contexts)
                left, right = tensor_record(a), tensor_record(b)
                maps[key] = {'prior0': left, 'prior001': right,
                             'dtype_shape_and_raw_bytes_equal': left == right,
                             'torch_equal_numerical_values': torch.equal(a, b),
                             'same_seed_same_order_max_abs_difference': float((a-b).abs().max())}
            inputs = {}
            for key in ('image', 'depth_m', 'valid_depth_mask', 'K', 'initial_w2c',
                        'prior_reference_w2c', 'icp_prefix_c2w_float64'):
                left, right = (tensor_record(c[key]) for c in contexts)
                inputs[key] = {'prior0': left, 'prior001': right,
                               'dtype_shape_and_raw_bytes_equal': left == right}
                require(left == right, 'same-seed common sensor/reference/prefix/source bytes ' + key)
            require(contexts[0]['source_frame_ids_per_gaussian'] == contexts[1]['source_frame_ids_per_gaussian'],
                    'same-seed source ID lists')
            pairs.append({'scene': scene, 'seed': seed, 'context_files': hashes,
                          'warmup_map': maps, 'shared_inputs': inputs,
                          'all_warmup_map_payloads_identical': all(x['dtype_shape_and_raw_bytes_equal'] for x in maps.values()),
                          'all_shared_sensor_reference_prefix_tensor_bytes_identical': True,
                          'source_id_lists_exactly_equal': True})
    return {'status': 'passed', 'audit_source_sha256': sha(Path(__file__)),
            'protocol_sha256': PROTOCOL_SHA, 'pairs': pairs, 'signed_zero_negative_control': control,
            'scope': 'Exact dtype, shape and raw C-contiguous payload SHA for six same-seed pairs. Hashes complement the old torch.equal numerical checks; no historical record is rewritten.',
            'limits': ['Same-seed array order only. Cross-seed arrays can reorder or subsample physical points.',
                       'Different actual warmup maps make full-pipeline arms observational; seed/config does not establish shared-map causal isolation.']}


def check_moment(saved, values, label):
    require(saved['values'] == values and saved['n'] == len(values), label + ' exact observations')
    expected = {'mean': math.fsum(values) / len(values),
                'sample_standard_deviation_ddof1': statistics.stdev(values) if len(values) > 1 else None,
                'min': min(values), 'max': max(values)}
    errors = {key: close(saved[key], value, label + '/' + key) if value is not None else 0.
              for key, value in expected.items()}
    require('not SEM, CI or significance' in saved['interpretation'], label + ' interpretation')
    return {'recomputed_python_math_statistics': expected, 'maximum_absolute_numeric_difference': max(errors.values())}


def statistics_audit(p, rows):
    folder = ROOT / 'outputs/tum-online-v4-seed-study-summary-001'
    report = read(folder / 'metrics.json')
    require(report['script_sha256'] == sha(ROOT / 'scripts/summarize_v4_seed_replication.py'), 'statistics source hash')
    require(report['protocol_sha256'] == PROTOCOL_SHA and report['all_metrics_posthoc_only'], 'statistics protocol/posthoc')
    require(report['status'] == 'all_predeclared_completed_runs_described' and
            (report['predeclared_runs'], report['new_attempts'], report['reused_existing_seed0_runs'],
             report['complete_runs_in_statistics'], report['process_incomplete_or_failed_runs']) == (12,8,4,12,0), 'complete denominators')
    require([r['run'] for r in report['runs']] == [r['run'] for r in rows], 'no run/seed selected or dropped')
    originals = {}
    for row, public in zip(rows, report['runs']):
        path = ROOT / 'outputs' / row['run'] / 'metrics.json'
        original = read(path); originals[row['run']] = original
        require(public['configuration'] == row['configuration'] == original['configuration'], 'full original config')
        require(public['metrics_sha256'] == sha(path), 'run original metric SHA')
        if row['seed'] == 0:
            require(row['seed0_metrics_sha256'] == sha(path), 'reused seed0 immutable SHA')
        require(public['scene'] == row['scene'] and public['seed'] == row['seed'] and
                public['weight'] == row['weight'], 'labels from declared protocol')
        require(original['status'] in ('completed', 'completed_with_tracking_or_mapping_failures') and
                public['status'] == original['status'] and public['included_in_statistics'] is True, 'terminal run kept')
        require(original['frames_processed'] == 100 and original['selected_frame_ids'] == list(range(200,300)) and
                len(original['frame_metrics']) == len(original['trajectory']['per_view']) == 100 and
                len(original['trajectory']['adjacent_rpe']) == 99, 'all frames and pairs retained')
        failures = sum(bool(f['failure_flags']) for f in original['frame_metrics'])
        require(not original['frame_metrics'][0]['failure_flags'] and
                public['failure_frame_count'] == original['failure_frame_count'] == failures and
                public['tracked_frame_denominator_excludes_fixed_first'] == 99 and
                public['failure_frame_fraction'] == failures/99. and
                public['failure_flags'] == dict(Counter(v for f in original['frame_metrics'] for v in f['failure_flags'])), 'failures retained / denominator99')
        for name in ('pose_gradient_checks', 'initial_point_count', 'final_point_count',
                     'online_loop_seconds_excludes_warmup', 'total_seconds_before_metrics_export'):
            destination = 'probe' if name == 'pose_gradient_checks' else name
            require(public[destination] == original[name] and type(public[destination]) is type(original[name]), 'run field ' + name)
        require(public['actual_first_context_sha256'] == original['first_tracking_context']['checkpoint_sha256'], 'context SHA retained')
        require(public['trajectory_metrics'] == {key: original['trajectory'][key] for key in KEYS}, 'original trajectory scalar fidelity')
    moment_checks, tables = {}, []
    for scene in ('desk', 'xyz'):
        for weight in (0., .01):
            name = f'{scene}_prior{weight:g}'; group = report['groups'][name]
            selected = sorted((r for r in rows if r['scene'] == scene and r['weight'] == weight), key=lambda r:r['seed'])
            require(group['seeds'] == [0,1,2] and group['runs'] == [r['run'] for r in selected] and
                    group['scene'] == scene and group['prior_weight'] == weight and group['complete_runs'] == 3 and
                    group['nonanchor_frame_count'] == 297 and
                    group['failure_frame_counts_by_seed'] == [originals[r['run']]['failure_frame_count'] for r in selected] and
                    group['failed_nonanchor_frames'] == sum(group['failure_frame_counts_by_seed']), 'group membership / denominator297')
            moment_checks[name] = {}
            for key in KEYS:
                values = [originals[r['run']]['trajectory'][key] for r in selected]
                moment_checks[name][key] = check_moment(group['metrics'][key], values, name+'/'+key)
            tables.append({'scene':scene,'weight':weight,'n':3,
                'translation_m_and_mm': {key:{'mean_m':group['metrics'][key]['mean'],
                    'sample_sd_m':group['metrics'][key]['sample_standard_deviation_ddof1'],
                    'mean_mm':group['metrics'][key]['mean']*1000.,
                    'sample_sd_mm':group['metrics'][key]['sample_standard_deviation_ddof1']*1000.} for key in TABLE_TRANSLATION_KEYS},
                'rotation_degrees':{key:{'mean_degrees':group['metrics'][key]['mean'],
                    'sample_sd_degrees':group['metrics'][key]['sample_standard_deviation_ddof1']} for key in KEYS[3:]}})
    delta_checks, icp_checks = {}, {}
    for scene in ('desk', 'xyz'):
        delta_checks[scene] = {}
        baseline = report['ICP_reference'][scene]
        standalone_folder = ROOT / 'outputs' / ('tum-icp-transfer100-001' if scene == 'desk' else 'tum-xyz-icp-transfer100-001')
        standalone = read(standalone_folder/'metrics.json')['trajectory']
        aliases = {}
        for key in KEYS:
            prefix_values = [originals[r['run']]['icp_prefix_trajectory_evaluation'][key] for r in rows if r['scene'] == scene]
            require(all(v == baseline['metrics'][key] for v in prefix_values), 'same ICP metrics across six runs and two groups')
            error = close(baseline['metrics'][key], standalone[ICP_KEYS[key]], 'standalone ICP scalar '+scene+'/'+key, abs_tol=2e-14)
            aliases[key] = {'standalone_key':ICP_KEYS[key], 'online_prefix_value':baseline['metrics'][key],
                            'standalone_value':standalone[ICP_KEYS[key]], 'absolute_difference':error}
            off = report['groups'][f'{scene}_prior0']['metrics'][key]['values']
            on = report['groups'][f'{scene}_prior0.01']['metrics'][key]['values']
            delta_checks[scene][key] = {'on_minus_off':check_moment(report['paired_differences'][scene]['metrics'][key],
                [b-a for a,b in zip(off,on)], scene+'/'+key+'/paired'),
                'on_minus_ICP_reference':check_moment(report['paired_differences'][scene]['on_minus_ICP_reference'][key],
                [v-baseline['metrics'][key] for v in on], scene+'/'+key+'/vsICP')}
        icp_checks[scene] = {'standalone_metric_sha256':sha(standalone_folder/'metrics.json'), 'aliases':aliases,
                            'all_six_online_prefix_metric_dictionaries_exactly_equal':True,
                            'two_statistic_group_references_share_one_baseline':True}
    ledger_path = ROOT / 'outputs/tum-online-v4-seed-replication-001/metrics.json'
    ledger = read(ledger_path); deadline=dt.datetime.fromisoformat(p['deadline_beijing']); previous_end=None
    require(ledger['status'] == 'all_eight_predeclared_runs_completed' and ledger['GPU_runs_started'] == 8 and
            ledger['current_run'] is None and ledger['protocol_sha256'] == PROTOCOL_SHA and len(ledger['run_records']) == 8, 'terminal serial process ledger')
    ledger_rows=[]
    for declared, record in zip(p['new_runs'], ledger['run_records']):
        start=dt.datetime.fromisoformat(record['started_at']['beijing']); end=dt.datetime.fromisoformat(record['ended_at']['beijing'])
        require(start < deadline and end >= start and (previous_end is None or start >= previous_end), 'recorded serial deadline times')
        previous_end=end
        metric_path=ROOT/'outputs'/declared['run']/'metrics.json'; log_path=ROOT/declared['log_relative_path']
        require(record['run'] == declared['run'] and record['order'] == declared['order'] and
                record['configuration'] == declared['configuration'] and record['returncode'] == 0 and
                record['status'] == originals[record['run']]['status'] and
                record['metrics_sha256'] == sha(metric_path) and record['log_sha256'] == sha(log_path), 'ledger record hash/config/exit')
        ledger_rows.append({'run':record['run'],'returncode':record['returncode'],
                            'start_beijing':record['started_at']['beijing'],'end_beijing':record['ended_at']['beijing'],
                            'metrics_sha256':sha(metric_path),'log_sha256':sha(log_path)})
    require(report['process_ledger']['batch_metrics_sha256'] == sha(ledger_path) and
            report['process_ledger']['returncodes'] == [0]*8 and
            report['process_ledger']['driver_sha256'] == sha(ROOT/'scripts/run_v4_seed_replication.py') == p['driver_sha256'], 'public process ledger bindings')
    return {'status':'passed','audit_source_sha256':sha(Path(__file__)),
            'report_metrics_sha256':sha(folder/'metrics.json'), 'statistics_source_sha256':report['script_sha256'],
            'protocol_sha256':PROTOCOL_SHA,'12_original_run_metric_fields_exact':True,
            'complete_runs':12,'new_process_records':8,'groups_with_seeds012':4,
            'python_math_statistics_moments':moment_checks,'matching_seed_paired_deltas':delta_checks,
            'ICP_references':icp_checks,'ledger':{'batch_metrics_sha256':sha(ledger_path),'rows':ledger_rows,
                'scope':'Recorded process timestamps, return codes, source/config/metrics/log hashes; not an OS scheduler timing proof.'},
            'm_to_mm_table':tables,'unit_rule':'Translation m * 1000 = mm; rotation remains degrees. Sample SD is descriptive ddof1, not SEM/CI/significance.',
            'limits':['CPU posthoc artifact/statistical audit; no rerender, registration, trainer, GPU or GT-based selection.',
                      'Individual trajectory audits reuse audit_pose_arrays, also used by ICP; not a second independent ICP formula.',
                      'Actual full-pipeline warmup maps differ. n=3 in two same-sensor sequences does not establish IID, significance or broad generalization.']}


def main():
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'fixed protocol SHA')
    require(sha(ROOT/'scripts/audit_tum_v4_seed_replication.py') == PUBLIC_AUDITOR_SHA, 'frozen generic/public auditor SHA')
    p=read(PROTOCOL)
    for name,digest in p['source_sha256'].items():
        require(sha(ROOT/name) == digest, 'frozen source '+name)
    rows=inventory(p)
    public=[]
    for row in p['new_runs']:
        result=summary_audit(row['run'])
        require(result['status'] == 'passed','public summary available and exact')
        public.append(result)
    raw=raw_context_audit(rows)
    report=statistics_audit(p,rows)
    require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    report['CUDA_initialized']=False
    for result in public:
        write(ROOT/'outputs'/result['run']/'independent_seed_public_summary_audit.json',result)
    write(ROOT/'outputs/seed-replication-public-summary-audit.json',
          {'status':'passed','audit_source_sha256':sha(Path(__file__)),'generic_summary_auditor_sha256':PUBLIC_AUDITOR_SHA,
           'protocol_sha256':PROTOCOL_SHA,'all_eight_public_summaries_original_scalars_types_and_lists_exact':True,'runs':public})
    write(ROOT/'outputs/seed-replication-warmup-raw-bytes-audit.json',raw)
    write(ROOT/'outputs/tum-online-v4-seed-study-summary-001/independent_statistics_audit.json',report)
    print(json.dumps({'status':'passed','public_summaries':len(public),'statistics_complete_runs':12,
                      'raw_byte_pairs':6,'m_to_mm_table':report['m_to_mm_table'],'CUDA_initialized':False},indent=2))


if __name__ == '__main__':
    main()

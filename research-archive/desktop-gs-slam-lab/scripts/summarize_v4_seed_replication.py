"""Descriptive CPU statistics for the predeclared V4 two-sequence seed study.

Never trains, changes an experiment, chooses a weight, or treats pending runs
as observations. All twelve complete runs are required before aggregation;
frame failures stay in each trajectory. Sample SD uses ddof=1, not a CI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'manifests/tum-online-v4-seed-replication-protocol-001.json'
PROTOCOL_SHA = 'b86496b7731352457e5d86becdd4d06e4709efee794c1ae3d81546cdccc75eb9'
KEYS = ('raw_anchored_translation_rmse_m', 'ate_rmse_m_se3_no_scale',
        'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_degrees',
        'raw_anchored_rotation_rmse_degrees')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def moments(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Statistics require nonempty finite scalar observations')
    return {'values': a.tolist(), 'n': len(a), 'mean': float(a.mean()),
            'sample_standard_deviation_ddof1': float(a.std(ddof=1)) if len(a) >= 2 else None,
            'min': float(a.min()), 'max': float(a.max()),
            'interpretation': 'descriptive variation across complete runs; not SEM, CI or significance'}


def cpu_checks() -> dict:
    a = moments([1., 2., 3.])
    assert a['mean'] == 2 and a['sample_standard_deviation_ddof1'] == 1
    assert moments([1.])['sample_standard_deviation_ddof1'] is None
    rejected = False
    try:
        moments([float('nan')])
    except ValueError:
        rejected = True
    assert rejected
    return {'sample_sd_known_three_values': True, 'single_run_sd_none': True,
            'nonfinite_rejected': True, 'CUDA_imported_or_initialized': False}


def inventory(protocol: dict) -> list[dict]:
    rows = []
    for scene, old_rows in protocol['existing_seed0_records'].items():
        for row in old_rows:
            rows.append({'run': row['run'], 'scene': scene, 'seed': 0,
                         'weight': row['weight'], 'configuration': row['configuration'],
                         'existing_seed0_reused': True, 'expected_metrics_sha256': row['metrics_sha256']})
    for row in protocol['new_runs']:
        rows.append({'run': row['run'], 'scene': row['scene'],
                     'seed': row['configuration']['seed'],
                     'weight': row['configuration']['pose_prior_weight'],
                     'configuration': row['configuration'], 'existing_seed0_reused': False})
    assert len(rows) == 12 and len({r['run'] for r in rows}) == 12
    assert {(r['scene'], r['seed'], r['weight']) for r in rows} == {
        (s, n, w) for s in ('desk', 'xyz') for n in (0, 1, 2) for w in (0., .01)}
    return rows


def collect(protocol: dict) -> tuple[list[dict], dict]:
    rows, baselines = inventory(protocol), {}
    for row in rows:
        folder = ROOT / 'outputs' / row['run']
        path = folder / 'metrics.json'
        if not path.is_file():
            row.update({'status': 'incomplete_or_not_started',
                        'failure_artifact_exists': (folder / 'failure.json').exists(),
                        'included_in_statistics': False})
            continue
        m = read(path)
        if row.get('expected_metrics_sha256') and sha(path) != row['expected_metrics_sha256']:
            raise ValueError('Seed0 artifact changed after the protocol was frozen')
        if m['configuration'] != row['configuration'] or m['script_sha256'] != protocol['source_sha256']['scripts/train_tum_online_gs_v4.py']:
            raise ValueError('Result does not match protocol configuration/source: ' + row['run'])
        if m['status'] not in ('completed', 'completed_with_tracking_or_mapping_failures') or m['frames_processed'] != 100 or m['selected_frame_ids'] != list(range(200, 300)):
            raise ValueError('Not a complete predeclared 100-frame trajectory: ' + row['run'])
        flags = Counter(flag for frame in m['frame_metrics'] for flag in frame['failure_flags'])
        failed = sum(bool(f['failure_flags']) for f in m['frame_metrics'])
        assert failed == m['failure_frame_count']
        assert len(m['frame_metrics']) == 100 and not m['frame_metrics'][0]['failure_flags']
        assert len(m['trajectory']['per_view']) == 100 and len(m['trajectory']['adjacent_rpe']) == 99
        baseline = {key: m['icp_prefix_trajectory_evaluation'][key] for key in KEYS}
        if row['scene'] in baselines:
            if baseline != baselines[row['scene']]['metrics']:
                raise ValueError('ICP prefix scalar metrics unexpectedly changed across seeds')
        else:
            baselines[row['scene']] = {'metrics': baseline, 'witness_run': row['run'],
                                      'scope': 'same-input independent depth ICP prefix; one geometric reference, not three random baseline observations'}
        row.update({'status': m['status'], 'included_in_statistics': True,
                    'metrics_relative_path': path.relative_to(ROOT).as_posix(),
                    'metrics_sha256': sha(path), 'trajectory_metrics': {key: m['trajectory'][key] for key in KEYS},
                    'failure_frame_count': failed, 'tracked_frame_denominator_excludes_fixed_first': 99,
                    'failure_frame_fraction': failed / 99., 'failure_flags': dict(flags),
                    'probe': m['pose_gradient_checks'], 'initial_point_count': m['initial_point_count'],
                    'final_point_count': m['final_point_count'],
                    'actual_first_context_sha256': m['first_tracking_context']['checkpoint_sha256'],
                    'online_loop_seconds_excludes_warmup': m['online_loop_seconds_excludes_warmup'],
                    'total_seconds_before_metrics_export': m['total_seconds_before_metrics_export']})
    return rows, baselines


def summarize(rows: list[dict], baselines: dict) -> dict:
    if len(rows) != 12 or any(not r['included_in_statistics'] for r in rows):
        raise ValueError('All twelve predeclared complete runs are required; no partial means')
    groups, deltas = {}, {}
    for scene in ('desk', 'xyz'):
        for weight in (0., .01):
            group = sorted((r for r in rows if r['scene'] == scene and r['weight'] == weight), key=lambda r: r['seed'])
            assert [r['seed'] for r in group] == [0, 1, 2]
            groups[f'{scene}_prior{weight:g}'] = {
                'scene': scene, 'prior_weight': weight, 'seeds': [0, 1, 2], 'runs': [r['run'] for r in group],
                'metrics': {key: moments([r['trajectory_metrics'][key] for r in group]) for key in KEYS},
                'failure_frame_counts_by_seed': [r['failure_frame_count'] for r in group],
                'complete_runs': 3, 'nonanchor_frame_count': 297,
                'failed_nonanchor_frames': sum(r['failure_frame_count'] for r in group)}
        off = groups[f'{scene}_prior0']; on = groups[f'{scene}_prior0.01']
        deltas[scene] = {
            'direction': 'prior0.01 minus prior0, per matching seed; lower errors are better',
            'metrics': {key: moments([b-a for a, b in zip(off['metrics'][key]['values'], on['metrics'][key]['values'])]) for key in KEYS},
            'on_minus_ICP_reference': {key: moments([v-baselines[scene]['metrics'][key] for v in on['metrics'][key]['values']]) for key in KEYS},
            'shared_actual_warmup_map_not_assumed': True}
    return {'groups': groups, 'paired_differences': deltas, 'ICP_reference': baselines,
            'status': 'all_predeclared_completed_runs_described', 'predeclared_runs': 12,
            'new_attempts': 8, 'reused_existing_seed0_runs': 4, 'complete_runs_in_statistics': 12,
            'process_incomplete_or_failed_runs': 0,
            'limitations': ['n=3 per method/sequence is descriptive, not a significance or IID claim',
                           'Seed changes point ordering/subsampling and GPU numerical histories; not isolated deterministic variation',
                           'Actual warmup map differences must be separately measured, not assumed equal from seed/configuration',
                           'Two same-sensor Freiburg1 sequences, both diagnostic; no global generalization or final unseen-test claim',
                           'All frame failures remain in trajectories; zero flags is not convergence or localization accuracy',
                           'ICP is an independent geometric reference, not an optimized GT-derived method choice']}


def verify_process_ledger(protocol: dict, rows: list[dict]) -> dict:
    path = ROOT / 'outputs/tum-online-v4-seed-replication-001/metrics.json'
    ledger = read(path)
    if ledger['status'] != 'all_eight_predeclared_runs_completed' or ledger['GPU_runs_started'] != 8 or ledger['current_run'] is not None:
        raise ValueError('The batch driver is not terminal with all eight processes collected')
    if ledger['protocol_sha256'] != PROTOCOL_SHA or len(ledger['run_records']) != 8:
        raise ValueError('Wrong batch execution ledger')
    by_run = {row['run']: row for row in rows}
    previous_end = None
    deadline = dt.datetime.fromisoformat(protocol['deadline_beijing'])
    for declared, record in zip(protocol['new_runs'], ledger['run_records']):
        start = dt.datetime.fromisoformat(record['started_at']['beijing'])
        end = dt.datetime.fromisoformat(record['ended_at']['beijing'])
        assert start < deadline and end >= start
        assert previous_end is None or start >= previous_end
        previous_end = end
        assert record['run'] == declared['run'] and record['order'] == declared['order']
        assert record['configuration'] == declared['configuration'] and record['returncode'] == 0
        assert record['status'] == by_run[record['run']]['status']
        assert record['metrics_sha256'] == by_run[record['run']]['metrics_sha256']
        assert record['log_sha256'] == sha(ROOT / declared['log_relative_path'])
    return {'status': 'all_eight_process_exits_and_original_metric_log_hashes_checked',
            'batch_metrics_relative_path': path.relative_to(ROOT).as_posix(),
            'batch_metrics_sha256': sha(path), 'driver_sha256': protocol['driver_sha256'],
            'returncodes': [record['returncode'] for record in ledger['run_records']],
            'serial_ledger_and_start_before_deadline_checked': True,
            'scope': 'Recorded process start timestamps, terminal return codes, original metrics and logs; not an OS scheduler timing proof'}


def plot(report: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, scene in enumerate(('desk', 'xyz')):
        for row, key in enumerate(KEYS[:2]):
            ax = axes[row, column]
            for x, weight in enumerate((0., .01)):
                values = report['groups'][f'{scene}_prior{weight:g}']['metrics'][key]
                a = np.asarray(values['values']) * 1000
                ax.scatter(np.arange(3)*.10-.10+x, a, color=('#ce7643', '#309ca0')[x], label='seeds 0/1/2' if x == 0 else None)
                ax.errorbar(x, values['mean']*1000, yerr=values['sample_standard_deviation_ddof1']*1000,
                            fmt='k_', capsize=7, linewidth=1.5, label='mean +/- sample SD' if x == 0 else None)
            ax.axhline(report['ICP_reference'][scene]['metrics'][key]*1000, color='#777777', linestyle='--', label='independent ICP')
            ax.set_xticks([0, 1], ['prior off', 'prior 0.01'])
            ax.set_title(f'{scene}: ' + ('first-GT-anchored raw position RMSE' if row == 0 else 'SE3-aligned ATE, no scale'))
            ax.set_ylabel('mm'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Fixed V4 configuration: all three seeds, two Freiburg1 sequences\nDots = actual complete runs; error bars describe variation, not confidence')
    fig.savefig(destination, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--run', default='tum-online-v4-seed-study-summary-001')
    args = parser.parse_args()
    checks = cpu_checks()
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Unknown protocol bytes')
    protocol = read(PROTOCOL)
    for relative, digest in protocol['source_sha256'].items():
        assert sha(ROOT / relative) == digest
    rows, baselines = collect(protocol)
    complete = sum(r['included_in_statistics'] for r in rows)
    if args.check_only:
        print(json.dumps({'status': 'CPU_controls_passed', 'complete_runs': complete,
                          'pending_or_incomplete': 12-complete, 'statistics_not_yet_computed': True, 'checks': checks}))
        return
    if not args.run or Path(args.run).name != args.run or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run):
        raise ValueError('Unsafe output run name')
    ledger = verify_process_ledger(protocol, rows)
    report = {**summarize(rows, baselines), 'runs': rows, 'cpu_checks': checks,
              'process_ledger': ledger,
              'script_sha256': sha(Path(__file__)), 'protocol_sha256': PROTOCOL_SHA,
              'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'all_metrics_posthoc_only': True}
    folder = ROOT / 'outputs' / args.run
    folder.mkdir(exist_ok=False)
    with (folder / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    plot(report, folder / 'seed-comparison.png')
    print(json.dumps({'status': report['status'], 'complete_runs': complete,
                      'output': str(folder), 'groups': report['groups']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

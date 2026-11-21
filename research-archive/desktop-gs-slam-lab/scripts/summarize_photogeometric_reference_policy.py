"""Read-only complete policy comparison after both recovery rounds terminate.

Exports compact complete per-frame records, causal reference/candidate ledgers
and posthoc figures. No registration, candidate selection or alignment fitting
is performed. Large original iterations remain local and hash referenced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'outputs/rgbd-photogeometric-last-accepted-cpu-preparation-001/protocol.json'
PROTOCOL_SHA = 'd2adb632c2261e50bc963e3b84c31d44e92233b3f399bd45a23fafd457ce13c1'
KEYS = ('raw_anchored_translation_rmse_m', 'se3_aligned_ate_translation_rmse_m',
        'adjacent_rpe_translation_rmse_m', 'adjacent_rpe_rotation_rmse_deg')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(payload) < 1024 * 1024, 'Compact artifact under 1MiB: ' + str(path))
    with path.open('xb') as stream: stream.write(payload)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(payload), 'sha256': sha(path)}


def ledger_check(data, recovery):
    frames = data['frame_metrics']; poses = np.asarray([r['estimated_c2w'] for r in frames])
    require(len(frames) == 100 and [r['frame_id'] for r in frames] == list(range(200, 300)), 'All 100 requested frames')
    require(np.array_equal(poses[0], np.eye(4)), 'First camera fixed identity')
    reference = 0; ledger = []; reconstruction_diff = 0.
    for index, row in enumerate(frames):
        if index == 0:
            ledger.append({'current_frame_id': 200, 'reference_frame_id': 200, 'reference_gap_frames': 0,
                           'last_accepted_frame_id_after': 200, 'accepted': True, 'failure_flags': [],
                           'candidate_reference_from_current': None, 'estimated_c2w': row['estimated_c2w']})
            continue
        if not recovery: reference = index - 1
        require(row['accepted'] == (not row['failure_flags']), 'Acceptance/flags retained')
        if recovery:
            require(row['current_frame_id'] == 200 + index and row['reference_frame_id'] == 200 + reference and
                    row['reference_gap_frames'] == index - reference and row['last_accepted_frame_id_before'] == 200 + reference,
                    'Causal last accepted reference IDs/gap')
            require(row['maximum_sensor_frame_used_for_this_pair'] == row['frame_id'] and
                    row['reference_sensor_metadata'] == data['selected_sensor_metadata'][reference], 'No future reference sensor')
            require(row['candidate_reference_from_current'] == row['candidate_previous_from_current'], 'Unmodified candidate matrix')
        candidate = row['candidate_previous_from_current']
        next_reference = index if row['accepted'] else reference
        expected = poses[reference] @ np.asarray(candidate) if row['accepted'] else poses[index - 1]
        diff = float(np.max(np.abs(expected - poses[index])))
        reconstruction_diff = max(reconstruction_diff, diff)
        require(diff < 2e-14, 'Actual accepted compose or rejected-prefix propagation')
        if not row['accepted']: require(np.array_equal(poses[index], poses[index - 1]), 'Exact rejected prefix')
        if recovery:
            require(row['last_accepted_frame_id_after'] == 200 + next_reference, 'Updated last accepted ID')
        ledger.append({'current_frame_id': row['frame_id'], 'reference_frame_id': 200 + reference,
                       'reference_gap_frames': index - reference, 'last_accepted_frame_id_after': 200 + next_reference,
                       'accepted': row['accepted'], 'failure_flags': row['failure_flags'],
                       'candidate_reference_from_current': candidate, 'estimated_c2w': row['estimated_c2w'],
                       'reference_ID_scope': 'recorded and checked' if recovery else 'derived from frozen previous-frame source policy'})
        reference = next_reference
    return ledger, reconstruction_diff


def collect(batch):
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'Reviewed protocol actual bytes')
    protocol = read(PROTOCOL)
    ledger = read(batch / 'metrics.json')
    require(ledger['status'] == 'all_two_predeclared_CPU_rounds_completed' and ledger['runs_started'] == 2 and
            ledger['current_run'] is None and ledger['GPU_runs'] == 0 and len(ledger['run_records']) == 2, 'Both actual processes terminal before aggregate')
    require(ledger['protocol_sha256'] == PROTOCOL_SHA, 'Actual batch reviewed protocol')
    for name, digest in protocol['source_sha256'].items(): require(sha(ROOT / 'scripts' / name) == digest, 'Frozen source ' + name)
    methods = []; watched = {PROTOCOL: sha(PROTOCOL), batch / 'metrics.json': sha(batch / 'metrics.json')}
    for frozen, child in zip(protocol['rounds'], ledger['run_records']):
        require(child['round_id'] == frozen['round_id'] and child['run'] == frozen['run'] and child['order'] == frozen['order'] and
                child['configuration'] == frozen['configuration'] and child['returncode'] == 0, 'Exact two-round order/config/exit status')
        log = batch / (frozen['round_id'] + '.log')
        require(sha(log) == child['log_sha256'], 'Actual child log hash')
        for recovery in (False, True):
            reference = protocol['original_previous_frame_photo1_references'][frozen['scene']]
            run = frozen['run'] if recovery else reference['run']
            folder = ROOT / 'outputs' / run; metric = folder / 'metrics.json'; data = read(metric)
            require(sha(metric) == (child['metrics_sha256'] if recovery else reference['metrics_sha256']), 'Original metric bytes')
            require(data['status'] in ('completed', 'completed_with_pair_rejections') and data['frames_processed'] == 100 and
                    data['GT_pose_used_in_optimizer'] is False, 'Complete GT-free optimizer records')
            if recovery:
                require((folder / 'protocol_used.json').read_bytes() == PROTOCOL.read_bytes() and
                        data['configuration'] == frozen['configuration'] and data['source_sha256'] == protocol['source_sha256'], 'Actual new config/source/protocol')
            else:
                require(data['configuration'] == reference['configuration'] and sha(folder / 'trajectory_estimated.txt') == reference['trajectory_sha256'], 'Immutable previous policy reference')
            require(data['configuration']['odometry'] == frozen['configuration']['odometry'] and
                    data['selected_sensor_metadata'] == protocol['datasets'][frozen['scene']]['selected_sensor_metadata'] and
                    data['input_array_hashes'] == protocol['datasets'][frozen['scene']]['input_array_hashes'], 'Same engineering objective and actual sensors')
            failures = sum(bool(r['failure_flags']) for r in data['frame_metrics'])
            require(failures == data['failure_frame_count'], 'Complete flags/failure denominator')
            causal, difference = ledger_check(data, recovery)
            compact_frames = []
            for row in data['frame_metrics']:
                saved = {k: v for k, v in row.items() if k not in ('iteration_history', 'configuration')}
                if 'iteration_history' in row:
                    saved['iteration_history_summary'] = {'records': len(row['iteration_history']),
                        'accepted_updates': sum(bool(h['accepted']) for h in row['iteration_history']),
                        'stop_reasons': dict(Counter(h.get('stop_reason', 'continued') for h in row['iteration_history']))}
                compact_frames.append(saved)
            compact = {k: v for k, v in data.items() if k not in ('frame_metrics', 'observed_wall_seconds', 'timing_scope')}
            compact['frame_metrics'] = compact_frames
            compact['public_summary_provenance'] = {'original_metrics_relative_path': metric.relative_to(ROOT).as_posix(),
                'original_metrics_sha256': sha(metric), 'original_metrics_bytes': metric.stat().st_size,
                'summarizer_sha256': sha(Path(__file__)), 'local_iterations_sha256': sha(folder / 'iterations.jsonl'),
                'scope': 'All frames/flags/counts/levels/candidates/prefix and reference fields retained. Full per-iteration matrices and trial costs stay local. No duration comparison.'}
            token = 'last-accepted' if recovery else 'previous-frame'
            methods.append({'name': frozen['scene'] + '-' + token, 'scene': frozen['scene'], 'run': run,
                            'policy': token, 'folder': folder, 'data': data, 'compact': compact, 'ledger': causal,
                            'max_prefix_reconstruction_difference': difference, 'failure_count': failures})
            for path in folder.iterdir():
                if path.is_file(): watched[path] = sha(path)
    require(all(sha(path) == digest for path, digest in watched.items()), 'Original artifacts unchanged during collect')
    return protocol, ledger, methods, watched


def figures(methods, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'previous-frame': '#bd743d', 'last-accepted': '#238d96'}
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for row, scene in enumerate(('desk', 'xyz')):
        selected = [m for m in methods if m['scene'] == scene]
        truth = np.loadtxt(selected[0]['folder'] / 'trajectory_groundtruth_anchored.txt')[:, 1:4]
        points = np.concatenate([truth] + [np.asarray([r['estimated_c2w'] for r in m['data']['frame_metrics']])[:, :3, 3] for m in selected])
        lo, hi = points.min(axis=0), points.max(axis=0); margin = np.maximum((hi - lo) * .05, .002)
        for col, m in enumerate(selected):
            estimated = np.asarray([r['estimated_c2w'] for r in m['data']['frame_metrics']])[:, :3, 3]
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection='3d')
            ax.plot(*truth.T, color='#444444', label='GT, first-camera anchored')
            ax.plot(*estimated.T, color=colors[m['policy']], label='estimated prefix')
            rejected = np.array([bool(r['failure_flags']) for r in m['data']['frame_metrics']])
            if rejected.any(): ax.scatter(*estimated[rejected].T, color='#d04438', s=25, label='rejected, retained')
            ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=m['name'] + f' | {m["failure_count"]}/99 rejected')
            ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0]); ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1]); ax.set_zlim(lo[2] - margin[2], hi[2] + margin[2])
            ax.view_init(elev=22, azim=-60); ax.legend(fontsize=8)
    fig.suptitle('Fixed photo1 objective: two reference-sensor policies\nAll 100 frames retained; raw common first-camera gauge, no fitted alignment')
    fig.savefig(output / 'trajectories-and-rejections.png', dpi=145); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, scene in enumerate(('desk', 'xyz')):
        selected = [m for m in methods if m['scene'] == scene]
        for band, m in enumerate(selected):
            frames = np.arange(200, 300)
            axes[row, 0].plot(frames, np.asarray(m['data']['trajectory']['per_frame_raw_translation_error_m']) * 1000,
                              color=colors[m['policy']], label=m['policy'])
            axes[row, 1].step(frames[1:], [r['reference_gap_frames'] for r in m['ledger'][1:]], where='mid',
                              color=colors[m['policy']], label=m['policy'])
            axes[row, 2].broken_barh([(200.5, 99)], (band - .25, .5), facecolors='#ededed')
            for record in m['ledger'][1:]:
                if record['failure_flags']: axes[row, 2].broken_barh([(record['current_frame_id'] - .5, 1)], (band - .25, .5), facecolors='#d04438')
        axes[row, 0].set(title=scene + ': raw anchored 3D position error', xlabel='associated frame ID', ylabel='mm'); axes[row, 0].legend(); axes[row, 0].grid(alpha=.2)
        axes[row, 1].set(title=scene + ': current - reference sensor ID', xlabel='associated frame ID', ylabel='gap in associated frames'); axes[row, 1].legend(); axes[row, 1].grid(alpha=.2)
        axes[row, 2].set(title=scene + ': all requested frames / rejected flags', xlabel='associated frame ID', yticks=[0, 1],
                        yticklabels=['previous-frame', 'last-accepted'], xlim=(200.5, 299.5))
    fig.suptitle('Full reference-gap and failure ledger\nGray = all tracked frames 201..299; red = rejected, kept with unchanged prefix')
    fig.savefig(output / 'position-error-reference-gap-and-flags.png', dpi=145); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, default=ROOT / 'outputs/rgbd-photogeometric-last-accepted-validation-001')
    p.add_argument('--run', default='rgbd-photogeometric-reference-policy-comparison-001'); args = p.parse_args()
    require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique simple output name')
    protocol, process, methods, watched = collect(args.batch)
    output = ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False)
    artifacts = [save(output / (m['name'] + '-summary.json'), m['compact']) for m in methods]
    artifacts.append(save(output / 'complete-reference-candidate-ledgers.json', {
        'status': 'complete_all_100_frame_ledgers', 'summarizer_sha256': sha(Path(__file__)),
        'methods': [{'name': m['name'], 'run': m['run'], 'frames': m['ledger'],
                     'max_prefix_reconstruction_difference': m['max_prefix_reconstruction_difference']} for m in methods]}))
    table = [{'name': m['name'], 'run': m['run'], 'scene': m['scene'], 'policy': m['policy'], 'frames': 100,
              'tracked_nonanchor_denominator': 99, 'failure_count': m['failure_count'],
              'all_failure_frames': [{'frame_id': r['frame_id'], 'flags': r['failure_flags']} for r in m['data']['frame_metrics'] if r['failure_flags']],
              'reference_gap_counts': dict(Counter(r['reference_gap_frames'] for r in m['ledger'][1:])),
              'max_reference_gap_frames': max(r['reference_gap_frames'] for r in m['ledger'][1:]),
              'trajectory_metrics': {k: m['data']['trajectory'][k] for k in KEYS},
              'original_metrics_sha256': sha(m['folder'] / 'metrics.json'),
              'original_trajectory_text_sha256': sha(m['folder'] / 'trajectory_estimated.txt')} for m in methods]
    report = {'status': 'both_predeclared_recovery_CPU_rounds_terminal', 'protocol_sha256': PROTOCOL_SHA,
              'batch_metrics_sha256': sha(args.batch / 'metrics.json'), 'summarizer_sha256': sha(Path(__file__)),
              'process_returncodes': [r['returncode'] for r in process['run_records']], 'recovery_rounds': 2, 'previous_policy_references': 2,
              'methods': table, 'public_compact_artifacts': artifacts, 'all_frames_and_flags_retained': True,
              'limits': protocol['limits'] + ['Trajectory formulas reuse audit_pose_arrays; this summary is not an independent metric evaluator.',
                  'Original previous-frame reference IDs are derived from its frozen source policy; new IDs/gaps are recorded and checked.',
                  'Only terminal rounds are compared. No GT candidate selection, threshold changes, re-registration, retry, speed claim, or SE3/Sim3 plot fitting.']}
    figures(methods, output)
    require(all(sha(path) == digest for path, digest in watched.items()), 'Original artifacts unchanged after summary/plots')
    report['original_files_unchanged_count'] = len(watched)
    report['plot_artifacts'] = [{'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in output.glob('*.png')]
    save(output / 'metrics.json', report)
    print(json.dumps({'status': report['status'], 'output': str(output), 'summary_sizes': [r['bytes'] for r in artifacts],
                      'methods': table}, indent=2))


if __name__ == '__main__': main()

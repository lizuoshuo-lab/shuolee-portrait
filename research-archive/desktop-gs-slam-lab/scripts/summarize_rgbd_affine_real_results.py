"""Read-only compact comparison after independent actual affine audit passes.

Retains four full100-frame methods (frozen6D last-accepted + new8D affine in
each scene), all failures/candidates/gain/bias/reference IDs. Large per-trial
ledgers remain authoritative locally. No across-scene mean, speed or tuning.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/rgbd-affine-photogeometric-real-cpu-preparation-001'
BATCH = ROOT / 'outputs/rgbd-affine-photogeometric-real-validation-001'
OUTPUT = ROOT / 'outputs/rgbd-affine-photogeometric-comparison-002'
AUDITOR_SHA = 'b549a0af2642b2e203e8f3b5f0deae46dcaf2458f608d7e27435c87b33f8c6d5'
AUDIT_SHA = 'd2c355e61bb9cac52158594fac9bedfcfde199ee00b1a249e4eec28093881346'
PROTOCOL_SHA = '9d8e674827a83f2c29e7dc901f2bcb8e1203b5513d89692b5aa92b34db0d8a63'


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'); require(len(raw) < 1024 * 1024, 'Public JSON below1MiB without dropping frames')
    with path.open('xb') as stream: stream.write(raw)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def history_summary(history):
    return {'records': len(history), 'accepted_updates': sum(bool(r['accepted']) for r in history),
            'stop_reasons': dict(Counter(r.get('stop_reason', 'continued') for r in history)),
            'joint_rank_histogram': dict(Counter(r['joint_rank'] for r in history if 'joint_rank' in r)),
            'line_search_trial_count': sum(len(r['line_search_trials']) for r in history),
            'invalid_trial_counts': dict(Counter(t['invalid_reason'] for r in history for t in r['line_search_trials'] if t.get('invalid_reason')))}


def compact_frames(frames):
    rows = []
    for full in frames:
        row = {k: v for k, v in full.items() if k not in ('configuration', 'iteration_history')}
        if 'iteration_history' in full: row['iteration_history_summary'] = history_summary(full['iteration_history'])
        rows.append(row)
    return rows


def nuisance_stats(rows):
    result = {}
    for key in ('candidate_log_relative_gain', 'candidate_relative_gain', 'candidate_bias_intensity'):
        values = [r[key] for r in rows[1:] if r.get(key) is not None]
        result[key] = {'defined_pair_denominator': len(values), 'all_pair_denominator': 99,
                       'mean': math.fsum(values) / len(values) if values else None,
                       'min': min(values) if values else None, 'max': max(values) if values else None}
    result['includes_rejected_candidates'] = True
    return result


def plots(data, truth):
    colors = {'old6': '#2463a6', 'affine8': '#cf6713'}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for ax, scene in zip(axs, ('desk', 'xyz')):
        gt = truth[scene][:, :3, 3]; ax.plot(gt[:, 0], gt[:, 1], color='#222222', label='Raw GT, first-camera gauge', lw=1.5)
        for key, label in (('old6', 'Frozen6D last-accepted'), ('affine8', 'Joint8D affine last-accepted')):
            rows = data[scene][key]['frame_metrics']; poses = np.asarray([r['estimated_c2w'] for r in rows]); points = poses[:, :3, 3]
            ax.plot(points[:, 0], points[:, 1], color=colors[key], lw=1.3, label=label)
            indices = [i for i, r in enumerate(rows) if r['failure_flags']]
            ax.scatter(points[indices, 0], points[indices, 1], marker='x', color=colors[key], s=38, zorder=5)
            position_groups = {}
            for i in indices: position_groups.setdefault(tuple(points[i, :2]), []).append(str(rows[i]['frame_id']))
            for position, ids in position_groups.items(): ax.annotate(','.join(ids), position, xytext=(3, 3), textcoords='offset points', fontsize=7, color=colors[key])
        ax.set(title=scene + ': sensors200..299, XY projection', xlabel='Camera center x (m)', ylabel='Camera center y (m)'); ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    axs[0].legend(fontsize=7, loc='best'); fig.suptitle('Pair acceptance and trajectory accuracy differ; all rejection markers retained', fontsize=10)
    fig.savefig(OUTPUT / 'trajectory-comparison.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(4, 2, figsize=(11, 10), constrained_layout=True)
    for col, scene in enumerate(('desk', 'xyz')):
        for key, label in (('old6', 'Frozen6D'), ('affine8', 'Joint8D affine')):
            row = data[scene][key]; frames = row['frame_metrics']; xs = np.asarray([r['frame_id'] for r in frames]); bad = np.asarray([bool(r['failure_flags']) for r in frames])
            errors = np.asarray(row['trajectory']['per_frame_raw_translation_error_m']) * 1000
            axs[0, col].plot(xs, errors, color=colors[key], lw=1.2, label=label)
            axs[0, col].scatter(xs[bad], errors[bad], marker='x', color=colors[key], s=27)
            axs[3, col].step(xs[1:], [r['reference_gap_frames'] for r in frames[1:]], where='mid', color=colors[key], lw=1.1, label=label)
        frames = data[scene]['affine8']['frame_metrics']; xs = np.asarray([r['frame_id'] for r in frames[1:]]); bad = np.asarray([bool(r['failure_flags']) for r in frames[1:]])
        for row_index, key, baseline, label in ((1, 'candidate_relative_gain', 1., 'Pair relative gain exp(a)'), (2, 'candidate_bias_intensity', 0., 'Pair bias (gray0..1 units)')):
            values = np.asarray([r.get(key) if r.get(key) is not None else np.nan for r in frames[1:]])
            axs[row_index, col].plot(xs, values, color=colors['affine8'], lw=1.1); axs[row_index, col].scatter(xs[bad], values[bad], marker='x', color=colors['affine8'], s=27); axs[row_index, col].axhline(baseline, color='#666666', ls=':', lw=.7); axs[row_index, col].set_ylabel(label)
        axs[0, col].set(title=scene + ': all100 frames, raw gauge', ylabel='Raw translation error (mm)'); axs[0, col].legend(fontsize=8)
        axs[3, col].set(xlabel='Associated frame ID', ylabel='Current - reference (frames)'); axs[3, col].legend(fontsize=8); axs[3, col].set_yticks([1, 2, 3])
        for ax in axs[:, col]: ax.grid(alpha=.2); ax.set_xlim(200, 299)
    fig.suptitle('Brightness correction resets each pair; x marks rejected candidate/held trajectory', fontsize=10)
    fig.savefig(OUTPUT / 'errors-brightness-reference-gaps.png', dpi=160); plt.close(fig)


def main():
    require(sha(PREP / 'protocol.json') == PROTOCOL_SHA and sha(BATCH / 'independent_affine_result_audit.json') == AUDIT_SHA and
            sha(ROOT / 'scripts/audit_rgbd_affine_real_results.py') == AUDITOR_SHA, 'Reviewed source and passed actual audit exact')
    protocol = read(PREP / 'protocol.json'); batch = read(BATCH / 'metrics.json'); audit = read(BATCH / 'independent_affine_result_audit.json')
    require(audit['status'] == 'passed_independent_input_reference_8D_step_and_trajectory_audit' and audit['batch_metrics_sha256'] == sha(BATCH / 'metrics.json') and
            batch['status'] == 'all_two_predeclared_CPU_rounds_completed' and len(batch['run_records']) == 2, 'Both terminal audited before aggregation')
    OUTPUT.mkdir(exist_ok=False); watched = {}; data = {}; truth = {}; methods = []; compact = []; all_ledgers = []
    for scene in ('desk', 'xyz'):
        data[scene] = {}; reference = protocol['original_lastaccepted_photo1_references'][scene]; record = next(r for r in batch['run_records'] if r['round_id'] == scene + '-affine-lastaccepted')
        for key, run, expected in (('old6', reference['run'], reference['metrics_sha256']), ('affine8', record['run'], record['metrics_sha256'])):
            folder = ROOT / 'outputs' / run; require(sha(folder / 'metrics.json') == expected, 'Actual original metrics SHA'); watched[folder / 'metrics.json'] = expected
            actual = read(folder / 'metrics.json'); data[scene][key] = actual; frames = compact_frames(actual['frame_metrics'])
            require(actual['frames_processed'] == len(frames) == 100 and actual['status'] in ('completed', 'completed_with_pair_rejections'), 'All100 original frame rows retained')
            summary = {k: v for k, v in actual.items() if k not in ('frame_metrics', 'observed_wall_seconds')}
            summary.update(original_metrics_sha256=expected, source_run=run, frame_metrics=frames,
                           removed_large_fields=['per-frame configuration duplicate', 'full per-trial iteration_history remains in immutable source'],
                           endpoint_raw_translation_error_m=actual['trajectory']['per_frame_raw_translation_error_m'][-1])
            artifact = write(OUTPUT / (scene + '-' + key + '-summary.json'), summary); compact.append(artifact)
            ledger = {'scene': scene, 'method': key, 'run': run, 'original_metrics_sha256': expected, 'all100_frame_rows': frames}
            # Portable full ledger omits duplicated frame metadata/stats only;
            # every ID/candidate/estimated pose/gain/bias/flag is retained here.
            ledger_keys = ['local_index', 'frame_id', 'current_frame_id', 'reference_frame_id', 'reference_gap_frames', 'last_accepted_frame_id_before',
                           'last_accepted_frame_id_after', 'accepted', 'failure_flags', 'estimated_c2w', 'candidate_reference_from_current',
                           'candidate_log_relative_gain', 'candidate_relative_gain', 'candidate_bias_intensity', 'pair_initial_transform',
                           'pair_initial_log_relative_gain', 'pair_initial_bias_intensity', 'brightness_accumulated_across_pairs', 'pair_initialized_from_GT']
            ledger['all100_frame_rows'] = [{k: r[k] for k in ledger_keys if k in r} for r in frames]; all_ledgers.append(ledger)
            metrics = {k: v for k, v in actual['trajectory'].items() if not isinstance(v, list)}
            methods.append({'scene': scene, 'method': key, 'run': run, 'state_dimension': 8 if key == 'affine8' else 6,
                            'metrics_sha256': expected, 'complete_frames': 100, 'tracked_pairs': 99, 'failure_frame_count': actual['failure_frame_count'],
                            'failure_frame_ids': [r['frame_id'] for r in frames if r['failure_flags']], 'failure_flag_counts': actual['failure_flag_counts'],
                            'reference_gap_counts': actual['reference_gap_counts'], 'trajectory_metrics': metrics,
                            'endpoint_raw_translation_error_m': summary['endpoint_raw_translation_error_m'],
                            'candidate_brightness_statistics': nuisance_stats(frames) if key == 'affine8' else None,
                            'level_stop_counts_all_pairs': dict(Counter(level['stop_reason'] for r in actual['frame_metrics'][1:] for level in r['levels'])),
                            'final_persisted_joint_rank_counts': dict(Counter(r['final_statistics'].get('joint_rank') for r in actual['frame_metrics'][1:])),
                            'public_compact_summary': artifact})
        with np.load(ROOT / 'outputs' / record['run'] / 'trajectory_arrays.npz', allow_pickle=False) as saved: truth[scene] = saved['evaluation_only_anchored_GT_c2w'].copy()
    compact.append(write(OUTPUT / 'complete-reference-candidate-ledgers.json', {'status': 'all400_frame_rows_retained', 'all_frame_count': 400, 'tracked_pair_count': 396, 'methods': all_ledgers,
                       'scope': 'All IDs/gaps/flags/full pose candidates/estimated prefix/defined gain-bias values; metadata/stats remain in the four compact summaries and full trials locally.'}))
    plots(data, truth); figures = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in (OUTPUT / 'trajectory-comparison.png', OUTPUT / 'errors-brightness-reference-gaps.png')]
    require(all(sha(path) == digest for path, digest in watched.items()), 'Original methods unchanged')
    result = {'status': 'completed_read_only_four_method_comparison_after_actual_affine_audit', 'summarizer_sha256': sha(Path(__file__)),
              'protocol_sha256': PROTOCOL_SHA, 'batch_metrics_sha256': sha(BATCH / 'metrics.json'), 'affine_independent_audit_sha256': AUDIT_SHA,
              'actual_methods': 4, 'all_frame_count': 400, 'tracked_pair_count': 396, 'methods': methods, 'public_compact_summaries': compact, 'figures': figures,
              'limits': ['Complete desk/xyz200..299 clips previously examined; same-sequence diagnostic, not unseen test or full trajectory.',
                         'Frozen6D versus joint8D changes rank, nuisance convergence and correspondence/trial paths; gain/bias/reset are pair-local, no radiometric calibration.',
                         'Old rejection illumination cause has not been established. Pair acceptance is not GT proof of accuracy or recovery.',
                         'All400 frames/rejects/candidates retained. Full per-trial history stays local and hash-bound; no >1MiB public JSON.',
                         'Metrics mix meters and degrees with explicit names; plot errors use mm. No across-scene mean, single-seed SD, FPS or isolated speed comparison.',
                         'The actual affine audit explicitly reuses prior independent e807 metric formulas; only new SE3/8D artifact algebra is added.']}
    artifact = write(OUTPUT / 'metrics.json', result); (OUTPUT / 'source_script.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'methods': 4, 'rows': 400, 'summary': artifact, 'figures': figures}))


if __name__ == '__main__': main()

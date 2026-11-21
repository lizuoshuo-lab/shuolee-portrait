"""Read-only scientific figures from audited compact rows and saved GT arrays.

No registration, GT metric calculation, alignment, reanchoring or rescaling.
Each scene has its own coordinate limits; XY axes use equal metre scaling.
Every stored frame, including repeated rejected estimates, remains in the plot.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = 'outputs/rgbd-affine-full-comparison-001'
FIELD_AUDIT = 'outputs/rgbd-affine-full-public-field-audit-001/metrics.json'
FIELD_AUDIT_SHA = '45f3e5b378bea7340fa82472d87da57ed5e31d246210e5b99d062fa8595cb92d'
SUMMARY_SHA = 'd6a5f9e304579e98b0b97912762eb6c7618273c89398bf3c95e09809e76635d3'
NPZ_SHA = {
    ('desk', 6): 'acb6cec358b7db978d8821f43c94564ec5da6a368120000033f46a68c05bc3bf',
    ('desk', 8): '15b69f60954f6a3025e1e247ba1d086b8625f45eb338a40d6bfb0fe0c435d908',
    ('xyz', 6): 'fcc2f81845261c7eec3790278dcbaa4dcf5a44d9521b9f598cec04f979e8b79d',
    ('xyz', 8): 'ebd1479d2df6ff858dbfd205ec79c818585c1e68a474cf657b145f576fe566c3',
}
COLORS = {6: '#1764ad', 8: '#da7626'}
STYLES = {6: '-', 8: '--'}
DPI = 160
FIGSIZE = (15, 7)


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.bindings = {}

    def read(self, relative, expected=None):
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Existing lab input: ' + relative)
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, 'Input hash: ' + relative)
        value = {'sha256': digest, 'bytes': len(raw)}
        if relative in self.bindings:
            require(self.bindings[relative] == value, 'Repeated input unchanged: ' + relative)
        self.bindings[relative] = value
        return raw

    def json(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def recheck(self):
        for relative, value in list(self.bindings.items()):
            self.read(relative, value['sha256'])


def load_scene(scene, n, summary, inputs):
    scene_arms = summary['scene_results'][scene]['arms']
    require(len(scene_arms) == 2, 'Both predeclared model arms')
    models = {}; stored_gt = None; stored_times = None
    for dimension in (6, 8):
        matches = [arm for arm in scene_arms if arm['model_dimension'] == dimension]
        require(len(matches) == 1, 'Unique scene/model arm')
        arm = matches[0]; require(arm['frames'] == n, 'Complete associated sequence')
        compact = inputs.json(PUBLIC + '/' + arm['all_frame_file'], arm['all_frame_file_sha256'])
        columns = compact['columns']; rows = compact['rows']
        require(len(columns) == len(set(columns)) == 24 and len(rows) == n, 'All audited compact rows')
        records = [dict(zip(columns, row)) for row in rows]
        require(all(len(row) == 24 for row in rows), 'All public cells present')
        ids = np.array([row['frame_id'] for row in records], dtype=np.int64)
        gap = np.array([row['reference_gap_frames'] for row in records], dtype=np.int64)
        accepted = np.array([row['accepted'] for row in records], dtype=bool)
        poses = np.array([row['estimated_c2w'] for row in records], dtype=np.float64)
        times = np.array([row['rgb_timestamp'] for row in records], dtype=np.float64)
        require(np.array_equal(ids, np.arange(n)) and poses.shape == (n, 4, 4) and np.isfinite(poses).all(),
                'Complete ordered finite saved camera matrices')
        require(gap[0] == 0 and accepted[0] and all(row['accepted'] == (not row['failure_flags']) for row in records),
                'Explicit anchor gap and rejected status')
        require(int((~accepted).sum()) == arm['failure_frame_count'], 'Every rejection retained')
        require([row['frame_id'] for row in records if not row['accepted']] == arm['all_rejected_frame_ids'],
                'All rejected frame IDs, unchanged')
        relative = 'outputs/' + arm['run'] + '/trajectory_arrays.npz'
        with np.load(io.BytesIO(inputs.read(relative, NPZ_SHA[(scene, dimension)])), allow_pickle=False) as data:
            require(set(data.files) == {'estimated_c2w', 'evaluation_only_anchored_GT_c2w', 'rgb_timestamps'},
                    'Exact original trajectory NPZ fields')
            require(np.array_equal(poses, data['estimated_c2w']) and np.array_equal(times, data['rgb_timestamps']),
                    'Audited public cameras/timestamps equal original saved arrays')
            gt = data['evaluation_only_anchored_GT_c2w'].copy()
        require(gt.shape == (n, 4, 4) and np.isfinite(gt).all(), 'Saved finite anchored GT matrices')
        if stored_gt is None:
            stored_gt = gt; stored_times = times.copy()
        else:
            require(np.array_equal(stored_gt, gt) and np.array_equal(stored_times, times),
                    'Same stored GT and timestamps across the two scene arms')
        models[dimension] = {'poses': poses, 'ids': ids, 'gaps': gap, 'accepted': accepted,
                             'rejected': int((~accepted).sum()), 'run': arm['run'], 'records': records}
    if scene == 'desk':
        for model in models.values():
            require(np.all(~model['accepted'][134:549]) and len(model['accepted'][134:549]) == 415 and
                    all(row['reference_frame_id'] == 133 for row in model['records'][134:549]),
                    'The annotated shared 415-frame rejection interval is an actual stored fact')
    return models, stored_gt


def draw_scene(scene, n, models, gt, destination):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelsize': 11, 'legend.fontsize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, (xy, gaps) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={'width_ratios': [1.05, 1]})
        fig.subplots_adjust(left=.065, right=.98, top=.83, bottom=.20, wspace=.28)
        fig.suptitle(f'TUM fr1_{scene}: complete {n} associated frames', fontsize=16, fontweight='bold', y=.965)
        fig.text(.5, .912, 'Last-accepted 6D / 8D  |  Saved first-camera coordinates  |  No alignment or metric recomputation',
                 ha='center', fontsize=11, color='#444444')
        # c2w translation is already the camera center; no inversion or fitting.
        xy.plot(gt[:, 0, 3], gt[:, 1, 3], color='#222222', linewidth=1.45, label='Stored anchored GT')
        xy.scatter(gt[-1, 0, 3], gt[-1, 1, 3], color='#222222', marker='x', s=55, zorder=6)
        for dimension in (6, 8):
            model = models[dimension]; positions = model['poses'][:, :3, 3]
            xy.plot(positions[:, 0], positions[:, 1], color=COLORS[dimension], linestyle=STYLES[dimension],
                    linewidth=1.7, alpha=.95, label=f'{dimension}D ({model["rejected"]} rejected frames)')
            xy.scatter(positions[-1, 0], positions[-1, 1], color=COLORS[dimension],
                       marker='s' if dimension == 6 else 'D', s=44, edgecolors='white', linewidths=.6, zorder=7)
            gaps.plot(model['ids'], model['gaps'], color=COLORS[dimension], linestyle=STYLES[dimension],
                      linewidth=1.6, label=f'{dimension}D stored gap')
            rejected = ~model['accepted']
            gaps.scatter(model['ids'][rejected], model['gaps'][rejected], s=12, marker='x',
                         color=COLORS[dimension], alpha=.55, linewidths=.6)
        xy.scatter(gt[0, 0, 3], gt[0, 1, 3], color='#15936a', marker='^', s=80, zorder=8, label='Shared first camera')
        xy.set(title='XY projection of every stored camera center', xlabel='X / m', ylabel='Y / m')
        xy.set_aspect('equal', adjustable='box'); xy.grid(alpha=.22); xy.margins(.10)
        xy.legend(loc='best', framealpha=.93)
        gaps.set(title='Reference gap at every associated frame', xlabel='Associated frame ID',
                 ylabel='Current ID - held reference ID / frames', xlim=(0, n - 1))
        gaps.grid(alpha=.22); gaps.set_ylim(-.04 * max(1, max(int(m['gaps'].max()) for m in models.values())),
                                             max(int(m['gaps'].max()) for m in models.values()) * 1.22)
        if scene == 'desk':
            gaps.axvspan(134, 548, facecolor='#e8b5b5', alpha=.22, zorder=0)
            gaps.axvline(134, color='#925757', linestyle=':', linewidth=1)
            gaps.text(.97, .95, 'Frames 134-548: 415 consecutive rejects\nBoth models hold reference 133',
                      transform=gaps.transAxes, ha='right', va='top', fontsize=10,
                      bbox={'facecolor':'white', 'edgecolor':'#dcbaba', 'alpha':.96, 'pad':5})
            gaps.legend(loc='upper left', bbox_to_anchor=(0, .80), framealpha=.93)
        else:
            gaps.set_yticks([0, 1, 2, 3]); gaps.legend(loc='upper left', framealpha=.93)
        fig.text(.065, .103, f'All {n} frames / {n - 1} intervals retained. Rejected poses repeat the preceding estimate; overplotting is expected.',
                 fontsize=10, color='#444444')
        fig.text(.065, .070, 'Gap anchor = 0; x marks = rejected frame. Endpoint markers: GT x, 6D square, 8D diamond. No frame trimming or reanchoring.',
                 fontsize=10, color='#444444')
        fig.text(.065, .037, 'Per-scene plot limits; XY has equal metre scaling. XY projection hides Z. GT is the stored associated evaluation reference.',
                 fontsize=10, color='#444444')
        fig.savefig(destination, dpi=DPI, facecolor='white', metadata={'Software': 'Matplotlib read-only saved-trajectory figure'})
        plt.close(fig)


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as file:
        file.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='rgbd-affine-full-trajectory-figures-001')
    args = parser.parse_args()
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}', args.run) is not None, 'Safe exclusive output name')
    output = ROOT / 'outputs' / args.run; output.mkdir(exist_ok=False)
    inputs = Inputs(); source = inputs.read('scripts/plot_rgbd_affine_full_trajectories.py')
    (output / 'source_script.py').write_bytes(source)
    try:
        audit = inputs.json(FIELD_AUDIT, FIELD_AUDIT_SHA)
        require(audit['status'] == 'passed_independent_full_public_field_preservation_check' and
                audit['frames_checked'] == 2594 and audit['typed_cells_checked'] == 62256 and
                audit['summary_sha256'] == SUMMARY_SHA, 'Actual passed complete field audit')
        summary = inputs.json(PUBLIC + '/summary.json', SUMMARY_SHA)
        require(summary['status'] == 'completed_after_independent_full_result_audit', 'Completed original summary')
        figures = []
        for scene, n in [('desk', 549), ('xyz', 748)]:
            models, gt = load_scene(scene, n, summary, inputs)
            name = scene + '-full-trajectory-and-reference-gap.png'
            draw_scene(scene, n, models, gt, output / name)
            raw = (output / name).read_bytes()
            require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG output signature')
            dimensions = [int.from_bytes(raw[16:20], 'big'), int.from_bytes(raw[20:24], 'big')]
            require(dimensions == [2400, 1120], 'Declared output dimensions')
            figures.append({'scene': scene, 'file': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                            'bytes': len(raw), 'pixels': dimensions, 'associated_frames_per_model': n,
                            'rejected_frames_per_model': {str(dim): models[dim]['rejected'] for dim in (6, 8)},
                            'actual_gap_anchor': 0, 'stored_GT_equal_between_models': True,
                            'NPZ_estimated_cameras_equal_audited_compact': True})
        inputs.recheck()
        result = {'status': 'completed_readonly_saved_trajectory_figures',
                  'source_sha256': hashlib.sha256(source).hexdigest(), 'field_audit_sha256': FIELD_AUDIT_SHA,
                  'summary_sha256': SUMMARY_SHA, 'input_bytes_rehashed_unchanged': inputs.bindings,
                  'figures': figures, 'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
                  'registration_calls': 0, 'GT_metric_recalculations': 0, 'alignment_or_reanchoring_calls': 0,
                  'GPU_calls': 0, 'old_artifacts_modified': False,
                  'limits': ['All stored frames plotted, including coincident held estimates; line overlap is not row removal.',
                             'Uses c2w translation and pre-existing anchored GT directly; XY only, equal metre axes per scene.',
                             'Scenes have independent plot limits and trajectories; no cross-scene method ranking or speed comparison.',
                             'No raw/ATE/RPE or GT-derived candidate selection was computed.']}
        write_json(output / 'figures.json', result)
        print(json.dumps({'status': result['status'], 'run': args.run, 'figures': figures,
                          'manifest_sha256': hashlib.sha256((output / 'figures.json').read_bytes()).hexdigest()}))
    except BaseException as error:
        write_json(output / 'failure.json', {'status': 'readonly_plot_failed_partial_outputs_preserved',
                   'exception': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                   'input_bindings': inputs.bindings, 'GPU_calls': 0})
        raise


if __name__ == '__main__':
    main()

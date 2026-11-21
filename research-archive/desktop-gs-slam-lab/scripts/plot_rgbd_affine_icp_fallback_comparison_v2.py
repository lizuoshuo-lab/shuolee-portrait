"""Plot saved full-sequence errors and reference gaps; no GT metric calculation.

--prepare only reads/hash-locks inputs and writes a new CPU preflight.
--execute requires its exact SHA after root source review. No optimization,
registration, GT-matrix-based error calculation, fit, reanchoring, filtering or GPU.
Whole original JSON is decoded, including saved GT-derived alignment and
association fields; those matrices are not used to compute new errors.
The gap panel uses only stored 8D/hybrid gaps; ICP has no fabricated gap field.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/rgbd-affine-icp-fallback-figure-preparation-002'
OUTPUT = ROOT / 'outputs/rgbd-affine-icp-fallback-figures-001'
NEW = 'outputs/rgbd-affine-icp-fallback-comparison-001'
OLD = 'outputs/rgbd-affine-full-comparison-001'
ICP = 'outputs/tum-online-v4-full-sequence-recovery-summary-001'
NEW_SHA = '9701897445669ddbd2fa9b11d6140e2d253c9abc4cba4f33dc10619e3e1cbb3f'
NEW_AUDIT = '69b3856ca466d079c3dc485a6327d668aa88067d161b534b8f4d9c899da87715'
OLD_SHA = 'd6a5f9e304579e98b0b97912762eb6c7618273c89398bf3c95e09809e76635d3'
OLD_FIELDS = '45f3e5b378bea7340fa82472d87da57ed5e31d246210e5b99d062fa8595cb92d'
ICP_SHA = '97b65b413abd5b18eb8677a20ff7d1eef422205289d929d373f4ea5edd10c81d'
ICP_AUDIT = '70f65dfced84eabf1ec313e93a908062d565a19401b1ae4c49b0d4d71801417e'


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self):
        self.files = {}

    def raw(self, relative, expected=None):
        path = (ROOT / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(ROOT) and path.is_file(), 'Existing lab input')
        raw = path.read_bytes(); value = {'sha256': sha(raw), 'bytes': len(raw)}
        require(expected is None or value['sha256'] == expected, 'Frozen input: ' + relative)
        if relative in self.files:
            require(self.files[relative] == value, 'Read-again input unchanged')
        self.files[relative] = value
        return raw

    def json(self, relative, expected=None):
        return json.loads(self.raw(relative, expected))

    def recheck(self):
        for relative, binding in list(self.files.items()):
            self.raw(relative, binding['sha256'])


def read_inputs(inputs):
    source = inputs.raw('scripts/plot_rgbd_affine_icp_fallback_comparison_v2.py')
    new = inputs.json(NEW + '/summary.json', NEW_SHA)
    audit = inputs.json('outputs/rgbd-affine-icp-fallback-validation-001/independent_affine_icp_fallback_result_audit.json', NEW_AUDIT)
    old = inputs.json(OLD + '/summary.json', OLD_SHA)
    old_field = inputs.json('outputs/rgbd-affine-full-public-field-audit-001/metrics.json', OLD_FIELDS)
    old_icp = inputs.json(ICP + '/summary.json', ICP_SHA)
    old_icp_audit = inputs.json(ICP + '/independent_summary_audit.json', ICP_AUDIT)
    require(new['status'] == 'completed_after_actual_independent_hybrid_result_audit' and new['actual_independent_audit_sha256'] == NEW_AUDIT and
            audit['status'] == 'passed_independent_terminal_affine_ICP_fallback_result_audit', 'Actual completed new full-sequence audit')
    require(old_field['status'] == 'passed_independent_full_public_field_preservation_check' and old_field['summary_sha256'] == OLD_SHA and
            old_icp_audit['status'] == 'passed_independent_lineage_full_trajectory_chunk_and_tensor_audit' and old_icp_audit['summary_sha256'] == ICP_SHA,
            'Actual old full field audit and original complete ICP audit')
    ledger = inputs.json(ICP + '/input_hashes.json', old_icp['input_hashes_file_sha256'])
    old_hashes = {key.replace('\\', '/'): value for key, value in ledger['input_sha256'].items()}
    scenes = {}; checks = {}
    for scene, n in [('desk', 549), ('xyz', 748)]:
        current = new['scenes'][scene]['new_hybrid']; page_rows = []
        for index, page in enumerate(current['public_frame_pages']):
            data = inputs.json(NEW + '/' + page['path'], page['sha256'])
            require(data['scene'] == scene and data['page_index'] == index and len(data['rows']) == page['rows'], 'All original ordered pages')
            page_rows.extend(data['rows'])
        require(len(page_rows) == n and [row['frame']['frame_id'] for row in page_rows] == list(range(n)), 'All hybrid frames including any hold')
        old_arm = next(row for row in old['scene_results'][scene]['arms'] if row['model_dimension'] == 8)
        compact = inputs.json(OLD + '/' + old_arm['all_frame_file'], old_arm['all_frame_file_sha256'])
        old_rows = [dict(zip(compact['columns'], row)) for row in compact['rows']]
        require(len(old_rows) == n and [row['frame_id'] for row in old_rows] == list(range(n)), 'All original 8D frames including holds')
        icp_run = old_icp['scenes'][scene]['baseline']['run']
        metric_path = 'outputs/' + icp_run + '/metrics.json'
        metadata_path = 'outputs/' + icp_run + '/selected_sensor_metadata.json'
        icp = inputs.json(metric_path, old_hashes[metric_path])
        metadata = inputs.json(metadata_path, old_hashes[metadata_path])['selected_frames']
        require(icp['frames_processed'] == len(metadata) == n and [r['frame_id'] for r in icp['frame_metrics']] == list(range(n)),
                'Complete original ICP indices and metadata')
        require(metadata == [row['sensor_metadata'] for row in page_rows], 'Exact original ICP and hybrid sensor identities/timestamps')
        for i, row in enumerate(old_rows):
            require(all(row[key] == metadata[i][key] for key in ('rgb_index', 'depth_index', 'gt_index', 'rgb_timestamp')),
                    'Original 8D compact matches same associated sensor sequence')
        curves = {'old_ICP': icp['trajectory']['per_frame_raw_translation_error_m'],
                  'old_8D': [row['raw_translation_error_m'] for row in old_rows],
                  'hybrid': [row['stored_errors']['raw_translation_error_m'] for row in page_rows]}
        require(all(len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in v)
                    for v in curves.values()), 'Complete finite stored error vectors; no error recomputation')
        gaps = {'old_8D': [row['reference_gap_frames'] for row in old_rows],
                'hybrid': [row['frame']['reference_gap_frames'] for row in page_rows]}
        require(all(v[0] == 0 and len(v) == n and all(type(x) is int and x >= 0 for x in v) for v in gaps.values()), 'Stored gaps including anchor zero')
        old_rejected = [row['frame_id'] for row in old_rows if not row['accepted']]
        new_rejected = [row['frame']['frame_id'] for row in page_rows if not row['frame']['accepted']]
        fallback_ids = [row['frame']['frame_id'] for row in page_rows if row['frame']['fallback_selected']]
        require(old_rejected == old_arm['all_rejected_frame_ids'] and new_rejected == current['final_rejected_frame_ids'] and
                fallback_ids == current['fallback_selected_frame_ids'], 'Every existing rejection/selection retained')
        if scene == 'desk':
            require(all(i in old_rejected for i in range(134, 549)) and all(old_rows[i]['reference_frame_id'] == 133 for i in range(134, 549)),
                    'Annotated original415-hold segment really exists')
        # The old arm endpoint was copied from an independent recomputation;
        # compact error values are the ORIGINAL stored arrays. Keep those exact
        # plot values; compare the two provenances only at the old audit's 1e-11m
        # tolerance, rather than pretending these were identical float bytes.
        old_endpoint_difference_m = abs(curves['old_8D'][-1] - old_arm['last_camera']['endpoint_raw_translation_m'])
        require(old_endpoint_difference_m <= 1e-11 and curves['hybrid'][-1] == current['endpoint_raw_translation_m'],
                'Stored final error at actual final sensor time; original audited metre tolerance')
        scenes[scene] = {'n': n, 'curves': curves, 'gaps': gaps, 'old_rejected': old_rejected,
                         'new_rejected': new_rejected, 'fallback_ids': fallback_ids}
        checks[scene] = {'associated_frames_per_method': n, 'plotted_saved_error_values': n * 3, 'plotted_saved_gap_values': n * 2,
                         'old_8D_rejected_frames': len(old_rejected), 'hybrid_final_rejected_frames': len(new_rejected),
                         'hybrid_fallback_selected_frames': len(fallback_ids), 'complete_sensor_metadata_equal_to_original_ICP': True,
                         'final_sensor_timestamp': metadata[-1]['rgb_timestamp'], 'ICP_gap_not_invented': True,
                         'old_stored_array_vs_independently_audited_endpoint_difference_m': old_endpoint_difference_m,
                         'old_endpoint_check_absolute_tolerance_m': 1e-11,
                         'original_ICP_metrics_top_level_keys_loaded': list(icp),
                         'original_ICP_trajectory_keys_loaded': list(icp['trajectory']),
                         'sensor_metadata_keys_loaded': list(metadata[0]),
                         'saved_posthoc_alignment_matrix_present_in_decoded_JSON': 'alignment_estimated_to_ground_truth' in icp['trajectory'],
                         'saved_posthoc_alignment_matrix_used_to_plot_or_calculate': False}
    inputs.recheck()
    return source, scenes, checks


def draw(scene, data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'old_ICP': '#313131', 'old_8D': '#dc7929', 'hybrid': '#1764ad'}
    labels = {'old_ICP': 'Original full ICP', 'old_8D': 'Original full 8D', 'hybrid': '8D + adjacent ICP fallback'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'legend.fontsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, (error, gap) = plt.subplots(1, 2, figsize=(15, 7))
        fig.subplots_adjust(left=.065, right=.98, bottom=.21, top=.83, wspace=.25)
        fig.suptitle(f'TUM fr1_{scene}: saved full-sequence error and reference state', fontsize=16, fontweight='bold', y=.965)
        fig.text(.5, .91, f'{data["n"]} associated frames per method  |  Raw first-camera error  |  No fit, reanchoring or error recomputation', ha='center', fontsize=11)
        ids = list(range(data['n']))
        for method in ('old_ICP', 'old_8D', 'hybrid'):
            error.plot(ids, data['curves'][method], color=colors[method], linewidth=1.4,
                       linestyle='--' if method == 'old_8D' else '-', label=labels[method])
        error.scatter(data['fallback_ids'], [data['curves']['hybrid'][i] for i in data['fallback_ids']],
                      color=colors['hybrid'], marker='x', s=24, linewidths=.8, label='Hybrid selected fallback')
        error.set(title='Stored per-frame raw position error', xlabel='Associated frame ID', ylabel='Position error / m', xlim=(0, data['n'] - 1))
        error.set_ylim(bottom=0); error.grid(alpha=.22); error.legend(loc='best', framealpha=.93)
        for method in ('old_8D', 'hybrid'):
            gap.plot(ids, data['gaps'][method], color=colors[method], linewidth=1.5,
                     linestyle='--' if method == 'old_8D' else '-', label=labels[method])
        if scene == 'desk':
            for ax in (error, gap):
                ax.axvspan(134, 548, color='#c78888', alpha=.13, zorder=0)
                ax.axvline(134, color='#9e7777', linestyle=':', linewidth=1)
            gap.text(.98, .95, 'Old 8D: frames 134-548 rejected\n415 holds; reference remains 133', transform=gap.transAxes,
                     ha='right', va='top', fontsize=10, bbox={'facecolor':'white', 'edgecolor':'#ddbbbb', 'alpha':.94})
            gap.legend(loc='upper left', bbox_to_anchor=(0, .78), framealpha=.93)
        else:
            gap.set_yticks([0, 1, 2, 3]); gap.legend(loc='upper left', framealpha=.93)
        gap.set(title='Stored last-accepted reference gap', xlabel='Associated frame ID',
                ylabel='Current ID - reference ID / frames', xlim=(0, data['n'] - 1))
        gap.set_ylim(-.04 * max(data['gaps']['old_8D']), 1.22 * max(data['gaps']['old_8D'])); gap.grid(alpha=.22)
        fig.text(.065, .105, 'All frames and held estimates retained. Lines use saved errors; x markers identify actual selected fallback frames.', fontsize=10, color='#444444')
        fig.text(.065, .071, f'Hybrid: {len(data["fallback_ids"])} primary rejections / fallback selections; {len(data["new_rejected"])} final holds. Zero final holds does not establish accuracy.', fontsize=10, color='#444444')
        fig.text(.065, .037, 'Gap panel uses stored 8D/hybrid values only (anchor 0); no invented ICP gap. Per-scene axes; no speed or cross-scene ranking.', fontsize=10, color='#444444')
        fig.savefig(path, dpi=160, facecolor='white', metadata={'Software': 'Read-only saved-error comparison'})
        plt.close(fig)


def write_new(path, obj):
    raw = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(raw) < 1024 * 1024, 'Small evidence JSON without truncation')
    with path.open('xb') as stream:
        stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-sha')
    args = parser.parse_args()
    require(args.execute == (args.preflight_sha is not None), 'Execution requires exact reviewed preflight SHA')
    inputs = Inputs(); source, scenes, checks = read_inputs(inputs)
    preflight = {'status': 'prepared_readonly_saved_error_figures_no_plot_created', 'source_sha256': sha(source),
                 'input_sha256': inputs.files, 'scene_checks': checks, 'proposed_output': OUTPUT.relative_to(ROOT).as_posix(),
                 'figures': ['desk-full-raw-error-and-reference-gap.png', 'xyz-full-raw-error-and-reference-gap.png'],
                 'pixels_per_figure': [2400, 1120], 'new_metric_calculations': 0, 'new_GT_metric_calculations': 0, 'GT_camera_arrays_used_for_new_error_calculation': 0,
                 'GT_related_JSON_fields_loaded': True,
                 'loaded_GT_related_payload_scope': 'Original ICP metrics JSON includes trajectory.alignment_estimated_to_ground_truth, a saved posthoc 4x4 fit, plus existing GT-derived errors. Metadata includes GT indices/timestamps/quaternion norms. Whole JSON decoding loads these fields; no saved alignment matrix is applied and no GT camera sequence is loaded or used to calculate new errors.',
                 'historical_source_and_preflight_preserved': 'scripts/plot_rgbd_affine_icp_fallback_comparison.py and outputs/rgbd-affine-icp-fallback-figure-preparation-001/',
                 'registration_calls': 0, 'GPU_calls': 0, 'scope': 'Saved scalar arrays only; old full ICP, old full8D and actual hybrid have complete identical associated sensor metadata. No alignment or GT fitting; all rejected/held rows remain.'}
    if args.prepare:
        PREP.mkdir(exist_ok=False); (PREP / 'source_script.py').write_bytes(source)
        write_new(PREP / 'preflight.json', preflight)
        print(json.dumps({'status': preflight['status'], 'source_sha256': sha(source),
                          'preflight_sha256': sha((PREP / 'preflight.json').read_bytes()), 'input_files': len(inputs.files)}))
        return
    raw_preflight = (PREP / 'preflight.json').read_bytes()
    require(sha(raw_preflight) == args.preflight_sha and json.loads(raw_preflight) == preflight,
            'Actual reviewed preflight/source/input bytes unchanged')
    OUTPUT.mkdir(exist_ok=False); (OUTPUT / 'source_script.py').write_bytes(source)
    try:
        files = []
        for scene in ('desk', 'xyz'):
            name = scene + '-full-raw-error-and-reference-gap.png'; draw(scene, scenes[scene], OUTPUT / name)
            raw = (OUTPUT / name).read_bytes()
            require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'PNG signature')
            shape = [int.from_bytes(raw[16:20], 'big'), int.from_bytes(raw[20:24], 'big')]
            require(shape == [2400, 1120], 'Exact declared figure pixels')
            files.append({'scene': scene, 'path': name, 'bytes': len(raw), 'sha256': sha(raw), 'pixels': shape})
        inputs.recheck()
        write_new(OUTPUT / 'figures.json', {'status': 'completed_readonly_stored_error_figures', **{k:v for k,v in preflight.items() if k != 'status'},
                                          'preflight_sha256': args.preflight_sha, 'figures': files, 'all_input_bytes_unchanged': True})
        print(json.dumps({'status': 'completed_readonly_stored_error_figures', 'manifest_sha256': sha((OUTPUT / 'figures.json').read_bytes()), 'figures': files}))
    except BaseException as error:
        write_new(OUTPUT / 'failure.json', {'status': 'plot_failure_partial_outputs_preserved', 'type': type(error).__name__,
                  'message': str(error), 'traceback': traceback.format_exc(), 'old_results_modified': False})
        raise


if __name__ == '__main__':
    main()

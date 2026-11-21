"""CPU-only raw-input/array audit of20 fixed seen-view full-map renders.

Does not import evaluator, renderer, solver, loader or optimizer. Metric math
explicitly reuses reviewed aabd independent NumPy/math helper, not a third
formula. --self-check uses synthetic arrays only; actual --audit requires the
reviewed terminal20-render packet and writes a new result once with xb.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

import audit_v4_final_map_quality as independent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tum-online-v4-full-seen-map-evaluation-004'
PROTOCOL = ROOT / 'manifests/tum-online-v4-full-seen-map-evaluation-protocol-004.json'
PROTOCOL_SHA = '7f07d29ebbc96b9f96d48d84e97bc2f9c9c042186a70f1f8a5fa87e23c8f6e40'
EVALUATOR = ROOT / 'scripts/evaluate_tum_v4_full_seen_maps.py'
EVALUATOR_SHA = '41ee8343536096b38e6effbaa415a2fbcadc43ea7ffe7dcb5d73cdc9e51c41d2'
HELPER_SHA = 'aabd7c5197711567e679457ce6d203939252395945f558bc7e2b120f83b8f695'
TERMINAL_SHA = 'f2b0c23f57df18a968901f77dcba881425cc7062cbf194027d444e3808da7159'
VIEWS = {'desk': [0, 137, 274, 411, 548], 'xyz': [0, 187, 374, 561, 747]}
require, sha, read, array_record, compare = independent.require, independent.sha, independent.read, independent.array_record, independent.compare


def save(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'); require(len(raw) < 1024 * 1024, 'Audit JSON under1MiB without dropping rows')
    with path.open('xb') as stream: stream.write(raw)


def raw_rows(path): return [r.split() for r in path.read_text().splitlines() if r.strip() and not r.lstrip().startswith('#')]


def compatible_quaternion_pose(row):
    """Explicit normalized xyzw formula to reproduce declared input bytes.

Separate closed-form helper below cross-checks its geometry. Byte-compatible
arithmetic is not claimed as an additional independent quaternion convention.
"""
    values = np.asarray(row[1:], dtype=np.float64); q = values[3:]; x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([[1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)
    result = np.eye(4); result[:3, :3] = rotation; result[:3, 3] = values[:3]; return result


def association(dataset):
    rgb, depth, gt = [raw_rows(dataset / name) for name in ('rgb.txt', 'depth.txt', 'groundtruth.txt')]
    depth_ts = np.asarray([float(row[0]) for row in depth]); gt_ts = np.asarray([float(row[0]) for row in gt])
    def nearest(ts, value):
        right = int(np.searchsorted(ts, value)); choices = [i for i in (right - 1, right) if 0 <= i < len(ts)]
        return min(choices, key=lambda i: (abs(float(ts[i]) - value), i))
    used = set(); previous = -math.inf; threshold = 0; rate_skips = 0; reused = 0; selected = []
    for i, row in enumerate(rgb):
        stamp = float(row[0]); j, k = nearest(depth_ts, stamp), nearest(gt_ts, stamp)
        if abs(depth_ts[j] - stamp) >= .02 or abs(gt_ts[k] - stamp) >= .02: continue
        threshold += 1
        if stamp - previous <= 1 / 32: rate_skips += 1; continue
        if j in used: reused += 1; continue
        selected.append({'frame_id': len(selected), 'rgb_index': i, 'depth_index': j, 'gt_index': k,
                         'rgb_timestamp': stamp, 'depth_timestamp': float(depth_ts[j]), 'gt_timestamp': float(gt_ts[k]),
                         'rgb_depth_abs_dt_s': abs(stamp - float(depth_ts[j])), 'rgb_gt_abs_dt_s': abs(stamp - float(gt_ts[k])),
                         'rgb_path': row[1], 'depth_path': depth[j][1], 'gt_quaternion_norm_before_normalization': float(np.linalg.norm(np.asarray(gt[k][4:], dtype=float)))})
        used.add(j); previous = stamp
    stats = {'method': 'nearest depth and nearest GT at RGB timestamp; chronological filtering', 'depth_max_abs_dt_s_strict': .02,
             'gt_max_abs_dt_s_strict': .02, 'frame_rate_upper_limit_hz': 32., 'unique_depth_files': True, 'pose_interpolation': False,
             'raw_rgb_count': len(rgb), 'raw_depth_count': len(depth), 'raw_gt_count': len(gt), 'threshold_passes': threshold,
             'rate_skips': rate_skips, 'repeated_depth_skips': reused, 'associated_count': len(selected),
             'difference_from_splatam_default': '20 ms instead of 80 ms; prohibit depth reuse'}
    return selected, gt, stats


def inputs(scene, witness):
    dataset = ROOT / witness['dataset']; sensors, gt, stats = association(dataset)
    count = 549 if scene == 'desk' else 748
    require(len(sensors) == count and stats == witness['association'], 'Independent full association/denominators')
    chosen = [sensors[i] for i in VIEWS[scene]]; require(chosen == witness['evaluation_sensor_metadata'], 'All ten fixed raw-index/paired-time views')
    images = []; depths = []; masks = []
    for frame in chosen:
        with Image.open(dataset / frame['rgb_path']) as source:
            require(source.size == (640, 480), 'Actual raw RGB size')
            images.append(np.asarray(source.convert('RGB').resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255.))
        with Image.open(dataset / frame['depth_path']) as source:
            require(source.size == (640, 480), 'Actual raw depth size')
            depths.append(np.asarray(source.resize((160, 120), Image.Resampling.NEAREST), dtype=np.float32) / np.float32(5000.))
        masks.append(np.isfinite(depths[-1]) & (depths[-1] >= .2) & (depths[-1] <= 5.))
    rawK = np.array([[517.3, 0., 318.6], [0., 516.5, 255.3], [0., 0., 1.]])
    K = np.array([[.25, 0., -.375], [0., .25, -.375], [0., 0., 1.]]) @ rawK
    anchor = compatible_quaternion_pose(gt[sensors[0]['gt_index']]); absolute = np.stack([compatible_quaternion_pose(gt[f['gt_index']]) for f in chosen])
    relative = independent.inverse_rigid(anchor)[None] @ absolute; relative[:, 3, :] = [0., 0., 0., 1.]
    w2c = np.stack([independent.inverse_rigid(pose) for pose in relative])
    arrays = {'rgb': np.stack(images), 'depth': np.stack(depths), 'valid_depth_mask': np.stack(masks), 'K_render_float32': K.astype(np.float32)[None],
              'GT_anchor_c2w': anchor, 'GT_absolute_c2w': absolute, 'GT_relative_c2w': relative, 'GT_relative_w2c': w2c, 'render_w2c_float32': w2c.astype(np.float32)}
    require({k: array_record(v) for k, v in arrays.items()} == witness['input_tensors'], 'All18 independent sensor/camera array byte witnesses exact')
    compare(witness['K'], K.tolist(), scene + '/K_pixel_centers'); compare(witness['anchor_GT_c2w'], anchor.tolist(), scene + '/GT_anchor')
    alternative = np.stack([independent.independent_tum_pose(gt[f['gt_index']]) for f in chosen]); alternative_anchor = independent.independent_tum_pose(gt[sensors[0]['gt_index']])
    alternative_relative = independent.inverse_rigid(alternative_anchor)[None] @ alternative
    gauge_error = float(np.max(np.abs(relative - alternative_relative))); require(gauge_error < 1e-12, 'Distinct closed-form normalized xyzw/gauge cross-check')
    wrong_order_error = float(np.max(np.abs(absolute @ independent.inverse_rigid(anchor) - relative))); require(wrong_order_error > .001, 'Actual noncommuting wrong gauge order detectable')
    return arrays, sensors, {'scene': scene, 'association_count': count, 'evaluation_frame_ids': VIEWS[scene], 'all9_input_arrays_raw_bytes_equal_protocol': True,
                            'alternative_xyzw_gauge_max_abs_difference': gauge_error, 'wrong_gauge_order_max_abs_difference': wrong_order_error,
                            'valid_depth_pixels_per_view': [int(v.sum()) for v in arrays['valid_depth_mask']],
                            'depth_unit': 'raw/5000 meters, inclusive valid[.2,5]m', 'K_resize': 'u_new=.25*(u_old+.5)-.5; matrix translation-.375',
                            'role': 'All five views were seen during full-sequence training; known-GT-camera map diagnosis, no localization claim.'}


def aggregate(rows):
    good = [r['metrics'] for r in rows if r['status'] == 'completed']
    result = {'requested_views': 5, 'completed_views': len(good), 'failed_views': 5 - len(good), 'scope': 'successful views only; all requested rows retained'}
    if good:
        sse = math.fsum(r['rgb']['squared_error_sum'] for r in good); channels = sum(r['rgb']['channel_values'] for r in good)
        valid = sum(r['valid_depth_pixels'] for r in good); covered = sum(r['covered_valid_pixels'] for r in good)
        result['pooled'] = {'rgb': {'squared_error_sum': sse, 'channel_values': channels, **independent.psnr(sse / channels)},
                            'valid_depth_pixels': valid, 'covered_valid_pixels': covered, 'sensor_valid_coverage': covered / valid,
                            'covered_depth': independent.pooled_depth([r['covered_depth'] for r in good]),
                            'all_valid_holes_as_zero_depth': independent.pooled_depth([r['all_valid_holes_as_zero_depth'] for r in good])}
    return result


def common(a, b, observed, valid):
    depth_a, cover_a = a; depth_b, cover_b = b; selected = cover_a & cover_b
    da = independent.depth_statistics(depth_a, observed, selected); db = independent.depth_statistics(depth_b, observed, selected)
    n, c = int(valid.sum()), int(selected.sum())
    return {'sensor_valid_pixels': n, 'intersection_pixels': c, 'intersection_sensor_valid_coverage': c / n, 'prior0_depth': da, 'prior001_depth': db,
            'mae_difference_prior001_minus_prior0_m': db['mae_m'] - da['mae_m'] if c else None,
            'rmse_difference_prior001_minus_prior0_m': db['rmse_m'] - da['rmse_m'] if c else None}


def source_review(protocol):
    tree = ast.parse(EVALUATOR.read_text()); execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute'); text = ast.unparse(execute)
    for snippet in ('with torch.inference_mode()', "value.to('cuda').detach()", "result['parameters_after'] == result['parameters_before']",
                    "recovery.sha(ROOT / record['checkpoint']) == record['checkpoint_sha256']", "inputs[scene].items()", "protocol['datasets'][scene]['input_tensors']"):
        require(snippet in text, 'Read-only parameter/input guard source ' + snippet)
    require(not any(isinstance(n, ast.Attribute) and n.attr in ('backward', 'step') for n in ast.walk(execute)), 'No optimization calls')
    frozen = ast.parse((ROOT / 'scripts/train_tum_online_gs_v4.py').read_text()); main = next(n for n in frozen.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    render = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'render')
    require(hashlib.sha256(ast.dump(render, include_attributes=False).encode()).hexdigest() == protocol['source']['render_ast_sha256'], 'Frozen V4 renderer AST')
    return {'evaluation_script_sha256': EVALUATOR_SHA, 'frozen_render_AST_sha256': protocol['source']['render_ast_sha256'],
            'detached_upload_inference_mode_no_optimizer_and_parameter_checkpoint_input_beforeafter_guards_source_reviewed': True,
            'boundary': 'Audits saved before/after guard records and current checkpoint bytes, no GPU rerender or reconstruction of runtime memory.'}


def history_check(frozen, training, checkpoint):
    count = frozen['configuration']['frames']; rows = training['frame_metrics']; history = read(ROOT / frozen['history']['path'])
    require(training['configuration'] == checkpoint['configuration'] == frozen['configuration'] and len(rows) == len(checkpoint['completed_frame_metrics']) == count and
            checkpoint['completed_frame_metrics'] == rows and checkpoint['frame_ids_with_stored_pose'] == list(range(count)), 'Complete original causal training rows/config')
    require(checkpoint['ground_truth_included'] is False and not any('evaluation_only_gt' in k for k in checkpoint), 'Render checkpoint GT-free')
    compare({k: array_record(v) for k, v in checkpoint['parameters'].items()}, frozen['parameter_tensors'], frozen['run'] + '/actual_parameter_bytes')
    require(array_record(checkpoint['K']) == frozen['K_tensor'] and array_record(np.asarray(checkpoint['source_frame_ids_per_gaussian'], dtype=np.int64)) == frozen['source_frame_ids_tensor'], 'Actual K/sourceID bytes')
    ids = checkpoint['source_frame_ids_per_gaussian']; numeric_histogram = Counter(ids); histogram = {str(k): v for k, v in sorted(numeric_histogram.items())}; require(histogram == frozen['source_frame_histogram'] == history['source_frame_histogram'], 'All Gaussian source histogram retained')
    require(len(ids) == frozen['point_count'] and set(ids).issubset(range(count)), 'All Gaussian sources observed during training')
    retained = []; cumulative = training['initial_point_count']; flags = Counter(); rejected = []; first_cap = None
    for index, row in enumerate(rows):
        require(row['frame_id'] == row['local_index'] == index, 'All training IDs including failures')
        added = int(row['new_points_added']); cumulative += added if index else 0
        require((index or added == 0) and row['point_count'] == cumulative <= frozen['point_cap'], 'Exact cumulative points/60000 budget')
        require(numeric_histogram[index] == (added if index else training['initial_point_count']), 'Actual source histogram vs additions')
        if row.get('pose_rejected'): require(added == 0, 'Rejected pose cannot add source points'); rejected.append(index)
        flags.update(row['failure_flags'])
        if cumulative == frozen['point_cap'] and first_cap is None: first_cap = index
        retained.append({'frame_id': index, 'rgb_timestamp': row['rgb_timestamp'], 'new_points_added': added, 'point_count': row['point_count'], 'failure_flags': row['failure_flags'],
                         'map_coverage_warning_flags': row.get('map_coverage_warning_flags'), 'pose_rejected': row.get('pose_rejected'),
                         'optional_tracking_field_presence': {'map_coverage_warning_flags': 'map_coverage_warning_flags' in row, 'pose_rejected': 'pose_rejected' in row}})
    require(retained == history['all_frames_including_all_flags'] and history['training_frame_count'] == count and history['final_point_count'] == cumulative == frozen['point_count'], 'All2594 flags/source/point history exact')
    failures = sum(bool(r['failure_flags']) for r in rows); require(failures == training['failure_frame_count'] == frozen['training_failure_frame_count'] == history['training_failure_frame_count'], 'No training failure dropped')
    require(frozen['cap_reached'] == (first_cap is not None) and frozen['first_cap_frame'] == first_cap, 'Cap reached diagnostic exact')
    cameras = checkpoint['estimated_c2w'].numpy(); inverses = checkpoint['estimated_w2c'].numpy(); inverse_error = float(np.max(np.abs(cameras @ inverses - np.eye(4))))
    require(np.array_equal(cameras[0], np.eye(4)) and inverse_error < 2e-6 and array_record(checkpoint['K']) == frozen['K_tensor'], 'Frame0 identity and canonical camera inverses')
    return {'run': frozen['run'], 'training_frame_count': count, 'all_history_flags_optional_absence_and_source_counts_exact': True, 'failure_frame_count': failures,
            'failure_flag_counts': dict(flags), 'rejected_frame_ids': rejected, 'final_point_count': cumulative, 'point_cap': frozen['point_cap'], 'cap_reached': first_cap is not None,
            'first_cap_frame': first_cap, 'camera_inverse_max_abs_difference': inverse_error, 'all5_frozen_parameter_byte_records_exact': True}


def controls():
    base = independent.cpu_controls(); rgb = np.zeros((120, 160, 3), np.float32); depth = np.full((120, 160), 2., np.float32); valid = np.zeros((120, 160), bool); valid[0, :3] = True
    alpha = np.zeros((120, 160), np.float32); alpha[0, :3] = [.8, .5, .8]
    measured, selected = independent.measure(rgb, depth, alpha, rgb, depth, valid)
    require(measured['covered_valid_pixels'] == 2 and measured['rgb']['psnr_db'] is None and measured['rgb']['psnr_positive_infinity'] and measured['all_valid_holes_as_zero_depth']['squared_error_sum_m2'] == 4., 'Fullshape zero RGB/strict .5/holes hand control')
    empty, empty_c = independent.measure(rgb, depth, np.zeros_like(alpha), rgb, depth, valid)
    require(empty['covered_depth']['pixels'] == 0 and empty['covered_depth']['rmse_m'] is None and empty['all_valid_holes_as_zero_depth']['rmse_m'] == 2., 'EmptyC denominator/null preserved')
    other = np.zeros_like(selected); other[0, 1] = True; common_stats = common((depth, selected), (depth, other), depth, valid)
    require(common_stats['intersection_pixels'] == 0 and common_stats['prior0_depth']['rmse_m'] is None and common_stats['rmse_difference_prior001_minus_prior0_m'] is None, 'Disjoint commonC null hand control')
    views = []
    for value in (.1, .2):
        measured, _ = independent.measure(np.full_like(rgb, value), depth, alpha, rgb, depth, valid); views.append({'status': 'completed', 'metrics': measured})
    views += [{'status': 'failed'}] * 3; pool = aggregate(views)
    require(pool['requested_views'] == 5 and pool['completed_views'] == 2 and pool['failed_views'] == 3, 'Failed full requested denominator retained')
    mean_psnr = independent.mean([r['metrics']['rgb']['psnr_db'] for r in views[:2]])
    require(abs(pool['pooled']['rgb']['psnr_db'] - mean_psnr) > .9, 'Pooled PSNR is not mean view dB')
    require(not torch.cuda.is_initialized() and 'gsplat' not in sys.modules, 'No GPU/backend import')
    return {'status': 'passed_CPU_synthetic_only', 'reused_aabd_math_controls': base, 'strict05_holes_zeroRGB_empty_and_disjoint_commonC_controls': True,
            'failed_views_retained': 3, 'pooled_RGB_psnr_db': pool['pooled']['rgb']['psnr_db'], 'mean_defined_view_psnr_db': mean_psnr, 'GPU_calls': 0}


def audit():
    require((OUT / 'metrics.json').exists() and not (OUT / 'independent_full_seen_map_audit.json').exists(), 'Only terminal actual evidence/new audit')
    require(sha(PROTOCOL) == PROTOCOL_SHA and sha(EVALUATOR) == EVALUATOR_SHA and sha(Path(independent.__file__)) == HELPER_SHA, 'Locked reviewed evaluator/protocol/independent math')
    protocol = read(PROTOCOL); metrics = read(OUT / 'metrics.json'); require(sha(OUT / 'metrics.json') == TERMINAL_SHA, 'Mapping reported terminal metrics SHA')
    require(metrics['status'] in ('completed', 'completed_with_render_failures') and metrics['render_attempts'] == 20 and len(metrics['maps']) == 4 and len(metrics['pairs']) == 2 and
            metrics['successful_render_count'] + metrics['failed_render_count'] == 20 and metrics['optimization_steps'] == 0 and metrics['protocol_sha256'] == PROTOCOL_SHA, 'All terminal requested frames/failed denominator preserved')
    require(metrics['input_tensor_bytes_before_after_unchanged'] and metrics['original_map_and_failure_artifacts_unchanged'], 'Persisted unchanged input/source guards')
    require(datetime.fromisoformat(metrics['GPU_started_at']['utc']) < datetime.fromisoformat(protocol['deadline_beijing']), 'Actual GPU start before fixed cutoff')
    watched = {}
    def watch(path, expected=None):
        path = path.resolve(); require(path.is_relative_to(ROOT), 'Safe evidence within lab'); digest = sha(path); require(expected is None or digest == expected, 'Actual hash ' + str(path)); watched[path] = digest
    watch(PROTOCOL, PROTOCOL_SHA); watch(EVALUATOR, EVALUATOR_SHA); watch(Path(independent.__file__), HELPER_SHA); watch(OUT / 'metrics.json', TERMINAL_SHA)
    for name, digest in protocol['source_sha256'].items(): watch(ROOT / name, digest)
    for name, digest in protocol['lineage']['evidence_sha256'].items(): watch(ROOT / name, digest)
    for name, digest in protocol['historical_CPU_evidence_sha256'].items(): watch(ROOT / name, digest)
    watch(ROOT / protocol['cpu_preflight']['path'], protocol['cpu_preflight']['sha256'])
    require(protocol['fixed_views'] == VIEWS and protocol['evaluation_role'] == metrics['evaluation_role'] == 'seen_training_view_diagnostic', 'Fixed full seen role, not heldout')
    source = source_review(protocol); cpu = controls(); arrays = {}; data_checks = []; map_checks = []; predictions = {}; raw_payloads = []; nviews = 0; nhistory = 0
    for scene, witness in protocol['datasets'].items():
        for name, digest in witness['input_file_sha256'].items(): watch(ROOT / name, digest)
        arrays[scene], _, checked = inputs(scene, witness); data_checks.append(checked)
    for frozen, actual in zip(protocol['maps'], metrics['maps']):
        require(all(actual[k] == frozen[k] for k in ('run', 'scene', 'prior_weight', 'point_count', 'history', 'training_failure_frame_count')), 'Map/arm/failure/history actual identity')
        for name, digest in frozen['artifact_sha256'].items(): watch(ROOT / name, digest)
        watch(ROOT / frozen['history']['path'], frozen['history']['sha256'])
        checkpoint = torch.load(ROOT / frozen['checkpoint'], map_location='cpu', weights_only=True); training = read((ROOT / frozen['checkpoint']).parent / 'metrics.json')
        checked = history_check(frozen, training, checkpoint); nhistory += checked['training_frame_count']
        require(actual['parameters_before'] == actual['parameters_after'] == frozen['parameter_tensors'] and actual['parameter_bytes_unchanged'], 'All frozen parameters before/after SHA records exact')
        scene = frozen['scene']; require(array_record(checkpoint['K']) == array_record(arrays[scene]['K_render_float32']), 'Actual map and independent pixel center K')
        require([v['frame_id'] for v in actual['views']] == VIEWS[scene], 'All5 fixed requested view rows retained')
        rows = []; predictions[frozen['run']] = {}
        for index, view in enumerate(actual['views']):
            nviews += 1
            require(view['rgb_timestamp'] == protocol['datasets'][scene]['evaluation_sensor_metadata'][index]['rgb_timestamp'] and view['role'] == 'seen_training_view_diagnostic', 'Actual fixed view paired timestamp/role')
            if view['status'] != 'completed': require(view['status'] == 'failed' and 'exception_type' in view, 'Failed render retained, never zero filled'); rows.append({'status': 'failed'}); continue
            file = ROOT / view['prediction']['path']; require(file.resolve().is_relative_to((OUT / 'raw_predictions').resolve()), 'Safe actual NPZ path'); watch(file, view['prediction']['sha256'])
            with np.load(file, allow_pickle=False) as packet:
                require(set(packet.files) == {'rgb', 'depth_ED_m', 'alpha'}, 'Exact3 raw prediction channels')
                payload = {k: packet[k].copy() for k in packet.files}
            require(all(v.dtype == np.float32 for v in payload.values()), 'Raw float32 unmodified render channels')
            raw_payloads.append({'run': frozen['run'], 'frame_id': view['frame_id'], 'NPZ_sha256': view['prediction']['sha256'], 'arrays': {k: array_record(v) for k, v in payload.items()}})
            measured, covered = independent.measure(payload['rgb'], payload['depth_ED_m'], payload['alpha'], arrays[scene]['rgb'][index], arrays[scene]['depth'][index], arrays[scene]['valid_depth_mask'][index])
            compare(view['metrics'], measured, frozen['run'] + '/' + str(view['frame_id']) + '/ownC_RGB_holes'); rows.append({'status': 'completed', 'metrics': measured})
            predictions[frozen['run']][view['frame_id']] = payload['depth_ED_m'], covered
        pooled = aggregate(rows); compare(actual['aggregate'], pooled, frozen['run'] + '/pooled')
        good = [r['metrics'] for r in rows if r['status'] == 'completed']
        checked.update(aggregate_recomputed=pooled, completed_render_count=len(good), failed_render_count=5 - len(good),
                       diagnostic_mean_finite_per_view_PSNR_db=independent.defined_mean([r['rgb']['psnr_db'] for r in good]),
                       diagnostic_any_view_PSNR_positive_infinity=any(r['rgb']['psnr_positive_infinity'] for r in good))
        map_checks.append(checked)
    paired = []
    for pair in metrics['pairs']:
        scene = pair['scene']; require([v['frame_id'] for v in pair['views']] == VIEWS[scene], 'All5 commonC rows')
        selected = [f for f in protocol['maps'] if f['scene'] == scene]; require([pair['prior0_run'], pair['prior001_run']] == [f['run'] for f in selected], 'Declared same-scene weight pair')
        rows = []
        for index, view in enumerate(pair['views']):
            frame = view['frame_id']; a = predictions[pair['prior0_run']].get(frame); b = predictions[pair['prior001_run']].get(frame)
            if a is None or b is None: require(view['status'] == 'unavailable_due_to_retained_render_failure', 'Unavailable commonC denominator retained'); continue
            require(view['status'] == 'completed', 'Available pair completed')
            measured = common(a, b, arrays[scene]['depth'][index], arrays[scene]['valid_depth_mask'][index]); compare(view['metrics'], measured, scene + '/' + str(frame) + '/commonC'); rows.append(measured)
        pooled = {'completed_paired_views': len(rows), 'requested_paired_views': 5}
        if rows:
            valid = sum(r['sensor_valid_pixels'] for r in rows); common_count = sum(r['intersection_pixels'] for r in rows)
            pooled.update(sensor_valid_pixels=valid, intersection_pixels=common_count, intersection_sensor_valid_coverage=common_count / valid,
                          prior0_depth=independent.pooled_depth([r['prior0_depth'] for r in rows]), prior001_depth=independent.pooled_depth([r['prior001_depth'] for r in rows]))
        compare(pair['pooled_common_C'], pooled, scene + '/pooled_commonC'); paired.append({'scene': scene, 'all5_requested_views_retained': True, 'pooled_commonC_recomputed': pooled})
    require(nviews == 20 and nhistory == 2594 and len(raw_payloads) == metrics['successful_render_count'] and not torch.cuda.is_initialized() and 'gsplat' not in sys.modules, 'Complete CPU-only actual audit')
    require(all(sha(path) == digest for path, digest in watched.items()), 'Every consumed raw/source/checkpoint/NPZ unchanged')
    result = {'status': 'passed_independent_full_seen_map_audit', 'audit_source_sha256': sha(Path(__file__)), 'independent_math_helper_sha256': HELPER_SHA,
              'evaluation_script_sha256': EVALUATOR_SHA, 'protocol_sha256': PROTOCOL_SHA, 'metrics_sha256': TERMINAL_SHA, 'source_review': source, 'CPU_controls': cpu,
              'map_count': 4, 'requested_view_count': 20, 'actual_NPZ_count': len(raw_payloads), 'actual_prediction_payload_array_count': 3 * len(raw_payloads),
              'all_training_rows_including_flags': nhistory, 'requested_commonC_views': 10, 'datasets': data_checks, 'maps': map_checks, 'pairs': paired,
              'raw_prediction_payload_hash_records': raw_payloads, 'numeric_scalar_comparison_count': len(independent.DIFFERENCES),
              'maximum_absolute_numeric_scalar_difference': max(r['absolute_difference'] for r in independent.DIFFERENCES),
              'largest10_numeric_scalar_differences': sorted(independent.DIFFERENCES, key=lambda r: r['absolute_difference'], reverse=True)[:10],
              'consumed_files_rehashed_unchanged': len(watched), 'GPU_calls': 0,
              'limits': ['Explicit aabd independent metric helper reuse, not a third formula; no evaluator/render metric functions, optimization, registration or GPU.',
                         'All five views were seen during full-sequence map training: known-GT-camera map diagnosis, not heldout generalization or localization accuracy.',
                         'Exact raw bytes use explicit compatible quaternion arithmetic, cross-checked with distinct closed form; known gauge is inverse(GT0)@GTview with no fitted alignment.',
                         'NPZ file SHA binds all3 raw array payloads; this audit computes raw payload SHAs, not a second GPU prediction or backend parity.',
                         'Parameter/input guards are source-reviewed and saved before/after/current bytes checked; no reconstruction of GPU memory/execution causality.',
                         'Own-C/common-C depths are conditional; holes-as-zero use all sensor-valid pixels. Empty/null and zeroMSE/infinity controls remain explicit.',
                         'Retained original failure and explicit frame0 replacement are lineage-bound. No selection by metric, retry, threshold tuning, cross-scene average or speed claim.']}
    save(OUT / 'independent_full_seen_map_audit.json', result)
    print(json.dumps({'status': result['status'], 'map_count': 4, 'view_count': 20, 'NPZ_arrays': 3 * len(raw_payloads), 'training_rows': nhistory,
                      'max_scalar_difference': result['maximum_absolute_numeric_scalar_difference']}))


def main():
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true'); mode.add_argument('--audit', action='store_true'); p.add_argument('--run', default='v4-full-seen-map-independent-auditor-cpu-001'); args = p.parse_args()
    if args.self_check:
        require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique simple CPU output')
        folder = ROOT / 'outputs' / args.run; folder.mkdir(exist_ok=False); value = controls(); value.update(audit_source_sha256=sha(Path(__file__)), independent_math_helper_sha256=sha(Path(independent.__file__)), actual_map_audits=0)
        save(folder / 'metrics.json', value); (folder / 'source_script.py').write_bytes(Path(__file__).read_bytes()); print(json.dumps({'status': value['status'], 'source_sha256': value['audit_source_sha256']}))
    else: audit()


if __name__ == '__main__': main()

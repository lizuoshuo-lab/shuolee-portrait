"""CPU artifact audit of SAME frozen map under saved-estimated versus GT cameras.

No evaluator, loader, renderer or optimizer is imported. Arithmetic explicitly
reuses reviewed aabd NumPy/math helper; raw association/PIL/GT gauge routines
are adapted from the earlier0204 independent audit, not evaluator functions.
--self-check uses synthetic arrays only; --audit waits for root authorization.
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
import traceback

import numpy as np
from PIL import Image
import torch
import audit_v4_final_map_quality as independent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tum-online-v4-estimated-seen-map-evaluation-002'
PROTOCOL = ROOT / 'manifests/tum-online-v4-estimated-seen-map-evaluation-protocol-002.json'
PROTOCOL_SHA = '124571c2835445a08e696e9a08f4f64495a7fb88aa35072f260404cefa8b2094'
EVALUATOR = ROOT / 'scripts/evaluate_tum_v4_estimated_seen_maps.py'
EVALUATOR_SHA = '7dfe57310c334d5e917dee5e12e0a68f9d83381b1096aeb00d0fc29bfa1bab74'
HELPER_SHA = 'aabd7c5197711567e679457ce6d203939252395945f558bc7e2b120f83b8f695'
TERMINAL_SHA = 'daec22d74d01c8f208907817b1067cbe61f8a6eb9ff9c23db53eca773d11b14f'
GT_OUT = ROOT / 'outputs/tum-online-v4-full-seen-map-evaluation-004'
GT_METRICS_SHA = 'f2b0c23f57df18a968901f77dcba881425cc7062cbf194027d444e3808da7159'
GT_AUDIT_SHA = '834ab16cea0efb298c44c512d3a7e0c5497c43c4075ad8caf44941a2a183ae65'
GT_PROTOCOL_SHA = '7f07d29ebbc96b9f96d48d84e97bc2f9c9c042186a70f1f8a5fa87e23c8f6e40'
OLD_AUDITOR_SHA = '020451555333f7137aa568cd8792db9e28818aba6e612d15bb87458b91271abe'
VIEWS = {'desk': [0, 137, 274, 411, 548], 'xyz': [0, 187, 374, 561, 747]}
require, sha, read, array_record, compare = independent.require, independent.sha, independent.read, independent.array_record, independent.compare


def save(path, obj):
    data = (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(data) < 1024 * 1024, 'Evidence JSON under1MiB without dropping requested cases')
    with path.open('xb') as stream: stream.write(data)



# Raw input routines adapted from locked0204 auditor; no runtime import of it.
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


def not_imported():
    forbidden = ('evaluate_tum_v4_estimated_seen_maps', 'evaluate_tum_v4_full_seen_maps', 'evaluate_tum_v4_final_maps',
                 'tum_rgbd', 'gsplat', 'train_tum_online_gs_v4', 'track_rgbd_icp')
    require(not any(k in sys.modules for k in forbidden) and not torch.cuda.is_initialized(), 'No evaluator/loader/backend/solver import or CUDA initialization')


def rigid(poses):
    p = np.asarray(poses, dtype=np.float64); require(p.shape[-2:] == (4, 4) and np.isfinite(p).all(), 'Finite rigid camera')
    r = p[..., :3, :3]
    result = {'maximum_orthogonality_element_error': float(np.max(np.abs(np.swapaxes(r, -2, -1) @ r - np.eye(3)))),
              'maximum_determinant_error': float(np.max(np.abs(np.linalg.det(r) - 1.))),
              'maximum_bottom_row_error': float(np.max(np.abs(p[..., 3, :] - [0, 0, 0, 1]))), 'absolute_tolerance': 2e-6}
    require(max(result[k] for k in result if k != 'absolute_tolerance') < 2e-6, 'Proper saved SO3/homogeneous camera')
    return result


def camera_common(gt, est, observed, valid):
    gt_depth, gt_cover = gt; est_depth, est_cover = est; selected = gt_cover & est_cover
    require(np.all(~selected | valid), 'Same-map commonC inside sensor-valid support')
    a = independent.depth_statistics(gt_depth, observed, selected); b = independent.depth_statistics(est_depth, observed, selected)
    n, c = int(valid.sum()), int(selected.sum()); require(n > 0, 'Sensor-valid denominator nonzero')
    return {'sensor_valid_pixels': n, 'intersection_pixels': c, 'intersection_sensor_valid_coverage': c / n,
            'GT_camera_depth': a, 'saved_estimated_camera_depth': b,
            'mae_difference_estimated_minus_GT_m': b['mae_m'] - a['mae_m'] if c else None,
            'rmse_difference_estimated_minus_GT_m': b['rmse_m'] - a['rmse_m'] if c else None}


def pooled_common(rows):
    good = [r['metrics'] for r in rows if r['status'] == 'completed']
    result = {'requested_same_map_paired_views': 5, 'completed_same_map_paired_views': len(good),
              'scope': 'successful pairs only; every requested failed row retained'}
    if good:
        n, c = sum(r['sensor_valid_pixels'] for r in good), sum(r['intersection_pixels'] for r in good)
        result.update(sensor_valid_pixels=n, intersection_pixels=c, intersection_sensor_valid_coverage=c / n,
                      GT_camera_depth=independent.pooled_depth([r['GT_camera_depth'] for r in good]),
                      saved_estimated_camera_depth=independent.pooled_depth([r['saved_estimated_camera_depth'] for r in good]))
    return result


def source_review(protocol):
    tree = ast.parse(EVALUATOR.read_text())
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    execute = nodes['execute']; text = ast.unparse(execute); accepted = ast.unparse(nodes['accepted_camera_inputs'])
    for snippet in ("w2c_array = checkpoint['render_saved_accepted_w2c_float32'][index]", 'torch.from_numpy(w2c_array)',
                    'with torch.inference_mode()', "result['parameters_after'] == result['parameters_before']",
                    "recovery.sha(ROOT / record['checkpoint']) == record['checkpoint_sha256']",
                    "assert_camera_record(checkpoint[name], record['camera_tensors'][name], module)",
                    "inputs[scene].items()", "protocol['datasets'][scene]['input_tensors']"):
        require(snippet in text, 'Actual saved-camera/map/input beforeafter source guard ' + snippet)
    for snippet in ("checkpoint['estimated_w2c'].numpy()", 'selected_w2c = w2c[frame_ids].copy()', 'inverse_error < 2e-06', 'differences[0] < 1e-12'):
        require(snippet in accepted, 'Original accepted-camera exact payload/direction source ' + snippet)
    require(not any(isinstance(n, ast.Attribute) and n.attr in ('backward', 'step') for n in ast.walk(execute)), 'No evaluator optimizer')
    frozen = ast.parse((ROOT / 'scripts/train_tum_online_gs_v4.py').read_text()); main = next(n for n in frozen.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    render = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'render')
    require(hashlib.sha256(ast.dump(render, include_attributes=False).encode()).hexdigest() == protocol['source']['render_ast_sha256'], 'Original frozen V4 render AST')
    return {'frozen_render_ast_sha256': protocol['source']['render_ast_sha256'], 'exact_saved_w2c_source_not_recomputed_or_fit': True,
            'inference_no_optimizer_parameter_checkpoint_camera_input_guards_source_reviewed': True,
            'boundary': 'Static source and persisted/current byte checks; no rerender or GPU runtime-memory reconstruction.'}


def camera_check(record, checkpoint, arrays):
    count = record['configuration']['frames']; ids = VIEWS[record['scene']]
    require(checkpoint['ground_truth_included'] is False and not any('evaluation_only_gt' in k.lower() for k in checkpoint), 'Causal camera source no eval GT')
    require(checkpoint['frame_ids_with_stored_pose'] == list(range(count)), 'All causal camera IDs')
    c2w, w2c = checkpoint['estimated_c2w'].numpy(), checkpoint['estimated_w2c'].numpy()
    require(c2w.dtype == w2c.dtype == np.float32 and c2w.shape == w2c.shape == (count, 4, 4) and
            not checkpoint['estimated_c2w'].requires_grad and not checkpoint['estimated_w2c'].requires_grad, 'Exact detached float32 camera source')
    selected_c2w, selected_w2c = c2w[ids].copy(), w2c[ids].copy()
    records = {'saved_all_c2w': array_record(c2w), 'saved_all_w2c': array_record(w2c),
               'selected_accepted_c2w': array_record(selected_c2w), 'render_saved_accepted_w2c_float32': array_record(selected_w2c)}
    require(records == record['camera_tensors'], 'Actual four saved-camera payload SHA records exact')
    require(array_record(np.asarray(record['selected_accepted_c2w'], dtype=np.float32)) == records['selected_accepted_c2w'] and
            array_record(np.asarray(record['render_saved_accepted_w2c_float32'], dtype=np.float32)) == records['render_saved_accepted_w2c_float32'], 'Protocol float32 camera payload/list exact')
    inverse_error = float(np.max(np.abs(c2w.astype(np.float64) @ w2c.astype(np.float64) - np.eye(4))))
    checks = {'all_saved_c2w': rigid(c2w), 'all_saved_w2c': rigid(w2c), 'selected_c2w': rigid(selected_c2w), 'selected_w2c': rigid(selected_w2c),
              'maximum_saved_inverse_product_error': inverse_error}
    compare(record['camera_checks'], checks, record['run'] + '/saved_camera_structure')
    require(inverse_error < 2e-6 and np.array_equal(c2w[0], np.eye(4)) and np.array_equal(w2c[0], np.eye(4)), 'Causal identity gauge/inverse direction')
    differences = [float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) for x, y in zip(selected_w2c, arrays['render_w2c_float32'])]
    compare(record['GT_vs_saved_estimated_w2c_max_absolute_difference_per_fixed_view'], differences, record['run'] + '/GT_vs_saved_camera')
    require(differences[0] < 1e-12 and max(differences[1:]) > 1e-5 and record['actual_GT_camera_substitution_rejected'] and
            array_record(arrays['render_w2c_float32']) != records['render_saved_accepted_w2c_float32'], 'GT-substitution must be detectably wrong across five views')
    return selected_c2w, selected_w2c, {'run': record['run'], 'full_camera_count': count, 'actual_camera_payload_records': records,
                                     'GT_vs_saved_w2c_max_difference_per_view': differences, 'camera_structure_recomputed': checks}


def controls(folder):
    base = independent.cpu_controls(); rgb = np.zeros((120, 160, 3), np.float32); target = np.full((120, 160), 2., np.float32)
    valid = np.zeros((120, 160), bool); valid[0, :3] = True
    gt_depth = target.copy(); gt_depth[0, 2] = 2.5
    est_depth = target.copy(); est_depth[0, 1:3] = 3.
    ag, ae = np.zeros_like(target), np.zeros_like(target); ag[0, :3] = [.6, .5, .6]; ae[0, :3] = [.5, .7, .6]
    g, cg = independent.measure(rgb, gt_depth, ag, rgb, target, valid); e, ce = independent.measure(rgb, est_depth, ae, rgb, target, valid)
    common = camera_common((gt_depth, cg), (est_depth, ce), target, valid)
    require(g['covered_valid_pixels'] == e['covered_valid_pixels'] == 2 and common['intersection_pixels'] == 1 and
            common['GT_camera_depth']['rmse_m'] == .5 and common['saved_estimated_camera_depth']['rmse_m'] == 1. and
            g['all_valid_holes_as_zero_depth']['squared_error_sum_m2'] == 4.25 and e['all_valid_holes_as_zero_depth']['squared_error_sum_m2'] == 6., 'Same-map labelled commonC/ownC/holes hand sums')
    require(common['mae_difference_estimated_minus_GT_m'] == .5 and g['rgb']['psnr_positive_infinity'] and g['rgb']['psnr_db'] is None, 'Camera label sign and zeroMSE')
    empty = camera_common((gt_depth, cg), (est_depth, np.zeros_like(ce)), target, valid)
    require(empty['intersection_pixels'] == 0 and empty['GT_camera_depth']['rmse_m'] is None and empty['rmse_difference_estimated_minus_GT_m'] is None, 'Empty commonC null not zero error')
    rows = []
    for v in [.1, .2]:
        metrics, _ = independent.measure(np.full_like(rgb, v), gt_depth, ag, rgb, target, valid); rows.append({'status': 'completed', 'metrics': metrics})
    rows += [{'status': 'failed'}] * 3; pooled = aggregate(rows)
    mean_db = independent.mean([r['metrics']['rgb']['psnr_db'] for r in rows[:2]])
    require(pooled['requested_views'] == 5 and pooled['failed_views'] == 3 and abs(pooled['pooled']['rgb']['psnr_db'] - mean_db) > .9, 'Failures denominator and pooled-vs-mean PSNR')
    pair_pool = pooled_common([{'status': 'completed', 'metrics': common}] + [{'status': 'unavailable_due_to_retained_estimated_camera_render_failure'}] * 4)
    require(pair_pool['completed_same_map_paired_views'] == 1 and pair_pool['requested_same_map_paired_views'] == 5, 'Failed same-map pair denominator')
    camera = np.eye(4, dtype=np.float32); camera[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; camera[:3, 3] = [1, 2, 3]
    inverse = independent.inverse_rigid(camera.astype(np.float64)); point = np.array([3., 5., 7., 1.]); local = inverse @ point
    require(np.allclose(camera @ local, point, atol=1e-14) and np.max(np.abs(camera @ point - local)) > 1., 'c2w/w2c direction negative control')
    signed = camera.copy(); signed[0, 0] = np.float32(-0.)
    require(np.array_equal(camera, signed) and array_record(camera) != array_record(signed), 'Raw-byte sign-zero guard stronger than numerical equal')
    selected = np.stack([camera, np.eye(4, dtype=np.float32)])
    require(array_record(np.asarray(selected.tolist(), dtype=np.float32)) == array_record(selected), 'Float32 list roundtrip exact with signed zero')
    fixture = folder / 'synthetic-association'; fixture.mkdir()
    (fixture / 'rgb.txt').write_text('1.0 rgb/a.png\n1.01 rgb/b.png\n1.04 rgb/c.png\n1.08 rgb/d.png\n')
    (fixture / 'depth.txt').write_text('1.0 depth/a.png\n1.04 depth/b.png\n1.08 depth/c.png\n')
    (fixture / 'groundtruth.txt').write_text('1.0 0 0 0 0 0 0 1\n1.04 0 0 0 0 0 0 1\n1.08 0 0 0 0 0 0 1\n')
    sensors, _, stats = association(fixture)
    require([s['rgb_index'] for s in sensors] == [0, 2, 3] and stats['rate_skips'] == 1 and stats['associated_count'] == 3, 'Synthetic chronological nearest/rate support')
    np.savez_compressed(folder / 'synthetic_arrays.npz', rgb=rgb, target=target, valid=valid, gt_depth=gt_depth, estimated_depth=est_depth,
                        GT_alpha=ag, estimated_alpha=ae, selected_cameras=selected, signed_zero_camera=signed)
    not_imported()
    return {'status': 'passed_CPU_synthetic_only', 'reused_aabd_math_controls': base, 'same_map_camera_commonC_hand_metrics': common,
            'strict05_ownC_holes_and_empty_commonC': True, 'failed_views_retained': 3, 'failed_common_pairs_retained': 4,
            'pooled_psnr_db': pooled['pooled']['rgb']['psnr_db'], 'mean_view_psnr_db': mean_db,
            'camera_direction_and_raw_signed_zero_negative_controls': True, 'synthetic_association': stats, 'actual_map_audits': 0,
            'actual_NPZ_reads': 0, 'actual_checkpoint_reads': 0, 'GPU_calls': 0}


def audit():
    destination = OUT / 'independent_estimated_seen_map_audit.json'; require(not destination.exists(), 'Single exclusive actual audit')
    watched = {}; independent.DIFFERENCES.clear()
    def watch(path, expected=None, byte_count=None):
        path = path.resolve(); require(path.is_relative_to(ROOT), 'Contained evidence'); digest = sha(path)
        require(expected is None or digest == expected, 'Actual file SHA ' + str(path)); require(byte_count is None or path.stat().st_size == byte_count, 'Actual bytes ' + str(path)); watched[path] = digest
    watch(Path(__file__)); watch(PROTOCOL, PROTOCOL_SHA); watch(EVALUATOR, EVALUATOR_SHA); watch(Path(independent.__file__), HELPER_SHA)
    watch(OUT / 'metrics.json', TERMINAL_SHA); watch(OUT / 'progress.json'); require(read(OUT / 'metrics.json') == read(OUT / 'progress.json'), 'Terminal progress exact')
    protocol, metrics = read(PROTOCOL), read(OUT / 'metrics.json')
    require(metrics['status'] in ('completed', 'completed_with_render_failures') and metrics['protocol_sha256'] == PROTOCOL_SHA and metrics['evaluation_script_sha256'] == EVALUATOR_SHA and
            metrics['render_attempts'] == 20 and len(metrics['maps']) == len(metrics['same_map_camera_pairs']) == 4 and
            metrics['successful_render_count'] + metrics['failed_render_count'] == 20 and metrics['optimization_steps'] == metrics['new_GT_camera_renders'] == 0 and
            metrics['observation_map_camera_and_existing_GT_baseline_bytes_unchanged'] and protocol['fixed_views'] == VIEWS, 'Complete actual camera-choice scope and all20 denominator')
    require(metrics['evaluation_role'] == protocol['evaluation_role'] == 'same_frozen_map_camera_choice_seen_training_view_diagnostic', 'In-sample camera choice scope')
    require(datetime.fromisoformat(metrics['GPU_started_at']['utc']) < datetime.fromisoformat(protocol['deadline_beijing']), 'Actual experiment before cutoff')
    for group in ('source_sha256', 'GT_control_evidence_sha256', 'historical_CPU_evidence_sha256'):
        for name, digest in protocol[group].items(): watch(ROOT / name, digest)
    for name, digest in protocol['lineage']['evidence_sha256'].items(): watch(ROOT / name, digest)
    watch(ROOT / protocol['cpu_preflight']['path'], protocol['cpu_preflight']['sha256'])
    review_path = ROOT / protocol['root_review_relative_path']; watch(review_path)
    reviewed = read(review_path); require(reviewed['status'] == 'passed' and reviewed['protocol_sha256'] == PROTOCOL_SHA and reviewed['evaluation_script_sha256'] == EVALUATOR_SHA, 'Actual root evaluator protocol review')
    watch(OUT / 'execution_started.json'); require(read(OUT / 'execution_started.json')['root_review_sha256'] == sha(review_path), 'Actual execution binds same root review')
    watch(GT_OUT / 'metrics.json', GT_METRICS_SHA); watch(GT_OUT / 'independent_full_seen_map_audit.json', GT_AUDIT_SHA)
    gt_metrics, gt_audit = read(GT_OUT / 'metrics.json'), read(GT_OUT / 'independent_full_seen_map_audit.json')
    gt_protocol = read(ROOT / protocol['GT_control']['protocol']); require(sha(ROOT / protocol['GT_control']['protocol']) == GT_PROTOCOL_SHA, 'Original GT protocol exact')
    require(gt_audit['status'] == 'passed_independent_full_seen_map_audit' and gt_audit['audit_source_sha256'] == OLD_AUDITOR_SHA and gt_audit['all_training_rows_including_flags'] == 2594 and
            gt_audit['metrics_sha256'] == GT_METRICS_SHA and gt_metrics['successful_render_count'] == 20 and gt_metrics['failed_render_count'] == 0, 'Explicit previous independent GT20/history audit reuse')
    require(protocol['datasets'] == gt_protocol['datasets'] and protocol['lineage'] == gt_protocol['lineage'], 'Same observation inputs/original failure lineage')
    snapshot = read(ROOT / 'outputs/tum-online-v4-full-sequence-recovery-001/original_failure_snapshot/manifest.json')
    require(snapshot['status'] == 'copied_and_hash_verified' and len(snapshot['files']) == protocol['lineage']['original_failure_snapshot_file_count'] == 1067, 'All original failed artifacts retained')
    for row in snapshot['files']:
        for key in ('original', 'snapshot'): watch(ROOT / row[key], row['sha256'], row['bytes'])
    old_failed = ROOT / 'outputs' / protocol['lineage']['original_failed_run']
    require(not (old_failed / 'metrics.json').exists() and not (old_failed / 'online_map.pt').exists(), 'Original partial failure never reclassified complete')
    lineage_audit = read(ROOT / 'outputs/tum-online-v4-full-sequence-recovery-summary-001/independent_summary_audit.json')
    require(lineage_audit['status'] == 'passed_independent_lineage_full_trajectory_chunk_and_tensor_audit' and lineage_audit['original_returncodes'] == [0, 0, 0, 1] and
            lineage_audit['replacement_returncode'] == 0 and lineage_audit['original_failed_partial_frame_count'] == 521 and lineage_audit['attempt_count'] == 5, 'Full trajectory numerical audit reuse, failed attempt still disclosed')
    source = source_review(protocol); arrays = {}; dataset_checks = []
    for scene, witness in protocol['datasets'].items():
        for name, digest in witness['input_file_sha256'].items(): watch(ROOT / name, digest)
        arrays[scene], _, checked = inputs(scene, witness); dataset_checks.append(checked)
    def prediction(view, folder, run, frame, camera):
        path = ROOT / view['prediction']['path']; require(path.resolve().is_relative_to((folder / 'raw_predictions' / run).resolve()) and path.name == f'view_{frame:04d}.npz', 'Same-map same-frame prediction source')
        watch(path, view['prediction']['sha256'])
        with np.load(path, allow_pickle=False) as packet:
            require(set(packet.files) == {'rgb', 'depth_ED_m', 'alpha'}, 'All3 raw prediction arrays'); value = {k: packet[k].copy() for k in packet.files}
        require(all(v.dtype == np.float32 for v in value.values()), 'Unmodified float32 predictions')
        payloads.append({'run': run, 'frame_id': frame, 'camera_choice': camera, 'NPZ_sha256': sha(path), 'array_payload_records': {k: array_record(v) for k, v in value.items()}})
        return value
    payloads = []; map_checks = []; pair_checks = []; nhistory = nviews = nest = ngt = 0
    for frozen, actual, gt_record, gt_map, pair in zip(protocol['maps'], metrics['maps'], gt_protocol['maps'], gt_metrics['maps'], metrics['same_map_camera_pairs']):
        run, scene = frozen['run'], frozen['scene']; require(run == actual['run'] == gt_record['run'] == gt_map['run'] == pair['run'] and scene == actual['scene'] == pair['scene'], 'One SAME map, no weight/map substitution')
        require({k: frozen[k] for k in gt_record} == gt_record, 'Same full original map record plus camera fields')
        for name, digest in frozen['artifact_sha256'].items(): watch(ROOT / name, digest)
        watch(ROOT / frozen['history']['path'], frozen['history']['sha256'], frozen['history']['bytes'])
        checkpoint = torch.load(ROOT / frozen['checkpoint'], map_location='cpu', weights_only=True)
        require({k: array_record(v) for k, v in checkpoint['parameters'].items()} == frozen['parameter_tensors'] == actual['parameters_before'] == actual['parameters_after'] == gt_map['parameters_before'] == gt_map['parameters_after'], 'Exact5 fixed parameter payloads both cameras')
        require(actual['parameter_bytes_unchanged'] and actual['selected_camera_bytes_unchanged'] and actual['saved_camera_tensors_before'] == frozen['camera_tensors'] and
                array_record(checkpoint['K']) == frozen['K_tensor'] == array_record(arrays[scene]['K_render_float32']), 'Actual camera/map/K beforeafter guards')
        training = read((ROOT / frozen['checkpoint']).parent / 'metrics.json'); history = read(ROOT / frozen['history']['path']); count = frozen['configuration']['frames']
        require(checkpoint['configuration'] == training['configuration'] == frozen['configuration'] and len(training['frame_metrics']) == count and
                checkpoint['completed_frame_metrics'] == training['frame_metrics'] and history['training_frame_count'] == count, 'Full2594 original training rows preserved')
        prior_checked = next(x for x in gt_audit['maps'] if x['run'] == run)
        require(prior_checked['training_frame_count'] == count and prior_checked['all_history_flags_optional_absence_and_source_counts_exact'] and
                prior_checked['failure_frame_count'] == training['failure_frame_count'] == frozen['training_failure_frame_count'] == history['training_failure_frame_count'], 'Explicit old audited full flags/history numerical scope')
        ids = checkpoint['source_frame_ids_per_gaussian']; require(array_record(np.asarray(ids, dtype=np.int64)) == frozen['source_frame_ids_tensor'] and len(ids) == frozen['point_count'], 'Actual Gaussian source payload unchanged')
        require({str(k): v for k, v in sorted(Counter(ids).items())} == frozen['source_frame_histogram'] == history['source_frame_histogram'], 'Actual source histogram retained')
        nhistory += count; c2w, w2c, camera_evidence = camera_check(frozen, checkpoint, arrays[scene])
        require(actual['existing_GT_camera_aggregate'] == gt_map['aggregate'] and actual['point_count'] == frozen['point_count'] and actual['history'] == frozen['history'], 'GT aggregate/history whole reference exact')
        require([v['frame_id'] for v in actual['views']] == [v['frame_id'] for v in gt_map['views']] == [v['frame_id'] for v in pair['views']] == VIEWS[scene] and pair['same_map_parameter_bytes'], 'All fixed5 own/common rows retained')
        est_rows = []; gt_rows = []; common_rows = []
        for index, (view, gt_view, pair_view) in enumerate(zip(actual['views'], gt_map['views'], pair['views'])):
            frame = view['frame_id']; nviews += 1
            require(view['rgb_timestamp'] == protocol['datasets'][scene]['evaluation_sensor_metadata'][index]['rgb_timestamp'] and view['role'] == 'seen_training_view_diagnostic' and
                    view['camera_choice'] == 'saved accepted estimated w2c', 'Actual observation timestamp and camera role')
            require(array_record(np.asarray(view['saved_accepted_c2w'], np.float32)) == array_record(c2w[index]) and
                    array_record(np.asarray(view['render_saved_accepted_w2c_float32'], np.float32)) == array_record(w2c[index]), 'Per-view actual saved camera bits')
            require(view['same_map_GT_prediction'] == gt_view['prediction'] and view['existing_GT_camera_metrics'] == gt_view['metrics'], 'Per-view old GT NPZ/metric exact source')
            gt_value = prediction(gt_view, GT_OUT, run, frame, 'GT_camera'); ngt += 1
            gm, gc = independent.measure(gt_value['rgb'], gt_value['depth_ED_m'], gt_value['alpha'], arrays[scene]['rgb'][index], arrays[scene]['depth'][index], arrays[scene]['valid_depth_mask'][index])
            compare(gt_view['metrics'], gm, run + '/' + str(frame) + '/GT_ownC'); gt_rows.append({'status': 'completed', 'metrics': gm})
            if view['status'] != 'completed':
                require(view['status'] == 'failed' and 'exception_type' in view and pair_view['status'] == 'unavailable_due_to_retained_estimated_camera_render_failure', 'Failed new view and unavailable pair retained')
                est_rows.append({'status': 'failed'}); common_rows.append(pair_view); continue
            est_value = prediction(view, OUT, run, frame, 'saved_estimated_camera'); nest += 1
            em, ec = independent.measure(est_value['rgb'], est_value['depth_ED_m'], est_value['alpha'], arrays[scene]['rgb'][index], arrays[scene]['depth'][index], arrays[scene]['valid_depth_mask'][index])
            compare(view['metrics'], em, run + '/' + str(frame) + '/EST_ownC'); est_rows.append({'status': 'completed', 'metrics': em})
            measured = camera_common((gt_value['depth_ED_m'], gc), (est_value['depth_ED_m'], ec), arrays[scene]['depth'][index], arrays[scene]['valid_depth_mask'][index])
            require(pair_view['status'] == 'completed', 'Successful same-map common pair'); compare(pair_view['metrics'], measured, run + '/' + str(frame) + '/GT_EST_commonC')
            common_rows.append({'status': 'completed', 'metrics': measured})
        gt_pool, est_pool, common_pool = aggregate(gt_rows), aggregate(est_rows), pooled_common(common_rows)
        compare(gt_map['aggregate'], gt_pool, run + '/GT_pooled'); compare(actual['aggregate'], est_pool, run + '/EST_pooled'); compare(pair['pooled_same_map_common_C'], common_pool, run + '/same_map_commonC_pooled')
        map_checks.append({**camera_evidence, 'prior_weight': frozen['prior_weight'], 'point_count': frozen['point_count'], 'all5_parameter_payloads_same_as_GT': True,
                           'all_training_frames_including_flags': count, 'reused_prior_history_check': prior_checked,
                           'GT_ownC_RGB_holes_pooled': gt_pool, 'saved_EST_ownC_RGB_holes_pooled': est_pool})
        pair_checks.append({'run': run, 'scene': scene, 'requested_same_map_views': 5, 'pooled_same_map_commonC_recomputed': common_pool})
    require(nviews == 20 and ngt == 20 and nest == metrics['successful_render_count'] and nhistory == 2594 and len(payloads) == ngt + nest, 'All40 requested predictions/120 channels and2594 rows scope')
    not_imported(); require(all(sha(path) == digest for path, digest in watched.items()), 'Every consumed source/raw/history/checkpoint/NPZ unchanged')
    differences = independent.DIFFERENCES
    result = {'status': 'passed_independent_estimated_seen_map_audit', 'audit_source_sha256': sha(Path(__file__)), 'independent_math_helper_sha256': HELPER_SHA,
              'evaluation_script_sha256': EVALUATOR_SHA, 'protocol_sha256': PROTOCOL_SHA, 'metrics_sha256': TERMINAL_SHA,
              'existing_GT_metrics_sha256': GT_METRICS_SHA, 'existing_GT_independent_audit_sha256': GT_AUDIT_SHA,
              'map_count': 4, 'requested_view_count': 20, 'actual_estimated_NPZ_count': nest, 'existing_GT_NPZ_count': ngt,
              'actual_prediction_payload_array_count': 3 * len(payloads), 'requested_same_map_commonC_views': nviews,
              'all_training_rows_including_flags': nhistory, 'source_review': source, 'datasets': dataset_checks, 'maps': map_checks, 'same_map_pairs': pair_checks,
              'raw_prediction_payload_hash_records': payloads, 'original_failure_snapshot_files_verified_original_and_copy': 1067,
              'consumed_files_rehashed_unchanged': len(watched), 'numeric_scalar_comparison_count': len(differences),
              'maximum_absolute_numeric_scalar_difference': max(r['absolute_difference'] for r in differences),
              'largest10_numeric_scalar_differences': sorted(differences, key=lambda r: r['absolute_difference'], reverse=True)[:10], 'GPU_calls': 0,
              'limits': ['Explicit aabd independent NumPy/math metric reuse and0204 raw-input routine adaptation, not a third independent formula.',
                         'Original full trajectory math and2594 training histories reuse locked prior independent audits; this audit rehashes files, current source payloads and all cameras without re-solving.',
                         'Same-map saved camera versus GT camera at seen training views is in-sample consistency, not localization accuracy or unbiased world geometry.',
                         'Only SAME map/camera pair shares exact map bytes; different prior-weight maps/warmups are not claimed byte-identical.',
                         'GT rendering occurred previously; no GT rerender, GPU, optimization, retry or numerical nondeterminism guarantee.',
                         'OwnC/commonC/holes use distinct persisted denominators; failures/empty intersections/perfect RGB infinity remain explicit.',
                         'Exact camera payload uses original saved float32 w2c, not re-inverting c2w; scalar maximum spans mixed units and is not an accuracy score.']}
    save(destination, result); print(json.dumps({'status': result['status'], 'artifact_sha256': sha(destination), 'new_NPZ': nest, 'old_GT_NPZ': ngt, 'arrays': 3 * len(payloads)}))


def main():
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true'); mode.add_argument('--audit', action='store_true')
    p.add_argument('--run', default='v4-estimated-seen-map-independent-auditor-cpu-001'); args = p.parse_args()
    if args.self_check:
        require(args.run and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run), 'Unique CPU output')
        folder = ROOT / 'outputs' / args.run; folder.mkdir(exist_ok=False); (folder / 'source_script.py').write_bytes(Path(__file__).read_bytes())
        try:
            require(sha(PROTOCOL) == PROTOCOL_SHA and sha(EVALUATOR) == EVALUATOR_SHA and sha(Path(independent.__file__)) == HELPER_SHA, 'SOURCE/protocol/math-helper frozen bytes')
            value = controls(folder); value.update(audit_source_sha256=sha(Path(__file__)), independent_math_helper_sha256=HELPER_SHA,
                protocol_sha256=PROTOCOL_SHA, evaluation_script_sha256=EVALUATOR_SHA, actual_metrics_sha256_binding=TERMINAL_SHA,
                source_review=source_review(read(PROTOCOL)))
            save(folder / 'metrics.json', value)
            save(folder / 'completion.json', {'status': 'completed_synthetic_only', 'files': {str(q.relative_to(folder)).replace('\\', '/'): {'sha256': sha(q), 'bytes': q.stat().st_size} for q in sorted(folder.rglob('*')) if q.is_file()}, 'actual_audits': 0, 'GPU_calls': 0})
            print(json.dumps({'status': value['status'], 'audit_source_sha256': value['audit_source_sha256'], 'actual_map_audits': 0}))
        except BaseException as error:
            save(folder / 'failure.json', {'status': 'CPU_auditor_preparation_failure_preserved', 'exception_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'GPU_calls': 0}); raise
    else:
        try: audit()
        except BaseException as error:
            path = OUT / 'independent_estimated_seen_map_audit_failure-001.json'
            save(path, {'status': 'actual_independent_audit_failure_preserved', 'audit_source_sha256': sha(Path(__file__)), 'exception_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'GPU_calls': 0}); raise


if __name__ == '__main__': main()

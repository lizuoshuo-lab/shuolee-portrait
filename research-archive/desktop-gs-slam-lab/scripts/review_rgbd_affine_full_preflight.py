"""Independent read-only source/protocol/CPU-evidence review, never registration.

Reuses the frozen association and PIL sensor loader, explicitly, for byte parity.
No main/prepare/execute/driver is called. Registration and normal-system entry
points are replaced in this process by aborting sentinels. No real child/GPU.
"""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import track_rgbd_affine_full as pipeline
import run_rgbd_affine_full as driver
from tum_rgbd import associate_frames, load_frames, scaled_intrinsics, safe_data_path

PREP = ROOT / 'outputs/rgbd-affine-full-cpu-preparation-001'
MOCK = ROOT / 'outputs/rgbd-affine-full-driver-cpu-controls-001'
PROTOCOL_SHA = 'ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a'
NEW_SHA = {
    'track_rgbd_affine_full.py': '879eafcceaa2ed03150755bae3bfc9f0063fded86280e27ad2f273087885086c',
    'run_rgbd_affine_full.py': 'be32a3b675ba3597369d3243123399b8c4e298b4f1e54baaf4a8bcea7b3da866',
    'check_rgbd_affine_full_preflight.py': '87f7cd48d952f05990f21742bdf4932c798330803a49ee271ad18991dffe1428'}


def need(ok, label):
    if not ok: raise AssertionError(label)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def array_record(a):
    a = np.ascontiguousarray(a)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': a.nbytes, 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def write(path, obj):
    with path.open('xb') as stream: stream.write((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())


def main():
    destination = PREP / 'root-independent-full-preflight-review.json'
    need(not destination.exists(), 'Exclusive independent review')
    watched = {}; status = {'status': 'running_readonly_CPU_preflight_review', 'real_pair_registration_calls': 0, 'real_children_started': 0, 'GPU_calls': 0}
    def watch(path, expected=None, nbytes=None):
        path = path.resolve(); need(path.is_relative_to(ROOT), 'Contained evidence path'); digest = sha(path)
        need(expected is None or digest == expected, 'Actual SHA ' + str(path)); need(nbytes is None or path.stat().st_size == nbytes, 'Actual bytes ' + str(path))
        watched[path] = digest; return digest
    def forbidden(*args, **kwargs): raise AssertionError('Independent preflight prohibits registration/normal equations/real Popen')
    try:
        watch(Path(__file__)); watch(PREP / 'protocol.json', PROTOCOL_SHA)
        protocol = read(PREP / 'protocol.json'); sources = protocol['source_sha256']
        need(len(sources) == 18 and all(sources[k] == v for k, v in NEW_SHA.items()), '18 exact source stack incl reviewed new3')
        for name, digest in sources.items():
            watch(ROOT / 'scripts' / name, digest); watch(PREP / 'source_snapshot' / name, digest)
        watch(PREP / 'preparation_source.py', NEW_SHA['track_rgbd_affine_full.py'])
        manifest = read(PREP / 'source_manifest.json'); watch(PREP / 'source_manifest.json')
        need(manifest == {'source_sha256': sources, 'protocol_sha256': PROTOCOL_SHA}, 'Source snapshot manifest exact')
        need(pipeline.source_checks() == sources, 'Runtime source-chain guards exact')
        watch(pipeline.SHORT_PROTOCOL, pipeline.SHORT_PROTOCOL_SHA); short = read(pipeline.SHORT_PROTOCOL)
        need({k: sources[k] for k in short['source_sha256']} == short['source_sha256'], 'Previous15 frozen stack unchanged')
        need(protocol['CPU_registration_runs'] == protocol['GPU_runs'] == 0, 'Preparation not actual execution')
        frozen6, frozen8 = pipeline.affine.frozen, pipeline.affine
        need(frozen6.register_pair is pipeline.recovery.odom.register_pair, 'Exact6D frozen solver object')
        # Abort even accidental optimization/subprocess attempts in this reviewer.
        for module, key in ((frozen8, 'register_pair_affine'), (frozen8, 'normalized_system'),
                            (frozen6, 'register_pair'), (frozen6, 'weighted_system'), (frozen6.icp, 'register_pair')):
            setattr(module, key, forbidden)
        driver.subprocess.Popen = forbidden
        round_checks = []
        cli = read(PREP / 'cli_checks.json'); watch(PREP / 'cli_checks.json')
        need(cli['status'] == 'passed_no_execution' and cli['protocol_sha256'] == PROTOCOL_SHA and cli['registration_and_normal_equation_calls_forbidden'], 'Actual CLI preparation gate')
        need(protocol['round_order'] == ['desk-lastaccepted-6d-full', 'desk-lastaccepted-8d-full', 'xyz-lastaccepted-6d-full', 'xyz-lastaccepted-8d-full'], 'Fixed four-order')
        for row, parsed_evidence in zip(protocol['rounds'], cli['rounds']):
            scene, dim, cfg = row['scene'], row['model_dimension'], row['configuration']; frames = 549 if scene == 'desk' else 748
            need(cfg == pipeline.configuration_for(scene, dim), 'Actual complete runtime config')
            original_cfg = pipeline.frozen.configuration_for(scene) if dim == 8 else pipeline.recovery.configuration_for(scene)
            differences = [k for k in cfg if cfg[k] != original_cfg[k]]
            need(set(differences) == {'frames', 'start_frame', 'frame_ids'}, 'Only full sequence length/start/IDs differ from frozen short config')
            need(cfg['frames'] == frames and cfg['start_frame'] == 0 and cfg['frame_ids'] == list(range(frames)) and
                 cfg['odometry'] == pipeline.original.config_record(1.) and cfg['reference_policy'] == pipeline.recovery.POLICY and cfg['first_c2w'] == 'identity', 'Full IDs/identity/frozen6D params/reference')
            if dim == 8:
                need(cfg['joint_affine'] == pipeline.frozen.config_record() and cfg['brightness_reset_for_every_pair'] is True and cfg['brightness_accumulated_across_pairs'] is False and
                     cfg['pair_brightness_initialization'] == {'log_relative_gain': 0., 'relative_gain': 1., 'bias_intensity': 0.}, 'Frozen8D params and pair-local reset')
            args = pipeline.arguments(parsed_evidence['argv'])
            need(args.execute and not args.prepare and args.round_id == row['round_id'] and args.run == row['run'] and args.protocol_sha == PROTOCOL_SHA, 'Actual four child CLI parse')
            need(not (ROOT / 'outputs' / row['run']).exists(), 'No actual output created/overwritten')
            round_checks.append({'round_id': row['round_id'], 'model_dimension': dim, 'frames': frames, 'start_frame': 0, 'only_config_changes': differences, 'CLI_exact': True, 'actual_output_absent': True})
        need(len(round_checks) == len(cli['rounds']) == 4, 'All4 configs/CLI')
        deadlines = []
        for offset, allowed in [(-1, True), (0, False), (1, False)]:
            stamp = pipeline.DEADLINE + dt.timedelta(microseconds=offset)
            try: pipeline.before_deadline(stamp); actual = True
            except ValueError: actual = False
            need(actual == allowed, 'Independent strict cutoff +/-1us'); deadlines.append({'timestamp': stamp.isoformat(), 'allowed': actual})
        watch(PREP / 'synthetic_policy_checks.json', protocol['synthetic_policy_sha256'])
        toy = read(PREP / 'synthetic_policy_checks.json'); need(toy['real_registration_or_linear_solves'] == 0 and toy['different_pair_nuisance_same_pose_prefix_exact'], 'Synthetic policy only')
        for ledger in toy['reviewed_noncommuting_and_rejection_ledgers']['complete_synthetic_ledgers'].values():
            pose = np.eye(4); ref = 200; poses = {200: pose}
            for item in ledger:
                current = item['current_frame_id']; need(item['reference_frame_id'] == ref and item['reference_gap_frames'] == current - ref, 'Saved synthetic causal IDs')
                if item['accepted']: pose = poses[ref] @ np.asarray(item['candidate_reference_from_current']); ref = current
                need(np.allclose(pose, item['estimated_c2w'], atol=2e-15, rtol=2e-15) and item['last_accepted_frame_id_after'] == ref, 'Independent synthetic accepted composition/reject prefix')
                poses[current] = pose.copy()
        noncommute = toy['reviewed_noncommuting_and_rejection_ledgers']['noncommuting_skip']
        need(np.allclose(np.asarray(noncommute['T12']) @ np.asarray(noncommute['T23']), noncommute['T13_T12_times_T23'], atol=1e-15) and
             np.max(np.abs(np.asarray(noncommute['T23']) @ np.asarray(noncommute['T12']) - np.asarray(noncommute['T13_T12_times_T23']))) > .001, 'Actual noncommuting toy direction')
        mock = read(MOCK / 'metrics.json'); watch(MOCK / 'metrics.json'); completion = read(MOCK / 'completion.json'); watch(MOCK / 'completion.json')
        need(mock['status'] == 'passed_all_mock_full_driver_controls' and mock['source_sha256'] == sources and mock['checker_sha256'] == NEW_SHA['check_rgbd_affine_full_preflight.py'] and
             mock['protocol_sha256'] == PROTOCOL_SHA and mock['real_children_started'] == mock['real_pair_registration_runs'] == mock['GPU_calls'] == 0, 'Actual mock-only11 controls source/status')
        for name, record in completion['files'].items(): watch(MOCK / name, record['sha256'], record['bytes'])
        need(len(mock['fixtures']) == 11, 'All11 mock controls retained')
        for name, fixture in mock['fixtures'].items():
            events = fixture['events']; need(fixture['status'] == 'passed_mock_control', 'Mock control status')
            started = 0; waiting = False
            for i, event in enumerate(events):
                if event.startswith('Popen:'):
                    need(not waiting and i > 0 and events[i-1].startswith('guard:') and dt.datetime.fromisoformat(events[i-1][6:]) < pipeline.DEADLINE, 'Recorded guard immediately before serialized Popen')
                    waiting = True; started += 1
                elif event.startswith('wait:'): waiting = False
            need(started == fixture['mock_children_started'] and not waiting, 'Every mock started child collected')
            terminal = MOCK / name / 'outputs/mock-batch' / ('metrics.json' if fixture['exception_type'] is None else 'failure.json')
            if fixture['terminal_sha256']:
                watch(terminal, fixture['terminal_sha256']); packet = read(terminal)
                need(packet['runs_started'] == started, 'Mock terminal count exact')
            if name == 'first_child_exit1': need(started == 1 and packet['run_records'][0]['returncode'] == 1, 'Immediate stop on first exit1')
            if name == 'second_child_exit1': need(started == 2 and packet['run_records'][-1]['returncode'] == 1, 'Stop on second exit1')
            if name == 'interrupt_started_child': need(started == 1 and packet['run_records'][0]['collected_after_interrupt'], 'Interrupted child collected')
        # Actual full sensor witnesses: reused frozen loader, independent grayscale/hash implementation.
        sensor_findings = []; raw_count = 0; full_arrays = 0; short_arrays = 0
        for scene, expected_count in [('desk', 549), ('xyz', 748)]:
            print(json.dumps({'reviewing_scene_inputs': scene, 'real_registration_calls': 0}), flush=True)
            witness_spec = protocol['datasets'][scene]; path = ROOT / witness_spec['path']; watch(path, witness_spec['sha256']); witness = read(path)
            dataset = ROOT / f'data/tum/rgbd_dataset_freiburg1_{scene}'
            for name, digest in witness['index_sha256'].items(): watch(dataset / name, digest)
            frames, association = associate_frames(dataset, .02, .02, 32.)
            need(len(frames) == expected_count and [f.frame_id for f in frames] == list(range(expected_count)) and witness['association'] == association, 'Actual full associated IDs')
            metadata = [pipeline.original.metadata(f) for f in frames]
            need(metadata == witness['selected_sensor_metadata'] and metadata[200:300] == short['datasets'][scene]['selected_sensor_metadata'], 'Actual all1297 metadata + original100 slice')
            baseline = ROOT / 'outputs' / witness['full_ICP_reference']['baseline']
            for name in ['metrics', 'validation', 'metadata']:
                filename = 'selected_sensor_metadata.json' if name == 'metadata' else name + '.json'
                watch(baseline / filename, witness['full_ICP_reference'][name + '_sha256'])
            baseline_meta = read(baseline / 'selected_sensor_metadata.json')['selected_frames']
            need(metadata == [{k: row[k] for k in metadata[0]} for row in baseline_meta], 'Actual entire full ICP metadata parity')
            need(len(witness['raw_sensor_files']) == expected_count, 'All raw witness rows retained')
            for frame, row in zip(frames, witness['raw_sensor_files']):
                need(row['frame_id'] == frame.frame_id, 'Actual raw witness frame')
                for key in ('rgb_path', 'depth_path'):
                    need(row[key]['path'] == getattr(frame, key), 'Actual raw filename'); watch(safe_data_path(dataset, row[key]['path']), row[key]['sha256']); raw_count += 1
            level_counts = []
            for level, (width, height) in enumerate(((40, 30), (80, 60), (160, 120))):
                rgb, depth, valid = load_frames(dataset, frames, width, height, .2, 5.)
                gray = np.asarray(rgb, dtype=np.float64) @ np.array([.299, .587, .114], dtype=np.float64)
                arrays = {'rgb': rgb, 'depth': depth, 'valid': valid, 'gray': gray, 'K': scaled_intrinsics(width, height)}
                for key, value in arrays.items():
                    need(array_record(value) == witness['input_array_hashes'][str(level)][key], 'Actual full array ' + scene + str(level) + key); full_arrays += 1
                    slice_record = array_record(value if key == 'K' else value[200:300])
                    need(slice_record == witness['short200_299_slice_array_hashes'][str(level)][key] == short['datasets'][scene]['input_array_hashes'][str(level)][key], 'Full short-slice byte parity ' + key); short_arrays += 1
                # Correspondence construction at identity is not a registration/normal solve.
                config = pipeline.original.config_for(1.)
                src = frozen6.make_frame(depth[1], valid[1], arrays['K'], rgb[1], config); target = frozen6.make_frame(depth[0], valid[0], arrays['K'], rgb[0], config)
                pair = frozen6.pair_correspondences(np.eye(4), src, target, config.geometry.distance_gates_m[level], config)
                measured = {'counts': pair['counts'], 'ratio': pair['ratio'], 'arrays': {k: array_record(pair[k]) for k in ('source', 'transformed', 'target', 'normals', 'source_intensity')}}
                need(measured == witness['first_identity_support'][str(level)], 'Actual first-pair support exact without solve')
                level_counts.append({'level': level, 'joint_correspondences': pair['counts']['joint_correspondences']})
                del rgb, depth, valid, gray, arrays, src, target, pair
            sensor_findings.append({'scene': scene, 'frames': expected_count, 'raw_PNG_hashes': 2 * expected_count, 'metadata_exact_with_full_ICP': True,
                                    'full_array_payload_hashes': 15, 'short_slice_payload_hashes': 15, 'first_identity_support': level_counts})
        need(not any(k in sys.modules for k in ('torch', 'gsplat')), 'No GPU stack imported')
        need(all(sha(path) == digest for path, digest in watched.items()), 'All consumed inputs/source/mock evidence unchanged')
        source_review = {
            'solver_calls': '6D directly calls frozen8016 register_pair;8D calls frozen15c register_pair_affine with unchanged frozen313d/bf219 integration.',
            'logging_only': 'Full wrapper writes immutable pair JSON, identical existing iteration rows, compact progress and partial pose text after integration; no logging output or clock enters solver acceptance.',
            'causal_reference': 'targets and reference_index update together only on accepted result; accepted prefix=poses[reference]@reference_from_current; rejection propagates previous prefix and keeps sensor reference.',
            'brightness': 'Frozen8D starts identity/a=0/b=0 every register_pair call; full wrapper never accumulates brightness.',
            'numeric_exceptions': 'Original numeric exception family retained as rejected/unknown candidate+traceback; unexpected outer failures preserve all completed rows and abort child/batch.',
            'GT_barrier': 'Association honestly reads GT fields/matrices; first support uses identity sensor arrays; execute accesses frame.c2w only after full estimated trajectory export.',
            'deadline': 'Open-log then strict cutoff immediately before Popen; full child rechecks just before first pair; no mid-sequence cutoff; already-started child waited/collected.',
            'failed_exit': 'code0 required before terminal metrics; first exit1 stops batch with no retry and no later round; completed-with-rejections is valid terminal, all flags retained.',
            'prepare_guards': 'Preparation binds register_pair_affine/normalized_system/register_pair/weighted_system/oldICP register_pair to forbidden sentinel; identity correspondence construction and synthetic ledger algebra remain allowed.'}
        result = {**status, 'status': 'passed_independent_full_preflight_review_no_actual_execution', 'review_source_sha256': sha(Path(__file__)),
                  'protocol_sha256': PROTOCOL_SHA, 'source_sha256': sources, 'source_and_snapshot_files_verified': 36,
                  'round_checks': round_checks, 'strict_deadline_controls': deadlines, 'mock_controls_verified': 11,
                  'mock_artifact_hashes_verified': len(completion['files']), 'actual_associated_sensor_frames': 1297,
                  'actual_raw_PNG_hashes_verified': raw_count, 'actual_full_array_payloads_verified': full_arrays,
                  'actual_short_slice_array_payloads_verified': short_arrays, 'sensor_findings': sensor_findings,
                  'consumed_files_rehashed_unchanged': len(watched), 'source_review_findings': source_review,
                  'limits': ['SOURCE/CPU preparation evidence only; no full-sequence localization outcome is established.',
                             'Frozen association/PIL loader reused explicitly; array hashes/grayscale implementation and saved mock-ledger checks independently recomputed.',
                             'Mocks exercise authored driver flow, not OS-scheduler timing or numerical solver accuracy.',
                             'All lastaccepted gaps may grow after rejection; wider gap is not guaranteed to register.',
                             '6D versus8D is a whole-model/path comparison; observability/rank/stopping differ, not pure brightness causality.',
                             'Expected future ordinary pair rejections are retained; infrastructure failure terminates partial history, not a successful full run.']}
        write(destination, result); print(json.dumps({'status': result['status'], 'review': str(destination), 'sha256': sha(destination)}))
    except BaseException as error:
        status.update(status='independent_preflight_failure_preserved', exception_type=type(error).__name__, exception_message=str(error), traceback=traceback.format_exc())
        write(PREP / 'root-independent-full-preflight-review-failure-001.json', status); raise


if __name__ == '__main__': main()

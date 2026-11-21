"""Summarize saved local replay evidence; never start ROS or infer GT accuracy."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

LAB = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def reference(path):
    return {'path': path.relative_to(LAB).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


spec = importlib.util.spec_from_file_location('replay_runner', LAB/'scripts/fast_livo2_linux/replay_bright.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
observed = datetime.now(timezone.utc).isoformat()
runs = []
preflight_failures = []
for group in ('fast-livo2-local-replay-001', 'fast-livo2-local-replay-002'):
    for status_path in sorted((LAB/'outputs'/group/'runs').glob('*/status.json')):
        status = read(status_path)
        if not status.get('finished_utc'):
            continue
        folder = status_path.parent
        if not status.get('fusion_started'):
            preflight_failures.append({'run_name': status['run_name'], 'group': group,
                'mode': status['mode'], 'error': status.get('error'),
                'fusion_started': False, 'evidence': reference(status_path)})
            continue
        child = runner.child_exit_summary(folder, status.get('fusion_node_pid'))
        checks = dict(status.get('runtime_evidence_checks', {}))
        checks['required_roslaunch_children_exit_zero'] = child['all_required_children_exit_zero']
        params = read(folder/'actual-parameter-checks.json')
        checks['actual_parameter_checks_passed'] = all(row['passed'] for row in params.values())
        accepted = bool(checks) and all(checks.values()) and not status.get('error')
        topics = read(folder/'topic-observation.json')
        markers = status.get('visual_log_evidence', {}).get('markers', {})
        reduced_markers = {}
        for key, value in markers.items():
            reduced_markers[key] = {k: v for k, v in value.items() if k != 'values_in_log_order'} if isinstance(value, dict) else value
        runs.append({'run_name': status['run_name'], 'group': group, 'mode': status['mode'],
            'started_utc': status['started_utc'], 'finished_utc': status['finished_utc'],
            'original_runner_status': status['status'],
            'reviewed_status': 'complete_local_functional_replay_accepted' if accepted else 'actual_outputs_observed_shutdown_failed_or_incomplete',
            'accepted': accepted, 'rate': status['rate'], 'binary': status['binary'], 'bag': status['bag'],
            'checks': checks, 'child_exits': child,
            'pre_vio_trajectory': status.get('pre_vio_trajectory'),
            'sampled_post_vio_trajectory': status.get('sampled_post_vio_trajectory'),
            'observed_topics': {topic: {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                for topic, row in topics['topics'].items()},
            'visual_markers': reduced_markers,
            'evidence': [reference(folder/name) for name in
                ['status.json', 'source-before.json', 'source-after.json', 'bag-index.json',
                 'actual-parameters.yaml', 'actual-parameter-checks.json', 'topic-observation.json', 'runner-source.py']]})

environment = LAB/'outputs/fast-livo2-linux-bootstrap-001/actual-build-readback-001.json'
environment_record = read(environment)
accepted_runs = [run for run in runs if run['accepted']]
patch_path = LAB/'outputs/fast-livo2-shutdown-fix-001/source-override-receipt.json'
patch = reference(patch_path) if patch_path.is_file() else None
audits = []
for group in ('001', '002'):
    folder = LAB/('outputs/fast-livo2-local-replay-independent-audit-'+group)
    for name in ('metrics.json', 'report.md'):
        path = folder/name
        if path.is_file():
            audits.append(reference(path))
summary = {'observed_utc': observed, 'provenance': 'actual_local_saved_execution_logs',
    'compiled_environment_accepted': environment_record['status'] == 'actual_linux_build_and_environment_acceptance_independent_readback_passed',
    'environment_evidence': reference(environment), 'source_local_patch': patch,
    'actual_local_replay_runs': len(runs), 'accepted_local_replay_runs': len(accepted_runs),
    'preflight_failures': preflight_failures,
    'accepted_modes': sorted({run['mode'] for run in accepted_runs}), 'runs': runs, 'independent_audits': audits,
    'ground_truth_available': False, 'local_accuracy_measured': False,
    'scope': 'Bright Screen half-speed functional replay only. Observer transport counts do not certify estimator processing or drop freedom. LIO trajectory is sensor-time IMU-body output after LiDAR and before same-cycle VIO. Sampled image poses are post-VIO. Numerical differences are not accuracy improvements. Original 001 false-positive receipts remain preserved; actual child -11 overrides their incomplete acceptance.'}
fix_summary = LAB/'outputs/fast-livo2-shutdown-fix-001/summary.json'
if fix_summary.is_file():
    summary['shutdown_fix_evidence'] = reference(fix_summary)
accepted_binary_hashes = {run['binary']['sha256'] for run in accepted_runs}
matching_reverifications = []
for path in sorted((LAB/'outputs/fast-livo2-linux-bootstrap-001/linux-executions').glob('*/environment-acceptance.json')):
    binary_hash_path = path.parent/'fastlivo-binary.sha256'
    if binary_hash_path.is_file() and read(path).get('status') == 'compiled_environment_acceptance_passed':
        binary_hash = binary_hash_path.read_text().split()[0]
        if binary_hash in accepted_binary_hashes:
            matching_reverifications.append({'acceptance': reference(path),
                'binary_hash': reference(binary_hash_path), 'accepted_binary_sha256': binary_hash})
if matching_reverifications:
    summary['accepted_binary_environment_reverification'] = matching_reverifications[-1]
for group in ('001', '002'):
    write(LAB/'outputs'/('fast-livo2-local-replay-'+group)/'summary.json', summary)
runtime_path = LAB/'manifests/fast-livo2-local-runtime.json'
runtime = {key: summary[key] for key in
    ['observed_utc', 'provenance', 'compiled_environment_accepted', 'environment_evidence',
     'source_local_patch', 'actual_local_replay_runs', 'accepted_local_replay_runs',
     'accepted_modes', 'preflight_failures', 'ground_truth_available', 'local_accuracy_measured', 'independent_audits', 'scope']}
runtime['replay_summary'] = reference(LAB/'outputs/fast-livo2-local-replay-002/summary.json')
if matching_reverifications:
    runtime['accepted_binary_environment_reverification'] = matching_reverifications[-1]
runtime['run_statuses'] = [{'run_name': run['run_name'], 'mode': run['mode'],
    'reviewed_status': run['reviewed_status'], 'fusion_child_exit_codes': run['child_exits']['children']['laserMapping-1']['exit_codes']} for run in runs]
write(runtime_path, runtime)
post_path = LAB/'outputs/fast-livo2-wsl-setup/post-reboot-status.json'
post = read(post_path)
post.update(observed_utc=observed, compiled_environment_accepted=summary['compiled_environment_accepted'],
    fusion_started=bool(runs), actual_local_replay_runs=len(runs), accepted_local_replay_runs=len(accepted_runs),
    status='post_reboot_local_replay_accepted_no_GT_accuracy' if accepted_runs else 'post_reboot_actual_outputs_observed_shutdown_not_accepted')
write(post_path, post)
subprocess.run([sys.executable, str(LAB/'scripts/update_fast_livo2_work_record.py')], check=True)
progress_path = LAB/'manifests/progress.json'
progress = read(progress_path)
result = {'id': 'fast-livo2-local-bright-replay-002', 'title': '本机 Bright Screen LIO/LIVO 融合回放与退出验收',
    'status': 'accepted_functional_replay_no_ground_truth_accuracy' if accepted_runs else 'actual_outputs_observed_shutdown_failure',
    'metrics': {'actual_replay_runs': len(runs), 'accepted_replay_runs': len(accepted_runs),
        'input_bag_messages': 14850, 'input_bag_duration_s': 66.6815, 'playback_rate': 0.5,
        'local_GT_accuracy_evaluated': False},
    'artifacts': ['manifests/fast-livo2-local-runtime.json', 'outputs/fast-livo2-local-replay-002/summary.json',
        'outputs/fast-livo2-replay-runner-controls-002/controls.json'] + [row['path'] for row in audits],
    'scope': summary['scope']}
for path in ('outputs/fast-livo2-shutdown-fix-001/summary.json',
             'outputs/fast-livo2-shutdown-fix-001/report.md',
             'outputs/fast-livo2-local-replay-002/trajectory-and-evidence.png',
             'outputs/fast-livo2-local-replay-002/chart-metrics.json'):
    if (LAB/path).is_file():
        result['artifacts'].append(path)
progress['results'] = [result]+[row for row in progress['results'] if row['id'] != result['id']]
write(progress_path, progress)
print(json.dumps({key: summary[key] for key in
    ['compiled_environment_accepted', 'actual_local_replay_runs', 'accepted_local_replay_runs', 'accepted_modes']}, indent=2))

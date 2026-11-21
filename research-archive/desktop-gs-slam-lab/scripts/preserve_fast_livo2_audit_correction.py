"""Preserve the original observed script and explicitly record a unit-label fix."""
import hashlib
import json
from pathlib import Path
import shutil

lab = Path(__file__).resolve().parent.parent
script = lab / 'scripts/audit_fast_livo2_bag.py'
old = script.read_text(encoding='utf-8')
old = old.replace('import shutil\n', '')
start = old.index('\ndef imu_brackets(')
end = old.index('\ndef ouster_scan_duration_ns(', start)
old = old[:start] + old[end:]
old = old.replace('    shutil.copy2(Path(__file__), args.output / "audit_source.py")\n', '')
old = old.replace('    scan_spans, livox_timebase_minus_header = [], []\n', '')
old = old.replace('                    scan_spans.append(span)\n                    if lidar_type == 1:\n                        livox_timebase_minus_header.append(int(msg.timebase) - timestamp)\n', '')
start = old.index('        report[f"{kind}_local_imu_brackets"]')
end = old.index('    if imu_norms:', start)
old = old[:start] + old[end:]
old = old.replace('imu_acceleration_norm_native_units', 'imu_acceleration_norm_m_s2')
old = old.replace('imu_angular_velocity_norm_native_units', 'imu_angular_velocity_norm_rad_s')
old = old.replace('        report["imu_units_scope"] = "Message type alone does not verify publisher units. Raw acceleration norm near one here; pinned IMU processor normalizes mean_acc.norm() to gravity. No SI conversion or unit calibration performed."\n', '')
out = lab / 'outputs/fast-livo2-bright-timing-audit-001'
metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
actual = hashlib.sha256(old.encode('utf-8')).hexdigest()
if actual != metrics['source_sha256']:
    raise AssertionError({'reconstructed': actual, 'expected': metrics['source_sha256']})
(out / 'audit_source.py').write_bytes(old.encode('utf-8'))
correction = {'status': 'preserved_original_metrics_with_explicit_unit_label_correction',
    'original_source_sha256_verified': actual,
    'incorrect_original_field': 'imu_acceleration_norm_m_s2',
    'correct_interpretation': 'Raw native publisher units; near-one norm does not prove m/s2. No SI calibration.',
    'angular_velocity_units': 'Original rad_s field was also schema-assumed, not independently verified.',
    'successor': 'outputs/fast-livo2-bright-timing-audit-002',
    'original_metrics_and_csv_unchanged': True}
(out / 'unit-label-correction.json').write_text(json.dumps(correction, indent=2)+'\n', encoding='utf-8')
for target in ('fast-livo2-position-evaluator-controls-002', 'fast-livo2-author-eee03-position-reanalysis-001'):
    folder = lab / 'outputs' / target
    position_script = lab / 'scripts/evaluate_fast_livo2_position.py'
    recorded = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))['source_sha256']
    if hashlib.sha256(position_script.read_bytes()).hexdigest() != recorded:
        raise AssertionError('Evaluation source no longer matches its results')
    shutil.copy2(position_script, folder / 'evaluator_source.py')
print(json.dumps(correction))

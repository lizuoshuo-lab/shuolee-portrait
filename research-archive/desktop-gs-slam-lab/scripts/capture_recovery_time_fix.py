"""Preserve the rejected recovery version and the observed compatibility fix."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/splatam_local/resume_ablation_batch.py'
OUT = ROOT / 'outputs/splatam-ablation-repeat-001/executions/20261010T093555Z_b8bac22d/controller-recovery'
source = SOURCE.read_text(encoding='utf-8')
old = source.replace('import re\n', '')
begin = old.index('            # PowerShell emits seven fractional digits;')
end = old.index('            epoch =', begin)
old = old[:begin] + '            cim_time = dt.datetime.fromisoformat(first["CreationUTC"].replace("Z", "+00:00"))\n' + old[end:]
old_bytes = old.encode('utf-8')
old_sha = hashlib.sha256(old_bytes).hexdigest()
assert old_sha == 'bafb411bea3edbf0154fb0a9b9024f16e25e17d520c2e0f9bf87faa07fdaca0c'
archive = OUT / 'resume-source-before-python310-time-fix.py'
with archive.open('xb') as stream:
    stream.write(old_bytes)
receipt = {
    'status': 'actual_recovery_pre_adoption_failure_preserved_and_fixed',
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'evidence_origin': 'Root observed actual exec terminal response, then actual read-only Windows handle check',
    'first_recovery_actual_exit_code': 1,
    'first_recovery_error': "Invalid isoformat string: '2026-10-10T10:50:25.8347670+00:00'",
    'original_receipt_or_training_mutated_by_failed_attempt': False,
    'previous_source_sha256': old_sha,
    'previous_source_archive_method': 'Reconstructed by reversing the documented four-line patch, byte SHA verified against pre-fix source',
    'fixed_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'actual_read_only_handle_validation_exit_code': 0,
    'actual_read_only_validation_utc': '2026-10-10T11:12:58.771043+00:00',
    'observed_process_creation_FILETIME_ticks': 134361030258347675,
    'fix': 'Trim only the CIM fractional seconds used for Python3.10 parsing; retain full raw CIM and exact FILETIME identity',
    'algorithm_or_plan_changed': False,
}
path = OUT / 'python310-cim-time-fix.json'
with path.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'receipt': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))

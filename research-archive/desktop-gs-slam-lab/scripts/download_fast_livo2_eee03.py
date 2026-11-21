"""Fetch the public official archive, preserving interrupted attempts; no SLAM."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

lab = Path(__file__).resolve().parent.parent
out = lab / 'outputs/fast-livo2-eee03-download-retry-001'
out.mkdir(parents=True, exist_ok=True)
archive = lab / 'data/fast-livo2/ntu-viral-eee03.zip'
expected_bytes = 1995367404
official_md5 = '9bf10b5455ee6d2962324ef557f86433'
url = 'https://researchdata.ntu.edu.sg/api/access/datafile/68132'
now = lambda: datetime.now(timezone.utc).isoformat()
record = {'status': 'preparing', 'observed_utc': now(), 'helper_pid': os.getpid(),
          'source': url, 'expected_bytes': expected_bytes, 'official_md5': official_md5,
          'scope': 'Public archive acquisition only; no fusion run, calibration or trajectory accuracy.'}
receipt = out / 'receipt.json'
def save():
    record['observed_utc'] = now()
    receipt.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
try:
    if archive.exists():
        previous = archive.stat().st_size
        # Never truncate a previous download. The endpoint was observed to ignore Range.
        preserved = archive.with_name(f'ntu-viral-eee03.interrupted-{previous}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.zip')
        if not preserved.resolve().is_relative_to(archive.parent.resolve()):
            raise ValueError('Destination escapes data directory')
        if preserved.exists():
            raise FileExistsError(preserved)
        archive.rename(preserved)
        record['preserved_previous_file'] = str(preserved)
        record['preserved_previous_bytes'] = previous
    curl = shutil.which('curl.exe')
    if not curl:
        raise FileNotFoundError('curl.exe unavailable')
    command = [curl, '--location', '--fail', '--silent', '--show-error',
               '--connect-timeout', '30', '--max-time', '14400',
               '--output', str(archive), '--dump-header', str(out/'headers.txt'), url]
    with (out/'curl.stdout.log').open('wb') as stdout, (out/'curl.stderr.log').open('wb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW)
        record.update({'status': 'downloading', 'curl_pid': process.pid, 'started_utc': now()})
        save()
        code = process.wait()
    record.update({'curl_returncode': code, 'actual_bytes': archive.stat().st_size if archive.exists() else 0})
    if code != 0 or record['actual_bytes'] != expected_bytes:
        raise RuntimeError('Transfer did not complete with the official archive size')
    record['status'] = 'verifying_complete_archive'; save()
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive.open('rb') as data:
        for block in iter(lambda: data.read(8 << 20), b''):
            md5.update(block); sha.update(block)
    record.update({'actual_md5': md5.hexdigest(), 'sha256': sha.hexdigest()})
    if md5.hexdigest() != official_md5:
        raise RuntimeError('Official MD5 differs')
    record.update({'status': 'complete_official_size_and_MD5_verified', 'completed_utc': now(),
                   'extracted': False, 'ROS1_index_read': False,
                   'next_validation': 'ZIP CRC and safe extraction into a fresh directory, then bag index and timing audit'})
    save()
except Exception as error:
    record.update({'status': 'incomplete_or_failed', 'error': repr(error)})
    save()
    raise

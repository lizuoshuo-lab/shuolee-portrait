"""One native Linux HTTPS EEE03 attempt. Never resume, retry, or replace old data."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import ssl
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import zipfile

SOURCE_URL = 'https://researchdata.ntu.edu.sg/api/access/datafile/68132'
EXPECTED_BYTES = 1995367404
EXPECTED_MD5 = '9bf10b5455ee6d2962324ef557f86433'
CHUNK_BYTES = 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 60
TOTAL_WALL_SECONDS = 1800
PROGRESS_INTERVAL_SECONDS = 2
LAB = Path(__file__).resolve().parents[2]
DESTINATION = LAB / 'data/fast-livo2/ntu-viral-eee03.native-20261010.zip'
OUTPUT = LAB / 'outputs/fast-livo2-linux-native-download-001'


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(str(temporary), str(path))


def public_source_url(value):
    # Keep the observed host/path while omitting user information and query credentials.
    parsed = urllib.parse.urlsplit(value)
    host = parsed.hostname or ''
    if parsed.port is not None:
        host += ':' + str(parsed.port)
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, '', ''))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    state = {
        'status': 'starting', 'run_id': run_id, 'started_utc': utc_now(),
        'source_url': SOURCE_URL, 'observed_source_url': None,
        'destination': str(DESTINATION), 'expected_bytes': EXPECTED_BYTES,
        'official_md5': EXPECTED_MD5, 'bytes_written': 0,
        'created_destination': False, 'destination_exists_before_attempt': DESTINATION.exists(),
        'chunk_bytes': CHUNK_BYTES, 'socket_timeout_seconds': SOCKET_TIMEOUT_SECONDS,
        'total_wall_seconds': TOTAL_WALL_SECONDS, 'progress_interval_seconds': PROGRESS_INTERVAL_SECONDS,
        'resume': False, 'automatic_retry': False,
        'tls': {'certificate_verification': True, 'hostname_verification': True,
                'context': 'ssl.create_default_context', 'openssl_version': ssl.OPENSSL_VERSION},
        'size_matches': False, 'official_md5_matches': False, 'zip_crc_all_entries_passed': False,
        'scope': 'Acquisition only. No extraction, SLAM, time synchronization, extrinsic calibration, or GT accuracy certification.',
    }
    lock = threading.Lock()
    stop_progress = threading.Event()
    worker_errors = []

    def progress_snapshot():
        with lock:
            return dict(state, elapsed_seconds=time.monotonic() - start, observed_utc=utc_now())

    def write_progress():
        snapshot = progress_snapshot()
        atomic_json(OUTPUT / 'progress.json', snapshot)
        print(json.dumps({k: snapshot[k] for k in ['status', 'bytes_written', 'elapsed_seconds',
                                                  'observed_utc', 'source_url', 'observed_source_url']}), flush=True)

    def progress_worker():
        while not stop_progress.wait(PROGRESS_INTERVAL_SECONDS):
            try:
                write_progress()
            except Exception as error:
                worker_errors.append(repr(error))
                return

    def wall_timeout(_signum, _frame):
        raise TimeoutError('Total wall budget of {} seconds exhausted'.format(TOTAL_WALL_SECONDS))

    worker = None
    timer_armed = False
    previous_alarm_handler = None
    exit_code = 1
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    try:
        if sys.platform != 'linux' or not hasattr(signal, 'setitimer'):
            raise RuntimeError('Run with native Linux Python 3.8+; SIGALRM is required for the total wall limit')
        if sys.version_info < (3, 8):
            raise RuntimeError('Python 3.8 or newer is required')
        # Reject an unexpected configured proxy without changing any environment or settings.
        if any(key.lower() in ('http', 'https', 'all') for key in urllib.request.getproxies()):
            raise RuntimeError('Native direct path required, but a proxy is configured; no settings were changed')
        context = ssl.create_default_context()
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise RuntimeError('Default TLS certificate/hostname verification is required')
        previous_alarm_handler = signal.signal(signal.SIGALRM, wall_timeout)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, TOTAL_WALL_SECONDS - (time.monotonic() - start)))
        timer_armed = True
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        # xb prevents a repeated invocation from silently replacing a previous attempt.
        with DESTINATION.open('xb') as output:
            with lock:
                state.update({'created_destination': True, 'status': 'connecting'})
            write_progress()
            worker = threading.Thread(target=progress_worker, name='download-progress', daemon=True)
            worker.start()
            request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'GS-SLAM-Lab-native-EEE03/1.0'})
            with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_SECONDS, context=context) as response:
                if response.status != 200:
                    raise RuntimeError('Expected an ordinary HTTP 200 full GET, received {}'.format(response.status))
                if not response.geturl().startswith('https://'):
                    raise RuntimeError('HTTPS is required for the final download source')
                kept_headers = {key: value for key, value in response.headers.items()
                                if key.lower() in ('content-type', 'content-length', 'transfer-encoding',
                                                   'content-range', 'content-disposition', 'date', 'server')}
                with lock:
                    state.update({'status': 'downloading', 'observed_source_url': public_source_url(response.geturl()),
                                  'http_status': response.status, 'response_headers': kept_headers})
                first_block = True
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    if first_block and not block.startswith(b'PK\x03\x04'):
                        raise RuntimeError('Response does not begin with the expected ZIP local-header signature')
                    first_block = False
                    if state['bytes_written'] + len(block) > EXPECTED_BYTES:
                        raise RuntimeError('Transfer exceeds the official expected byte count')
                    output.write(block)
                    md5.update(block)
                    sha256.update(block)
                    with lock:
                        state['bytes_written'] += len(block)
                output.flush()
                os.fsync(output.fileno())
        with lock:
            state.update({'status': 'verifying_size_and_md5', 'actual_md5': md5.hexdigest(),
                          'sha256': sha256.hexdigest(),
                          'size_matches': state['bytes_written'] == EXPECTED_BYTES,
                          'official_md5_matches': md5.hexdigest() == EXPECTED_MD5})
        if not state['size_matches'] or not state['official_md5_matches']:
            raise RuntimeError('Transfer ended without matching both official size and MD5')
        with lock:
            state['status'] = 'verifying_all_zip_crcs'
        with zipfile.ZipFile(DESTINATION, 'r') as archive:
            members = archive.infolist()
            if not members:
                raise RuntimeError('Expected a nonempty EEE03 ZIP archive')
            first_bad_member = archive.testzip()
            if first_bad_member is not None:
                raise RuntimeError('ZIP CRC failed for member: ' + first_bad_member)
            manifest = [{'name': member.filename, 'uncompressed_bytes': member.file_size,
                         'compressed_bytes': member.compress_size, 'zip_crc32': member.CRC,
                         'directory': member.is_dir()} for member in members]
        with lock:
            state.update({'status': 'complete_verified', 'zip_crc_all_entries_passed': True,
                          'zip_members': manifest, 'finished_utc': utc_now(),
                          'scope': 'Actual complete official ZIP size, MD5, and every ZIP member CRC verified. No extraction or sensor/SLAM/GT conclusions.'})
        exit_code = 0
    except Exception as error:
        with lock:
            state.update({'status': 'failed_preserved_attempt', 'error_type': type(error).__name__,
                          'error': str(error), 'exception_repr': repr(error),
                          'traceback': traceback.format_exc(), 'finished_utc': utc_now(),
                          'scope': 'This attempt did not satisfy all acquisition gates. Keep partial data and logs; do not promote or resume it automatically.'})
    finally:
        if timer_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
        stop_progress.set()
        if worker is not None:
            worker.join()
        if worker_errors:
            with lock:
                state['progress_logging_errors'] = worker_errors
        with lock:
            state.update({'exit_code': exit_code, 'elapsed_seconds': time.monotonic() - start,
                          'observed_destination_bytes': DESTINATION.stat().st_size if DESTINATION.exists() else 0,
                          'attempt_stream_hashed_bytes': state['bytes_written'],
                          'attempt_stream_md5': md5.hexdigest(),
                          'attempt_stream_sha256': sha256.hexdigest(),
                          'attempt_stream_hash_scope': 'Only blocks successfully written by this invocation; not a checksum of any prior existing destination or a resume checkpoint.',
                          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        snapshot = progress_snapshot()
        atomic_json(OUTPUT / ('receipt-' + run_id + '.json'), snapshot)
        atomic_json(OUTPUT / 'receipt.json', snapshot)
        atomic_json(OUTPUT / 'progress.json', snapshot)
        print(json.dumps(snapshot, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

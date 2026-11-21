"""Resume a user-authorized ENeRF-Outdoor archive from the author's Drive folder."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
from datetime import datetime, timezone
import hashlib, json, os, re, shutil, sys, time
import requests

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / 'outputs/dynamic-human-enerf-full-023'
ROOT = Path('D:/GS-SLAM-Data/enerf_outdoor')
ARCHIVE = ROOT / 'downloads/actor1_4.tar.gz'
PART = ARCHIVE.with_suffix('.gz.partial')
FILE_ID = '1YMqsNP-erPyL2T_KetpALUfMHDwptXIX'
EXPECTED = 12029956569

def now(): return datetime.now(timezone.utc).isoformat()
def write(name, value):
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temp, p)

class Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form': self.action = a.get('action')
        if tag == 'input' and a.get('type') == 'hidden' and a.get('name'):
            self.fields[a['name']] = a.get('value', '')

def response(session, offset):
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    r = session.get('https://drive.usercontent.google.com/download',
        params={'id': FILE_ID, 'export': 'download'}, headers=headers, stream=True, timeout=(20, 90))
    r.raise_for_status()
    if 'text/html' in r.headers.get('Content-Type', ''):
        doc = r.text; r.close()
        if 'Quota exceeded' in doc: raise RuntimeError('google_drive_download_quota_exceeded')
        f = Form(); f.feed(doc)
        if not f.action or urlparse(f.action).hostname != 'drive.usercontent.google.com' or f.fields.get('id') != FILE_ID:
            raise RuntimeError('Expected official large-file confirmation form missing')
        r = session.get(f.action, params=f.fields, headers=headers, stream=True, timeout=(20, 90))
        r.raise_for_status()
    if 'text/html' in r.headers.get('Content-Type', ''):
        doc = r.text; r.close()
        raise RuntimeError('google_drive_download_quota_exceeded' if 'Quota exceeded' in doc else 'unexpected_html_response')
    if offset:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
        if r.status_code != 206 or not match or int(match[1]) != offset or int(match[3]) != EXPECTED:
            r.close(); raise RuntimeError('Server did not honor exact resume range; existing partial preserved')
    length = int(r.headers.get('Content-Length', 0))
    if length and length != EXPECTED - offset:
        r.close(); raise RuntimeError(f'Unexpected content length {length}')
    return r

def main():
    OUT.mkdir(parents=True, exist_ok=True); ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if (OUT/'acquisition.json').exists():
        print('Already complete; inspect acquisition receipt.', flush=True); return
    if not (OUT/'source.json').exists():
        write('source.json', {'dataset': 'ENeRF-Outdoor', 'sequence_archive': ARCHIVE.name,
            'source': 'Author full-data folder provided by user after official application',
            'folder_url': 'https://drive.google.com/drive/folders/1CORowHi56hsfwpFOaH_SS3r-5s501CIt',
            'file_id': FILE_ID, 'metadata_size_bytes': EXPECTED,
            'citations': ['https://zju3dv.github.io/4k4d/', 'https://github.com/zju3dv/EasyVolcap'],
            'local_only': True, 'archive_path': str(ARCHIVE), 'received_utc': now()})
    offset = PART.stat().st_size if PART.exists() else 0
    if not ARCHIVE.exists():
        if offset > EXPECTED: raise RuntimeError('Partial exceeds official archive size')
        if shutil.disk_usage(ROOT).free < EXPECTED - offset + 60*1024**3:
            raise RuntimeError('Insufficient free space for download plus bounded extracted subset')
        for attempt in range(1, 5):
            offset = PART.stat().st_size if PART.exists() else 0
            if offset == EXPECTED: break
            write('state.json', {'status': 'connecting', 'bytes': offset, 'expected_bytes': EXPECTED,
                'pid': os.getpid(), 'attempt': attempt, 'updated_utc': now()})
            try:
                with requests.Session() as s, response(s, offset) as r:
                    start = time.monotonic(); tick = start; transferred = 0
                    with PART.open('ab' if offset else 'xb') as f:
                        for chunk in r.iter_content(1024*1024):
                            if offset + transferred + len(chunk) > EXPECTED: raise RuntimeError('Response exceeds metadata size')
                            f.write(chunk); transferred += len(chunk)
                            if time.monotonic() - tick > 10:
                                f.flush(); elapsed = time.monotonic() - start
                                state = {'status': 'downloading', 'bytes': offset+transferred, 'expected_bytes': EXPECTED,
                                    'bytes_per_second': transferred/elapsed, 'pid': os.getpid(), 'attempt': attempt, 'updated_utc': now()}
                                write('state.json', state); print(json.dumps(state), flush=True); tick = time.monotonic()
                    if PART.stat().st_size != EXPECTED: raise requests.ConnectionError('Incomplete response body')
                break
            except (requests.RequestException, OSError) as e:
                write(f'network-attempt-{attempt}-{os.getpid()}.json', {'error': str(e), 'updated_utc': now()})
                if attempt == 4: raise
                time.sleep(5)
        if PART.stat().st_size != EXPECTED: raise RuntimeError('Incomplete archive')
        PART.replace(ARCHIVE)
    if ARCHIVE.stat().st_size != EXPECTED: raise RuntimeError('Archive size mismatch')
    write('state.json', {'status': 'hashing', 'bytes': EXPECTED, 'pid': os.getpid(), 'updated_utc': now()})
    h = hashlib.sha256()
    with ARCHIVE.open('rb') as f:
        if f.read(2) != b'\x1f\x8b': raise RuntimeError('Not a gzip archive')
        f.seek(0)
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    write('acquisition.json', {'status': 'download_complete_inventory_pending', 'archive': str(ARCHIVE),
        'bytes': EXPECTED, 'sha256': h.hexdigest(), 'official_checksum_available': False, 'completed_utc': now(),
        'verification': 'Official folder byte count, gzip signature and local receipt hash; full gzip/tar integrity pending inventory'})
    write('state.json', {'status': 'download_complete_inventory_pending', 'bytes': EXPECTED, 'pid': os.getpid(), 'updated_utc': now()})
    print('COMPLETE', str(ARCHIVE), h.hexdigest(), flush=True)

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write('state.json', {'status': 'download_interrupted', 'reason': str(exc),
            'partial_bytes': PART.stat().st_size if PART.exists() else 0, 'pid': os.getpid(), 'updated_utc': now()})
        raise

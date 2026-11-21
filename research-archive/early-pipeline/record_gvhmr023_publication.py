from pathlib import Path
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BUILD = Path(r'C:\Users\Administrator\Documents\New project\shuolee-slide-update')
BASE = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
CURRENT = BASE / 'current/smplx-gvhmr-shoe-motion-fresh-023'
VIEWER = BASE.parent / 'viewer'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

receipt = read(BUILD / 'gvhmr023r3-deck-receipt.json')
validation = read(BUILD / 'gvhmr023r3-final-validation.json')
cleanup = read(CURRENT / 'publication-cleanup.json')
web = read(CURRENT / 'web-publication.json')
run = read(CURRENT / 'run-state.json')
canonical = Path(receipt['source'])
assert sha(canonical) == receipt['final_sha256'] == validation['finalSha256']
assert canonical.stat().st_size == validation['byteCount']
assert run['step'] == 1200 and run['running'] is False
assert sha(CURRENT / 'fitted.npz') == run['output_sha256']
assert cleanup['state'] == 'removed_and_protected_hashes_verified'
assert receipt['unchanged_rendered_pixels'] == list(range(1, 55))
assert (BUILD / 'gvhmr023-page-proof.png').is_file()
for asset in web['assets']:
    assert sha(VIEWER / 'body-fitting-method/gvhmr-motion' / asset['name']) == asset['sha256']

urls = ['body-smplx.html'] + ['body-fitting-method/gvhmr-motion/' + a['name'] for a in web['assets']]
def head(path):
    request = urllib.request.Request('http://127.0.0.1:8870/' + path, method='HEAD')
    with urllib.request.urlopen(request, timeout=8) as response:
        assert response.status == 200, (path, response.status)
        return path
with ThreadPoolExecutor(max_workers=4) as pool:
    http_verified = list(pool.map(head, urls))

now = datetime.now().astimezone().isoformat(timespec='seconds')
publication = {
    'created': now,
    'page': 'http://127.0.0.1:8870/body-smplx.html#gvhmr-motion-023',
    'page_file': str(VIEWER / 'body-smplx.html'),
    'browser_screenshot': str(BUILD / 'gvhmr023-page-proof.png'),
    'canonical_presentation': str(canonical),
    'presentation_sha256': receipt['final_sha256'],
    'changed_slides': [55],
    'preserved_slides': list(range(1, 55)),
    'validation_receipt': str(BUILD / 'gvhmr023r3-final-validation.json'),
    'slide55_preview': str(BUILD / 'gvhmr023r3-final-renders/55.png'),
    'rendering_boundary': 'All55 rendered; first54 XML/media/notes and rendered pixels unchanged; final55 visually inspected. Native PowerPoint not executed.',
    'cleanup': {'manifest': str(CURRENT / 'publication-cleanup.json'), 'files': cleanup['files'], 'bytes': cleanup['bytes']},
    'retention': 'One canonical PPT, final slide and browser proof, reproducible source/receipts, current023 and matched022 model arrays, compact022 review images. Unpublished draft PPTs, duplicate exports and superseded022 renders removed.',
    'verified_http_200': http_verified,
    'body_geometry_accepted': False,
    'new_body_fit_updates_this_turn': {'022': 1200, '023': 1200},
    'new_gaussian_updates': 0,
    'remote_deployed': False,
    'viewer_model_replaced': False,
}
save(CURRENT / 'publication.json', publication)
state_path = BASE / 'work-status.json'
state = read(state_path)
assert state['current_phase'] == 'gvhmr_shoe_motion_fresh_023', 'Another phase is active; do not overwrite its state.'
state['updated'] = now
state['latest_report'] = {
    'updated': now,
    'canonical_path': str(canonical),
    'final_sha256': receipt['final_sha256'],
    'final_bytes': validation['byteCount'],
    'slides': 55,
    'changed_slides': [55],
    'all55_rendered': True,
    'first54_pixel_identical': True,
    'slide55_visually_reviewed': True,
    'nativePowerPointVerified': False,
    'validated_output': str(canonical),
}
state['gvhmr_motion_023']['publication'] = publication
save(state_path, state)
print(json.dumps({'state': run['state'], 'http_ok': len(http_verified), 'canonical_sha256': sha(canonical), 'removed_files': cleanup['files'], 'removed_bytes': cleanup['bytes'], 'publication': str(CURRENT / 'publication.json')}, ensure_ascii=False))

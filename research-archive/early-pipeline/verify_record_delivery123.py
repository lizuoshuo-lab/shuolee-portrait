"""Verify image-only record delivery and refresh its existing viewer notice."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
from PIL import Image

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REF = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
RECORD = Path('D:/00/记录/005')
COORD = H.parent / 'shuolee-coordination'
stamp = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def windows_path(p):
    return Path(p.replace('/mnt/d/', 'D:/', 1))

rules = Path('D:/00/记录/规则.md')
previous = read(REF / 'observation-contract/record-delivery.json')
assert sha(rules) == previous['rules_sha256'], 'Record rules changed; review before publishing.'
expected = {f['path'].replace('\\', '/'): f['sha256'] for f in previous['files']}
membership = REF / 'solver-track-membership'
sources = [membership / 'rendered-evidence.json', membership / 'false-family-bridge/result.json']
for source in sources:
    data = read(source)
    for f in data.get('figures', data.get('images', [])):
        p = windows_path(f['path'])
        relative = p.relative_to(RECORD).as_posix()
        assert relative not in expected
        expected[relative] = f['sha256']

files = sorted(p for p in RECORD.rglob('*') if p.is_file())
assert len(files) == len(expected) == 81
assert all(p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} for p in files)
entries = []
for p in files:
    relative = p.relative_to(RECORD).as_posix()
    digest = sha(p)
    assert digest == expected[relative], relative
    with Image.open(p) as im:
        dimensions = list(im.size)
        im.verify()
    entries.append(dict(path=relative, sha256=digest, pixels=dimensions))

# Reserve the requested stage folders without substituting mesh images for GS.
for child in ['3DGS/原图', '3DGS/高斯']:
    (RECORD / child).mkdir(parents=True, exist_ok=True)
counts = dict(Counter(p.relative_to(RECORD).parts[0] for p in files))
counts['3DGS'] = 0
receipt = dict(
    updated=stamp, rule=str(rules), rules_sha256=sha(rules), dataset='video5.mp4',
    root=str(RECORD), image_count=len(files), stage_counts=counts,
    image_only=True, files_decoded_and_source_hashes_verified=True,
    image_text_policy='No burned-in words; stage, source frame and comparison order are in filenames.',
    source_receipts=[str(REF / 'observation-contract/record-delivery.json')] + [str(p) for p in sources],
    gaussian_render_generated=False, new_work_log_images=False,
    historical_work_log_preserved=True, cloud_sync_verified=False, files=entries)
receipt_path = REF / 'image-record-delivery.json'
save(receipt_path, receipt)

destinations = [
    T / 'shuolee/current/robust-portrait-122/video10-global-initializer',
    T / 'viewer/diagnostics/video10-global-initializer',
    Path('D:/00/工作/查看器/diagnostics/video10-global-initializer')]
for d in destinations:
    target = d / 'smplifyx-reference-control/image-record-delivery.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(receipt_path, target)
    p = d / 'index.html'
    page = p.read_text(encoding='utf-8')
    old = re.search(r'<section id="smplify-observation-contract">.*?</section>', page, flags=re.S)
    assert old is not None
    section = old.group()
    updated, n = re.subn(r'当前共\d+张图', f'当前共{len(files)}张图', section, count=1)
    assert n == 1
    updated = updated.replace('smplifyx-reference-control/observation-contract/record-delivery.json',
                              'smplifyx-reference-control/image-record-delivery.json')
    page = page[:old.start()] + updated + page[old.end():]
    p.write_text(page, encoding='utf-8')

for p in [T / 'viewer/body-smplx.html', Path('D:/00/工作/查看器/progress.html'),
          T / 'shuolee/current/robust-portrait-122/video10-global-initializer/viewer-article.html']:
    page = p.read_text(encoding='utf-8')
    page = page.replace('图片归档：D:/00/记录/005，43张，按步骤分类。',
                        f'图片归档：D:/00/记录/005，{len(files)}张，按步骤分类。')
    p.write_text(page, encoding='utf-8')

p = COORD / 'image-record-policy.json'
policy = read(p)
policy.update(updated=stamp, latest_verified_dataset='005', latest_image_count=len(files),
              latest_delivery_receipt=str(receipt_path), new_work_log_image_writes=False)
save(p, policy)
p = COORD / 'local-research-status.json'
state = read(p)
state.update(image_delivery_root='D:/00/记录', latest_image_record=str(RECORD),
             latest_image_count=len(files), latest_image_delivery_receipt=str(receipt_path),
             new_work_log_images=False, image_delivery_updated=stamp)
save(p, state)
print(json.dumps({k: receipt[k] for k in ['root', 'image_count', 'stage_counts',
      'image_only', 'files_decoded_and_source_hashes_verified', 'new_work_log_images',
      'gaussian_render_generated', 'cloud_sync_verified']}, ensure_ascii=False))

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = Path('D:/00/工作/SMPL-X')
CACHE = Path('D:/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap')
CACHE.mkdir(parents=True, exist_ok=True)
VERSIONS = {'pip': '25.1.1', 'setuptools': '79.0.1', 'wheel': '0.45.1'}
items = []
for package, version in VERSIONS.items():
    metadata_url = f'https://pypi.org/pypi/{package}/{version}/json'
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    matches = [item for item in metadata['urls']
               if item['packagetype'] == 'bdist_wheel' and item['filename'].endswith('-none-any.whl')]
    assert len(matches) == 1, matches
    item = matches[0]
    assert item['url'].startswith('https://files.pythonhosted.org/')
    path = CACHE/item['filename']
    digest = item['digests']['sha256']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        with urllib.request.urlopen(item['url'], timeout=60) as response:
            data = response.read()
        assert len(data) == item['size'] and hashlib.sha256(data).hexdigest() == digest
        path.write_bytes(data)
    items.append(dict(filename=path.name, sha256=digest, bytes=path.stat().st_size,
                      source=item['url'], metadata_url=metadata_url))
    print('OFFICIAL_WHEEL_VERIFIED', path.name, path.stat().st_size, flush=True)

manifest = dict(created=datetime.now(timezone.utc).isoformat(), versions=VERSIONS,
                wheels=items, purpose='Offline pip bootstrap only; no CUDA, weights or training data.')
out = DEST/'server113_pip_repair.pyz'
temp = out.with_suffix('.partial')
with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(ROOT/'server113_pip_repair_main.py', '__main__.py')
    archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    for item in items:
        archive.write(CACHE/item['filename'], 'wheels/'+item['filename'])
temp.replace(out)
import runpy
# Include the reviewed revision3 patch on every future rebuild.
runpy.run_path(str(ROOT/'build_server113_header_patch.py'), run_name='__main__')
digest = hashlib.sha256(out.read_bytes()).hexdigest()
(DEST/'server113_pip_repair.pyz.sha256').write_text(digest+'  '+out.name+'\n', encoding='ascii', newline='\n')
print(json.dumps(dict(path=str(out), bytes=out.stat().st_size, sha256=digest)))

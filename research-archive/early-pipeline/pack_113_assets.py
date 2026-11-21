"""Pack the user's existing, hash-verified assets outside WPS and Git."""
from pathlib import Path
import hashlib
import json
import zipfile

source = Path('D:/GS-SLAM-Data/shuolee-113-assets')
destination = Path('D:/GS-SLAM-Data/shuolee-transfer')
manifest = json.loads(Path('D:/00/工作/人体重建/configs/route113-inputs.json').read_text(encoding='utf-8'))['asset_files']
destination.mkdir(parents=True, exist_ok=True)
archive = destination / 'shuolee-113-assets.zip'
for item in manifest:
    data = (source / item['path']).read_bytes()
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
if not archive.exists():
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for item in manifest:
            handle.write(source / item['path'], item['path'])
with zipfile.ZipFile(archive) as handle:
    assert set(handle.namelist()) == {item['path'] for item in manifest}
    for item in manifest:
        assert hashlib.sha256(handle.read(item['path'])).hexdigest() == item['sha256'], item['path']
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(destination / (archive.name + '.sha256')).write_text(digest + '  ' + archive.name + '\n', encoding='ascii', newline='\n')
report = {'files': len(manifest), 'zip_bytes': archive.stat().st_size, 'sha256': digest,
          'source_and_archive_hashes_verified': True, 'in_git': False, 'in_wps': False,
          'copied_to_company00': False, 'uploaded_to_server': False}
(destination / 'asset-transfer-status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))

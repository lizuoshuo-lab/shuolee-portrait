"""Verify the handoff archive and copy only its declared source files."""
import hashlib
import json
from pathlib import Path
import zipfile

handoff=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/torch-clothing-consumer-123-20261029')
repo=Path('D:/Projects/Shuolee-torch-clothing-consumer-123')
out=Path('D:/Projects/Shuolee-validation-20261029/torch-clothing-consumer')
metadata=json.loads((handoff/'manifest.json').read_text(encoding='utf-8-sig'))
archive=handoff/'torch-clothing-consumer-123-source.zip'
sha=lambda value:hashlib.sha256(value).hexdigest()
assert sha(archive.read_bytes())==metadata['source_package_sha256']=='d6868ca069aa681a8b653018d3c350eb3f7ceb785c4d0375fdbd2e65ec816dba'
assert repo.is_dir() and (repo/'.git').is_file()
receipt={'archive_sha256':metadata['source_package_sha256'],'files':{}}
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    assert set(z.namelist())==set(metadata['files']),z.namelist()
    for name,expected in metadata['files'].items():
        original=handoff/'source'/name
        data=z.read(name)
        assert sha(data)==expected and original.read_bytes()==data,name
        target=repo/name
        assert not target.exists(),target
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        assert sha(target.read_bytes())==expected
        receipt['files'][name]={'original_sha256':expected,'copied_sha256':sha(target.read_bytes())}
(out/'handoff-import.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))

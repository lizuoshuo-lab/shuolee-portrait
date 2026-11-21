"""Check the research follow-up patch against preserved sources in an isolated copy."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

out=Path('D:/Projects/Shuolee-validation-20261029/world-units').resolve()
source=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/world-units-123-20261029/source')
target=Path('D:/Projects/Shuolee-world-units-contract-123')
paths=['src/shuolee_portrait/sequence_fitting.py','tests/test_camera_world_units.py']
patch=out/'world-units-followup.patch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='patch-check-',dir=out) as temp:
    root=Path(temp).resolve();root.relative_to(out)
    for path in paths:
        destination=root/path;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/path,destination)
    for args in (['--check'],[]):
        subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply',*args,str(patch)],cwd=root,check=True)
    files=[dict(path=p,sha256=sha(root/p),matches_published_source=sha(root/p)==sha(target/p),
                crlf_lines=(root/p).read_bytes().count(b'\r\n'),
                normalized_matches=(root/p).read_bytes().replace(b'\r\n',b'\n')==(target/p).read_bytes()) for p in paths]
    assert all(f['matches_published_source'] for f in files)
receipt=dict(passed=True,files=files,original_sources_modified=False,temporary_copy_removed=not Path(temp).exists(),
    line_endings='LF, matching repository attributes',
    initial_harness_note='Standalone Git apply used Windows autocrlf; normalized content matched. Rechecked with explicit repository LF policy.')
(out/'followup-patch-verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))

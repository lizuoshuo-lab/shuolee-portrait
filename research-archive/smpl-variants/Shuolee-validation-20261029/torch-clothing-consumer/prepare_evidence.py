"""Preserve handoff bytes and verify source-only fixes in an isolated copy."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

repo=Path('D:/Projects/Shuolee-torch-clothing-consumer-123')
handoff=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/torch-clothing-consumer-123-20261029')
source=handoff/'source'
out=Path('D:/Projects/Shuolee-validation-20261029/torch-clothing-consumer')
expected=json.loads((handoff/'manifest.json').read_text(encoding='utf-8-sig'))['files']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'files':{},'originals_preserved':True};patch=''
for name,original_sha in expected.items():
    original,final=source/name,repo/name
    assert sha(original)==original_sha,name
    final.write_bytes(final.read_bytes().replace(b'\r\n',b'\n'))
    data=final.read_text(encoding='utf-8')
    assert 'codex' not in data.lower() and 'D:/' not in data and 'C:/' not in data,name
    receipt['files'][name]={'original_sha256':original_sha,'final_sha256':sha(final)}
    if name.endswith('.py'):
        patch+=''.join(difflib.unified_diff(original.read_text(encoding='utf-8').splitlines(True),data.splitlines(True),
                                          fromfile='a/'+name,tofile='b/'+name))
patch_path=out/'torch-clothing-followup.patch';patch_path.write_bytes(patch.encode('utf-8'))
with tempfile.TemporaryDirectory(prefix='torch-clothing-patch-',dir=out) as temp:
    temp=Path(temp)
    for name in expected:
        if name.endswith('.py'):
            target=temp/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((source/name).read_bytes().replace(b'\r\n',b'\n'))
    r=subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply',str(patch_path)],
                     cwd=temp,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    for name in expected:
        if name.endswith('.py'):assert sha(temp/name)==sha(repo/name),name
receipt['patch']={'path':str(patch_path),'sha256':sha(patch_path),'apply_verified':True,
                  'temporary_checkout_removed':not temp.exists()}
(out/'source-and-patch-verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))

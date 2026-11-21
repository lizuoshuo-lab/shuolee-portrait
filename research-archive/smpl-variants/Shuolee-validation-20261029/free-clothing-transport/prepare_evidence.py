"""Verify originals and prepare the deployable three-file increment."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

repo = Path('D:/Projects/Shuolee-free-clothing-transport-123')
source = Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/handoffs/free-clothing-transport-123-20261029/source')
out = Path('D:/Projects/Shuolee-validation-20261029/free-clothing-transport')
expected = {
    'src/shuolee_portrait/free_clothing_transport.py': 'b0e271a3f66c15551e9aa0d0203f9fb31f8b46fced890eb4efed063f0e596c81',
    'tests/test_free_clothing_transport.py': 'f9239877b284f091986e38ce22cd7c484dae21f4e7438f358ae280fc85e184b3',
    'docs/自由衣物高斯逐帧运输.md': 'da9029014952694152d64994062b4c2f03e8b631fe3c5a17ddfea64ace3abd4c',
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'files': {}, 'originals_preserved': True}
patch = ''
for name, original_sha in expected.items():
    original, final = source/name, repo/name
    assert sha(original) == original_sha, name
    # Repository text uses LF; preserve original handoff byte-for-byte.
    final.write_bytes(final.read_bytes().replace(b'\r\n', b'\n'))
    data = final.read_text(encoding='utf-8')
    assert 'codex' not in data.lower(), name
    assert 'D:/' not in data and 'C:/' not in data, name
    receipt['files'][name] = {'original_sha256': original_sha, 'final_sha256': sha(final)}
    if name.endswith('.py'):
        patch += ''.join(difflib.unified_diff(original.read_text(encoding='utf-8').splitlines(True),
                           data.splitlines(True), fromfile='a/'+name, tofile='b/'+name))
patch_path = out/'free-clothing-followup.patch'
patch_path.write_bytes(patch.encode('utf-8'))
with tempfile.TemporaryDirectory(prefix='transport-patch-', dir=out) as temp:
    temp = Path(temp)
    for name in expected:
        if name.endswith('.py'):
            target = temp/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source/name).read_bytes().replace(b'\r\n',b'\n'))
    r = subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply',str(patch_path)],
                       cwd=temp,capture_output=True,text=True)
    assert r.returncode == 0, r.stderr
    for name in expected:
        if name.endswith('.py'):
            assert sha(temp/name) == sha(repo/name), name
receipt['patch'] = {'path': str(patch_path), 'sha256': sha(patch_path), 'apply_verified': True,
                    'temporary_checkout_removed': not temp.exists()}
(out/'source-and-patch-verification.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2,ensure_ascii=False))

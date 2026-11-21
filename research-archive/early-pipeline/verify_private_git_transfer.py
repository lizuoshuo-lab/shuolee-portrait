"""Verify the authorized private Git transfer using only the small probe commit."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
from datetime import datetime

PROJECT = Path('D:/00/工作/人体重建')
REPOSITORY = 'lizuoshuo-lab/shuolee-portrait'
EXPECTED_LOGIN = 'lizuoshuo-lab'
EXPECTED_HASH = '44cc0f49c4162fd52948756ab946ec003510b866e4ce53b4131aad99be820581'
RECORD = Path('D:/00/传送验证/传送状态.json')
CHECKOUT = Path(__file__).parent / 'transfer-verification' / 'github-clone'
ENV = {k: v for k, v in os.environ.items() if k.upper() not in ('GH_TOKEN', 'GITHUB_TOKEN')}
GIT_AUTH = ['git', '-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential']

def run(args, *, cwd=None, check=True):
    p = subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    if check and p.returncode:
        raise RuntimeError(f'{args[0]} {args[1]} failed: {p.stderr.strip()}')
    return p

def save(**fields):
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    record.update(fields)
    record['updated'] = datetime.now().astimezone().isoformat()
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

login = run(['gh', 'api', 'user', '--jq', '.login']).stdout.strip()
if login != EXPECTED_LOGIN:
    raise RuntimeError(f'Expected GitHub account {EXPECTED_LOGIN}; received {login}. No repository changed.')
tracked = run(['git', 'ls-files'], cwd=PROJECT).stdout.splitlines()
if sorted(tracked) != sorted(['.gitignore', '.gitattributes', 'transfer-check.txt']):
    raise RuntimeError('Probe-only file allowlist failed; do not upload formal materials during preflight.')
if run(['git', 'status', '--porcelain'], cwd=PROJECT).stdout.strip():
    raise RuntimeError('Local test repository has uncommitted changes. Review them before transmission.')
metadata = run(['gh', 'repo', 'view', REPOSITORY, '--json', 'isPrivate,nameWithOwner,url'], check=False)
if metadata.returncode:
    # Only a specific missing-repository response is sufficient to create a new one.
    if 'Could not resolve to a Repository' not in metadata.stderr:
        raise RuntimeError('Repository visibility check failed for a reason other than absence: '+metadata.stderr.strip())
    run(['gh', 'repo', 'create', REPOSITORY, '--private', '--description', 'Human Gaussian reconstruction code and configurations'])
    metadata = run(['gh', 'repo', 'view', REPOSITORY, '--json', 'isPrivate,nameWithOwner,url'])
repo = json.loads(metadata.stdout)
if repo['nameWithOwner'] != REPOSITORY or repo['isPrivate'] is not True:
    raise RuntimeError('Private repository verification failed. No push attempted.')
url = repo['url']+'.git'
origin = run(['git', 'remote', 'get-url', 'origin'], cwd=PROJECT, check=False)
if origin.returncode:
    run(['git', 'remote', 'add', 'origin', url], cwd=PROJECT)
elif origin.stdout.strip() != url:
    raise RuntimeError('Existing origin differs from the intended private repository.')
run(GIT_AUTH+['push', '-u', 'origin', 'main'], cwd=PROJECT)
commit = run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).stdout.strip()
save(github_remote=repo['url'], github_private_verified=True, github_push_verified=True, github_commit=commit)
if CHECKOUT.exists():
    raise RuntimeError('Clean-clone destination already exists; preserve it and review before repeating.')
CHECKOUT.parent.mkdir(exist_ok=True)
run(GIT_AUTH+['clone', '--depth', '1', url, str(CHECKOUT)])
actual = hashlib.sha256((CHECKOUT/'transfer-check.txt').read_bytes()).hexdigest()
if actual != EXPECTED_HASH:
    raise RuntimeError('Downloaded test-file hash does not match its local source.')
if run(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT).stdout.strip() != commit:
    raise RuntimeError('Cloned revision differs from the pushed revision.')
save(github_fresh_clone_verified=True, github_download_sha256=actual, remote_server_verified=False)
print(json.dumps({'repository':repo['url'], 'private':True, 'push':True, 'clean_clone':True, 'sha256':actual, 'remote_server_verified':False}, ensure_ascii=False))

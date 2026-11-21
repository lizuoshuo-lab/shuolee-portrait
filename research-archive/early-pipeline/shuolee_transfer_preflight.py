from pathlib import Path
import hashlib
import json
from datetime import datetime

root = Path('D:/00')
probe = root / '传送验证'
project = root / '工作' / '人体重建'
probe.mkdir(exist_ok=True)
project.mkdir(exist_ok=True)
content = 'shuolee transfer verification\nUTF-8: 传送验证，内容一致。\nRevision: 1\n'.encode('utf-8')
for destination in (probe / 'shuolee-transfer-check.txt', project / 'transfer-check.txt'):
    if destination.exists() and destination.read_bytes() != content:
        raise RuntimeError(f'Refusing to overwrite changed test file: {destination}')
    destination.write_bytes(content)
ignore = project / '.gitignore'
if not ignore.exists():
    ignore.write_text('''# Only code and configuration belong in Git.
*.ply
*.splat
*.spz
*.pt
*.pth
*.ckpt
*.pkl
*.npz
*.mp4
*.mov
*.zip
data/
datasets/
models/
weights/
outputs/
private/
.env
.env.*
!.env.example
__pycache__/
.venv/
''', encoding='utf-8')
record = {
    'updated': datetime.now().astimezone().isoformat(),
    'purpose': 'Small-file round-trip validation before preparing or uploading formal materials',
    'filename': 'shuolee-transfer-check.txt',
    'bytes': len(content),
    'sha256': hashlib.sha256(content).hexdigest(),
    'github_account': 'lizuoshuo-lab',
    'github_remote': None,
    'github_push_verified': False,
    'github_fresh_clone_verified': False,
    'wps_folder_sync_verified': False,
    'wps_download_verified': False,
    'remote_server_verified': False,
}
evidence = probe / '传送状态.json'
if not evidence.exists():
    evidence.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
state_file = root / '整理状态.json'
state = json.loads(state_file.read_text(encoding='utf-8-sig')) if state_file.exists() else {}
state.update(updated=record['updated'], status='transfer_preflight', current_priority='First validate Git private push/clean clone and WPS small-file sync/download. Do not prepare additional materials yet.', human_git='awaiting_cli_web_authorization', wps_sync='not_configured', evidence=str(evidence))
state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'test_file':str(probe / record['filename']), 'sha256':record['sha256'], 'bytes':record['bytes']}, ensure_ascii=False))

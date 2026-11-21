"""Exercise the documented CPU install against Git in a temporary Linux home."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path('/mnt/d/Projects/Shuolee-frame-pose-contract-123')
out = Path('/mnt/d/Projects/Shuolee-validation-20261029/frame-pose-contract')
doc = (root/'docs/逐帧高斯输入约定.md').read_text(encoding='utf-8')
commands = re.findall(r'```bash\n(.*?)\n```', doc, re.S)
command = next(c for c in commands if c.startswith("bash -c 'set -eu; git clone"))
receipt = dict(kind='local WSL fresh Git checkout using documented command',
               expected_commit=os.environ.get('FRAME_POSE_EXPECTED_COMMIT', '22cea46d86a3f0fac24a6e922ba77b15a133a203'),
               existing_network_proxy_used=bool(os.environ.get('HTTPS_PROXY')),
               git_version=subprocess.check_output(['git','--version'], text=True).strip(),
               server_verified=False, training_started=False)
with tempfile.TemporaryDirectory(prefix='frame-pose-git-') as temp:
    env = dict(os.environ, HOME=temp, SMPLX_ENV_DIR=str(Path(sys.executable).parent.parent),
               GIT_TERMINAL_PROMPT='0')
    with (out/'git-install.log').open('w', encoding='utf-8') as log:
        r = subprocess.run(['bash', '-c', command], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    receipt['returncode'] = r.returncode
    repo = Path(temp)/'smpl-x-frame-contract'
    report = repo/'test-results/frame-pose-cpu.json'
    if report.exists():
        receipt['tests'] = json.loads(report.read_text(encoding='utf-8'))
    if (repo/'.git').exists():
        git = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
        receipt['actual_commit'] = git('rev-parse','HEAD')
        receipt['status'] = git('status','--porcelain')
        receipt['sparse_directories'] = git('sparse-checkout','list').splitlines()
receipt['temporary_checkout_removed'] = not Path(temp).exists()
receipt['passed'] = receipt['returncode']==0 and receipt.get('actual_commit')==receipt['expected_commit'] and receipt.get('tests',{}).get('passed') is True
(out/'git-install.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if receipt['passed'] else 1)

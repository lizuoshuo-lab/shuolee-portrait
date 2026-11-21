"""Replay the published CPU check command in a temporary Linux home."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path('/mnt/d/Projects/Shuolee-world-units-contract-123')
out = Path('/mnt/d/Projects/Shuolee-validation-20261029/world-units')
doc = (root/'docs/共同相机单位与部署边界.md').read_text(encoding='utf-8')
command = next(c for c in re.findall(r'```bash\n(.*?)\n```',doc,re.S)
               if c.startswith("bash -c 'set -eu;"))
receipt = dict(kind='local WSL fresh Git checkout using published CPU command',
    expected_commit='8d54497c0395a0a8a8f378a33d4895f5ffcd56be',
    git_version=subprocess.check_output(['git','--version'],text=True).strip(),
    existing_network_proxy_used=bool(os.environ.get('HTTPS_PROXY')),
    command_disables_cuda=True, server_verified=False, training_started=False)
with tempfile.TemporaryDirectory(prefix='world-units-git-') as temp:
    env = dict(os.environ,HOME=temp,SMPLX_ENV_DIR=str(Path(sys.executable).parent.parent),
               GIT_TERMINAL_PROMPT='0')
    with (out/'git-install.log').open('w',encoding='utf-8') as log:
        run = subprocess.run(['bash','-c',command],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    receipt['returncode']=run.returncode
    repo=Path(temp)/'smpl-x-world-units'
    report=repo/'test-results/world-units-cpu.json'
    if report.exists(): receipt['tests']=json.loads(report.read_text(encoding='utf-8'))
    if (repo/'.git').exists():
        git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        receipt['actual_commit']=git('rev-parse','HEAD')
        receipt['status']=git('status','--porcelain')
        receipt['sparse_directories']=git('sparse-checkout','list').splitlines()
receipt['temporary_checkout_removed']=not Path(temp).exists()
receipt['passed']=(receipt['returncode']==0 and receipt.get('actual_commit')==receipt['expected_commit']
                   and receipt.get('tests',{}).get('passed') is True)
(out/'git-install.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if receipt['passed'] else 1)

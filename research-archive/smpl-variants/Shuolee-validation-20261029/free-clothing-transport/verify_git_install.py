"""Replay the published CPU command in a temporary Linux home."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args()
root=Path('/mnt/d/Projects/Shuolee-free-clothing-transport-123')
out=Path('/mnt/d/Projects/Shuolee-validation-20261029/free-clothing-transport')
doc=(root/'docs/自由衣物高斯逐帧运输.md').read_text(encoding='utf-8')
command=next(c for c in re.findall(r'```bash\n(.*?)\n```',doc,re.S) if c.startswith("bash -c 'set -eu;"))
receipt=dict(kind='local WSL fresh Git checkout using published CPU command',expected_commit=args.commit,
    git_version=subprocess.check_output(['git','--version'],text=True).strip(),
    existing_network_proxy_used=bool(os.environ.get('HTTPS_PROXY')),
    command_disables_cuda=True,server_verified=False,training_started=False)
with tempfile.TemporaryDirectory(prefix='free-clothing-git-') as temp:
    env=dict(os.environ,HOME=temp,SMPLX_ENV_DIR=str(Path(sys.executable).parent.parent),GIT_TERMINAL_PROMPT='0')
    with (out/'git-install.log').open('w',encoding='utf-8') as log:
        run=subprocess.run(['bash','-c',command],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    receipt['returncode']=run.returncode
    repo=Path(temp)/'smpl-x-free-clothing'
    report=repo/'test-results/free-clothing-transport-cpu.json'
    if report.exists():receipt['tests']=json.loads(report.read_text(encoding='utf-8'))
    if (repo/'.git').exists():
        git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()
        receipt['actual_commit']=git('rev-parse','HEAD')
        receipt['status']=git('status','--porcelain')
        receipt['sparse_directories']=git('sparse-checkout','list').splitlines()
receipt['temporary_checkout_removed']=not Path(temp).exists()
receipt['passed']=(receipt['returncode']==0 and receipt.get('actual_commit')==args.commit
    and receipt.get('tests',{}).get('passed') is True and receipt.get('status')=='')
(out/'git-install.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if receipt['passed'] else 1)

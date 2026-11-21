"""Execute the documented Git command in a disposable Linux HOME."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--http1',action='store_true');a=p.parse_args()
repo=Path('/mnt/d/Projects/Shuolee-torch-clothing-consumer-123')
out=Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer')
doc=(repo/'docs/逐帧Torch消费层.md').read_text(encoding='utf-8')
command=next(c for c in re.findall(r'```bash\n(.*?)\n```',doc,re.S) if c.startswith("bash -c 'set -eu;"))
receipt={'kind':'local WSL fresh Git checkout using documented CPU command','expected_commit':a.commit,
         'git_version':subprocess.check_output(['git','--version'],text=True).strip(),
         'existing_network_proxy_used':bool(os.environ.get('HTTPS_PROXY')),
         'server_verified':False,'training_started':False,'command_disables_cuda':True,
         'process_local_git_http_version':'HTTP/1.1' if a.http1 else 'default'}
with tempfile.TemporaryDirectory(prefix='torch-clothing-git-') as temp:
    env=dict(os.environ,HOME=temp,SMPLX_ENV_DIR=str(Path(sys.executable).parent.parent),GIT_TERMINAL_PROMPT='0')
    if a.http1:env['GIT_CONFIG_PARAMETERS']="'http.version=HTTP/1.1'"
    with (out/'git-install.log').open('w',encoding='utf-8') as log:
        with subprocess.Popen(['bash','-c',command],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as process:
            try:returncode=process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                receipt['timed_out']=True
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                returncode=124
    receipt['returncode']=returncode
    checkout=Path(temp)/'smpl-x-torch-clothing'
    report=checkout/'test-results/torch-clothing-consumer-cpu.json'
    if report.exists():receipt['tests']=json.loads(report.read_text(encoding='utf-8'))
    if returncode==0 and (checkout/'.git').exists():
        # The handoff explicitly requires the full suite from a fresh Git install.
        check_env=dict(env,CUDA_VISIBLE_DEVICES='')
        with (out/'fresh-regression-runner.log').open('w',encoding='utf-8') as log:
            full=subprocess.run([sys.executable,'-B',str(out/'verify_branch.py'),'--repo',str(checkout),
                                 '--out',str(out/'fresh-regression')],env=check_env,stdout=log,
                                stderr=subprocess.STDOUT,timeout=120)
        receipt['fresh_full_regression_returncode']=full.returncode
        full_report=out/'fresh-regression/linux-verification.json'
        if full_report.exists():receipt['full_regression']=json.loads(full_report.read_text(encoding='utf-8'))
    if (checkout/'.git').exists():
        git=lambda *args:subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
        receipt['actual_commit']=git('rev-parse','HEAD');receipt['status']=git('status','--porcelain')
        receipt['checkout_mode']='complete shallow clone'
receipt['temporary_checkout_removed']=not Path(temp).exists()
receipt['passed']=(receipt['returncode']==0 and receipt.get('actual_commit')==a.commit
                  and receipt.get('tests',{}).get('outcome')=='passed' and receipt.get('status')==''
                  and receipt.get('fresh_full_regression_returncode')==0
                  and receipt.get('full_regression',{}).get('passed') is True)
(out/'git-install.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if receipt['passed'] else 1)

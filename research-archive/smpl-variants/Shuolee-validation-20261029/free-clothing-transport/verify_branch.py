"""Local CPU verification receipt; kept outside the source checkout."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

import numpy
import PIL
import torch

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--real-input', type=Path)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
os.chdir(a.repo)
sys.path.insert(0, str(a.repo / 'src'))
sys.path.insert(0, str(a.repo))
receipt = dict(hostname=platform.node(), platform=platform.platform(),
               python=sys.version, executable=sys.executable,
               numpy=numpy.__version__, pillow=PIL.__version__,
               torch=torch.__version__, torch_cuda_build=torch.version.cuda,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               cuda_available=torch.cuda.is_available(),
               repo=str(a.repo), checks=[], training_started=False,
               server_verified=False, physical_quality_accepted=False)
assert not receipt['cuda_available'], 'This verification must run with CUDA hidden.'
start = time.monotonic()
with (a.out / 'linux-regression.log').open('w', encoding='utf-8') as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        suite = unittest.defaultTestLoader.discover(str(a.repo / 'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
receipt['regression'] = dict(tests_run=result.testsRun, passed=result.wasSuccessful(),
                             failures=len(result.failures), errors=len(result.errors),
                             skipped=len(result.skipped), seconds=time.monotonic()-start)
commands = [
    [sys.executable, '-B', 'scripts/check_frame_pose_inputs.py', '--help'],
    [sys.executable, '-B', 'run_smplx.py', 'train', '--help'],
    [sys.executable, '-B', 'watch.py', '--help'],
    ['bash', '-n', 'deploy.sh', 'scripts/setup_server.sh', 'scripts/train.sh'],
    ['bash', 'deploy.sh', '--help'],
    ['bash', 'scripts/setup_server.sh', '--help'],
]
for i, command in enumerate(commands):
    r = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (a.out / f'entrypoint-{i}.log').write_text(r.stdout+r.stderr, encoding='utf-8')
    receipt['checks'].append(dict(command=command, returncode=r.returncode))
if a.real_input:
    r = subprocess.run([sys.executable, '-B', 'scripts/check_frame_pose_inputs.py',
                        str(a.real_input)], capture_output=True, text=True, timeout=60)
    receipt['real_input'] = dict(path=str(a.real_input),
        sha256=hashlib.sha256(a.real_input.read_bytes()).hexdigest(),
        returncode=r.returncode, stdout=r.stdout, stderr=r.stderr,
        expected='unsupported legacy schema; no manifest invented',
        contract_passed=False, training_admitted=False)
receipt['passed'] = receipt['regression']['passed'] and all(c['returncode']==0 for c in receipt['checks'])
(a.out/'linux-verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if receipt['passed'] else 1)

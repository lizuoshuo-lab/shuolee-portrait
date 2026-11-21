"""Resume only the authorized read-only route comparison after CUDA recovery."""
import json
from pathlib import Path
import subprocess
import time
import os

root=Path(__file__).resolve().parent/'wsl-service-repair'
state_path=root/'route-review-status.json'
ready=json.loads((root/'recovery-validation.json').read_text(encoding='utf-8'))
assert ready['cuda_passed'] and ready['phase']=='cuda_verified'
state=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),phase='rendering',
           optimizer_updates=0,quality_accepted=False,windows_reboot=False)
state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
started=time.monotonic()
with (root/'route-review-stdout.log').open('wb') as out, (root/'route-review-stderr.log').open('wb') as err:
    proc=subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/wsl.exe'),'-d','Ubuntu-20.04',
                         '--cd','/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee',
                         '--exec','/opt/gs-slam-lab/dynamic-human/.venv/bin/python','compare_full_body_routes.py'],
                        stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
state.update(phase='rendered' if proc.returncode==0 else 'failed',exit_code=proc.returncode,
             seconds=time.monotonic()-started,finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')

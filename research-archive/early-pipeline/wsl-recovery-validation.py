"""Validate the existing CUDA environment after a reversible WSLg workaround."""
import json
from pathlib import Path
import subprocess
import time
import os

root = Path(__file__).resolve().parent / 'wsl-service-repair'
state_path = root / 'recovery-validation.json'
state = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(), phase='cuda_probe',
             windows_reboot=False, distro_reinstalled=False, gpu_support_disabled=False,
             persistent_change='[wsl2] guiApplications=false', quality_accepted=False)
state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
code = '''import json, torch, time
assert torch.cuda.is_available()
x=torch.randn((256,256),device="cuda",requires_grad=True)
y=(x@x.T).square().mean()
y.backward()
torch.cuda.synchronize()
print(json.dumps({"marker":"WSL_TORCH_OK","torch":torch.__version__,"device":torch.cuda.get_device_name(0),"value":y.item(),"gradient_norm":x.grad.norm().item()}),flush=True)
'''
start = time.monotonic()
result = subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/wsl.exe'), '-d','Ubuntu-20.04','--exec',
                         '/opt/gs-slam-lab/dynamic-human/.venv/bin/python','-c',code],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
def decode(data):
    return data.decode('utf-16le' if b'\x00' in data else 'utf-8', errors='replace').lstrip('\ufeff')
(root/'cuda-probe-stdout.bin').write_bytes(result.stdout)
(root/'cuda-probe-stderr.bin').write_bytes(result.stderr)
(root/'cuda-probe-output.txt').write_text(decode(result.stdout), encoding='utf-8')
(root/'cuda-probe-stderr.txt').write_text(decode(result.stderr), encoding='utf-8')
state.update(exit_code=result.returncode, seconds=time.monotonic()-start,
             cuda_passed=result.returncode==0 and 'WSL_TORCH_OK' in decode(result.stdout),
             phase='cuda_verified' if result.returncode==0 and 'WSL_TORCH_OK' in decode(result.stdout) else 'cuda_probe_failed',
             finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')

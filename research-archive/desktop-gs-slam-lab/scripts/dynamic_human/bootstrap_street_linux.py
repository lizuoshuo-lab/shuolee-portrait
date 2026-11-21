"""Create an isolated, Blackwell-compatible Linux environment for street work.

Reuses previously verified CUDA/PyTorch downloads; never modifies SplaTAM's env.
This first stage installs Python dependencies only. CUDA operators and complete
DriveStudio imports must pass a later smoke test before any training claim.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path('/opt/gs-slam-lab/dynamic-human')
EXISTING = Path('/opt/gs-slam-lab/splatam')
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB/'outputs/dynamic-human-street-005'
PYTHON_BASE = EXISTING/'python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10'
UV = EXISTING/'tools/uv'
PYTHON = ROOT/'.venv/bin/python'


def run(command, name, env=None):
    log = ROOT/'logs'/name
    print('STREET ENV', name, flush=True)
    with log.open('a') as handle:
        subprocess.run([str(x) for x in command], stdout=handle, stderr=subprocess.STDOUT, check=True, env=env)


def main():
    assert sys.platform == 'linux' and PYTHON_BASE.exists() and UV.exists()
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    if not PYTHON.exists():
        run([UV, 'venv', '--python', PYTHON_BASE, ROOT/'.venv'], 'venv.log')
    env = os.environ.copy()
    env['UV_HTTP_TIMEOUT'] = '180'
    env['UV_CONCURRENT_DOWNLOADS'] = '3'
    versions = json.loads((EXISTING/'downloads/pytorch-wheel-selection.json').read_text())
    wheels = json.loads((EXISTING/'downloads/pytorch-gpu-local-wheels.json').read_text())
    core = [versions[name]['download_url'] for name in ['torch', 'torchvision']]+list(wheels.values())
    if not (ROOT/'torch-installed.json').exists():
        run([UV, 'pip', 'install', '--python', PYTHON, *core], 'torch-cu128.log', env)
        result = subprocess.check_output([str(PYTHON), '-c', 'import json,torch;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda)))'], text=True)
        value = json.loads(result)
        assert value['torch'] == '2.7.1+cu128' and value['cuda'] == '12.8'
        (ROOT/'torch-installed.json').write_text(json.dumps(value, indent=2)+'\n')
    requirements = [
        'pip==25.2', 'setuptools==80.9.0', 'wheel==0.45.1', 'ninja==1.13.0',
        'numpy==1.26.4', 'scipy==1.14.1', 'pandas==2.2.3', 'pillow==11.1.0',
        'timm==0.9.5', 'pytorch_msssim==1.0.0', 'omegaconf==2.3.0',
        'torchmetrics==0.10.3', 'tensorboard==2.11.0', 'wandb==0.15.8',
        'matplotlib==3.8.4', 'plotly==5.13.1', 'viser==0.2.1',
        'imageio==2.37.0', 'imageio-ffmpeg==0.6.0', 'scikit-image==0.20.0',
        'opencv-python-headless==4.10.0.84', 'open3d==0.18.0', 'pyquaternion==0.9.9',
        'kornia==0.7.2', 'gdown==5.2.0', 'nerfview==0.0.3', 'lpips==0.1.4',
        'trimesh==4.6.8', 'joblib==1.4.2', 'plyfile==1.1', 'rich==13.9.4',
        'fvcore==0.1.5.post20221221', 'iopath==0.1.10', 'gsplat==1.3.0',
    ]
    constraints = ROOT/'constraints.txt'
    constraints.write_text('torch==2.7.1+cu128\ntorchvision==0.22.1+cu128\nnumpy==1.26.4\n')
    run([UV, 'pip', 'install', '--python', PYTHON, '--constraint', constraints, *requirements], 'python-dependencies.log', env)
    destination = ROOT/'repos/drivestudio'
    if not destination.exists():
        shutil.copytree(LAB/'repos/drivestudio', destination, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    run([UV, 'pip', 'install', '--python', PYTHON, '--no-deps', '-e', destination/'third_party/smplx'], 'smplx.log', env)
    freeze = subprocess.check_output([str(UV), 'pip', 'freeze', '--python', str(PYTHON)], text=True)
    (ROOT/'requirements.freeze.txt').write_text(freeze)
    cuda = EXISTING/'cuda-extracted/usr/local/cuda-12.8'
    (ROOT/'env.sh').write_text(
        f'export CUDA_HOME={cuda}\n'
        f'export PATH={ROOT}/.venv/bin:$CUDA_HOME/bin:$PATH\n'
        'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n'
        "export TORCH_CUDA_ARCH_LIST='12.0'\nexport MAX_JOBS=4\nexport FORCE_CUDA=1\n"
        f'export TORCH_EXTENSIONS_DIR={ROOT}/torch-extensions\n'
        'export WANDB_MODE=disabled\nexport PYTHONUNBUFFERED=1\n')
    receipt = dict(status='python_dependencies_installed_cuda_operators_and_training_imports_pending',
        root=str(ROOT), python=str(PYTHON), torch=json.loads((ROOT/'torch-installed.json').read_text()),
        cuda_toolkit=str(cuda), torch_source_indices=versions,
        source_commit='e59bda4fa681f829dbb1d65f0de582b0f633c450',
        source_train_sha256=hashlib.sha256((destination/'tools/train.py').read_bytes()).hexdigest(),
        requested_dependencies=requirements, environment_freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest(),
        deviations=['Torch 2.7.1 CUDA 12.8 for RTX 5080 instead of author Torch 2.0 CUDA 11.7',
            'NumPy 1.26.4, matplotlib 3.8.4 and Open3D 0.18.0 for Python 3.10',
            'xformers/HMR pose extraction dependencies are not installed; author-provided poses already acquired'],
        pending=['PyTorch3D compiled operators', 'nvdiffrast compiled operators', 'gsplat forward/backward CUDA smoke',
            'full DriveStudio dataset/model imports', 'official SMPL template', 'street training and evaluation'])
    (OUT/'linux-environment-stage1.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['torch_source_indices','requested_dependencies']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

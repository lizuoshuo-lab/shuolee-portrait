"""Build pinned CUDA dependencies in the isolated street environment."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path('/opt/gs-slam-lab/dynamic-human')
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB/'outputs/dynamic-human-street-005'
PYTHON = ROOT/'.venv/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    deadline = time.monotonic()+7200
    while not (OUT/'linux-environment-stage1.json').exists():
        if args.wait_pid is None:
            raise RuntimeError('Python environment is not ready')
        process = Path(f'/proc/{args.wait_pid}/cmdline')
        if not process.exists() or b'bootstrap_street_linux.py' not in process.read_bytes():
            raise RuntimeError('Prerequisite installer is no longer live and has no completion receipt')
        if time.monotonic() > deadline:
            raise TimeoutError('Python prerequisite installation did not finish in two hours')
        print('WAITING FOR LIVE PYTHON INSTALLER', args.wait_pid, flush=True)
        time.sleep(30)
    cuda = Path('/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8')
    # CUDA runtime wheels contain cuSPARSE/cuBLAS/cuSOLVER headers outside
    # the minimal nvcc toolkit. Expose those installed official headers.
    nvidia = ROOT/'.venv/lib/python3.10/site-packages/nvidia'
    include_dirs = sorted(p for p in nvidia.glob('*/include') if p.is_dir())
    library_dirs = sorted(p for p in nvidia.glob('*/lib') if p.is_dir())
    for package, header in [('cusparse', 'cusparse.h'), ('cublas', 'cublas_v2.h'),
                            ('cusolver', 'cusolverDn.h')]:
        if not (nvidia/package/'include'/header).is_file():
            raise FileNotFoundError(f'Missing installed NVIDIA development header: {header}')
    env = os.environ.copy()
    env.update(CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST='12.0', FORCE_CUDA='1', MAX_JOBS='4',
        CUB_HOME=str(cuda/'include'), TORCH_EXTENSIONS_DIR=str(ROOT/'torch-extensions'),
        WANDB_MODE='disabled', PYTHONUNBUFFERED='1',
        PATH=f'{ROOT}/.venv/bin:{cuda}/bin:'+env.get('PATH',''),
        LD_LIBRARY_PATH='/usr/lib/wsl/lib:'+str(cuda/'lib64')+':'+env.get('LD_LIBRARY_PATH',''))
    env['CPATH'] = ':'.join(map(str, include_dirs))
    env['LIBRARY_PATH'] = ':'.join(map(str, library_dirs))
    env['LD_LIBRARY_PATH'] += ':' + env['LIBRARY_PATH']
    env_path = ROOT/'env.sh'
    env_lines = env_path.read_text().splitlines()
    env_lines = [line for line in env_lines if not line.startswith(('export CPATH=', 'export LIBRARY_PATH='))]
    env_lines.extend([f"export CPATH='{env['CPATH']}'", f"export LIBRARY_PATH='{env['LIBRARY_PATH']}'"])
    env_path.write_text('\n'.join(env_lines)+'\n')
    sources = json.loads((OUT/'linux-extension-sources.json').read_text())
    rows = []
    for item in sources:
        source = LAB/'repos'/f"{item['name']}-street"
        destination = ROOT/'repos'/item['name']
        if not destination.exists():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '.git', 'build'))
        log = ROOT/'logs'/f"build-{item['name']}.log"
        print('BUILDING STREET CUDA DEPENDENCY', item['name'], item['commit'], flush=True)
        started = time.monotonic()
        with log.open('a') as output:
            subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', str(destination)],
                cwd=destination, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        rows.append(dict(name=item['name'], commit=item['commit'], seconds=time.monotonic()-started,
            log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        (OUT/'linux-extension-build-progress.json').write_text(json.dumps(rows, indent=2)+'\n')
    code = ('import torch,pytorch3d,nvdiffrast,gsplat; '
        'from pytorch3d import _C; from datasets.argoverse.argoverse_sourceloader import ArgoVerseCameraData; '
        'from models.nodes.smpl import SMPLNodes; import tools.train; '
        'print("FULL STREET TRAINING IMPORTS PASSED", torch.__version__,pytorch3d.__version__,gsplat.__version__)')
    log = ROOT/'logs/full-training-imports.log'
    with log.open('w') as output:
        subprocess.run([str(PYTHON), '-c', code], cwd=ROOT/'repos/drivestudio', env=env,
            stdout=output, stderr=subprocess.STDOUT, check=True)
    receipt = dict(status='compiled_extensions_and_full_imports_passed_cuda_math_smoke_pending',
        builds=rows, imports_log=str(log), imports_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        pending=['CUDA forward/backward numeric smoke', 'camera capture adapter in full image loader',
            'SMPL neutral template and mesh alignment', 'street joint reconstruction and held-out evaluation'])
    (OUT/'linux-extension-build-completion.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()

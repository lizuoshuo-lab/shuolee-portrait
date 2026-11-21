"""Run a lab script with the local CUDA/MSVC toolchain, without global changes."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
VSDEV = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat')

def main():
    env = os.environ.copy()
    if VSDEV.exists():
        command = f'call "{VSDEV}" -arch=x64 -host_arch=x64 -vcvars_ver=14.29 >nul && set'
        result = subprocess.run(command, shell=True, capture_output=True,
                                text=True, errors='replace')
        if result.returncode:
            raise RuntimeError(f'MSVC setup failed: {result.stdout}\n{result.stderr}')
        for line in result.stdout.splitlines():
            if '=' in line:
                name, value = line.split('=', 1)
                env[name.upper()] = value
    env['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
    env['CUDA_PATH'] = env['CUDA_HOME']
    # A Ninja executable already accompanies the read-only linked GPU runtime.
    ninja_dir = r'C:\Users\Administrator\Documents\New project\job_materials\tools\ai3d\venv\Lib\site-packages\ninja\data\bin'
    env['PATH'] = str(PYTHON.parent) + os.pathsep + ninja_dir + os.pathsep + env['CUDA_HOME'] + '\\bin' + os.pathsep + env.get('PATH', '')
    env['TORCH_CUDA_ARCH_LIST'] = '12.0'
    env['TORCH_EXTENSIONS_DIR'] = str(ROOT / '.cache' / 'torch_extensions')
    env['MAX_JOBS'] = '4'
    env['DISTUTILS_USE_SDK'] = '1'
    env['VSLANG'] = '1033'
    env['VERBOSE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['GS_LAB_ROOT'] = str(ROOT)
    arguments = sys.argv[1:]
    if not arguments:
        raise SystemExit('Usage: python scripts/run_local.py path/to/experiment.py [args]')
    raise SystemExit(subprocess.call([str(PYTHON), *arguments], cwd=ROOT, env=env))

if __name__ == '__main__':
    main()

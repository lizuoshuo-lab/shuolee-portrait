"""Run the large-scene track in its own environment and CUDA build cache."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.venv-large-gs' / 'Scripts' / 'python.exe'
VSDEV = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat')

def main():
    env = os.environ.copy()
    if VSDEV.exists():
        result = subprocess.run(
            f'call "{VSDEV}" -arch=x64 -host_arch=x64 -vcvars_ver=14.29 >nul && set',
            shell=True, capture_output=True, text=True, errors='replace')
        if result.returncode:
            raise RuntimeError(f'MSVC setup failed: {result.stdout}\n{result.stderr}')
        for line in result.stdout.splitlines():
            if '=' in line:
                name, value = line.split('=', 1)
                env[name.upper()] = value
    env['CUDA_HOME'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
    env['CUDA_PATH'] = env['CUDA_HOME']
    ninja_dir = r'C:\Users\Administrator\Documents\New project\job_materials\tools\ai3d\venv\Lib\site-packages\ninja\data\bin'
    env['PATH'] = str(PYTHON.parent) + os.pathsep + ninja_dir + os.pathsep + env['CUDA_HOME'] + '\\bin' + os.pathsep + env.get('PATH', '')
    env['TORCH_CUDA_ARCH_LIST'] = '12.0'
    env['TORCH_EXTENSIONS_DIR'] = str(ROOT / '.cache' / 'large_gs_torch_extensions')
    env['TORCH_HOME'] = str(ROOT / '.cache' / 'large_gs_torch')
    env['TQDM_MININTERVAL'] = '5'
    env['MAX_JOBS'] = '4'
    env['DISTUTILS_USE_SDK'] = '1'
    env['VSLANG'] = '1033'
    env['VERBOSE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['GS_LAB_ROOT'] = str(ROOT)
    official_extensions = ROOT / 'outputs' / 'large-scene-gs-001' / 'hierarchy-build' / 'python-extensions' / 'cu128'
    if official_extensions.is_dir():
        env['PYTHONPATH'] = str(official_extensions) + os.pathsep + env.get('PYTHONPATH', '')
    if not sys.argv[1:]:
        raise SystemExit('Usage: run_local.py script.py [args] or -m module [args]')
    raise SystemExit(subprocess.call([str(PYTHON), *sys.argv[1:]], cwd=ROOT, env=env))

if __name__ == '__main__':
    main()

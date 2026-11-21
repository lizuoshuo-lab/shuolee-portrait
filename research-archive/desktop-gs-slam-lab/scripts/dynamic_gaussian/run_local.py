"""Reuse read-only CUDA 12.8 Torch packages with a separate pinned Warp overlay."""
import os
import subprocess
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[2]
env = os.environ.copy()
env['PYTHONPATH'] = str(LAB / '.dynamic-deps') + os.pathsep + str(LAB / 'repos/PhysGaussian')
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUNBUFFERED'] = '1'
raise SystemExit(subprocess.call([str(LAB / '.venv-large-gs/Scripts/python.exe'), str(LAB / 'scripts/large_scene_gs/run_local.py'), *sys.argv[1:]], env=env))

"""Create a separate overlay venv; never install into the existing experiment env."""
from pathlib import Path
import os, sys, json, subprocess, venv, datetime

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-4k4d-official-035'
BASE=Path('/opt/gs-slam-lab/4k4d')
BASE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
VENV=BASE/'.venv'
if not (VENV/'bin/python').exists(): venv.EnvBuilder(with_pip=False).create(VENV)
# Relocatable standalone CPython needs the true base directory rather than
# the launching experiment venv directory written by Python3.10's venv module.
(VENV/'pyvenv.cfg').write_text('home = '+str(Path(sys._base_executable).resolve().parent)+'\ninclude-system-site-packages = false\nversion = 3.10.19\n')
site=VENV/'lib/python3.10/site-packages'
(site/'existing-torch-runtime.pth').write_text('/opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages\n')
envscript='''source /opt/gs-slam-lab/dynamic-human/env.sh
export PATH=/opt/gs-slam-lab/4k4d/.venv/bin:$PATH
export TORCH_EXTENSIONS_DIR=/opt/gs-slam-lab/4k4d/torch-extensions
export TCNN_CUDA_ARCHITECTURES=120
export TCNN_BUILD_NO_NETWORKS=1
export MAX_JOBS=3
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export LIBRARY_PATH=/opt/gs-slam-lab/4k4d/lib:/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
'''
(BASE/'env.sh').write_text(envscript)
(BASE/'lib').mkdir(exist_ok=True)
nvrtc=BASE/'lib/libnvrtc.so'
if not nvrtc.exists():nvrtc.symlink_to('/opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),venv=str(VENV), inherited_read_only_packages='/opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages',purpose='Separate runtime for official4K4D, no training or image-quality claim',repositories={})
for repo in ['4K4D','EasyVolcap']:
    p=LAB/'repos'/repo
    receipt['repositories'][repo]=dict(path=str(p),commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),url=subprocess.check_output(['git','-C',str(p),'remote','get-url','origin'],text=True).strip())
(OUT/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
py=str(VENV/'bin/python')
subprocess.run([py,'-m','pip','install','pdbr','ujson','ruamel.yaml','PyGLM','PyTurboJPEG==1.7.7','h5py','yapf','av','PyMCubes','pyperclip','pyntcloud'],check=True)
subprocess.run([py,'-m','pip','install','-e',str(LAB/'repos/4K4D'),'--no-build-isolation','--no-deps'],check=True)
print(json.dumps(receipt),flush=True)

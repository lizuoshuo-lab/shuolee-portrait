from pathlib import Path
import sys,venv,subprocess,json
BASE=Path('/opt/gs-slam-lab/sam2');BASE.mkdir(exist_ok=True)
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038';OUT.mkdir(exist_ok=True)
VENV=BASE/'.venv'
if not (VENV/'bin/python').exists():venv.EnvBuilder(with_pip=False).create(VENV)
(VENV/'pyvenv.cfg').write_text('home = '+str(Path(sys.base_prefix)/'bin')+'\ninclude-system-site-packages = false\nversion = 3.10.19\n')
(VENV/'lib/python3.10/site-packages/existing-torch-runtime.pth').write_text('/opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages\n')
(BASE/'env.sh').write_text('''source /opt/gs-slam-lab/dynamic-human/env.sh
export PATH=/opt/gs-slam-lab/sam2/.venv/bin:$PATH
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/sam2:/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export SAM2_BUILD_CUDA=0
''')
subprocess.run([str(VENV/'bin/python'),'-m','pip','install','hydra-core==1.3.2'],check=True)
(OUT/'cpu-runtime.json').write_text(json.dumps(dict(venv=str(VENV),inherited_packages_read_only='/opt/gs-slam-lab/dynamic-human/.venv/lib/python3.10/site-packages',device='CPU only; CUDA_VISIBLE_DEVICES empty',purpose='Training-camera segmentation candidate; no RGB enhancement and no change to ongoing035baseline'),indent=2)+'\n')

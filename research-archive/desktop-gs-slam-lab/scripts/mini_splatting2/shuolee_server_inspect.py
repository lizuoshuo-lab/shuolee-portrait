"""Prepare the authorized namespace and inspect existing Linux dependencies."""
import getpass
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

home = Path('/home/zhouhaoyi')
assert platform.system() == 'Linux' and getpass.getuser() == 'zhouhaoyi'
assert os.getuid() != 0 and home.resolve() == home
root = home / 'shuolee'
assert not root.is_symlink()
if root.exists():
    assert (root / 'workspace.json').is_file()
    assert json.loads((root / 'workspace.json').read_text())['owner'] == 'shuolee'
else:
    root.mkdir()
    (root / 'workspace.json').write_text(json.dumps({'owner': 'shuolee', 'purpose': 'controlled reconstruction experiments'}))
for name in ['code', 'data', 'runs', 'best']:
    target = root / name
    assert not target.is_symlink()
    target.mkdir(exist_ok=True)

def run(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=40)
        return {'returncode': p.returncode, 'text': p.stdout[-16000:]}
    except Exception as exc:
        return {'error': str(exc)}

torch_probe = "import sys,torch,json,importlib.util;print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available(),modules={n:bool(importlib.util.find_spec(n)) for n in ['plyfile','cv2','scipy','PIL','tqdm','lpips','diff_gaussian_rasterization','simple_knn']})))"
result = {
    'owner': 'shuolee', 'hostname': platform.node(),
    'disk': shutil.disk_usage(home)._asdict(),
    'gpu': run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,driver_version', '--format=csv,noheader']),
    'gpu_processes': run(['nvidia-smi', '--query-compute-apps=pid,used_memory,process_name', '--format=csv,noheader']),
    'runtime_3dgs': run([str(home / 'miniconda3/envs/3dgs/bin/python'), '-c', torch_probe]),
    'compiler': run(['nvcc', '--version']),
    'compiler_dirs': sorted(str(p) for p in Path('/usr/local').glob('cuda*')),
    'locations': {}, 'training_started': False,
}
for rel in ['videos', 'video_utils', 'dataset/dataset', 'gaussian-splatting']:
    p = home / rel
    result['locations'][rel] = sorted(x.name for x in p.iterdir())[:80] if p.is_dir() else None
(root / 'inspection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))

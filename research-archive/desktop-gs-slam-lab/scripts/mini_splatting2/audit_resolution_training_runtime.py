"""Verify already running native training libraries via its loaded shared objects."""
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
rows=[]
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():continue
    try:args=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    if 'train_depth_guided.py' not in args or 'native-resolution-full-18000-s0' not in args or not args.startswith('/opt/gs-slam-lab/dynamic-human/.venv/bin/python '):continue
    libraries=sorted({line.split()[-1] for line in (folder/'maps').read_text().splitlines()
        if 'diff_gaussian_rasterization' in line and line.split()[-1].endswith('.so')})
    rows.append(dict(pid=int(folder.name),libraries=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in libraries]))
assert len(rows)==1 and rows[0]['libraries'],rows
assert any('/diff_gaussian_rasterization_ms/' in m['path'] for m in rows[0]['libraries'])
assert all('/mini-splatting2-native-mip-001/deps/' in m['path'] for m in rows[0]['libraries'])
out=ROOT/'native-resolution-evaluator-correction-001/training-runtime.json'
out.write_text(json.dumps(dict(status='verified',created=time.time(),processes=rows,training_uses_correct_mip_libraries=True),indent=2));print(out.read_text())

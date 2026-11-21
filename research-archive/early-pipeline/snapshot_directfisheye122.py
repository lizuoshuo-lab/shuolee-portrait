"""Current official DirectFisheye-GS source review, no install or execution."""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,hashlib
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/directfisheye-research')
O.mkdir(exist_ok=True)
if (O/'sources.json').exists():raise RuntimeError('Already snapshotted; inspect instead of repeating')
repo='Metaverse-AI-Lab-THU/DirectFisheye-GS'
def fetch(url,limit=2000000):
 with urlopen(Request(url,headers={'User-Agent':'Shuolee-ReadOnly-Research'}),timeout=30) as r:data=r.read(limit+1)
 if len(data)>limit:raise RuntimeError('Bounded source size exceeded')
 return data
head=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/main'))['sha']
tree=json.loads(fetch('https://api.github.com/repos/'+repo+'/git/trees/'+head+'?recursive=1'))
(O/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
names=['README.md','LICENSE.md','environment.yml','training_cli.py','training_core.py','train.py','convert.py','arguments/__init__.py','scene/__init__.py','scene/cameras.py','scene/dataset_readers.py','scene/gaussian_model.py','utils/camera_utils.py','utils/graphics_utils.py','gaussian_renderer/__init__.py','submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu','submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu','submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h','submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py']
existing={r['path'] for r in tree['tree'] if r['type']=='blob'}
def one(name):
 url='https://raw.githubusercontent.com/'+repo+'/'+head+'/'+name
 if name not in existing:return dict(path=name,error='Not in official tree')
 try:
  data=fetch(url);p=O/name.replace('/','__');p.write_bytes(data)
  return dict(path=name,permalink='https://github.com/'+repo+'/blob/'+head+'/'+name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),snapshot=str(p))
 except Exception as exc:return dict(path=name,error=repr(exc),url=url)
with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(one,names))
out=dict(updated=datetime.now(timezone.utc).isoformat(),repo=repo,commit=head,files=rows,scope='Public-source research only; no training/runtime/install/weights',previous_commit='db9b3a25bff79a59f02ce90f5bd235cb59c03aaa')
(O/'sources.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(commit=head,files=len(rows),downloaded=sum('sha256'in r for r in rows),errors=[r for r in rows if 'error'in r]),indent=2))

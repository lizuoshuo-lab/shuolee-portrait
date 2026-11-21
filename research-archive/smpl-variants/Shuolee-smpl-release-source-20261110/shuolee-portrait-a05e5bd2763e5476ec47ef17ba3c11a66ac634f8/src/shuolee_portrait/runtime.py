from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys
PROJECT=Path(__file__).resolve().parents[2]
CORE=PROJECT/'src/avatar_core'
ASSETS=PROJECT/'assets'
sys.path.insert(0,str(CORE))
import shuolee_common as c

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def configure(assets,root=None):
 global ASSETS
 if os.environ.get('SHUOLEE_EXTRA_PYTHONPATH','').strip():
  raise RuntimeError('SHUOLEE_EXTRA_PYTHONPATH is not supported in Git deployment. Unset it; install the pinned dependencies in the dedicated environment.')
 ASSETS=Path(assets).resolve()
 sys.path[:0]=[str(PROJECT/'third_party/gvhmr'),str(PROJECT/'third_party/sam2'),str(PROJECT/'third_party/human_body_prior')]
 c.PRIVATE=ASSETS;c.MODEL=ASSETS/'smplx/SMPLX_NEUTRAL.npz';c.LAB=PROJECT/'third_party'
 if root:c.ROOT=Path(root);c.BASE=Path(root)
 def torch_runtime():
  import torch
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable. Run doctor on the target server.')
  return torch
 c.runtime=torch_runtime
 import shuolee_vposer as vp
 vp.MODEL=ASSETS/'vposer/V02_05';vp.CODE=PROJECT/'third_party/human_body_prior'

def verify_assets(required_paths=None):
 manifest=read(PROJECT/'configs/server-assets.json')
 rows=manifest['files']
 if required_paths is not None:
  required=set(required_paths)
  if not required or not required <= {r['path'] for r in rows}:raise ValueError('Unknown or empty required asset selection.')
  rows=[r for r in rows if r['path'] in required]
 missing=[];bad=[]
 for row in rows:
  p=ASSETS/row['path']
  if not p.is_file():missing.append(row['path']);continue
  if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:bad.append(row['path'])
 if missing or bad:raise RuntimeError(f'Asset check failed. Missing={missing}; checksum mismatch={bad}. Model directory: {ASSETS}. Run bash deploy.sh to prepare server models; see docs/models.md and configs/server-assets.json.')
 return {'files':len(rows),'sha256_verified':True}

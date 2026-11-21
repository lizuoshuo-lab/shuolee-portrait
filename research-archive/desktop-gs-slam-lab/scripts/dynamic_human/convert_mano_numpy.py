"""One-time official MANO Python2/chumpy -> numeric arrays; originals retained."""
from pathlib import Path
import json,pickle,inspect,hashlib,shutil,builtins
import numpy as np
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
 if name not in np.__dict__:setattr(np,name,value)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
import chumpy
from scipy import sparse
root=Path('/mnt/d/GS-SLAM-Data/models/mano_v1_2');out=root/'numpy';out.mkdir(exist_ok=True)
records=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['MANO_RIGHT.pkl','MANO_LEFT.pkl']:
 src=root/name;dest=out/name
 with src.open('rb') as f:old=pickle.load(f,encoding='latin1')
 new={k:np.array(v.r,copy=True) if isinstance(v,chumpy.Ch) else v for k,v in old.items()}
 assert all(not isinstance(v,chumpy.Ch) for v in new.values())
 if not dest.exists():
  with dest.open('xb') as f:pickle.dump(new,f,protocol=4)
 with dest.open('rb') as f:check=pickle.load(f)
 assert check.keys()==old.keys()
 for k,v in old.items():
  if isinstance(v,chumpy.Ch):v=v.r
  if sparse.issparse(v):assert (v!=check[k]).nnz==0
  elif isinstance(v,np.ndarray):np.testing.assert_array_equal(v,check[k])
  else:assert v==check[k]
 assert check['v_template'].shape==(778,3)
 records.append(dict(file=name,original_sha256=sha(src),numpy_sha256=sha(dest),converted_keys=[k for k,v in old.items() if isinstance(v,chumpy.Ch)],all_values_exact=True))
shutil.copy2(root/'LICENSE.txt',out/'LICENSE.txt')
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-hand-prior-096');(O/'mano-conversion.json').write_text(json.dumps(dict(status='two_templates_numpy_conversion_exact_originals_preserved',records=records,license_preserved=True,no_site_export=True,goal_complete=False),indent=2)+'\n');print(json.dumps(records),flush=True)

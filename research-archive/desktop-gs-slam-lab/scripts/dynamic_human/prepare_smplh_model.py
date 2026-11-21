"""Register user-provided official SMPL+H neutral model locally only."""
from pathlib import Path
import zipfile,json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-prior-096'
src=Path('C:/Users/Administrator/Downloads/smplh_300.zip')
dest=Path('D:/GS-SLAM-Data/models/smplh_300').resolve();dest.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
with zipfile.ZipFile(src) as z:
 for name in ['smplh/LICENSE.txt','smplh/SMPLH_NEUTRAL.npz']:
  path=(dest/Path(name).name).resolve();assert path.parent==dest
  if not path.exists():
   temp=path.with_suffix(path.suffix+'.part')
   with z.open(name) as inp,temp.open('wb') as out:shutil.copyfileobj(inp,out,1024*1024)
   assert temp.stat().st_size==z.getinfo(name).file_size;temp.replace(path)
  else:assert path.stat().st_size==z.getinfo(name).file_size
  records.append(dict(file=str(path),bytes=path.stat().st_size,sha256=sha(path)))
p=dest/'SMPLH_NEUTRAL.npz'
with np.load(p,allow_pickle=False) as z:
 arrays={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in ['v_template','f','shapedirs','posedirs','weights','J_regressor','kintree_table','hands_componentsr','hands_meanr']}
 assert z['v_template'].shape==(6890,3) and z['weights'].shape==(6890,52)
 assert z['shapedirs'].shape==(6890,3,300)
out=dict(status='user_SMLPH_neutral_registered_not_MANO',archive=str(src),archive_sha256=sha(src),files=records,arrays=arrays,MANO_RIGHT_present=False,license_kept_locally=True,no_model_export_to_site=True,scope='SMPL+H full-body template with hand joints. Not a drop-in substitute for the778-vertex MANO layer required by HaMeR. Existing SMPL-fitted scene parameters need geometric compatibility/alignment checks before any model substitution.',goal_complete=False)
(O/'smplh-registration.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(out['arrays']))

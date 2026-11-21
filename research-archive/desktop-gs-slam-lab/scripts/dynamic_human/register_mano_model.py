from pathlib import Path
import zipfile,hashlib,json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-prior-096';src=Path('C:/Users/Administrator/Downloads/mano_v1_2.zip');root=Path('D:/GS-SLAM-Data/models/mano_v1_2').resolve();root.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();records=[]
with zipfile.ZipFile(src) as z:
 for name in ['mano_v1_2/models/MANO_RIGHT.pkl','mano_v1_2/models/MANO_LEFT.pkl','mano_v1_2/models/LICENSE.txt','mano_v1_2/models/info.txt']:
  path=(root/Path(name).name).resolve();assert path.parent==root
  with z.open(name) as source:
   if not path.exists():
    with path.open('xb') as out:shutil.copyfileobj(source,out)
   else:assert sha(path)==hashlib.sha256(source.read()).hexdigest()
  assert path.stat().st_size==z.getinfo(name).file_size;records.append(dict(file=str(path),bytes=path.stat().st_size,sha256=sha(path)))
(O/'mano-registration.json').write_text(json.dumps(dict(status='user_provided_MANO_v1_2_templates_registered',archive=str(src),archive_sha256=sha(src),files=records,models_local_only=True,no_site_export=True,goal_complete=False),indent=2)+'\n');print('MANO_RIGHT and MANO_LEFT locally registered')

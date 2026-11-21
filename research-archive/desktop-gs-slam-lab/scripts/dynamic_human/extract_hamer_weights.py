"""Verify author archive and extract only HaMeR/mean parameters, not ViTPose."""
from pathlib import Path,PurePosixPath
import json,tarfile,gzip,shutil,hashlib
R=Path('D:/GS-SLAM-Data/models/hamer').resolve();O=Path(__file__).resolve().parents[2]/'outputs/dynamic-human-hand-prior-096'
status=json.loads((R/'download-progress.json').read_text());p=R/'hamer_demo_data.tar.gz';assert status['status']=='download_complete_archive_not_yet_extracted' and p.stat().st_size==6037554929
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
assert sha(p)==status['sha256'];records=[];extracted=[]
with gzip.open(p,'rb') as gz:
 with tarfile.open(fileobj=gz,mode='r|') as tar:
  for info in tar:
   rel=PurePosixPath(info.name);assert not rel.is_absolute() and '..' not in rel.parts
   selected=info.isfile() and (str(rel).startswith('_DATA/hamer_ckpts/') or ('mean_params' in rel.name and rel.suffix=='.npz'))
   records.append(dict(name=str(rel),bytes=info.size,selected=selected))
   if not selected:continue
   dest=(R/Path(*rel.parts)).resolve();assert dest.is_relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True)
   temp=dest.with_suffix(dest.suffix+'.part');assert not dest.exists(),str(dest)
   source=tar.extractfile(info);assert source is not None
   with temp.open('wb') as target:shutil.copyfileobj(source,target,8*1024*1024)
   assert temp.stat().st_size==info.size;temp.replace(dest);extracted.append(dict(file=str(dest),bytes=info.size,sha256=sha(dest)));print('EXTRACTED',str(rel),info.size,flush=True)
 while gz.read(8*1024*1024):pass
assert any(r['file'].endswith('hamer.ckpt') for r in extracted)
(O/'weights-extraction.json').write_text(json.dumps(dict(status='author_archive_SHA_gzip_CRC_and_selected_files_verified',archive_sha256=status['sha256'],members=records,extracted=extracted,MANO_source='user-provided separate local archive',goal_complete=False),indent=2)+'\n');print('ARCHIVE_VERIFIED',len(extracted),flush=True)

"""Acquire the hand model referenced by the installed official rtmlib Hand API."""
from pathlib import Path
import json,urllib.request,zipfile,hashlib,shutil,time
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-observability-073'
R=Path('D:/GS-SLAM-Data/models/rtmhand');R.mkdir(parents=True,exist_ok=True)
url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip'
archive=R/'rtmpose-m-hand5-256x256.zip';partial=archive.with_suffix('.zip.partial')
if not archive.exists():
 with urllib.request.urlopen(url,timeout=40) as response,partial.open('wb') as target:
  n=0;last=time.monotonic()
  while True:
   block=response.read(4*1024*1024)
   if not block:break
   target.write(block);n+=len(block)
   if time.monotonic()-last>8:print('DOWNLOADED',n,flush=True);last=time.monotonic()
 partial.rename(archive)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 members=[n for n in z.namelist() if n.endswith('.onnx')];assert len(members)==1
 model=R/'rtmpose-m-hand5-256x256.onnx'
 if not model.exists():
  with z.open(members[0]) as source,model.open('wb') as target:shutil.copyfileobj(source,target)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(status='official_OpenMMLab_archive_CRC_checked',source=url,reference='Installedrtmlib0.0.16 tools/solution/hand.py',archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),model=str(model),model_bytes=model.stat().st_size,model_sha256=sha(model),archive_member=members[0],purpose='Local observable-hand detector comparison only, notGT or rendered detail synthesis')
(O/'hand-model-acquisition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

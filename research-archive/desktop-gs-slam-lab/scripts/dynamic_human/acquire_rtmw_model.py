from pathlib import Path
import json,urllib.request,zipfile,hashlib,shutil,time
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-wholebody-059';OUT.mkdir(exist_ok=True)
ROOT=Path('D:/GS-SLAM-Data/models/rtmw');ROOT.mkdir(parents=True,exist_ok=True)
url='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip'
archive=ROOT/'rtmw-l-384x288.zip';partial=ROOT/'rtmw-l-384x288.zip.partial'
if not archive.exists():
 with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as target:
  count=0;last=time.monotonic()
  while True:
   block=response.read(4*1024*1024)
   if not block:break
   target.write(block);count+=len(block)
   if time.monotonic()-last>8:print('DOWNLOADED',count,flush=True);last=time.monotonic()
 partial.rename(archive)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 members=[n for n in z.namelist() if n.endswith('.onnx')];assert len(members)==1,members
 model=ROOT/'rtmw-l-384x288.onnx'
 if not model.exists():
  with z.open(members[0]) as source,model.open('wb') as target:shutil.copyfileobj(source,target)
report=dict(status='official_openmmlab_ONNX_archive_CRC_checked',source=url,source_documentation='https://github.com/Tau-J/rtmlib#pose-estimators',
 archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
 model=str(model),model_bytes=model.stat().st_size,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),archive_member=members[0],
 runtime='isolated rtmlib0.0.16 andonnxruntime1.23.2 CPU',purpose='133wholebody keypoint observations, not GT or appearance generation')
(OUT/'model-acquisition.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(report))

from pathlib import Path
import shutil,json,hashlib,zipfile,urllib.request
P=Path('D:/00/工作/人体重建');LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');A=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cp(s,d):
 d.parent.mkdir(parents=True,exist_ok=True)
 if not d.exists() or d.stat().st_size!=s.stat().st_size:shutil.copy2(s,d)
for s in (P/'assets').rglob('*'):
 if s.is_file():cp(s,A/s.relative_to(P/'assets'))
for name in ['GVHMR-shuolee','sam2']:
 source=LAB/'repos'/name;target=P/'third_party'/('gvhmr' if name.startswith('GVHMR') else 'sam2')
 for s in source.rglob('*'):
  rel=s.relative_to(source)
  if any(x in {'.git','inputs','checkpoints','outputs','shuolee','__pycache__','build','dist','.venv','notebooks','assets','demo','tests'} or x.endswith('.egg-info') for x in rel.parts):continue
  if s.is_file() and s.suffix.lower() in {'.py','.yaml','.yml','.json','.cu','.cuh','.cpp','.h','.hpp','.txt','.toml','.cfg','.md'} or s.is_file() and s.name.startswith('LICENSE'):cp(s,target/rel)
for rel in ['gvhmr/gvhmr_siga24_release.ckpt','hmr2/epoch=10-step=25000.ckpt','vitpose/vitpose-h-multi-coco.pth']:
 cp(Path('D:/GS-SLAM-Data/models/gvhmr')/rel,A/'gvhmr'/rel)
for s in (LAB/'repos/GVHMR-shuolee/hmr4d').rglob('*'):
 if s.is_file() and s.suffix in ['.npz','.npy','.pt','.pth']:cp(s,A/'gvhmr/hmr4d'/s.relative_to(LAB/'repos/GVHMR-shuolee/hmr4d'))
# This is a licensed template copied from the user's official download, not Git content.
cp(A/'smplx/SMPLX_NEUTRAL.npz',A/'gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz')
cp(LAB/'models/sam2/sam2.1_hiera_tiny.pt',A/'sam2/sam2.1_hiera_tiny.pt')
model=Path('D:/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx');cp(model,A/'rtmw'/model.name)
with zipfile.ZipFile(model.with_suffix('.zip')) as z:pipe=json.loads(z.read('pipeline.json'))
norm=next(t for t in pipe['pipeline']['tasks'][0]['transforms'] if t['type']=='Normalize')
(A/'rtmw/normalization.json').write_text(json.dumps(norm,indent=2)+'\n')
target=A/'detection/yolov8n.pt';target.parent.mkdir(exist_ok=True)
url='https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt'
if not target.exists():
 with urllib.request.urlopen(url,timeout=60) as r,target.with_suffix('.download').open('wb') as f:shutil.copyfileobj(r,f)
 target.with_suffix('.download').replace(target)
# Runtime weights stay outside D:/00. Never mirror them into WPS.
rows=[]
for p in sorted(A.rglob('*')):
 if p.is_file():rows.append(dict(path=p.relative_to(A).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
(P/'configs/server-assets.json').write_text(json.dumps({'files':rows,'yolo_source':url,'licensed_assets_user_supplied':True,'weights_are_not_uploaded_to_Git':True},indent=2)+'\n')
print(json.dumps({'asset_files':len(rows),'bytes':sum(x['bytes'] for x in rows)}))

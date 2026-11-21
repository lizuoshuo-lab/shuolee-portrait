from pathlib import Path
import hashlib,json,os
from datetime import datetime,timezone

P=Path('D:/00/工作/人体重建').resolve()
R=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets').resolve()
B=P/'weight-parts'
LIMIT=512*1024*1024
rels=['gvhmr/hmr2/epoch=10-step=25000.ckpt','gvhmr/vitpose/vitpose-h-multi-coco.pth']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

rows=[]
for rel in rels:
 source=(P/'assets'/rel).resolve();outside=(R/rel).resolve()
 original=(Path('D:/GS-SLAM-Data/models')/rel).resolve()
 assert source.is_relative_to(P/'assets') and outside.is_relative_to(R)
 # Verify both our newly made copy and its original before moving out of WPS.
 expected=sha(original)
 if source.exists():
  assert sha(source)==expected and source.stat().st_size==original.stat().st_size
  if outside.exists():raise FileExistsError(outside)
  outside.parent.mkdir(parents=True,exist_ok=True)
  source.rename(outside)
 assert sha(outside)==expected
 print('Retained outside WPS: '+str(outside),flush=True)
 parts=[];folder=B/Path(rel).parent/Path(rel).name
 folder.mkdir(parents=True,exist_ok=True)
 with outside.open('rb') as stream:
  index=1
  while True:
   block=stream.read(LIMIT)
   if not block:break
   part=folder/f'{index:03d}.part';digest=hashlib.sha256(block).hexdigest()
   if part.exists():assert part.stat().st_size==len(block) and sha(part)==digest
   else:
    temp=part.with_suffix('.writing')
    with temp.open('xb') as f:f.write(block)
    temp.replace(part)
   parts.append({'path':part.relative_to(B).as_posix(),'bytes':len(block),'sha256':digest})
   print(f'Part {index}: {len(block)} bytes',flush=True);index+=1
 rows.append({'target':rel,'bytes':outside.stat().st_size,'sha256':expected,'parts':parts})
manifest={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'format':'raw_concatenation','max_part_bytes':LIMIT,'files':rows,'lossless':True,'originals_retained_outside_wps':True}
(B/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
large=[{'path':str(f),'bytes':f.stat().st_size} for f in Path('D:/00').rglob('*') if f.is_file() and f.stat().st_size>=2_000_000_000]
receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'oversized_files_remaining':large,'split_files':len(rows),'parts':sum(len(r['parts']) for r in rows),'originals_retained':True,'cloud_upload_verified':False}
(P/'provenance/wps-weight-transfer.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False),flush=True)

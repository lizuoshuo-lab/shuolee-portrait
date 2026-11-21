from pathlib import Path
import json,hashlib
from PIL import Image
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-visibility-100';V=O/'review';V.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=read(O/'comparison/summary.json')['panels'];output=[]
def sheets(group,paths):
 for i in range(0,len(paths),3):
  rr=paths[i:i+3];images=[Image.open(x) for x in rr];im=Image.new('RGB',(max(x.width for x in images),sum(x.height for x in images)),'white');y=0
  for a in images:im.paste(a,(0,y));y+=a.height
  path=V/f'{group}-{i//3:02d}.png';im.save(path);output.append(dict(file=str(path.relative_to(O)),sources=[str(x.relative_to(O)) for x in rr],sha256=sha(path)))
sheets('hands',[O/'comparison'/r['file'] for r in p if r['name'] in ['same_time_hand','target_right_hand']])
sheets('temporal',[O/'temporal'/r['file'] for r in read(O/'temporal/summary.json')['panels']])
for c,f in [('06',732),('10',729),('12',729),('12',731)]:
 r=next((r for r in p if r['camera']==c and r['frame']==f and r['name'] in ['full-person','worst-full-person']),None)
 if r is None:continue
 path=O/'comparison'/r['file'];output.append(dict(file=str(path.relative_to(O)),sources=[str(path.relative_to(O))],sha256=sha(path)))
(O/'review-plan.json').write_text(json.dumps(dict(status='fixed_contact_sheet_inventory',files=output),indent=2)+'\n');print('review files',len(output))

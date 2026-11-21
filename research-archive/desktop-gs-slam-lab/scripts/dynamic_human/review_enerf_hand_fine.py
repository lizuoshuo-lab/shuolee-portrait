from pathlib import Path
import json
from PIL import Image
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-fine-085';T=O/'temporal'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
files=[]
for c in ['04','13']:
    for part,start in enumerate(range(720,733,4)):
        ims=[Image.open(T/'panels'/f'camera{c}-frame{t:06d}-panel.png').convert('RGB') for t in range(start,min(start+4,733))]
        sheet=Image.new('RGB',(ims[0].width,sum(im.height for im in ims)),'white');y=0
        for im in ims:sheet.paste(im,(0,y));y+=im.height
        p=T/f'camera{c}-review-{part+1}.png';sheet.save(p);files.append(str(p.relative_to(O)).replace('\\','/'))
    files.append(f'temporal/camera{c}-comparison.png')
    files.extend(f'comparison/camera{c}-frame{t:06d}-target_right_hand.png' for t in [721,727,731])
files.extend(f'comparison/camera{c}-frame{t:06d}-both_hands_and_objects.png' for c in ['04','06','07','12'] for t in [721,727,731])
c=read(O/'comparison/summary.json');cases={('04',731),('13',731)}
for split in ['validation_time','validation_camera']:
    for b in ['fine_photo','fine_flow']:
        cases.update((r['camera'],r['frame']) for r in c['splits'][split]['versus_parent'][b]['worst'])
for c,t in sorted(cases):
    name=f'comparison/camera{c}-frame{t:06d}-full-person.png'
    if not (O/name).is_file():name=name.replace('-full-person.png','-worst-full-person.png')
    files.append(name)
assert len(files)==len(set(files)) and all((O/p).is_file() for p in files)
(O/'review-plan.json').write_text(json.dumps(dict(files=files,status='actual_visual_review_pending',count=len(files)),indent=2)+'\n')
print(json.dumps(dict(files=files,count=len(files))))

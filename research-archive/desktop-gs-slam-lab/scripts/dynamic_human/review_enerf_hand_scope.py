from pathlib import Path
import json
from PIL import Image
L=Path(__file__).resolve().parents[2]; O=L/'outputs/dynamic-human-enerf-hand-time-scope-088'; R=O/'review';R.mkdir(exist_ok=True)
plans=[]
def sheets(name,paths,n):
    for j in range(0,len(paths),n):
        group=paths[j:j+n];ims=[Image.open(p).convert('RGB') for p in group];canvas=Image.new('RGB',(max(x.width for x in ims),sum(x.height for x in ims)+12*(len(ims)-1)),(190,190,190));y=0
        for im in ims:canvas.paste(im,(0,y));y+=im.height+12
        dest=R/f'{name}-{j//n+1:02d}.png';canvas.save(dest);plans.append(dict(file=str(dest.relative_to(O)),sources=[str(p.relative_to(O)) for p in group]))
sheets('same-time',sorted((O/'comparison').glob('*same_time_hand.png')),3)
sheets('target',sorted((O/'comparison').glob('*target_right_hand.png')),3)
sheets('hand-object',sorted((O/'comparison').glob('*both_hands_and_objects.png')),3)
for c in ['04','13']:sheets('temporal-'+c,sorted((O/'temporal/panels').glob(f'camera{c}-*-panel.png')),4)
full=[O/'comparison'/f'camera{c}-frame000732-full-person.png' for c in ['00','06','12']]+sorted((O/'comparison').glob('*worst-full-person.png'))
plans += [dict(file=str(p.relative_to(O)),sources=[str(p.relative_to(O))]) for p in full]
(O/'review-plan.json').write_text(json.dumps(dict(sheets=plans,status='prepared_not_yet_viewed'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(sheets=len(plans),files=[x['file'] for x in plans])))

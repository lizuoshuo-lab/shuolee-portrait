"""Show the largest measured regression and its adjacent states without rescaling."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029'
summary=json.loads((out/'paired-statistics.json').read_text());metrics=json.loads((out/'evaluation/metrics.json').read_text())
worst=next(r for r in summary['comparisons'] if r['baseline']=='same-index')['worst_pair']
cam=worst['camera'];t=worst['common_frame'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
groups=['reference','static','same-index','offset-index'];labels=['Reference','Static / multi-time','Dynamic / same index','Dynamic / local offset'];records=[]
for frame in [t-1,t,t+1]:
    row=next(r for r in metrics['views'] if r['camera']==cam and r['common_frame']==frame and r['model']=='offset-index')
    images=[Image.open(out/'evaluation'/f'camera{cam}-frame{frame:03d}-{g}.png').convert('RGB') for g in groups]
    for p in row['people']:
        box=p['upper_body_bbox'];w=box[2]-box[0];h=box[3]-box[1]
        image=Image.new('RGB',(w*4,h+44),'white');draw=ImageDraw.Draw(image)
        for j,(im,label) in enumerate(zip(images,labels)):
            image.paste(im.crop(box),(j*w,44));draw.text((j*w+6,12),label,font=font,fill='black')
        name=f'camera{cam}-frame{frame:03d}-person{p["person"]}.png';image.save(out/'comparison'/name)
        records.append(dict(file=name,camera=cam,common_frame=frame,person=p['person'],crop=box,resized=False))
(out/'comparison/worst-pair-panels.json').write_text(json.dumps(dict(selection=worst,panels=records),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(worst=worst,panels=6),indent=2))

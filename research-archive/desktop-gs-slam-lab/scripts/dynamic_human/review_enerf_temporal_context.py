"""Larger unchanged-pixel views for fingertips outside old forearm windows."""
from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-temporal-basis-078';B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
O=R/'expanded-detail';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
rows=[]
for source,sub in [('dynamic-human-enerf-articulated-control-064','residual_graph'),('dynamic-human-enerf-heldout-cameras-068','heldout')]:
    S=L/'outputs'/source
    for row in read(S/'review-protocol.json')['regions']:
        if row['frame']!=751 or not any(k in row['name'] for k in ['forearm','shoe']): continue
        x0,y0,x1,y1=row['bbox'];box=[max(0,x0-64),max(0,y0-64),min(1920,x1+64),min(1080,y1+64)]
        rows.append(dict(**row,expanded_bbox=box,sub=sub,input_root=source))
protocol=dict(status='rule_fixed_before_candidate_panels_or_metrics_inspection',rule='All existing frame751 forearm/shoe regions, expanded64pixels on every side. Same original pixels; no rescaling/sharpening.',regions=rows,goal_complete=False)
file=O/'protocol.json'
if not file.exists():file.write_text(json.dumps(protocol,indent=2)+'\n')
else:assert read(file)==protocol
if '--prepare-only' in __import__('sys').argv:
    print('Fixed expanded-detail plan:',len(rows));raise SystemExit
for row in rows:
    c,t,sub=row['camera'],row['frame'],row['sub'];name=f'camera{c}-frame{t:06d}';x0,y0,x1,y1=row['expanded_bbox'];w,h=x1-x0,y1-y0
    with np.load(L/'outputs'/row['input_root']/'native-input'/f'{name}.npz') as z:source=z['rgb']
    images=[source,np.asarray(Image.open(B/sub/'native-evaluation'/f'{name}-trained.png')),np.asarray(Image.open(R/sub/'native-evaluation'/f'{name}-trained.png'))]
    panel=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(panel)
    for i,(label,im) in enumerate(zip(['Original photograph','Corrected RGB pose','Local temporal basis'],images)):
        panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*w,32));d.text((i*w+4,9),label,fill='black')
    out=O/f"{name}-{row['name']}.png";panel.save(out);actual=np.asarray(Image.open(out))
    for i,im in enumerate(images):assert np.array_equal(actual[32:,i*w:(i+1)*w],im[y0:y1,x0:x1])
print('Verified expanded native-pixel panels:',len(rows))

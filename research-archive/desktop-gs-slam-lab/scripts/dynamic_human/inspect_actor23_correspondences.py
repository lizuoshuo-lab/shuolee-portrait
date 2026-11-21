from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';Q=O/'geometry-time';V=Q/'review';V.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));s=read(Q/'summary.json');a={(r['camera'],r['frame']):r for r in read(O/'training-input-audit.json')['records']};rr=s['clock_fit']['validation_pairs']
selected=[min(rr,key=lambda r:r['after']['median']),max(rr,key=lambda r:r['after']['median']),min(rr,key=lambda r:r['after']['median']-r['before']['median'])];unique=[]
for r in selected:
    if r['file'] not in [x['file'] for x in unique]:unique.append(r)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);files=[];palette=['#dc2626','#0369a1','#16a34a','#a21caf','#d97706','#0891b2']
for r in unique:
    z=np.load(Q/r['file']);ids=np.linspace(0,len(z['a'])-1,min(12,len(z['a'])),dtype=int);panes=[]
    for c,key in zip(r['cameras'],['a','b']):
        q=np.load(O/a[c,612]['input']);x0,y0,x1,y1=q['bbox'];im=Image.fromarray(q['rgb'][y0:y1,x0:x1]);panel=Image.new('RGB',(im.width,im.height+35),'white');panel.paste(im,(0,35));d=ImageDraw.Draw(panel);d.text((5,5),f'Original {c}/612 | tracked descriptor pairs',font=font,fill='black')
        for j,i in enumerate(ids):
            x,y=z[key][i,1];x-=x0;y+=35-y0;color=palette[j%len(palette)];d.ellipse([x-3,y-3,x+3,y+3],outline=color,width=2);d.text((x+4,y-15),str(j),font=font,fill=color,stroke_width=1,stroke_fill='white')
        panes.append(panel)
    out=Image.new('RGB',(sum(x.width for x in panes),max(x.height for x in panes)),'white');x=0
    for im in panes:out.paste(im,(x,0));x+=im.width
    p=V/(Path(r['file']).stem+'.png');out.save(p);files.append(dict(file=str(p.relative_to(Q)),selected_match_indices=ids.tolist(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pair=r))
(V/'inventory.json').write_text(json.dumps(dict(files=files,selection='Validation median minimum, maximum, and greatest reduction; up to12uniformly spaced fixed correspondences per pair. Original frame612pixels with markers; these are candidate observations, not RGB reconstructions or semantic groundtruth.'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(files=[r['file'] for r in files],background_pair_count=len(s['background_pairs']),background_median_range=[min(r['sampson_pixel_error']['median'] for r in s['background_pairs']),max(r['sampson_pixel_error']['median'] for r in s['background_pairs'])],validation_pair_improved=sum(r['after']['median']<r['before']['median'] for r in rr),validation_pair_count=len(rr))))

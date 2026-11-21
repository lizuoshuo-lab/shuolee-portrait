from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';Q=O/'evaluation-time';V=Q/'review';V.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));s=read(Q/'summary.json');a={(r['camera'],r['frame']):r for r in read(Q/'calibration-input-audit.json')['records']};rr=s['clock_fit']['validation_pairs']
selected=[max([r for r in rr if c in r['cameras']],key=lambda r:r['after']['median']) for c in ['00','06','12']];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);palette=['#dc2626','#0369a1','#16a34a','#a21caf','#d97706','#0891b2'];files=[]
for r in selected:
    z=np.load(Q/r['file']);ids=np.linspace(0,len(z['a'])-1,min(12,len(z['a'])),dtype=int);panes=[]
    for c,key in zip(r['cameras'],['a','b']):
        q=np.load(O/a[c,588]['input']);yy,xx=np.where((q['matte']>.05)&q['valid']);x0=max(0,xx.min()-24);y0=max(0,yy.min()-24);x1=min(1920,xx.max()+25);y1=min(1080,yy.max()+25);im=Image.fromarray(q['rgb'][y0:y1,x0:x1]);panel=Image.new('RGB',(im.width,im.height+35),'white');panel.paste(im,(0,35));d=ImageDraw.Draw(panel);d.text((5,5),f'Original {c}/588 | calibration only',font=font,fill='black')
        for j,i in enumerate(ids):
            x,y=z[key][i,1];x-=x0;y+=35-y0;color=palette[j%len(palette)];d.ellipse([x-3,y-3,x+3,y+3],outline=color,width=2);d.text((x+4,y-15),str(j),font=font,fill=color,stroke_width=1,stroke_fill='white')
        panes.append(panel)
    out=Image.new('RGB',(sum(im.width for im in panes),max(im.height for im in panes)),'white');x=0
    for im in panes:out.paste(im,(x,0));x+=im.width
    p=V/(Path(r['file']).stem+'.png');out.save(p);files.append(dict(file=p.relative_to(Q).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),selected_match_indices=ids.tolist(),pair=r))
(V/'inventory.json').write_text(json.dumps(dict(files=files,selection='For each other camera00/06/12, select validation pair with largest corrected median residual, then12uniform correspondences; before inspecting any model image.',scope='Observed calibration RGB plus small diagnostic point marks; not theoretical illustrations or reconstruction groundtruth.'),indent=2)+'\n',encoding='utf8');print(json.dumps([r['file'] for r in files]))

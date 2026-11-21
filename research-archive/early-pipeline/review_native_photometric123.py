from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from PIL import Image
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122');D=R/'video10-correspondence-contrast/head-roi-contrast';O=D/'native-photometric-contrast';V=O/'visual-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
p=read(D/'plan.json');result=read(O/'result.json');assert not V.exists();V.mkdir()
images={int(f):Image.open(c['source']).convert('RGB') for f,c in p['crops'].items()}
def crop(f,uv,size=192):
    if uv is None or any(v is None for v in uv):return Image.new('RGB',(size,size),(128,128,128)),None
    x,y=np.floor(np.array(uv)+.5).astype(int)-size//2;box=[int(x),int(y),int(x+size),int(y+size)];return images[f].crop(box),box
sheets=[];records=[]
for row in result['rows']:
    i=row['row'];a,b,c,_=row['cycle'];old=row['baseline'];new=row['refined']
    pieces=[(a,row['uv']),(b,old['b_uv']),(b,new['b_uv']),(c,old['c_uv']),(c,new['c_uv'])]
    if i%4==0:sheets.append(Image.new('RGB',(960,768),(235,235,235)))
    boxes=[]
    for col,(f,uv) in enumerate(pieces):
        im,box=crop(f,uv);sheets[-1].paste(im,(col*192,(i%4)*192));boxes.append(dict(frame=f,uv=uv,box=box))
    records.append(dict(row=i,frame=a,family=row['family'],split=row['split'],file=f'comparison_{i//4+1:02d}.png',sheet_row=i%4,crops=boxes,old_pass=old['eligible'],new_pass=new['eligible']))
    if i==19:
        fig=Image.new('RGB',(1152,384))
        for col,(f,uv) in enumerate(pieces[:3]):fig.paste(crop(f,uv,384)[0],(col*384,0))
        fig.save(V/'row19_source_baseline_refined.png')
for i,im in enumerate(sheets):im.save(V/f'comparison_{i+1:02d}.png')
(V/'index.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),rows=records,columns=['source','baselineB','photometricB','baselineC','photometricC'],native_crop=192,no_resize=True,no_text_drawn=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sheets=len(sheets),rows=len(records),extra='row19_source_baseline_refined.png')))

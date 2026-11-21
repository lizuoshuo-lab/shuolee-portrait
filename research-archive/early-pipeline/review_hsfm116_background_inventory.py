from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
J=H/'joint-diagnostic-reserved';rr={r['raw_frame']:r for r in json.loads((H/'input-pilot/inputs.json').read_text())['records']}
z=np.load(J/'background-tracks.npz');images={f:Image.open(rr[f]['image']).convert('RGB') for f in rr}
for start in range(0,len(z['frames']),24):
    out=Image.new('RGB',(900,960),'white');d=ImageDraw.Draw(out)
    for q in range(start,min(start+24,len(z['frames']))):
        col=(q-start)%3;row=(q-start)//3;x0=col*300;y0=row*120
        d.text((x0+4,y0+2),f"{q} | raw {list(z['frames'][q])}",fill='black')
        for i,(f,p) in enumerate(zip(z['frames'][q],z['xy'][q])):
            box=np.rint(np.r_[p-110,p+110]).astype(int);crop=images[f].crop(tuple(box)).resize((96,96))
            out.paste(crop,(x0+i*100,y0+20));u=(p-box[:2])*96/220+[x0+i*100,y0+20]
            d.ellipse((u[0]-3,u[1]-3,u[0]+3,u[1]+3),outline='cyan',width=1)
    out.save(J/f'background-inventory-{start//24+1}.jpg',quality=96)
print('Six complete background audit sheets saved')

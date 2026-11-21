from pathlib import Path
import json
import numpy as np
from PIL import Image
from datetime import datetime,timezone
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge')
I=O.parent/'video9-10-native-intake/video10'
rows=json.loads((O/'run/visual-index.json').read_text(encoding='utf-8'))['rows']
selected=[0,11,12,17,23,34,45,51]
out=O/'native-context';out.mkdir()
records=[]
for n in selected:
 r=rows[n];obs=r['observations'];im=Image.new('RGB',(1152,384*((len(obs)+2)//3)),(80,80,80))
 for j,ob in enumerate(obs):
  f=ob['frame'];p=O/'native'/f'native_frame_{f:06d}.png'
  if not p.exists():p=I/p.name
  with Image.open(p) as src:
   x,y=np.rint(ob['uv']).astype(int);im.paste(src.convert('RGB').crop((x-192,y-192,x+192,y+192)),(384*(j%3),384*(j//3)))
 name=f'row_{n:03d}_family_{r["family"]}.png';im.save(out/name)
 records.append(dict(row=n,family=r['family'],frames=r['cell_frames'],file=name))
(out/'index.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),rows=records,cell_size=384,columns=3,native_pixels=True,burned_text=False,budget_used=8,budget_limit=16),indent=2)+'\n',encoding='utf-8')
print(json.dumps(records))

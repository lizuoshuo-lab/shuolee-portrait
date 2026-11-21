"""Read-only native contexts for the preregistered ambiguous-row review budget."""
from pathlib import Path
from PIL import Image
import json, hashlib

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'video10-static-material'; I=R/'video9-10-native-intake/video10'
idx=json.loads((O/'run/visual-index.json').read_text())
selected=[0,1,2,4,5,6,8,9,10,12,15,16,20,21,28,35]
assert len(selected)==16
out=O/'native-context';out.mkdir(exist_ok=False)
manifest=[]
for row in selected:
    t=idx['rows'][row];obs=t['observations'];page=Image.new('RGB',(384*len(obs),384),(80,80,80))
    for j,ob in enumerate(obs):
        x,y=map(round,ob['uv']);src=I/f'native_frame_{ob["frame"]:06d}.png'
        with Image.open(src) as im:page.paste(im.crop((x-192,y-192,x+192,y+192)),(384*j,0))
    name=f'row_{row:03d}_family_{t["family"]}.png';page.save(out/name)
    manifest.append(dict(row=row,family=t['family'],split=t['split'],frames=[x['frame'] for x in obs],file=name,center_pixel=[192,192],sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
(out/'index.json').write_text(json.dumps(dict(rows=manifest,cell_size=384,resized=False,burned_text=False,black_area='PIL outside-image padding',reviewed=False),indent=2)+'\n')
print(json.dumps(manifest))

"""Native context only; no annotations, resampling, matching or camera fitting."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
A=I/'clip03-structure-allocation'; M=I/'clip03-scene-support'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (A/'context-index.json').exists()
v=read(A/'visual-index.json'); frames=v['columns']
rows=[0,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19]
assert len(rows)==16
images={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in frames}
out=[]
for n in rows:
    r=v['rows'][n]; dst=Image.new('RGB',(256*7,256),(80,80,80)); old=Image.open(A/r['file'])
    for o in r['observations']:
        x,y=np.rint(o['uv']).astype(int); c=frames.index(o['frame'])
        crop=images[o['frame']].crop((x-128,y-128,x+128,y+128))
        assert np.array_equal(np.array(crop.crop((64,64,192,192))),np.array(old.crop((c*128,0,(c+1)*128,128))))
        dst.paste(crop,(c*256,0))
    name=f'context_{n:03d}_family_{r["family"]}.png';dst.save(A/name)
    out.append(dict(row=n,family=r['family'],file=name,sha256=sha(A/name)))
data=dict(rows=out,columns=frames,cell_size=256,annotations_in_images=False,source_center_pixels_equal=True,
          frozen_hashes={n:sha(A/n) for n in ['result.json','allocation.json','visual-index.json']})
(A/'context-index.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print('N12C_CONTEXT_READY',len(out))

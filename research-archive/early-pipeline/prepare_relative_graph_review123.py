from pathlib import Path
from PIL import Image
import numpy as np,json,hashlib
import match_video10_temporal_bridge123 as b
G=b.R/'video10-global-initializer';D=G/'visual-review';U=b.R/'video10-remaining-material-review/global-input-preflight'
read=b.read;save=b.save;sha=b.sha
D.mkdir(exist_ok=True);assert not (D/'index.json').exists()
root=read(G/'minimal-root-diagnosis/result.json');ids=root['fit_families']+root['roots'][0]['heldout_families']
tracks={t['family']:t for name in ['fit-observations.json','heldout-observations.json'] for t in read(U/name)['tracks']}
rows=[]
for fid in ids:
    tr=tracks[fid];obs=[o for o in tr['observations'] if o['frame'] in [698,752]];assert len(obs)==2
    strip=Image.new('RGB',(768,384),(235,235,235));sources=[]
    for j,o in enumerate(obs):
        path=b.imgpath(o['frame']);im=Image.open(path).convert('RGB');assert im.size==(2704,1520)
        x,y=np.rint(o['uv']).astype(int);strip.paste(im.crop((x-192,y-192,x+192,y+192)),(j*384,0))
        sources.append(dict(frame=o['frame'],uv=o['uv'],rounded_center=[int(x),int(y)],source=str(path),source_sha256=sha(path)))
    name=f"{tr['split']}_family_{fid}_frames_698_752.png";strip.save(D/name)
    rows.append(dict(family=fid,split=tr['split'],file=name,sha256=sha(D/name),sources=sources))
for page in range(3):
    group=rows[3*page:3*page+3];im=Image.new('RGB',(768,384*len(group)),(235,235,235))
    for j,r in enumerate(group):im.paste(Image.open(D/r['file']),(0,384*j))
    im.save(D/f'page_{page+1}.png')
save(D/'index.json',dict(updated=b.now(),selection='Post-hoc diagnostic of a five-point ambiguous root. All fiveFIT and bothHELD shown; no point removed or reclassified.',native_crop=384,no_scaling=True,burned_text=False,rows=rows))
print(json.dumps(dict(pages=[str(D/f'page_{n}.png') for n in range(1,4)],families=ids)))

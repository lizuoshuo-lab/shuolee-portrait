from pathlib import Path
import numpy as np
from PIL import Image
import match_video10_temporal_bridge123 as b
O=b.R/'video10-global-initializer/joint-track-candidate/convergence-repair';D=O/'visual-review'
assert not (D/'supplementary-context.json').exists()
rows=b.read(D/'index.json')['rows'];r=next(x for x in rows if x['family']==834)
canvas=Image.new('RGB',(192*3,192),(235,235,235));meta=[]
for i,s in enumerate(r['sources']):
    im=Image.open(s['source']).convert('RGB');x,y=np.rint(s['uv']).astype(int);box=(x-96,y-96,x+96,y+96)
    canvas.paste(im.crop(box),(i*192,0));meta.append(dict(frame=s['frame'],uv=s['uv'],crop=list(map(int,box))))
canvas.save(D/'heldout_834_exact192.png')
b.save(D/'supplementary-context.json',dict(updated=b.now(),family=834,reason='Inspect center identity on the already selected native crop; no source/observation change.',no_resize=True,no_text=True,views=meta))

"""Freeze up to16 context requests after all original pages were viewed."""
from pathlib import Path
from PIL import Image
import numpy as np,json
import match_video10_temporal_bridge123 as b
from review_video10_remaining123 import notes,U
read=b.read;save=b.save;sha=b.sha
idx=read(U/'review/index.json');rows=idx['rows'];p=read(U/'plan.json')
assert set(notes)==set(range(222))
selected=sorted(i for i,(status,note) in notes.items() if status=='q')
assert len(selected)==16==p['budget']['expanded_native_context_rows_max']
out=U/'review/contexts';assert not out.exists();out.mkdir()
save(out/'allocation.json',dict(registered=b.now(),rows=selected,reason='Observed native physical-identity ambiguity; no camera or geometry residual exists or was used.',initial_notes_sha256=sha(Path(__file__).with_name('review_video10_remaining123.py')),additional_matching=0))
ims={f:Image.open(b.imgpath(f)).convert('RGB') for f in sorted({o['frame'] for i in selected for o in rows[i]['observations']})}
result=[]
for i in selected:
    r=rows[i];strip=Image.new('RGB',(384*len(r['observations']),384),(235,235,235))
    for j,o in enumerate(r['observations']):
        x,y=np.rint(o['uv']).astype(int);strip.paste(ims[o['frame']].crop((x-192,y-192,x+192,y+192)),(384*j,0))
    name=f'row_{i:03d}.png';strip.save(out/name);result.append(dict(row=i,family=r['family'],split=r['split'],frames=r['frames'],file=name,sha256=sha(out/name)))
save(out/'index.json',dict(updated=b.now(),native_crop_px=384,center=[192,192],no_scaling_or_warp=True,burned_text=False,rows=result))
print(json.dumps(dict(rows=selected,context_review_pending=True)))

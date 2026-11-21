"""Persist visual N9 evidence; never modify observations or camera parameters."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
from PIL import Image, ImageDraw
from audit_offplane_support122 import B, O, read, save, sha, now

def prepare():
    p=O/'track399-context-plan.json'
    assert not p.exists()
    rows=[r for page in read(O/'patch-index.json') for r in page['rows']]
    t=next(r for r in rows if r['track']==399)
    save(p,dict(registered=now(),reason='Actual three-view review shows a discontinuous material context for family399; inspect larger native crops, without using reprojection residuals to choose or discard observations.',
        track=399,observations=t['observations'],crop_size=320,source_sha256=sha(Path(__file__)),
        prohibits=['No optimization','No point deletion','No changed FIT/heldout identity']))
    canvas=Image.new('RGB',(960,320))
    for i,ob in enumerate(t['observations']):
        im=Image.open(B/f'frame_{ob["frame"]:06d}.png').convert('RGB')
        u,v=ob['uv'];x,y=round(u)-160,round(v)-160
        crop=im.crop((x,y,x+320,y+320));d=ImageDraw.Draw(crop)
        d.ellipse((u-x-4,v-y-4,u-x+4,v-y+4),outline=(0,255,80),width=1)
        canvas.paste(crop,(320*i,0))
    canvas.save(O/'track399_native_context.png')
    save(O/'track399-context-index.json',t)
    print('N9_TRACK399_CONTEXT_READY')

if __name__=='__main__':prepare()

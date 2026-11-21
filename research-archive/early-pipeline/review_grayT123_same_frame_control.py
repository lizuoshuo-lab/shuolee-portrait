"""Native all-node crops for the already-selected asphalt HELD negative control."""
import numpy as np
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,native

out=REC/'grayT-skin-texture-probe'
target=out/'asphalt-all-node-index.json'
if target.exists():raise RuntimeError('Control review prepared already.')
rows=read(ROOT/'data/training-inputs.json')['records']
tr=next(r for r in read(ROOT/'camera/held-evaluation.json')['records'] if r['id'].startswith('8036d2b7f840'))
canvas=Image.new('RGB',(128*len(tr['nodes']),128),(245,245,245));rawcanvas=canvas.copy();entries=[]
for n,(i,j) in enumerate(tr['nodes']):
    xy=np.load(ROOT/f'camera/features-{i:03d}.npz')['xy'][j];cx,cy=map(int,np.round(xy));crop=[cx-64,cy-64,cx+64,cy+64]
    with Image.open(native(rows[i]['image'])) as im:tile=im.convert('RGB').crop(crop)
    rawcanvas.paste(tile,(128*n,0));p=xy-crop[:2];d=ImageDraw.Draw(tile);d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),outline=(0,255,255),width=1);canvas.paste(tile,(128*n,0))
    entries.append(dict(frame=rows[i]['frame'],role='ANCHOR' if n%2==0 else 'HELD',xy=xy.tolist(),crop=crop))
canvas.save(out/'asphalt_all_nodes_locator.png');rawcanvas.save(out/'asphalt_all_nodes_native.png')
save(target,dict(track_id=tr['id'],observations=entries,point_identity_not_assumed_from_numeric_error=True))
print(entries)

"""Document a concrete raw-photo vs released-matte discrepancy before correction."""
from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw,ImageFont
LAB=Path(__file__).resolve().parents[2]
SOURCE=LAB/'outputs/dynamic-human-4k4d-official-035/native-evaluation'
OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038';OUT.mkdir(exist_ok=True)
prefix='camera01-frame000750';z=np.load(SOURCE/f'{prefix}-raw.npz')
rgb=z['source'];matte=z['matte'];ref=np.array(Image.open(SOURCE/f'{prefix}-reference.png'))
assert np.array_equal(ref,np.rint((rgb.astype(np.float32)/255*matte[...,None]+.5*(1-matte[...,None]))*255).clip(0,255).astype(np.uint8))
box=[736,430,824,610];x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
panes=[('Photo',rgb),('Matte',np.repeat(np.rint(matte*255).astype(np.uint8)[...,None],3,axis=2)),('Composite',ref)]
canvas=Image.new('RGB',(3*w,h+30),'white');draw=ImageDraw.Draw(canvas)
for j,(title,arr) in enumerate(panes):
    canvas.paste(Image.fromarray(arr[y0:y1,x0:x1]),(j*w,30));draw.text((j*w+4,8),title,fill='black')
canvas.save(OUT/'camera01-forearm-source-matte.png')
points=[(775,495),(771,515),(767,535),(767,555),(774,573)]
rows=[]
for x,y in points:
    rows.append(dict(x=x,y=y,rgb=rgb[y,x].tolist(),released_matte=float(matte[y,x]),patch3x3_matte_mean=float(matte[y-1:y+2,x-1:x+2].mean())))
report=dict(status='native_source_and_matte_comparison_written_manual_point_review_pending',camera='01',frame=750,crop=box,points=rows,
    source_raw_sha256=hashlib.sha256((SOURCE/f'{prefix}-raw.npz').read_bytes()).hexdigest(),
    scope='Training-view hand/forearm diagnostic selected after inspecting rawphoto. Points are tentative manual interior samples, not a full semantic ground-truth annotation.',
    initialization='035visual hull requires all15trainingcamera projections to have releasedmatte>0.5 when all15are in view. A false-negative camera can remove corresponding candidategeometry; exact affected3Dvolume not yet measured.',
    modification='No source pixels or masks changed; panel contains only native crops and labels.',goal_complete=False)
(OUT/'missing-forearm-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

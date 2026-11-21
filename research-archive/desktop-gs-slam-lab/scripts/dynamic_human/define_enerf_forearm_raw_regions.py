"""Define source-only visible-skin interiors, never using model predictions."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';O=R/'raw-forearm-evaluation';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
assert not (O/'summary.json').exists()
polygons={749:[[790,480],[796,482],[787,506],[778,527],[774,526],[778,502]],751:[[777,480],[784,481],[773,508],[761,531],[757,529],[767,506]],753:[[764,476],[771,478],[759,503],[747,529],[742,527],[753,503]],755:[[750,477],[756,480],[745,506],[734,533],[729,531],[737,507]]}
rows=[];panel=Image.new('RGB',(4*210,252),'white');draw=ImageDraw.Draw(panel)
for i,(t,poly) in enumerate(polygons.items()):
 file=I/'native-input'/f'camera01-frame{t:06d}.npz';z=np.load(file);m=Image.new('L',(1920,1080),0);ImageDraw.Draw(m).polygon([tuple(p) for p in poly],fill=255);mask=np.array(m)>0;assert np.all(z['valid'][mask])
 im=Image.fromarray(z['rgb'][420:640,660:870].copy());ImageDraw.Draw(im).polygon([(x-660,y-420) for x,y in poly],outline=(0,220,255),width=1);panel.paste(im,(i*210,32));draw.text((i*210+4,10),f'Original photograph frame{t}',fill='black')
 rows.append(dict(camera='01',frame=t,polygon_xy=poly,pixels=int(mask.sum()),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
panel.save(O/'source-only-regions-review.png')
report=dict(status='source_only_regions_defined_before_model_evaluation_visual_check_pending',defined_utc=datetime.now(timezone.utc).isoformat(),records=rows,metric='Native predicted RGB composited onto0.5gray versus original normalized photograph; average RGB MSE, PSNR, MAE and opacity inside manually selected visible-skin interiors. Do not divide by predicted alpha.',scope='Four odd-frame visible forearm interiors from known training camera01. Development diagnostic, not broad novel-view or finger accuracy. Selection reads only original RGB. No polygons used in training.',goal_complete=False)
(O/'protocol.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('Saved source-only visible-forearm polygons; review pending.')

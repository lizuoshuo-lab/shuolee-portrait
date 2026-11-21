from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
T=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';I=R/'video-update-20261027';B=I/'background-geometry'
O=B/'material-region-comparison';Z=B/'residual-structure';M=I/'native-temporal-material'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not (O/'anchor-semantic-review-plan.json').exists()
save(O/'anchor-semantic-review-plan.json',dict(updated=stamp,reason='The6 preregistered footwear patches include skirt hem, boot shaft and ground. Region name is not a physical part annotation.',
    action='Inspect all20 original anchor seed centers at native96x96. No retracking or residual filtering. Preserve existing foot candidate label as historical origin, add semantic audit separately.',
    tracks=list(range(287,307)),training=False))
im=Image.open(B/'frame_000278.png').convert('RGB');seeds=read(M/'seeds.json')['seeds']
sheet=Image.new('RGB',(5*96,4*192),'white');index=[]
for i,s in enumerate(s for s in seeds if s['group']=='footwear_candidate'):
    u,v=s['uv'];x,y=round(u)-48,round(v)-48;crop=im.crop((x,y,x+96,y+96));col,row=i%5,i//5
    sheet.paste(crop,(col*96,row*192));marked=crop.copy();d=ImageDraw.Draw(marked);d.ellipse((u-x-2,v-y-2,u-x+2,v-y+2),outline=(0,255,80));sheet.paste(marked,(col*96,row*192+96))
    index.append(dict(track=s['id'],row=row,column=col,uv=s['uv'],crop=[x,y,x+96,y+96]))
sheet.save(O/'foot_candidate_anchor_all20.png');save(O/'foot_candidate_anchor_all20-index.json',index)
save(Z/'visual-review.json',dict(updated=stamp,state='reviewed_support_limitation',
    reviewed_all7_nearest_background_sheets=True,frames=[230,246,262,278,294,310,326],
    native_crop=True,training_image_warp=False,
    findings=['Nearest heldout background points are337to720 nativepx fromfacecenter, not adjacent negativecontrols.',
        'Several nearest patches show water/rippled or reflective appearance and shoreline edges, so static material identity is not certified. No observations removed.',
        'All935FIT conditional3Dpoints have a dominant plane with medianabsdistance0.109 baselineunits; face points median5.250 fromsameplane. The cloud also has off-plane points, so notallbackgroundisplanar.',
        '96.93percent offace squared residual is its per-view commonvector. This isdescription, notcorrection orproof ofmotion.'],
    all_physical_identities_certified=False,unique_cause_proven=False,training_admitted=False,
    result_sha256=sha(Z/'result.json')))
print('Anchor review sheet and N7 visual review saved')

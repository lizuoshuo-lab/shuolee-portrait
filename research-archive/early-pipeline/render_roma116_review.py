"""Original-pixel evidence for the preregistered fixed sample, never warp a portrait."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
B=TOP/'shuolee';O=B/'current/portrait-alignment-open-source-116';R=O/'roma';S=B/'current/full-scene-localization-074';D=R/'smoke'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
assert read(D/'result.json')['state']=='completed_waiting_original_image_review'
reg=read(R/'smoke-preregistration.json');all_rows=[];figures=[]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
labels=['background','lower face','upper face / glasses','torso','legs / shoes']
crop_size=192;block_width=600;row_height=292
for ti,triple in enumerate(reg['triples_raw_frame_ids']):
    with np.load(D/f'triple-{ti:02d}.npz') as z:data={k:z[k] for k in z.files}
    assert list(data['raw_frames'])==triple
    originals=[Image.open(S/'images'/f'raw_{f:06d}.png').convert('RGB') for f in triple]
    board=Image.new('RGB',(1800,90+row_height*5),'#ffffff');d=ImageDraw.Draw(board)
    d.text((12,8),f'RoMa fixed sample {ti+1}/6 | Raw frames {triple[0]} / {triple[1]} / {triple[2]} | Same original pixel scale',font=title,fill='#152a38')
    d.text((12,38),'Green: inferred image correspondence. Orange:074 projection in held B, fit from A/C only. No image warping.',font=font,fill='#333333')
    d.text((12,59),'Each source crop centers on its own observed texture for identity review; this display does not establish 3D alignment. All registered failures retained.',font=font,fill='#555555')
    for group,label in enumerate(labels):
        eligible_ids=np.flatnonzero(data['groups']==group)
        assert len(eligible_ids)>=3
        chosen=eligible_ids[[0,len(eligible_ids)//2,len(eligible_ids)-1]]
        for col,idx in enumerate(chosen):
            x0=col*block_width;y0=90+group*row_height;pid=int(data['probe_ids'][idx]);good=bool(data['eligible'][idx]);point=data['points'][idx];proj=data['projection'][idx]
            held=float(np.linalg.norm(proj[1]-point[1]));conf=float(data['confidence'][idx]);fb=float(data['fb'][idx].max());cycle=float(data['cycle'][idx])
            d.text((x0+8,y0),f'{label} | probe {pid} | numeric {"PASS" if good else "FAIL"}',font=font,fill='#11663d' if good else '#a22424')
            d.text((x0+8,y0+20),f'conf {conf:.3f} | FB {fb:.2f}px | cycle {cycle:.2f}px | held B {held:.2f}px',font=small,fill='#333333')
            crop_origins=[]
            for j,(im,xy) in enumerate(zip(originals,point)):
                cx=x0+8+j*196;cy=y0+55
                if not np.isfinite(xy).all():
                    d.text((cx,cy),'NONFINITE MATCH',font=font,fill='#a22424');crop_origins.append(None);continue
                origin=np.floor(xy-crop_size/2).astype(int);crop_origins.append(origin.tolist())
                crop=im.crop((*origin,*(origin+crop_size)));board.paste(crop,(cx,cy))
                px,py=xy-origin+[cx,cy]
                d.ellipse((px-5,py-5,px+5,py+5),outline='#00f070',width=2)
                if j==1 and np.isfinite(proj[j]).all():
                    u,v=proj[j]-origin+[cx,cy]
                    if cx<=u<cx+crop_size and cy<=v<cy+crop_size:
                        d.line((u-5,v,u+5,v),fill='#ff8c00',width=2);d.line((u,v-5,u,v+5),fill='#ff8c00',width=2)
                d.text((cx,y0+39),f'{"ABC"[j]} raw{triple[j]:06d}',font=small,fill='#152a38')
                d.text((cx,y0+250),f'x={xy[0]:.1f}, y={xy[1]:.1f}',font=small,fill='#333333')
            d.line((x0+3,y0+282,x0+block_width-3,y0+282),fill='#cccccc')
            all_rows.append(dict(triple_index=ti,probe_id=pid,group=group,label=label,selected_rule='first/middle/last in fixed hash rank',points=point.tolist(),crop_origins=crop_origins,
                numeric_pass=good,confidence=conf,max_fb_raw_px=fb,cycle_raw_px=cycle,held_projection_raw_px=held,visual_material_identity='pending',motion='not_determined'))
    name=f'fixed-original-review-{ti+1}.jpg';board.save(D/name,quality=96);figures.append(name)
    for im in originals:im.close()
manifest=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),selection_rule=reg['visual_review'],tracks=len(all_rows),figures=figures,
    crop_policy='192x192 original pixels, no stretching or warping; each texture crop follows its observed source coordinate, all coordinates and origins recorded. Held prediction shown in the unchanged B crop.',
    rows=all_rows,full_dataset_validation=False,quality_accepted=False,root_cause_resolved=False)
(D/'visual-sample-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(figures=figures,tracks=len(all_rows),visual_review='pending')))

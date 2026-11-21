from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
from PIL import Image
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=REC/'pattern-preflight/coverage';V=O/'visual-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
result=read(O/'result.json');obs=read(O/'observations.json');z=np.load(O/'observations.npz')
assert result['frames']==99 and len(obs['rows'])==99 and not V.exists();V.mkdir()
records=[];sheets={};selected=[0,24,49,74,98]
for row in obs['rows']:
    slot=row['slot'];im=Image.open(row['image']).convert('RGB');assert im.size==(2704,1520)
    sheet=slot//20;position=slot%20
    if sheet not in sheets:sheets[sheet]=dict(context=Image.new('RGB',(2028,1425),(235,235,235)),face=Image.new('RGB',(1536,1920),(235,235,235)))
    # Exact uniform3/16 overview only. Native face crops are never rescaled.
    sheets[sheet]['context'].paste(im.resize((507,285),Image.Resampling.LANCZOS),((position%4)*507,(position//4)*285))
    face=row.get('face_box');box=None
    if face is not None:
        center=(np.array(face[:2])+np.array(face[2:]))*.5
        x,y=np.floor(center+.5).astype(int)-192;box=[int(x),int(y),int(x+384),int(y+384)]
        faceim=im.crop(box)
    else:faceim=Image.new('RGB',(384,384),(128,128,128))
    sheets[sheet]['face'].paste(faceim,((position%4)*384,(position//4)*384))
    record=dict(slot=slot,source_frame=row['source_frame_index'],context_sheet=f'context_{sheet+1:02d}.png',face_sheet=f'face_{sheet+1:02d}.png',row=position//4,column=position%4,face_crop_xyxy=box,detector_face_ROI_not_physical=True)
    if slot in selected and row['chosen'] is not None:
        foot=z['keypoints'][slot,17:23];cx,cy=np.floor(np.mean(foot,axis=0)+.5).astype(int)
        b=[int(cx-256),int(cy-256),int(cx+256),int(cy+256)]
        im.crop(b).save(V/f'slot{slot:03d}_predicted_foot_native.png')
        record['foot_crop']=b;record['foot_uv']=foot.tolist();record['foot_scores']=z['scores'][slot,17:23].tolist()
    records.append(record)
for n,group in sheets.items():
    for kind,im in group.items():im.save(V/f'{kind}_{n+1:02d}.png')
(V/'index.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),records=records,frames=99,context_scale=3/16,context_exact_uniform_scale=True,face_crop_native_size=384,face_resize=False,image_text_drawn=False,foot_selection=selected,foot_black_padding_is_outside_source=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sheets=10,frames=99,extra_foot_crops=5)))

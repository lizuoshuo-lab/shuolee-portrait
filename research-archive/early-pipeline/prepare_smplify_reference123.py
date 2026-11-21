from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageDraw
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
R=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
plan=json.loads((O/'plan.json').read_text(encoding='utf-8-sig'))
rows=json.loads((R/'data/inputs.json').read_text())['records']
obs=np.load(R/'data/observations.npz')
out=[]
for slot in plan['views']:
    row=rows[slot]; im=Image.open(row['image']).convert('RGB')
    xy=obs['keypoints'][slot]; conf=obs['scores'][slot]
    boxes={'body':row['initial_box_xyxy']}
    for name,ids,pad in [('face',np.arange(23,91),25),('left_hand',np.arange(91,112),25),('right_hand',np.arange(112,133),25)]:
        pts=xy[ids][conf[ids]>.25]
        if not len(pts): pts=xy[ids]
        boxes[name]=[*np.min(pts,axis=0)-pad,*np.max(pts,axis=0)+pad]
    rec={'slot':slot,'source_frame':row['source_frame_index'],'regions':{}}
    for name,b in boxes.items():
        b=np.round(b).astype(int);b[[0,2]]=np.clip(b[[0,2]],0,im.width);b[[1,3]]=np.clip(b[[1,3]],0,im.height)
        raw=im.crop(tuple(b)); overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        ids=np.arange(133) if name=='body' else np.arange(23,91) if name=='face' else np.arange(91,112) if name=='left_hand' else np.arange(112,133)
        for i in ids:
            p=xy[i]-b[:2]
            if conf[i]>.25: draw.ellipse((p[0]-1,p[1]-1,p[0]+1,p[1]+1),fill=(0,255,80))
        scale=min(1,600/raw.height) if name=='body' else 4
        wh=(round(raw.width*scale),round(raw.height*scale))
        panels=[x.resize(wh,Image.Resampling.NEAREST if scale>1 else Image.Resampling.LANCZOS) for x in [raw,overlay]]
        sheet=Image.new('RGB',(wh[0]*2,wh[1]),(230,230,230));sheet.paste(panels[0],(0,0));sheet.paste(panels[1],(wh[0],0))
        path=O/f'input_slot{slot:03d}_{name}_raw_observations.png';sheet.save(path)
        rec['regions'][name]={'crop_integer':b.tolist(),'uniform_scale':scale,'file':str(path),'native_crop_wh':list(raw.size)}
    out.append(rec)
(O/'input-crops.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

"""Unwarped, identical-image-window evidence for the finished fixed-K run."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
from PIL import Image

ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
OUT=REC/'fixedk-gaussian-review'
LOG=Path('/mnt/d/00/工作日志/2026-10-27/新人物118/过程图片')
OUT.mkdir(exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records']
obs=dict(np.load(ROOT/'data/observations.npz'))
sel=[0,17,35,52,70,88]
layout=[]
for kind in ['body','face','feet']:
    sheet=Image.new('RGB',(800,360*len(sel)),(48,48,48))
    for j,i in enumerate(sel):
        row=rows[i];w,h=row['wh']
        if kind=='body':
            xy=obs['boxes'][i].copy();xy[:2]-=30;xy[2:]+=30
        else:
            pts=obs['keypoints'][i,40:91] if kind=='face' else obs['keypoints'][i,17:23]
            margin=35 if kind=='face' else 65
            xy=np.r_[pts.min(0)-margin,pts.max(0)+margin]
        x0,y0,x1,y1=map(int,[max(0,xy[0]),max(0,xy[1]),min(w,xy[2]),min(h,xy[3])])
        assert x1>x0 and y1>y0
        for col,suffix in enumerate(['input','viewer']):
            src=ROOT/'export/review'/f"{row['frame']}_{suffix}.png"
            with Image.open(src) as im:
                sx,sy=im.width/w,im.height/h
                crop=im.convert('RGB').crop((round(x0*sx),round(y0*sy),round(x1*sx),round(y1*sy)))
                display_scale=min(400/crop.width,360/crop.height)
                crop=crop.resize((max(1,round(crop.width*display_scale)),max(1,round(crop.height*display_scale))),Image.Resampling.LANCZOS)
                sheet.paste(crop,(400*col+(400-crop.width)//2,360*j+(360-crop.height)//2))
            if kind=='body':
                shutil.copy2(src,LOG/f'{row["frame"]}_{"原图" if col==0 else "固定新内参_18000步渲染"}.png')
        layout.append(dict(kind=kind,frame=row['frame'],source_frame=row['source_frame_index'],
                           crop_xyxy=[x0,y0,x1,y1],columns=['original','fixedK18000'],
                           identical_pixel_crop=True,aspect_ratio_preserved=True,text_burned_in=False))
    sheet.save(OUT/f'{kind}_original_fixedK18000.jpg',quality=96)
    shutil.copy2(OUT/f'{kind}_original_fixedK18000.jpg',LOG/f'{kind}_左原图_右18000步_同窗口对比.jpg')
for batch in range(3):
    sheet=Image.new('RGB',(256*6,384*2),(48,48,48))
    for j in range(12):
        i=batch*12+j
        with Image.open(ROOT/'export/review'/f'novel_{i:03d}.png') as im:
            im=im.convert('RGB');im.thumbnail((256,384),Image.Resampling.LANCZOS)
            sheet.paste(im,(256*(j%6)+(256-im.width)//2,384*(j//6)+(384-im.height)//2))
    sheet.save(OUT/f'novel_{batch*12:03d}_{batch*12+11:03d}.jpg',quality=95)
    shutil.copy2(OUT/f'novel_{batch*12:03d}_{batch*12+11:03d}.jpg',LOG/f'未知视角_{batch*12:03d}_{batch*12+11:03d}_每格10度.jpg')
validation=json.loads((ROOT/'export/validation.json').read_text())
result=dict(source_sha256=validation['sha256'],same_camera_layout=layout,
            novel_views=dict(count=36,step_degrees=10,ground_truth=False),
            actual_visual_review_pending=True,quality_accepted=False)
(OUT/'layout.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(out=str(OUT),images=6,sha256=validation['sha256'])))

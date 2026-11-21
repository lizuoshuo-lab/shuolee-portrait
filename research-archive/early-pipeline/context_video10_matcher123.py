"""One bounded context expansion after inspecting all72 initial rows."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
Q=T/'shuolee/current/robust-portrait-122/video10-correspondence-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
D=Q/'review/contexts';D.mkdir();p=read(Q/'plan.json');rows=read(Q/'review/index.json')['rows']
chosen=[2,3,6,8,19,25,26,38,39,42,48,49,57,58,59,61]
save(D/'allocation.json',dict(updated=datetime.now(timezone.utc).isoformat(),rows=chosen,max_rows=16,
    reason='After viewing all12pages: clarify reflective black/glass object, shelf identities, headROI versus actualbackground centers, occlusion and the positivecontrol with absent directSIFT. Selection based on visible identity ambiguity; no confidence/FB/cycle threshold selection.'))
paths=[Path(s) for s in p['input_sha256']];images={f:Image.open(next(x for x in paths if x.name==f'native_frame_{f:06d}.png')).convert('RGB') for f in p['frames']}
out=[];pages=[]
for k,i in enumerate(chosen):
    r=rows[i];a=r['source'];b=r['target'];strip=Image.new('RGB',(1152,384),(226,226,226))
    for j,(f,uv) in enumerate([(a,r['source_uv']),(b,r['baseline_uv']),(b,r['roma_uv'])]):
        if uv is None or not np.isfinite(uv).all():continue
        x,y=np.rint(uv).astype(int);strip.paste(images[f].crop((x-192,y-192,x+192,y+192)),(384*j,0))
    name=f'row_{i:03d}.png';strip.save(D/name);out.append(dict(row=i,family=r['family'],frames=[a,b,b],file=name,center=[192,192]))
    if k%4==0:page=Image.new('RGB',(1152,1552),(245,245,245));ids=[]
    page.paste(strip,(0,(k%4)*388));ids.append(i)
    if k%4==3:
        name=f'page_{len(pages):02d}.png';page.save(D/name);pages.append(dict(file=name,rows=ids.copy()))
save(D/'index.json',dict(rows=out,pages=pages,native_crop_px=384,burned_text=False,warped=False))
print('Prepared16bounded native contexts in4pages.')

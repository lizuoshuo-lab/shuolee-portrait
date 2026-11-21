from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
R=B/'person-static-shared-support-fresh-102';O=B/'shared-camera-points-allview-104'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
win=lambda p:Path(p.replace('/mnt/c/','C:/').replace('/mnt/d/','D:/'))
rows=read(R/'data/training-inputs.json')['records'];a=np.load(R/'shared-targets/shared-face-points.npz')
visibility=read(O/'visibility-audit.json');bad={(r['raw'],r['feature']):r for r in visibility['outside']}
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
sheet=Image.new('RGB',(1200,820),'#fafafa');dr=ImageDraw.Draw(sheet)
for j,raw in enumerate([1,34,70,100,124,136]):
    i=raw-1;row=rows[i];im=Image.open(win(row['image'])).convert('RGB');mask=Image.open(win(row['mask'])).convert('L')
    center=np.median(a['observed_uv'][i],axis=0);origin=np.rint(center-[180,160]).astype(int);box=(*origin,*(origin+[360,340]))
    im=im.crop(box);mask=mask.crop(box).point(lambda x:255 if x>=128 else 0)
    boundary=np.array(mask)>np.array(mask.filter(ImageFilter.MinFilter(3)))
    z=np.array(im);z[boundary]=[0,230,230];im=Image.fromarray(z);d=ImageDraw.Draw(im)
    for k in np.flatnonzero(a['observation_valid'][i]&a['point_available']):
        ident=int(a['point_ids'][k]);xy=a['observed_uv'][i,k]-origin
        color='#ff3030' if (raw,ident) in bad else '#44ee55'
        d.ellipse((*list(xy-2),*list(xy+2)),outline=color,width=1)
        if ident in [27,29,30,51,56,57]:d.text(tuple(xy+[4,-17]),str(ident),fill=color,font=small)
    x=(j%3)*400;y=(j//3)*400
    sheet.paste(im,(x+20,y+45));dr.text((x+20,y+10),f'原视频第 {raw} 帧',font=font,fill='#25313a')
dr.text((20,799),'青线：原 mask 边界；红点：mask 外检测；绿点：mask 内检测。图像与坐标均未修改。',font=small,fill='#25313a')
sheet.save(O/'source-mask-feature-review.jpg',quality=97)
print(O/'source-mask-feature-review.jpg')

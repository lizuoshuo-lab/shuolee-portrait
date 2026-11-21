"""A fixed training-view preview, explicitly separate from heldout evidence."""
from pathlib import Path
import numpy as np,cv2,json
from PIL import Image,ImageDraw,ImageFont
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029';dest=out/'comparison';dest.mkdir(exist_ok=True);d=np.load(out/'same-index/train-camera-01.npz');a=np.load(out/'offset-index/train-camera-01.npz');assert np.array_equal(d['rgb'],a['rgb']) and int(d['source_frame'])==int(a['source_frame'])==34
target=np.rint(d['rgb']*d['matte'][...,None]+127.5*(1-d['matte'][...,None])).clip(0,255).astype('uint8');images=[Image.fromarray(target)]+[Image.open(out/g/'first-frame/train-01-step-06000.png').convert('RGB') for g in ['same-index','offset-index']];_,_,stats,_=cv2.connectedComponentsWithStats((d['matte']>.5).astype('uint8'));regions=sorted([s for s in stats[1:] if s[4]>10000],key=lambda s:s[0]);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);scores={}
for j,s in enumerate(regions):
    x,y,w,h,_=map(int,s);box=[x-16,y-16,x+w+16,y+h//2];pw=box[2]-box[0];ph=box[3]-box[1];can=Image.new('RGB',(3*pw,ph+42),'white');dr=ImageDraw.Draw(can)
    for k,(im,label) in enumerate(zip(images,['Training reference','Same-index fit','Offset-index fit'])):can.paste(im.crop(box),(k*pw,42));dr.text((k*pw+7,12),label,font=font,fill='black')
    can.save(dest/f'first-fit-training-camera01-person{j+1}.png')
inside=(d['matte']>.9)&d['valid']
for group,im in zip(['same-index','offset-index'],images[1:]):scores[group]=float(-10*np.log10((((np.array(im).astype(float)-target)/255)**2)[inside].mean()))
(dest/'first-fit-training-view.json').write_text(json.dumps(dict(status='training_view_only_not_heldout_quality',camera='01',source_frame=34,foreground_psnr_db=scores),indent=2)+'\n',encoding='utf8');print(json.dumps(scores))

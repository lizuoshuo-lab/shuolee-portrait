"""Inspect native training photographs, provided matte, and exact grey target."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-matte-diagnosis-080';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows={(r['camera'],r['frame']):r for r in read(R/'input-audit.json')['records']}
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
panels=[]
for c in ['04','07','13']:
    for f in [726,732]:
        src=R/'native-input'/f'camera{c}-frame{f:06d}.npz'
        assert rows[c,f]['split']=='train' and sha(src)==rows[c,f]['input_sha256']
        z=np.load(src);rgb=z['rgb'];a=z['matte'];x0,y0,x1,y1=map(int,z['bbox'])
        target=np.rint(np.clip(rgb.astype(np.float32)*a[...,None]+127.5*(1-a[...,None]),0,255)).astype(np.uint8)
        arrays=[rgb,np.repeat(np.rint(a[...,None]*255).astype(np.uint8),3,axis=2),target]
        w,h=x1-x0,y1-y0;im=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(im)
        for k,(arr,label) in enumerate(zip(arrays,['Original training photograph','Provided matte (white = foreground)','Exact target on grey background'])):
            im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(w*k,32));d.text((w*k+4,7),label,font=font,fill='black')
        path=O/f'camera{c}-frame{f:06d}-training-target.png';im.save(path)
        got=np.array(Image.open(path))
        for k,arr in enumerate(arrays):assert np.array_equal(got[32:,k*w:(k+1)*w],arr[y0:y1,x0:x1])
        panels.append(dict(camera=c,frame=f,file=path.name,bbox=[x0,y0,x1,y1],sha256=sha(path),input_sha256=sha(src)))
save(O/'protocol.json',dict(status='six_native_training_targets_prepared_for_visual_inspection',source=R.name,cameras=['04','07','13'],frames=[726,732],
    purpose='Check whether visible background/finger errors already occur in provided foreground mattes. Read-only diagnosis, no target changes and no retraining.',
    source_mask='Author-provided bgmtv2 JPEG, same rectification and actual array as the matched training; not a new segmentation model.',
    panels=panels,goal_complete=False))
print(json.dumps(dict(panels=len(panels),output=str(O))))

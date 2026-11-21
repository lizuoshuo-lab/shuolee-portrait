"""Show already reserved review windows without claiming full evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';P=O/'preview-review';P.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
B=['static','dynamic_nominal','dynamic_clock'];labels=['Original','Static GS','Dynamic GS','Dynamic GS + time offset'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);plan=read(O/'review-plan/protocol.json');inputs={(r['camera'],r['frame']):r for r in read(O/'evaluation-protocol.json')['records']};reports={b:{(r['camera'],r['frame']):r for r in read(O/b/'native-evaluation/progress.json')['records']} for b in B};panels=[];checks=0
assert not (P/'inventory.json').exists()
for r in plan['full_person']:
    c,f=r['camera'],r['frame'];key=c,f;source=O/inputs[key]['input'];assert sha(source)==inputs[key]['input_sha256'];z=np.load(source);arrays=[z['rgb']];stem=f'camera{c}-frame{f:06d}'
    for b in B:
        root=O/b/'native-evaluation';record=reports[b][key];assert record['input_sha256']==inputs[key]['input_sha256']
        for name,digest in record['output_sha256'].items():assert sha(root/name)==digest
        pred=np.load(root/f'{stem}-prediction.npz');raw=np.clip(pred['rgb']+.5*(1-pred['alpha'][...,None]),0,1);arr=u8(raw);assert np.array_equal(arr,np.array(Image.open(root/f'{stem}-trained.png')));checks+=1;arrays.append(arr)
    windows=[dict(name='full-person',bbox=r['bbox'])]+[d for d in plan['details'] if (d['camera'],d['frame'])==key]
    for window in windows:
        x0,y0,x1,y1=window['bbox'];w=x1-x0;h=y1-y0;im=Image.new('RGB',(4*w,h+32),'white');draw=ImageDraw.Draw(im)
        for k,(arr,label) in enumerate(zip(arrays,labels)):
            im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(w*k,32));draw.text((w*k+4,7),label,font=font,fill='black')
        path=P/f"{stem}-{window['name']}.png";im.save(path);actual=np.array(Image.open(path))
        for k,arr in enumerate(arrays):assert np.array_equal(actual[32:,k*w:(k+1)*w],arr[y0:y1,x0:x1])
        panels.append(dict(camera=c,frame=f,name=window['name'],bbox=window['bbox'],file=path.name,sha256=sha(path)))
    print('ACTOR23_PREVIEW_PANEL',c,f,flush=True)
assert checks==54 and len(panels)==90
(P/'inventory.json').write_text(json.dumps(dict(status='18fixedcontexts_90nativepanels_ready_not_full_evaluation',PNG_checks=checks,panels=panels,scope='Raw original photos and native gray-background GS renders. No resizing, sharpening or synthesis; allwindowsreservedbeforemodelrendering. Full264context evaluation remains separately required.',goal_complete=False),indent=2)+'\n',encoding='utf8');print('ACTOR23_PREVIEW_PANELS_COMPLETE')

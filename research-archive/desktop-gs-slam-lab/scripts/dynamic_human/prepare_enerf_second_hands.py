"""Native training photographs for independent hand visibility review."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

L=Path(__file__).resolve().parents[2]
S=L/'outputs/dynamic-human-enerf-second-interval-079'
O=L/'outputs/dynamic-human-enerf-hand-motion-082'
O.mkdir(exist_ok=True); (O/'source-panels').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'source-index.json').exists()
obs=read(S/'pose/observations.json')['records']
rows=[]; font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
colors=[(235,85,30),(210,40,160),(30,150,210),(65,180,55),(165,75,210)]
for c in sorted({r['camera'] for r in obs}):
    sheet=Image.new('RGB',(1040,430),'white');d=ImageDraw.Draw(sheet)
    d.text((8,4),f'Camera {c} | Light-clothed person | Each pair: native photograph / RGB whole-body prediction (not truth)',fill='black',font=font)
    for ti,t in enumerate([720,726,732]):
        r=next(r for r in obs if r['camera']==c and r['frame']==t)
        det=next(q for q in r['detections'] if q['person']=='person_light_clothing')
        xy=np.asarray(det['keypoints']);scores=np.asarray(det['scores'])
        path=S/'native-input'/f'camera{c}-frame{t:06d}.npz'
        assert sha(path)==r['input_sha256']
        with np.load(path) as z:rgb=z['rgb']
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rectified_sha256']
        for hi,(hand,base,wrist) in enumerate([('left',91,9),('right',112,10)]):
            pts=xy[base:base+21]
            center=np.median(np.r_[pts,xy[wrist:wrist+1]],axis=0)
            x0=int(np.clip(round(center[0])-80,0,1760));y0=int(np.clip(round(center[1])-80,0,920));crop=[x0,y0,x0+160,y0+160]
            raw=Image.fromarray(rgb).crop(crop);over=raw.copy();dr=ImageDraw.Draw(over);local=pts-[x0,y0]
            for f in range(5):
                chain=[0]+list(range(1+4*f,5+4*f))
                for i,j in zip(chain[:-1],chain[1:]):dr.line([tuple(local[i]),tuple(local[j])],fill=colors[f],width=1)
            for x,y in local:dr.ellipse((x-1,y-1,x+1,y+1),fill=(255,20,20))
            x,y=xy[wrist]-[x0,y0];dr.rectangle((x-2,y-2,x+2,y+2),outline=(0,200,255))
            px=8+ti*344;py=48+hi*190
            d.text((px,py-18),f'{t}: {hand}; 1 source pixel = 1 panel pixel',fill='black',font=font)
            sheet.paste(raw,(px,py));sheet.paste(over,(px+168,py))
            rows.append(dict(camera=c,frame=t,person='person_light_clothing',hand=hand,crop=crop,source_sha256=r['rectified_sha256']))
    sheet.save(O/'source-panels'/f'camera{c}-source.png')
save(O/'source-index.json',dict(records=rows,scale='Native 1:1; no photo enhancement; overlays are predictions, not annotations.'))
save(O/'protocol.json',dict(status='native_source_review_before_inference',scene='actor5_6',frames=list(range(720,733)),person='person_light_clothing',training_frames=list(range(720,733,2)),diagnostic_times=list(range(721,732,2)),excluded_cameras=['00','06','12'],source_observations_sha256=sha(S/'pose/observations.json'),purpose='Qualify visible hand observations on a moving interval before considering articulated hand supervision. No Gaussian updates authorized by scores alone.',goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name)
print('PREPARED',len(rows),'native training crops')

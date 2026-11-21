"""Fixed source-defined lower-body context, preserving native pixels."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
L=Path(__file__).resolve().parents[2];I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';R=O/'lower-body-review';R.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
if a.prepare:
    assert not (R/'protocol.json').exists()
    save(R/'protocol.json',dict(camera='00',frames=list(range(600,625)),bbox=[710,540,1040,1040],prepared_before107rendering=True,basis='Original source panel camera00/601 and existing106failure: fixed lower body includes raised knee/foot and support foot, instead of the bottom32percent window which misses the raised foot. The rectangle is context, not a semantic groundtruth mask. 106outputs were inspected; this is development.',goal_complete=False))
    print('FIXED_LOWER_BODY_REVIEW_PREPARED');raise SystemExit(0)
assert not (R/'summary.json').exists()
protocol=read(R/'protocol.json');x0,y0,x1,y1=protocol['bbox'];w=x1-x0;h=y1-y0
B=['dynamic_clock','path_rigidity','path_rigidity_binding'];roots={b:(I/b if b=='dynamic_clock' else O/b) for b in B};labels=['Original','Euclidean graph','Path edges','Path edges + binding'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
sequence=read(I/'continuity-protocol.json')['sequences']['00']['records'];rows=[]
for r in sequence:
    c,f=r['camera'],r['frame'];z=np.load(I/r['input']);assert sha(I/r['input'])==r['input_sha256'];mask=z['valid'][y0:y1,x0:x1]&(z['matte'][y0:y1,x0:x1]>.5);target=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);arrays=[z['rgb']];metrics={}
    for b in B:
        q=np.load(roots[b]/r['prediction_directory']/f'camera{c}-frame{f:06d}-prediction.npz');p=np.clip(q['rgb']+.5*(1-q['alpha'][...,None]),0,1);arrays.append(u8(p));e=(p[y0:y1,x0:x1].astype(np.float64)-target[y0:y1,x0:x1].astype(np.float64))[mask];metrics[b]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(e*e))),mae=float(abs(e).mean()))
    im=Image.new('RGB',(w*4,h+32),'white');draw=ImageDraw.Draw(im)
    for k,(arr,label) in enumerate(zip(arrays,labels)):
        im.paste(Image.fromarray(arr[y0:y1,x0:x1]),(k*w,32));draw.text((k*w+4,7),label,font=font,fill='black')
    file=R/f'camera00-frame{f:06d}-lower-body.png';im.save(file);actual=np.array(Image.open(file))
    for k,arr in enumerate(arrays):assert np.array_equal(actual[32:,k*w:(k+1)*w],arr[y0:y1,x0:x1])
    rows.append(dict(camera=c,frame=f,file=file.name,sha256=sha(file),metrics=metrics))
save(R/'summary.json',dict(status='25unaltered_native_lower_body_panels_visual_review_required',records=rows,mean_psnr_db={b:float(np.mean([r['metrics'][b]['psnr_db'] for r in rows])) for b in B},protocol_sha256=sha(R/'protocol.json'),goal_complete=False))
print('LOWER_BODY_REVIEW_PANELS_COMPLETE')

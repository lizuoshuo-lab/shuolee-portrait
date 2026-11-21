"""Enlarge matched face display panels uniformly; no alignment or new training."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/avatar-repository-pair-114/gaussianavatar')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=root/'evaluation';receipt=out/'face-display-scale-review.json'
assert not receipt.exists(), 'Already applied'
rows=read(root/'data/training-inputs.json')['records'];obs=np.load(root/'data/observations.npz',allow_pickle=False)
records=[]
for i in [0,33,69,99,135,174]:
    r=rows[i];w=round(r['wh'][0]*1536/r['wh'][1]);h=1536
    xy=obs['keypoints'][i,23:91]*np.array([w/r['wh'][0],h/r['wh'][1]])
    lo=np.maximum(0,np.floor(xy.min(0)-[24,35]).astype(int));hi=np.minimum([w,h],np.ceil(xy.max(0)+[24,25]).astype(int))
    cw,ch=(hi-lo).tolist();W,H=420,460
    assert cw<=W and ch<=H
    p=out/f'face-{i+1:06d}.jpg';before=sha(p);image=Image.open(p).convert('RGB');canvas=Image.new('RGB',image.size,(35,35,35));canvas.paste(image.crop((0,0,image.width,30)),(0,0))
    factor=min(W/cw,H/ch);nw,nh=round(cw*factor),round(ch*factor)
    for col in range(3):
        x=col*W+(W-cw)//2;y=30+(H-ch)//2;crop=image.crop((x,y,x+cw,y+ch))
        canvas.paste(crop.resize((nw,nh),Image.Resampling.LANCZOS),(col*W+(W-nw)//2,30+(H-nh)//2))
    canvas.save(p,quality=95);records.append(dict(path=str(p),before_sha256=before,after_sha256=sha(p),display_scale=factor))
receipt.write_text(json.dumps(dict(operation='Equal isotropic display enlargement of all three existing matched crops; no recovered detail claim',geometric_warp=False,metrics_changed=False,files=records),indent=2),encoding='utf-8')
print(str(receipt))

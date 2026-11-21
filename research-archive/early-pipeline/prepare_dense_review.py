from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/gaussianavatars-smplx-dense-alignment-fresh-005/comparison')
cap = cv2.VideoCapture(str(root/'fixed-pose-path.mp4'))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
assert count == 72 and abs(fps-12)<1e-6
selected = [0,8,17,26,35,44,53,62,71]
tiles=[]
for idx in selected:
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ok, frame = cap.read()
    assert ok, idx
    width=600
    frame=cv2.resize(frame,(width,round(frame.shape[0]*width/frame.shape[1])),interpolation=cv2.INTER_AREA)
    band=np.full((30,width,3),255,np.uint8)
    cv2.putText(band,f'video frame {idx}/71; 002 | 005',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
    tiles.append(np.concatenate([band,frame]))
cap.release()
sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,9,3)],axis=0)
cv2.imwrite(str(root/'path-review-contact-sheet.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
(root/'path-metadata.json').write_text(json.dumps(dict(frames=count,fps=fps,seconds=count/fps,sampled_frames=selected,all_frames_visually_reviewed=False),indent=2))

source=Image.open(root/'fixed-pose-comparison.jpg').convert('RGB')
# Existing native renders only. Each image retains the same crop and scale.
parts=[source.crop((1040,820,1560,1580)),source.crop((1560,820,2080,1580)),source.crop((0,1610,520,2370)),source.crop((520,1610,1040,2370))]
mask=np.logical_or.reduce([np.min(np.asarray(im),axis=2)<225 for im in parts])
ys,xs=np.where(mask)
box=[max(0,int(xs.min())-18),max(0,int(ys.min())-18),min(520,int(xs.max())+19),min(760,int(ys.max())+19)]
target=Image.new('RGB',(1720,790),'white')
draw=ImageDraw.Draw(target)
font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',29)
labels=['共享姿态：侧面','表面约束：侧面','共享姿态：俯视','表面约束：俯视']
for i,(part,label) in enumerate(zip(parts,labels)):
    pic=part.crop(box)
    pic.thumbnail((420,720),Image.Resampling.LANCZOS)
    target.paste(pic,(430*i+(430-pic.width)//2,58+(720-pic.height)//2))
    draw.text((430*i+18,15),label,font=font,fill='#183140')
target.save(root/'dense-method-summary.png')
(root/'summary-image-provenance.json').write_text(json.dumps(dict(source='fixed-pose-comparison.jpg',pose=33,identical_world_cameras=True,common_crop=box,meaning='Actual 002 and 005 native renders; shared crop and image scale, no geometry or appearance modification.'),indent=2))
print(json.dumps(dict(frames=count,summary=str(root/'dense-method-summary.png'))))

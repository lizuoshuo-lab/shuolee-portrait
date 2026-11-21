"""Native source-coordinate limb overlay: observations are not physical truth."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/officewoman-circle24-input127')
p=json.loads((ROOT/'plan-before-source-review.json').read_text());c=json.loads((ROOT/'source-review/source-crops.json').read_text());rows={r['frame']:r for r in p['source_rows']};crops={r['frame']:r for r in c['source_rows']}
with np.load(p['source_observations'],allow_pickle=False) as f:o=dict(f)
ids=o['frame_ids'].astype(str).tolist();out=ROOT/'source-review/lower-limb-proof';out.mkdir(exist_ok=True);receipt=[]
for n in [32,34,37,41,47,49,52,56,71,74,76,80,82,87]:
 frame='frame_%06d'%n;r=rows[frame];im=Image.open(r['image']).convert('RGB');d=ImageDraw.Draw(im);i=ids.index(frame)
 points=[]
 for j in range(13,23):
  x,y=map(float,o['keypoints'][i,j]);color='#00ffff' if j in (13,15,17,18,19) else '#ff66ff'
  d.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3);d.text((x+10,y-12),str(j),fill=color,stroke_width=1,stroke_fill='black')
  points.append(dict(index=j,name=str(o['joint_names'][j]),xy=[x,y],score=float(o['scores'][i,j])))
 b=crops[frame]['paths']['body']['crop_original_xyxy'];box=[b[0],b[1]+int((b[3]-b[1])*.43),b[2],b[3]]
 path=out/(frame+'-observed-points.png');im.crop(box).save(path)
 receipt.append(dict(frame=frame,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crop_original_xyxy=box,raw_points=points,source_image_sha256=r['source_sha256']))
(out/'receipt.json').write_text(json.dumps(dict(raw_coordinates_unchanged=True,points_are_not_ground_truth=True,records=receipt),indent=2)+'\n')
print('SOURCE_LIMB_PROOFS_READY',out)

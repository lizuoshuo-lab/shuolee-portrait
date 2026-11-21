"""Hand-only SAM proposals for photograph review, not segmentation truth."""
from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-enerf-hand-motion-082';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083/hand-masks';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'completion.json').exists();torch.set_num_threads(4);assert torch.cuda.is_available()
weights=L/'models/sam2/sam2.1_hiera_tiny.pt';model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device='cuda',apply_postprocessing=False);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
obs=[r for r in read(H/'observations.json')['records'] if r['frame'] in [728,730,732]];cams=sorted({r['camera'] for r in obs});font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
save(O/'protocol.json',dict(status='fixed_before_segmentation',weights_sha256=sha(weights),cameras=cams,frames=[728,730,732],source='Native160square handcrop; SAMinternal resizer unchanged. Outputmask only, no color target changes.',prompts='Box encloses21handcandidates+4px. Two positive seeds: midpoint ofwrist andmeanMCP; meanfourMCP. Allkeypoints arecandidates, outputs requirevisualreview.',goal_complete=False))
out=[];start=time.monotonic()
with torch.inference_mode():
 for c in cams:
  sheet=Image.new('RGB',(520,610),'white');draw=ImageDraw.Draw(sheet);draw.text((8,4),f'Camera{c}: native / predicted hand mask / mask contour',font=font,fill='black')
  for ti,t in enumerate([728,730,732]):
   r=next(r for r in obs if r['camera']==c and r['frame']==t);p=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(p)==r['input_sha256']
   with np.load(p) as z:rgb=z['rgb']
   x0,y0,x1,y1=r['crop'];crop=rgb[y0:y1,x0:x1];xy=np.array(r['keypoints'])-[x0,y0];mc=xy[[5,9,13,17]].mean(0);points=np.array([(xy[0]+mc)/2,mc]);box=np.r_[xy.min(0)-4,xy.max(0)+4];box=np.clip(box,0,159)
   predictor.set_image(crop);m,score,_=predictor.predict(box=box,point_coords=points,point_labels=np.ones(2,np.int32),multimask_output=False);mask=m[0].astype(bool)
   stem=f'camera{c}-frame{t:06d}';np.savez_compressed(O/f'{stem}.npz',mask=mask,crop=r['crop']);out.append(dict(camera=c,frame=t,input_sha256=r['input_sha256'],filename=stem+'.npz',sha256=sha(O/f'{stem}.npz'),box=box.tolist(),points=points.tolist(),score=float(score[0]),mask_pixels=int(mask.sum())))
   foreground=np.rint(crop*mask[...,None]+127.5*(1-mask[...,None])).astype(np.uint8);cont=crop.copy();contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(cont,contours,-1,(0,255,60),1)
   y=40+ti*190;draw.text((8,y-16),f'Frame{t} | SAM candidate, not GT',font=font,fill='black')
   for k,im in enumerate([crop,foreground,cont]):sheet.paste(Image.fromarray(im),(8+170*k,y))
  sheet.save(O/f'camera{c}-review.png');print('HAND_SEGMENT',c,flush=True)
save(O/'completion.json',dict(status='45nativehandmasks_visualreview_pending',records=out,seconds=time.monotonic()-start,goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name)

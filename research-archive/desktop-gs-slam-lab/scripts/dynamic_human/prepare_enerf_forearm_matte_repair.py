"""Prepare seven training-only local matte candidates; do not train before review."""
from pathlib import Path
import json,hashlib,shutil,numpy as np,cv2,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');I=L/'outputs/dynamic-human-enerf-articulated-control-064';R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';R.mkdir(exist_ok=True);O=R/'candidates';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (R/'preparation.json').exists()
protocol=dict(status='fixed_before_segmentation',camera='01',frames=list(range(744,757,2)),region_xyxy=[660,420,870,640],box=[660,230,1060,960],positive_point=[850,420],model='SAM2.1 Hiera Tiny',rule='Only inside fixed forearm rectangle: maximum of original released matte and3x3-eroded SAM dark-person foreground. Outside region unchanged. Native RGB unchanged. Requires each proposed mask to be visually reviewed.',selection='Existing known camera01 visible forearm false-negative diagnosed before predictions; seven original even training images only. No newcamera or oddframe images.',qualification='Additional weak training labels, not GT. No model trained and no completeness claim.',goal_complete=False)
save(R/'protocol.json',protocol)
torch.set_num_threads(4);torch.manual_seed(0)
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(L/'models/sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);rows=[]
with torch.inference_mode():
 for t in protocol['frames']:
  f=I/'native-input'/f'camera01-frame{t:06d}.npz';z=np.load(f);rgb=z['rgb'];matte=z['matte'];predictor.set_image(rgb)
  masks,scores,_=predictor.predict(box=np.array(protocol['box']),point_coords=np.array([protocol['positive_point']]),point_labels=np.array([1]),multimask_output=False)
  core=cv2.erode(masks[0].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
  x0,y0,x1,y1=protocol['region_xyxy'];region=np.zeros_like(core);region[y0:y1,x0:x1]=True;support=core&region&z['valid'];revised=matte.copy();revised[support]=1
  changed=revised>matte;assert np.count_nonzero(changed)<10000;assert np.array_equal(revised[~region],matte[~region])
  out=O/f'camera01-frame{t:06d}.npz';np.savez_compressed(out,matte=revised,raw_mask=masks[0],changed=changed)
  def composite(m):return np.rint(rgb.astype(float)*m[...,None]+127.5*(1-m[...,None])).clip(0,255).astype(np.uint8)
  ims=[rgb,composite(matte),composite(revised),np.repeat((changed*255).astype(np.uint8)[...,None],3,2)]
  labels=['Original native photograph','Current training RGB target','Proposed local correction','Changed pixels only']
  w=x1-x0;h=y1-y0;panel=Image.new('RGB',(4*w,h+32),'white');draw=ImageDraw.Draw(panel)
  for k,im in enumerate(ims):panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(k*w,32));draw.text((k*w+3,10),labels[k],fill='black')
  panel.save(O/f'camera01-frame{t:06d}-review.png')
  rows.append(dict(camera='01',frame=t,source_sha256=sha(f),candidate_sha256=sha(out),changed_pixels=int(changed.sum()),raised_from_below_half=int((support&(matte<.5)).sum()),sam_score=float(scores[0])))
  print('FOREARM_MASK',json.dumps(rows[-1]),flush=True)
save(R/'preparation.json',dict(status='seven_training_only_candidates_visual_review_required_no_training',records=rows,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)

"""Training-only conservative instance cores from exact current native inputs."""
from pathlib import Path
import json,hashlib,time,shutil,numpy as np,cv2,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-actor-supervision-070';I=L/'outputs/dynamic-human-enerf-articulated-control-064';OLD=L/'outputs/dynamic-human-enerf-mask-quality-038/train15'
R.mkdir(exist_ok=True);M=R/'masks';M.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'mask-preparation.json').exists()
torch.set_num_threads(4);torch.manual_seed(0);began=time.monotonic()
prompts=read(OLD/'prompt-protocol.json')['prompts'];rows=[r for r in read(I/'input-audit.json')['records'] if r['frame']==750 and r['split']=='train'];assert len(rows)==15 and set(prompts)=={r['camera'] for r in rows}
checkpoint=L/'models/sam2/sam2.1_hiera_tiny.pt'
save(R/'mask-protocol.json',dict(status='fixed_before_instance_predictions',source='Exact064native-input RGB, frame750 only; manual training prompts reused from038, masks recomputed.',
 prompt_sha256=sha(OLD/'prompt-protocol.json'),sam2_sha256=sha(checkpoint),device='cuda float32',rule='Exclusive person-mask core eroded7x7; exclude7x7dilated other-person and held-object masks; require releasedmatte>.95 and valid calibration pixels. Others ignored, not background.',
 classes=['dark person','light person'],qualification='SAM2 predictions and manual prompts are additional weak training supervision, not instanceGT. No odd frames or cameras00/06/12 read.',goal_complete=False))
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device='cuda',apply_postprocessing=False)
predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);records=[];kernel=np.ones((7,7),np.uint8)
with torch.inference_mode():
 for r in rows:
  c=r['camera'];source=I/'native-input'/f'camera{c}-frame000750.npz';assert sha(source)==r['input_sha256'];z=np.load(source);rgb=z['rgb'];masks=[];scores=[]
  existing=M/f'camera{c}-frame000750.npz'
  if existing.exists():
   masks=np.load(existing)['raw_masks'];scores=None
  else:
   predictor.set_image(rgb)
   for prompt in prompts[c]:
    kw={} if 'point' not in prompt else dict(point_coords=np.array([prompt['point']]),point_labels=np.array([1]))
    mask,score,_=predictor.predict(box=np.array(prompt['box']),multimask_output=False,**kw);masks.append(mask[0].astype(bool));scores.append(float(score[0]))
  masks=np.stack(masks);labels=np.full((1080,1920),-1,np.int8)
  for a in [0,1]:
   other=cv2.dilate((masks[1-a]|masks[2]).astype(np.uint8),kernel).astype(bool);core=cv2.erode(masks[a].astype(np.uint8),kernel).astype(bool)&~other&(z['matte']>.95)&z['valid'];labels[core]=a
  assert all((labels==a).sum()>100 for a in [0,1]);np.savez_compressed(M/f'camera{c}-frame000750.npz',labels=labels,raw_masks=masks)
  x0,y0,x1,y1=z['bbox'];w=x1-x0;h=y1-y0;overlay=rgb.copy();palette=np.array([[255,135,25],[40,140,255]])
  for a in [0,1]:sel=labels==a;overlay[sel]=np.rint(.55*rgb[sel]+.45*palette[a]).astype(np.uint8)
  panel=Image.new('RGB',(2*w,h+36),'white');draw=ImageDraw.Draw(panel)
  for j,(label,img) in enumerate([('Original native training photograph',rgb),('Orange dark / blue light; uncolored ignored',overlay)]):panel.paste(Image.fromarray(img[y0:y1,x0:x1]),(j*w,36));draw.text((j*w+4,12),label,fill='black')
  panel.save(M/f'camera{c}-labels-review.png');records.append(dict(camera=c,frame=750,input_sha256=r['input_sha256'],labels_sha256=sha(M/f'camera{c}-frame000750.npz'),counts=[int((labels==a).sum()) for a in [0,1]],scores=scores,bbox=z['bbox'].tolist(),recovered_saved_prediction=scores is None))
  print('ACTOR_SUPERVISION',c,records[-1]['counts'],flush=True)
save(R/'mask-preparation.json',dict(status='15training_only_instance_core_candidates_ready_visual_review_required',records=records,seconds=time.monotonic()-began,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)

"""Reproduce camera10 frame750 mask, then use identical prompts/rule at749."""
from pathlib import Path
import json,time,cv2,torch,numpy as np
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from audit_enerf_input import read_cameras,digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');RAW=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
OLD=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
OUT=LAB/'outputs/dynamic-human-enerf-camera10-time-050';OUT.mkdir(exist_ok=True)
MASK=OUT/'input-mask';MASK.mkdir(exist_ok=True)
assert not (MASK/'summary.json').exists() and not torch.cuda.is_available()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
prompts=json.loads((OLD/'prompt-protocol.json').read_text())['prompts']['10']
cam=read_cameras(RAW/'optimized')['10'];K=np.array(cam['K']);D=np.array(cam['D'])
mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
save(OUT/'protocol.json',dict(status='fixed_before_mask_and_training',canonical_frame=750,changed_camera='10',source_frame_before=750,source_frame_after=749,
 evidence='049: all6camera-pair edges touching10 select the same signed relative lag and improve760moving-point residual. Candidate empirical frame index, not physical clock calibration.',
 intervention='Replace only camera10 RGB+mask observation, regenerate mask using identical750prompts and SAM2/closed-core rule. Other14observations unchanged.',
 fixed='048initial Gaussian shell/colors,6000steps,seed73,optimizer/densification/loss and original cameras. Heldout00/06/12photos at750 unchanged.',
 scope='Single-camera training-observation control; temporal state differs by design, not a fullsequence retiming or a fresh final test.',goal_complete=False))
torch.set_num_threads(4);model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(LAB/'models/sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0);records=[];start=time.time()
with torch.inference_mode():
 for t in [750,749]:
  raw=cv2.imread(str(RAW/'images/10'/f'{t:06d}.jpg'))
  rgb=cv2.remap(raw,mx,my,cv2.INTER_LINEAR)[...,::-1].copy()
  matte=cv2.remap(cv2.imread(str(RAW/'bgmtv2/10'/f'{t:06d}.jpg'),0).astype(np.float32)/255,mx,my,cv2.INTER_LINEAR)
  if t==750:
   assert np.array_equal(rgb,np.array(Image.open(OLD/'camera10-source.png')))
   assert np.array_equal(matte,np.load(OLD/'camera10-released.npy'))
  predictor.set_image(rgb);parts=[]
  for p in prompts:
   m,s,l=predictor.predict(box=np.array(p['box']),point_coords=np.array([p['point']]),point_labels=np.array([1]),multimask_output=False)
   parts.append(m[0].astype(bool))
  union=np.any(parts,axis=0)
  support=cv2.morphologyEx(union.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49))).astype(bool)
  mask=union|((matte>.95)&support)
  if t==750:
   assert np.array_equal(union,np.array(Image.open(OLD/'camera10-candidate.png'))>0)
   assert np.array_equal(mask,np.array(Image.open(OLD/'hybrid-closed-core/camera10-mask.png'))>0)
  Image.fromarray(mask.astype(np.uint8)*255).save(MASK/f'frame{t}-mask.png')
  Image.fromarray(rgb).save(MASK/f'frame{t}-source.png')
  # Same float32 OpenCV undistortion convention as039/048 training RGB.
  training_rgb=cv2.undistort(raw,K.astype(np.float32),D.astype(np.float32))[...,::-1].copy()
  if t==750:
   baseline=np.load(LAB/'outputs/dynamic-human-enerf-representation-048/train-camera-10.npz')
   assert np.array_equal(training_rgb,baseline['rgb']) and np.array_equal(mask.astype(np.float32),baseline['matte'])
  np.savez_compressed(MASK/f'frame{t}-training.npz',rgb=training_rgb,matte=mask.astype(np.float32),valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079),K=K,w2c=np.array(cam['w2c']))
  boxes=np.array([p['box'] for p in prompts]);x0,y0=boxes[:,:2].min(0);x1,y1=boxes[:,2:].max(0);w=x1-x0;h=y1-y0
  comp=np.rint(rgb.astype(np.float32)*mask[...,None]+127.5*(~mask[...,None])).astype(np.uint8)
  panel=Image.new('RGB',(2*w,h+32),'white');d=ImageDraw.Draw(panel)
  for j,(label,im) in enumerate([('Original input photograph',rgb),('Input mask only (not a render)',comp)]):
   panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(j*w,32));d.text((j*w+4,10),f'{t}: {label}',fill='black')
  panel.save(MASK/f'frame{t}-input-comparison.png')
  records.append(dict(frame=t,mask_pixels=int(mask.sum()),raw_rgb_sha256=digest(RAW/'images/10'/f'{t:06d}.jpg'),mask_sha256=digest(MASK/f'frame{t}-mask.png')))
  print('MASK',t,int(mask.sum()),flush=True)
save(MASK/'summary.json',dict(status='exact750SAM_mask_RGB_reproduced_749candidate_ready_for_visual_review',prompts=prompts,records=records,elapsed_seconds=time.time()-start,goal_complete=False))

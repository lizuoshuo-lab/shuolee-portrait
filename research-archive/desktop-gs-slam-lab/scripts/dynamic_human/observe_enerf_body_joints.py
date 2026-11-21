"""Training-view body observations for an articulated prior, never ground truth."""
from pathlib import Path
import hashlib,json,time,shutil
import torch,numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=LAB/'scripts/dynamic_human'
BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame';MASK=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
OUT=LAB/'outputs/dynamic-human-enerf-body-observations-057';OUT.mkdir(exist_ok=True)
assert not (OUT/'observations.json').exists(), 'Preserve completed observations'
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
types=['person_dark_clothing','person_light_clothing'];names=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
links=[(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7),(6,8),(8,10),(7,9),(9,11)]
weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
save(OUT/'protocol.json',dict(status='frozen_before_inference',frame=750,training_cameras=[f.stem[-2:] for f in sorted(BASE.glob('train-camera-*.npz'))],
 method='Official torchvision KeypointRCNN ResNet50 FPN COCO_V1, CPU, standard model resize. Body joints5..16 only; original output coordinates retained.',
 weights_url=weights.url,threshold_detection=.8,
 association='Hungarian maximizing body joint support in separately annotated SAM person masks, plus0.2boxIoU. Tentative, must visually review occluded views.',
 limits='Predicted joints and SAM association are not truth. No heldout photos, no model geometry modified, no SMPL fitting until observation review.',goal_complete=False))
torch.set_num_threads(4);torch.manual_seed(73)
model=keypointrcnn_resnet50_fpn(weights=weights,progress=False).cpu().eval();rows=[];started=time.monotonic()
for f in sorted(BASE.glob('train-camera-*.npz')):
 c=f.stem[-2:];z=np.load(f);rgb=z['rgb'];h,w=rgb.shape[:2]
 assert c not in ['00','06','12']
 with torch.inference_mode():pred=model([to_tensor(Image.fromarray(rgb))])[0]
 ids=torch.where(pred['scores']>=.8)[0].tolist();candidates=[];masks=[];mbox=[]
 for typ in types:
  m=np.array(Image.open(MASK/f'camera{c}-{typ}.png'))>0;masks.append(m);yy,xx=np.where(m);mbox.append(np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1]))
 for idx in ids:
  xy=pred['keypoints'][idx,5:17,:2].numpy();conf=pred['keypoints_scores'][idx,5:17].numpy();bbox=pred['boxes'][idx].numpy()
  scores=[]
  for m,b in zip(masks,mbox):
   px=np.rint(xy).astype(int);px[:,0]=px[:,0].clip(0,w-1);px[:,1]=px[:,1].clip(0,h-1)
   support=float(np.mean(m[px[:,1],px[:,0]]));inter=np.prod(np.maximum(np.minimum(b[2:],bbox[2:])-np.maximum(b[:2],bbox[:2]),0));union=np.prod(b[2:]-b[:2])+np.prod(bbox[2:]-bbox[:2])-inter
   scores.append(support+.2*float(inter/max(union,1)))
  candidates.append(dict(index=idx,score=float(pred['scores'][idx]),box=bbox.tolist(),body_xy=xy.tolist(),body_confidence=conf.tolist(),association_scores=scores))
 assoc=[]
 if candidates:
  costs=-np.array([x['association_scores'] for x in candidates]).T;rr,cc=linear_sum_assignment(costs)
  for r,k in zip(rr,cc):assoc.append(dict(person=types[r],candidate=int(k),association_score=float(-costs[r,k]),status='tentative_requires_visual_review'))
 canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas);colors=[(255,100,20),(0,180,255)]
 for row in assoc:
  pi=types.index(row['person']);cand=candidates[row['candidate']];color=colors[pi];xy=cand['body_xy']
  draw.rectangle(cand['box'],outline=color,width=2)
  for a,b in links:draw.line([tuple(xy[a]),tuple(xy[b])],fill=color,width=2)
  for j,(x,y) in enumerate(xy):draw.ellipse([x-3,y-3,x+3,y+3],fill=color);draw.text((x+4,y),str(j+5),fill=color)
 yy,xx=np.where(masks[0]|masks[1]);box=[max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(w,int(xx.max())+31),min(h,int(yy.max())+31)]
 canvas.save(OUT/f'camera{c}-body-observations.png');canvas.crop(box).save(OUT/f'camera{c}-person-body-observations.png')
 rows.append(dict(camera=c,input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),body_joint_names=names,candidates=candidates,associations=assoc,K=z['K'].tolist(),w2c=z['w2c'].tolist()))
 save(OUT/'progress.json',dict(cameras_done=len(rows),latest_camera=c,elapsed_seconds=time.monotonic()-started));print('BODY_OBSERVATION',c,len(candidates),flush=True)
save(OUT/'observations.json',dict(status='15training_views_inferred_tentative_association_visual_review_pending',frame=750,records=rows,goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)

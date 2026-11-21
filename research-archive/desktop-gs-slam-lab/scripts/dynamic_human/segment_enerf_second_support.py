"""Training-only foreground support candidates. SAM predictions are not truth."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
SRC=LAB/'outputs/dynamic-human-enerf-second-interval-079'
OUT=LAB/'outputs/dynamic-human-enerf-support-correction-081'
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');a=ap.parse_args()
 OUT.mkdir(exist_ok=True);(OUT/'candidates').mkdir(exist_ok=True)
 records=[r for r in read(SRC/'pose/observations.json')['records'] if r['split']=='train' and (a.all or (r['camera'] in ['04','07','13'] and r['frame'] in [726,732]))]
 assert all(r['camera'] not in ['00','06','12'] and r['frame']%2==0 for r in records)
 save(OUT/'protocol.json',dict(status='training_support_candidates_visual_review_required',training_images=105,cameras=15,source_interval=[720,732],heldout_inputs_unchanged=True,training_rgb_unchanged=True,method='SAM2.1 tiny per-person and group union; original matte retained inside 8px support band; no sharpening',uncertainty='Segmentation proposals require visual review, not semantic ground truth',goal_complete=False))
 torch.set_num_threads(4);model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(LAB/'models/sam2/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False);predictor=SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
 rows=[];began=time.monotonic()
 with torch.inference_mode():
  for r in records:
   c,t=r['camera'],r['frame'];stem=f'camera{c}-frame{t:06d}';dest=OUT/'candidates'/stem
   if dest.with_suffix('.json').exists():rows.append(read(dest.with_suffix('.json')));continue
   path=SRC/'native-input'/f'{stem}.npz';assert sha(path)==r['input_sha256'];z=np.load(path);rgb=z['rgb'];matte=z['matte'];predictor.set_image(rgb);masks=[];prompts=[];points=[]
   for d in r['detections']:
    k=np.array(d['keypoints']);s=np.array(d['scores']);idx=[i for i in [0,5,6,9,10,11,12,15,16] if s[i]>.7 and i not in d.get('disabled_fitting_indices',[])]
    q=k[idx,:2]
    if not len(q):continue
    m,score,_=predictor.predict(box=np.array(d['box']),point_coords=q,point_labels=np.ones(len(q),np.int32),multimask_output=False)
    masks.append(m[0].astype(bool));points.extend(q.tolist());prompts.append(dict(person=d['person'],box=d['box'],positive_points=q.tolist(),predicted_iou=float(score[0])))
   assert masks
   box=z['bbox'];m,score,_=predictor.predict(box=box,point_coords=np.array(points),point_labels=np.ones(len(points),np.int32),multimask_output=False);masks.append(m[0].astype(bool))
   union=np.any(masks,axis=0);support=cv2.dilate(union.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0
   candidate=matte*support;removed=(matte>.5)&~support
   np.savez_compressed(dest.with_suffix('.npz'),support=support,sam_union=union,matte=candidate)
   x0,y0,x1,y1=box;w=int(x1-x0);h=int(y1-y0)
   panel=Image.new('RGB',(3*w,h+32),'white');draw=ImageDraw.Draw(panel)
   old=rgb*matte[...,None]+127.5*(1-matte[...,None]);new=rgb*candidate[...,None]+127.5*(1-candidate[...,None])
   for j,(name,im) in enumerate([('Native training photograph',rgb),('Original training target',old),('Candidate training target',new)]):
    panel.paste(Image.fromarray(np.rint(im).astype(np.uint8)[y0:y1,x0:x1]),(j*w,32));draw.text((j*w+4,10),stem+' '+name,fill='black')
   panel.save(dest.with_name(stem+'-comparison.png'))
   row=dict(camera=c,frame=t,input_sha256=sha(path),candidate_sha256=sha(dest.with_suffix('.npz')),prompts=prompts,group_predicted_iou=float(score[0]),removed_foreground_pixels=int(removed.sum()),original_foreground_pixels=int((matte>.5).sum()),bbox=box.tolist(),visual_review='pending',goal_complete=False)
   save(dest.with_suffix('.json'),row);rows.append(row);print('CANDIDATE',c,t,row['removed_foreground_pixels'],flush=True)
 save(OUT/('all-candidates.json' if a.all else 'pilot-candidates.json'),dict(status='generated_visual_review_pending',records=rows,elapsed_seconds=time.monotonic()-began,goal_complete=False))
if __name__=='__main__':main()

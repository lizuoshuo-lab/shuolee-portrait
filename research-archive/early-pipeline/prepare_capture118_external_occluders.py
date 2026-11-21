"""Observe visible furniture; never turn SAM/YOLO masks into depth truth."""
from pathlib import Path
import argparse,sys,time,json
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from capture118_all_subjects import REC,rt,ASSETS,state
SOURCE=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
OUT=REC/'indoor118-external-occluders'
ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu'],default='cpu');args=ap.parse_args()
assert not OUT.exists(),'Preserve existing evidence; never overwrite or duplicate its job.'
OUT.mkdir();(OUT/'masks').mkdir();(OUT/'review').mkdir()
rt.configure(ASSETS,SOURCE)
plan=dict(created=rt.now(),purpose='Prepare a single-variable alpha-supervision candidate for indoor118 using visible furniture, independently of fitted mesh errors.',
 inputs='Frozen89 raw images, original person masks and detector boxes. No camera, RGB, person-mask or learned-point changes.',
 policy='YOLO chair/couch/dining-table boxes at conf0.25; SAM2 mask score>=0.5; conservative2px erosion; remove all person-mask pixels; restrict to the lower55percent of the person detector box expanded10percent in width. No fitted-mesh projection is used to choose ignored pixels.',
 interpretation='Only a candidate uncertainty mask. Object labels and 2D segmentation cannot establish occluder depth. Review actual pixels before any loss change.',
 control='All positive person-alpha pixels and all RGB loss remain unchanged; alpha outside these masks retains its original supervision. Facial pixels cannot enter the ignore region.',
 rejection='Reject if furniture is mistaken for person or background with no plausible occlusion. Do not repair by per-frame manual masks, shifting points or changing K.',
 device=args.device,budget_seconds=900,max_frames=89,training_updates=0,physical_calibration_accepted=False)
rt.save(OUT/'plan.json',plan);state('indoor118',state='external_occluder_observation',training_running=False,device='cpu',evidence=str(OUT))
import torch,cv2
torch.set_num_threads(4)
from shuolee_portrait.observations import load_detector
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
detector=load_detector().to('cpu')
assert {detector.names[i] for i in [56,57,60]}=={'chair','couch','dining table'}
sam=rt.ASSETS/'sam2/sam2.1_hiera_tiny.pt'
entry=next(x for x in rt.read(rt.PROJECT/'configs/server-assets.json')['files'] if x['path']=='sam2/sam2.1_hiera_tiny.pt')
assert rt.sha(sam)==entry['sha256']
net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(sam),device='cpu',apply_postprocessing=False)
segmenter=SAM2ImagePredictor(net,max_hole_area=0,max_sprinkle_area=0)
rows=rt.read(SOURCE/'data/training-inputs.json')['records'];assert len(rows)==89
selected=set(np.linspace(0,len(rows)-1,6).astype(int));records=[];start=time.monotonic()
try:
 with torch.inference_mode():
  for i,r in enumerate(rows):
   if time.monotonic()-start>900:raise RuntimeError('CPU furniture observation budget exhausted; evidence preserved.')
   assert rt.sha(r['mask'])==r['mask_sha256']
   rgb=np.asarray(Image.open(r['image']).convert('RGB'));H,W=rgb.shape[:2]
   person=np.asarray(Image.open(r['mask']).convert('L'))>127
   detected=detector.predict(rgb[:,:,::-1].copy(),classes=[56,57,60],conf=.25,verbose=False,device='cpu')[0]
   boxes=detected.boxes.xyxy.cpu().numpy();cls=detected.boxes.cls.cpu().numpy();confidence=detected.boxes.conf.cpu().numpy()
   furniture=np.zeros((H,W),bool);objects=[]
   if len(boxes):
    segmenter.set_image(rgb)
    for box,category,prob in zip(boxes,cls,confidence):
     masks,scores,_=segmenter.predict(box=box,multimask_output=False)
     accepted=float(scores[0])>=.5
     if accepted:furniture|=masks[0].astype(bool)
     objects.append(dict(class_id=int(category),label=detector.names[int(category)],box=box.tolist(),detector_confidence=float(prob),sam_score=float(scores[0]),included=accepted))
   furniture=cv2.erode(furniture.astype('uint8'),np.ones((5,5),np.uint8),iterations=1).astype(bool)&~person
   b=np.array(r['initial_box_xyxy']);bw=b[2]-b[0];bh=b[3]-b[1]
   roi=np.rint(np.clip([b[0]-.1*bw,b[1]+.45*bh,b[2]+.1*bw,b[3]],[0,0,0,0],[W,H,W,H])).astype(int)
   region=np.zeros((H,W),bool);region[roi[1]:roi[3],roi[0]:roi[2]]=True
   ignore=furniture&region;assert not (ignore&person).any()
   filename=r['frame']+'.png';path=OUT/'masks'/filename;Image.fromarray(ignore.astype('uint8')*255).save(path)
   record=dict(frame=r['frame'],image=r['image'],image_sha256=rt.sha(r['image']),person_mask_sha256=r['mask_sha256'],ignore_mask=str(path),ignore_sha256=rt.sha(path),
    roi_xyxy=roi.tolist(),ignore_pixels=int(ignore.sum()),person_positive_removed=0,objects=objects)
   records.append(record)
   if i in selected:
    overlay=rgb.astype(float);overlay[ignore]=.45*overlay[ignore]+.55*np.array([255,145,0]);overlay=Image.fromarray(overlay.astype('uint8'))
    Image.fromarray(rgb).save(OUT/'review'/f'{r["frame"]}_raw.png');overlay.save(OUT/'review'/f'{r["frame"]}_candidate_ignore_overlay.png')
    Image.fromarray(ignore.astype('uint8')*255).save(OUT/'review'/f'{r["frame"]}_ignore_mask.png')
   rt.save(OUT/'progress.json',dict(updated=rt.now(),done=i+1,total=len(rows),seconds=time.monotonic()-start));print('occluder',i+1,len(rows),int(ignore.sum()),flush=True)
 result=dict(created=rt.now(),state='actual_pixel_review_required',records=records,total_ignore_pixels=sum(r['ignore_pixels'] for r in records),frames_with_ignore=sum(r['ignore_pixels']>0 for r in records),
  camera_sha256=rt.sha(SOURCE/'data/training-inputs.json'),model_mask_is_depth_truth=False,training_admitted=False,seconds=time.monotonic()-start)
 rt.save(OUT/'result.json',result);state('indoor118',state='external_occluder_review_pending',training_running=False,evidence=str(OUT))
except BaseException as e:
 rt.save(OUT/'failure.json',dict(created=rt.now(),error=repr(e),completed_frames=len(records)));state('indoor118',state='external_occluder_failed',training_running=False,error=repr(e));raise

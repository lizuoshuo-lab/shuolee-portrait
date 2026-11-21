"""Fresh person masks and native COCO-133 observations; no historical subject files."""
import json,zipfile,time
import numpy as np
from PIL import Image
from . import runtime as rt

def face_region_box(keypoints,scores,wh):
 """An approximate detector ROI only; it does not establish material identity."""
 keypoints=np.asarray(keypoints);scores=np.asarray(scores)
 if keypoints.shape!=(133,2) or scores.shape!=(133,):raise ValueError('Expected native COCO-133 observations.')
 face_points=keypoints[23:91];face_scores=scores[23:91]
 finite=np.isfinite(face_points).all(1)&np.isfinite(face_scores)
 if not finite.any():return None
 threshold=.5*max(float(np.max(face_scores[finite])),1e-6)
 valid=finite&(face_scores>=threshold)
 if valid.sum()<8:return None
 box=np.r_[np.min(face_points[valid],axis=0),np.max(face_points[valid],axis=0)]
 W,H=wh;box=np.clip(box,[0,0,0,0],[W-1,H-1,W-1,H-1])
 return None if (box[2:]-box[:2]<4).any() else box.tolist()


def load_detector():
 """Load the pinned official YOLO model without disabling weights-only globally."""
 import importlib,torch
 from ultralytics import YOLO
 path=rt.ASSETS/'detection/yolov8n.pt'
 entry=next(x for x in rt.read(rt.PROJECT/'configs/server-assets.json')['files'] if x['path']=='detection/yolov8n.pt')
 if rt.sha(path)!=entry['sha256']:raise RuntimeError('Detector checkpoint checksum mismatch.')
 allowed={
  'torch.nn.modules.pooling.MaxPool2d','torch.nn.modules.container.ModuleList',
  'torch.nn.modules.container.Sequential','torch.nn.modules.batchnorm.BatchNorm2d',
  'torch.nn.modules.activation.SiLU','torch.nn.modules.conv.Conv2d',
  'torch.nn.modules.upsampling.Upsample','ultralytics.nn.tasks.DetectionModel',
  'ultralytics.nn.modules.block.Bottleneck','ultralytics.nn.modules.block.C2f',
  'ultralytics.nn.modules.block.SPPF','ultralytics.nn.modules.block.DFL',
  'ultralytics.nn.modules.conv.Conv','ultralytics.nn.modules.conv.Concat',
  'ultralytics.nn.modules.head.Detect'}
 found=set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
 if not found<=allowed:raise RuntimeError(f'Unreviewed detector globals: {sorted(found-allowed)}')
 objects=[getattr(importlib.import_module(name.rsplit('.',1)[0]),name.rsplit('.',1)[1]) for name in sorted(found)]
 with torch.serialization.safe_globals(objects):return YOLO(str(path))

def observe(root,cfg):
 import torch,onnxruntime as ort
 from rtmlib import RTMPose
 from rtmlib.visualization.skeleton.coco133 import coco133
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 inputs=rt.read(root/'data/inputs.json');rows=inputs['records'];maskdir=root/'data/masks';maskdir.mkdir()
 detector=load_detector()
 net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(rt.ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
 segmenter=SAM2ImagePredictor(net,max_hole_area=0,max_sprinkle_area=0)
 model=rt.ASSETS/'rtmw/rtmw-l-384x288.onnx'
 norm=rt.read(rt.ASSETS/'rtmw/normalization.json')
 class Pose(RTMPose):
  def __init__(self):
   opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
   self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
   self.onnx_model=str(model);self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
 pose=Pose();names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
 flip=np.array([names.index(coco133['keypoint_info'][i]['swap']) if coco133['keypoint_info'][i]['swap'] else i for i in range(133)])
 points=[];scores=[];disagreement=[];boxes=[];reports=[];previous=None
 with torch.inference_mode():
  for i,row in enumerate(rows):
   rgb=np.array(Image.open(row['image']).convert('RGB'));H,W=rgb.shape[:2]
   det=detector.predict(rgb[:,:,::-1].copy(),classes=[0],conf=.45,verbose=False,device=0)[0]
   b=det.boxes.xyxy.cpu().numpy()
   if len(b)==0:raise RuntimeError(f'No person detected in {row["name"]}. No interpolated bbox.')
   areas=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
   if previous is None:
    order=np.argsort(-areas)
    if len(order)>1 and areas[order[1]]>.65*areas[order[0]]:raise RuntimeError('Multiple similarly sized people in the first frame. Provide a single-person clip.')
    box=b[order[0]]
   else:
    lo=np.maximum(b[:,:2],previous[:2]);hi=np.minimum(b[:,2:],previous[2:]);inter=np.maximum(hi-lo,0).prod(1)
    iou=inter/(areas+np.prod(previous[2:]-previous[:2])-inter+1e-8);order=np.argsort(-iou)
    if iou[order[0]]<.1 or (len(order)>1 and iou[order[0]]-iou[order[1]]<.05):raise RuntimeError(f'Person track ambiguous at {row["name"]}. No identity switch is silently accepted.')
    box=b[order[0]]
   other_boxes=[item.tolist() for index,item in enumerate(b) if index!=int(order[0])]
   previous=box.copy();margin=.04*max(box[2:]-box[:2]);box=np.clip(box+[-margin,-margin,margin,margin],[0,0,0,0],[W,H,W,H])
   a,sa=pose(rgb,bboxes=box[None]);a,sa=a[0],sa[0]
   fb=np.array([W-1-box[2],box[1],W-1-box[0],box[3]])
   b,sb=pose(rgb[:,::-1].copy(),bboxes=fb[None]);b,sb=b[0][flip],sb[0][flip];b[:,0]=W-1-b[:,0]
   kp=(a+b)/2;err=np.linalg.norm(a-b,axis=-1)
   if not np.isfinite(kp).all():raise RuntimeError('Non-finite landmark observations.')
   segmenter.set_image(rgb);center=kp[[5,6,11,12]].mean(0)[None]
   masks,sc,_=segmenter.predict(box=box,point_coords=center,point_labels=np.ones(1,dtype=np.int32),multimask_output=False)
   mask=masks[0].astype(bool);area=int(mask.sum());fraction=area/(W*H)
   if not .005<fraction<.95 or sc[0]<.5:raise RuntimeError(f'Person mask failed a coarse sanity check in {row["name"]}.')
   path=maskdir/row['name'];Image.fromarray(mask.astype('uint8')*255).save(path)
   # Face landmarks define an approximate region only. They never define the
   # identity of material correspondences in the appearance-track stage.
   face_box=face_region_box(kp,np.minimum(sa,sb),(W,H))
   row.update(mask=str(path),mask_sha256=rt.sha(path),mask_kind='new_SAM2_observation',initial_box_xyxy=box.tolist(),
              other_person_boxes_xyxy=other_boxes,face_box_xyxy=face_box,face_region_source='RTMW face bounds: ROI only, not material identity')
   points.append(kp);scores.append(np.minimum(sa,sb));disagreement.append(err);boxes.append(box)
   reports.append(dict(frame=row['frame'],mask_fraction=fraction,segmentation_score=float(sc[0]),model_scores_are_not_truth=True))
   rt.save(root/'progress.json',dict(stage='observations',done=i+1,total=len(rows),updated=rt.now()))
   print(f'observations {i+1}/{len(rows)}',flush=True)
 np.savez_compressed(root/'data/observations.npz',frame_ids=np.array([r['frame'] for r in rows]),keypoints=points,scores=scores,flip_disagreement=disagreement,boxes=boxes,joint_names=names)
 rt.save(root/'data/inputs.json',inputs);rt.save(root/'data/training-inputs.json',inputs)
 rt.save(root/'mask-audit.json',{'records':reports,'new_masks':len(rows),'old_masks_loaded':False,'visual_quality_accepted':False})

"""Training-camera RTMW tracks; reserve odd times before timing fitting."""
from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageDraw
from rtmlib import RTMPose
from audit_enerf_input import read_cameras
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');RAW=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');OUT=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';OUT.mkdir(exist_ok=True)
OLD=LAB/'outputs/dynamic-human-enerf-wholebody-059';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'observations.json').exists()
prompts=read(LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15/prompt-protocol.json')['prompts'];cams=read_cameras(RAW/'optimized');types=['person_dark_clothing','person_light_clothing'];old={r['camera']:r for r in read(OLD/'observations.json')['records']}
class BoundedRTMW(RTMPose):
 def __init__(self):
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  self.onnx_model='/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx';self.session=ort.InferenceSession(self.onnx_model,sess_options=opts,providers=['CPUExecutionProvider']);self.model_input_size=(288,384);self.mean=np.array([123.675,116.28,103.53]);self.std=np.array([58.395,57.12,57.375]);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=BoundedRTMW();frames=list(range(744,757));fitframes=frames[::2];valframes=frames[1::2]
save(OUT/'protocol.json',dict(status='fixed_before_sequence_inference',frames=frames,fit_frames=fitframes,validation_frames=valframes,training_cameras=sorted(old),excluded_cameras=['00','06','12'],
 model='Same059 RTMW-l384x288 133landmarks and fixed038 person boxes. No tracking based on heldout observations. CPU ONNX inference.',
 images='Native1920x1080 precomputed with original optimizedK/D castfloat32 and cv2.undistort, exactly039/048/050 officialloader convention; all15images750 match050bytewise. Originalfloat64K/R/T used for geometry. No image enhancement.',
 timing_hypothesis='Fit low-order continuous3D body trajectories and camera time offsets on even frames; compare fixed offsets against learned offsets on reserved odd frames. Camera01 defines zero offset. Offset unit is frame index, FPS not certified.',
 limits='Predicted keypoints can drift with occlusion or camera viewpoint; a timing estimate requires independent-time improvement and does not prove synchronized RGB.',goal_complete=False))
inputs={(r['camera'],r['frame']):r for r in read(OUT/'native-input/input-audit.json')['records']}
rows=[];start=time.monotonic();maxold=0
for c in sorted(old):
 cal=cams[c];K=np.array(cal['K']);D=np.array(cal['D']);mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
 for frame in frames:
  f=RAW/'images'/c/f'{frame:06d}.jpg';rgb=np.load(OUT/'native-input'/f'camera{c}-frame{frame:06d}.npz')['rgb'];bgr=np.ascontiguousarray(rgb[...,::-1]);assert bgr.shape==(1080,1920,3);detections=[]
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==inputs[c,frame]['rectified_sha256']
  if frame==750:assert np.array_equal(rgb,np.load(BASE/f'train-camera-{c}.npz')['rgb'])
  for pi,typ in enumerate(types):
   box=next(p['box'] for p in prompts[c] if p['label']==typ);xy,s=pose(bgr,bboxes=[box]);assert np.isfinite(xy).all() and np.isfinite(s).all()
   detections.append(dict(person=typ,box=box,keypoints=xy[0].tolist(),scores=s[0].tolist()))
   if frame==750:maxold=max(maxold,float(abs(xy[0]-np.array(old[c]['detections'][pi]['keypoints'])).max()))
  if frame in [744,750,756]:
   im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
   for det,col in zip(detections,[(255,100,20),(0,180,255)]):
    d.rectangle(det['box'],outline=col,width=2)
    for j,((x,y),score) in enumerate(zip(det['keypoints'][:23],det['scores'][:23])):
     if score<.4:continue
     d.ellipse([x-3,y-3,x+3,y+3],fill=col);d.text((x+3,y-8),str(j),fill=col)
   boxes=[d['box'] for d in detections];x0=max(0,min(b[0] for b in boxes)-40);y0=max(0,min(b[1] for b in boxes)-40);x1=min(1920,max(b[2] for b in boxes)+40);y1=min(1080,max(b[3] for b in boxes)+40)
   im.crop((x0,y0,x1,y1)).save(OUT/f'camera{c}-frame{frame:06d}-observations.png')
  rows.append(dict(camera=c,frame=frame,split='fit' if frame in fitframes else 'validation_time',source_sha256=sha(f),rectified_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),K=cal['K'],w2c=cal['w2c'],detections=detections))
 save(OUT/f'camera{c}-observations.json',dict(records=[r for r in rows if r['camera']==c]));save(OUT/'progress.json',dict(cameras_completed=sum(r['frame']==756 for r in rows),images=len(rows),elapsed_seconds=time.monotonic()-start));print('POSE_SEQUENCE',c,flush=True)
assert maxold<1e-4,maxold
save(OUT/'observations.json',dict(status='195native_training_images_390person_predictions_complete',records=rows,frame750_max_difference_from059_px=maxold,goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(dict(images=len(rows),old750_max_px=maxold,elapsed_seconds=time.monotonic()-start)))

"""Native training photographs -> separate person crops -> RTMW133 observations."""
from pathlib import Path
import json,time,hashlib,shutil,importlib.metadata
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageDraw
from rtmlib import RTMPose
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-wholebody-059';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame';MASK=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
assert not (OUT/'observations.json').exists()
prompts=read(MASK/'prompt-protocol.json')['prompts'];types=['person_dark_clothing','person_light_clothing']
modelpath='/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx'
# Keep the package preprocess, decode and inference path; bound the CPU session threads.
class BoundedRTMW(RTMPose):
 def __init__(self):
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  self.session=ort.InferenceSession(modelpath,sess_options=opts,providers=['CPUExecutionProvider'])
  self.onnx_model=modelpath;self.model_input_size=(288,384);self.mean=np.array([123.675,116.28,103.53]);self.std=np.array([58.395,57.12,57.375]);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=BoundedRTMW();save(OUT/'protocol.json',dict(status='fixed_before_observation_inference',frame=750,train_cameras=sorted(prompts),
 method='RTMW-l384x288 133wholebody; each previouslydefined person prompt box passedseparately,1.25affine crop padding. Official rtmlib BGR input/preprocessing/SimCC decoding.',
 input='Unmodified full nativeRGB in050 convertedRGBtoBGR for documented model API. No candidate mesh/heldout RGB used.',
 box_source='038/train15/prompt-protocol.json; fixed before prior GS controls. Boxes selectperson, no new coordinate annotation.',
 runtime=dict(rtmlib=importlib.metadata.version('rtmlib'),onnxruntime=ort.__version__,opencv=cv2.__version__,numpy=np.__version__,CPU_threads=4),
 dependencies='opencv-python-headless providescv2; duplicateopencv-python/contribpipnames intentionally not installed. Actualcv2 preprocessing verified.',
 keypoint_layout='COCO133:0..16body,17..22feet,23..90face68,91..111lefthand,112..132righthand',
 limits='Neural predictions, notGT. Behind-person detections may attach tofrontperson; visibility/confidence and geometryreview required.',goal_complete=False))
rows=[];start=time.monotonic()
for f in sorted(BASE.glob('train-camera-*.npz')):
 c=f.stem[-2:];assert c not in ['00','06','12'];z=np.load(f);rgb=z['rgb'];bgr=np.ascontiguousarray(rgb[...,::-1]);detections=[]
 for typ in types:
  box=next(p['box'] for p in prompts[c] if p['label']==typ)
  xy,score=pose(bgr,bboxes=[box]);assert xy.shape==(1,133,2) and score.shape==(1,133)
  assert np.isfinite(xy).all() and np.isfinite(score).all()
  detections.append(dict(person=typ,box=box,keypoints=xy[0].tolist(),scores=score[0].tolist()))
 image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
 for d,col in zip(detections,[(255,100,20),(0,180,255)]):
  draw.rectangle(d['box'],outline=col,width=2)
  for j,((x,y),s) in enumerate(zip(d['keypoints'],d['scores'])):
   if s<.4:continue
   rad=3 if j<23 else 1;draw.ellipse([x-rad,y-rad,x+rad,y+rad],fill=col)
   if j<23:draw.text((x+3,y-8),str(j),fill=col)
 yy,xx=np.where(z['matte']>.5);box=[max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(1920,int(xx.max())+26),min(1080,int(yy.max())+26)]
 image.crop(box).save(OUT/f'camera{c}-wholebody-observations.png')
 rows.append(dict(camera=c,input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),K=z['K'].tolist(),w2c=z['w2c'].tolist(),detections=detections))
 save(OUT/'progress.json',dict(cameras=len(rows),elapsed_seconds=time.monotonic()-start));print('RTMW',c,flush=True)
save(OUT/'observations.json',dict(status='15training_views_two_person133keypoints_inferred_review_pending',records=rows,goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name)

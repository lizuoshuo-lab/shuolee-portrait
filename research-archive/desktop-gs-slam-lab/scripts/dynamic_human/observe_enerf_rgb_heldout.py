from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,onnxruntime as ort
from PIL import Image,ImageDraw
from rtmlib import RTMPose
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'observations.json').exists();inputs=read(R/'input-audit.json')['records'];boxes=read(R/'person-boxes.json')['boxes']
class BoundedRTMW(RTMPose):
 def __init__(self):
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  self.onnx_model='/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx';self.session=ort.InferenceSession(self.onnx_model,sess_options=opts,providers=['CPUExecutionProvider']);self.model_input_size=(288,384);self.mean=np.array([123.675,116.28,103.53]);self.std=np.array([58.395,57.12,57.375]);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=BoundedRTMW();rows=[];start=time.monotonic()
for record in inputs:
 c,t=record['camera'],record['frame'];f=I/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(f)==record['input_sha256'];z=np.load(f);rgb=z['rgb'];bgr=np.ascontiguousarray(rgb[...,::-1]);detections=[]
 for person in [0,1]:
  xy,s=pose(np.ascontiguousarray(rgb),bboxes=[boxes[c][person]]);assert np.isfinite(xy).all() and np.isfinite(s).all();detections.append(dict(person=person,box=boxes[c][person],keypoints=xy[0].tolist(),scores=s[0].tolist()))
 if t in [744,750,756]:
  im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
  for d,color in zip(detections,[(255,110,20),(0,180,255)]):
   draw.rectangle(d['box'],outline=color,width=2)
   for j,((x,y),score) in enumerate(zip(d['keypoints'][:23],d['scores'][:23])):
    if score<.4:continue
    draw.ellipse([x-3,y-3,x+3,y+3],fill=color);draw.text((x+4,y-7),str(j),fill=color)
  im.crop(record['bbox']).save(R/f'camera{c}-frame{t:06d}-observations.png')
 rows.append(dict(camera=c,frame=t,split=record['split'],K=z['K'].tolist(),w2c=z['w2c'].tolist(),input_sha256=record['input_sha256'],detections=detections))
 if t==756:save(R/f'camera{c}-observations.json',dict(records=[r for r in rows if r['camera']==c]));print('HELDOUT_POSE',c,flush=True)
save(R/'observations.json',dict(status='39images78person_RTMW_predictions_complete',records=rows,model_sha256=sha(Path(pose.onnx_model)),elapsed_seconds=time.monotonic()-start,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)

"""Re-extract existing pose observations with archive-specified RGB normalization.

No Gaussian renderer or archived observations are modified. Official pipeline
contains stale crop-size defaults; ONNX input shape and decoder dimensions win.
"""
from pathlib import Path
import json,hashlib,time,zipfile,inspect,shutil
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rtmlib import RTMPose
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs,top_down_affine
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-pose-sequence-062';O=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'observations.json').exists()
model=Path('/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx');archive=model.with_suffix('.zip')
with zipfile.ZipFile(archive) as z:
 pipeline=json.loads(z.read('pipeline.json'));save(O/'official-pipeline.json',pipeline)
transforms=pipeline['pipeline']['tasks'][0]['transforms'];norm=next(t for t in transforms if t['type']=='Normalize')
assert norm['to_rgb'] is True
class Pose(RTMPose):
 def __init__(self):
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  self.onnx_model=str(model);self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider']);self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=Pose();shape=pose.session.get_inputs()[0].shape;assert shape[1:]==[3,384,288] and (shape[0]==1 or isinstance(shape[0],str)),shape
obs=read(S/'observations.json')['records'];ref=next(r for r in obs if r['camera']=='01' and r['frame']==750)
rgb=np.load(S/'native-input/camera01-frame000750.npz')['rgb'];bgr=np.ascontiguousarray(rgb[...,::-1]);box=ref['detections'][0]['box']
# Independent official path: start in OpenCV BGR, warp, convert to RGB, normalize.
center,scale=bbox_xyxy2cs(np.array(box),padding=1.25);warped,scale=top_down_affine((288,384),scale,center,bgr)
official=(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)-np.asarray(norm['mean']))/np.asarray(norm['std'])
correct,_,_=pose.preprocess(rgb,box);historical,_,_=pose.preprocess(bgr,box)
assert np.array_equal(correct,official)
oldxy,oldscore=pose(bgr,bboxes=[box]);xyerr=float(np.max(np.abs(oldxy[0]-np.asarray(ref['detections'][0]['keypoints']))));serr=float(np.max(np.abs(oldscore[0]-np.asarray(ref['detections'][0]['scores']))))
assert xyerr<1e-4 and serr<1e-4,(xyerr,serr)
audit=dict(status='official_RGB_input_exactly_reproduced_historical_BGR_coordinates_reproduced',official_archive_sha256=sha(archive),model_sha256=sha(model),onnx_input_shape=shape,archive_to_rgb=norm['to_rgb'],
 corrected_normalized_tensor_max_abs_difference_from_official=float(np.max(np.abs(correct-official))),historical_normalized_tensor_max_abs_difference_from_official=float(np.max(np.abs(historical-official))),
 historical_xy_reproduction_max_px=xyerr,historical_scores_reproduction_max=serr,
 implementation_paths={str(Path(inspect.getfile(RTMPose))):sha(Path(inspect.getfile(RTMPose)))},
 scope='Changes neural pose input only. Gaussian training always used original RGB. No retroactive rewrite of historic metrics. Coordinates change; predictive score is not calibrated probability.',
 crop_size_note='Archivepreprocessimage_size192x256isstale;ONNX1x3x384x288anddecoder288x384matchlocalruntime.',goal_complete=False)
save(O/'preprocessing-audit.json',audit)
save(O/'protocol.json',dict(status='fixed_before_corrected_full_sequence_inference',source_sha256=sha(S/'observations.json'),images=len(obs),fit_frames=list(range(744,757,2)),validation_frames=list(range(745,756,2)),
 cameras=sorted({r['camera'] for r in obs}),change='Only BGR->RGB at existing RTMW normalization input. Fixed boxes/model/CPU/intrinsics/nativeimages remain identical.',
 heldout='00,06,12 excluded; odd-time observations for diagnosis only, not later motion fitting.',
 fitting='No motion refit or Gaussian retrain in this script; corrected body must be evaluated before replacing067.',goal_complete=False))
rows=[];shifts=[];start=time.monotonic();font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
for r in obs:
 rgb=np.load(S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz")['rgb'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rectified_sha256']
 new=[]
 for person,d in enumerate(r['detections']):
  xy,score=pose(np.ascontiguousarray(rgb),bboxes=[d['box']]);xy=xy[0];score=score[0]
  assert xy.shape==(133,2) and score.shape==(133,) and np.isfinite(xy).all() and np.isfinite(score).all()
  new.append(dict(person=d['person'],box=d['box'],keypoints=xy.tolist(),scores=score.tolist()))
  shifts.append(dict(camera=r['camera'],frame=r['frame'],person=person,shift_px=np.linalg.norm(xy-np.asarray(d['keypoints']),axis=1).tolist()))
 nr={k:v for k,v in r.items() if k!='detections'};nr['detections']=new;rows.append(nr)
 if r['frame']==750:
  boxes=[d['box'] for d in r['detections']];bb=[max(0,min(b[0] for b in boxes)-20),max(0,min(b[1] for b in boxes)-20),min(1920,max(b[2] for b in boxes)+20),min(1080,max(b[3] for b in boxes)+20)]
  raw=Image.fromarray(rgb);ims=[raw.copy(),raw.copy()]
  for im,ds in zip(ims,[r['detections'],new]):
   draw=ImageDraw.Draw(im)
   for det,color in zip(ds,[(255,100,20),(0,180,255)]):
    for j,(x,y) in enumerate(np.asarray(det['keypoints'])[:23]):
     draw.ellipse([x-2,y-2,x+2,y+2],fill=color)
     draw.text((x+3,y-8),str(j),font=font,fill=color)
  ims=[raw.crop(bb)]+[im.crop(bb) for im in ims];w,h=ims[0].size
  sheet=Image.new('RGB',(w*3,h+32),'white');draw=ImageDraw.Draw(sheet)
  for i,(im,label) in enumerate(zip(ims,['NativeRGB','HistoricalBGRpose','CorrectRGBpose'])):sheet.paste(im,(w*i,32));draw.text((w*i+5,5),label,font=font,fill='black')
  sheet.save(O/f"camera{r['camera']}-body750.png")
 if r['frame']==756:
  save(O/'progress.json',dict(images=len(rows),cameras_complete=len(rows)//13,elapsed_seconds=time.monotonic()-start));print('RGB_POSE',r['camera'],len(rows),flush=True)
save(O/'observations.json',dict(status='195native_images_390RGBpose_predictions_complete',records=rows,goal_complete=False));save(O/'keypoint-shifts.json',dict(records=shifts))
out=dict(status='RGBpose_extraction_complete_motion_refit_not_started',images=len(rows),seconds=time.monotonic()-start,body_shift_median_px=float(np.median([x for r in shifts for x in r['shift_px'][:23]])),hand_shift_median_px=float(np.median([x for r in shifts for x in r['shift_px'][91:]])),source_unchanged=sha(S/'observations.json')==read(O/'protocol.json')['source_sha256'],goal_complete=False)
save(O/'summary.json',out);shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(out))

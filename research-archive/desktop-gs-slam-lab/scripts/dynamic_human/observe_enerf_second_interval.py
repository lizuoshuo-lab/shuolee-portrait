"""New-interval RGB pose predictions from photograph-reviewed person boxes."""
from pathlib import Path
import json, hashlib, time, zipfile, shutil
import numpy as np, cv2, onnxruntime as ort
from rtmlib import RTMPose
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
from PIL import Image, ImageDraw, ImageFont

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
R=L/'outputs/dynamic-human-enerf-second-interval-079'; O=R/'pose'; O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'observations.json').exists(), 'Preserve completed observations'
protocol=read(R/'protocol.json'); det=read(R/'box-detection/observations.json')['records']
lookup={(r['camera'],r['frame']):r for r in det}
types=['person_dark_clothing','person_light_clothing']; anchors=[720,726,732]
# Actual fifteen contact sheets were inspected, including duplicates and shadows.
# Values are detector candidate IDs, NOT identities inferred from detector order.
assign={
 '01':[[1,1,1],[0,0,0]], '02':[[0,1,1],[1,0,0]],
 '03':[[1,1,1],[0,0,0]], '04':[[1,1,1],[0,0,0]],
 '05':[[0,0,0],[1,1,1]], '07':[[0,0,0],[1,1,1]],
 '08':[[1,1,1],[0,0,0]], '09':[[None,1,1],[0,0,0]],
 '10':[[None,1,1],[0,0,0]], '11':[[None,None,1],[0,0,0]],
 '13':[[1,1,1],[0,0,0]], '14':[[1,None,None],[0,0,0]],
 '15':[[0,1,1],[1,0,0]], '16':[[1,1,None],[0,0,0]],
 '17':[[0,1,1],[1,0,0]]}
exclude_dark={'08','09','10','11','14','16'}
review=dict(status='45_actual_training_keyframe_photos_visually_reviewed',
 identities=types,anchor_frames=anchors,candidate_ids=assign,
 excluded_dark_person_cameras=sorted(exclude_dark),
 exclusion_reason='Dark person severely occluded, incomplete/missing boxes, or overlap makes predicted limbs ambiguous. Do not substitute wall-shadow detections or extrapolate absent boxes.',
 joint_exclusions={'person_dark_clothing':{'01':list(range(13,23)),'03':list(range(13,23))}},
 joint_reason='Dark lower legs/feet overlap the light person in01/03; disable both knees, ankles and feet throughout interval conservatively.',
 observations='Dark full-body candidates01/03 are distinct from duplicate upper-body detections. Wall-shadow false positives09/11 excluded. Light person identity is unambiguous in all15views; partial foot edges are covered by padded RTMW crop.',
 crop='Linearly interpolate each reviewed bbox between720/726/732; add12native pixels on all sides, clip image bounds; RTMW affine crop retains standard1.25padding. No manual landmark coordinate edits.',
 reviewed_images=[dict(camera=c,path=f'box-detection/camera{c}-keyframe-boxes.png',sha256=sha(R/'box-detection'/f'camera{c}-keyframe-boxes.png')) for c in protocol['training_cameras']],
 scope='Photograph inspection is identity/crop QC, not a GT skeleton or proof that all predicted joints are visible.',goal_complete=False)
save(R/'box-detection/identity-review.json',review)
save(O/'protocol.json',dict(status='fixed_before_new_interval_pose_inference',
 input_split='195known-camera images; even times only for fitting, odd times diagnostic. Other cameras00/06/12 excluded entirely.',
 model='Existing official RTMW-L 384x288 ONNX; CPU; RGB normalization; no model fine-tuning.',
 identity_review_sha256=sha(R/'box-detection/identity-review.json'),
 fitting='Confidence>=0.4 and per-time geometric consensus; explicit joint exclusions applied before fit. Predicted face/finger landmarks never SMPL body fitting targets.',
 goal_complete=False))
model=Path('/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx'); archive=model.with_suffix('.zip')
with zipfile.ZipFile(archive) as z:pipeline=json.loads(z.read('pipeline.json'))
norm=next(t for t in pipeline['pipeline']['tasks'][0]['transforms'] if t['type']=='Normalize');assert norm['to_rgb'] is True
class Pose(RTMPose):
 def __init__(self):
  opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
  self.onnx_model=str(model);self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
  self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=Pose();assert pose.session.get_inputs()[0].shape[1:]==[3,384,288]
def box_at(c,pi,t):
 boxes=np.array([lookup[c,f]['candidates'][idx]['box'] for f,idx in zip(anchors,assign[c][pi])])
 b=np.array([np.interp(t,anchors,boxes[:,i]) for i in range(4)])+[-12,-12,12,12]
 return np.clip(b,[0,0,0,0],[1919,1079,1919,1079]).tolist()
with np.load(R/'native-input/camera04-frame000726.npz') as z:probe=z['rgb']
box=box_at('04',0,726);bgr=np.ascontiguousarray(probe[...,::-1])
center,scale=bbox_xyxy2cs(np.array(box),padding=1.25);warped,_=top_down_affine((288,384),scale,center,bgr)
official=(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)-np.asarray(norm['mean']))/np.asarray(norm['std'])
actual,_,_=pose.preprocess(probe,box);assert np.array_equal(actual,official)
save(O/'preprocessing-audit.json',dict(status='RGB_tensor_exactly_matches_official_BGL_read_then_RGB_conversion',
 model_sha256=sha(model),archive_sha256=sha(archive),max_tensor_difference=float(abs(actual-official).max()),
 onnx_shape=pose.session.get_inputs()[0].shape,reference_camera='04',reference_frame=726))
inputs={(r['camera'],r['frame']):r for r in read(R/'input-audit.json')['records']}
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',13);rows=[];start=time.monotonic()
for c in protocol['training_cameras']:
 panels=[]
 for t in protocol['frames']:
  path=R/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(path)==inputs[c,t]['input_sha256']
  with np.load(path) as z:rgb=z['rgb'];K=z['K'];W=z['w2c'];bb=z['bbox']
  predictions=[]
  for pi,typ in enumerate(types):
   if pi==0 and c in exclude_dark:continue
   box=box_at(c,pi,t);xy,score=pose(np.ascontiguousarray(rgb),bboxes=[box]);xy=xy[0];score=score[0]
   assert xy.shape==(133,2) and score.shape==(133,) and np.isfinite(xy).all() and np.isfinite(score).all()
   disabled=list(range(13,23)) if pi==0 and c in {'01','03'} else []
   predictions.append(dict(person=typ,box=box,keypoints=xy.tolist(),scores=score.tolist(),disabled_fitting_indices=disabled))
  rows.append(dict(camera=c,frame=t,split=inputs[c,t]['split'],input_sha256=sha(path),K=K.tolist(),w2c=W.tolist(),rectified_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),detections=predictions))
  if t in anchors:
   raw=Image.fromarray(rgb);im=raw.copy();draw=ImageDraw.Draw(im)
   for d in predictions:
    col=(255,100,20) if d['person']==types[0] else (0,160,255)
    for j in range(5,23):
     if j in d['disabled_fitting_indices'] or d['scores'][j]<.4:continue
     x,y=d['keypoints'][j];draw.ellipse([x-3,y-3,x+3,y+3],fill=col);draw.text((x+4,y-10),str(j),font=font,fill=col,stroke_width=1,stroke_fill='white')
   raw=raw.crop(bb);im=im.crop(bb);panel=Image.new('RGB',(raw.width*2,raw.height+30),'white');panel.paste(raw,(0,30));panel.paste(im,(raw.width,30))
   ImageDraw.Draw(panel).text((5,5),f'Camera{c} frame{t}: original | RGB pose predictions (not GT)',font=font,fill='black');panel.save(O/f'camera{c}-frame{t:06d}-pose.png')
 save(O/'progress.json',dict(images=len(rows),predictions=sum(len(r['detections']) for r in rows),elapsed_seconds=time.monotonic()-start));print('SECOND_RGB_POSE',c,len(rows),flush=True)
save(O/'observations.json',dict(status='195native_images_312reviewed_identity_predictions_complete',records=rows,goal_complete=False))
save(O/'completion.json',dict(images=len(rows),predictions=sum(len(r['detections']) for r in rows),seconds=time.monotonic()-start,motion_fitted=False,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name)
print('SECOND_INTERVAL_RGB_OBSERVATIONS_COMPLETE',len(rows),flush=True)

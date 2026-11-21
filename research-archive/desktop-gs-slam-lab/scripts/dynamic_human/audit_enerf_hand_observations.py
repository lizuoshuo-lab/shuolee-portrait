"""Read-only hand observability study; no new pose labels or Gaussian training."""
from pathlib import Path
import json, hashlib, shutil, inspect, itertools, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rtmlib.visualization.skeleton.coco133 import coco133
from rtmlib.tools.pose_estimation.rtmpose import RTMPose
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
O=L/'outputs/dynamic-human-enerf-hand-observability-073'
S=L/'outputs/dynamic-human-enerf-pose-sequence-062'
O.mkdir(exist_ok=True); (O/'source-panels').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'summary.json').exists(), 'Keep completed diagnostics immutable.'
metadata=Path(inspect.getfile(inspect.getmodule(RTMPose)))
skel=Path('/opt/gs-slam-lab/wholebody-pose/venv/lib/python3.10/site-packages/rtmlib/visualization/skeleton/coco133.py')
assert coco133['keypoint_info'][91]['name']=='left_hand_root'
assert coco133['keypoint_info'][112]['name']=='right_hand_root'
assert coco133['keypoint_info'][111]['name']=='left_pinky_finger4'
assert coco133['keypoint_info'][132]['name']=='right_pinky_finger4'
obs=read(S/'observations.json')['records']; by={(r['camera'],r['frame']):r for r in obs}
cams=sorted({r['camera'] for r in obs}); assert len(obs)==195 and len(cams)==15
assert not set(cams)&{'00','06','12'}
motion=L/'outputs/dynamic-human-enerf-head-motion-066/motion-fit.npz'
z=np.load(motion); offsets=dict(zip(z['cameras'].astype(str),z['camera_offsets']))
assert abs(float(offsets['01']))<1e-7
save(O/'protocol.json', dict(status='fixed_before_hand_statistics_and_geometry',
 source=str(S/'observations.json'),source_sha256=sha(S/'observations.json'),
 metadata=dict(source=str(skel),sha256=sha(skel),preprocessor=str(metadata),preprocessor_sha256=sha(metadata)),
 layout={str(i):coco133['keypoint_info'][i]['name'] for i in [9,10]+list(range(91,133))},
 images='Existing195native1920x1080 rectified training-camera images. No sharpening, upscaling or new inference.',
 scores='RTMW SimCC maximum scores, not calibrated probabilities or visibility ground truth. Primary filter >=0.4.',
 source_review='Frame750, every15camera, both people, both hands:60native160x160 photographs alongside score-filtered predictions. Crop centers derived from predictions and body wrist; failures can therefore miscenter crops.',
 diagnostic='Report all780 hand observations and420 even-time observations separately. Measure prediction extent in native pixels and exact padded288x384 detector-input coordinates, duplicate wrist disagreement, and finger segment lengths.',
 geometry='At reference physical frame750, interpolate even-time2D predictions at nominal750-frozen_offset. Perjoint leave-one-camera-out robust DLT, all training pairs, score>=0.4,6px inlier gate, >=3support cameras, cheirality. No test cameras and no odd-time data enter triangulation. Positive result is prediction consistency, not accuracy.',
 timing_sha256=sha(motion), limits='Interpolation and fitted timing are approximations. Occluded hallucinations may be mutually consistent. No learned hand motion, SMPL-X/MANO, new Gaussian fitting or clarity acceptance.',goal_complete=False))
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
colors=[(237,90,40),(200,55,145),(45,140,220),(85,175,55),(175,90,210)]
rows=[]; panels=[]; start=time.monotonic()
for r in obs:
 for person,d in enumerate(r['detections']):
  xy=np.asarray(d['keypoints'],float); scores=np.asarray(d['scores'],float)
  box=np.asarray(d['box'],float); center,scale=bbox_xyxy2cs(box,padding=1.25)
  w,h=scale; scale=np.array([max(w,h*.75),max(h,w/.75)])
  factor=np.array([288,384])/scale
  for hand,base,wrist in [('left',91,9),('right',112,10)]:
   pts=xy[base:base+21]; conf=scores[base:base+21]; good=conf>=.4
   extent=np.ptp(pts,axis=0); segments=[]; input_segments=[]
   for f in range(5):
    chain=[0]+list(range(1+f*4,5+f*4))
    for i,j in zip(chain[:-1],chain[1:]):
     if good[i] and good[j]:
      segments.append(float(np.linalg.norm(pts[i]-pts[j])))
      input_segments.append(float(np.linalg.norm((pts[i]-pts[j])*factor)))
   rows.append(dict(camera=r['camera'],frame=r['frame'],split=r['split'],person=person,hand=hand,
    scores_median=float(np.median(conf)),scores_min=float(conf.min()),points_ge04=int(good.sum()),
    extent_native_px=extent.tolist(),extent_detector_px=(extent*factor).tolist(),native_px_per_detector_px=(1/factor).tolist(),
    duplicate_wrist_distance_px=float(np.linalg.norm(xy[wrist]-pts[0])),both_wrist_scores_ge04=bool(conf[0]>=.4 and scores[wrist]>=.4),
    confident_extent_native_px=np.ptp(pts[good],axis=0).tolist() if good.sum()>1 else None,
    finger_segments_native_px=segments,finger_segments_detector_px=input_segments))
 if r['frame']!=750:continue
 native=S/'native-input'/f"camera{r['camera']}-frame000750.npz"
 rgb=np.load(native)['rgb'];assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==r['rectified_sha256']
 image=Image.fromarray(rgb); sheet=Image.new('RGB',(700,434),'white');draw=ImageDraw.Draw(sheet)
 draw.text((10,3),f"Camera{r['camera']} frame750 | Native photo / RTMW prediction >=0.4 | 1 source pixel = 1 panel pixel",font=font,fill='black')
 for person,d in enumerate(r['detections']):
  xy=np.asarray(d['keypoints']); scores=np.asarray(d['scores'])
  for hi,(hand,base,wrist) in enumerate([('left',91,9),('right',112,10)]):
   pts=xy[base:base+21]; conf=scores[base:base+21]
   mid=np.median(np.concatenate([pts,xy[wrist:wrist+1]],axis=0),axis=0)
   cx,cy=mid.round().astype(int); x0=int(np.clip(cx-80,0,1760));y0=int(np.clip(cy-80,0,920));crop=[x0,y0,x0+160,y0+160]
   raw=image.crop(crop);overlay=raw.copy();dr=ImageDraw.Draw(overlay)
   for f in range(5):
    chain=[0]+list(range(1+f*4,5+f*4))
    for i,j in zip(chain[:-1],chain[1:]):
     if conf[i]>=.4 and conf[j]>=.4:dr.line([tuple(pts[i]-[x0,y0]),tuple(pts[j]-[x0,y0])],fill=colors[f],width=1)
   for i,p in enumerate(pts):
    if conf[i]<.4:continue
    px,py=p-[x0,y0];dr.ellipse([px-1,py-1,px+1,py+1],fill=(255,50,30))
   px,py=xy[wrist]-[x0,y0];dr.rectangle([px-2,py-2,px+2,py+2],outline=(0,225,255),width=1)
   x=10+hi*350;y=35+person*195
   draw.text((x,y-15),f"{'dark' if person==0 else 'light'} {hand}; {int((conf>=.4).sum())}/21; wrist gap {np.linalg.norm(pts[0]-xy[wrist]):.1f}px",font=font,fill='black')
   sheet.paste(raw,(x,y));sheet.paste(overlay,(x+170,y))
   panels.append(dict(camera=r['camera'],person=person,hand=hand,crop=crop,source_sha256=r['rectified_sha256']))
 sheet.save(O/'source-panels'/f"camera{r['camera']}-hands.png")

save(O/'hand-statistics.json',dict(records=rows))
save(O/'source-panel-index.json',dict(records=panels,scale='1x native pixels; colored landmarks are predictions, not truth'))

# Diagnose a single physical time with even-time interpolation; exclude each
# target camera from all hypotheses and inlier refinement for that prediction.
P={c:np.asarray(by[c,750]['K'])@np.asarray(by[c,750]['w2c'])[:3] for c in cams}
W={c:np.asarray(by[c,750]['w2c']) for c in cams}
def project(X,c):
 q=P[c]@np.r_[X,1.];return q[:2]/q[2],float((W[c]@np.r_[X,1.])[2])
def dlt(points,weights,cs):
 a=[]
 for uv,w,c in zip(points,weights,cs):
  a.extend([(uv[0]*P[c][2]-P[c][0])*np.sqrt(w),(uv[1]*P[c][2]-P[c][1])*np.sqrt(w)])
 _,_,v=np.linalg.svd(np.asarray(a));v=v[-1]
 if abs(v[3])<1e-10:return None
 return v[:3]/v[3]
interpolated={}
for c in cams:
 nominal=750-float(offsets[c]);lo=2*int(np.floor(nominal/2));hi=lo+2
 assert 744<=lo<hi<=756
 fraction=(nominal-lo)/2
 ds=[]
 for person in [0,1]:
  a=by[c,lo]['detections'][person];b=by[c,hi]['detections'][person]
  ds.append((np.asarray(a['keypoints'])*(1-fraction)+np.asarray(b['keypoints'])*fraction,
             np.minimum(a['scores'],b['scores'])))
 interpolated[c]=ds
save(O/'reference-interpolation.json',dict(physical_frame=750,even_source_only=True,offsets={c:float(offsets[c]) for c in cams},records={c:[dict(points=p.tolist(),scores=s.tolist()) for p,s in ds] for c,ds in interpolated.items()}))
geometry=[]
for person in [0,1]:
 for joint in [9,10]+list(range(91,133)):
  valid=[c for c in cams if interpolated[c][person][1][joint]>=.4]
  for target in valid:
   cs=[c for c in valid if c!=target];best=None
   for a,b in itertools.combinations(cs,2):
    X=dlt([interpolated[c][person][0][joint] for c in [a,b]],[interpolated[c][person][1][joint] for c in [a,b]],[a,b])
    if X is None:continue
    err=[];depth=[]
    for c in cs:
     uv,z=project(X,c);err.append(np.linalg.norm(uv-interpolated[c][person][0][joint]));depth.append(z)
    err=np.asarray(err);inside=(err<6)&(np.asarray(depth)>0);n=int(inside.sum())
    if n<3:continue
    rank=(n,-float(np.minimum(err,6).sum()))
    if best is None or rank>best[0]:best=(rank,inside)
   row=dict(person=person,joint=joint,name=coco133['keypoint_info'][joint]['name'],camera=target,available_other_cameras=len(cs),valid=False)
   if best is not None:
    ins=[c for c,s in zip(cs,best[1]) if s]
    X=dlt([interpolated[c][person][0][joint] for c in ins],[interpolated[c][person][1][joint] for c in ins],ins)
    uv,z=project(X,target);e=float(np.linalg.norm(uv-interpolated[target][person][0][joint]))
    row.update(valid=bool(z>0 and np.isfinite(e)),support_cameras=ins,heldout_reprojection_px=e,target_positive_depth=bool(z>0))
   geometry.append(row)
save(O/'leave-one-camera-out.json',dict(status='reference_time_prediction_consistency_only',records=geometry))
def quant(v):
 v=np.asarray(v,float)
 return dict(n=int(v.size),p10=float(np.percentile(v,10)),median=float(np.median(v)),p90=float(np.percentile(v,90)),max=float(v.max())) if len(v) else dict(n=0)
stats={}
for split in ['all','fit','validation_time']:
 rs=[r for r in rows if split=='all' or r['split']==split]
 segments=[v for r in rs for v in r['finger_segments_detector_px']]
 stats[split]=dict(hand_observations=len(rs),scores_median=quant([r['scores_median'] for r in rs]),
  points_ge04=quant([r['points_ge04'] for r in rs]),native_extent_long_axis_px=quant([max(r['extent_native_px']) for r in rs]),
  detector_extent_long_axis_px=quant([max(r['extent_detector_px']) for r in rs]),
  confident_finger_segment_detector_px=quant(segments),
  confident_finger_segments_below2_detector_pixels_fraction=float(np.mean(np.asarray(segments)<2)),
  duplicate_wrist_distance_both_confident_px=quant([r['duplicate_wrist_distance_px'] for r in rs if r['both_wrist_scores_ge04']]))
geo={}
for category,indices in [('body_wrist',[9,10]),('hand_root',[91,112]),('finger_joints',[i for i in range(91,133) if i not in [91,112]])]:
 rs=[r for r in geometry if r['joint'] in indices];v=[r['heldout_reprojection_px'] for r in rs if r['valid']]
 geo[category]=dict(confident_target_count=len(rs),valid_count=len(v),reprojection_px=quant(v),under6_px_count=sum(x<6 for x in v),invalid_count=len(rs)-len(v))
out=dict(status='hand_input_statistics_and_cross_view_diagnostic_complete_visual_review_pending',statistics=stats,geometry=geo,source_panels=15,source_hand_crops=60,elapsed_seconds=time.monotonic()-start,no_training_or_label_updates=True,goal_complete=False)
save(O/'summary.json',out);shutil.copy2(__file__,O/Path(__file__).name)
print(json.dumps(out,indent=2))

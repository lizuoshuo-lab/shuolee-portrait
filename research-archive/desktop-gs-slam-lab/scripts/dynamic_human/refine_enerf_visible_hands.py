"""Pilot hand-specific observations on source-reviewed, visible training hands."""
from pathlib import Path
import json,hashlib,time,shutil,argparse
import numpy as np, onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rtmlib import RTMPose
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-enerf-hand-observability-073';S=L/'outputs/dynamic-human-enerf-pose-sequence-062'
args=argparse.ArgumentParser();args.add_argument('--rgb',action='store_true');args=args.parse_args()
P=O/('hand-pilot-rgb' if args.rgb else 'hand-pilot');P.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (P/'observations.json').exists()
obs=read(S/'observations.json')['records'];by={r['camera']:r for r in obs if r['frame']==750}
selected={0:['01','03','04','08','09','10','11','14','16'],1:['01','02','03','05','08','09','10','11','13','14','15','16','17']}
save(O/'source-visual-review.json',dict(status='all15frame750panels_60raw_and_overlay_hand_crops_actually_reviewed',
 files=[f'source-panels/camera{c}-hands.png' for c in sorted(by)],
 observations_zh=['全身模型在被玩偶遮住的手上仍输出完整21个点。','机位02、13、15中暗衣人物的预测经常对应浅衣人物；机位07有交叉手部归属歧义。','机位04浅衣人物右手预测落在衣服附近；不能作为可见手指约束。','机位17暗衣人物左右手预测接近同一只可见手；不纳入先导。','可见右手在若干机位保留手掌与手指轮廓，可用于局部检测候选；不代表每个指尖都可直接辨认。'],
 pilot_source_only_selection=selected,selection_scope='Onlyframe750;notautomaticallyvalidforothertimes. This is qualitative observability filtering, nothandpointGT.',
 rejected='Allleft hands excluded from initial pilot due to toy/contact/occlusion. Right-hand views omitted if identity or visibility ambiguous.',goal_complete=False))
model=Path('/mnt/d/GS-SLAM-Data/models/rtmhand/rtmpose-m-hand5-256x256.onnx')
acq=read(O/'hand-model-acquisition.json');assert sha(model)==acq['model_sha256']
save(P/'protocol.json',dict(status='fixed_before_pilot_inference',model_sha256=sha(model),frame=750,selected=selected,color_order='official_RGB' if args.rgb else 'historical_BRG_package_example_path_mismatch_found_afterward',
 crop='Nativehandboxderivedfromoldhand21predictionsandbodywrist; square side max(96px,1.4extent), capped160px. OfficialRTMPoseadds1.25paddingandresizesonlydetectorinputto256square. OriginalRGBandGaussiantrainingunchanged.',
 purpose='Comparewholebodyvsdedicatedhandobservationson22visibletraininghands. Noappearanceoptimization, noheldoutcameras, nooddframes, noGTclaim.',goal_complete=False))
class HandPose(RTMPose):
 def __init__(self):
  opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
  self.onnx_model=str(model);self.session=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider']);self.model_input_size=(256,256);self.mean=np.array([123.675,116.28,103.53]);self.std=np.array([58.395,57.12,57.375]);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=HandPose();rows=[];start=time.monotonic()
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);colors=[(237,90,40),(200,55,145),(45,140,220),(85,175,55),(175,90,210)]
def overlay(raw,xy,s,crop):
 im=raw.copy();d=ImageDraw.Draw(im);xy=xy-np.array(crop[:2])
 for f in range(5):
  chain=[0]+list(range(1+4*f,5+4*f))
  for i,j in zip(chain[:-1],chain[1:]):
   if s[i]>=.4 and s[j]>=.4:d.line([tuple(xy[i]),tuple(xy[j])],fill=colors[f],width=1)
 for i,(x,y) in enumerate(xy):
  if s[i]>=.4:d.ellipse([x-1,y-1,x+1,y+1],fill=(255,50,30))
 return im
for c in sorted(set(selected[0]+selected[1])):
 r=by[c];rgb=np.load(S/'native-input'/f'camera{c}-frame000750.npz')['rgb'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rectified_sha256']
 selected_people=[person for person in [0,1] if c in selected[person]]
 sheet=Image.new('RGB',(520,35+len(selected_people)*200),'white');d=ImageDraw.Draw(sheet)
 d.text((10,3),f'Camera{c} frame750 | Native / Whole-body / Hand-specific',font=font,fill='black')
 for row,person in enumerate(selected_people):
  det=r['detections'][person];allxy=np.asarray(det['keypoints']);old=allxy[112:133];s=np.asarray(det['scores'])[112:133]
  pts=np.concatenate([old,allxy[10:11]]);center=(pts.min(0)+pts.max(0))/2;side=float(np.clip(np.ptp(pts,axis=0).max()*1.4,96,160));box=np.r_[center-side/2,center+side/2]
  detector_rgb=rgb if args.rgb else rgb[...,::-1]
  xy,score=pose(np.ascontiguousarray(detector_rgb),bboxes=[box.tolist()]);xy=xy[0];score=score[0];assert xy.shape==(21,2) and np.isfinite(xy).all()
  # Compare in unchanged native coordinate space; detector resizing is not
  # applied to any displayed photograph or reconstruction target.
  x0=int(np.clip(round(center[0])-80,0,1760));y0=int(np.clip(round(center[1])-80,0,920));crop=[x0,y0,x0+160,y0+160]
  raw=Image.fromarray(rgb).crop(crop);y=50+row*200
  d.text((10,y-18),f"{'dark' if person==0 else 'light'} right; detector box {side:.0f}px; native comparison",font=font,fill='black')
  for i,im in enumerate([raw,overlay(raw,old,s,crop),overlay(raw,xy,score,crop)]):sheet.paste(im,(10+170*i,y))
  rows.append(dict(camera=c,frame=750,person=person,hand='right',source_sha256=r['rectified_sha256'],box=box.tolist(),panel_crop=crop,
    keypoints=xy.tolist(),scores=score.tolist(),old_keypoints=old.tolist(),old_scores=s.tolist(),median_shift_px=float(np.median(np.linalg.norm(xy-old,axis=1)))))
 sheet.save(P/f'camera{c}-pilot.png')
save(P/'observations.json',dict(status='22visiblehand_candidates_inferred_review_pending',records=rows,elapsed_seconds=time.monotonic()-start,goal_complete=False));shutil.copy2(__file__,P/Path(__file__).name)
print(json.dumps(dict(hands=len(rows),cameras=len(set(r['camera'] for r in rows)),seconds=time.monotonic()-start)))

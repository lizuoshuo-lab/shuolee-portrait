"""Freeze a camera-heldout appearance evaluation before fitting new clock offsets."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from audit_enerf_input import read_cameras
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-heldout-cameras-068';R.mkdir(exist_ok=True)
RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');BASE=L/'outputs/dynamic-human-enerf-articulated-control-064';H=L/'outputs/dynamic-human-enerf-head-motion-066';G=L/'outputs/dynamic-human-enerf-residual-graph-067'
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'input-audit.json').exists();cams=['00','06','12'];frames=list(range(744,757));cal=read_cameras(RAW/'optimized')
train=json.loads((BASE/'input-audit.json').read_text())['records'];assert not set(cams)&{r['camera'] for r in train}
save(R/'protocol.json',dict(status='fixed_before_new_camera_inference_and_any_render',cameras=cams,frames=frames,clock_fit_frames=frames[::2],appearance_evaluation_frames=frames[1::2],
 scope='3cameras absent from064-067appearance, initialization and pose fitting. These camera IDs/images appeared in older development work: this is camera-heldout appearance evaluation for frozen066/067, not untouched finalbenchmark.',
 clock='Fit only oneoffsetpernewcamera against frozen0623Dbodyfoot trajectories using even-time RTMWpredictions; bound[-2,2]frameindices. No3Dtrajectory/body/graph/appearance updates. Oddtime keypoints andRGB are evaluation only.',
 gate='Atleast4even-time scores>=0.4 and zerooffset median compatibility<=40px percamera/person/bodyfootjoint, frozen before odd-time evaluation. Robust softL1scale3px and offsetresidual0.1*delta. Also fit eachperson independently to expose disagreement.',
 branches=['066_head_only_with_estimated_clocks','067_graph_with_estimated_clocks','067_graph_with_zero_clocks'],
 fixed_visual_review='All3cameras xframes745,751,755 fullperson, plus two projected forearm boxes perimage. Also worstgraph-vs-head regression. No sharpening or resizing of sourcepixels.',
 checkpoints={'head':sha(H/'head_only/resume.pt'),'graph':sha(G/'residual_graph/resume.pt')},body_motion_sha256=sha(H/'motion-fit.npz'),trajectory_sha256=sha(L/'outputs/dynamic-human-enerf-pose-sequence-062/estimated_offsets-fit.npz'),
 qualification='Releasedmattes and predictedlandmarks notGT; unknown shutter/exposure/cameraappearance effects remain. Foreground evaluation, backgroundnottrained; no largescenecompletion.',goal_complete=False))
z=np.load(H/'capture-affines.npz');select=[i for i,(c,t) in enumerate(zip(z['camera'],z['frame'])) if c=='01' and t in [744,750,756]];verts=z['vertices'][select].reshape(3,2,6890,3);boxes={}
for c in cams:
 K=np.array(cal[c]['K']);W=np.array(cal[c]['w2c']);P=K@W[:3];boxes[c]=[]
 for person in [0,1]:
  X=verts[:,person].reshape(-1,3);q=X@P[:,:3].T+P[:,3];assert np.min(q[:,2])>0;uv=q[:,:2]/q[:,2:];lo=uv.min(0);hi=uv.max(0);pad=(hi-lo)*np.array([.12,.04])+20
  box=np.r_[np.maximum([0,0],np.floor(lo-pad)),np.minimum([1920,1080],np.ceil(hi+pad))].astype(int).tolist();boxes[c].append(box)
save(R/'person-boxes.json',dict(source='Union of066camera01physical744750756bodymesh projected through provided newcamera calibration; padded12percentwidth4percentheight+20px. No newcamera RGB used to derive boxes.',boxes=boxes))
rows=[];IN=R/'native-input';IN.mkdir(exist_ok=True)
for c in cams:
 K=np.array(cal[c]['K']);D=np.array(cal[c]['D']);W=np.array(cal[c]['w2c']);mx,my=cv2.initUndistortRectifyMap(K.astype(np.float32),D.astype(np.float32),None,K.astype(np.float32),(1920,1080),cv2.CV_32FC1);valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
 for t in frames:
  a=RAW/'images'/c/f'{t:06d}.jpg';b=RAW/'bgmtv2'/c/f'{t:06d}.jpg';rgb=cv2.undistort(cv2.imread(str(a)),K.astype(np.float32),D.astype(np.float32))[...,::-1].copy();matte=cv2.undistort(cv2.imread(str(b),0).astype(np.float32)/255,K.astype(np.float32),D.astype(np.float32));assert rgb.shape==(1080,1920,3)
  ys,xs=np.where(valid&(matte>.5));bbox=[max(0,int(xs.min())-32),max(0,int(ys.min())-32),min(1920,int(xs.max())+33),min(1080,int(ys.max())+33)]
  f=IN/f'camera{c}-frame{t:06d}.npz';np.savez_compressed(f,rgb=rgb,matte=matte,valid=valid,K=K,w2c=W,bbox=np.array(bbox))
  rows.append(dict(camera=c,frame=t,split='clock_calibration' if t%2==0 else 'heldout_appearance_time',source_rgb_sha256=sha(a),source_matte_sha256=sha(b),rectified_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),input_sha256=sha(f),bbox=bbox))
 print('HELDOUT_INPUT',c,flush=True)
save(R/'input-audit.json',dict(status='39native_newcamera_inputs21clockcalibration18appearance_evaluation',records=rows,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)

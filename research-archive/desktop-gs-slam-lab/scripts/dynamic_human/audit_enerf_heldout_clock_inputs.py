from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from scipy.optimize import minimize_scalar
from audit_enerf_input import read_cameras
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-heldout-cameras-068';RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
protocol=read(R/'protocol.json');cal=read_cameras(RAW/'optimized');inputs={(r['camera'],r['frame']):r for r in read(R/'input-audit.json')['records']};observ=read(R/'observations.json')['records'];assert len(observ)==39
for r in observ:
 c,t=r['camera'],r['frame'];row=inputs[c,t];p=R/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(p)==row['input_sha256']==r['input_sha256'];z=np.load(p);K=np.array(cal[c]['K']);D=np.array(cal[c]['D']);assert np.array_equal(z['K'],r['K']) and np.array_equal(z['w2c'],r['w2c'])
 f=RAW/'images'/c/f'{t:06d}.jpg';assert sha(f)==row['source_rgb_sha256'];rgb=cv2.undistort(cv2.imread(str(f)),K.astype(np.float32),D.astype(np.float32))[...,::-1].copy();assert np.array_equal(rgb,z['rgb']);assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rectified_rgb_sha256'];assert [d['person'] for d in r['detections']]==[0,1]
 for d in r['detections']:assert np.array(d['keypoints']).shape==(133,2) and np.isfinite(d['keypoints']).all() and np.isfinite(d['scores']).all()
support=read(R/'timing-support.json');fit=support['fit_observations'];assert fit and all(r['frame']%2==0 and r['camera'] in ['00','06','12'] for r in fit)
source=L/'outputs/dynamic-human-enerf-pose-sequence-062/estimated_offsets-fit.npz';assert sha(source)==protocol['trajectory_sha256'];co=np.load(source)['coefficients'];gate={(r['camera'],r['joint']) for r in support['gates'] if r['accepted']};timing=read(R/'timing-summary.json');audits=[]
for c in ['00','06','12']:
 rows=[]
 for r in observ:
  if r['camera']!=c or r['frame']%2:continue
  for d in r['detections']:
   for j in range(5,23):
    joint=d['person']*18+j-5
    if (c,joint) in gate and d['scores'][j]>=.4:rows.append((r['frame'],joint,np.array(d['keypoints'][j]),np.array(r['K'])@np.array(r['w2c'])[:3],min(1,d['scores'][j])))
 assert len(rows)==sum(r['camera']==c for r in fit)
 def cost(delta):
  total=9*(np.sqrt(1+(.1*delta/3)**2)-1)
  for t,j,xy,P,w in rows:
   tau=(t+delta-750)/6;X=np.polynomial.polynomial.polyval(tau,co[j]);q=P@np.r_[X,1.];res=(q[:2]/q[2]-xy)*np.sqrt(w);total+=np.sum(9*(np.sqrt(1+(res/3)**2)-1))
  return float(total)
 grid=np.linspace(-2,2,81);values=np.array([cost(x) for x in grid]);i=int(values.argmin());low=grid[max(0,i-1)];high=grid[min(80,i+1)];result=minimize_scalar(cost,bounds=(low,high),method='bounded',options={'xatol':1e-8});delta=timing['offsets_frames'][c];diff=abs(delta-result.x);assert diff<1e-4,(c,diff)
 audits.append(dict(camera=c,fit_observations=len(rows),recorded_offset=delta,independent_offset=float(result.x),difference_frames=float(diff),fit_objective=cost(delta)))
save(R/'clock-input-independent-audit.json',dict(status='39original_native_images78pose_records_even_only_clockfit_and_independent1Doptima_verified',images=39,calibration_images=21,appearance_evaluation_images=18,pose_predictions=78,offset_audits=audits,appearance_or_body_updates=0,qualification='Confirms computation and split; not true synchronization or GT keypoints.',goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(audits))

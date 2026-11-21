"""Independent projection/crop checks and near-time04/17 skeletal consistency."""
from pathlib import Path
import json,hashlib
import numpy as np,cv2
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-prior-096';S=L/'outputs/dynamic-human-enerf-second-interval-079'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
obs=read(O/'observations.json')['records'];manifest=read(O/'input-manifest.json')['records'];by={(r['camera'],r['frame']):r for r in manifest};assert len(obs)==105
maxerr=0.;rows=[];preds={}
for r in obs:
 key=r['camera'],r['frame'];z=np.load(O/r['file']);preds[key]=z
 assert hashlib.sha256((O/r['file']).read_bytes()).hexdigest()==r['sha256']
 for k in range(3):
  J=z['pred_keypoints_3d'][k].astype(np.float64)+z['pred_cam_t'][k].astype(np.float64);xy=J[:,:2]/J[:,2:]*5000/256*z['box_sizes'][k]+z['box_centers'][k];error=float(abs(xy-z['source_keypoints_2d'][k]).max());assert error<1e-3;maxerr=max(maxerr,error)
  rotations=np.concatenate([z['mano_global_orient'][k],z['mano_hand_pose'][k]],axis=0);assert np.max(abs(rotations@rotations.transpose(0,2,1)-np.eye(3)))<1e-5 and np.allclose(np.linalg.det(rotations),1,atol=1e-5)
 spread=np.max(np.linalg.norm(z['source_keypoints_2d']-z['source_keypoints_2d'][1:2],axis=-1),axis=0);assert abs(float(np.median(spread))-r['crop_keypoint_spread_median_px'])<1e-8
 rows.append(dict(camera=key[0],frame=key[1],median_crop_variation_px=float(np.median(spread)),p95_crop_variation_px=float(np.quantile(spread,.95))))
times=np.load(S/'heldout/capture-affines-estimated.npz');tm={(str(c),int(f)):float(t) for c,f,t in zip(times['camera'],times['frame'],times['capture_time'])};pairs=[]
for t in range(720,733,2):
 a,b=preds['04',t],preds['17',t];wa,wb=a['w2c'],b['w2c'];R=wb[:3,:3]@wa[:3,:3].T;v=wb[:3,3]-R@wa[:3,3];cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);F=np.linalg.inv(b['K']).T@cross@R@np.linalg.inv(a['K']);P=a['K']@wa[:3];Q=b['K']@wb[:3];ca=-wa[:3,:3].T@wa[:3,3];cb=-wb[:3,:3].T@wb[:3,3]
 # Independently reconstruct F using the projective camera center.
 _,_,vh=np.linalg.svd(P);e=Q@vh[-1];E=np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]]);Fp=E@Q@np.linalg.pinv(P);assert min(abs(F/np.linalg.norm(F)-Fp/np.linalg.norm(Fp)).max(),abs(F/np.linalg.norm(F)+Fp/np.linalg.norm(Fp)).max())<1e-9
 pred=[]
 for k in range(3):
  x=a['source_keypoints_2d'][k].astype(np.float64);y=b['source_keypoints_2d'][k].astype(np.float64);xx=np.c_[x,np.ones(21)];yy=np.c_[y,np.ones(21)];lx=yy@F;ly=xx@F.T;err=abs(np.sum(yy*ly,axis=1))/np.sqrt(np.sum(lx[:,:2]**2+ly[:,:2]**2,axis=1));X=cv2.triangulatePoints(P,Q,x.T,y.T);assert np.isfinite(X).all() and abs(X[3]).min()>1e-10;X=(X[:3]/X[3]).T;XA=np.c_[X,np.ones(21)]@P.T;XB=np.c_[X,np.ones(21)]@Q.T;reprojection=np.stack([np.linalg.norm(XA[:,:2]/XA[:,2:]-x,axis=-1),np.linalg.norm(XB[:,:2]/XB[:,2:]-y,axis=-1)],1);ra=X-ca;rb=X-cb;angles=np.degrees(np.arccos(np.clip(np.sum(ra*rb,axis=-1)/np.linalg.norm(ra,axis=-1)/np.linalg.norm(rb,axis=-1),-1,1)));lengths=[]
  for finger in range(5):
   chain=[0]+list(range(1+4*finger,5+4*finger));lengths.extend(np.linalg.norm(np.diff(X[chain],axis=0),axis=-1).tolist())
  pred.append(dict(crop_variant=k,sampson_px=err.tolist(),median_sampson_px=float(np.median(err)),p95_sampson_px=float(np.quantile(err,.95)),reprojection_px=reprojection.tolist(),positive_depth_joints=int(((XA[:,2]>0)&(XB[:,2]>0)).sum()),triangulation_angle_degrees=angles.tolist(),joint_positions=X.tolist(),bone_lengths_world_units=lengths))
 pairs.append(dict(cameras=['04','17'],frame=t,estimated_capture_times=[tm['04',t],tm['17',t]],estimated_time_difference_frames=abs(tm['04',t]-tm['17',t]),baseline_world_units=float(np.linalg.norm(ca-cb)),variants=pred))
save(O/'projection-audit.json',dict(status='105images315crop_projections_rotation_matrices_and_metadata_verified',maximum_projection_error_px=maxerr,MANO_buffer_differences=[r for r in read(O/'network-load.json')['MANO_buffer_comparison'] if not r['exact']],goal_complete=False))
save(O/'crop-robustness.json',dict(records=rows,worst=sorted(rows,key=lambda r:r['median_crop_variation_px'],reverse=True)[:10],scope='Crop perturbation is not an independent view or confidence calibration.',goal_complete=False))
save(O/'near-time-geometry.json',dict(status='seven_training_time_pairs_evaluated_no_GT_or_labels_accepted',records=pairs,scope='Frozen estimated camera times, not verified synchronized exposures.04/17 see different surfaces; semantic joints are inferred, often occluded. Small epipolar/reprojection residual alone cannot validate visible fingers or depth.',goal_complete=False))
print(json.dumps(dict(max_projection_error=maxerr,pair_median_px=[r['variants'][1]['median_sampson_px'] for r in pairs],worst_crop=sorted(rows,key=lambda r:r['median_crop_variation_px'],reverse=True)[:3])),flush=True)

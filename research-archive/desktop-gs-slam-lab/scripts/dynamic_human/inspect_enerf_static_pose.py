"""Training-only PnP diagnostic in the actual032 world, original cameras preserved."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image
from audit_enerf_input import read_cameras,digest
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-matte-background-034';seed=LAB/'outputs/dynamic-human-enerf-background-032';root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
assert not (out/'static-pose-diagnostic.json').exists()
cv2.setNumThreads(8);cv2.setRNGSeed(77);cal=read_cameras(root/'optimized')['08'];K=np.array(cal['K']);w2c=np.array(cal['w2c']);xyz=np.load(seed/'background-init.npz')['points'].astype(np.float64)
tracks=json.loads((seed/'triangulation.json').read_text())['tracks'];mapping={o['feature']:i for i,t in enumerate(tracks) for o in t['observations'] if o['camera']=='08'}
features=np.load(seed/'features-08.npz');db=features['descriptors'];sift=cv2.SIFT_create(nfeatures=10000);rows=[]
for t in [740,750,760]:
    rgb=np.array(Image.open(out/f'camera08-frame{t:06d}-source.png').convert('RGB'));m=np.array(Image.open(out/f'camera08-frame{t:06d}-matte.png').convert('L'));mask=cv2.erode((m<3).astype('uint8'),np.ones((11,11),np.uint8));mask[:20]=0;mask[820:]=0;mask[:,:20]=0;mask[:,1900:]=0
    kp,ds=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask*255);ys=np.array([k.pt for k in kp]);matcher=cv2.BFMatcher(cv2.NORM_L2);ab=matcher.knnMatch(db,ds,k=2);ba=matcher.knnMatch(ds,db,k=2);rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.7*n.distance};matches=[(mapping[m.queryIdx],m.trainIdx) for m,n in ab if m.queryIdx in mapping and m.distance<.7*n.distance and rev.get(m.trainIdx)==m.queryIdx]
    assert len(matches)>100
    X=xyz[[i for i,j in matches]];y=ys[[j for i,j in matches]].astype('float64');check=np.arange(len(X))%5==0;rv=cv2.Rodrigues(w2c[:3,:3])[0];tv=w2c[:3,3,None].copy()
    original=cv2.projectPoints(X,rv,tv,K,None)[0].reshape(-1,2);ok,r,tvec,inliers=cv2.solvePnPRansac(X[~check],y[~check],K,None,rv.copy(),tv.copy(),True,iterationsCount=2000,reprojectionError=1.5,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok and len(inliers)>100
    r,tvec=cv2.solvePnPRefineLM(X[~check][inliers.ravel()],y[~check][inliers.ravel()],K,None,r,tvec)
    fitted=cv2.projectPoints(X,r,tvec,K,None)[0].reshape(-1,2);R=cv2.Rodrigues(r)[0];relative=R@w2c[:3,:3].T;angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))));oldcenter=-w2c[:3,:3].T@tv[:,0];newcenter=-R.T@tvec[:,0]
    proposal=np.eye(4);proposal[:3,:3]=R;proposal[:3,3]=tvec[:,0];np.savez_compressed(out/f'camera08-frame{t:06d}-static-pnp.npz',X=X,y=y,check=check,original_w2c=w2c,proposed_w2c=proposal,K=K,inliers=inliers)
    rows.append(dict(frame=t,correspondences=len(X),check_points=int(check.sum()),inliers=int(len(inliers)),original_check_median_px=float(np.median(np.linalg.norm(original[check]-y[check],axis=1))),pnp_check_median_px=float(np.median(np.linalg.norm(fitted[check]-y[check],axis=1))),pnp_check_p90_px=float(np.quantile(np.linalg.norm(fitted[check]-y[check],axis=1),.9)),rotation_change_deg=angle,camera_center_change_released_units=float(np.linalg.norm(newcenter-oldcenter)),positive_depth_fraction=float(((X@R.T+tvec[:,0])[:,2]>.1).mean()),proposed_w2c=proposal.tolist()))
report=dict(status='training_camera_world_pose_candidates_measured_not_applied',camera='08',rows=rows,points_sha256=digest(seed/'background-init.npz'),method='Known032 3D points matched to low-matte wall image features.80percent tracks PnP+LM,20percent same-image tracks geometric checks, fixed original K, initial original extrinsics.',limits=['Mostly planar wall support; this is not a calibrated uncertainty estimate or proven physical camera movement.','No original camera files, foreground model or heldout images modified/read.','Three selected training images do not establish whole-sequence stability or all-camera corrections.'])
(out/'static-pose-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k!='proposed_w2c'} for r in rows]},indent=2))

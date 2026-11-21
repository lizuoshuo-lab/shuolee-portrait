"""Train-view static texture diagnosis; candidate poses never replace original data."""
from pathlib import Path
import cv2,json,numpy as np,time,traceback,shutil
from audit_enerf_input import read_cameras,digest
LAB=Path(__file__).resolve().parents[2]
ROOT=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
SEED=LAB/'outputs/dynamic-human-enerf-background-032'
OUT=LAB/'outputs/dynamic-human-enerf-static-camera-036'
OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists(),'Preserve previous run; explicit recovery required.'
cv2.setNumThreads(4);cv2.setRNGSeed(3601)
cal=read_cameras(ROOT/'optimized');train=sorted(set(cal)-{'00','06','12'})
protocol=dict(status='fixed_before_all_camera_diagnosis',dataset='actor5_6',cameras=train,appearance_heldout_images_read=0,
    fit_frame=750,temporal_check_frames=[740,760],source='Released bkgd image features and032training-only triangulated3Dpoints; source frame images and released mattes.',
    features='SIFT<=10000; eroded11x11 valid native pixels with released matte<3/255; mutual BF ratio0.7 to032background descriptors.',
    spatial_split='240x180native reference-image cells; (column+3*row)%5==0 reserved for frame750check. Other cells fit.740and760are additional-image checks, not untouched scene tests.',
    pose_fit='Original K fixed. PnP RANSAC1.5px thenLM on fit inliers. Original pose initialization. Fixed032points, no bundle adjustment.',
    homography='Same frame750fit correspondences, RANSAC1.5px. Diagnostic only; not applied to people.',
    evaluation='All descriptor-qualified check correspondences scored without candidate-residual filtering; median/p90 and fractions reported.',
    preservation='No camera, image, matte, geometry or active training file modified. Candidate poses remain local diagnostic artifacts.',
    cautions=['032points were triangulated using these background cameras; they are not independent metric ground truth.',
              'Mostly planar geometry can fit projection while giving an incorrect physical camera pose.',
              'Background matches do not verify moving body synchronization or clothing correspondences.'],
    source_geometry_sha256=digest(SEED/'background-init.npz'),calibration_sha256={n:digest(ROOT/'optimized'/n) for n in ['intri.yml','extri.yml']})
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
xyz=np.load(SEED/'background-init.npz')['points'].astype(np.float64)
tracks=json.loads((SEED/'triangulation.json').read_text())['tracks']
sift=cv2.SIFT_create(nfeatures=10000);matcher=cv2.BFMatcher(cv2.NORM_L2)
rows=[];start=time.time()
def stats(e):
    return dict(n=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),fraction_below1px=float((e<1).mean())) if len(e) else dict(n=0)
def project(X,K,w):
    p=X@w[:3,:3].T+w[:3,3];p=p@K.T
    return p[:,:2]/p[:,2:]
try:
    for camera in train:
        c=cal[camera];K=np.array(c['K']);D=np.array(c['D']);original=np.array(c['w2c'])
        mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
        valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
        features=np.load(SEED/f'features-{camera}.npz');db=features['descriptors'];bx=features['xy']
        mapping={o['feature']:i for i,t in enumerate(tracks) for o in t['observations'] if o['camera']==camera}
        cases={};hashes=[]
        for frame in [750,740,760]:
            path=ROOT/'images'/camera/f'{frame:06d}.jpg';mp=ROOT/'bgmtv2'/camera/f'{frame:06d}.jpg'
            rgb=cv2.remap(cv2.imread(str(path)),mx,my,cv2.INTER_LINEAR)
            matte=cv2.remap(cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE),mx,my,cv2.INTER_LINEAR)
            mask=cv2.erode(((matte<3)&valid).astype(np.uint8),np.ones((11,11),np.uint8))*255
            kp,ds=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),mask)
            assert ds is not None
            ys=np.array([k.pt for k in kp]);ab=matcher.knnMatch(db,ds,k=2);ba=matcher.knnMatch(ds,db,k=2)
            rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.7*n.distance}
            pairs=np.array([(m.queryIdx,m.trainIdx) for m,n in ab if m.queryIdx in mapping and m.distance<.7*n.distance and rev.get(m.trainIdx)==m.queryIdx],dtype=int).reshape(-1,2)
            point_ids=np.array([mapping[i] for i,j in pairs],int)
            X=xyz[point_ids];y=ys[pairs[:,1]];x=bx[pairs[:,0]]
            cells=np.floor(x/[240,180]).astype(int);check=(cells[:,0]+3*cells[:,1])%5==0
            cases[frame]=dict(X=X,y=y,reference_xy=x,check=check,point_ids=point_ids)
            np.savez_compressed(OUT/f'camera{camera}-frame{frame:06d}-matches.npz',**cases[frame])
            hashes.append(dict(frame=frame,rgb_sha256=digest(path),matte_sha256=digest(mp),source_features=len(kp),static_matches=len(X)))
        fit=cases[750];keep=~fit['check'];X=fit['X'][keep];y=fit['y'][keep]
        row=dict(camera=camera,sources=hashes,fit_count=len(X),check_count=int(fit['check'].sum()))
        if len(X)<30:
            row['status']='insufficient_static_3D_matches';rows.append(row);continue
        rv=cv2.Rodrigues(original[:3,:3])[0];tv=original[:3,3,None].copy()
        ok,r,t,inliers=cv2.solvePnPRansac(X,y,K,None,rv.copy(),tv.copy(),True,iterationsCount=2000,reprojectionError=1.5,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers)<20:
            row['status']='insufficient_PnP_inliers';rows.append(row);continue
        r,t=cv2.solvePnPRefineLM(X[inliers.ravel()],y[inliers.ravel()],K,None,r,t)
        candidate=np.eye(4);candidate[:3,:3]=cv2.Rodrigues(r)[0];candidate[:3,3]=t[:,0]
        H,hi=cv2.findHomography(fit['reference_xy'][keep],y,cv2.RANSAC,1.5,maxIters=10000,confidence=.999)
        s=np.linalg.svd(X[inliers.ravel()]-X[inliers.ravel()].mean(0),compute_uv=False)
        evaluations=[]
        for frame,case in cases.items():
            split=case['check'] if frame==750 else np.ones(len(case['X']),bool)
            px=case['X'][split];py=case['y'][split];refxy=case['reference_xy'][split]
            baseline=np.linalg.norm(project(px,K,original)-py,axis=1)
            proposed=np.linalg.norm(project(px,K,candidate)-py,axis=1)
            ident=np.linalg.norm(refxy-py,axis=1)
            herror=np.linalg.norm(cv2.perspectiveTransform(refxy[None],H)[0]-py,axis=1) if H is not None and len(refxy) else np.array([])
            evaluations.append(dict(frame=frame,split='spatial_holdout' if frame==750 else 'additional_image_all_qualified_matches',
                original_pose=stats(baseline),candidate_pose=stats(proposed),identity_reference=stats(ident),candidate_homography=stats(herror),
                median_pairwise_improvement_px=float(np.median(baseline-proposed)) if len(px) else None,
                positive_depth_fraction=float(((px@candidate[:3,:3].T+candidate[:3,3])[:,2]>.1).mean()) if len(px) else None))
        rel=candidate[:3,:3]@original[:3,:3].T
        angle=float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1))))
        C0=-original[:3,:3].T@original[:3,3];C1=-candidate[:3,:3].T@candidate[:3,3]
        np.savez_compressed(OUT/f'camera{camera}-candidate.npz',original_w2c=original,candidate_w2c=candidate,K=K,H=H,fit_inlier_indices=inliers)
        row.update(status='candidate_measured_not_applied',fit_inliers=len(inliers),point_geometry_singular_values=s.tolist(),point_smallest_largest_ratio=float(s[-1]/s[0]),
            rotation_change_deg=angle,center_change_released_units=float(np.linalg.norm(C1-C0)),evaluations=evaluations)
        rows.append(row)
        (OUT/'progress.json').write_text(json.dumps(dict(status='running',completed_cameras=len(rows),rows=rows,elapsed_seconds=time.time()-start),indent=2)+'\n')
        print('CAMERA',json.dumps(dict(camera=camera,inliers=len(inliers),point_ratio=s[-1]/s[0],spatial_original=evaluations[0]['original_pose'],spatial_candidate=evaluations[0]['candidate_pose'])),flush=True)
    summary=dict(status='all_training_camera_static_diagnosis_complete_not_applied',rows=rows,elapsed_seconds=time.time()-start,protocol='protocol.json',goal_complete=False,
        conclusion='Geometric consistency diagnosis only; no candidate accepted as physically correct and no reconstruction improvement claimed.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copy2(__file__,OUT/Path(__file__).name)
except Exception:
    (OUT/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),completed_cameras=len(rows)),indent=2)+'\n');raise

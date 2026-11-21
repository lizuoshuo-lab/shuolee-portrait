"""Separate-time, static-background calibration for a future matched experiment."""
from pathlib import Path
import json,time,cv2,numpy as np,traceback,shutil
from audit_enerf_input import read_cameras,digest
LAB=Path(__file__).resolve().parents[2];ROOT=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
OUT=LAB/'outputs/dynamic-human-enerf-static-calibration-037';OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists(),'Preserve completed or partial calibration.'
SEED=LAB/'outputs/dynamic-human-enerf-background-032'
cv2.setNumThreads(4);cv2.setRNGSeed(3701);sift=cv2.SIFT_create(nfeatures=8000);matcher=cv2.BFMatcher(cv2.NORM_L2)
cal=read_cameras(ROOT/'optimized');frames=[700,710,720]
protocol=dict(status='fixed_before_disjoint_time_calibration',dataset='actor5_6',cameras=sorted(cal),fit_frame=700,check_frames=[710,720],
    calibration_RGB_scope='All18camera static-background features only, including appearance-heldout cameras00/06/12. Source frames700/710/720 are outside reconstruction frame750 and walking diagnostic740/760.',
    split_qualification='This candidate uses extra calibration photographs from heldout cameras. They are not texture/foreground training inputs. A future comparison must explicitly report this calibration-data access.',
    pixel_domain='Native1920x1080 with original fixedK/D undistortion. Fit a separate2Dprojection offset per camera; leave decoding/rectification unchanged.',
    fit='SIFT mutual BF ratio0.7, source mask eroded11x11 from valid matte<3/255. Spatial cells240x180 with(column+3row)%5==0reserved. Median source-minus-background xy on remaining frame700matches.',
    checks='Reserved frame700cells and all qualified matches from710/720, no candidate-dependent filtering. Homography diagnostic shares samefit split.',
    projection_contract='For a future model-only control, add delta_x/delta_y to decoded target and source projection K principal points AFTER undistortion/cropping. Never reuse adjustedK as undistortion input.',
    limitations=['Released bkgd provenance is not claimed to be an original empty capture.','Offsets are empirical local image-plane registration, not identified physical extrinsic/intrinsic corrections.','Static calibration does not verify camera synchronization for moving bodies.'],
    evaluation_frame750_RGB_read=False,applied_to_training=False,calibration_sha256={n:digest(ROOT/'optimized'/n) for n in ['intri.yml','extri.yml']})
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[];start=time.time()
def stats(e):return dict(n=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),fraction_below1px=float((e<1).mean()))
try:
    for camera in sorted(cal):
        c=cal[camera];K=np.asarray(c['K']);D=np.asarray(c['D'])
        mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
        valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
        bgpath=ROOT/'bkgd'/f'{camera}.jpg';cached=SEED/f'features-{camera}.npz'
        if cached.exists():
            z=np.load(cached);bx=z['xy'];db=z['descriptors']
        else:
            bg=cv2.remap(cv2.imread(str(bgpath)),mx,my,cv2.INTER_LINEAR)
            kp,db=sift.detectAndCompute(cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY),None)
            bx=np.array([k.pt for k in kp])
        np.savez_compressed(OUT/f'background-features-{camera}.npz',xy=bx,descriptors=db)
        cases={};sources=[]
        for frame in frames:
            rp=ROOT/'images'/camera/f'{frame:06d}.jpg';mp=ROOT/'bgmtv2'/camera/f'{frame:06d}.jpg'
            rgb=cv2.remap(cv2.imread(str(rp)),mx,my,cv2.INTER_LINEAR)
            matte=cv2.remap(cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE),mx,my,cv2.INTER_LINEAR)
            mask=cv2.erode(((matte<3)&valid).astype(np.uint8),np.ones((11,11),np.uint8))*255
            kp,ds=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),mask)
            xy=np.array([k.pt for k in kp]);ab=matcher.knnMatch(db,ds,k=2);ba=matcher.knnMatch(ds,db,k=2)
            rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.7*n.distance}
            pairs=np.array([(m.queryIdx,m.trainIdx) for m,n in ab if m.distance<.7*n.distance and rev.get(m.trainIdx)==m.queryIdx],int).reshape(-1,2)
            x=bx[pairs[:,0]];y=xy[pairs[:,1]]
            cells=np.floor(x/[240,180]).astype(int);check=(cells[:,0]+3*cells[:,1])%5==0
            assert len(x)>50 and (~check).sum()>30
            cases[frame]=dict(reference_xy=x,source_xy=y,check=check,feature_pairs=pairs)
            np.savez_compressed(OUT/f'camera{camera}-frame{frame:06d}-matches.npz',**cases[frame])
            sources.append(dict(frame=frame,rgb_sha256=digest(rp),matte_sha256=digest(mp),matches=len(x)))
        fit=cases[700];keep=~fit['check'];shift=np.median(fit['source_xy'][keep]-fit['reference_xy'][keep],axis=0)
        H,inliers=cv2.findHomography(fit['reference_xy'][keep],fit['source_xy'][keep],cv2.RANSAC,1.5,maxIters=10000,confidence=.999)
        checks=[]
        for frame,z in cases.items():
            use=z['check'] if frame==700 else np.ones(len(z['check']),bool)
            x=z['reference_xy'][use];y=z['source_xy'][use]
            checks.append(dict(frame=frame,split='spatial_holdout' if frame==700 else 'additional_image',
                identity=stats(np.linalg.norm(x-y,axis=1)),translation=stats(np.linalg.norm(x+shift-y,axis=1)),
                homography=stats(np.linalg.norm(cv2.perspectiveTransform(x[None],H)[0]-y,axis=1)),
                empirical_check_translation_xy=np.median(y-x,axis=0).tolist()))
        row=dict(camera=camera,background_sha256=digest(bgpath),sources=sources,shift_xy_px=shift.tolist(),H=H.tolist(),checks=checks,
            note='Appearance-heldout camera contributes static calibration photographs only' if camera in ['00','06','12'] else 'Training camera static calibration')
        rows.append(row)
        (OUT/'progress.json').write_text(json.dumps(dict(status='running',rows=rows,elapsed_seconds=time.time()-start),indent=2)+'\n')
        print('CAMERA',json.dumps(dict(camera=camera,shift=shift.tolist(),checks=[dict(frame=c['frame'],before=c['identity']['median_px'],after=c['translation']['median_px']) for c in checks])),flush=True)
    result=dict(status='18camera_static_projection_offsets_measured_not_applied',rows=rows,elapsed_seconds=time.time()-start,protocol='protocol.json',goal_complete=False)
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,OUT/Path(__file__).name)
except Exception:
    (OUT/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),completed_cameras=len(rows)),indent=2)+'\n');raise

"""Freeze a second moving interval and prepare original native inputs only.

No body fit, Gaussian initialization/training or heldout metrics run here.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import cv2,numpy as np
from audit_enerf_input import read_cameras

L=Path(__file__).resolve().parents[2]
R=L/'outputs/dynamic-human-enerf-second-interval-079'
RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'protocol.json').exists(),'Do not replace a frozen interval protocol'
assert shutil.disk_usage(L).free>10*1024**3
R.mkdir(exist_ok=True);I=R/'native-input';I.mkdir(exist_ok=True)
cal=read_cameras(RAW/'optimized');assert len(cal)==18
excluded=['00','06','12'];traincams=[c for c in sorted(cal) if c not in excluded]
frames=list(range(720,733));trainframes=frames[::2];valframes=frames[1::2]
assert set(frames).isdisjoint(range(744,757))
reviewed=[720,726,732,780,786,792]
selection=[dict(camera='04',frame=t,source_sha256=sha(RAW/'images/04'/f'{t:06d}.jpg')) for t in reviewed]
protocol=dict(status='second_interval_protocol_fixed_before_pose_fitting_and_training',
    scene='ENeRF-Outdoor actor5_6',reference_frame=726,frames=frames,
    training_cameras=traincams,appearance_train_frames=trainframes,
    validation_time_frames=valframes,other_cameras=excluded,
    counts=dict(appearance_train=105,known_camera_validation=90,other_camera_validation=18,total_native_inputs=213),
    selection=dict(actual_original_images_reviewed=selection,
        decision='Use720..732: both people move, raised shoes, carried toys and body overlap remain.780..792 is more stationary with dark-clothed hands behind torso; not selected as the main motion validation.',
        scope='Only training camera04 source photos used for this selection. No new heldout photos or metrics inspected for selection. Same scene/cameras as development; this is another motion interval, not independent scene generalization.'),
    planned_control=dict(branches=['global_cubic','global_cubic_plus_reference_centered_7_Bsplines'],
        steps_each=15000,seed=73,spatial_nodes=384,bindings_per_gaussian=4,
        time='tau=(capture_frame-726)/6; retain078centers,width,coefficientbounds and074loss/regularizers/densityschedule',
        preparation='Refit motion from new training-time observations. Shared newly prepared training-only initialization for both branches. Do not extrapolate old750bodymotion to726 or reuse old appearance checkpoint.',
        body_preprocessing='RTMW must receive correct RGB as repaired074; no BGR regression. Any new person boxes must cover actual new poses and be visually checked using training views.',
        evaluation='All90oddtime knowncamera plus18othercamera original images. Fixed full-person and hand/shoe/occlusion windows chosen before candidate predictions. Preserve original744..756failures.'),
    images='1920x1080 original JPEG, original bgmtv2 matte. cv2.undistort withfloat32K/D; float64K/R/T geometry; no enhancement or resampling of time.',
    preparation_only=True,training_started=False,goal_complete=False,
    created_utc=datetime.now(timezone.utc).isoformat())
save(R/'protocol.json',protocol)
rows=[]
for c in sorted(cal):
    K=np.array(cal[c]['K']);D=np.array(cal[c]['D']);W=np.array(cal[c]['w2c'])
    mx,my=cv2.initUndistortRectifyMap(K.astype(np.float32),D.astype(np.float32),None,K.astype(np.float32),(1920,1080),cv2.CV_32FC1)
    valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
    for t in (valframes if c in excluded else frames):
        rgbfile=RAW/'images'/c/f'{t:06d}.jpg';mattefile=RAW/'bgmtv2'/c/f'{t:06d}.jpg'
        source=cv2.imread(str(rgbfile));mask=cv2.imread(str(mattefile),0)
        assert source is not None and source.shape==(1080,1920,3)
        assert mask is not None and mask.shape==(1080,1920)
        rgb=cv2.undistort(source,K.astype(np.float32),D.astype(np.float32))[...,::-1].copy()
        matte=cv2.undistort(mask.astype(np.float32)/255,K.astype(np.float32),D.astype(np.float32))
        assert np.isfinite(matte).all() and matte.min()>=0 and matte.max()<=1
        ys,xs=np.where((matte>.5)&valid);assert len(xs)>1000
        box=[max(0,int(xs.min())-32),max(0,int(ys.min())-32),min(1920,int(xs.max())+33),min(1080,int(ys.max())+33)]
        target=I/f'camera{c}-frame{t:06d}.npz'
        np.savez_compressed(target,rgb=rgb,matte=matte,valid=valid,K=K,w2c=W,bbox=np.array(box))
        split='validation_camera' if c in excluded else ('train' if t in trainframes else 'validation_time')
        rows.append(dict(camera=c,frame=t,split=split,bbox=box,foreground_valid_pixels=int(len(xs)),
            source_rgb_sha256=sha(rgbfile),source_matte_sha256=sha(mattefile),input_sha256=sha(target),
            rectified_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    print('SECOND_INTERVAL_NATIVE',c,flush=True)
assert len(rows)==213
assert {k:sum(r['split']==k for r in rows) for k in ['train','validation_time','validation_camera']}==dict(train=105,validation_time=90,validation_camera=18)
save(R/'input-audit.json',dict(status='213original_native_inputs_prepared_for_second_interval',records=rows,
    protocol_sha256=sha(R/'protocol.json'),opencv_version=cv2.__version__,training_started=False,goal_complete=False))
save(R/'progress.json',dict(status='native_inputs_prepared_pose_and_initialization_pending',training_started=False,goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
print('213native inputs complete; no pose fit, model training or validation score yet.')

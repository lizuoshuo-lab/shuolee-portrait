"""Disjoint calibration-only images; never initialization or appearance training."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import cv2,numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';D=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor2_3');Q=O/'evaluation-time';N=Q/'calibration-input';N.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
existing=(Q/'calibration-input-audit.json').exists();policy=read(O/'comparison-protocol.json');assert '582..594' in policy['evaluation_clock_policy'];reserved=read(O/'validation-reservation.json');frames=list(range(582,595,2));assert not set(frames)&set(reserved['frame_window']);cameras=read(O/'cameras.json');cams=sorted(cameras)
if not (Q/'access-policy.json').exists():save(Q/'access-policy.json',dict(status='recorded_before_calibration_RGB_access',recorded_utc=datetime.now(timezone.utc).isoformat(),calibration_cameras=cams,calibration_frames=frames,fit_anchors=[584,592],check_anchor=588,reserved_appearance_window=[600,624],appearance_training_permitted=False,geometry_initialization_permitted=False,all_cameras_now_calibration_visible=True,reserved_appearance_RGB_opened=False,scope='Only disjoint-time clock qualification. Never include these images in a Gaussian appearance loader or geometry initialization. Novel cameras mean appearance-heldout, not completely unseen sensors.',goal_complete=False))
cv2.setNumThreads(2);rows=read(Q/'calibration-input-audit.json')['records'] if existing else []
for c in ([] if existing else cams):
    K=np.array(cameras[c]['K'],float);W=np.array(cameras[c]['w2c'],float);dist=np.array(cameras[c]['D'],float);mx,my=cv2.initUndistortRectifyMap(K,dist,None,K,(1920,1080),cv2.CV_32FC1);valid=cv2.remap(np.ones((1080,1920),np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>.999
    for f in frames:
        p=D/'images'/c/f'{f:06d}.jpg';mp=D/'bgmtv2'/c/f'{f:06d}.jpg';rgb=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB);matte=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255;assert rgb.shape==(1080,1920,3) and matte.shape==(1080,1920)
        rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);matte=cv2.remap(matte,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);out=N/f'camera{c}-frame{f:06d}.npz';assert not out.exists();np.savez_compressed(out,rgb=rgb,matte=matte,valid=valid,K=K,w2c=W);rows.append(dict(camera=c,frame=f,input=out.relative_to(O).as_posix(),input_sha256=sha(out),source_sha256=sha(p),source_matte_sha256=sha(mp),purpose='calibration_only'))
    print('ACTOR23_CLOCK_INPUT',c,len(rows),flush=True)
assert len(rows)==126
save(Q/'calibration-input-audit.json',dict(status='126_disjoint_calibration_inputs_complete',records=rows,appearance_training_permitted=False,geometry_initialization_permitted=False,reserved_appearance_RGB_opened=False,goal_complete=False))
# Reuse the exact qualified camera/flow mathematics, with explicit substitutions.
src=L/'scripts/dynamic_human/qualify_actor23_geometry_time.py';code=src.read_text(encoding='utf8')
changes={"Q=O/'geometry-time'":"Q=O/'evaluation-time'","cams=plan['train_cameras']":"cams=sorted(read(O/'cameras.json'))","read(O/'training-input-audit.json')":"read(Q/'calibration-input-audit.json')","604":"584","612":"588","620":"592","range(15)":"range(18)","np.zeros(15)":"np.zeros(18)","np.zeros(14)":"np.zeros(17)","(1500,930)":"(1500,1240)","training_cameras=cams,heldout_RGB_opened=False":"calibration_cameras=cams,reserved_appearance_RGB_opened=False,heldout_cameras_calibration_visible=True","same_index_sift_geometry.npz":"calibration_only_geometry_not_for_training.npz","Train-only camera/clock diagnostics":"Disjoint calibration-only camera/clock diagnostics","No heldout images or author surfaces;":"No reserved appearance-window images or author surfaces;","training_only_geometry_and_empirical_clock_diagnostic_visual_review_pending":"disjoint_calibration_only_clock_diagnostic_stationarity_check_required","No heldout image, author surface or actor5_6 geometry/time/pose was used.":"No reserved600..624appearance image, author surface or actor5_6 geometry/time/pose was used. Derived geometry is calibration-only and must not initialize training."}
for old,new in changes.items():
    assert old in code,old;code=code.replace(old,new)
dst=L/'scripts/dynamic_human/qualify_actor23_eval_clocks.py';assert not dst.exists();dst.write_text(code,encoding='utf8')
save(Q/'source-generation.json',dict(parent_sha256=sha(src),generated_sha256=sha(dst),substitutions=changes))
shutil.copy2(__file__,Q/Path(__file__).name);print('ACTOR23_EVAL_CLOCK_INPUTS_COMPLETE',flush=True)

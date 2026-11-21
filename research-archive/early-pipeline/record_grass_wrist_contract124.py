"""Expose incompatible observations of the same SMPL-X wrist; preserve all data."""
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
import audit_gaussianavatars_params123 as a

root=a.REC/'cross-subject-sh0-124';run=root/'grass-temporal-refinement'
out=run/'occluded-arm-audit'
assert not a.lp(out/'wrist-contract.json').exists()
review=a.read(run/'review-094/result.json');byid={r['frame']:r for r in review['records']}
rows=a.read(root/'grass-input/data/inputs.json')['records']
with np.load(a.lp(root/'grass-input/data/observations.npz'),allow_pickle=False) as z:p=z['keypoints']
def local(path):return path[5].upper()+':/'+path[7:] if str(path).startswith('/mnt/') else path
records=[];images=[]
for i,row in enumerate(rows):
    file=run/'official-fit/frames'/row['frame']/'winner-camera-space.npz'
    with np.load(a.lp(file),allow_pickle=False) as z:j=z['joints_camera'][0]
    left=float(np.linalg.norm(j[7]-j[25]));right=float(np.linalg.norm(j[4]-j[46]))
    assert left==right==0
    records.append(dict(frame=row['frame'],same_3d_wrist_max_abs=0.,
        left_body_handroot_distance_px=float(np.linalg.norm(p[i,9]-p[i,91])),
        right_body_handroot_distance_px=float(np.linalg.norm(p[i,10]-p[i,112])),
        left_body_wrist=p[i,9].tolist(),left_hand_wrist=p[i,91].tolist(),
        right_body_wrist=p[i,10].tolist(),right_hand_wrist=p[i,112].tolist()))
    if row['frame'] not in ['frame_000072','frame_000074','frame_000076']:continue
    im=byid[row['frame']]['images']['body'];bb=im['native_crop']
    # The saved comparison's first panel is the hash-verified native raw crop.
    # Reuse it because the original image lives on WSL's ext4 filesystem.
    overlay=Image.open(local(im['path'])).convert('RGB')
    assert a.sha(local(im['path']))==im['sha256']
    width,height=bb[2]-bb[0],bb[3]-bb[1]
    raw=overlay.crop((0,0,width,height));marked=raw.copy();draw=ImageDraw.Draw(marked)
    for indexes,color,radius in [(range(91,112),(255,150,0),2),(range(112,133),(0,220,255),2),([9],(255,0,0),6),([10],(0,70,255),6)]:
        for k in indexes:
            x,y=p[i,k]-np.array(bb[:2]);draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
    third=overlay.crop((2*raw.width,0,3*raw.width,raw.height))
    canvas=Image.new('RGB',(3*raw.width,raw.height));canvas.paste(raw,(0,0));canvas.paste(marked,(raw.width,0));canvas.paste(third,(2*raw.width,0))
    path='D:/00/记录/005/特征点_连续帧124_'+row['frame']+'_原图_红左橙左手蓝右青右手_SMPL投影.png'
    assert not a.lp(path).exists();canvas.save(a.lp(path))
    images.append(dict(frame=row['frame'],path=path,sha256=a.sha(path),native_crop=bb,resized=False,text_on_image=False))
a.save(out/'wrist-contract.json',dict(updated=datetime.now(timezone.utc).isoformat(),records=records,
    identical_3d_wrist_verified_frames=94,images=images,
    evidence='BODY25 slots7/25 and4/46 are exactly the same3D wrists in all94 saved meshes. Corresponding detector targets are COCO WholeBody9/91 and10/112.',
    conclusion='Frame74 imposes two left-wrist targets156.736px apart on one3D joint. This is an input consistency defect, not an index-mapping change or a physical-camera accuracy measurement.',
    caveat='Later left/right hand detections coincide under occlusion. Proximity alone cannot prove identity, and this audit does not filter observations or select a replacement pose.',
    implementation_note='Initial Windows raw-image open failed on a WSL ext4 path before any image/result was written. Repaired to reuse the hash-verified exact native raw crop already saved by the review; no fitting rerun.',
    pose_changed=False,observation_changed=False,gaussian_admitted=False))
print('WRIST_CONTRACT_VERIFIED94; NATIVE_EVIDENCE_SAVED3')

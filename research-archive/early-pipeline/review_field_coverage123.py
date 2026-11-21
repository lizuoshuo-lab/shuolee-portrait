from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from PIL import Image

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
F=T/'shuolee/current/local-clothed-surface-123/field-preflight'
C=F/'coverage'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
r=read(C/'result.json');obs=read(C/'observations.json');plan=read(F/'coverage-plan.json')
assert r['frames']==r['resolved_subject_frames']==177 and not r['training_started']
assert len(obs['rows'])==177
data=np.load(C/'observations.npz',allow_pickle=False)
assert data['keypoints'].shape==(177,133,2)
slots=plan['review_slots'];panels=[];items=[]
for slot in slots:
    row=obs['rows'][slot];src=F/f'slot{slot:03d}_source{row["source_frame_index"]:04d}.png'
    assert sha(src)==row['source_sha256']
    im=Image.open(src).convert('RGB');assert im.size==(2704,1520)
    crop=[1160,550,1530,1390];panels.append(im.crop(crop))
    face=data['keypoints'][slot,23:91];feet=data['keypoints'][slot,17:23]
    items.append(dict(slot=slot,source_frame=row['source_frame_index'],source_sha256=sha(src),
        body_roi=row['body_box'],face_roi=row['face_box'],
        face_native_bounds=[*face.min(0).tolist(),*face.max(0).tolist()],
        feet_native_bounds=[*feet.min(0).tolist(),*feet.max(0).tolist()],
        raw_face_response=row['regions']['face'],raw_feet_response=row['regions']['feet'],
        crop_xyxy=crop,resize=False,annotations=False))
canvas=Image.new('RGB',(370*len(slots),840));
for i,panel in enumerate(panels):canvas.paste(panel,(i*370,0))
canvas.save(C/'five-native-person-crops.png')
save(C/'review-inventory.json',dict(updated=datetime.now(timezone.utc).isoformat(),
    result_sha256=sha(C/'result.json'),observations_sha256=sha(C/'observations.json'),
    keypoints_sha256=sha(C/'observations.npz'),items=items,
    automatic=dict(single_detection_frames=sum(len(x['detections_xyxy'])==1 for x in obs['rows']),
        all_frames_subject_resolved=r['resolved_subject_frames'],zero_out_of_raster=r['frames_with_region_out_of_image'],
        numerical_inverse_affine_parity_max_px=r['native_inverse_affine_parity_max_px']),
    limitation='These are image/ROI/decoder checks. Raw SimCC maxima are not calibrated probabilities or measured localization uncertainty. No physical visibility, material correspondence, camera or fitting acceptance.'))
print(json.dumps(dict(review=str(C/'review-inventory.json'),image=str(C/'five-native-person-crops.png'),items=items),ensure_ascii=False))

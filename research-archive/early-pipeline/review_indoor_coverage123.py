"""Build native diagnostic crops and explicit, inactive 2D-to-SMPL-X bindings."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'indoor-preflight';C=PRE/'coverage'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (C/'review-inventory.json').exists()
r=read(C/'result.json');o=read(C/'observations.json');p=read(PRE/'coverage-plan.json')
assert r['frames']==r['resolved_subject_frames']==177 and r['training_started'] is False
z=np.load(C/'observations.npz',allow_pickle=False);xy=z['keypoints'];assert xy.shape==(177,133,2)
bind=read(REC/'direct-smplx-bindings/bindings.json');anchors=bind['anchors'];assert len(anchors)==51
assert o['keypoint_names'][40:91]==[f'face-{j}' for j in range(17,68)]
points=[30,33,36,39,42,45,48,54]
head_sheet=Image.new('RGB',(512,1600),'white');foot_sheet=Image.new('RGB',(1800,1280),'white')
items=[]
for i,slot in enumerate(p['review_slots']):
    row=o['rows'][slot];src=PRE/f'slot{slot:03d}_source{row["source_frame_index"]:04d}.png';assert sha(src)==row['source_sha256']
    raw=Image.open(src).convert('RGB');assert raw.size==(2704,1520)
    fb=np.array(row['face_box']);mid=(fb[:2]+fb[2:])/2;left,top=np.floor(mid-[128,160]).astype(int)
    hbox=[int(left),int(top),int(left+256),int(top+320)]
    panel=raw.crop(hbox);marked=panel.copy();draw=ImageDraw.Draw(marked)
    for j in points:
        x,y=xy[slot,23+j]-[left,top];draw.ellipse((x-2,y-2,x+2,y+2),outline='#168aff',width=1)
    head_sheet.paste(panel,(0,320*i));head_sheet.paste(marked,(256,320*i))
    bb=np.array(row['body_box']);center=int(np.floor((bb[0]+bb[2])/2));fbox=[center-450,1264,center+450,1520]
    fp=raw.crop(fbox);fm=fp.copy();draw=ImageDraw.Draw(fm)
    for j in range(15,23):
        x,y=xy[slot,j]-[fbox[0],fbox[1]]
        # Out-of-crop predictions are recorded below, never clamped into view.
        if 0<=x<900 and 0<=y<256:draw.ellipse((x-3,y-3,x+3,y+3),outline='#168aff',width=1)
    foot_sheet.paste(fp,(0,i*256));foot_sheet.paste(fm,(900,i*256))
    items.append(dict(slot=slot,source_frame=row['source_frame_index'],source_sha256=row['source_sha256'],head_crop=hbox,feet_crop=fbox,
        marked_face_indices=points,face_native_xy={str(j):xy[slot,23+j].tolist() for j in points},
        foot_native_xy={o['keypoint_names'][j]:xy[slot,j].tolist() for j in range(15,23)},
        head_region=row['face_box'],raw_face_response=row['regions']['face'],raw_feet_response=row['regions']['feet']))
head_sheet.save(C/'five-native-head-predictions.png');foot_sheet.save(C/'five-native-feet-predictions.png')
records=[]
for slot,row in enumerate(o['rows']):
    for b in anchors:
        idx=b['rtmw133_index'];uv=xy[slot,idx];assert idx==b['ibug68_index']+23
        inside=bool(0<=uv[0]<2704 and 0<=uv[1]<1520)
        records.append(dict(frame=row['frame'],source_frame=row['source_frame_index'],source_sha256=row['source_sha256'],
            anchor_id=b['anchor_id'],uv_native=uv.tolist(),inside_raster=inside,
            visibility='unknown' if inside else 'outside',semantic_mapping_verified=True,
            actual_anatomical_visibility_verified=False,physical_material_identity_verified=False,
            localization_sigma_px=None,active_for_fit=False,smplx_binding=b))
save(C/'2d-smplx-bindings.json',dict(updated=datetime.now(timezone.utc).isoformat(),
    observations_sha256=sha(C/'observations.npz'),template_sha256=bind['template_sha256'],topology_bindings_sha256=sha(REC/'direct-smplx-bindings/bindings.json'),
    input_domain='native_integer_pixel_centres',source='RTMW region-coverage predictions; not the higher-resolution face fit input',
    correspondence='Fixed anatomical triangle and barycentric location shared across frames; not nearest 3D vertex or screen point',
    frames=177,anchors_per_frame=51,records=records,real_fitting_run=False,independent_material_truth=False))
save(C/'review-inventory.json',dict(updated=datetime.now(timezone.utc).isoformat(),items=items,
    panel_order='Source0,220,440,660,880 top to bottom; raw left, detector predictions right; no text or resampling',
    face_image='five-native-head-predictions.png',feet_image='five-native-feet-predictions.png',
    displayed_markers_are_predictions_not_ground_truth=True,all_observations_inactive_until_visibility_identity_review=True,
    total_bindings=len(records),training=False))
print(json.dumps(dict(prepared_crops=10,total_2d_smplx_bindings=len(records),all_inactive=True,training=False)))

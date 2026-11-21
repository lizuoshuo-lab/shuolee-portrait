"""Read-only evidence after the frozen coverage pass. No detector/optimizer call."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
from PIL import Image

BASE=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=BASE/'shuolee/current/local-clothed-surface-123/pattern-preflight/coverage'
SRC=Path('/opt/gs-slam-lab/shuolee-local-surface-123/runtime/src/shuolee_portrait')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

assert not (O/'support-audit.json').exists(), 'Already audited; inspect saved evidence.'
observations=read(O/'observations.json'); data=np.load(O/'observations.npz',allow_pickle=False)
assert len(observations['rows'])==99 and data['keypoints'].shape==(99,133,2)
xy=data['keypoints']; finite=np.isfinite(xy).all(2)
inside=finite & (xy[:,:,0]>=0)&(xy[:,:,0]<=2703)&(xy[:,:,1]>=0)&(xy[:,:,1]<=1519)
assert np.array_equal(inside,data['in_image'])
examples=[]
for slot,key in [(24,17),(49,21),(74,21),(98,17)]:
    row=observations['rows'][slot];source=Path(row['image']);assert sha(source)==row['source_sha256']
    uv=xy[slot,key];left,top=np.floor(uv).astype(int)-96
    path=O/'visual-review'/f'slot{slot:03d}_key{key:03d}_native192.png'
    Image.open(source).convert('RGB').crop((left,top,left+192,top+192)).save(path)
    examples.append(dict(slot=slot,source_frame=row['source_frame_index'],keypoint_index=key,
        keypoint_name=observations['keypoint_names'][key],uv=uv.tolist(),score=float(data['scores'][slot,key]),
        image=path.name,crop_xyxy=[int(left),int(top),int(left+192),int(top+192)],
        actual_prediction_position_in_crop=(uv-[left,top]).tolist(),source_sha256=row['source_sha256'],
        no_resize=True,no_image_text=True,physical_annotation='pending_actual_pixel_review'))
parts={}
for name,needles in {
    'observation_support.py':['def image_support','width - 1','height - 1','mask','external'],
    'fitting.py':['image_support','anatomical_visibility','external_occluders','feet_weight','support[:,17:23]','face_weight'],
    'observations.py':['keypoints','visibility','score'],
}.items():
    p=SRC/name;lines=p.read_text().splitlines();selected=set()
    for i,line in enumerate(lines):
        if any(n in line for n in needles):selected.update(range(max(0,i-2),min(len(lines),i+4)))
    parts[name]=dict(path=str(p),sha256=sha(p),excerpts=[dict(line=i+1,text=lines[i]) for i in sorted(selected)])
result=dict(updated=datetime.now(timezone.utc).isoformat(),source_observations_sha256=sha(O/'observations.npz'),
    operation='Read saved native coordinates, source images and current support code. No new predictions, K, poses, geometry or training.',
    predicted_feet_total=int(inside[:,17:23].size),predicted_feet_outside=int((~inside[:,17:23]).sum()),
    frames_with_outside_feet=int((~inside[:,17:23]).any(1).sum()),
    predicted_feet_inside=int(inside[:,17:23].sum()),inside_does_not_certify_anatomical_visibility=True,
    examples=examples,source_code=parts,
    conclusion='Existing image_support rejects off-image predictions and uses distance to the person mask. It is not an independent anatomical or external-occlusion label. No new pattern mask or fit exists, so actual fitting weights for this person have not been evaluated.',
    camera_candidates_started=0,SMPLX_fits_started=0,training_started=False,physical_face_HELD=0)
save(O/'support-audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['source_code','examples']},ensure_ascii=False))

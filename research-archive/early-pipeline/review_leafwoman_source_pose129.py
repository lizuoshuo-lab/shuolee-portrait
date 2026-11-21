"""One CPU source-only57-frame pose/contact preparation; no roster or fitting.

The original native files/NPZ/masks remain untouched. Crops are integer pixel
rectangles and contact resizing is display-only. No detector or GPU executes.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import time
from datetime import datetime,timezone

from PIL import Image

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
DATA=Path('/opt/gs-slam-lab/shuolee-cross-subject-129/leafwoman')
OUT=ROOT/'leafwoman-native129/source-pose-review'
VIDEO_SHA='c5aac80e574dba892137d59186ff17713ab0403ec1a969d788b054a364f68cd0'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not OUT.exists(),'Finite source review already exists; inspect it, do not recreate.'
    started=time.monotonic();inputs=DATA/'data/inputs.json';obs=DATA/'data/observations.npz'
    native=ROOT/'leafwoman-native129/result.json';admission=ROOT/'new-person-intake129/video22-arc/root-source-admission.json'
    p=read(inputs);r=read(native);a=read(admission)
    assert r['observations_ready'] is True and r['frames']==57 and r['geometry_fitted'] is False
    assert p['video_sha256']==r['video_sha256']==VIDEO_SHA
    assert a['source_only_stable_pose_group_review_required_before_fit'] is True and a['fitting_admitted'] is False
    assert sha(inputs)==r['observation_manifest_sha256'] and sha(obs)==r['observations_sha256']
    rows=p['records'];assert len(rows)==57 and len({x['frame'] for x in rows})==57
    OUT.mkdir();records=[]
    for row in rows:
        image,mask=Path(row['image']),Path(row['mask'])
        assert sha(image)==row['source_sha256'] and sha(mask)==row['mask_sha256']
        raw=Image.open(image).convert('RGB');m=Image.open(mask).convert('L');assert raw.size==m.size==(2704,1520)
        b=row['initial_box_xyxy'];x0=max(0,math.floor(b[0])-32);y0=max(0,math.floor(b[1])-32)
        x1=min(raw.width,math.ceil(b[2])+32);y1=min(raw.height,math.ceil(b[3])+32);h=y1-y0
        assert x1>x0 and h>0
        boxes=dict(body=[x0,y0,x1,y1],mask=[x0,y0,x1,y1],upper=[x0,y0,x1,min(y1,y0+math.ceil(.70*h))])
        face=row.get('face_box_xyxy')
        if face:
            boxes['face']=[max(0,math.floor(face[0])-40),max(0,math.floor(face[1])-40),min(raw.width,math.ceil(face[2])+40),min(raw.height,math.ceil(face[3])+40)]
        paths={}
        for name,box in boxes.items():
            dest=OUT/(row['frame']+'-'+name+'.png');(m if name=='mask' else raw).crop(box).save(dest)
            paths[name]=dict(path=str(dest),sha256=sha(dest),crop_original_xyxy=box,native_integer_crop=True,resized=False)
        records.append(dict(frame=row['frame'],source_frame_index=row['source_frame_index'],time_seconds=row['time_seconds'],
            source_image=str(image),source_sha256=row['source_sha256'],source_mask=str(mask),source_mask_sha256=row['mask_sha256'],
            initial_detector_box=row['initial_box_xyxy'],paths=paths,pose_group=None,actual_visual_review=False,
            physical_landmark_truth=False,cropped_bottom_boundary=row['initial_box_xyxy'][3]>=raw.height))
    pose_pages=[];context_pages=[]
    for start in range(0,len(records),4):
        batch=records[start:start+4];sheet=Image.new('RGB',(1800,1420),'#ececec')
        context=Image.new('RGB',(2160,math.ceil(len(batch))*304),'#ececec')
        for j,record in enumerate(batch):
            ox=(j%2)*900;oy=(j//2)*710
            for name,x,y,w,h in [('body',0,0,240,700),('mask',245,0,240,700),('upper',495,0,390,430),('face',495,435,390,270)]:
                if name not in record['paths']:continue
                pic=Image.open(record['paths'][name]['path']).convert('RGB');pic.thumbnail((w,h),Image.Resampling.LANCZOS)
                sheet.paste(pic,(ox+x+(w-pic.width)//2,oy+y+(h-pic.height)//2))
            for k,key in enumerate(['source_image','source_mask']):
                pic=Image.open(record[key]).convert('RGB');pic.thumbnail((1080,304),Image.Resampling.LANCZOS)
                context.paste(pic,(k*1080+(1080-pic.width)//2,j*304))
        path=OUT/f'pose-contact-{start//4+1:02d}.jpg';sheet.save(path,quality=94)
        full=OUT/f'full-context-{start//4+1:02d}.jpg';context.save(full,quality=94)
        common=dict(frames=[x['frame'] for x in batch],display_only_resize=True,embedded_text=False)
        pose_pages.append(dict(path=str(path),sha256=sha(path),**common))
        context_pages.append(dict(path=str(full),sha256=sha(full),**common))
    assert sha(inputs)==r['observation_manifest_sha256'] and sha(obs)==r['observations_sha256']
    assert all(sha(x['source_image'])==x['source_sha256'] and sha(x['source_mask'])==x['source_mask_sha256'] for x in records)
    save(OUT/'source-contacts.json',dict(updated=datetime.now(timezone.utc).isoformat(),subject='leafwoman',record_id='022',
        video_sha256=VIDEO_SHA,input_manifest_sha256=sha(inputs),observation_npz_sha256=sha(obs),native_result_sha256=sha(native),
        root_admission_sha256=sha(admission),helper_sha256=sha(__file__),records=records,pose_pages=pose_pages,full_context_pages=context_pages,
        actual_visual_review=False,pose_group_classes=['chest_leaf_hands','raised_hand','unknown_or_transition'],
        group_scope='Source-only qualitative broad arm/hand gesture. No fitted residual, old person geometry, exact pose equality or material correspondence truth.',
        mask_scope='Original masks inspected as-is; leaf/body/accessory ownership requires review, not an automatic segmentation truth.',
        crop_scope='Fixed source detector boxes and original face ROI for display only; body may be truncated at native bottom. Cropped dress/legs/feet remain unknown, never blank negative supervision.',
        final_training_or_heldout_roster_created=False,source_inputs_or_NPZ_masks_changed=False,CPU_only=True,GPU_started=False,
        fitting_or_Gaussian_training_started=False,quality_accepted=False,seconds=time.monotonic()-started))
    print('LEAFWOMAN129_SOURCE57_CONTACTS_READY',OUT,sha(OUT/'source-contacts.json'))


if __name__=='__main__':main()

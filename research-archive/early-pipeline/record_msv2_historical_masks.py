"""Read-only classification and frame binding of historical graywoman masks."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-cross-subject-128/graywoman')
RECORD=Path('D:/00/记录/2026-11-06_MSV2_video15_整头子母图训练/input_audit')
DISPLAY=Path('D:/00/记录/2026-11-05_MSV2_video15/展示区域.json')
CROPS=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/graywoman-half12-input128/source-review/source-crops.json')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def linux(p):return Path('//wsl.localhost/Ubuntu-20.04'+p)
training=ROOT/'data/training-inputs.json'
rows=read(training)['records']
display=read(DISPLAY)
crop=read(CROPS)
before={str(p):sha(p) for p in [training,DISPLAY,CROPS,ROOT/'mask-audit.json',ROOT/'data/observations.npz']}
records=[]
for row in rows:
    mask=linux(row['mask']);image=linux(row['image'])
    new=Path('D:/GS-SLAM-Data/msv2-video15-20261105/dataset/images')/row['name']
    assert sha(mask)==row['mask_sha256']
    assert sha(image)==row['source_sha256']
    before[str(mask)]=sha(mask);before[str(image)]=sha(image);before[str(new)]=sha(new)
    with Image.open(mask) as im:
        mode=im.mode;wh=list(im.size);bbox=list(im.getbbox());values=np.unique(np.asarray(im)).tolist()
    with Image.open(image) as im:old_rgb=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
    with Image.open(new) as im:new_rgb=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
    assert old_rgb==new_rgb
    records.append(dict(frame=row['frame'],source_frame_index=row['source_frame_index'],
        mask_path=row['mask'],mask_sha256=row['mask_sha256'],mode=mode,wh=wh,binary_values=values,
        nonzero_bbox=bbox,mask_kind=row['mask_kind'],whole_head_semantic_mask=False,
        existing_face_field=row.get('face_region_source'),old_rgb_sha256=old_rgb,new_rgb_sha256=new_rgb,
        current_native_png=str(new),RGB_bitwise_identical=True))
npz=np.load(ROOT/'data/observations.npz',allow_pickle=False)
npz_fields={k:dict(shape=list(npz[k].shape),dtype=str(npz[k].dtype)) for k in npz.files}
head=crop['source_rows'][0]['paths']['head']
headpath=Path(head['path'].replace('/mnt/c/','C:/'))
with Image.open(headpath) as im:preview=dict(mode=im.mode,wh=list(im.size))
after={p:sha(Path(p)) for p in before}
assert after==before
result=dict(schema='msv2-history-head-mask-discovery-v1',state='read_only_audit_completed',
    checked_scope=[str(ROOT),str(CROPS.parent),str(CROPS.parent.parent.parent/'graywoman-half12-paired18k128/data')],
    conclusion='Known128 artifacts contain SAM2 whole-person binary masks and RGB head review crops; no existing whole-head semantic mask located in this checked scope',
    whole_head_mask_found=False,model_or_roi_code_modified=False,GPU_used=False,
    old_mask_count=len(records),RGB_bitwise_matches=len(records),new_source_frames=65,
    missing_source=dict(frame='frame_000065',source_index=512,nearest_old_source_index=505,
        nearest_old_mask_must_not_be_reused=True),
    binary_mask_values=sorted(set(v for r in records for v in r['binary_values'])),
    body_mask_nonzero_height_range=[min(r['nonzero_bbox'][3]-r['nonzero_bbox'][1] for r in records),max(r['nonzero_bbox'][3]-r['nonzero_bbox'][1] for r in records)],
    head_named_preview=dict(path=str(headpath),**preview,source_crop=head,
        count=len(crop['source_rows']),kind='RGB original-pixel review crop, not a semantic mask',
        generation='select_graywoman_half12_sources128.py137-149: padded person box +24px, upper30percent RGB crop; proposal_not_visibility_truth=True'),
    observations_fields=npz_fields,records=records,protected_before=before,protected_after=after,protected_unchanged=True,
    limitations='Search is bounded to the named128 data/downstream artifacts; it does not prove no head mask exists anywhere else')
(RECORD/'历史mask溯源.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('whole_head_mask_found','old_mask_count','RGB_bitwise_matches','binary_mask_values','body_mask_nonzero_height_range','protected_unchanged')},ensure_ascii=False))

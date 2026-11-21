"""Correct report units only; preserve the once-only numerical audit and sources."""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import hashlib,json
import numpy as np
from PIL import Image

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/grayT-final-control-review'
W=Path('//wsl.localhost/Ubuntu-20.04')
A=W/'opt/gs-slam-lab/shuolee-local-surface-123/grayT/data/training-inputs.json'
B=W/'opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2/data/training-inputs.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ref(p):return dict(path=str(p),sha256=sha(p))
assert not (O/'unit-correction.json').exists(),'Read the existing correction; do not repeat.'
original=read(O/'result.json');before=sha(O/'result.json')
a=read(A)['records'];b=read(B)['records'];assert len(a)==len(b)==95
ratios=[];kerr=[]
for x,y in zip(a,b):
    assert (x['frame'],x['source_frame_index'],x['time_seconds'])==(y['frame'],y['source_frame_index'],y['time_seconds'])
    assert x['wh']==[1520,2704] and y['wh']==[899,1600]
    ratio=np.asarray(x['wh'],float)/y['wh'];ratios.append(ratio)
    target=np.array(y['K']);target[:2]*=ratio[:,None]
    kerr.append(float(np.max(np.abs(target-np.array(x['K'])))))
assert max(kerr)<1e-8
image_checks=[]
for i in (0,40):
    p=W/a[i]['image'].lstrip('/')
    assert list(Image.open(p).size)==a[i]['wh'] and sha(p)==a[i]['source_sha256']
    image_checks.append(ref(p))
lo,hi=float(np.min(ratios)),float(np.max(ratios))
def bound(s):
    return {k:(v if k=='n' else {'lower':float(v*lo),'upper':float(v*hi)}) for k,v in s.items()}
project=read(O/'canonical-versus-deformed.json')
fixed=deepcopy(project)
for row in fixed['frames']:
    row['working_width']=row.pop('native_width');row['working_height']=row.pop('native_height')
    row['native_dimensions']=[1520,2704]
    for kind in ('head','whole'):
        row[kind+'_working_px']=row.pop(kind)
        row[kind+'_native_px_bounds']=bound(row[kind+'_working_px'])
fixed['units']='Working projections were 899x1600; native1520x2704 Euclidean norms are bounded by the two exact axis scale factors.'
fixed['absolute_pixel_center_offset']='Pixel-center translations cancel in displacement; this is not a claim about absolute observed feature error.'
save(O/'canonical-versus-deformed-corrected.json',fixed)
corrected=deepcopy(original);corrected['working_camera_dimensions']=corrected.pop('native_camera_dimensions')
corrected['native_camera_dimensions']=[[1520,2704]]
for kind in ('head','whole'):
    key=f'projection_delta_{kind}_per_frame_median'
    corrected[key+'_working_px']=corrected.pop(key)
    corrected[key+'_native_px_bounds']=bound(corrected[key+'_working_px'])
corrected['state']='units_corrected_visual_review_pending'
corrected['updated']=datetime.now(timezone.utc).isoformat()
corrected['correction']=str(O/'unit-correction.json')
correction=dict(updated=corrected['updated'],kind='report_unit_label_correction_only',original=ref(O/'result.json'),
    original_preserved=True,source_camera_files=[ref(A),ref(B)],source_image_checks=image_checks,
    matching_frame_identity=95,full_K_row_scale_max_abs_error=max(kerr),
    working_dimensions=[899,1600],native_dimensions=[1520,2704],axis_scale=[1520/899,2704/1600],
    formula='min(rx,ry)*working_norm <= native_norm <= max(rx,ry)*working_norm; applies to each monotone order statistic.',
    exact_native_vectors_recomputed=False,reran_numerical_audit=False,changed_camera=False,new_fit=False,new_training=False,
    limitation='This compares saved model states; neither is independent observed material or physical motion ground truth. A and B extrinsics are not asserted equal.')
save(O/'unit-correction.json',correction);save(O/'result-completed.json',corrected)
assert sha(O/'result.json')==before
print(json.dumps(dict(frame_count=95,K_error=max(kerr),head_native=corrected['projection_delta_head_per_frame_median_native_px_bounds']),ensure_ascii=False))

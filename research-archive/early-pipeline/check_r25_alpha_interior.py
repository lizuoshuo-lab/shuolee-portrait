"""Measure same native held-out alpha away from the source mask boundary."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from audit_r23_numerical import LAB, alpha_image, _pin, _read


subject = sys.argv[1]
step = int(sys.argv[2]) if len(sys.argv)>2 else 18000
old_run = LAB/f'shuolee-smpl-delivery-round22-{subject}'/'run'
source = {row['frame']:row for row in _read(old_run/'data/inputs.json')['records']}
paths = [LAB/f'shuolee-smpl-delivery-round{r}-{subject}'/'run'/'gaussian-run'/f'raw-step-{step:06d}'/'drive-replay' for r in (24,25)]
records = [[row for row in _read(p/'result.json')['records'] if row['role']=='pose_conditioned_holdout'] for p in paths]
result = []
for a,b in zip(*records,strict=True):
    frame = a['frame']
    row = source[frame]
    assert all(a[k]==b[k] for k in ('frame','mask_sha256','source_sha256','box_xyxy','pose_index'))
    assert all(a[k]==row[k] for k in ('mask_sha256','source_sha256'))
    mask_file = Path(row['mask'])
    _pin(mask_file,a['mask_sha256'])
    with Image.open(mask_file) as image:
        mask = np.asarray(image.convert('L'))
    interior = distance_transform_edt(mask>=253)>3.
    x0,y0,x1,y1 = a['box_xyxy']
    selected = interior[y0:y1,x0:x1]
    assert selected.sum()==interior.sum()>0
    values = []
    for path,r in zip(paths,(a,b),strict=True):
        alpha = alpha_image(path/r['files']['alpha'],(x1-x0,y1-y0))[selected].astype(np.float64)/255.
        values.append(dict(mean=float(alpha.mean()),below_point9=float(np.mean(alpha<.9))))
    result.append(dict(frame=frame,confident_interior_pixels=int(selected.sum()),r24=values[0],r25=values[1]))
print(json.dumps(result,indent=2))

"""One read-only pixel-mapping check of existing008 inputs, no inference or fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np
from PIL import Image

P=Path('/mnt/d/00/工作/人体重建')
sys.path.insert(0,str(P/'src'))
from shuolee_portrait import runtime as rt
rt.configure(P/'unused-assets-for-crop-audit')
from shuolee_portrait.crops import prepare_prior_crop

root=Path('/opt/gs-slam-lab/shuolee-portrait-server-check-008')
out=Path('/mnt/d/GS-SLAM-Data/shuolee-server-install-check-002/prior-crop-audit')
out.mkdir(parents=True,exist_ok=False)
manifest=root/'data/training-inputs.json'
rows=rt.read(manifest)['records']
samples=np.array([[0,0,1],[255,0,1],[0,255,1],[255,255,1],[127.5,127.5,1]],float)
checks=[]
for row in rows:
    assert rt.sha(row['image'])==row['source_sha256']
    rgb=np.array(Image.open(row['image']).convert('RGB'))
    crop,cs,mapping=prepare_prior_crop(rgb,row['initial_box_xyxy'])
    A=np.eye(3);s=cs[2];c=cs[:2]
    A[:2]=[[256/s,0,127.5-c[0]*256/s],[0,256/s,127.5-c[1]*256/s]]
    before=samples@np.linalg.inv(A).T
    after=samples@np.array(mapping['crop_to_source']).T
    shift=float(np.max(np.abs(before[:,:2]-after[:,:2])))
    checks.append(dict(frame=row['frame'],source_frame_index=row['source_frame_index'],
                       wh=row['wh'],roi_size=float(s),previous_map_axis_shift_px_max=shift,
                       source_sha256=row['source_sha256'],**{'mapping':mapping}))
result=dict(created=datetime.now(timezone.utc).isoformat(),state='read_only_prior_crop_contract_audit',
            frames=len(rows),manifest_sha256=rt.sha(manifest),records=checks,
            maximum_axis_shift_in_008_rectified_pixels=max(r['previous_map_axis_shift_px_max'] for r in checks),
            training_images_modified=False,model_inference_run=False,new_fit_run=False,
            historical_114_mapping_retested=False,historical_ghosting_root_cause_proven=False,
            camera_calibration_changed=False,quality_accepted=False,
            scope='New server adapter crop only. Full-image cameras and RTMW observations unchanged.')
rt.save(out/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

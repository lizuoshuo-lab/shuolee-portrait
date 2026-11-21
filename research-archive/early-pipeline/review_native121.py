"""Copy bounded, unannotated native image review assets. Does not change observations."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121/visual-review')
OUT.mkdir(exist_ok=True)
receipt=[]
for subject in ['grass','sofa','portrait','indoor']:
    obs=json.loads((ROOT/subject/'observations.json').read_text())
    panels=[]
    for i in [0,11,23]:
        r=obs['records'][i];im=cv2.imread(r['image']);h,w=im.shape[:2]
        target=OUT/f'{subject}_native_frame{r["frame"]}.jpg'
        cv2.imwrite(str(target),im,[cv2.IMWRITE_JPEG_QUALITY,95])
        scale=min(640/w,640/h);small=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        panel=np.full((640,640,3),240,np.uint8);yy=(640-small.shape[0])//2;xx=(640-small.shape[1])//2
        panel[yy:yy+small.shape[0],xx:xx+small.shape[1]]=small;panels.append(panel)
        receipt.append(dict(subject=subject,frame=r['frame'],file=str(target),pixel_processing='JPEG encoding only; no geometric resampling',features_unchanged=True))
    cv2.imwrite(str(OUT/f'{subject}_overview_first_middle_last.jpg'),np.concatenate(panels,axis=1),[cv2.IMWRITE_JPEG_QUALITY,94])
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(str(OUT))

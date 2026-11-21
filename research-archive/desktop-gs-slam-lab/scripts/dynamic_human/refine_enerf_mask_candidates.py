"""Conservative union with nearby high-alpha released pixels; candidate, not GT."""
from pathlib import Path
import json,cv2,numpy as np,os
from PIL import Image,ImageDraw
from audit_enerf_input import digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
SRC=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
mode=os.environ.get('MASK_REPAIR_MODE','nearby-core')
OUT=SRC/('hybrid-closed-core' if mode=='closed-core' else 'hybrid');OUT.mkdir(exist_ok=True)
protocol=dict(status='fixed_before_combination',sam2_variant='Three manually prompted objects (boxes plus point except camera01)',
    rule='candidate = SAM2_union OR ((released_alpha>0.95) AND distance_to_SAM2<=24_native_pixels)',
    motivation='SAM2 may omit body interiors at inter-person occlusions; original strong-alpha pixels may preserve them. Restrict to local support to reject far wall ghosts.',
    limitation='Empirical hard segmentation, may preserve near-body shadows; not ground truth or soft matting. Source photos remain unchanged; no heldout masks used.',
    frame=750,heldout_excluded=['00','06','12'],goal_complete=False)
assert not (OUT/'summary.json').exists()
if mode=='closed-core':
    protocol['rule']='candidate = SAM2_union OR ((released_alpha>0.95) AND morphological_close(SAM2_union,ellipse49px))'
    protocol['motivation']='Recover supported interior holes/narrow notches at overlaps without a uniform outward dilation of body silhouettes.'
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[]
for record in json.loads((SRC/'segmentation.json').read_text())['records']:
    c=record['camera'];sam=np.array(Image.open(SRC/f'camera{c}-candidate.png'))>0
    matte=np.load(SRC/f'camera{c}-released.npy');rgb=np.array(Image.open(SRC/f'camera{c}-source.png'))
    distance=cv2.distanceTransform((~sam).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    support=(distance<=24.) if mode!='closed-core' else cv2.morphologyEx(sam.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49))).astype(bool)
    union=sam|((matte>.95)&support)
    Image.fromarray((union*255).astype(np.uint8)).save(OUT/f'camera{c}-mask.png')
    new=np.rint(rgb.astype(np.float32)*union[...,None]+127.5*(~union[...,None])).astype(np.uint8)
    Image.fromarray(new).save(OUT/f'camera{c}-composite.png')
    old=np.rint(rgb.astype(np.float32)*matte[...,None]+127.5*(1-matte[...,None])).astype(np.uint8)
    x0,y0,x1,y1=record['comparison_bbox'];w=x1-x0;h=y1-y0
    panel=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(panel)
    for j,(label,arr) in enumerate([('Photo',rgb),('Released matte',old),('Hybrid input mask candidate',new)]):
        panel.paste(Image.fromarray(arr[y0:y1,x0:x1]),(j*w,32));d.text((j*w+4,10),f'{c}: {label}',fill='black')
    panel.save(OUT/f'camera{c}-comparison.png')
    rows.append(dict(camera=c,mask_sha256=digest(OUT/f'camera{c}-mask.png'),sam2_pixels=int(sam.sum()),candidate_pixels=int(union.sum()),additional_high_alpha_pixels=int((union&~sam).sum()),removed_released_core_pixels=int(((matte>.95)&~union).sum()),bbox=record['comparison_bbox']))
(OUT/'summary.json').write_text(json.dumps(dict(status='15_hybrid_input_candidates_visual_review_pending',records=rows,goal_complete=False),indent=2)+'\n')
print('15hybridinputcandidatesready')

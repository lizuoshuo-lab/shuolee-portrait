"""Read-only training-camera photographs for explicit SAM2 box selection."""
from pathlib import Path
import json, cv2, numpy as np
from PIL import Image, ImageDraw
from audit_enerf_input import read_cameras,digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
ROOT=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15'
OUT.mkdir(exist_ok=True)
cameras=read_cameras(ROOT/'optimized');rows=[];images=[]
for name,cam in cameras.items():
    if name in ['00','06','12']:continue
    rp=ROOT/'images'/name/'000750.jpg';mp=ROOT/'bgmtv2'/name/'000750.jpg'
    K=np.array(cam['K']);D=np.array(cam['D'])
    maps=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
    rgb=cv2.remap(cv2.cvtColor(cv2.imread(str(rp)),cv2.COLOR_BGR2RGB),*maps,cv2.INTER_LINEAR)
    matte=cv2.remap(cv2.imread(str(mp),0).astype(np.float32)/255,*maps,cv2.INTER_LINEAR)
    Image.fromarray(rgb).save(OUT/f'camera{name}-source.png')
    np.save(OUT/f'camera{name}-released.npy',matte)
    images.append((name,rgb))
    rows.append(dict(camera=name,rgb_sha256=digest(rp),matte_sha256=digest(mp),rectified_rgb_sha256=digest(OUT/f'camera{name}-source.png')))
for s in range(5):
    im=Image.new('RGB',(1920,392),'white');draw=ImageDraw.Draw(im)
    for j,(name,rgb) in enumerate(images[s*3:s*3+3]):
        im.paste(Image.fromarray(rgb).resize((640,360)),(j*640,32))
        draw.text((j*640+10,10),f'Camera {name}; preview x/y multiply by 3 for native coordinates',fill='black')
    im.save(OUT/f'source-sheet-{s}.png')
(OUT/'source-receipt.json').write_text(json.dumps(dict(status='train_only_photographs_prepared',frame=750,heldout_excluded=['00','06','12'],records=rows),indent=2)+'\n')
print('Prepared15trainingphotographs; originaldatasetunchanged')

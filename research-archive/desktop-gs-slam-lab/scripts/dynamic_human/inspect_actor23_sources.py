"""Read-only independent-scene preparation; heldout RGB stays unopened."""
from pathlib import Path
import json, hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from audit_enerf_input import read_cameras

L=Path(__file__).resolve().parents[2]
D=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor2_3')
O=L/'outputs/dynamic-human-actor23-independent-106'
O.mkdir(exist_ok=True)
assert not (O/'source-inventory.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
cameras=read_cameras(D/'optimized')
assert len(cameras)==18 and set(cameras)=={f'{i:02d}' for i in range(18)}
inventories={}; common=None
for c in sorted(cameras):
    images=sorted(int(p.stem) for p in (D/'images'/c).glob('*.jpg'))
    masks=sorted(int(p.stem) for p in (D/'bgmtv2'/c).glob('*.jpg'))
    assert len(images)==len(set(images)) and images and masks
    inventories[c]=dict(images=len(images),masks=len(masks),first=images[0],last=images[-1],mask_only_frames=sorted(set(masks)-set(images)),image_only_frames=sorted(set(images)-set(masks)))
    paired=set(images)&set(masks)
    common=paired if common is None else common & paired
save(O/'cameras.json',cameras)
frames=[0,200,400,600,800,1000]; c='04'
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
sheet=Image.new('RGB',(1440,600),'white'); draw=ImageDraw.Draw(sheet); rows=[]
for i,f in enumerate(frames):
    p=D/'images'/c/f'{f:06d}.jpg'; m=D/'bgmtv2'/c/f'{f:06d}.jpg'
    with Image.open(p) as im:
        im.load(); assert im.size==(1920,1080)
        sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),((i%3)*480,(i//3)*300+30))
    draw.text(((i%3)*480+8,(i//3)*300+5),f'Original camera {c} | frame {f} | preview 25%',font=font,fill='black')
    mask=np.array(Image.open(m)); assert mask.shape[:2]==(1080,1920)
    rows.append(dict(camera=c,frame=f,path=str(p),sha256=sha(p),mask_sha256=sha(m)))
sheet.save(O/'training-source-survey.png')
save(O/'source-inventory.json',dict(status='source_inventory_and_matrix_checks_only',dataset='actor2_3',cameras=inventories,common_frame_count=len(common),calibration_hashes={n:sha(D/'optimized'/n) for n in ['intri.yml','extri.yml']},source_survey=rows,source_survey_file='training-source-survey.png',heldout_cameras=['00','06','12'],heldout_RGB_opened=False,source_selection_only=True,training_started=False,hardware_synchronization_verified=False,scope='Camera matrix algebra and file inventory do not prove reprojection accuracy or synchronization. Source thumbnails are resized only for inspection; future evaluation uses native pixels. Released precomputed surfaces may include heldout cameras and are not used for training initialization.',goal_complete=False))
print(json.dumps(dict(cameras=len(cameras),common_frames=len(common),inspected_training_photos=len(rows),heldout_RGB_opened=False)))

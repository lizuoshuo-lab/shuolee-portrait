from pathlib import Path
import json,shutil
from audit_enerf_input import read_cameras,digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');ROOT=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
OUT=LAB/'outputs/dynamic-human-enerf-actor56-time-049';OUT.mkdir(exist_ok=True)
assert not (OUT/'manifest.json').exists()
frames=sorted({a+d for a in [740,750,760] for d in range(-3,4)});rows=[]
for c in [f'{i:02d}' for i in range(18) if i not in [0,6,12]]:
 for t in frames:
  image=f'images/{c}/{t:06d}.jpg';mask=f'bgmtv2/{c}/{t:06d}.jpg'
  rows.append(dict(camera=c,frame=t,width=1920,height=1080,path=image,sha256=digest(ROOT/image),mask_path=mask,mask_sha256=digest(ROOT/mask),
    split='train',mask_semantics='Released background-subtraction matte, imperfect foreground support including held objects.'))
manifest=dict(root=str(ROOT),cameras=read_cameras(ROOT/'optimized'),heldout_cameras=['00','06','12'],heldout_frames=[],images=rows,
 calibration_hashes={f'optimized/{n}':digest(ROOT/'optimized'/n) for n in ['intri.yml','extri.yml']},
 purpose='Actor5_6 train-only local timing consistency near walking frame750; no dataset retiming and no heldout appearance opened.')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for n in ['prepare_enerf_actor56_time_diagnostic.py','audit_enerf_camera_time.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,OUT/n)
print('049:',len(rows),'native training observations; anchors740/750 fit,760 independent check')

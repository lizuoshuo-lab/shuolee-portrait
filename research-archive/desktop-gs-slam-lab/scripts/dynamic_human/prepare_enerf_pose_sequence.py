from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from audit_enerf_input import read_cameras
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
IN=OUT/'native-input';IN.mkdir(exist_ok=True);assert not (IN/'input-audit.json').exists()
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cal=read_cameras(RAW/'optimized');rows=[]
for c in sorted(cal):
 if c in ['00','06','12']:continue
 K=np.array(cal[c]['K']);D=np.array(cal[c]['D']);mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1)
 for t in range(744,757):
  f=RAW/'images'/c/f'{t:06d}.jpg';rgb=cv2.undistort(cv2.imread(str(f)),K.astype(np.float32),D.astype(np.float32))[...,::-1].copy()
  if t==750:assert np.array_equal(rgb,np.load(BASE/f'train-camera-{c}.npz')['rgb'])
  target=IN/f'camera{c}-frame{t:06d}.npz';np.savez_compressed(target,rgb=rgb)
  rows.append(dict(camera=c,frame=t,source_sha256=sha(f),rectified_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),input_sha256=sha(target),K=cal[c]['K'],w2c=cal[c]['w2c']))
 print('NATIVE_INPUT',c,flush=True)
save(IN/'input-audit.json',dict(status='195native_training_images_15frame750_exactly_match_frozen050',rectification='cv2.undistort withfloat32K/D as039/048/050 officialloader; not float64 map remap',opencv_version=cv2.__version__,records=rows,goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name)

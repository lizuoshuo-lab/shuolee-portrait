"""Native, fixed-camera background references, explicit train/evaluation selection."""
from pathlib import Path
import cv2
import numpy as np
from audit_enerf_input import read_cameras, digest

HELDOUT = ['00', '06', '12']
def load_background(root, cameras):
    root=Path(root); calibration=read_cameras(root/'optimized'); result=[]
    for camera in cameras:
        path=root/'bkgd'/f'{camera}.jpg'; bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.shape == (1080,1920,3)
        c=calibration[camera]; K=np.array(c['K']); D=np.array(c['D'])
        mx,my=cv2.initUndistortRectifyMap(K,D,np.eye(3),K,(1920,1080),cv2.CV_32FC1)
        valid=(mx>=0)&(my>=0)&(mx<1919)&(my<1079)&np.isfinite(mx)&np.isfinite(my)
        rgb=cv2.cvtColor(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),cv2.COLOR_BGR2RGB)
        result.append(dict(camera=camera,rgb=rgb,valid=valid,K=K,w2c=np.array(c['w2c']),source=path.relative_to(root).as_posix(),sha256=digest(path)))
    return result

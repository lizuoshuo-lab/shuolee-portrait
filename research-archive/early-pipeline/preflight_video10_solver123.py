"""Inventory only: no camera solve, package install, or GPU execution."""
from pathlib import Path
import sys,hashlib,json,shutil,importlib.util
from datetime import datetime,timezone
out=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-remaining-material-review/global-input-preflight')
out.mkdir(exist_ok=True)
assert not (out/'runtime.json').exists()
sys.path.insert(0,'/opt/gs-slam-lab/camera-contract120-tools/poselib-2.0.5')
import poselib,pycolmap,scipy
file_sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=dict(updated=datetime.now(timezone.utc).isoformat(),python=sys.executable,
 poselib=dict(version=poselib.__version__,path=poselib.__file__,file_sha256=file_sha(poselib.__file__),
   selected_api=[x for x in dir(poselib) if any(s in x.lower() for s in ['global','averag','relative','absolute','bundle','three'])]),
 pycolmap=dict(version=pycolmap.__version__,path=pycolmap.__file__,file_sha256=file_sha(pycolmap.__file__),
   selected_api=[x for x in dir(pycolmap) if any(s in x.lower() for s in ['global','averag','mapper','mapping'])]),
 scipy=dict(version=scipy.__version__,path=scipy.__file__),
 glomap_on_PATH=shutil.which('glomap'),colmap_on_PATH=shutil.which('colmap'),
 optional_modules={k:bool(importlib.util.find_spec(k)) for k in ['gtsam','pyceres']},
 scope='Only existing clean environment, explicit existing PoseLib2.0.5 source and current PATH inspected. No exhaustive filesystem claim.',
 camera_estimates=0,installations=0,downloads=0,training=0)
(out/'runtime.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result))

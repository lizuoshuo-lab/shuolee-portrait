from pathlib import Path
import json, zipfile
import numpy as np
from scipy.spatial.transform import Rotation
import pycolmap
H=Path(__file__).parent;D=H/'camera-formats121-test';results=[]
for name in ['new-capture-119-sofa-clothed','new-capture-119-portrait-clothed','branches']:
    source=json.loads((D/(name+'.json')).read_text());rec=pycolmap.Reconstruction(str(D/name));assert len(rec.images)==len(source)
    max_pose=max_k=0.
    for i,s in enumerate(source):
        im=rec.images[i+1];cam=rec.cameras[im.camera_id]
        pose=im.cam_from_world
        if callable(pose):pose=pose()
        p=pose.matrix()
        r=np.asarray(s['rotation']).T;t=-r@np.asarray(s['position'])
        max_pose=max(max_pose,float(abs(p-np.column_stack([r,t])).max()))
        max_k=max(max_k,float(abs(cam.params-np.asarray([s['fx'],s['fy'],s['cx'],s['cy']])).max()))
    assert max_pose<1e-10 and max_k<1e-10
    if name!='branches':
        with zipfile.ZipFile(D/(name+'.zip')) as z:
            assert z.testzip() is None
            for n in z.namelist():assert z.read(n)==(D/name/n).read_bytes()
    results.append(dict(name=name,cameras=len(source),pycolmap_read=True,max_pose_error=max_pose,max_K_error=max_k))
(H/'camera-formats121-external-reader.json').write_text(json.dumps(dict(pycolmap_version=pycolmap.__version__,results=results,external_viewer_ui_tested=False),indent=2)+'\n')
print(json.dumps(results,indent=2))

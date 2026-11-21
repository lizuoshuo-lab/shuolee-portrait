from pathlib import Path
import json,zipfile,hashlib
import numpy as np
import pycolmap
H=Path(__file__).parent
D=Path('/mnt/c/Users/Administrator/Downloads')
arr=json.loads((D/'new-capture-119-sofa-clothed_cameras (1).json').read_text())
assert isinstance(arr,list) and len(arr)==89
T=H/'camera-formats121-test/browser-download';T.mkdir(exist_ok=True)
with zipfile.ZipFile(D/'new-capture-119-sofa-clothed_cameras_colmap.zip') as z:
    assert z.testzip() is None
    assert set(z.namelist())=={'cameras.txt','images.txt','points3D.txt','camera_provenance.json','README.txt'}
    for n in z.namelist():(T/n).write_bytes(z.read(n))
r=pycolmap.Reconstruction(str(T));assert len(r.images)==89
mx=0.
for i,c in enumerate(arr):
    im=r.images[i+1];pose=im.cam_from_world
    if callable(pose):pose=pose()
    p=pose.matrix();R=np.asarray(c['rotation']).T;t=-R@np.asarray(c['position'])
    mx=max(mx,float(abs(p-np.c_[R,t]).max()))
    assert np.allclose(r.cameras[im.camera_id].params,[c['fx'],c['fy'],c['cx'],c['cy']],rtol=0,atol=1e-12)
assert mx<1e-12
out=dict(actual_browser_downloads=True,flat_3dgs_array=True,cameras=len(arr),colmap_reader='pycolmap'+pycolmap.__version__,
    camera_pose_max_error=mx,K_max_error=0,zip_crc_pass=True,remote00_verified=False,third_party_viewer_ui_tested=False,
    files=[dict(name=n,sha256=hashlib.sha256((D/n).read_bytes()).hexdigest()) for n in ['new-capture-119-sofa-clothed_cameras (1).json','new-capture-119-sofa-clothed_cameras_colmap.zip']])
(H/'camera-formats121-browser-download.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

"""Check training-only image/mask inputs and the two released camera conventions."""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
from shuolee_audit_diffuman_cameras import LAB, OUT
from shuolee_prepare_elevation_benchmark import digest, dump


def main():
    folder=OUT/'elevation-benchmark'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    staged=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    assert staged['status']=='staged_not_visualized'
    sys.path.insert(0,str(LAB/'scripts/dynamic_human'))
    from audit_enerf_input import read_cameras
    camera_root=OUT/protocol['scene'];evc=read_cameras(camera_root)
    ns=json.loads((camera_root/'transforms.json').read_text(encoding='utf-8'))
    cameras={f['camera_label']:f for f in ns['frames']}
    rows=[]
    for c in protocol['roles']['train_horizontal']:
        n=cameras[c];e=evc[c]
        cv_c2w=np.asarray(n['transform_matrix'])@np.diag([1.,-1.,-1.,1.])
        evc_c2w=np.linalg.inv(e['w2c'])
        pose_error=float(np.max(abs(cv_c2w-evc_c2w)))
        K=np.array([[n['fl_x'],0,n['cx']],[0,n['fl_y'],n['cy']],[0,0,1.]])
        Kerror=float(np.max(abs(K-np.asarray(e['K']))))
        assert pose_error<1e-5 and Kerror<1e-4
        photo=folder/'train_horizontal/images'/c/(protocol['frame']+'.webp')
        maskfile=folder/'train_horizontal/fmasks'/c/(protocol['frame']+'.png')
        a=np.asarray(Image.open(photo).convert('RGB'));mask=np.asarray(Image.open(maskfile).convert('L'))
        assert a.shape==(n['h'],n['w'],3) and mask.shape==a.shape[:2]
        foreground=mask>127;ys,xs=np.nonzero(foreground)
        assert len(xs)>1000
        inside=cv2.erode(foreground.astype('uint8'),np.ones((5,5),np.uint8))>0
        gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
        rows.append(dict(camera=c,resolution=[n['w'],n['h']],K=K.tolist(),distortion=e['D'],
            w2c=e['w2c'],camera_format_pose_max_error=pose_error,camera_format_K_max_error=Kerror,
            principal_point_offset_from_center=[n['cx']-n['w']/2,n['cy']-n['h']/2],
            foreground_fraction=float(foreground.mean()),bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
            touches_border=bool(foreground[0].any() or foreground[-1].any() or foreground[:,0].any() or foreground[:,-1].any()),
            grayscale_laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_32F)[inside].var()),
            source_sha256=digest(photo),mask_sha256=digest(maskfile)))
    result=dict(status='training_inputs_checked_not_training_ready',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        scene=protocol['scene'],frame=protocol['frame'],rows=rows,test_images_opened=False,
        boundary_cameras=[r['camera'] for r in rows if r['touches_border']],
        visual_inspection=dict(by='assistant',training_cameras=['02','26'],human_reviews=0,
            observations=['Subject and clothing are visible without obvious block redaction in inspected images.',
            'Some feet touch/cross the lower image boundary; these pixels cannot be treated as observed complete silhouettes.',
            'Foreground masks approximate fingers and sheer clothing; use a boundary uncertainty band.',
            'Face texture is limited by1024px source and human occupies part of image; do not promise missing fine detail.']),
        required_before_training=['Undistort photographs and masks with the same maps and recenter principal point, or use a renderer supporting the actual principal point.',
            'Initialize using only horizontal training images/masks; do not use author all-view sparse cloud.',
            'Verify multiview consistency and camera coverage; identical frame indices alone do not prove synchronization.'],
        laplacian_note='Input diagnostic affected by texture and noise; not a cross-method perceptual quality score.')
    dump(folder/'input-qa.json',result)
    staged.update(status='training_inputs_inspected',test_images_inspected=False,training_started=False,
        last_input_audit='input-qa.json')
    dump(folder/'status.json',staged)
    print(json.dumps(dict(status=result['status'],boundary_cameras=result['boundary_cameras'],
        max_pose_error=max(r['camera_format_pose_max_error'] for r in rows),
        max_K_error=max(r['camera_format_K_max_error'] for r in rows)),indent=2))


if __name__=='__main__':main()

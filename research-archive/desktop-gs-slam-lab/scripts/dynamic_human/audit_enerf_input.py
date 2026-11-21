"""Read-only native-image/calibration audit before any ENeRF model optimization.

Local adapter of the documented EasyVolcap OpenCV calibration format.
No dataset pixels, masks, points or poses are synthesized by this module.
"""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone
import cv2
import numpy as np
from PIL import Image

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def read_cameras(root):
    root = Path(root)
    for name in ('intri.yml', 'extri.yml'):
        if not (root/name).is_file(): raise FileNotFoundError(root/name)
    intri = cv2.FileStorage(str(root/'intri.yml'), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(str(root/'extri.yml'), cv2.FILE_STORAGE_READ)
    result = {}
    try:
        if not intri.isOpened() or not extri.isOpened(): raise ValueError('Cannot open calibration')
        names = intri.getNode('names')
        for i in range(names.size()):
            node = names.at(i); name = node.string() if node.isString() else str(int(node.real()))
            if name == 'none': continue
            if name in result: raise ValueError('Duplicate camera name')
            K = intri.getNode('K_'+name).mat()
            T = extri.getNode('T_'+name).mat()
            rv = extri.getNode('R_'+name).mat()
            R = cv2.Rodrigues(rv)[0] if rv is not None else extri.getNode('Rot_'+name).mat()
            if K is None or R is None or T is None: raise ValueError('Missing calibration '+name)
            if K.shape != (3,3) or R.shape != (3,3) or T.size != 3: raise ValueError('Invalid camera dimensions')
            if not all(np.isfinite(x).all() for x in (K,R,T)): raise ValueError('Nonfinite calibration')
            if not np.allclose(R@R.T,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
                raise ValueError('Invalid rotation '+name)
            if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]): raise ValueError('Invalid intrinsic matrix')
            D = intri.getNode('D_'+name).mat()
            if D is None: D = intri.getNode('dist_'+name).mat()
            if D is None: raise ValueError('Missing distortion; do not assume undistorted images')
            if D.size not in (4,5,8,12,14) or not np.isfinite(D).all(): raise ValueError('Unsupported distortion')
            w2c = np.eye(4); w2c[:3,:3] = R; w2c[:3,3] = T.reshape(3)
            center = -R.T@T.reshape(3)
            assert np.allclose(w2c@np.r_[center,1],[0,0,0,1],atol=1e-6)
            declared_w=int(intri.getNode('W_'+name).real())
            declared_h=int(intri.getNode('H_'+name).real())
            # EasyVolcap read_camera uses -1 for unspecified image dimensions.
            # This does not authorize resizing K: retain it and verify real geometry.
            result[name] = dict(K=K.tolist(), D=D.reshape(-1).tolist(), w2c=w2c.tolist(),
                camera_center_world=center.tolist(), W=declared_w if declared_w>0 else None,
                H=declared_h if declared_h>0 else None,
                declared_W=declared_w,declared_H=declared_h)
        if not result: raise ValueError('No named cameras')
    finally:
        intri.release(); extri.release()
    return result

def audit(root, out, count=60, calibration_subdir='.'):
    root = Path(root).resolve(); out = Path(out).resolve()
    if out.exists(): raise FileExistsError('Preserve prior audit; choose a new output')
    calibration_root=(root/calibration_subdir).resolve()
    if not calibration_root.is_relative_to(root):raise ValueError('Calibration outside dataset root')
    cameras = read_cameras(calibration_root); cam_names = list(cameras)
    if len(cam_names)!=18: raise ValueError('Expected official 18-camera capture; inspect actual archive')
    images = {}
    for cam in cam_names:
        image_dir = root/'images'/cam
        if not image_dir.is_dir(): raise ValueError('Missing calibrated image camera '+cam)
        entries = {}
        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
            if not p.stem.isdigit(): raise ValueError('Unexpected frame name '+p.name)
            t = int(p.stem)
            if t in entries: raise ValueError('Duplicate numeric frame '+cam+'/'+p.name)
            entries[t] = p
        images[cam] = entries
    common = sorted(set.intersection(*(set(d) for d in images.values())))
    selected = common[:count]
    if len(selected)!=count or any(b-a!=1 for a,b in zip(selected,selected[1:])):
        raise ValueError('Insufficient consecutive synchronized frames')
    held_cams = [cam_names[i] for i in (0,6,12)]
    held_times = selected[5::10]
    rows = []
    for cam in cam_names:
        sizes = set()
        for t in selected:
            p = images[cam][t]
            with Image.open(p) as im:
                im.load(); width,height = im.size
                if im.mode not in ('RGB','RGBA'): raise ValueError('Unexpected image channels')
            sizes.add((width,height))
            c=cameras[cam]
            if c['W'] and c['W']!=width: raise ValueError('Calibration/image width mismatch')
            if c['H'] and c['H']!=height: raise ValueError('Calibration/image height mismatch')
            split = 'heldout_time' if t in held_times else 'heldout_view' if cam in held_cams else 'train'
            row=dict(camera=cam,frame=t,path=p.relative_to(root).as_posix(),
                width=width,height=height,sha256=digest(p),split=split)
            mask=root/'bgmtv2'/cam/p.name
            if (root/'bgmtv2').exists():
                with Image.open(mask) as im:
                    im.load()
                    if im.size!=(width,height):raise ValueError('Matte/image size differs')
                row.update(mask_path=mask.relative_to(root).as_posix(),mask_sha256=digest(mask),
                    mask_semantics='Author supplied background-matting output, not manually verified human-only ground truth')
            rows.append(row)
        if len(sizes)!=1: raise ValueError('Variable image dimensions for camera '+cam)
    result = dict(status='native_images_and_numeric_calibration_audited_visual_review_pending',
        root=str(root),camera_count=len(cameras),cameras=cameras,selected_frames=selected,
        image_count=len(rows),split_counts={s:sum(x['split']==s for x in rows) for s in ['train','heldout_view','heldout_time']},
        heldout_cameras=held_cams,heldout_frames=held_times,images=rows,
        calibration_hashes={(calibration_root/n).relative_to(root).as_posix():digest(calibration_root/n) for n in ('intri.yml','extri.yml')},
        coordinate_convention='OpenCV world-to-camera X_c=R X_w+T; C_w=-R^T T',
        preprocessing='Native RGB unchanged; distortion retained. Undistortion with consistent new K required before gsplat pinhole rendering.',
        split_scope='Local pilot, not a claim to reproduce official paper evaluation',
        initialization='Exclude heldout RGB/masks/depth from all initialization and geometry fitting',
        source_geometry='Not approved for initialization until generating camera/time subset is established',
        unresolved=['native human clarity and motion blur visual audit','metric world units','capture frame rate','mask provenance','calibration reprojection on actual correspondences'],
        observed_utc=datetime.now(timezone.utc).isoformat(),local_only=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','camera_count','image_count','split_counts']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--frames',type=int,default=60);p.add_argument('--calibration-subdir',default='.')
    a=p.parse_args();audit(a.root,a.out,a.frames,a.calibration_subdir)

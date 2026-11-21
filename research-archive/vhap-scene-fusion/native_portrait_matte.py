"""Map the pre-dilation person matte to the actual training pinhole camera."""
import cv2
import numpy as np


def source_map(scene, frames, K, size):
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

    names = {f.name for f in frames}
    paths = [p for p in (scene/'colmap_bg/sparse').glob('*/cameras.bin')
             if {r.name for r in read_extrinsics_binary(p.with_name('images.bin')).values()} == names]
    if len(paths) != 1:
        raise ValueError('Expected a unique reconstruction of the native frame sequence')
    cam, = read_intrinsics_binary(paths[0]).values()
    if cam.model != 'OPENCV':
        raise ValueError(cam.model)
    fx, fy, cx, cy, *dist = cam.params
    raw = np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])
    return cv2.initUndistortRectifyMap(raw, np.asarray(dist), np.eye(3), K, size, cv2.CV_32FC1)


def portrait_matte(scene, frame, maps, undistorted):
    read = lambda directory: cv2.imread(str(scene/directory/frame.name), cv2.IMREAD_GRAYSCALE)
    expanded, soft = read('person_masks'), read('alpha_maps')
    if expanded is None or soft is None or expanded.shape != soft.shape:
        raise ValueError(f'Missing or incompatible source mattes for {frame.name}')
    mapped = cv2.remap(expanded, *maps, cv2.INTER_LINEAR)[frame.crop] >= 224
    target = undistorted[frame.crop] >= 224
    overlap = (mapped & target).sum() / max(1, (mapped | target).sum())
    if overlap < .98:
        raise ValueError(f'Matte registration failed for {frame.name}: IoU={overlap:.4f}')
    return cv2.remap(soft, *maps, cv2.INTER_LINEAR)

"""CPU-only common 3DGS and COLMAP PINHOLE camera export.

K uses COLMAP pixel corners. Extrinsics are proper OpenCV world-to-camera
transforms; 3DGS JSON stores camera-to-world rotation and camera center. No
image, sparse point or feature track is invented by this serializer.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


def _validate(row):
    wh = row.get('wh')
    if (not isinstance(wh, (list, tuple)) or len(wh) != 2 or
            any(type(value) is not int or value <= 0 for value in wh)):
        raise ValueError('Camera width and height must be positive integers.')
    K = np.asarray(row.get('K'), dtype=np.float64)
    E = np.asarray(row.get('w2c'), dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
            min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0) or
            abs(K[0, 1]) + abs(K[1, 0]) > 1e-12):
        raise ValueError('Common export requires finite zero-skew PINHOLE fx/fy/cx/cy.')
    if (E.shape != (4, 4) or not np.isfinite(E).all() or
            not np.allclose(E[3], [0, 0, 0, 1], atol=1e-12, rtol=0) or
            not np.allclose(E[:3, :3] @ E[:3, :3].T, np.eye(3), atol=1e-6, rtol=0) or
            abs(np.linalg.det(E[:3, :3]) - 1) > 1e-6):
        raise ValueError('Common export requires a proper rigid world-to-camera transform.')
    return K, E


def _image_name(row, index, used):
    original = row.get('name') or row.get('frame') or f'frame_{index:06d}'
    if not isinstance(original, str):
        raise ValueError('Camera image names must be strings.')
    # Keep a portable basename. Original names, including spaces and paths, are
    # preserved in the separate mapping rather than inserted into COLMAP rows.
    base = original.replace('\\', '/').rsplit('/', 1)[-1]
    base = re.sub(r'[^A-Za-z0-9_.-]', '_', base).strip(' ._')
    extension = re.search(r'\.[A-Za-z0-9]{1,8}$', base)
    suffix = extension.group(0) if extension else '.png'
    stem = base[:extension.start()] if extension else base
    stem = stem.strip(' ._')
    if not re.search('[A-Za-z0-9]', stem):
        stem = f'frame_{index:06d}'
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in range(1, 10)}
    if stem.split('.')[0].upper() in reserved:
        stem = 'frame_' + stem
    stem = stem[:100]
    name, counter = stem + suffix, 2
    while name.casefold() in used:
        name = f'{stem}_{counter:03d}{suffix}'
        counter += 1
    used.add(name.casefold())
    return name, original


def camera_documents(rows):
    """Build standard documents plus explicit original-name mappings."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('Cannot export an empty camera set.')
    cameras, mappings = [], []
    camera_lines = [
        '# Camera list with one line of data per camera:',
        '# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
        f'# Number of cameras: {len(rows)}',
    ]
    image_lines = [
        '# Image list with two lines of data per image:',
        '# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME',
        '# POINTS2D[] as (X, Y, POINT3D_ID); empty: no observations supplied',
        f'# Number of images: {len(rows)}, mean observations per image: 0',
    ]
    used = set()
    number = lambda value: format(float(value), '.17g')
    for i, row in enumerate(rows):
        K, E = _validate(row)
        width, height = row['wh']
        name, original = _image_name(row, i, used)
        R, t = E[:3, :3], E[:3, 3]
        camera_to_world = R.T
        center = -camera_to_world @ t
        cameras.append(dict(id=i, img_name=name, width=width, height=height,
            position=center.tolist(), rotation=camera_to_world.tolist(),
            fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            pixelConvention='pixel_corners', projectionModel='pinhole'))
        camera_id = image_id = i + 1
        camera_lines.append(f'{camera_id} PINHOLE {width} {height} ' +
                            ' '.join(number(value) for value in (K[0, 0], K[1, 1], K[0, 2], K[1, 2])))
        xyzw = Rotation.from_matrix(R).as_quat()
        if xyzw[3] < 0:
            xyzw = -xyzw
        pose = [xyzw[3], xyzw[0], xyzw[1], xyzw[2], *t]
        image_lines.append(f'{image_id} ' + ' '.join(number(value) for value in pose) + f' {camera_id} {name}')
        image_lines.append('')
        mappings.append(dict(index=i, camera_id=camera_id, image_id=image_id,
            original_name=original, original_frame=row.get('frame'), exported_name=name))
    return dict(
        cameras=cameras, name_mapping=mappings,
        colmap_cameras='\n'.join(camera_lines) + '\n',
        colmap_images='\n'.join(image_lines) + '\n',
        colmap_points='# 3D point list: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n'
                      '# Number of points: 0, mean track length: 0\n',
    )


def write_camera_exchange(directory, rows, *, world_to_viewer=None):
    """Write common files and optional provenance; refuse existing destinations."""
    documents = camera_documents(rows)  # Validate everything before any writes.
    directory = Path(directory)
    files = {
        'cameras.json': json.dumps(documents['cameras'], ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        'colmap/cameras.txt': documents['colmap_cameras'],
        'colmap/images.txt': documents['colmap_images'],
        'colmap/points3D.txt': documents['colmap_points'],
    }
    for relative in (*files, 'camera_provenance.json'):
        if (directory / relative).exists():
            raise FileExistsError(f'Preserve existing camera export: {directory / relative}')
    checksums = {}
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode('utf-8')
        path.write_bytes(encoded)
        checksums[relative] = hashlib.sha256(encoded).hexdigest()
    provenance = dict(schema='shuolee-camera-exchange-v1', records=rows,
        convention='OpenCV world-to-camera; COLMAP pixel corners',
        world_to_viewer=world_to_viewer, name_mapping=documents['name_mapping'],
        files_sha256=checksums, point_count=0, observations_per_image=0,
        image_data_included=False, cameras_share_model_world=True,
        quality_accepted=False, physical_calibration_accepted=False,
        limitations=[
            '3DGS JSON preserves cx/cy as explicit fields; some readers ignore them and assume a centered principal point.',
            'COLMAP PINHOLE keeps full fx/fy/cx/cy and Hamilton wxyz world-to-camera poses.',
            'No source images, sparse 3D points or feature tracks are invented or bundled.',
        ])
    (directory / 'camera_provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return provenance

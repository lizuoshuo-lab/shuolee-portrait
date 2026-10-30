"""Shared PTS sampling and explicit pose/GS pixel domains; no fitting model."""
from pathlib import Path
import json
import subprocess

import numpy as np
from PIL import Image

from . import runtime as rt
from .camera_pixel_contract import map_pixels, resize_matrix


SAMPLE_FPS = 3
RESOLUTION_RULES = {
    (1920, 1080): (2, 2),
    (3840, 2160): (4, 2),
    (2704, 1520): (2, 1),
    (1520, 2704): (2, 1),
}


def resolution_contract(source_wh):
    if (len(source_wh) != 2 or any(type(v) is not int for v in source_wh)
            or tuple(source_wh) not in RESOLUTION_RULES):
        raise ValueError(f'No approved pose/GS resolution rule for {source_wh}.')
    pose_div, gs_div = RESOLUTION_RULES[tuple(source_wh)]
    pose_wh = [v // pose_div for v in source_wh]
    gs_wh = [v // gs_div for v in source_wh]
    return dict(schema='smplx-preprocessing-v1', sample_fps=SAMPLE_FPS,
        source_wh=list(source_wh), pose_wh=pose_wh, gaussian_wh=gs_wh,
        pose_divisor=pose_div, gaussian_divisor=gs_div,
        source_to_pose_integer=resize_matrix(source_wh, pose_wh).tolist(),
        source_to_GS_integer=resize_matrix(source_wh, gs_wh).tolist(),
        pose_to_GS_integer=resize_matrix(pose_wh, gs_wh).tolist(),
        pose_to_GS_corner=np.diag([gs_wh[0]/pose_wh[0], gs_wh[1]/pose_wh[1], 1.]).tolist(),
        interpolation='INTER_AREA for RGB; INTER_NEAREST for masks',
        second_undistortion=False, physical_calibration_accepted=False)


def sampling_indices(times, sample_fps=SAMPLE_FPS):
    """Nearest real PTS to t0+n/3; ties choose the earlier source frame."""
    if type(sample_fps) not in (int, float) or sample_fps != SAMPLE_FPS:
        raise ValueError('The preprocessing contract requires sample_fps=3.')
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Require real strictly increasing decoded PTS; no synthetic timestamps.')
    grids = times[0] + np.arange(int(np.floor((times[-1]-times[0])*sample_fps))+1)/sample_fps
    hi = np.searchsorted(times, grids).clip(0, len(times)-1)
    lo = (hi-1).clip(0, len(times)-1)
    ids = np.where(abs(times[lo]-grids) <= abs(times[hi]-grids), lo, hi)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Source cannot supply distinct samples at 3 fps; frames are not duplicated.')
    return grids, ids


def extract_dual_resolution(video, root, cfg):
    """Decode each selected source frame once, then derive both image sizes."""
    import cv2
    from .video119 import validate_video_stream

    root, video = Path(root).resolve(), Path(video).resolve()
    for name in ('raw', 'pose', 'decoded.json', 'decoded-pose.json', 'pixel-contract.json'):
        if (root/name).exists():
            raise FileExistsError(f'Fresh preprocessing only; preserve {root/name}.')
    digest = rt.sha(video)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-show_entries',
        'frame=best_effort_timestamp_time:stream', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    if len(info.get('streams', [])) != 1:
        raise ValueError('Expected exactly one selected video stream.')
    stream = info['streams'][0]
    validate_video_stream(stream)
    contract = resolution_contract([int(stream['width']), int(stream['height'])])
    times = np.asarray([float(f['best_effort_timestamp_time']) for f in info['frames']])
    grids, ids = sampling_indices(times, cfg['sample_fps'])
    if not cfg['min_frames'] <= len(ids) <= cfg['max_frames']:
        raise ValueError(f'{len(ids)} samples at 3 fps exceed the declared '
                         f'{cfg["min_frames"]}..{cfg["max_frames"]} frame budget; no silent truncation.')
    (root/'raw').mkdir(); (root/'pose').mkdir()
    vf = 'select=' + '+'.join(f'eq(n\\,{int(i)})' for i in ids)
    with (root/'logs/decode.log').open('w') as log:
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-v', 'warning', '-n',
            '-noautorotate', '-i', str(video), '-vf', vf, '-pix_fmt', 'rgb24', '-vsync', '0',
            str(root/'raw/frame_%06d.png')], stdout=log, stderr=subprocess.STDOUT, check=True)
    files = sorted((root/'raw').glob('frame_*.png'))
    if len(files) != len(ids):
        raise ValueError('Decoded frame identity/count does not match sampled source PTS.')
    gs_rows, pose_rows = [], []
    for path, idx, grid in zip(files, ids, grids):
        with Image.open(path) as im:
            if list(im.size) != contract['source_wh']:
                raise ValueError('Decoded source dimensions changed; no implicit rotation or crop.')
            rgb = np.array(im.convert('RGB'))
        small = root/'pose'/path.name
        Image.fromarray(cv2.resize(rgb, contract['pose_wh'], interpolation=cv2.INTER_AREA)).save(small)
        if contract['gaussian_wh'] != contract['source_wh']:
            # This is our newly decoded copy. The source video is never modified.
            Image.fromarray(cv2.resize(rgb, contract['gaussian_wh'], interpolation=cv2.INTER_AREA)).save(path)
        identity = dict(frame=path.stem, name=path.name, source_frame_index=int(idx),
            time_seconds=float(times[idx]), target_time_seconds=float(grid),
            sampling_error_seconds=float(times[idx]-grid))
        gs_rows.append(dict(identity, image=str(path), source_sha256=rt.sha(path), wh=contract['gaussian_wh']))
        pose_rows.append(dict(identity, image=str(small), source_sha256=rt.sha(small), wh=contract['pose_wh']))
    if rt.sha(video) != digest:
        raise ValueError('Source video changed during preprocessing.')
    metadata = dict(video_sha256=digest, decoded_total=len(times), sample_fps=SAMPLE_FPS,
        source_indices=ids.tolist(), preprocessing=contract,
        sampling='Nearest real source PTS to t0+n/3; earlier frame on ties; no interpolation',
        decoder='FFmpeg noautorotate; shared source frames; explicit INTER_AREA resizing')
    rt.save(root/'decoded.json', dict(metadata, records=gs_rows, pixel_domain='GS'))
    rt.save(root/'decoded-pose.json', dict(metadata, records=pose_rows, pixel_domain='pose'))
    rt.save(root/'pixel-contract.json', contract)
    return contract


def camera_intrinsics(contract, K_pose):
    """Map the complete pose-domain K, including its principal point, to GS."""
    K_pose = np.asarray(K_pose, dtype=np.float64)
    if K_pose.shape != (3, 3) or not np.isfinite(K_pose).all():
        raise ValueError('Expected finite complete pose K.')
    A = np.asarray(contract['pose_to_GS_integer'])
    K_gs = A @ K_pose
    pose_corner, gs_corner = K_pose.copy(), K_gs.copy()
    pose_corner[:2, 2] += .5; gs_corner[:2, 2] += .5
    if not np.allclose(np.asarray(contract['pose_to_GS_corner']) @ pose_corner, gs_corner,
                       rtol=0, atol=1e-10):
        raise ValueError('Integer/corner resize contracts disagree.')
    return dict(contract, pose_K_integer_centers=K_pose.tolist(), pose_K_corner=pose_corner.tolist(),
                GS_K_integer_centers=K_gs.tolist(), GS_K_corner=gs_corner.tolist())


def derive_pose_observations(root):
    """Map GS detections once; do not redetect lower-resolution face/hands."""
    import cv2
    root = Path(root)
    native = rt.read(root/'data/inputs.json')
    pose = rt.read(root/'data/pose-inputs.json')
    contract = rt.read(root/'pixel-contract.json')
    rows, small_rows = native['records'], pose['records']
    ids, small_ids = [r['frame'] for r in rows], [r['frame'] for r in small_rows]
    if len(set(ids)) != len(ids) or ids != small_ids:
        raise ValueError('Pose/GS observation frame identities must match exactly once.')
    with np.load(root/'data/observations.npz', allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}
    obs_ids = obs['frame_ids'].astype(str).tolist()
    if len(set(obs_ids)) != len(obs_ids) or set(obs_ids) != set(ids):
        raise ValueError('Detector frame identities differ from the selected source frames.')
    order = [obs_ids.index(frame) for frame in ids]
    pobs = dict(obs)
    shapes = dict(keypoints=(len(ids), 133, 2), scores=(len(ids), 133),
                  boxes=(len(ids), 4), flip_disagreement=(len(ids), 133))
    for key, shape in shapes.items():
        if obs[key].shape != shape or not np.isfinite(obs[key]).all():
            raise ValueError('Invalid finite COCO-133 observation array: ' + key)
    for key in ('frame_ids', 'keypoints', 'scores', 'boxes', 'flip_disagreement'):
        if len(obs[key]) != len(ids):
            raise ValueError('Observation frame count differs: ' + key)
        pobs[key] = obs[key][order].copy()
    A = resize_matrix(contract['gaussian_wh'], contract['pose_wh'])
    if A[0, 0] != A[1, 1]:
        raise ValueError('Scalar flip disagreement requires an isotropic resize.')
    for key in ('keypoints', 'boxes'):
        values = pobs[key]
        pobs[key] = map_pixels(values.reshape(-1, 2), A).reshape(values.shape)
    pobs['flip_disagreement'] = pobs['flip_disagreement'] * A[0, 0]
    masks = root/'data/pose-masks'; masks.mkdir()
    for row, small in zip(rows, small_rows):
        for key in ('source_frame_index', 'time_seconds', 'target_time_seconds'):
            if row[key] != small[key]:
                raise ValueError('Pose/GS source identity differs: ' + row['frame'])
        for item, wh in ((row, contract['gaussian_wh']), (small, contract['pose_wh'])):
            if item['wh'] != wh or rt.sha(item['image']) != item['source_sha256']:
                raise ValueError('Pose/GS source pixels changed: ' + row['frame'])
        for item, prefix in ((row, 'GS'), (small, 'pose')):
            if (not np.array_equal(item['K_integer_centers'], contract[prefix+'_K_integer_centers'])
                    or not np.array_equal(item['K'], contract[prefix+'_K_corner'])):
                raise ValueError('Pose/GS complete K differs from the pixel contract: ' + row['frame'])
        if row['w2c'] != small['w2c']:
            raise ValueError('Image resizing must not alter camera extrinsics: ' + row['frame'])
        if rt.sha(row['mask']) != row['mask_sha256']:
            raise ValueError('GS mask changed: ' + row['frame'])
        with Image.open(row['mask']) as im:
            if list(im.size) != row['wh']:
                raise ValueError('GS mask dimensions differ: ' + row['frame'])
            mask = np.array(im.convert('L'))
        path = masks/small['name']
        Image.fromarray(cv2.resize(mask, contract['pose_wh'], interpolation=cv2.INTER_NEAREST)).save(path)
        small.update(mask=str(path.resolve()), mask_sha256=rt.sha(path),
            native_image_sha256=row['source_sha256'], native_mask_sha256=row['mask_sha256'],
            observation_origin='GS-domain detector coordinates mapped once; no second detection')
    pose.update(observation_source=dict(path=str((root/'data/observations.npz').resolve()),
        sha256=rt.sha(root/'data/observations.npz')), pixel_contract_sha256=rt.sha(root/'pixel-contract.json'))
    rt.save(root/'data/pose-inputs.json', pose)
    with (root/'data/pose-observations.npz').open('xb') as f:
        np.savez_compressed(f, **pobs)

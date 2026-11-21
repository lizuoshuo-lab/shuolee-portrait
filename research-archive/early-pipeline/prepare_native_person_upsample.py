"""Build direct native-video crops and 2x enlarged inputs for the sofa trainer."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

PREVIOUS = Path('/opt/gs-slam-lab/static-person-upsampling-20261102')
BASE = Path('/opt/gs-slam-lab/static-person-native-upsampling-20261102')
ROOT = BASE / 'sofa'
VIDEO = Path('/mnt/d/00/工作/数据集/video6.mp4')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    old = PREVIOUS / 'sofa'
    plan = read(old / 'plan.json')
    assert sha(VIDEO) == plan['source_video_sha256']
    source = Path(plan['source_run'])
    rows = read(source / 'data/training-inputs.json')['records']
    video = cv2.VideoCapture(str(VIDEO))
    assert video.isOpened()
    native_wh = np.array([int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))])
    assert native_wh.tolist() == [2704, 1520]
    scale_to_native = native_wh / np.asarray(rows[0]['wh'])
    old_box = np.asarray(plan['crop_xyxy'])
    box = np.r_[np.floor(old_box[:2] * scale_to_native), np.ceil(old_box[2:] * scale_to_native)].astype(int)
    box[:2] = np.maximum(0, box[:2]); box[2:] = np.minimum(native_wh, box[2:])
    width, height = (box[2:] - box[:2]).tolist()
    ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(old / 'source', ROOT / 'source', symlinks=True)
    shutil.copy2(old / 'source-manifest.json', ROOT / 'source-manifest.json')
    launcher = (old / 'experiment.py').read_text(encoding='utf-8')
    launcher = launcher.replace(str(PREVIOUS), str(BASE))
    launcher = launcher.replace("    cfg = read(out/'config.json')\n", "    cfg = read(out/'config.json')\n    rows = read(out/'data/training-inputs.json')['records']\n    assert all(r['wh'][1] == cfg['training_height'] for r in rows)\n    assert all(list(Image.open(r['image']).size) == r['wh'] for r in rows)\n")
    (ROOT / 'experiment.py').write_text(launcher, encoding='utf-8')
    shutil.copy2(__file__, ROOT / 'prepare-native-inputs.py')
    plan.update(created=datetime.now(timezone.utc).isoformat(), source_video=str(VIDEO),
        crop_xyxy=box.tolist(), crop_wh=[width, height], crop_coordinate_wh=native_wh.tolist(),
        input_source='Integer crop from native decoded source video, then optional 2x Lanczos enlargement; no RGB downsampling.',
        no_rgb_downsampling=True, previous_input_wh=rows[0]['wh'], native_source_wh=native_wh.tolist(),
        mask_source='Existing segmentation mask resized to native video with bilinear interpolation; no new segmentation model.',
        input_sizes={'original': [width, height], 'upsampled2x': [width * 2, height * 2]},
        intervention='Direct native-video crop enlarged 2x before training; loss and render resolution equal enlarged PNG resolution.',
        script_sha256=sha(__file__), source_manifest_sha256=sha(ROOT / 'source-manifest.json'))
    save(ROOT / 'plan.json', plan)
    adapted = {a: [] for a in ['original', 'upsampled2x']}
    proofs = []
    for arm, factor in [('original', 1), ('upsampled2x', 2)]:
        out = ROOT / arm
        for folder in ['data/images', 'data/masks', 'fit', 'clothing-init']:
            (out / folder).mkdir(parents=True)
        cfg = read(source / 'config.json')
        cfg.update(training_height=height * factor, image_cache_frames=len(rows), max_image_size=int(max(native_wh) * factor),
            iterations=18000, seed=6060, gaussian_representation='static_clothed_world',
            native_video_input=True, pre_training_enlargement=factor, no_rgb_downsampling=True)
        save(out / 'config.json', cfg)
        for rel in ['fit/fitted.npz', 'clothing-init/points.npz', 'clothing-init/motion.npz', 'clothing-init/audit.json']:
            shutil.copy2(source / rel, out / rel)
            assert sha(source / rel) == sha(out / rel)
    selected = set(np.linspace(0, len(rows) - 1, 6).astype(int).tolist())
    for i, row in enumerate(rows):
        assert sha(row['mask']) == row['mask_sha256']
        video.set(cv2.CAP_PROP_POS_FRAMES, row['source_frame_index'])
        ok, bgr = video.read(); assert ok
        raw = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        assert list(raw.size) == native_wh.tolist()
        rgb = raw.crop(tuple(box))
        assert np.array_equal(np.asarray(rgb), np.asarray(raw)[box[1]:box[3], box[0]:box[2]])
        full_mask = Image.open(row['mask']).convert('L').resize(tuple(native_wh), Image.Resampling.BILINEAR)
        mask = full_mask.crop(tuple(box))
        assert np.count_nonzero(np.asarray(mask) > 127) == np.count_nonzero(np.asarray(full_mask) > 127)
        proof = dict(frame=row['frame'], source_frame_index=row['source_frame_index'], native_wh=native_wh.tolist(),
            integer_crop_xyxy=box.tolist(), native_crop_exact_pixels=True, inputs={})
        if i in selected:
            dest = ROOT / 'source-frames' / (row['frame'] + '.png')
            dest.parent.mkdir(exist_ok=True); raw.save(dest)
        for arm, factor in [('original', 1), ('upsampled2x', 2)]:
            out = ROOT / arm
            im = rgb if factor == 1 else rgb.resize((width * factor, height * factor), Image.Resampling.LANCZOS)
            alpha = mask if factor == 1 else mask.resize(im.size, Image.Resampling.BILINEAR)
            ip = out / 'data/images' / row['name']; mp = out / 'data/masks' / row['name']
            im.save(ip); alpha.save(mp)
            # K uses corner coordinates; 2D observations use integer pixel centers.
            transform = np.array([[factor * scale_to_native[0], 0, -factor * box[0]],
                [0, factor * scale_to_native[1], -factor * box[1]], [0, 0, 1.]])
            K = transform @ np.asarray(row['K'])
            Ki = K.copy(); Ki[:2, 2] -= .5
            r = copy.deepcopy(row)
            r.update(image=str(ip), mask=str(mp), source_sha256=sha(ip), mask_sha256=sha(mp),
                wh=list(im.size), K=K.tolist(), K_integer_centers=Ki.tolist(),
                parent_mask_sha256=row['mask_sha256'], crop_xyxy=box.tolist(), crop_coordinate_wh=native_wh.tolist(),
                upsampling_scale=factor, no_rgb_downsampling=True, image_origin='direct_native_video_integer_crop')
            for key in ['face_box_xyxy', 'initial_box_xyxy']:
                if key in r:
                    xy = np.asarray(row[key]).reshape(2, 2)
                    r[key] = (((xy + .5) * scale_to_native - box[:2]) * factor - .5).reshape(-1).tolist()
            r['other_person_boxes_xyxy'] = [(((np.asarray(b).reshape(2, 2) + .5) * scale_to_native - box[:2]) * factor - .5).reshape(-1).tolist() for b in row.get('other_person_boxes_xyxy', [])]
            adapted[arm].append(r)
            proof['inputs'][arm] = {'wh': list(im.size), 'sha256': r['source_sha256'], 'training_height': height * factor}
        proofs.append(proof)
        if i % 20 == 0: print('native frames prepared', i + 1, '/', len(rows), flush=True)
    video.release()
    for arm in adapted:
        save(ROOT / arm / 'data/training-inputs.json', dict(records=adapted[arm], coordinate_contract='pixel_corners',
            source_video=str(VIDEO), source_video_sha256=plan['source_video_sha256'],
            transform='native video integer crop then 2x enlargement' if arm == 'upsampled2x' else 'native video integer crop',
            no_rgb_downsampling=True, fresh_gaussians=True, cached_fit_and_surface=True))
    errors = []
    for a, b in zip(adapted['original'], adapted['upsampled2x']):
        assert a['w2c'] == b['w2c']
        errors.append(float(np.max(abs(np.asarray(b['K']) - np.diag([2, 2, 1]) @ np.asarray(a['K'])))))
        expected = Image.open(a['image']).resize(tuple(b['wh']), Image.Resampling.LANCZOS)
        assert np.array_equal(np.asarray(expected), np.asarray(Image.open(b['image'])))
    save(ROOT / 'input-verification.json', dict(passed=True, frames=len(rows), records=proofs,
        native_video_wh=native_wh.tolist(), input_sizes=plan['input_sizes'], no_rgb_downsampling=True,
        enlarged_png_exact_lanczos=True, max_K_scaling_error_px=max(errors), same_cameras_and_initial_surface=True))
    print(json.dumps({'prepared': True, 'frames': len(rows), 'input_sizes': plan['input_sizes'], 'root': str(ROOT)}), flush=True)


if __name__ == '__main__':
    main()

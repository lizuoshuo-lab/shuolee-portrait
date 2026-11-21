"""Prepare demo for the sofa static representation using background cameras."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
import numpy as np

BASE = Path('/opt/gs-slam-lab/static-person-upsampling-20261102')
ROOT = BASE/'demo-static-input'
INPUT = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
PROJECT = Path('/mnt/d/Projects/Shuolee')
ASSETS = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8'); t.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def camera():
    import cv2
    ROOT.mkdir(exist_ok=False)
    for folder in ['data', 'logs', 'camera/background-masks', 'source']:
        (ROOT/folder).mkdir(parents=True)
    source = PROJECT/'src/shuolee_portrait/video119.py'
    shutil.copy2(source, ROOT/'source/video119.py')
    spec = importlib.util.spec_from_file_location('static_background_camera', ROOT/'source/video119.py')
    sfm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sfm)
    rows = read(INPUT/'data/training-inputs.json')['records']
    video = Path('/mnt/d/00/工作/数据集/demo.mp4')
    save(ROOT/'run.json', dict(video=str(video), video_sha256=sha(video), subject='demo',
        original_input=str(INPUT), reused_observations=True, old_cameras_loaded=False,
        old_fitted_person_loaded=False, old_gaussians_loaded=False, quality_accepted=False))
    cfg = read(Path('/opt/gs-slam-lab/shuolee-clothed-surface-119/sofa/config.json'))
    cfg.update(training_height=rows[0]['wh'][1], max_image_size=max(rows[0]['wh']), image_cache_frames=len(rows))
    save(ROOT/'config.json', cfg)
    start = time.monotonic()
    def status(stage, **kw):
        elapsed = time.monotonic()-start
        save(ROOT/'camera-progress.json', dict(stage=stage, seconds=elapsed, **kw))
        print(json.dumps(dict(stage=stage, seconds=elapsed, **kw)), flush=True)
        if elapsed > 1800: raise RuntimeError('Registered 1800-second camera budget exceeded.')
    for r in rows:
        assert sha(r['image']) == r['source_sha256'] and sha(r['mask']) == r['mask_sha256']
        mask = cv2.imread(r['mask'], cv2.IMREAD_GRAYSCALE)
        person = cv2.dilate((mask>127).astype(np.uint8), np.ones((65,65),np.uint8))
        for box in r.get('other_person_boxes_xyxy', []):
            x0,y0,x1,y1 = np.rint(box).astype(int)
            person[max(0,y0-32):min(person.shape[0],y1+32),max(0,x0-32):min(person.shape[1],x1+32)] = 1
        assert cv2.imwrite(str(ROOT/'camera/background-masks'/r['name']), (1-person)*255)
    save(ROOT/'camera/plan.json', dict(source_sha256=sha(source), frames=len(rows),
        background_only=True, main_person_mask_dilation_pixels=32, whole_track_holdout_fraction=.2,
        provided_K_unchanged=True, identity_extrinsics_not_used=True, body_pose_not_used_to_estimate_camera=True,
        max_ba_points=5000, max_ba_nfev=60, camera_seconds=1800, physical_calibration_accepted=False))
    xy, tracks = sfm._fresh_tracks(ROOT, rows, ROOT/'camera', status, ROOT/'camera/background-masks')
    K = np.asarray(rows[0]['K_integer_centers'])
    assert all(np.array_equal(np.asarray(r['K_integer_centers']), K) for r in rows)
    E, report = sfm.solve_camera_geometry(xy, tracks, K, ROOT/'camera',
        settings=dict(min_frames=12, max_ba_points=5000, max_ba_nfev=60, camera_seconds=1800), status=status)
    ids = report['retained_input_indices']
    selected = []
    for j, i in enumerate(ids):
        r = copy.deepcopy(rows[i]); r.update(w2c=E[j].tolist(), official_prediction_slot=j,
            camera_kind='background_only_SIFT_Essential_PnP_BA',
            camera_registration=dict(interpolation_used=False, pose_scope='Scene-background camera candidate'))
        selected.append(r)
    metadata = dict(records=selected, video_sha256=sha(video), background_camera_recovered=True,
        coordinate_contract='K pixel corners; observations integer centers', original_frame_count=len(rows),
        retained_frame_count=len(selected), camera_report=report, physical_calibration_accepted=False)
    save(ROOT/'data/inputs.json', metadata); save(ROOT/'data/training-inputs.json', metadata)
    obs = dict(np.load(INPUT/'data/observations.npz', allow_pickle=False))
    for key, a in obs.items():
        if a.ndim and a.shape[0] == len(rows): obs[key] = a[ids]
    assert obs['frame_ids'].astype(str).tolist() == [r['frame'] for r in selected]
    np.savez_compressed(ROOT/'data/observations.npz', **obs)
    save(ROOT/'camera-ready.json', dict(frames=len(selected), source_frames=len(rows), camera_report=report,
        observations_sha256=sha(ROOT/'data/observations.npz'), image_and_mask_pixels_unchanged=True,
        quality_accepted=False))
    print('BACKGROUND_CAMERA_READY', len(selected), flush=True)


def stage(action):
    assert (ROOT/'camera-ready.json').is_file()
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS, ROOT)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    free,total = torch.cuda.mem_get_info()
    if free < 7*1024**3: raise RuntimeError('Need 7 GiB free; existing experiments are not interrupted.')
    torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
    cfg=read(ROOT/'config.json')
    if action == 'prior':
        from shuolee_portrait.prior import infer
        infer(ROOT, cfg)
    elif action == 'fit':
        from shuolee_portrait.fitting import main
        main(ROOT, cfg['fit_steps'])
        from shuolee_portrait.fit_gate import evaluate
        evaluate(ROOT, cfg)
    elif action == 'surface':
        from shuolee_portrait.clothed_surface import initialize
        initialize(ROOT, cfg)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=['camera','prior','fit','surface'])
    args=parser.parse_args()
    try:
        if args.action == 'camera': camera()
        else: stage(args.action)
    except BaseException:
        save(ROOT/(args.action+'-failure.json'), dict(stage=args.action, error=traceback.format_exc(), quality_accepted=False))
        raise

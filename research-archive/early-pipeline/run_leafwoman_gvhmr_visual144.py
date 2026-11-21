"""Add the original HMR2 and ViTPose inputs to the frozen 022 GVHMR control."""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from official_smplifyx_entry123 import sha
from run_leafwoman_gvhmr143 import (
    ARCHIVE, ASSETS, CONTACTS, OUT as BASE, REPO, SOURCE_OBS, WEIGHT, WSL, atomic, read,
)


OUT = BASE.with_name('gvhmr-visual144')
HMR = ASSETS/'hmr2/epoch=10-step=25000.ckpt'
VIT = ASSETS/'vitpose/vitpose-h-multi-coco.pth'
HMR_SHA = '2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e'
VIT_SHA = '50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc'


def prepare():
    assert not WSL and not OUT.exists()
    baseline = read(BASE/'plan.json')
    result = read(BASE/'result.json')
    assert result['plan_sha256'] == sha(BASE/'plan.json')
    assert result['prediction_sha256'] == sha(BASE/'prediction-incam.npz')
    assert baseline['contacts_sha256'] == sha(CONTACTS) and baseline['baseline_sha256'] == sha(ARCHIVE)
    assert len(baseline['all_source_samples']) == 57 and len(baseline['comparison_roster']) == 24
    paths = [
        'hmr4d/network/hmr2/hmr2.py', 'hmr4d/network/hmr2/vit.py',
        'hmr4d/network/hmr2/configs/model_config.yaml', 'hmr4d/network/hmr2/configs/smpl_mean_params.npz',
        'hmr4d/utils/preproc/vitfeat_extractor.py', 'hmr4d/utils/preproc/vitpose.py',
        'hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py',
    ]
    plan = dict(
        baseline_plan_sha256=sha(BASE/'plan.json'), baseline_prediction_sha256=result['prediction_sha256'],
        runner_sha256=sha(__file__), baseline_sha256=baseline['baseline_sha256'],
        contacts_sha256=baseline['contacts_sha256'], comparison_roster=baseline['comparison_roster'],
        HMR2_sha256=HMR_SHA, ViTPose_sha256=VIT_SHA,
        upstream={name: sha(REPO/name) for name in paths}, K_integer_centers=baseline['K_integer_centers'],
        conditions=['HMR2+RTMW', 'HMR2+ViTPose'], feature_crop='Upstream get_batch on native RGB, img_ds=1',
        pose_input='Upstream flip-test and UDP mapping. Preserve the earlier exclusion of covered lower joints.',
        camera_assumption=baseline['camera_assumption'], world_camera_verified=False,
        same_input_samples=57, same_bbox=True, same_K=True, face_hand_fitted=False, gaussian_started=False,
    )
    atomic(OUT/'plan.json', plan)
    for name in ('hmr2-rtmw', 'hmr2-vitpose'):
        atomic(OUT/name/'plan.json', plan)
    print('GVHMR022_VISUAL_PLAN_READY', sha(OUT/'plan.json'), flush=True)


def checkpoint(torch, path, digest):
    assert sha(path) == digest and not torch.serialization.get_unsafe_globals_in_checkpoint(path)
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)['state_dict']


def progress(stage, count=0):
    atomic(OUT/'progress.json', dict(stage=stage, done=count, total=57, pid=os.getpid()))
    print('GVHMR022_VISUAL', stage, count, flush=True)


def frontends(torch, boxes, contacts):
    from PIL import Image
    from hmr4d.network.hmr2 import HMR2
    from hmr4d.network.hmr2.configs import get_config
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    from hmr4d.utils.preproc.vitpose_pytorch import build_model

    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 8*2**30
    progress('native_crops')
    crops = []
    for row, bbox in zip(contacts, boxes, strict=True):
        path = Path(row['source_image'])
        assert sha(path) == row['source_sha256']
        with Image.open(path) as source:
            assert source.size == (2704, 1520)
            rgb = np.asarray(source.convert('RGB'))
        image, box = get_batch(rgb[None], bbox[None], img_ds=1.0, path_type='np')
        assert torch.equal(box.float(), bbox[None])
        crops.append(image)
    imgs = torch.cat(crops)
    del crops

    progress('load_hmr2')
    cfg = get_config(str(REPO/'hmr4d/network/hmr2/configs/model_config.yaml'))
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.freeze()
    weights = checkpoint(torch, HMR, HMR_SHA)
    # Construct directly on CUDA to avoid a second resident copy of the model on the host.
    with torch.device('cuda'):
        model = HMR2(cfg)
    model.load_state_dict({k: v for k, v in weights.items() if k.split('.')[0] in ('backbone', 'smpl_head')}, strict=True)
    del weights
    model = model.cuda().eval()
    tokens = []
    with torch.inference_mode():
        for start in range(0, len(imgs), 4):
            tokens.append(model({'img': imgs[start:start+4].cuda()}).cpu())
            progress('hmr2_features', min(start+4, len(imgs)))
    features = torch.cat(tokens).numpy()
    assert features.shape == (57, 1024) and np.isfinite(features).all()
    del model
    torch.cuda.empty_cache()
    np.save(OUT/'hmr2-features.npy', features, allow_pickle=False)

    progress('load_vitpose')
    weights = checkpoint(torch, VIT, VIT_SHA)
    with torch.device('cuda'):
        pose = build_model('ViTPose_huge_coco_256x192', checkpoint=None)
    pose.load_state_dict(weights, strict=True)
    del weights
    pose = pose.cuda().eval()
    # Original extractor with the verified model instance and small bounded batches.
    extractor = VitPoseExtractor.__new__(VitPoseExtractor)
    extractor.pose, extractor.flip_test, extractor.tqdm_leave = pose, True, False
    points = []
    for start in range(0, len(imgs), 4):
        points.append(extractor.extract(imgs[start:start+4], boxes[start:start+4]))
        progress('vitpose_points', min(start+4, len(imgs)))
    keypoints = torch.cat(points).numpy()
    assert keypoints.shape == (57, 17, 3) and np.isfinite(keypoints).all()
    np.save(OUT/'vitpose-keypoints.npy', keypoints, allow_pickle=False)
    progress('frontends_complete', 57)


def reconstruct(torch, boxes, baseline, plan):
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.smplx_utils import make_smplx
    import hmr4d.model.gvhmr.gvhmr_pl_demo
    import hmr4d.model.gvhmr.utils.endecoder
    import hmr4d.network.gvhmr.relative_transformer

    features = np.load(OUT/'hmr2-features.npy', allow_pickle=False)
    vitpose = np.load(OUT/'vitpose-keypoints.npy', allow_pickle=False).astype(np.float32)
    assert features.shape == (57, 1024) and np.isfinite(features).all()
    assert vitpose.shape == (57, 17, 3) and np.isfinite(vitpose).all()
    vitpose[:, 11:, 2] = 0
    with np.load(BASE/'prediction-incam.npz', allow_pickle=False) as saved:
        rtmw, frames = saved['observed_keypoints'], saved['frame_ids']
    assert sha(SOURCE_OBS) == baseline['observations_sha256']
    with np.load(SOURCE_OBS, allow_pickle=False) as z:
        assert np.array_equal(frames, z['frame_ids'])
        assert np.array_equal(rtmw[..., :2], z['keypoints'][:, :17].astype(np.float32))
        assert np.array_equal(rtmw[:, :11, 2], z['scores'][:, :11].astype(np.float32))
        assert not rtmw[:, 11:, 2].any()

    register_store_gvhmr()
    with initialize_config_module(version_base='1.3', config_module='hmr4d.configs'):
        config = compose(config_name='demo', overrides=['video_name=video22_gvhmr143'])
    model = instantiate(config.model, _recursive_=False).eval()
    model.load_state_dict(checkpoint(torch, WEIGHT, baseline['checkpoint_sha256']), strict=True)
    body = make_smplx('supermotion').eval()
    K = torch.tensor(plan['K_integer_centers'], dtype=torch.float32)
    angles = compute_cam_angvel(torch.eye(3).repeat(len(frames), 1, 1))
    assert torch.equal(angles, torch.tensor([1., 0, 0, 0, 1, 0]).repeat(len(frames), 1))
    for name, points in (('hmr2-rtmw', rtmw), ('hmr2-vitpose', vitpose)):
        start = time.monotonic()
        progress(name)
        batch = dict(length=torch.tensor([len(frames)]), obs=normalize_kp2d(torch.from_numpy(points), boxes)[None],
                     bbx_xys=boxes[None], K_fullimg=K[None, None].expand(1, len(frames), 3, 3),
                     cam_angvel=angles[None], f_imgseq=torch.from_numpy(features)[None])
        with torch.inference_mode():
            raw = model.pipeline.forward(batch, train=False, postproc=False, static_cam=False)
            params = {k: v[0] for k, v in raw['pred_smpl_params_incam'].items()}
            chunks = [body(**{k: v[i:i+4] for k, v in params.items()}) for i in range(0, len(frames), 4)]
            vertices = torch.cat([c.vertices for c in chunks]).numpy()
            joints = torch.cat([c.joints for c in chunks]).numpy()
        assert vertices.shape == (57, 10475, 3) and np.isfinite(vertices).all() and (vertices[..., 2] > .05).all()
        assert np.isfinite(joints).all() and not torch.cuda.is_initialized()
        path = OUT/name/'prediction-incam.npz'
        arrays = {f'incam_{k}': v.numpy() for k, v in params.items()}
        np.savez_compressed(path, **arrays, vertices_camera=vertices, joints_camera=joints, faces=np.asarray(body.faces),
                           observed_keypoints=points, K_integer_centers=np.asarray(plan['K_integer_centers']), frame_ids=frames)
        atomic(OUT/name/'result.json', dict(plan_sha256=sha(OUT/name/'plan.json'), prediction_sha256=sha(path),
            source_samples=len(frames), inference=name, seconds=time.monotonic()-start, real_prediction=True,
            world_camera_verified=False, face_hand_fitted=False, gaussian_started=False, quality_accepted=False))
    progress('predictions_complete', 57)


def run(stage):
    assert WSL
    plan = read(OUT/'plan.json')
    baseline = read(BASE/'plan.json')
    assert plan['runner_sha256'] == sha(__file__) and plan['baseline_plan_sha256'] == sha(BASE/'plan.json')
    assert plan['baseline_prediction_sha256'] == sha(BASE/'prediction-incam.npz')
    assert plan['contacts_sha256'] == sha(CONTACTS) and all(sha(REPO/p) == h for p, h in plan['upstream'].items())
    contacts = read(CONTACTS)['records']
    assert len(contacts) == 57
    for row, entry in zip(contacts, baseline['all_source_samples'], strict=True):
        assert (row['frame'], row['source_sha256'], row['initial_detector_box']) == (
            entry['frame'], entry['source_rgb_sha256'], entry['box'])

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch
    import hmr4d
    hmr4d.PROJ_ROOT = ASSETS
    torch.set_num_threads(1)
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    bbox = torch.tensor([r['initial_detector_box'] for r in contacts], dtype=torch.float32)
    boxes = get_bbx_xys_from_xyxy(bbox, base_enlarge=1.2)
    if stage == 'frontends':
        assert not (OUT/'progress.json').exists()
        frontends(torch, boxes, contacts)
    else:
        assert read(OUT/'progress.json')['stage'] == 'frontends_complete'
        reconstruct(torch, boxes, baseline, plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'frontends', 'predict'))
    stage = parser.parse_args().stage
    prepare() if stage == 'prepare' else run(stage)

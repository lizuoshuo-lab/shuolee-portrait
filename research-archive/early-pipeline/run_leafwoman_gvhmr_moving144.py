"""Run the upstream moving-camera demo model with verified 022 input identities."""
import os
import sys
import time

import numpy as np

from official_smplifyx_entry123 import sha
from run_leafwoman_gvhmr143 import ASSETS, CONTACTS, OUT as BASE, REPO, WEIGHT, WSL, atomic, read
from run_leafwoman_gvhmr_visual144 import OUT as VISUAL, checkpoint


OUT = VISUAL/'hmr2-vitpose-moving'


def main():
    assert WSL and not OUT.exists()
    camera = read(VISUAL/'vo-result.json')
    assert camera['complete_numeric_chain'] and camera['valid_rotations'] == 56
    assert not camera['physical_camera_verified']
    base = read(BASE/'plan.json')
    plan = read(VISUAL/'plan.json')
    assert plan['contacts_sha256'] == sha(CONTACTS) and plan['baseline_plan_sha256'] == sha(BASE/'plan.json')
    assert camera['plan_sha256'] == sha(VISUAL/'vo-plan.json')
    contacts = read(CONTACTS)['records']
    names = np.asarray([r['frame'] for r in contacts])
    expected_pairs = list(zip(names[:-1], names[1:], strict=True))
    assert [tuple(r['pair']) for r in camera['pairs']] == expected_pairs
    rotations_path = VISUAL/'vo-rotations.npy'
    rotations = np.load(rotations_path, allow_pickle=False)
    assert rotations.shape == (57, 3, 3) and np.array_equal(rotations[0], np.eye(3))
    assert np.max(abs(np.matmul(rotations.transpose(0, 2, 1), rotations)-np.eye(3))) < 1e-6
    assert np.max(abs(np.linalg.det(rotations)-1)) < 1e-6
    for i, record in enumerate(camera['pairs']):
        assert np.allclose(rotations[i+1], np.asarray(record['relative_R'])@rotations[i], rtol=0, atol=1e-10)

    features_path = VISUAL/'hmr2-features.npy'
    pose_path = VISUAL/'vitpose-keypoints.npy'
    features = np.load(features_path, allow_pickle=False)
    points = np.load(pose_path, allow_pickle=False)
    assert features.shape == (57, 1024) and points.shape == (57, 17, 3)
    assert np.isfinite(features).all() and np.isfinite(points).all()
    main_model = REPO/'hmr4d/model/gvhmr/gvhmr_pl_demo.py'
    atomic(OUT/'plan.json', dict(baseline_sha256=base['baseline_sha256'], comparison_roster=base['comparison_roster'],
        runner_sha256=sha(__file__), visual_plan_sha256=sha(VISUAL/'plan.json'), vo_result_sha256=sha(VISUAL/'vo-result.json'),
        vo_rotations_sha256=sha(rotations_path), features_sha256=sha(features_path), keypoints_sha256=sha(pose_path),
        contacts_sha256=sha(CONTACTS), upstream_predictor_sha256=sha(main_model), K_integer_centers=plan['K_integer_centers'],
        graph='Original DemoPL.predict with HMR2 tokens, original unmasked ViTPose COCO17, VO angular conditioning, postprocess and IK',
        input='57 exact native 3fps samples with frozen detector boxes and K. No stock K estimation, no frame interpolation.',
        pose_prior_only=True, world_camera_verified=False, gaussian_started=False))
    atomic(OUT/'progress.json', dict(pid=os.getpid(), stage='moving_camera_inference'))

    started = time.monotonic()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch
    import hmr4d
    hmr4d.PROJ_ROOT = ASSETS
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.smplx_utils import make_smplx
    import hmr4d.model.gvhmr.gvhmr_pl_demo
    import hmr4d.model.gvhmr.utils.endecoder
    import hmr4d.network.gvhmr.relative_transformer

    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 5*2**30
    register_store_gvhmr()
    with initialize_config_module(version_base='1.3', config_module='hmr4d.configs'):
        config = compose(config_name='demo', overrides=['video_name=video22_gvhmr143'])
    model = instantiate(config.model, _recursive_=False)
    model.load_state_dict(checkpoint(torch, WEIGHT, base['checkpoint_sha256']), strict=True)
    model = model.cuda().eval()
    K = torch.tensor(plan['K_integer_centers'], dtype=torch.float32)
    bbox = torch.tensor([r['initial_detector_box'] for r in contacts], dtype=torch.float32)
    boxes = get_bbx_xys_from_xyxy(bbox, base_enlarge=1.2)
    angles = compute_cam_angvel(torch.from_numpy(rotations.astype(np.float32)))
    assert torch.isfinite(angles).all() and not torch.equal(angles, angles[:1].expand_as(angles))
    data = dict(length=torch.tensor(len(names)), kp2d=torch.from_numpy(points.astype(np.float32)),
                bbx_xys=boxes, K_fullimg=K.expand(len(names), 3, 3), cam_angvel=angles,
                f_imgseq=torch.from_numpy(features))
    with torch.inference_mode():
        result = model.predict(data, static_cam=False)
        params = result['smpl_params_incam']
        body = make_smplx('supermotion').cuda().eval()
        chunks = [body(**{k: v[i:i+4] for k, v in params.items()}) for i in range(0, len(names), 4)]
        vertices = torch.cat([c.vertices for c in chunks]).cpu().numpy()
        joints = torch.cat([c.joints for c in chunks]).cpu().numpy()
    assert vertices.shape == (57, 10475, 3) and np.isfinite(vertices).all() and (vertices[..., 2] > .05).all()
    assert np.isfinite(joints).all()
    arrays = {f'incam_{k}': v.cpu().numpy() for k, v in params.items()}
    path = OUT/'prediction-incam.npz'
    np.savez_compressed(path, **arrays, vertices_camera=vertices, joints_camera=joints, faces=np.asarray(body.faces),
                       observed_keypoints=points, K_integer_centers=np.asarray(plan['K_integer_centers']), frame_ids=names)
    atomic(OUT/'result.json', dict(plan_sha256=sha(OUT/'plan.json'), prediction_sha256=sha(path),
        source_samples=len(names), real_prediction=True, moving_rotation_used=True, upstream_postprocess_used=True,
        seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
        world_camera_verified=False, face_hand_fitted=False, gaussian_started=False, quality_accepted=False))
    atomic(OUT/'progress.json', dict(pid=None, stage='complete'))
    print('GVHMR022_MOVING_PREDICTION', vertices.shape, flush=True)


if __name__ == '__main__':
    main()

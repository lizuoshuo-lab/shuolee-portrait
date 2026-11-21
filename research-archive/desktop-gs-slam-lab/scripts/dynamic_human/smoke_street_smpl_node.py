"""Actual SMPLNodes initialization and pose gradients on CPU, no extra GPU job."""
from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
REPO = Path('/opt/gs-slam-lab/dynamic-human/repos/drivestudio')
OUT = LAB / 'outputs/dynamic-human-smpl-010'
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from models.nodes.smpl import SMPLNodes
from pytorch3d.transforms import matrix_to_quaternion
from street_capture_time_adapter import camera_to_reference


def main():
    torch.set_num_threads(4)
    torch.manual_seed(73)
    np.random.seed(73)
    started = time.perf_counter()
    archive = np.load(LAB / 'outputs/dynamic-human-street-005/humanpose-validated-arrays.npz')
    frames = list(range(128, 133))
    assert archive['79_valid_mask'][frames].all()
    records = json.loads((SCENE / 'instances/instances_info.json').read_text())['79']['frame_annotations']
    reference = np.linalg.inv(np.loadtxt(SCENE / 'ego_pose/000.txt'))
    cameras = {c: camera_to_reference(SCENE, c, 0, 156)[0] for c in range(7)}
    quats, centers, sizes = [], [], []
    for frame in frames:
        cam = int(archive['79_selected_cam_idx'][frame])
        root = cameras[cam][frame, :3, :3] @ torch.from_numpy(archive['79_global_orient'][frame, 0]).float()
        quats.append(matrix_to_quaternion(torch.cat([root[None], torch.from_numpy(archive['79_body_pose'][frame]).float()])))
        i = records['frame_idx'].index(frame)
        centers.append((reference @ np.asarray(records['obj_to_world'][i]))[:3, 3])
        sizes.append(records['box_size'][i])
    beta_frame = int(np.flatnonzero(archive['79_valid_mask'])[0])
    entry = dict(smpl_quats=torch.stack(quats), smpl_trans=torch.tensor(np.stack(centers), dtype=torch.float32),
        frame_info=torch.ones(5, dtype=torch.bool), smpl_betas=torch.from_numpy(archive['79_betas'][beta_frame]).float(),
        size=torch.tensor(np.mean(sizes, axis=0), dtype=torch.float32))
    cfg = OmegaConf.load(REPO / 'configs/omnire.yaml')
    ctrl = OmegaConf.merge(cfg.trainer.gaussian_ctrl_general_cfg, cfg.model.SMPLNodes.ctrl)
    node = SMPLNodes(class_name='SMPLNodes', ctrl=ctrl, reg=cfg.model.SMPLNodes.reg,
        scene_scale=105.91646575927734, num_train_images=987, device=torch.device('cpu'))
    node.create_from_pcd({79: entry})
    assert node.num_points == 6890 and node.num_instances == 1 and node.num_frames == 5
    assert node.use_voxel_deformer
    node.set_cur_frame(2)
    node.in_test_set = False
    means, rotations = node.transform_means_and_quats(node._means, node._quats)
    assert means.shape == (6890, 3) and rotations.shape == (6890, 4)
    assert torch.isfinite(means).all() and torch.isfinite(rotations).all()
    # A discarded algebraic gradient probe, not a photometric fit or alignment.
    probe = means[::97].square().mean() + 0.01 * rotations[::97].square().mean()
    probe.backward()
    gradients = {}
    for name in ('_means', 'instances_trans', 'instances_quats', 'smpl_qauts'):
        value = getattr(node, name).grad
        assert value is not None and torch.isfinite(value).all(), name
        gradients[name] = float(value.abs().max())
        assert gradients[name] > 0, name
    node.in_test_set = True
    with torch.no_grad():
        heldout_means, heldout_rotations = node.transform_means_and_quats(node._means, node._quats)
        assert torch.isfinite(heldout_means).all() and torch.isfinite(heldout_rotations).all()
    report = dict(status='actual_SMPLNodes_cpu_initialization_pose_forward_backward_and_heldout_interpolation_passed',
        frames=frames, instance=79, vertices=6890, actual_author_voxel_deformer=True,
        gradient_max=gradients, seconds=time.perf_counter() - started,
        heldout_mean_max_difference_m=float((heldout_means - means.detach()).abs().max()),
        device='cpu', optimized_steps=0,
        boundary='Single real person, five-frame algebraic module check. No joint scene rendering, alignment improvement, RGB training, or quality claim.')
    (OUT / 'smpl-node-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

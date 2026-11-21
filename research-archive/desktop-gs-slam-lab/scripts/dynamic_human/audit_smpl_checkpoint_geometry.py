"""CPU-only inspection of the actual trained SMPL node; no extra GPU job."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
REPO = Path('/opt/gs-slam-lab/dynamic-human/repos/drivestudio')
RUN = LAB / 'outputs/dynamic-human-street-smpl-joint-011/articulated'
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from models.nodes.smpl import SMPLNodes
from street_capture_time_adapter import camera_to_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=3000)
    args = parser.parse_args()
    checkpoint = RUN / f'checkpoint-{args.updates:05d}.pt'
    out = RUN.parent / f'geometry-{args.updates:05d}-cpu'
    assert checkpoint.exists()
    assert not (out / 'receipt.json').exists(), 'Preserve completed diagnostics'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    start = time.perf_counter()
    cfg = OmegaConf.load(RUN / 'config.yaml')
    ctrl = OmegaConf.merge(cfg.trainer.gaussian_ctrl_general_cfg, cfg.model.SMPLNodes.ctrl)
    node = SMPLNodes(class_name='SMPLNodes', ctrl=ctrl, reg=cfg.model.SMPLNodes.reg,
        scene_scale=105.91646575927734, num_train_images=987, device=torch.device('cpu'))
    # Locally generated model checkpoint; mmap avoids loading background tensors.
    checkpoint_state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert checkpoint_state['step'] == args.updates - 1
    state = checkpoint_state['models']['SMPLNodes']
    node.load_state_dict(dict(state), strict=True)
    restored = node.state_dict()
    assert state.keys() == restored.keys()
    for name in state:
        if isinstance(state[name], torch.Tensor):
            assert torch.equal(state[name], restored[name]), name
        else:
            assert state[name] == restored[name], name
    node.step = checkpoint_state['step']
    node.eval()
    initialization = json.loads((RUN / 'initialization.json').read_text())
    model_id = next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance'] == 79)
    point_mask = node.point_ids[:, 0] == model_id
    assert int(point_mask.sum()) == 6890
    reference_vertices, faces = node.template.get_init_vf()
    faces = faces.cpu().numpy()
    masks = node.instances_fv[:, model_id]
    assert int(masks.sum()) == 100
    rows = []
    with torch.no_grad():
        for frame, cam in [(129, 4), (130, 4), (131, 4), (149, 6), (150, 6), (151, 6)]:
            node.set_cur_frame(frame)
            node.in_test_set = frame % 10 == 0
            means, rotations = node.transform_means_and_quats(node._means, node._quats)
            world = means[point_mask]
            assert torch.isfinite(world).all() and torch.isfinite(rotations).all()
            matrix = camera_to_reference(SCENE, cam, 0, 156)[0][frame]
            inverse = torch.linalg.inv(matrix)
            xyz = (world @ inverse[:3, :3].T + inverse[:3, 3]).numpy()
            assert xyz[:, 2].min() > 0.1
            fx, fy, cx, cy = np.loadtxt(SCENE / 'intrinsics' / f'{cam}.txt')[:4]
            uv = xyz[:, :2] / xyz[:, 2:] * [fx, fy] + [cx, cy]
            image_path = SCENE / 'images' / f'{frame:03d}_{cam}.jpg'
            original = Image.open(image_path).convert('RGB')
            low = np.maximum(np.floor(uv.min(0) - 80), 0).astype(int)
            high = np.minimum(np.ceil(uv.max(0) + 80), original.size).astype(int)
            assert np.all(high > low)
            layer = Image.new('RGBA', original.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for face in faces[np.argsort(xyz[faces, 2].mean(1))[::-1]]:
                draw.polygon([tuple(map(float, p)) for p in uv[face]], fill=(20, 160, 230, 85))
            overlay = Image.alpha_composite(original.convert('RGBA'), layer).convert('RGB')
            crop = tuple(map(int, [*low, *high]))
            a, b = original.crop(crop), overlay.crop(crop)
            pair = Image.new('RGB', (a.width * 2, a.height))
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width, 0))
            name = f'frame{frame:03d}-camera{cam}-person79.png'
            pair.save(out / name)
            rows.append(dict(frame=frame, camera=cam, heldout=node.in_test_set, file=name,
                projected_bounds=[*uv.min(0).tolist(), *uv.max(0).tolist()], crop=list(crop),
                world_centroid_m=world.mean(0).tolist()))
    mean_offset = (node._means[point_mask] - reference_vertices[model_id]).norm(dim=-1)
    report = dict(status='actual_trained_SMPL_node_restored_exactly_and_projected_visual_review_pending',
        optimizer_updates=args.updates, device='cpu', vertices=6890, model_index=model_id,
        tensor_state_restoration='Exact torch.equal on every saved tensor',
        optimized_steps_in_this_diagnostic=0, views=rows,
        canonical_vertex_offset_m=dict(mean=float(mean_offset.mean()), max=float(mean_offset.max())),
        seconds=time.perf_counter()-start,
        boundaries=['Connects learned Gaussian centers using original SMPL topology; not the Gaussian RGB render.',
            'Includes training views129/131/149/151 and heldout130/150, explicitly labeled.',
            'Actual author heldout pose interpolation retained; no fitting to these images.',
            'No scene occlusion compositing, no clothing surface guarantee.',
            'Source face anonymization retained. Licensed template remains private.'])
    (out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

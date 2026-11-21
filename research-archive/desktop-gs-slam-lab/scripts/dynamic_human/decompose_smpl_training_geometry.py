"""Training-view-only CPU diagnostic: pose versus learned Gaussian geometry."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import audit_smpl_checkpoint_geometry as context


def main():
    torch.set_num_threads(3)
    updates = 15000
    out = context.LAB / 'outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000'
    assert not (out / 'receipt.json').exists(), 'Preserve finished diagnostics'
    out.mkdir(exist_ok=True)
    checkpoint = context.RUN / f'checkpoint-{updates:05d}.pt'
    start = time.perf_counter()
    cfg = OmegaConf.load(context.RUN / 'config.yaml')
    ctrl = OmegaConf.merge(cfg.trainer.gaussian_ctrl_general_cfg, cfg.model.SMPLNodes.ctrl)
    node = context.SMPLNodes(class_name='SMPLNodes', ctrl=ctrl, reg=cfg.model.SMPLNodes.reg,
        scene_scale=105.91646575927734, num_train_images=987, device=torch.device('cpu'))
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert saved['step'] == updates - 1
    state = saved['models']['SMPLNodes']
    node.load_state_dict(dict(state), strict=True)
    assert state.keys() == node.state_dict().keys()
    for k, value in node.state_dict().items():
        expected = state[k]
        assert torch.equal(value, expected) if isinstance(value, torch.Tensor) else value == expected
    node.step = saved['step']
    node.eval()
    initial = json.loads((context.RUN / 'initialization.json').read_text())
    model_id = next(r['model_index'] for r in initial['instances']['SMPLNodes'] if r['author_instance'] == 79)
    point_mask = node.point_ids[:, 0] == model_id
    canonical, faces = node.template.get_init_vf()
    faces = faces.cpu().numpy()
    assert point_mask.sum() == 6890
    rows = []
    with torch.no_grad():
        for frame, camera in [(129, 4), (131, 4), (149, 6), (151, 6)]:
            assert frame % 10 != 0, 'No heldout images used for this fitting diagnosis'
            node.set_cur_frame(frame)
            node.in_test_set = False
            learned = node.transform_means(node._means)[point_mask]
            # Transform untouched canonical vertices with optimized joint/root poses
            # and fixed original LBS weights. No learned offsets or voxel correction.
            node.use_voxel_deformer = False
            pose_only = node.transform_means(canonical.reshape(-1, 3))[point_mask]
            node.use_voxel_deformer = True
            assert torch.isfinite(learned).all() and torch.isfinite(pose_only).all()
            matrix = context.camera_to_reference(context.SCENE, camera, 0, 156)[0][frame]
            inverse = torch.linalg.inv(matrix)
            fx, fy, cx, cy = np.loadtxt(context.SCENE / 'intrinsics' / f'{camera}.txt')[:4]
            projected = []
            for world in [pose_only, learned]:
                xyz = (world @ inverse[:3, :3].T + inverse[:3, 3]).numpy()
                assert xyz[:, 2].min() > .1
                uv = xyz[:, :2] / xyz[:, 2:] * [fx, fy] + [cx, cy]
                projected.append((xyz, uv))
            source = context.SCENE / 'images' / f'{frame:03d}_{camera}.jpg'
            original = Image.open(source).convert('RGB')
            both_uv = np.concatenate([x[1] for x in projected])
            low = np.maximum(np.floor(both_uv.min(0) - 80), 0).astype(int)
            high = np.minimum(np.ceil(both_uv.max(0) + 80), original.size).astype(int)
            crop = tuple(map(int, [*low, *high]))
            panels = [original.crop(crop)]
            for xyz, uv in projected:
                layer = Image.new('RGBA', original.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)
                for face in faces[np.argsort(xyz[faces, 2].mean(1))[::-1]]:
                    draw.polygon([tuple(map(float, p)) for p in uv[face]], fill=(20, 160, 230, 85))
                panels.append(Image.alpha_composite(original.convert('RGBA'), layer).convert('RGB').crop(crop))
            width, height = panels[0].size
            canvas = Image.new('RGB', (3 * width, height + 74), 'white')
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
            for col, (label, panel) in enumerate(zip(['Observed training image', 'Optimized pose / fixed template', 'Actual learned Gaussian centers'], panels)):
                draw.text((col * width + 10, 9), label, fill='#182d39', font=font)
                canvas.paste(panel, (col * width, 36))
            draw.text((10, height + 43), f'Frame {frame} / camera {camera} | native pixels | projected geometry only; not RGB rendering', fill='#182d39', font=font)
            filename = f'frame{frame:03d}-camera{camera}-decomposition.png'
            canvas.save(out / filename)
            distance = (learned - pose_only).norm(dim=-1)
            rows.append(dict(frame=frame, camera=camera, heldout=False, file=filename, crop=list(crop),
                learned_vs_pose_only_vertex_distance_m=dict(mean=float(distance.mean()), median=float(distance.median()),
                    p95=float(torch.quantile(distance, .95)), max=float(distance.max())),
                pose_only_bounds=projected[0][1][[np.argmin(projected[0][1][:, 0]), np.argmax(projected[0][1][:, 0])]].tolist()))
    report = dict(status='training_only_geometry_decomposition_computed_visual_review_pending',
        updates=updates, model_index=model_id, checkpoint=str(checkpoint.relative_to(context.LAB)),
        tensor_restoration='Exact torch.equal on every saved SMPL tensor before diagnostic interventions',
        optimized_steps=0, rows=rows, seconds=time.perf_counter()-start,
        boundaries=['Fixed-template panel retains optimized root translation and all joint rotations.',
            'Fixed-template panel removes Gaussian offsets and learned voxel skinning corrections together; not a single-factor causal study.',
            'Actual learned centers connected by original topology are a diagnostic, not an extracted or textured surface.',
            'No occlusion compositor, original face anonymization retained.',
            'Four training views only; no new checkpoint, no changes to active training.'])
    (out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

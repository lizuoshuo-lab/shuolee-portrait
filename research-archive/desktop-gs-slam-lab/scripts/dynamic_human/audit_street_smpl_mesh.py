"""CPU-only SMPL loading and actual AV2 person-79 mesh projection audit.

Uses the author's canonical-pose LBS and bounding-box center correction. The
overlays are alignment diagnostics, not learned reconstructions or ground truth.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
REPO = Path('/opt/gs-slam-lab/dynamic-human/repos/drivestudio')
SCENE = LAB / 'data/dynamic-human/argoverse2/processed/training/000'
OUT = LAB / 'outputs/dynamic-human-smpl-010'
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from models.human_body import SMPLTemplate
from pytorch3d.transforms import matrix_to_quaternion
from street_capture_time_adapter import camera_to_reference


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', choices=['center', 'box-bottom'], default='center')
    args = parser.parse_args()
    if args.anchor == 'box-bottom':
        OUT = OUT / 'box-bottom-diagnostic'
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'mesh-projection-audit.json').exists(), 'Preserve the previous audit'
    torch.set_num_threads(4)
    model_path = REPO / 'smpl_models/SMPL_NEUTRAL.pkl'
    poses = np.load(LAB / 'outputs/dynamic-human-street-005/humanpose-validated-arrays.npz')
    identity = 79
    valid = poses[f'{identity}_valid_mask']
    frame_ids = np.flatnonzero(valid)
    # Match DrivingDataset.get_init_objects: first valid frame's shape is fixed.
    beta = torch.from_numpy(poses[f'{identity}_betas'][frame_ids[0]:frame_ids[0]+1]).float()
    template = SMPLTemplate(str(model_path), 1, beta, 'da_pose', use_voxel_deformer=False)
    vertices, faces = template.get_init_vf()
    assert vertices.shape == (1, 6890, 3) and faces.shape == (13776, 3)
    assert template.W.shape == (1, 6890, 24)
    assert torch.isfinite(vertices).all()
    torch.testing.assert_close(template.W.sum(-1), torch.ones(1, 6890), atol=1e-5, rtol=1e-5)
    selected_cams = poses[f'{identity}_selected_cam_idx']
    camera_matrices = {cam: camera_to_reference(SCENE, cam, 0, 156)[0] for cam in range(7)}
    annotations = json.loads((SCENE / 'instances/instances_info.json').read_text())[str(identity)]
    record = annotations['frame_annotations']
    reference_inv = np.linalg.inv(np.loadtxt(SCENE / 'ego_pose/000.txt'))
    sizes, overlays, anchor_offsets = [], [], []
    up = torch.from_numpy(reference_inv[:3, :3] @ np.array([0., 0., 1.])).float()
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    preview_frames = {60, 90, 120, 130, 140, 150}
    faces_np = faces.numpy()
    with torch.no_grad():
        for frame in frame_ids:
            frame = int(frame)
            camera = int(selected_cams[frame])
            root = camera_matrices[camera][frame, :3, :3] @ torch.from_numpy(poses[f'{identity}_global_orient'][frame, 0]).float()
            body = torch.from_numpy(poses[f'{identity}_body_pose'][frame]).float()
            quaternions = matrix_to_quaternion(torch.cat([root[None], body]))[None]
            W, A = template(masked_theta=quaternions, instances_mask=torch.tensor([True]))
            transform = torch.einsum('bnj,bjrc->bnrc', W, A)
            deformed = torch.einsum('bnij,bnj->bni', transform[:, :, :3, :3], vertices) + transform[:, :, :3, 3]
            lo, hi = deformed.amin(1), deformed.amax(1)
            local_shift = (lo + hi) / 2
            annotation_index = record['frame_idx'].index(frame)
            box_pose = reference_inv @ np.asarray(record['obj_to_world'][annotation_index])
            center = torch.tensor(box_pose[:3, 3], dtype=torch.float32)[None]
            world = deformed - local_shift[:, None] + center[:, None]
            assert torch.isfinite(world).all()
            torch.testing.assert_close((world.amin(1) + world.amax(1)) / 2, center, atol=1e-4, rtol=1e-5)
            corners = signs * np.asarray(record['box_size'][annotation_index])[None] / 2
            corners = corners @ box_pose[:3, :3].T + box_pose[:3, 3]
            box_bottom = float((corners @ up.numpy()).min())
            mesh_bottom = float((world[0] @ up).min())
            offset = box_bottom - mesh_bottom
            anchor_offsets.append(dict(frame=frame, offset_m=offset))
            if args.anchor == 'box-bottom':
                world = world + offset * up[None, None]
                assert abs(float((world[0] @ up).min()) - box_bottom) < 1e-4
            sizes.append((hi - lo)[0].tolist())
            if frame not in preview_frames:
                continue
            for cam in sorted({camera} | ({4, 6} if frame >= 130 else set())):
                image_path = SCENE / 'images' / f'{frame:03d}_{cam}.jpg'
                source = Image.open(image_path).convert('RGB')
                w, h = source.size
                inv_cam = torch.linalg.inv(camera_matrices[cam][frame])
                local = world[0] @ inv_cam[:3, :3].T + inv_cam[:3, 3]
                intrinsic = np.loadtxt(SCENE / 'intrinsics' / f'{cam}.txt')
                fx, fy, cx, cy = intrinsic[:4]
                xyz = local.numpy()
                if xyz[:, 2].min() <= 0.1:
                    continue
                uv = xyz[:, :2] / xyz[:, 2:] * [fx, fy] + [cx, cy]
                low = np.maximum(np.floor(uv.min(0) - [40, 40]), 0).astype(int)
                high = np.minimum(np.ceil(uv.max(0) + [40, 40]), [w, h]).astype(int)
                if np.any(high - low < 24):
                    continue
                layer = Image.new('RGBA', source.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(layer)
                # Actual posed mesh triangles sorted by camera-space depth.
                depth = xyz[faces_np, 2].mean(1)
                for face in faces_np[np.argsort(depth)[::-1]]:
                    triangle = [tuple(map(float, p)) for p in uv[face]]
                    draw.polygon(triangle, fill=(20, 160, 230, 85))
                overlay = Image.alpha_composite(source.convert('RGBA'), layer).convert('RGB')
                prefix = f'frame{frame:03d}-camera{cam}-person{identity}'
                overlay.save(OUT / f'{prefix}-overlay.png')
                crop_box = tuple(map(int, [*low, *high]))
                original_crop, overlay_crop = source.crop(crop_box), overlay.crop(crop_box)
                pair = Image.new('RGB', (original_crop.width * 2, original_crop.height), 'white')
                pair.paste(original_crop, (0, 0))
                pair.paste(overlay_crop, (original_crop.width, 0))
                pair.save(OUT / f'{prefix}-pair.png')
                overlays.append(dict(frame=frame, camera=cam, author_pose_camera=camera,
                    original=str(image_path), crop=list(crop_box), figure=f'{prefix}-pair.png',
                    mesh_projected_box=[*uv.min(0).tolist(), *uv.max(0).tolist()],
                    depth_m=[float(xyz[:, 2].min()), float(xyz[:, 2].max())]))
    report = dict(status='cpu_template_and_100_pose_forward_checks_passed_visual_review_pending',
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        device='cpu', vertex_count=6890, triangle_count=13776, joints=24, shape_coefficients=10,
        identity=identity, track_uuid=annotations['id'], valid_pose_frames=len(frame_ids),
        anchor=args.anchor, box_bottom_alignment_offsets=anchor_offsets,
        fixed_beta_source_frame=int(frame_ids[0]), posed_extent_min_m=np.min(sizes, axis=0).tolist(),
        posed_extent_max_m=np.max(sizes, axis=0).tolist(), overlays=overlays,
        implementation='Author SMPLTemplate DA-pose skinning; selected-camera root orientation transformed to reference world; author bounding-box center matching, optionally followed by a separate annotation-bottom translation along city vertical.',
        boundaries=['Overlay is an initialization diagnostic, not a trained result.',
            'Mesh does not model clothing, bags, or scene occluders. No scene-occlusion compositing.',
            'Camera poses at image exposure; object annotation centers at LiDAR time.',
            'The bottom-anchor option uses annotated box bounds, not measured foot/ground truth; no scaling or joint changes.',
            'Author supplied pose/shape priors include heldout times; this is not motion forecasting.',
            'No human sharpness or visual alignment claim until figures are inspected.'])
    write_json('mesh-projection-audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'overlays'}, indent=2))


if __name__ == '__main__':
    main()

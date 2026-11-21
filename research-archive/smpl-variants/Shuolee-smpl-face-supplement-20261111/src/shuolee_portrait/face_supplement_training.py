"""Train and replay independently admitted facial pixels on the shared scene."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from . import runtime as rt


INTERVAL = 4
UPDATE_LIMIT = 1800
RGB_WEIGHT = .30
ALPHA_WEIGHT = .15


def masked_face_loss(rgb, alpha, target_rgb, target_alpha, valid):
    """Index admitted pixels before color or alpha arithmetic."""
    if not bool(valid.any()):
        raise ValueError('Face-only view has zero verified pixels.')
    source = target_rgb[:, valid]
    target = target_alpha[valid]
    rgb, alpha = rgb[:, valid], alpha[valid]
    straight = (rgb/alpha.clamp_min(1e-4)-source).abs().mean()
    premultiplied = (rgb-source*target).abs().mean()
    return .30*straight+.70*premultiplied, (alpha-target).abs().mean()


@dataclass
class FaceTarget:
    row: dict
    slot: int
    camera: object
    camera_record: dict
    box: tuple
    valid: torch.Tensor
    alpha: torch.Tensor
    rgb: torch.Tensor
    guard_view: object
    record: dict

    @property
    def crop(self):
        x0, y0, x1, y1 = self.box
        return slice(y0, y1), slice(x0, x1)


def _read_source(row, box):
    with Image.open(row['image']) as image:
        if image.size != tuple(row['wh']):
            raise ValueError(f'Supplemental source dimension differs for {row["frame"]}.')
        return np.asarray(image.convert('RGB').crop(box)).copy()


def prepare_targets(items, first_slot, camera, directory):
    targets = []
    for index, item in enumerate(items):
        row, box = item.row, tuple(item.box)
        slot = first_slot+index
        rgb = torch.from_numpy(_read_source(row, box)).permute(2, 0, 1).contiguous().cuda()
        valid = torch.as_tensor(item.valid, dtype=torch.bool, device='cuda')
        alpha = torch.as_tensor(item.alpha, dtype=torch.float32, device='cuda')
        if valid.shape != alpha.shape or valid.shape != rgb.shape[1:] or not bool(valid.any()):
            raise ValueError(f'Invalid admitted face raster for {row["frame"]}.')
        if not bool((alpha[valid] >= .99).all()):
            raise ValueError(f'Unreliable foreground in admitted face raster {row["frame"]}.')
        x0, y0, x1, y1 = box
        width, height = row['wh']
        inside = torch.zeros((height, width), dtype=torch.bool, device=valid.device)
        inside[y0:y1, x0:x1] = valid
        # The verifier sees no exterior label from a face-only view.
        guard_view = SimpleNamespace(box=box, rgb=rgb, inside=inside,
                                     outside=torch.zeros_like(inside), face=None)
        cam, cam_record = camera(row, slot, pose_index=slot)
        name = f'{row["frame"]}-valid.png'
        Image.fromarray(np.asarray(item.valid, dtype=np.uint8)*255).save(directory/name)
        record = dict(frame=row['frame'], role='face_only', pose_index=slot, box=list(box),
                      source_frame_index=row['source_frame_index'], source_sha256=row['source_sha256'],
                      mask_sha256=row['mask_sha256'], valid_pixels=int(valid.count_nonzero()),
                      valid_mask=name, valid_mask_sha256=rt.sha(directory/name), camera=cam_record,
                      updates=0, density_observations=0, max_density_visible_rows=0)
        targets.append(FaceTarget(row, slot, cam, cam_record, box, valid, alpha,
                                  rgb.float().div(255), guard_view, record))
    return targets


def _face_scene(model, owner):
    """Keep occluding geometry in the raster and route gradients only to head-owned rows."""
    properties = ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features')
    values = {}
    for name in properties:
        value = getattr(model, name)
        mask = owner.reshape((len(owner),)+(1,)*(value.ndim-1))
        values[name] = torch.where(mask, value, value.detach())
    return SimpleNamespace(**values, active_sh_degree=model.active_sh_degree, max_sh_degree=model.max_sh_degree)


def train_face_view(target, model, head_faces, render, pipe, background, step, *, density):
    """Accumulate one masked view into the current main Adam update."""
    previous = model.timestep
    try:
        model.select_mesh_by_timestep(target.slot)
        owner = head_faces[model.binding]
        scene = _face_scene(model, owner)
        packet = render(target.camera, scene, pipe, background)
        alpha = render(target.camera, scene, pipe, background,
                       override_color=torch.ones_like(scene.get_xyz))['render'][0]
        y, x = target.crop
        photo, coverage = masked_face_loss(packet['render'][:, y, x], alpha[y, x],
                                          target.rgb, target.alpha, target.valid)
        loss = RGB_WEIGHT*photo+ALPHA_WEIGHT*coverage
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite face-only loss for {target.row["frame"]}.')
        loss.backward()
        with torch.no_grad():
            record = target.record
            record['updates'] += 1
            record.setdefault('first_main_step', step)
            record.update(last_main_step=step, last_rgb_loss=float(photo), last_alpha_loss=float(coverage))
            if density:
                screen = packet['viewspace_points']
                supported = owner & packet['visibility_filter'] & (screen.grad[:, :2].abs().sum(1) > 0)
                area = record['valid_pixels'] / (target.camera.image_width*target.camera.image_height)
                screen.grad.mul_(area)
                model.add_densification_stats(screen, supported)
                record['density_observations'] += 1
                record['max_density_visible_rows'] = max(record['max_density_visible_rows'],
                                                          int(supported.count_nonzero()))
    finally:
        model.select_mesh_by_timestep(previous)


def save_replay(directory, targets, live, reloaded, manifest, render, pipe, black):
    """Replay named supplemental poses using the round-tripped local bound PLY."""
    from .bound_replay import _compare_geometry, _geometry
    from .bound_review_panels import _mesh_panel

    records = []
    if targets:
        poses = live.supplement_vertices
        np.savez_compressed(directory/'poses.npz', frame_ids=np.asarray([view.row['frame'] for view in targets]),
                            vertices_world=poses.cpu().numpy(), faces=live.faces.cpu().numpy(),
                            pose_indices=np.asarray([view.slot for view in targets]))
        base_sequence = reloaded.vertices_sequence
        original_live, original_reload = live.timestep, reloaded.timestep
        reloaded.vertices_sequence = torch.cat((base_sequence, poses))
        try:
            for target in targets:
                live.select_mesh_by_timestep(target.slot)
                reloaded.select_mesh_by_timestep(target.slot)
                geometry = _compare_geometry(_geometry(live), _geometry(reloaded))
                y, x = target.crop
                native = render(target.camera, reloaded, pipe, black)['render'][:, y, x]
                alpha = render(target.camera, reloaded, pipe, black,
                               override_color=torch.ones_like(reloaded.get_xyz))['render'][0, y, x]
                live_native = render(target.camera, live, pipe, black)['render'][:, y, x]
                difference = (native-live_native).abs()[:, target.valid]
                if not bool(torch.isfinite(difference).all()) or float(difference.max()) >= .005:
                    raise RuntimeError('Reloaded face-only render differs on verified pixels.')
                source = Image.fromarray(_read_source(target.row, target.box))
                valid = Image.open(directory/target.record['valid_mask']).convert('RGB')
                gaussian = Image.fromarray(native.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
                opacity = Image.fromarray(alpha.clamp(0, 1).mul(255).round().byte().cpu().numpy())
                extrinsic = np.asarray(target.row['w2c'])
                vertices = poses[target.slot-len(base_sequence)].cpu().numpy()
                vertices = vertices@extrinsic[:3, :3].T+extrinsic[:3, 3]
                k = np.asarray(target.row['K']).copy()
                k[:2, 2] -= .5
                mesh = _mesh_panel(source, target.box, vertices, live.faces.cpu().numpy(), k)
                frame = target.row['frame']
                width, height = source.size
                panel = Image.new('RGB', (4*width, height+26), '#181818')
                labels = ImageDraw.Draw(panel)
                for col, (title, part) in enumerate(zip(
                        ('Native RGB', 'Verified face pixels', 'Frozen mesh', 'Same-frame bound Gaussian'),
                        (source, valid, mesh, gaussian), strict=True)):
                    panel.paste(part, (col*width, 26))
                    labels.text((col*width+4, 5), title, fill='white')
                files = {}
                for kind, image in (('panel', panel),
                                    ('gaussian', gaussian), ('alpha', opacity)):
                    path = directory/f'{frame}-{kind}.png'
                    image.save(path)
                    files[kind] = dict(file=path.name, sha256=rt.sha(path))
                records.append(dict(target.record, files=files, geometry_parity=geometry,
                                    valid_rgb_max_abs=float(difference.max())))
        finally:
            reloaded.vertices_sequence = base_sequence
            reloaded.select_mesh_by_timestep(original_reload)
            live.select_mesh_by_timestep(original_live)
    rt.save(directory/'replay.json', dict(schema='face-supplement-replay-v1', records=records,
        shared_bound_ply_sha256=manifest['bound_ply_sha256'],
        poses_sha256=rt.sha(directory/'poses.npz') if records else None))

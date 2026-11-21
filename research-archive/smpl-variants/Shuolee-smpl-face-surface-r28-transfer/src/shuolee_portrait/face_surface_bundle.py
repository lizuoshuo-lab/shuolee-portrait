"""Verified named-motion package and native replays for the face-only refinement."""
import shutil
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from PIL import Image

from . import runtime as rt
from .bound_replay import RAW_PARAMETERS, inspect_bound_ply, save_bound_snapshot
from .duration_checkpoint import _capture_rng
from .training import _native_foreground_target, camera, image

PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
SCHEMA = 'shuolee-face-bound-v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def draw(render, cam, model, black):
    packet = render(cam, model, PIPE, black)
    alpha = render(cam, model, PIPE, black, override_color=torch.ones_like(model.get_xyz))['render'][0]
    return packet, alpha


def _parity(actual, expected):
    difference = (actual-expected).abs()
    report = dict(mean_abs=float(difference.mean()), max_abs=float(difference.max()))
    require(report['mean_abs'] < 2e-6 and report['max_abs'] < .005, f'Native replay mismatch: {report}')
    return report


def _save_drive(drive, snapshot, prepared, labels, spill):
    drive.mkdir()
    model = prepared.model
    shutil.copyfile(snapshot/'bound-local.ply', drive/'bound-local.ply')
    np.savez_compressed(
        drive/'motion.npz', vertices_world=model.vertices_sequence.cpu().numpy(),
        faces=model.faces.cpu().numpy(), frame_ids=np.asarray([row['frame'] for row in prepared.rows]),
        stable_ids=model.densification_ids.cpu().numpy(), labels=labels.cpu().numpy(),
        spill=spill.cpu().numpy())
    rt.save(drive/'drive.json', dict(
        schema=SCHEMA, sh_degree=0, reference_pose_index=prepared.reference,
        cameras=prepared.camera_records, roles=['training']*prepared.main_count
        +['pose_conditioned_holdout']*prepared.holdout_count+['face_only']*prepared.aux_count,
        files={name: rt.sha(drive/name) for name in ('bound-local.ply', 'motion.npz')},
        coordinate_space='Triangle-local parameters and named fitted-world SMPL-X meshes',
        quality_accepted=False))


def load_face_drive(drive):
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    drive = Path(drive)
    meta = rt.read(drive/'drive.json')
    require(meta['schema'] == SCHEMA and meta['sh_degree'] == 0
            and set(meta['files']) == {'bound-local.ply', 'motion.npz'}
            and all(rt.sha(drive/name) == digest for name, digest in meta['files'].items()),
            'Motion package hash or schema differs.')
    with np.load(drive/'motion.npz', allow_pickle=False) as archive:
        motion = dict(archive)
    vertices, faces, ids = motion['vertices_world'], motion['faces'], motion['frame_ids']
    roles = meta['roles']
    main_count = roles.count('training')
    holdout_count = roles.count('pose_conditioned_holdout')
    aux_count = roles.count('face_only')
    require(vertices.ndim == 3 and vertices.shape[0] == len(ids) == len(meta['cameras']) == len(roles)
            and roles == ['training']*main_count+['pose_conditioned_holdout']*holdout_count+['face_only']*aux_count
            and vertices.shape[-1] == 3 and np.isfinite(vertices).all()
            and faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < vertices.shape[1]
            and 0 <= meta['reference_pose_index'] < main_count
            and len(set(ids.tolist())) == len(ids), 'Invalid motion topology or roster.')
    owners, counts = inspect_bound_ply(drive/'bound-local.ply', len(faces))
    lineage = motion['stable_ids']
    require(lineage.shape == motion['labels'].shape == motion['spill'].shape == owners.shape
            and len(np.unique(lineage)) == len(lineage)
            and np.isin(motion['labels'], np.arange(6)).all()
            and not np.any(motion['spill'] & (motion['labels'] != 0)),
            'Invalid material lineage.')
    require(all(row['frame'] == frame and row['pose_index'] == slot
                and row['pixel_convention'] == 'pixel_corners'
                for slot, (row, frame) in enumerate(zip(meta['cameras'], ids, strict=True))),
            'Camera roster differs from the named pose order.')
    model = SMPLXGaussianModel(0)
    model.vertices_sequence = torch.from_numpy(vertices).cuda()
    model.faces = torch.from_numpy(faces).long().cuda()
    model.num_timesteps = len(vertices)
    model.load_ply(str(drive/'bound-local.ply'))
    model.binding_counter = torch.from_numpy(counts).int().cuda()
    require(torch.equal(model.binding.long().cpu(), torch.from_numpy(owners)), 'Triangle IDs changed on replay.')
    for name in RAW_PARAMETERS:
        getattr(model, name).requires_grad_(False)
    model.select_mesh_by_timestep(meta['reference_pose_index'])
    return model, meta


def _world_ply(model, path):
    from scene.gaussian_model import GaussianModel

    world = GaussianModel(0)
    world._xyz, world._scaling, world._rotation = model.get_xyz, model.get_scaling.log(), model.get_rotation
    world._features_dc, world._features_rest, world._opacity = model._features_dc, model._features_rest, model._opacity
    world.save_ply(str(path))
    loaded = GaussianModel(0)
    loaded.load_ply(str(path))
    return loaded


def _replay(snapshot, prepared, loaded, meta, render, black):
    main, auxiliary = snapshot/'drive-replay', snapshot/'face-supplement'
    main.mkdir()
    auxiliary.mkdir()
    records = {main: [], auxiliary: []}
    model = prepared.model
    for slot, (row, saved_camera, role) in enumerate(zip(prepared.rows, meta['cameras'], meta['roles'], strict=True)):
        require(saved_camera == prepared.camera_records[slot], 'Native camera changed on replay.')
        cam, _ = camera(saved_camera, slot, pose_index=slot)
        model.select_mesh_by_timestep(slot)
        loaded.select_mesh_by_timestep(slot)
        for name in ('get_xyz', 'get_rotation', 'get_scaling', 'get_features', 'get_opacity'):
            require(torch.equal(getattr(model, name), getattr(loaded, name)), f'Bound replay changed {name}.')
        require(torch.equal(model.get_covariance(), loaded.get_covariance()), 'Bound covariance changed.')
        packet, alpha = draw(render, cam, loaded, black)
        rgb = packet['render']
        difference = _parity(rgb, render(prepared.cameras[slot], model, PIPE, black)['render'])
        if role == 'face_only':
            target = prepared.targets[slot-prepared.holdout_count]
            directory, box = auxiliary, target.box
        else:
            _, _, bounds = _native_foreground_target(row)
            x0, y0, x1, y1 = bounds
            w, h = row['wh']
            directory, box = main, (max(0, x0-24), max(0, y0-24), min(w, x1+24), min(h, y1+24))
        files = {}
        for name, value in (('black', rgb), ('white', (rgb+1-alpha).clamp(0, 1)), ('alpha', alpha.expand_as(rgb))):
            path = directory/f'{row["frame"]}-{name}.png'
            image(value).crop(box).save(path)
            files[name] = path.name
        for name, key, mode in (('source', 'image', 'RGB'), ('mask', 'mask', 'L')):
            path = directory/f'{row["frame"]}-{name}.png'
            with Image.open(row[key]) as source:
                source.convert(mode).crop(box).save(path)
            files[name] = path.name
        if role == 'face_only':
            path = directory/f'{row["frame"]}-valid.png'
            Image.fromarray(target.valid.cpu().numpy().astype(np.uint8)*255).save(path)
            files['valid'] = path.name
        records[directory].append(dict(
            frame=row['frame'], pose_index=slot, box_xyxy=box, box=box, files=files,
            files_sha256={name: rt.sha(directory/path) for name, path in files.items()},
            source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
            role=role, camera=saved_camera, exact_geometry_roundtrip=True, render_parity=difference))
        if 'evaluation_role' in row:
            records[directory][-1]['evaluation_role'] = row['evaluation_role']
    for directory, rows in records.items():
        rt.save(directory/'result.json', dict(records=rows, optimizer_updates=0, quality_accepted=False))


@torch.no_grad()
def save_endpoint(root, prepared, step, config, render, black, draws, splits, labels, spill, caps, report,
                  *, training_metadata=None):
    from .export import export

    model = prepared.model
    snapshot = root/'gaussian-run'/f'raw-step-{step:06d}'
    snapshot.mkdir()
    model.select_mesh_by_timestep(prepared.reference)
    state = snapshot/'main-state.pt'
    torch.save(dict(
        step=step, model=model.capture(), stable_ids=model.densification_ids,
        next_stable_id=model.next_densification_id, labels=labels, spill=spill, caps=caps,
        draws=draws.copy(), split_events=splits.copy(), rng=_capture_rng(), optimization=config['optimization'],
        pinned_inputs=config['pinned_inputs'], frozen_and_geometry=report, **(training_metadata or {})), state)
    world = _world_ply(model, snapshot/'world.ply')
    native = render(prepared.cameras[prepared.reference], model, PIPE, black)['render']
    difference = _parity(native, render(prepared.cameras[prepared.reference], world, PIPE, black)['render'])
    mesh = snapshot/'reference-mesh.npz'
    shutil.copyfile(prepared.source_snapshot/mesh.name, mesh)
    image(native).save(snapshot/'reference-render.png')
    rt.save(snapshot/'native-ply-parity.json', dict(
        passed=True, checks=[difference], sha256=rt.sha(snapshot/'world.ply'), mesh_sha256=rt.sha(mesh),
        reference_frame_id=prepared.rows[prepared.reference]['frame'], reference_pose_index=prepared.reference,
        mesh_role='frozen SMPL-X support for the named motion'))
    sequence = model.vertices_sequence
    try:
        model.vertices_sequence = sequence[:prepared.main_count]
        model.num_timesteps = prepared.main_count
        save_bound_snapshot(root, snapshot, model)
    finally:
        model.vertices_sequence = sequence
        model.num_timesteps = len(sequence)
        model.select_mesh_by_timestep(prepared.reference)
    export(root, config, raw_snapshot=snapshot, endpoint=step, points=len(model.binding))
    drive = snapshot/'drive'
    _save_drive(drive, snapshot, prepared, labels, spill)
    reloaded, meta = load_face_drive(drive)
    _replay(snapshot, prepared, reloaded, meta, render, black)
    model.select_mesh_by_timestep(prepared.reference)
    package = snapshot/'drive-and-reference.zip'
    with ZipFile(package, 'x', compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(drive.iterdir()):
            archive.write(path, 'drive/'+path.name)
        archive.write(snapshot/'export/scene.zip', 'reference/scene.zip')
    with ZipFile(package) as archive:
        require(archive.testzip() is None, 'Motion and reference package CRC check failed.')
    receipt = dict(step=step, points=len(model.binding), path=snapshot.name, state_sha256=rt.sha(state),
                   drive_package=package.name, drive_package_sha256=rt.sha(package),
                   quality_accepted=False, frozen_and_geometry=report)
    rt.save(snapshot/'result.json', receipt)
    return receipt

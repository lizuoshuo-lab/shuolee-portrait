"""Refit one free canonical garment layer on a frozen, named SMPL-X motion."""

import argparse
import fcntl
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from PIL import Image

from . import runtime as rt
from .bound_replay import RAW_PARAMETERS, load_bound_snapshot
from .duration_checkpoint import _capture_rng
from .heldout_pose import verify_holdouts
from .sh0_main import DCColorProjection
from .stable_selection import source_bin_training_order
from .training import (_native_foreground_target, _sh0_foreground_alpha_loss, camera, image,
                       named_pose_layout, verify_official_fit)


ENDPOINTS = (10000, 18000)
BASELINE_STEP = ENDPOINTS[-1]
SPLIT_STEPS = (3000, 4800, 6600, 8400)
MAX_PARENTS_PER_SPLIT = 768
SPLIT_AXIS_MAX_NORMAL_COSINE = .5
BASELINE_CONVERSION_AFFECTED_FRACTION = .001
PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
COVARIANCE_PIPE = SimpleNamespace(**{**vars(PIPE), 'compute_cov3D_python': True})


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256_tensor(value):
    from hashlib import sha256

    return sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _fixed_identity(model):
    return {name: _sha256_tensor(getattr(model, name)) for name in (*RAW_PARAMETERS, 'binding')}


def _prepare(baseline):
    from scene.clothing_gaussian_model import ClothingGaussianModel
    from .drivable_clothing_bundle import CombinedClothingScene

    out = baseline/'gaussian-run'
    snapshot = out/f'step-{BASELINE_STEP:06d}'
    model, manifest = load_bound_snapshot(baseline, snapshot)
    raw = out/f'raw-step-{BASELINE_STEP:06d}'
    state_receipt = rt.read(raw/'state.json')
    state_path = raw/'state.pt'
    _require(state_receipt['step'] == BASELINE_STEP and rt.sha(state_path) == state_receipt['state_sha256']
             and rt.sha(raw/'bound-local.ply') == manifest['bound_ply_sha256'],
             'The terminal model has no matching raw optimizer state and row lineage.')
    state = torch.load(state_path, map_location='cpu', weights_only=True)['state']
    _require(torch.equal(state['model']['binding']['tensor'].long(), model.binding.cpu().long()) and
             all(torch.equal(state['parameters'][name]['value']['tensor'], getattr(model, name).cpu())
                 for name in RAW_PARAMETERS), 'The saved row order differs from the bound baseline.')
    with np.load(baseline/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    train_rows = rt.read(baseline/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, train_rows)
    _, held_fit, holdout_rows = verify_holdouts(baseline)
    held = snapshot/'heldout-replay'
    held_receipt = rt.read(held/'result.json')
    held_mesh = held/held_receipt['mesh_file']
    _require(rt.sha(held_mesh) == held_receipt['mesh_sha256'] and
             held_receipt['bound_ply_sha256'] == manifest['bound_ply_sha256'],
             'Held-out motion is not from this baseline.')
    reference = layout['reference_pose_index']
    training_count = len(layout['frame_ids'])
    with np.load(held_mesh, allow_pickle=False) as archive:
        holdout_ids = [row['frame'] for row in holdout_rows]
        _require(archive['frame_ids'].tolist() == holdout_ids and
                 np.array_equal(archive['faces'], fit['faces']) and
                 archive['pose_indices'].tolist() == list(range(training_count, training_count+len(holdout_ids))),
                 'Held-out motion identity differs from its native camera roster.')
        extra_vertices = torch.as_tensor(archive['vertices_world'], device=model._xyz.device)
    model.vertices_sequence = torch.cat((model.vertices_sequence, extra_vertices))
    model.num_timesteps = len(model.vertices_sequence)
    model.select_mesh_by_timestep(reference)
    frame_ids = layout['frame_ids'] + holdout_ids
    by_name = {row['frame']: row for row in (*train_rows, *holdout_rows)}
    _require(len(by_name) == len(frame_ids), 'Training and held-out frame rosters overlap.')
    rows = [by_name[name] for name in frame_ids]
    checked_cameras = [camera(row, slot, pose_index=slot) for slot, row in enumerate(rows)]
    cameras, camera_records = map(list, zip(*checked_cameras, strict=True))
    for row, saved in zip(holdout_rows, held_receipt['records'], strict=True):
        _require(all(row[key] == saved[key] for key in ('frame', 'source_sha256', 'mask_sha256', 'K', 'w2c', 'wh')),
                 'Held-out archived raster does not bind the current source frame.')

    guide = rt.read(out/'clothing-surface.json')
    support = out/guide['support_file']
    _require(rt.sha(support) == guide['support_file_sha256'] and
             guide['active_mesh_sha256'] == manifest['inputs_sha256']['reference_mesh'],
             'Clothing eligibility is not from the frozen motion guide.')
    with np.load(support, allow_pickle=False) as archive:
        vertex_mask = torch.as_tensor(archive['eligible_vertices'], device=model._xyz.device)
    owners = model.binding.long()
    garment = vertex_mask[model.faces].all(dim=1)[owners]
    _require(bool(garment.any()) and bool((~garment).any()), 'Empty garment or protected layer.')
    face_ids = owners[garment]
    cloth = ClothingGaussianModel(0)
    cloth._xyz = torch.nn.Parameter(model.get_xyz[garment].detach().clone())
    cloth._scaling = torch.nn.Parameter(model.get_scaling[garment].detach().log())
    cloth._rotation = torch.nn.Parameter(model.get_rotation[garment].detach().clone())
    for name in ('_features_dc', '_features_rest', '_opacity'):
        setattr(cloth, name, torch.nn.Parameter(getattr(model, name)[garment].detach().clone()))
    cloth.configure_triangle_support(model.vertices_sequence, model.faces, reference, face_ids,
        stable_ids=state['model']['densification_ids']['tensor'][garment.cpu()],
        next_stable_id=state['model']['next_densification_id']['value'])
    support_scale = model.face_scaling.reshape(-1)[face_ids]
    cloth.spatial_lr_scale = float(support_scale.median())
    _require(np.isfinite(cloth.spatial_lr_scale) and cloth.spatial_lr_scale > 0, 'Invalid metre learning-rate factor.')
    caps = torch.full((len(model.faces),), -torch.inf, device=model._xyz.device)
    caps.scatter_reduce_(0, face_ids, cloth._scaling.detach().amax(dim=1), reduce='amax', include_self=True)
    reference_normals = model.face_orien_mat[:, :, 1].detach().clone()

    fixed = type(model)(0)
    for name in RAW_PARAMETERS:
        setattr(fixed, name, getattr(model, name)[~garment].detach())
    fixed.binding = owners[~garment]
    fixed.faces, fixed.vertices_sequence = model.faces, model.vertices_sequence
    fixed.num_timesteps = model.num_timesteps
    fixed.binding_counter = torch.bincount(fixed.binding, minlength=len(fixed.faces)).to(torch.int32)
    scene = CombinedClothingScene(fixed, cloth)
    scene.select_mesh_by_timestep(reference)
    permutation = torch.cat((torch.where(~garment)[0], torch.where(garment)[0]))
    root_transforms = {
        key: torch.as_tensor(np.concatenate((fit[key], held_fit[key])),
                             device=model._xyz.device, dtype=model._xyz.dtype)
        for key in ('root_rotation', 'world_translation')
    }
    return SimpleNamespace(model=model, fixed=fixed, cloth=cloth, scene=scene, reference=reference,
                           rows=rows, train_rows=train_rows, training_count=training_count, layout=layout,
                           cameras=cameras, camera_records=camera_records, frames=frame_ids, caps=caps,
                           source_snapshot=snapshot, source_manifest=manifest, permutation=permutation,
                           source_support=support, reference_normals=reference_normals,
                           root_rotations=root_transforms['root_rotation'],
                           root_translations=root_transforms['world_translation'])


def _draw(render, cam, scene, background, pipe=PIPE):
    packet = render(cam, scene, pipe, background)
    alpha = render(cam, scene, pipe, background,
                   override_color=torch.ones_like(scene.get_xyz))['render'][:1]
    return packet, alpha


def _parity(a, b):
    delta = (a-b).abs()
    return dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()))


def _require_parity(difference, message):
    _require(difference['mean_abs'] < 2e-6 and difference['max_abs'] < .005, f'{message}: {difference}')


@torch.no_grad()
def _initial_replay(prepared, render, background):
    """Check the same initially posed Gaussian rows before updating geometry."""
    p = prepared
    records = []
    for slot, cam in enumerate(p.cameras):
        p.scene.select_mesh_by_timestep(slot)
        p.model.select_mesh_by_timestep(slot)
        order = p.permutation
        expected_xyz = p.model.get_xyz[order]
        expected_covariance = p.model.get_covariance()[order]
        xyz_error = float((p.scene.get_xyz-expected_xyz).abs().max())
        covariance_error = float((p.scene.get_covariance()-expected_covariance).abs().max())
        _require(xyz_error < 3e-6 and covariance_error < 1e-7,
                 f'Canonical initialization changes world geometry at {p.frames[slot]}: '
                 f'{xyz_error:.4g} m and {covariance_error:.4g} m².')
        # Reordering the rows can change tie ordering at identical depths.
        # Verify the renderer after restoring the old order for this check.
        inverse = order.argsort()
        ordered = SimpleNamespace(active_sh_degree=0, max_sh_degree=0, **{
            field: getattr(p.scene, field)[inverse]
            for field in ('get_xyz', 'get_scaling', 'get_rotation', 'get_features', 'get_opacity')})
        original = render(cam, p.model, PIPE, background)['render']
        posed = render(cam, ordered, PIPE, background)['render']
        difference = _parity(original, posed)
        affected = (original-posed).abs().amax(dim=0) > 1/255
        occupied = (torch.maximum(original, posed).abs().amax(dim=0) > 1/255).sum()
        difference.update(pixels_above_one_code_value=int(affected.sum()), occupied_pixels=int(occupied),
                          affected_fraction=float(affected.sum()/occupied.clamp_min(1)))
        covariance_rgb = render(cam, p.scene, COVARIANCE_PIPE, background)['render']
        posed_rgb = render(cam, p.scene, PIPE, background)['render']
        covariance_parity = _parity(covariance_rgb, posed_rgb)
        records.append(dict(frame=p.frames[slot], center_max_abs_m=xyz_error,
                            covariance_max_abs_m2=covariance_error, baseline_rgb=difference,
                            explicit_covariance_rgb=covariance_parity))
    p.scene.select_mesh_by_timestep(p.reference)
    return records


def _face_box(row):
    if not row.get('face_box_xyxy'):
        return None
    x0, y0, x1, y1 = row['face_box_xyxy']
    w, h = row['wh']
    bounds = (max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1))))
    return None if bounds[0] >= bounds[2] or bounds[1] >= bounds[3] else (
        slice(None), slice(bounds[1], bounds[3]), slice(bounds[0], bounds[2]))


@torch.no_grad()
def _targets(prepared, render, background):
    result = []
    for row in prepared.train_rows:
        slot = prepared.layout['pose_by_frame'][row['frame']]
        _require(rt.sha(row['mask']) == row['mask_sha256'], f'Native mask changed: {row["frame"]}')
        prepared.scene.select_mesh_by_timestep(slot)
        rgb, alpha, box = _native_foreground_target(row)
        face = _face_box(row)
        reference_face = None
        if face is not None:
            packet, model_alpha = _draw(render, prepared.cameras[slot], prepared.scene, background)
            reference_face = packet['render'][face].clone(), model_alpha[face].clone()
        area = float((alpha > .99).count_nonzero()) / alpha.numel()
        result.append((slot, rgb.cuda(), alpha.cuda(), box, area, face, reference_face))
    return result


@torch.no_grad()
def _split(cloth, threshold, reference_normals):
    from utils.general_utils import build_rotation

    score = (cloth.xyz_gradient_accum / cloth.denom.clamp_min(1)).flatten()
    eligible = torch.where(torch.isfinite(score) & (score >= threshold))[0]
    rotation = build_rotation(cloth._rotation[eligible])
    normals = reference_normals[cloth.support_face_ids[eligible]]
    cosines = (rotation*normals[:, :, None]).sum(dim=1).abs()
    tangent = cosines <= SPLIT_AXIS_MAX_NORMAL_COSINE
    supported = tangent.any(dim=1)
    candidates = eligible[supported]
    order = score[candidates].argsort(descending=True, stable=True)[:MAX_PARENTS_PER_SPLIT]
    parents = candidates[order]
    if not len(parents):
        cloth.reset_densification_stats()
        return dict(eligible=len(eligible), tangent_eligible=0, parents=0, children=0)
    parent_owners = cloth.support_face_ids[parents].clone()
    scales = cloth._scaling[parents].exp()
    axis = scales.masked_fill(~tangent[supported][order], -torch.inf).argmax(dim=1)
    index = torch.arange(len(parents), device=axis.device)
    quaternion = cloth._rotation[parents]
    offsets = .5*scales[index, axis, None] * rotation[supported][order][index, :, axis]
    xyz = cloth._xyz[parents]
    child_scaling = cloth._scaling[parents].clone()
    child_scaling[index, axis] += .5*np.log(.75)
    # Two co-located children initially preserve combined transmittance.
    # The symmetric displacement and smaller major sigma preserve the spatial second moment.
    parent_alpha = cloth.get_opacity[parents]
    child_alpha = -torch.expm1(.5*torch.log1p(-parent_alpha))
    opacity = torch.logit(child_alpha, eps=1e-7)
    existing = len(cloth._xyz)
    parent_stable_ids = cloth.densification_ids[parents].cpu().tolist()
    first_child_id = cloth.next_densification_id
    cloth.densification_postfix(torch.cat((xyz-offsets, xyz+offsets)),
        cloth._features_dc[parents].repeat(2, 1, 1), cloth._features_rest[parents].repeat(2, 1, 1),
        opacity.repeat(2, 1), child_scaling.repeat(2, 1), quaternion.repeat(2, 1),
        constraint_parents=parents.repeat(2))
    _require(torch.equal(cloth.support_face_ids[existing:], parent_owners.repeat(2)),
             'New garment rows changed their parent motion support.')
    remove = torch.zeros(len(cloth._xyz), dtype=torch.bool, device=parents.device)
    remove[parents] = True
    cloth.prune_points(remove)
    cloth._opacity.requires_grad_(False)
    cloth.reset_densification_stats()
    return dict(eligible=len(eligible), tangent_eligible=len(candidates),
                parents=len(parents), children=2*len(parents),
                parent_ids=parent_stable_ids, first_child_id=first_child_id)


@torch.no_grad()
def _world_ply(scene, path):
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(0)
    model._xyz, model._scaling, model._rotation = scene.get_xyz, scene.get_scaling.log(), scene.get_rotation
    model._features_dc = scene.get_features
    model._features_rest = model._features_dc[:, :0, :]
    model._opacity = torch.cat((scene.fixed._opacity, scene.cloth._opacity))
    model.save_ply(str(path))
    model.load_ply(str(path))
    return model


@torch.no_grad()
def _replay(snapshot, prepared, loaded, metadata, render, background):
    p = prepared
    destination = snapshot/'drive-replay'
    destination.mkdir()
    records = []
    loaded.select_mesh_by_timestep(p.reference)
    reference = loaded.cloth.get_xyz.clone()
    reference_local = (reference-p.root_translations[p.reference]) @ p.root_rotations[p.reference]
    for slot, (row, saved_camera) in enumerate(zip(p.rows, metadata['cameras'], strict=True)):
        original_camera = p.camera_records[slot]
        _require(all(saved_camera[key] == original_camera[key] for key in
                     ('frame', 'K', 'w2c', 'wh', 'pose_index', 'pixel_convention')),
                 f'Driving bundle changed the named native camera at {row["frame"]}.')
        cam, _ = camera(saved_camera, slot, pose_index=slot)
        p.scene.select_mesh_by_timestep(slot)
        loaded.select_mesh_by_timestep(slot)
        for field in ('get_xyz', 'get_rotation', 'get_scaling', 'get_features', 'get_opacity'):
            _require(torch.equal(getattr(p.scene, field), getattr(loaded, field)),
                     f'Driving bundle changed {field} at {row["frame"]}.')
        _require(torch.equal(p.scene.get_covariance(), loaded.get_covariance()),
                 f'Driving bundle changed covariance at {row["frame"]}.')
        packet, alpha = _draw(render, cam, loaded, background)
        truth = render(p.cameras[slot], p.scene, PIPE, background)['render']
        diff = _parity(packet['render'], truth)
        _require_parity(diff, f'Native saved bundle replay differs at {row["frame"]}')
        _, target, bounds = _native_foreground_target(row)
        x0, y0, x1, y1 = bounds
        width, height = row['wh']
        box = (max(0, x0-24), max(0, y0-24), min(width, x1+24), min(height, y1+24))
        rgb = packet['render']
        files = {}
        for name, tensor in (('black', rgb), ('white', (rgb+1-alpha).clamp(0, 1)),
                             ('alpha', alpha.expand_as(rgb))):
            file = destination/f'{row["frame"]}-{name}.png'
            image(tensor).crop(box).save(file)
            files[name] = file.name
        with Image.open(row['image']) as source:
            source.convert('RGB').crop(box).save(destination/f'{row["frame"]}-source.png')
        with Image.open(row['mask']) as source:
            source.convert('L').crop(box).save(destination/f'{row["frame"]}-mask.png')
        movement = (loaded.cloth.get_xyz-reference).norm(dim=1)
        local = (loaded.cloth.get_xyz-p.root_translations[slot]) @ p.root_rotations[slot]
        articulated = (local-reference_local).norm(dim=1)
        foreground = target[0].cuda() > .99
        alpha_core = alpha[0][foreground]
        records.append(dict(frame=row['frame'], pose_index=slot, box_xyxy=box, files=files,
                            source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'],
                            role='training' if slot < p.training_count else 'pose_conditioned_holdout',
                            exact_geometry_roundtrip=True, render_parity=diff,
                            cloth_center_motion_median_m=float(movement.median()),
                            cloth_center_motion_max_m=float(movement.max()),
                            root_removed_center_motion_median_m=float(articulated.median()),
                            root_removed_center_motion_p95_m=float(articulated.quantile(.95)),
                            foreground_mean_alpha=float(alpha_core.mean()),
                            foreground_alpha_below_point9=float((alpha_core < .9).float().mean()),
                            full_mask_alpha_l1_per_foreground=float((alpha.cpu()-target).abs().sum()/target.sum())))
    _require(any(row['cloth_center_motion_median_m'] > .01 for row in records),
             'Saved clothing does not actually move in the fitted motion sequence.')
    rt.save(destination/'result.json', dict(records=records, optimizer_updates=0,
            independent_full_clothing_surface_ground_truth=False, quality_accepted=False))


def _endpoint(root, prepared, step, config, render, background, draw_indices, splits, fixed_identity):
    from .drivable_clothing_bundle import load_drive_bundle, save_drive_bundle
    from .export import export

    p = prepared
    snapshot = root/'gaussian-run'/f'raw-step-{step:06d}'
    snapshot.mkdir()
    p.scene.select_mesh_by_timestep(p.reference)
    _require(_fixed_identity(p.fixed) == fixed_identity, 'Protected Gaussian parameters changed.')
    torch.save(dict(step=step, garment=p.cloth.capture(), support_face_ids=p.cloth.support_face_ids,
                    stable_ids=p.cloth.densification_ids, next_stable_id=p.cloth.next_densification_id,
                    draws=draw_indices.copy(), rng=_capture_rng(), split_events=splits.copy(),
                    source_sha256=p.source_manifest['bound_ply_sha256'], options=config['optimization'],
                    fixed_identity=fixed_identity), snapshot/'main-state.pt')
    with torch.no_grad():
        world = _world_ply(p.scene, snapshot/'world.ply')
        cam = p.cameras[p.reference]
        native = render(cam, p.scene, PIPE, background)['render']
        difference = _parity(native, render(cam, world, PIPE, background)['render'])
        _require_parity(difference, 'Reference world PLY differs from the live driven scene')
        mesh = snapshot/'reference-mesh.npz'
        shutil.copyfile(p.source_snapshot/mesh.name, mesh)
        image(native).save(snapshot/'reference-render.png')
        rt.save(snapshot/'native-ply-parity.json', dict(passed=True, checks=[difference],
                sha256=rt.sha(snapshot/'world.ply'), mesh_sha256=rt.sha(mesh),
                reference_frame_id=p.frames[p.reference], reference_pose_index=p.reference,
                mesh_role='frozen motion support, not the reconstructed garment surface'))
        export(root, config, raw_snapshot=snapshot, endpoint=step, points=len(world._xyz))
        drive = snapshot/'drive'
        save_drive_bundle(drive, p.fixed, p.cloth, vertices_sequence=p.fixed.vertices_sequence,
            faces=p.fixed.faces, reference_frame=p.reference, support_face_ids=p.cloth.support_face_ids,
            frame_ids=p.frames, cameras=p.camera_records)
        loaded, metadata = load_drive_bundle(drive)
        _replay(snapshot, p, loaded, metadata, render, background)
        p.scene.select_mesh_by_timestep(p.reference)
    packaged = snapshot/'drive-and-reference.zip'
    with ZipFile(packaged, 'x', compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(drive.iterdir()):
            archive.write(path, 'drive/'+path.name)
        archive.write(snapshot/'export/scene.zip', 'reference/scene.zip')
    with ZipFile(packaged) as archive:
        _require(archive.testzip() is None, 'A packaged scene member cannot be read back.')
    return dict(step=step, points=len(world._xyz), free_garment_points=len(p.cloth._xyz),
                fixed_points=len(p.fixed._xyz), path=snapshot.name, quality_accepted=False,
                drive_package=packaged.name, drive_package_sha256=rt.sha(packaged))


def run(baseline, root, *, probe):
    from gaussian_renderer import render
    from utils.sh_utils import C0

    source = rt.read(baseline/'run.json')
    rt.configure(source['assets'], root)
    fit_provenance = verify_official_fit(baseline)
    config = rt.read(baseline/'gaussian-run/config.json')
    _require(config['official_fit_provenance'] == fit_provenance,
             'The active baseline is not from the same verified fit.')
    # The inherited schedule stays at 18k for both checkpoints.
    options = dict(config['optimization'], iterations=ENDPOINTS[-1], opacity_lr=0.0)
    config.update(optimization=options, warm_start='terminal R22 garment rows, fresh Adam',
                  output_representation='single depth-composited protected plus free canonical garment',
                  opacity_policy='frozen logits except co-located peak transmittance rule at split initialization',
                  splits=list(SPLIT_STEPS), max_parents_per_split=MAX_PARENTS_PER_SPLIT,
                  split_axis_max_motion_support_normal_cosine=SPLIT_AXIS_MAX_NORMAL_COSINE,
                  scale_cap='per-owner largest reference sigma at initialization',
                  protection='frozen non-garment rows and native reference-face RGB/alpha preservation',
                  face_preservation_weight=1.0, reference_pose='unchanged frozen fitted pose',
                  training_duration_policy=dict(checkpoints=list(ENDPOINTS)))
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    prepared = _prepare(baseline)
    p = prepared
    cloth = p.cloth
    background = torch.zeros(3, device=cloth._xyz.device)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('data', 'fit'):
        (root/name).symlink_to(baseline/name, target_is_directory=True)
    out = root/'gaussian-run'
    out.mkdir()
    config.update(baseline_run=str(baseline), baseline_step=BASELINE_STEP,
                  source_bound_sha256=p.source_manifest['bound_ply_sha256'],
                  source_guide_sha256=rt.sha(p.source_support), canonical_position_lr_scale_m=cloth.spatial_lr_scale,
                  fixed_points=len(p.fixed._xyz), initial_free_garment_points=len(cloth._xyz),
                  training_frames=p.frames[:p.training_count], heldout_frames=p.frames[p.training_count:],
                  implementation_sha256={path.name: rt.sha(path) for path in (
                      Path(__file__), Path(__file__).with_name('drivable_clothing_bundle.py'),
                      rt.CORE/'scene/clothing_gaussian_model.py')})
    rt.save(out/'config.json', config)
    rt.save(root/'status.json', dict(state='preflight', updated=rt.now()))
    native_parity = _initial_replay(p, render, background)
    fixed_identity = _fixed_identity(p.fixed)
    rt.save(out/'initial-replay.json', dict(records=native_parity, protected_parameters=fixed_identity,
            baseline_conversion_max_affected_fraction=BASELINE_CONVERSION_AFFECTED_FRACTION,
            baseline_conversion_mean_float_rgb=2e-6))
    for record in native_parity:
        difference = record['baseline_rgb']
        _require(difference['mean_abs'] < 2e-6 and difference['affected_fraction'] <=
                 BASELINE_CONVERSION_AFFECTED_FRACTION,
                 f'Baseline coordinate conversion affects too many pixels at {record["frame"]}: {difference}')
        _require_parity(record['explicit_covariance_rgb'],
                        f'CUDA scale/rotation disagrees with posed covariance at {record["frame"]}')
    cloth.training_setup(SimpleNamespace(**options))
    cloth._opacity.requires_grad_(False)
    cloth.max_radii2D = cloth._xyz.new_zeros(len(cloth._xyz))
    _require(not cloth.optimizer.state, 'R23 must use an empty Adam optimizer.')
    projection = DCColorProjection(cloth._features_dc, C0)
    initial_color = projection.measure(cloth)
    targets = _targets(p, render, background)
    selection = rt.read(baseline/'fit/official-fit-result.json')['selection']
    _, order = source_bin_training_order(p.train_rows, selection, seed=config['seed'])
    splits, draw_indices, endpoints = [], [], []
    start = time.monotonic()
    updates = 1 if probe else ENDPOINTS[-1]
    with (out/'loss.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        for step in range(1, updates+1):
            draw = next(order)
            draw_indices.append(draw)
            slot, target_rgb, target_alpha, box, area, face, reference_face = targets[draw]
            cloth.update_learning_rate(step)
            p.scene.select_mesh_by_timestep(slot)
            packet, alpha = _draw(render, p.cameras[slot], p.scene, background)
            rgb = packet['render']
            photo, l1, ssim, mask_loss = _sh0_foreground_alpha_loss(rgb, alpha, target_rgb, target_alpha, box)
            face_loss = rgb.new_zeros(())
            if face is not None:
                face_loss = (rgb[face]-reference_face[0]).abs().mean() + (alpha[face]-reference_face[1]).abs().mean()
            loss = photo + config['alpha_loss_weight']*mask_loss + config['face_preservation_weight']*face_loss
            loss.backward()
            with torch.no_grad():
                if step <= SPLIT_STEPS[-1]:
                    garment = slice(len(p.fixed._xyz), None)
                    visible = packet['visibility_filter'][garment]
                    g = packet['viewspace_points'].grad[garment, :2]
                    cloth.xyz_gradient_accum[visible] += area*g[visible].norm(dim=1, keepdim=True)
                    cloth.denom[visible] += 1
                    cloth.max_radii2D.copy_(torch.maximum(cloth.max_radii2D, packet['radii'][garment]))
                if step == 1:
                    gradients = {name: float(getattr(cloth, name).grad.abs().max())
                                 for name in ('_xyz', '_scaling', '_rotation', '_features_dc')}
                    _require(all(np.isfinite(v) and v > 0 for v in gradients.values()),
                             f'Native RGB/alpha does not guide all garment parameters: {gradients}')
                    rt.save(out/'cuda-gradient-proof.json', dict(frame=p.frames[slot], gradients=gradients))
                cloth.optimizer.step()
                projection.apply(cloth, after_step=True)
                cloth._scaling.clamp_(max=p.caps[cloth.support_face_ids, None])
                cloth.optimizer.zero_grad(set_to_none=True)
                if step in SPLIT_STEPS:
                    splits.append(dict(step=step, **_split(
                        cloth, options['densify_grad_threshold'], p.reference_normals)))
                    rt.save(out/'garment-splits.json', splits)
            if step == 1 or step % 100 == 0:
                receipt = dict(step=step, state='probe' if probe else 'training', frame=p.frames[slot],
                               loss=float(loss.detach()), l1=float(l1.detach()), ssim=float(ssim.detach()),
                               mask_loss=float(mask_loss.detach()), protected_face_loss=float(face_loss.detach()),
                               points=len(p.fixed._xyz)+len(cloth._xyz),
                               elapsed_seconds=time.monotonic()-start, updated=rt.now())
                log.write(json.dumps(receipt, allow_nan=False)+'\n')
                rt.save(root/'status.json', receipt)
            if step in ENDPOINTS or probe:
                endpoints.append(_endpoint(root, p, step, config if not probe else {
                    **config, 'training_duration_policy': {'checkpoints': [1]}},
                    render, background, draw_indices, splits, fixed_identity))
    rt.save(out/'color-constraint.json', projection.report(cloth, initial_color))
    rt.save(root/'result.json', dict(state='probe_completed' if probe else 'completed', endpoints=endpoints,
            quality_accepted=False, input_pose_updates=0, total_optimizer_updates=updates,
            elapsed_seconds=time.monotonic()-start))
    rt.save(root/'status.json', rt.read(root/'result.json'))


def main():
    from .cli import gpu_lock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease, gpu_lock():
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline, output = args.baseline_run.resolve(strict=True), args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        try:
            run(baseline, output, probe=args.probe_only)
        except Exception as error:
            if output.is_dir():
                status = output/'status.json'
                previous = rt.read(status) if status.is_file() else None
                rt.save(status, dict(state='failed', last_status=previous, error=str(error), updated=rt.now()))
            raise


if __name__ == '__main__':
    main()

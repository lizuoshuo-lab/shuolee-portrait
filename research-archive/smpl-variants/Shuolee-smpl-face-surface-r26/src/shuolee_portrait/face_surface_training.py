"""Refine facial splats on frozen R22 SMPL-X motion, cameras and SH0 appearance."""
import argparse
import random
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from . import runtime as rt
from .bound_replay import RAW_PARAMETERS, load_bound_snapshot
from .face_supplement_training import masked_face_loss
from .face_surface_bundle import PIPE, draw, require, save_endpoint
from .face_surface_geometry import face_surface_loss, face_surface_metrics
from .heldout_pose import verify_holdouts
from .stable_selection import source_bin_training_order
from .training import camera, named_pose_layout, verify_official_fit

ENDPOINTS = (10000, 18000)
SPLIT_STEPS = (3500, 6500, 9500, 13500, 16500)
FROZEN_GLOBAL = ('_opacity', '_features_rest')
TRAINABLE = ('_xyz', '_scaling', '_rotation', '_features_dc')
REPLAY_FIELDS = ('frame', 'source_sha256', 'mask_sha256', 'K', 'w2c', 'wh')
MIN_UNKNOWN_SCALE_RATIO = .85


def _snapshot_inputs(baseline):
    output = baseline/'gaussian-run'
    snapshot = output/'step-018000'
    raw = output/'raw-step-018000'
    model, manifest = load_bound_snapshot(baseline, snapshot)
    receipt = rt.read(raw/'state.json')
    state_path = raw/'state.pt'
    require(receipt['step'] == 18000 and rt.sha(state_path) == receipt['state_sha256']
            and rt.sha(raw/'bound-local.ply') == manifest['bound_ply_sha256'],
            'The R22 raw checkpoint does not match its terminal replay.')
    state = torch.load(state_path, map_location='cpu', weights_only=True)['state']
    require(torch.equal(state['model']['binding']['tensor'].long(), model.binding.cpu().long())
            and all(torch.equal(state['parameters'][name]['value']['tensor'], getattr(model, name).cpu())
                    for name in RAW_PARAMETERS),
            'R22 stable row IDs cannot be joined to the terminal model.')
    ids = state['model']['densification_ids']['tensor']
    next_id = state['model']['next_densification_id']['value']
    require(ids.ndim == 1 and len(ids) == len(model.binding) and len(torch.unique(ids)) == len(ids)
            and next_id > int(ids.max()), 'Invalid stable IDs.')
    model.densification_ids = ids.cuda()
    model.next_densification_id = next_id
    model.binding = model.binding.long()
    model.spatial_lr_scale = float(state['model']['spatial_lr_scale']['value'])
    return model, manifest, snapshot, raw


def _pose_receipt(path, row_names, faces, pose_indices, digest):
    require(rt.sha(path) == digest, f'Frozen pose file changed: {path}')
    with np.load(path, allow_pickle=False) as data:
        require(data['frame_ids'].tolist() == row_names and np.array_equal(data['faces'], faces)
                and data['pose_indices'].tolist() == pose_indices,
                f'Frozen pose order or topology changed: {path}')
        vertices = data['vertices_world']
    require(np.isfinite(vertices).all(), 'Frozen poses contain non-finite vertices.')
    return torch.from_numpy(vertices).cuda()


def _prepare(baseline):
    model, manifest, snapshot, raw = _snapshot_inputs(baseline)
    with np.load(baseline/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    main = rt.read(baseline/'data/training-inputs.json')['records']
    layout = named_pose_layout(fit, main)
    require(len(main) == 24 and layout['frame_ids'] == [row['frame'] for row in main], 'Invalid native training roster.')
    reference = layout['reference_pose_index']
    _, _, holdouts = verify_holdouts(baseline)
    held = snapshot/'heldout-replay'
    held_receipt = rt.read(held/'result.json')
    require(held_receipt['bound_ply_sha256'] == manifest['bound_ply_sha256'] and len(holdouts) == 4
            and all(all(row[key] == saved[key] for key in REPLAY_FIELDS)
                    for row, saved in zip(holdouts, held_receipt['records'], strict=True)), 'Held-out replay has wrong source.')
    held_mesh = held/held_receipt['mesh_file']
    held_vertices = _pose_receipt(held_mesh, [row['frame'] for row in holdouts], fit['faces'],
                                 list(range(24, 28)), held_receipt['mesh_sha256'])
    auxiliary = baseline/'gaussian-run/face-supplement'
    aux_replay = rt.read(auxiliary/'replay.json')
    aux_records = aux_replay['records']
    require(len(aux_records) == 4 and aux_replay['shared_bound_ply_sha256'] == manifest['bound_ply_sha256'],
            'R22 face-only replay does not bind the same terminal model.')
    originals = {row['frame']: row for row in rt.read(baseline/'data/inputs.json')['records']}
    aux_rows = []
    for saved in aux_records:
        row = originals[saved['frame']]
        require(saved['role'] == 'face_only'
                and all(row[key] == saved[key] for key in ('source_sha256', 'mask_sha256', 'source_frame_index'))
                and row['wh'] == saved['camera']['wh'], 'Auxiliary frame identity differs.')
        aux_rows.append({**row, **{key: saved['camera'][key] for key in ('K', 'w2c', 'wh')}})
    aux_vertices = _pose_receipt(auxiliary/'poses.npz', [row['frame'] for row in aux_rows], fit['faces'],
                                list(range(24, 28)), aux_replay['poses_sha256'])
    rows = main + holdouts + aux_rows
    require(len({row['frame'] for row in rows}) == len(rows) == 32, 'Frame roles overlap.')
    model.vertices_sequence = torch.cat((model.vertices_sequence, held_vertices, aux_vertices))
    model.num_timesteps = len(rows)
    cams, camera_records = map(list, zip(*(camera(row, slot, pose_index=slot)
                                           for slot, row in enumerate(rows)), strict=True))
    return SimpleNamespace(model=model, rows=rows, cameras=cams, camera_records=camera_records,
                           reference=reference, source_snapshot=snapshot, source_raw=raw,
                           source_manifest=manifest, auxiliary=auxiliary, auxiliary_records=aux_records,
                           baseline=baseline)


def _load_material(prepared, archive_path):
    receipt_path = archive_path.with_suffix('.json')
    receipt = rt.read(receipt_path)
    expected = prepared.source_manifest
    require(receipt['baseline_bound_ply_sha256'] == expected['bound_ply_sha256']
            and receipt['baseline_raw_ply_sha256'] == expected['inputs_sha256']['world_ply']
            and receipt['training_inputs_sha256'] == expected['inputs_sha256']['training_rows']
            and receipt['heldout_source_images_read'] == 0
            and receipt['material_evidence_npz_sha256'] == rt.sha(archive_path)
            and receipt['fit_sha256'] == expected['inputs_sha256']['fitted'],
            'Material evidence is not pinned to this fit and row order.')
    training = {row['frame'] for row in prepared.rows[:24]}
    require(set(receipt['classification_frames']) <= training
            and set(receipt['selected_frames']) == training
            and set(receipt['admitted_auxiliary_frames']) == {row['frame'] for row in prepared.rows[28:]},
            'Material evidence contains unapproved frame roles.')
    with np.load(archive_path, allow_pickle=False) as data:
        ids, labels, owners, spill = (data[name] for name in ('stable_ids', 'labels', 'owner_faces', 'silhouette_spill'))
    model = prepared.model
    require(ids.shape == labels.shape == owners.shape == spill.shape == (len(model.binding),)
            and np.array_equal(ids, model.densification_ids.cpu().numpy())
            and np.array_equal(owners, model.binding.cpu().numpy())
            and np.isin(labels, np.arange(6)).all() and (labels == 1).any()
            and spill.dtype == np.bool_ and not np.any(spill & (labels != 0)),
            'Material evidence has invalid labels, owners or stable IDs.')
    return torch.from_numpy(labels).to(device='cuda', dtype=torch.uint8), torch.from_numpy(spill).cuda(), receipt_path


def _face_box(row):
    value = np.asarray(row['face_box_xyxy'], dtype=float)
    require(value.shape == (4,) and np.isfinite(value).all() and (value[2:] > value[:2]).all(),
            f'Missing native face box for {row["frame"]}')
    pad = max(12., .34 * max(value[2:]-value[:2]))
    lo = np.maximum(np.floor(value[:2]-pad), 0).astype(int)
    hi = np.minimum(np.ceil(value[2:]+pad), row['wh']).astype(int)
    return tuple(map(int, (*lo, *hi)))


def _raster(row, box):
    arrays = []
    for key, mode, digest in (('image', 'RGB', 'source_sha256'), ('mask', 'L', 'mask_sha256')):
        require(rt.sha(row[key]) == row[digest], f'Native source pixels changed: {row["frame"]}')
        with Image.open(row[key]) as source:
            require(source.size == tuple(row['wh']), f'Wrong native source size: {row["frame"]}')
            arrays.append(torch.from_numpy(np.asarray(source.convert(mode).crop(box)).copy()).cuda().float()/255)
    return arrays[0].permute(2, 0, 1).contiguous(), arrays[1]


@torch.no_grad()
def _targets(prepared, render, black):
    targets = []
    model = prepared.model
    for slot in (*range(24), *range(28, 32)):
        row = prepared.rows[slot]
        saved = prepared.auxiliary_records[slot-28] if slot >= 28 else None
        box = tuple(saved['box']) if saved else _face_box(row)
        x0, y0, x1, y1 = box
        crop = (slice(y0, y1), slice(x0, x1))
        rgb, alpha = _raster(row, box)
        valid = alpha >= .99
        outside = -torch.nn.functional.max_pool2d(-((alpha <= .01).float())[None, None], 5, 1, 2)[0, 0] > .99
        if saved:
            path = prepared.auxiliary/saved['valid_mask']
            require(rt.sha(path) == saved['valid_mask_sha256'], f'Changed auxiliary mask: {path}')
            with Image.open(path) as mask:
                admitted = torch.from_numpy(np.asarray(mask.convert('L')).copy()).cuda() == 255
            require(admitted.shape == valid.shape and not bool((admitted & ~valid).any())
                    and int(admitted.sum()) == saved['valid_pixels'], 'Auxiliary foreground support differs.')
            valid = admitted
            outside.zero_()
        require(int(valid.sum()) > 0, f'No verified facial foreground for {row["frame"]}.')
        cam = prepared.cameras[slot]
        model.select_mesh_by_timestep(slot)
        _, base_alpha = draw(render, cam, model, black)
        targets.append(SimpleNamespace(
            row=row, cam=cam, slot=slot, crop=crop, box=box, valid=valid, outside=outside,
            rgb=rgb, alpha=alpha, baseline_alpha=base_alpha[crop].clone()))
    return targets


def _loss(target, model, render, black):
    model.select_mesh_by_timestep(target.slot)
    packet, full_alpha = draw(render, target.cam, model, black)
    rgb = packet['render'][(slice(None), *target.crop)]
    alpha = full_alpha[target.crop]
    color, coverage = masked_face_loss(rgb, alpha, target.rgb, target.alpha, target.valid)
    preserved = target.valid & (target.baseline_alpha >= .90)
    opaque_deficit = ((target.baseline_alpha - alpha).relu()*preserved).sum()/preserved.sum().clamp_min(1)
    exterior = (alpha*target.outside).sum()/target.outside.sum().clamp_min(1)
    return color + .15*coverage + .25*opaque_deficit + .15*exterior, packet


def _restrict_gradients(model, labels, spill):
    skin = labels == 1
    for name in TRAINABLE:
        gradient = getattr(model, name).grad
        require(gradient is not None, f'No gradient reaches {name}.')
        editable = skin | spill if name == '_scaling' else skin
        gradient[~editable] = 0
        require(bool(torch.isfinite(gradient).all()), f'Invalid {name} gradient.')


@torch.no_grad()
def _restrict_values(model, labels, spill, caps):
    skin = labels == 1
    scaling = model._scaling
    scaling[skin] = torch.minimum(scaling[skin], caps[skin].amax(dim=1, keepdim=True))
    scaling[spill] = torch.maximum(torch.minimum(scaling[spill], caps[spill]), caps[spill] + np.log(MIN_UNKNOWN_SCALE_RATIO))
    limit = .5/.28209479177387814
    model._features_dc[skin] = model._features_dc[skin].clamp(-limit, limit)


@torch.no_grad()
def _frozen(model, original, labels, spill, caps):
    current = model.densification_ids
    original_ids = original['ids']
    sorted_ids, order = original_ids.sort()
    positions = torch.searchsorted(sorted_ids, current).clamp_max(len(sorted_ids)-1)
    existing = sorted_ids[positions] == current
    original_row = order[positions]
    fixed = labels != 1
    require(bool(existing[fixed].all()), 'A non-skin row was inserted or reclassified.')
    require(torch.equal(model.binding[existing], original['binding'][original_row[existing]])
            and torch.equal(labels[existing], original['labels'][original_row[existing]])
            and torch.equal(spill[existing], original['spill'][original_row[existing]]),
            'An existing row changed anatomical or material ownership.')
    require(int(fixed.sum()) == int((original['labels'] != 1).sum()), 'A protected row was deleted.')
    for name in RAW_PARAMETERS:
        mask = existing if name in FROZEN_GLOBAL else fixed
        if name == '_scaling':
            mask = mask & ~spill
        require(torch.equal(getattr(model, name)[mask], original[name][original_row[mask]]),
                f'Protected values changed in {name}.')
    require(bool((model._scaling[spill] <= caps[spill]).all()) and
            bool((model._scaling[spill] >= caps[spill] + np.log(MIN_UNKNOWN_SCALE_RATIO)-1e-7).all()),
            'Unknown-row shrink escaped its bounded range.')
    return dict(frozen_non_skin_rows=int(fixed.sum()-spill.sum()), limited_unknown_rows=int(spill.sum()),
                modified_limited_unknown_rows=int((model._scaling[spill] != caps[spill]).any(dim=1).sum()),
                skin_rows=int((labels == 1).sum()), new_skin_rows=int((~existing).sum()))


@torch.no_grad()
def _measure(model, labels, slots):
    ids = torch.where(labels == 1)[0]
    output = []
    for slot in slots:
        model.select_mesh_by_timestep(slot)
        metrics = {name: value.float() for name, value in face_surface_metrics(model, ids).items()}
        metrics['center'] = metrics['center'].abs()
        output.append(dict(pose_index=slot, **{
            name: dict(p50=float(metrics[name].median()), p95=float(metrics[name].quantile(.95)),
                       max=float(metrics[name].max()))
            for name in ('center', 'normal_sigma', 'tangent_major_sigma', 'tangent_cap', 'axis_excess', 'sheet')}))
    return output


def _optimization(baseline):
    options = rt.read(baseline/'gaussian-run/config.json')['optimization']
    # Small local steps on a mature snapshot. The opacity and SH parameters stay fixed.
    return {**options, 'position_lr_init': .0005, 'position_lr_final': .000025,
            'scaling_lr': .002, 'rotation_lr': .0005, 'feature_lr': .0005, 'opacity_lr': 0.}


def _run(baseline, root, evidence, probe):
    from gaussian_renderer import render

    assets = rt.read(baseline/'run.json')['assets']
    rt.configure(assets, root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    prepared = _prepare(baseline)
    model = prepared.model
    labels, spill, material_receipt = _load_material(prepared, evidence)
    black = torch.zeros(3, device='cuda')
    prepared.targets = _targets(prepared, render, black)
    reference = prepared.reference
    select = rt.read(baseline/'fit/official-fit-result.json')['selection']
    row_bins, order = source_bin_training_order(prepared.rows[:24], select, seed=0)
    options = _optimization(baseline)
    out = root/'gaussian-run'
    root.mkdir(parents=True)
    for folder in ('fit', 'data'):
        (root/folder).symlink_to(baseline/folder, target_is_directory=True)
    out.mkdir()
    pinned = dict(
        bound_ply=prepared.source_manifest['bound_ply_sha256'],
        raw_state=rt.sha(prepared.source_raw/'state.pt'),
        fit=rt.sha(baseline/'fit/fitted.npz'), material=rt.sha(evidence), material_receipt=rt.sha(material_receipt),
        aux_replay=rt.sha(prepared.auxiliary/'replay.json'))
    config = dict(
        official_fit_provenance=verify_official_fit(baseline), pinned_inputs=pinned,
        training_duration_policy=dict(checkpoints=list(ENDPOINTS), schedule_steps=ENDPOINTS[-1],
                                      independent_postprocessing=False, single_main_training=True),
        optimization=options, seed=0, training_source_bins=row_bins,
        unknown_minimum_scale_ratio=MIN_UNKNOWN_SCALE_RATIO, split_steps=list(SPLIT_STEPS),
        train_frames=[row['frame'] for row in prepared.rows[:24]],
        holdout_frames=[row['frame'] for row in prepared.rows[24:28]],
        auxiliary_frames=[row['frame'] for row in prepared.rows[28:]],
        auxiliary_frequency=8, candidate='R26_face_surface', execution='cuda_probe' if probe else 'main')
    rt.save(out/'config.json', config)
    rt.save(root/'run.json', dict(rt.read(baseline/'run.json'), state='running', source=str(baseline), candidate='R26'))
    original = dict(
        ids=model.densification_ids.clone(), labels=labels.clone(), spill=spill.clone(), binding=model.binding.clone(),
        **{name: getattr(model, name).detach().clone() for name in RAW_PARAMETERS})
    caps = model._scaling.detach().clone()
    model.select_mesh_by_timestep(reference)
    model.max_radii2D = torch.zeros_like(model._opacity[:, 0])
    model.training_setup(SimpleNamespace(**options))
    for name in TRAINABLE:
        getattr(model, name).requires_grad_(True)
    require(not model.optimizer.state, 'R26 must start with an empty optimizer.')
    geom_slots = [reference, *[slot for slot in (0, 6, 12, 18, 23) if slot != reference]]
    geometry_before = _measure(model, labels, geom_slots)
    rt.save(out/'initial-geometry.json', dict(metrics=geometry_before, material_rows=labels.bincount(minlength=6).tolist(),
                                              evidence_spill_rows=int(spill.sum())))
    model.face_surface_view_support = torch.zeros_like(labels, dtype=torch.int32)
    # Track distinct supporting cameras, not the count of repeated training draws.
    seen_cameras = torch.zeros_like(labels, dtype=torch.int64)
    draws, splits, endpoints, aux_updates = [], [], [], Counter()
    start = time.monotonic()
    stop = 8 if probe else ENDPOINTS[-1]
    for step in range(1, stop+1):
        slot = next(order)
        draws.append(slot)
        model.update_learning_rate(step)
        model.optimizer.zero_grad(set_to_none=True)
        target = prepared.targets[slot]
        photo, packet = _loss(target, model, render, black)
        ids = torch.where(labels == 1)[0]
        geometry, _ = face_surface_loss(model, ids)
        loss = photo + .16*geometry
        if step % 8 == 0:
            auxiliary = prepared.targets[24 + (step//8-1) % 4]
            aux_photo, _ = _loss(auxiliary, model, render, black)
            loss = loss + .30*aux_photo
            aux_updates[auxiliary.row['frame']] += 1
        require(bool(torch.isfinite(loss)), f'Non-finite R26 loss at {step}.')
        loss.backward()
        _restrict_gradients(model, labels, spill)
        model.optimizer.step()
        _restrict_values(model, labels, spill, caps)
        with torch.no_grad():
            gradient = packet['viewspace_points'].grad
            if gradient is not None:
                weight = gradient[:, :2].norm(dim=1)
                valid = (labels == 1) & packet['visibility_filter'] & (weight > 0)
                flag = 1 << slot
                first = valid & ((seen_cameras & flag) == 0)
                model.face_surface_view_support += first
                seen_cameras[valid] |= flag
                model.xyz_gradient_accum[valid, 0] += weight[valid]
                model.denom[valid, 0] += 1
        model.optimizer.zero_grad(set_to_none=True)
        if step in SPLIT_STEPS or (probe and step == stop):
            from .face_surface_split import try_skin_split
            labels, spill, caps, event = try_skin_split(model, labels, spill, caps, prepared.targets, render, PIPE, black, step)
            splits.append(event)
            rt.save(out/'split-events.json', splits)
            model.face_surface_view_support = torch.zeros_like(labels, dtype=torch.int32)
            seen_cameras = torch.zeros_like(labels, dtype=torch.int64)
            model.xyz_gradient_accum.zero_()
            model.denom.zero_()
            for name in FROZEN_GLOBAL:
                getattr(model, name).requires_grad_(False)
        if step == 1 or step % 200 == 0 or step == stop:
            progress = dict(state='running', step=step, loss=float(loss.detach()), photo=float(photo.detach()),
                            geometry=float(geometry.detach()), points=len(model.binding),
                            auxiliary_updates=dict(aux_updates), elapsed_s=round(time.monotonic()-start, 2))
            rt.save(root/'progress.json', progress)
            print(progress, flush=True)
        if step in ENDPOINTS or (probe and step == stop):
            report = _frozen(model, original, labels, spill, caps)
            report['geometry_before'] = geometry_before
            report['geometry_after'] = _measure(model, labels, geom_slots)
            if probe:
                probe_config = {**config, 'training_duration_policy': dict(checkpoints=[step], schedule_steps=stop)}
                endpoint = save_endpoint(root, prepared, step, probe_config, render, black,
                                         draws, splits, labels, spill, caps, report)
            else:
                endpoint = save_endpoint(root, prepared, step, config, render, black,
                                         draws, splits, labels, spill, caps, report)
            endpoints.append(endpoint)
            rt.save(out/'result.json', dict(state='probe_complete' if probe else 'checkpoint_verified',
                                           steps=step, endpoints=endpoints, quality_accepted=False,
                                           auxiliary_updates=dict(aux_updates), split_events=splits))
    rt.save(root/'progress.json', dict(**{k: v for k, v in progress.items() if k != 'state'}, state='completed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    _run(args.baseline.resolve(), args.root.resolve(), args.material.resolve(), args.probe)


if __name__ == '__main__':
    main()

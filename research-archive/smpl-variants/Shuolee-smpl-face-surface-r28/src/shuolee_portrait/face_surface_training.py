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
from .face_surface_association_r28 import SurfaceAssociation
from .face_supplement_training import masked_face_loss
from .face_surface_bundle import PIPE, draw, require, save_endpoint
from .face_surface_geometry import face_surface_metrics
from .face_surface_geometry_r27 import (face_surface_limits, face_surface_loss_r27, project_skin_envelope,
                                       snapshot_face_geometry, unknown_support_loss)
from .face_surface_geometry_r28 import (SURFACE_R28, prepare_surface_geometry, project_surface_envelope,
                                       surface_loss_r28, surface_metrics_r28)
from .face_supplement import REGIONS
from .heldout_pose import verify_holdouts
from .stable_selection import source_bin_training_order
from .training import camera, named_pose_layout, verify_official_fit

ENDPOINTS = (10000, 18000)
SPLIT_STEPS = (3500, 6500, 9500, 13500, 16500)
FROZEN_GLOBAL = ('_opacity', '_features_rest')
TRAINABLE = ('_xyz', '_scaling', '_rotation', '_features_dc')
REPLAY_FIELDS = ('frame', 'source_sha256', 'mask_sha256', 'K', 'w2c', 'wh')
VISIBLE_SCALE_RANGE = (.55, 1.20)
UNSEEN_SCALE_FLOOR = .93
SH0_DC = .28209479177387814
COLOR_WITHOUT_REPEATED_SUPPORT = .025
COLOR_SUPPORTED_MAX_CHANGE = .12


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
                           baseline=baseline, fit=fit)


def _load_material(prepared, archive_path):
    receipt_path = archive_path.with_suffix('.json')
    receipt = rt.read(receipt_path)
    expected = prepared.source_manifest
    require(receipt['schema'] == 'shuolee-r27-native-face-material-visibility/1'
            and receipt['baseline_bound_ply_sha256'] == expected['bound_ply_sha256']
            and receipt['baseline_raw_ply_sha256'] == expected['inputs_sha256']['world_ply']
            and receipt['training_inputs_sha256'] == expected['inputs_sha256']['training_rows']
            and receipt['heldout_source_images_read'] == 0
            and receipt['material_visibility_npz_sha256'] == rt.sha(archive_path)
            and receipt['fit_sha256'] == expected['inputs_sha256']['fitted'],
            'Material evidence is not pinned to this fit and row order.')
    training = {row['frame'] for row in prepared.rows[:24]}
    auxiliary = {row['frame'] for row in prepared.rows[28:]}
    holdouts = {row['frame'] for row in prepared.rows[24:28]}
    require(set(receipt['classification_frames']) == training | auxiliary
            and set(receipt['selected_frames']) == training
            and set(receipt['admitted_auxiliary_frames']) == auxiliary
            and set(receipt['retained_holdouts']) == holdouts,
            'Material evidence contains unapproved frame roles.')
    fields = ('stable_ids', 'labels', 'owner_faces', 'silhouette_spill', 'observed', 'head_scope',
              'geometry_eligible', 'unknown_unobserved_elongated', 'evidence_interior_mass', 'visible_views',
              'possible_protected_material')
    with np.load(archive_path, allow_pickle=False) as data:
        records = {name: data[name] for name in fields}
    ids, labels, owners, spill, observed, scope, skin, unseen, mass, views, protected = records.values()
    model = prepared.model
    require(all(value.shape == (len(model.binding),) for value in records.values())
            and np.array_equal(ids, model.densification_ids.cpu().numpy())
            and np.array_equal(owners, model.binding.cpu().numpy())
            and np.isin(labels, np.arange(6)).all() and skin.any() and observed.any()
            and all(value.dtype == np.bool_ for value in (spill, observed, scope, skin, unseen, protected))
            and np.isfinite(mass).all() and (mass >= 0).all()
            and not np.any(spill & (labels != 0)) and not np.any(observed & (~scope | (views == 0)))
            and not np.any(skin & (~observed | (labels != 1)))
            and not np.any(protected & (skin | unseen | ~scope))
            and not np.any(unseen & (~scope | observed | (labels != 0))),
            'Material evidence has invalid labels, owners or stable IDs.')
    return SimpleNamespace(**{name: torch.from_numpy(value).cuda() for name, value in records.items()}), receipt_path


def _load_association(prepared, archive_path, material, material_path):
    receipt = rt.read(archive_path.with_suffix('.json'))
    require(receipt['schema'] == 'shuolee-r28-native-surface-association/1'
            and receipt['association_npz_sha256'] == rt.sha(archive_path)
            and receipt['material_receipt_sha256'] == rt.sha(material_path)
            and receipt['baseline_bound_ply_sha256'] == prepared.source_manifest['bound_ply_sha256']
            and receipt['baseline_raw_state_sha256'] == rt.sha(prepared.source_raw/'state.pt')
            and receipt['training_frames'] == [row['frame'] for row in prepared.rows[:24]]
            and receipt['auxiliary_frames'] == [row['frame'] for row in prepared.rows[28:]]
            and receipt['holdout_source_images_read'] == 0, 'R28 association has changed source evidence.')
    with np.load(archive_path, allow_pickle=False) as data:
        association = SurfaceAssociation.from_arrays(dict(data)).to('cuda')
    model = prepared.model
    selected = association.mask(model.densification_ids)
    require(bool(selected.any()) and not bool((selected & ~material.geometry_eligible).any())
            and int(selected.sum()) == receipt['association']['associated_rows'],
            'R28 association includes an unsupported material or wrong stable ID.')
    return association, receipt


@torch.no_grad()
def _color_limits(prepared, archive_path, material, material_path):
    receipt = rt.read(archive_path.with_suffix('.json'))
    expected = prepared.source_manifest
    require(receipt['schema'] == 'shuolee-r28-native-sh0-color-guard/1'
            and receipt['source_color_guard_npz_sha256'] == rt.sha(archive_path)
            and receipt['material_visibility_receipt_sha256'] == rt.sha(material_path)
            and receipt['material_visibility_npz_sha256'] == rt.sha(material_path.with_suffix('.npz'))
            and receipt['baseline_bound_ply_sha256'] == expected['bound_ply_sha256']
            and receipt['baseline_raw_ply_sha256'] == expected['inputs_sha256']['world_ply']
            and set(receipt['source_frames']) == {row['frame'] for row in prepared.rows[:24]+prepared.rows[28:]}
            and receipt['heldout_source_images_read'] == 0, 'R28 native color bounds have changed source evidence.')
    with np.load(archive_path, allow_pickle=False) as data:
        names = ('stable_ids','owner_faces','source_rgb_low','source_rgb_high','source_color_guard','source_color_bins')
        ids, owners, low, high, guarded, bins = (torch.from_numpy(data[name]).cuda() for name in names)
    require(torch.equal(ids, material.stable_ids) and torch.equal(owners, material.owner_faces)
            and guarded.dtype == torch.bool and guarded.shape == ids.shape
            and bins.shape == ids.shape and not bool((guarded & (~material.observed | (bins < 2))).any())
            and low.shape == high.shape == (len(ids),3) and torch.isfinite(low[guarded]).all()
            and torch.isfinite(high[guarded]).all() and bool(((low[guarded] >= 0) & (low[guarded] <= high[guarded])
                                                           & (high[guarded] <= 1)).all()),
            'R28 native color bounds do not align with observed Gaussian rows.')
    seed = prepared.model._features_dc[:,0,:].detach()*SH0_DC+.5
    fallback = COLOR_WITHOUT_REPEATED_SUPPORT
    lower = (seed-fallback).clamp(0,1)
    upper = (seed+fallback).clamp(0,1)
    lower[guarded] = torch.maximum(low[guarded].minimum(seed[guarded]), seed[guarded]-COLOR_SUPPORTED_MAX_CHANGE)
    upper[guarded] = torch.minimum(high[guarded].maximum(seed[guarded]), seed[guarded]+COLOR_SUPPORTED_MAX_CHANGE)
    return ((lower-.5)/SH0_DC)[:,None,:], ((upper-.5)/SH0_DC)[:,None,:], guarded


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


def _feature_pixels(box, points, valid):
    x0, y0, x1, y1 = box
    feature, eyes = torch.zeros_like(valid), torch.zeros_like(valid)
    for name in ('left_eye', 'right_eye', 'nose', 'mouth'):
        native = points[REGIONS[name]]
        native = native[np.isfinite(native).all(1) & (native[:, 2] >= .50), :2]
        if len(native) < 3:
            continue
        pad = max(2, round(.18*float(np.ptp(native, axis=0).max())))
        lo = np.maximum(np.floor(native.min(0)).astype(int)-pad, (x0, y0))
        hi = np.minimum(np.ceil(native.max(0)).astype(int)+pad+1, (x1, y1))
        if (hi > lo).all():
            region = np.s_[lo[1]-y0:hi[1]-y0, lo[0]-x0:hi[0]-x0]
            feature[region] = True
            if name.endswith('_eye'):
                eyes[region] = True
    return feature & valid, eyes & valid


@torch.no_grad()
def _targets(prepared, render, black):
    targets = []
    model = prepared.model
    with np.load(prepared.baseline/'fit/joint-candidates.npz', allow_pickle=False) as raw:
        auxiliary_points = {name: (points, image_hash, mask_hash) for name, points, image_hash, mask_hash in zip(
            raw['frame_ids'], raw['observed_native_keypoints'], raw['image_sha256'], raw['mask_sha256'], strict=True)
            if name in {item['frame'] for item in prepared.auxiliary_records}}
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
            points, image_hash, mask_hash = auxiliary_points[row['frame']]
            require(image_hash == row['source_sha256'] and mask_hash == row['mask_sha256'],
                    'Auxiliary landmarks do not match the verified source.')
        else:
            points = prepared.fit['observed_native_keypoints'][slot]
        require(int(valid.sum()) > 0, f'No verified facial foreground for {row["frame"]}.')
        cam = prepared.cameras[slot]
        model.select_mesh_by_timestep(slot)
        packet, base_alpha = draw(render, cam, model, black)
        feature, eyes = _feature_pixels(box, points, valid)
        base = base_alpha[crop].clone()
        luma = rgb.new_tensor((.2126, .7152, .0722))[:, None, None]
        source_luma = (rgb*luma).sum(0)
        prior_luma = (packet['render'][(slice(None), *crop)]*luma).sum(0)/base.clamp_min(1e-4)
        dark_eye = eyes & (source_luma < .20) & (base >= .95) & ((prior_luma-source_luma).abs() < .08)
        targets.append(SimpleNamespace(
            row=row, cam=cam, slot=slot, crop=crop, box=box, valid=valid, outside=outside,
            rgb=rgb, alpha=alpha, baseline_alpha=base, feature=feature, dark_eye=dark_eye,
            luma=luma, eye_limit=torch.maximum(source_luma+.01, prior_luma+.005)))
    return targets


def _image_terms(target, model, render, black):
    model.select_mesh_by_timestep(target.slot)
    packet, full_alpha = draw(render, target.cam, model, black)
    rgb = packet['render'][(slice(None), *target.crop)]
    alpha = full_alpha[target.crop]
    color, coverage = masked_face_loss(rgb, alpha, target.rgb, target.alpha, target.valid)
    preserved = target.valid & (target.baseline_alpha >= .95)
    opaque_deficit = ((target.baseline_alpha-alpha-.002).relu()*preserved).sum()/preserved.sum().clamp_min(1)
    holes = ((.95-alpha).relu()*preserved).sum()/preserved.sum().clamp_min(1)
    exterior = (alpha*target.outside).sum()/target.outside.sum().clamp_min(1)
    feature = ((rgb-target.rgb*target.alpha).abs().mean(0)*target.feature).sum()/target.feature.sum().clamp_min(1)
    luma = (rgb*target.luma).sum(0)/alpha.clamp_min(1e-4)
    black_eye = ((luma-target.eye_limit).relu()*target.dark_eye).sum()/target.dark_eye.sum().clamp_min(1)
    return color+.20*feature+.30*black_eye, .45*coverage+.95*opaque_deficit+.50*holes+.15*exterior, packet


def _restrict_gradients(model, observed):
    for name in TRAINABLE:
        gradient = getattr(model, name).grad
        require(gradient is not None, f'No gradient reaches {name}.')
        gradient[~observed] = 0
        require(bool(torch.isfinite(gradient).all()), f'Invalid {name} gradient.')


@torch.no_grad()
def _restrict_values(model, observed, geometry_only, associated, caps, dc_lower, dc_upper):
    scaling = model._scaling
    # Axis identities may rotate. Bound the sorted widths without assigning a material plane.
    material_rows = observed & ~associated
    widths, order = scaling[material_rows].sort(dim=1)
    prior = caps[material_rows].sort(dim=1).values
    widths = torch.maximum(widths, prior+np.log(VISIBLE_SCALE_RANGE[0]))
    widths = torch.minimum(widths, prior+np.log(VISIBLE_SCALE_RANGE[1]))
    scaling[material_rows] = torch.empty_like(widths).scatter(1, order, widths)
    # The associated skin uses a single covariance contract instead of an initial-width floor.
    scaling[associated] = scaling[associated].clamp(min=np.log(SURFACE_R28['min_axis_sigma_local']))
    rows = torch.where(geometry_only)[0]
    major = caps[rows].argmax(dim=1)
    base = caps[rows, major]
    bounded = scaling[rows, major].clamp(min=base+np.log(UNSEEN_SCALE_FLOOR), max=base)
    scaling[rows] = caps[rows]
    scaling[rows, major] = bounded
    model._features_dc[observed] = model._features_dc[observed].clamp(dc_lower[observed], dc_upper[observed])


def _marked_rays(path, expected):
    document = rt.read(path)
    require(document['baseline_bound_ply_sha256'] == expected['bound_ply_sha256'] and
            document['baseline_raw_ply_sha256'] == expected['inputs_sha256']['world_ply'],
            'Prior marked-ray inspection is not from this starting scene.')
    return {name: [[int(row['stable_id']) for row in pair['artifact']['top_rows']] for pair in item['pairs']]
            for name, item in document['previous_novel_ray_coverage'].items()}


def _gradient_audit(model, targets, render, black, evidence, rays):
    params = tuple(getattr(model, key) for key in TRAINABLE)
    receipt = {}
    observed = evidence.observed
    source_slots = {t.slot: t.row['frame'] for t in targets}
    for kind, term in (('rgb', 0), ('alpha', 1)):
        raw_seen = torch.zeros_like(observed)
        counts = torch.zeros_like(observed, dtype=torch.int32)
        sizes = {key: 0. for key in TRAINABLE}
        for target in targets:
            terms = _image_terms(target, model, render, black)
            gradients = torch.autograd.grad(terms[term], params, allow_unused=True)
            rows = torch.zeros_like(observed)
            for key, gradient in zip(TRAINABLE, gradients, strict=True):
                if gradient is None:
                    continue
                require(bool(torch.isfinite(gradient).all()), f'Non-finite {kind} gradient in {key}.')
                sizes[key] = max(sizes[key], float(gradient[observed].abs().max()))
                rows |= (gradient.flatten(1).abs() > 1e-12).any(1)
            raw_seen |= rows
            counts += rows & observed
        receipt[kind] = dict(raw_head_rows=int((raw_seen & evidence.head_scope).sum()),
                             active_rows_after_filter=int((counts > 0).sum()),
                             active_unknown_rows_after_filter=int(((counts > 0) & (evidence.labels == 0)).sum()),
                             active_mass_after_filter=float(evidence.evidence_interior_mass[counts > 0].sum()),
                             maximum_retained_gradient=sizes)
        receipt[kind]['rows_with_gradient_by_material'] = [int(((counts > 0) & (evidence.labels == label)).sum())
                                                          for label in range(6)]
        receipt[kind]['marked_rows'] = {}
        prior_rows = sorted({row for angle in rays.values() for ray in angle for row in ray})
        for index, stable_id in enumerate(model.densification_ids.cpu().tolist()):
            if stable_id in prior_rows:
                receipt[kind]['marked_rows'][str(stable_id)] = dict(raw=bool(raw_seen[index]),
                    retained_views=int(counts[index]), material=int(evidence.labels[index]),
                    photometric_eligible=bool(observed[index]), geometry_only=bool(evidence.unknown_unobserved_elongated[index]))
    require(all(receipt[key]['active_rows_after_filter'] > 0 for key in ('rgb', 'alpha')),
            'Observed rows must receive independent real image RGB and alpha gradients.')
    receipt['source_slots'] = source_slots
    receipt['observed_rows'] = int(observed.sum())
    receipt['observed_mass'] = float(evidence.evidence_interior_mass[observed].sum())
    receipt['head_mass'] = float(evidence.evidence_interior_mass[evidence.head_scope].sum())
    receipt['marked_rays'] = rays
    return receipt


@torch.no_grad()
def _frozen(model, original, labels, spill, observed, skin, unseen, lineage, geometry_ref, caps,
            associated, dc_lower, dc_upper):
    current = model.densification_ids
    original_ids = original['ids']
    sorted_ids, order = original_ids.sort()
    positions = torch.searchsorted(sorted_ids, current).clamp_max(len(sorted_ids)-1)
    existing = sorted_ids[positions] == current
    original_row = order[torch.searchsorted(sorted_ids, lineage)]
    require(torch.equal(original_ids[original_row], lineage) and torch.equal(current[existing], lineage[existing]),
            'Existing rows or children changed original source identity.')
    fixed = ~observed & ~unseen
    require(bool(existing[~observed].all()), 'A child has no native-observed ancestor.')
    for values, name in ((model.binding, 'binding'), (labels, 'labels'), (spill, 'spill'),
                         (observed, 'observed'), (skin, 'skin'), (unseen, 'unseen')):
        require(torch.equal(values, original[name][original_row]), f'Split altered {name} ownership.')
    require(int((~observed).sum()) == int((~original['observed']).sum()), 'A protected row was removed.')
    changes = {}
    for name in RAW_PARAMETERS:
        mask = existing if name in FROZEN_GLOBAL else fixed if name == '_scaling' else ~observed
        require(torch.equal(getattr(model, name)[mask], original[name][original_row[mask]]),
                f'Protected values changed in {name}.')
        current_values = getattr(model, name)[observed]
        moved = (current_values != original[name][original_row[observed]]).flatten(1).any(1)
        changes[name] = [int((moved & (labels[observed] == label)).sum()) for label in range(6)]
    ordinary_skin = torch.where(skin & ~associated)[0]
    limit = face_surface_limits(model, ordinary_skin, geometry_ref)
    center = (limit['center'].abs()-limit['initial_center'].abs()).clamp_min(0)
    thickness = (limit['normal_sigma']-limit['initial_normal_sigma']).clamp_min(0)
    center_increase = center.max() if len(center) else center.new_zeros(())
    thickness_increase = thickness.max() if len(thickness) else thickness.new_zeros(())
    require(float(center_increase) <= 1e-7 and float(thickness_increase) <= 1e-7,
            'Trusted skin exceeded its original normal envelope.')
    differences = model._scaling[unseen]-caps[unseen]
    require(bool((differences <= 0).all()) and bool((differences >= np.log(UNSEEN_SCALE_FLOOR)-1e-7).all())
            and bool(((differences != 0).sum(1) <= 1).all()), 'Unobserved shape prior exceeded its bound.')
    dc = model._features_dc[observed]
    require(bool(((dc >= dc_lower[observed]-1e-7) & (dc <= dc_upper[observed]+1e-7)).all()),
            'Observed SH0 updates exceeded their native color bounds.')
    return dict(frozen_non_skin_rows=int(fixed.sum()), limited_unknown_rows=int(unseen.sum()),
                modified_limited_unknown_rows=int((differences != 0).any(dim=1).sum()),
                observed_rows=int(observed.sum()), observed_unknown_rows=int((observed & (labels == 0)).sum()),
                skin_rows=int(skin.sum()), new_skin_rows=int((~existing & skin).sum()),
                new_observed_unknown_rows=int((~existing & (labels == 0)).sum()), changes_by_material=changes,
                maximum_skin_center_increase_m=float(center_increase),
                maximum_skin_thickness_increase_m=float(thickness_increase),
                owner_envelope_rows=len(ordinary_skin), associated_rows=int(associated.sum()))


@torch.no_grad()
def _measure(model, skin, slots):
    ids = torch.where(skin)[0]
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


@torch.no_grad()
def _association_metrics(model, association, reference, associated, step):
    ids = torch.where(associated)[0]
    saved_pose = model.timestep
    names = ('center','owner_plane','owner_closest_distance','target_closest_distance','anchor_distance',
             'normal_sigma','tangent_major_sigma','center_cap','normal_sigma_cap','tangent_cap','axis_excess')
    values = {name: [] for name in names}
    max_center = max_normal = 0.
    native_slots = set(association.source_slots.tolist())
    try:
        for slot in range(model.num_timesteps):
            model.select_mesh_by_timestep(slot)
            metrics = surface_metrics_r28(model, ids, reference, step)
            for name in names:
                values[name].append(metrics[name].abs())
            if slot in native_slots:
                for key, threshold in (('center','center_goal_m'),('normal_sigma','normal_sigma_goal_m')):
                    initial = metrics['initial_'+key].abs()
                    excess = ((metrics[key].abs()-initial)*(initial <= SURFACE_R28[threshold])).clamp_min(0)
                    maximum = float(excess.max())
                    if key == 'center':
                        max_center = max(maximum, max_center)
                    else:
                        max_normal = max(maximum, max_normal)
    finally:
        model.select_mesh_by_timestep(saved_pose)
    require(max(max_center,max_normal) <= 1e-7, 'Originally good native proxy geometry exceeded its ceiling.')
    summary = {}
    for name, rows in values.items():
        values_flat = torch.cat(rows)
        summary[name] = dict(p50=float(values_flat.median()), p95=float(values_flat.quantile(.95)),
                             max=float(values_flat.max()))
    return dict(rows=len(ids),poses=model.num_timesteps,source_slots=sorted(native_slots),
                maximum_initially_good_center_increase_m=max_center,
                maximum_initially_good_normal_sigma_increase_m=max_normal, metrics=summary)


@torch.no_grad()
def _native_coverage(model, targets, render, black):
    start_pose = model.timestep
    rows = []
    try:
        for target in targets:
            model.select_mesh_by_timestep(target.slot)
            alpha = render(target.cam, model, PIPE, black, override_color=torch.ones_like(model.get_xyz))['render'][0]
            valid = target.valid
            now, before = alpha[target.crop][valid], target.baseline_alpha[valid]
            opaque = before >= .95
            drop = (before[opaque]-now[opaque]).clamp_min(0)
            rows.append(dict(frame=target.row['frame'], pose_index=target.slot, valid=int(valid.sum()),
                originally_opaque=int(opaque.sum()),
                alpha_lt_095_before=int((before < .95).sum()), alpha_lt_095_after=int((now < .95).sum()),
                new_alpha_lt_095=int((opaque & (now < .95)).sum()),
                new_alpha_lt_080=int((opaque & (now < .80)).sum()),
                opaque_alpha_drop_p99=float(drop.quantile(.99)) if len(drop) else 0.))
    finally:
        model.select_mesh_by_timestep(start_pose)
    return rows


def _optimization(baseline):
    options = rt.read(baseline/'gaussian-run/config.json')['optimization']
    # Small local steps on a mature snapshot. The opacity and SH parameters stay fixed.
    return {**options, 'position_lr_init': .0005, 'position_lr_final': .000025,
            'scaling_lr': .002, 'rotation_lr': .0005, 'feature_lr': .0005, 'opacity_lr': 0.}


def _run(baseline, root, evidence, artifact_audit, surface_path, color_path, probe):
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
    material, material_receipt = _load_material(prepared, evidence)
    association, association_receipt = _load_association(prepared, surface_path, material, material_receipt)
    dc_lower, dc_upper, color_guarded = _color_limits(prepared, color_path, material, material_receipt)
    labels, spill, observed = material.labels, material.silhouette_spill, material.observed
    skin, unseen = material.geometry_eligible, material.unknown_unobserved_elongated
    protected = material.possible_protected_material
    associated = association.mask(model.densification_ids)
    lineage = material.stable_ids.clone()
    marked_rays = _marked_rays(artifact_audit, prepared.source_manifest)
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
        aux_replay=rt.sha(prepared.auxiliary/'replay.json'), marked_artifact_audit=rt.sha(artifact_audit),
        native_auxiliary_landmarks=rt.sha(baseline/'fit/joint-candidates.npz'),
        surface_association=rt.sha(surface_path), surface_association_receipt=rt.sha(surface_path.with_suffix('.json')),
        native_color_guard=rt.sha(color_path), native_color_guard_receipt=rt.sha(color_path.with_suffix('.json')))
    config = dict(
        official_fit_provenance=verify_official_fit(baseline), pinned_inputs=pinned,
        training_duration_policy=dict(checkpoints=list(ENDPOINTS), schedule_steps=ENDPOINTS[-1],
                                      independent_postprocessing=False, single_main_training=True),
        optimization=options, seed=0, training_source_bins=row_bins,
        visible_scale_range=list(VISIBLE_SCALE_RANGE), unseen_major_scale_floor=UNSEEN_SCALE_FLOOR,
        surface_schedule=SURFACE_R28.copy(), surface_geometry_weight=.20,
        color_bounds=dict(without_repeated_support=COLOR_WITHOUT_REPEATED_SUPPORT,
                          supported_max_change=COLOR_SUPPORTED_MAX_CHANGE),
        split_steps=list(SPLIT_STEPS),
        train_frames=[row['frame'] for row in prepared.rows[:24]],
        holdout_frames=[row['frame'] for row in prepared.rows[24:28]],
        auxiliary_frames=[row['frame'] for row in prepared.rows[28:]],
        auxiliary_frequency=8, candidate='R28_face_surface', execution='cuda_probe' if probe else 'main')
    rt.save(out/'config.json', config)
    rt.save(root/'run.json', dict(rt.read(baseline/'run.json'), state='running', source=str(baseline), candidate='R28'))
    original = dict(
        ids=model.densification_ids.clone(), labels=labels.clone(), spill=spill.clone(), binding=model.binding.clone(),
        observed=observed.clone(), skin=skin.clone(), unseen=unseen.clone(),
        **{name: getattr(model, name).detach().clone() for name in RAW_PARAMETERS})
    caps = model._scaling.detach().clone()
    model.select_mesh_by_timestep(reference)
    model.max_radii2D = torch.zeros_like(model._opacity[:, 0])
    model.training_setup(SimpleNamespace(**options))
    for name in TRAINABLE:
        getattr(model, name).requires_grad_(True)
    require(not model.optimizer.state, 'R28 must start with an empty optimizer.')
    geometry_ref = snapshot_face_geometry(model)
    surface_ref = prepare_surface_geometry(model, association)
    surface_before = _association_metrics(model, association, surface_ref, associated, 0)
    audit = _gradient_audit(model, prepared.targets, render, black, material, marked_rays)
    rt.save(out/'native-gradient-audit.json', audit)
    require(not model.optimizer.state and all(getattr(model, name).grad is None for name in TRAINABLE),
            'Gradient audit must not step or accumulate in the live optimizer.')
    geom_slots = [reference, *[slot for slot in (0, 6, 12, 18, 23) if slot != reference]]
    geometry_before = _measure(model, skin, geom_slots)
    rt.save(out/'initial-geometry.json', dict(metrics=geometry_before, material_rows=labels.bincount(minlength=6).tolist(),
        evidence_spill_rows=int(spill.sum()), proxy=surface_before, association=association_receipt['association'],
        observed_color_rows=int(observed.sum()), repeated_color_rows=int(color_guarded.sum())))
    model.face_surface_view_support = torch.zeros_like(labels, dtype=torch.int32)
    # Track distinct supporting cameras, not the count of repeated training draws.
    seen_cameras = torch.zeros_like(labels, dtype=torch.int64)
    draws, splits, endpoints, aux_updates = [], [], [], Counter()
    start = time.monotonic()
    stop = 24 if probe else ENDPOINTS[-1]
    for step in range(1, stop+1):
        slot = next(order)
        draws.append(slot)
        model.update_learning_rate(step)
        model.optimizer.zero_grad(set_to_none=True)
        target = prepared.targets[slot]
        rgb, alpha, packet = _image_terms(target, model, render, black)
        ids = torch.where(associated)[0]
        owner_ids = torch.where(skin & ~associated)[0]
        geometry, _ = surface_loss_r28(model, ids, surface_ref, step)
        owner_geometry, _ = face_surface_loss_r27(model, owner_ids, geometry_ref)
        unknown, _ = unknown_support_loss(model, torch.where(observed & (labels == 0) & ~protected)[0],
                                          geometry_ref, observed=True)
        photo = rgb+alpha
        loss = photo + .20*geometry + .16*len(owner_ids)/(len(owner_ids)+len(ids))*owner_geometry + .05*unknown
        if step % 8 == 0:
            auxiliary = prepared.targets[24 + (step//8-1) % 4]
            aux_rgb, aux_alpha, _ = _image_terms(auxiliary, model, render, black)
            loss = loss + .30*(aux_rgb+aux_alpha)
            aux_updates[auxiliary.row['frame']] += 1
        require(bool(torch.isfinite(loss)), f'Non-finite R28 loss at {step}.')
        loss.backward()
        _restrict_gradients(model, observed)
        shape_only, _ = unknown_support_loss(model, torch.where(unseen)[0], geometry_ref, observed=False)
        if shape_only.requires_grad:
            (.04*shape_only).backward()
        model.optimizer.step()
        _restrict_values(model, observed, unseen, associated, caps, dc_lower, dc_upper)
        project_skin_envelope(model, owner_ids, geometry_ref)
        project_surface_envelope(model, ids, surface_ref)
        with torch.no_grad():
            gradient = packet['viewspace_points'].grad
            if gradient is not None:
                weight = gradient[:, :2].norm(dim=1)
                valid = observed & packet['visibility_filter'] & (weight > 0)
                flag = 1 << slot
                first = valid & ((seen_cameras & flag) == 0)
                model.face_surface_view_support += first
                seen_cameras[valid] |= flag
                model.xyz_gradient_accum[valid, 0] += weight[valid]
                model.denom[valid, 0] += 1
        model.optimizer.zero_grad(set_to_none=True)
        if step in SPLIT_STEPS or (probe and step == stop):
            from .face_surface_split import try_skin_split
            previous_ids = model.densification_ids
            split_eligible = observed & ~protected & ~associated
            labels, spill, caps, event = try_skin_split(model, labels, spill, caps, prepared.targets,
                render, PIPE, black, step, observed=split_eligible, skin=skin & ~associated)
            if event['committed_parents']:
                parents = previous_ids.new_tensor(event['parent_ids'])
                child_ids = previous_ids.new_tensor(event['child_ids'])
                ids_sorted, indices = previous_ids.sort()
                parent_rows = indices[torch.searchsorted(ids_sorted, parents)]
                require(torch.equal(previous_ids[parent_rows], parents), 'Split lineage refers to an absent row.')
                keep = ~torch.isin(previous_ids, parents)
                children = parent_rows.repeat_interleave(child_ids.shape[1])
                require(not bool(associated[parent_rows].any()), 'A split would change a fixed surface anchor.')
                observed, skin, unseen, protected, lineage, dc_lower, dc_upper, color_guarded = (
                    torch.cat((values[keep], values[children])) for values in
                    (observed, skin, unseen, protected, lineage, dc_lower, dc_upper, color_guarded))
                geometry_ref = geometry_ref.with_children(parents, child_ids)
                associated = association.mask(model.densification_ids)
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
            report = _frozen(model, original, labels, spill, observed, skin, unseen, lineage, geometry_ref, caps,
                             associated, dc_lower, dc_upper)
            report['geometry_before'] = geometry_before
            report['geometry_after'] = _measure(model, skin, geom_slots)
            report['proxy_geometry_before'] = surface_before
            report['proxy_geometry_after'] = _association_metrics(model, association, surface_ref, associated, step)
            report['native_coverage_after_training'] = _native_coverage(model, prepared.targets, render, black)
            report['repeated_source_color_rows'] = int(color_guarded.sum())
            report['native_gradient_audit_sha256'] = rt.sha(out/'native-gradient-audit.json')
            metadata = dict(observed=observed, skin=skin, geometry_only=unseen, original_row_ids=lineage,
                            possible_protected_material=protected, geometry_reference_ids=geometry_ref.stable_ids,
                            geometry_reference=geometry_ref.geometry, associated=associated,
                            surface_association={key:torch.as_tensor(value) for key,value in association.arrays().items()},
                            surface_geometry_reference={key:torch.as_tensor(value) for key,value in surface_ref.arrays().items()},
                            dc_lower=dc_lower, dc_upper=dc_upper, source_color_guarded=color_guarded,
                            seen_cameras=seen_cameras, face_surface_view_support=model.face_surface_view_support,
                            reference_pose_index=reference, auxiliary_updates=dict(aux_updates),
                            recovery_schema='shuolee-r28-optimizer-rng-topology/1')
            if probe:
                probe_config = {**config, 'training_duration_policy': dict(checkpoints=[step], schedule_steps=stop)}
                endpoint = save_endpoint(root, prepared, step, probe_config, render, black,
                                         draws, splits, labels, spill, caps, report, training_metadata=metadata)
            else:
                endpoint = save_endpoint(root, prepared, step, config, render, black,
                                         draws, splits, labels, spill, caps, report, training_metadata=metadata)
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
    parser.add_argument('--artifact-audit', type=Path, required=True)
    parser.add_argument('--surface-association', type=Path, required=True)
    parser.add_argument('--native-color-guard', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    _run(args.baseline.resolve(), args.root.resolve(), args.material.resolve(), args.artifact_audit.resolve(),
         args.surface_association.resolve(), args.native_color_guard.resolve(), args.probe)


if __name__ == '__main__':
    main()

"""Apply training-only native skin evidence and initialize local eye coverage."""

from dataclasses import fields

import numpy as np
import torch

from . import runtime as rt
from .bound_replay import RAW_PARAMETERS
from .face_surface_association_r28 import SurfaceAssociation
from .face_surface_bundle import require


COVERAGE_R29 = dict(seed_opacity=.35, seed_normal_ratio=.55, seed_rgb_margin=.025,
                    eye_radius_px=2, initial_hole_alpha=.94, target_alpha=.95,
                    eye_alpha_weight=.30)


def _packet(prepared, path, prior_path, material_path):
    receipt = rt.read(path.with_suffix('.json'))
    aux_start = prepared.main_count + prepared.holdout_count
    training = [row['frame'] for row in prepared.rows[:prepared.main_count]]
    auxiliary = [row['frame'] for row in prepared.rows[aux_start:]]
    require(receipt['schema'] == 'shuolee-r29-native-coverage/1'
            and receipt['archive_sha256'] == rt.sha(path)
            and receipt['baseline_bound_ply_sha256'] == prepared.source_manifest['bound_ply_sha256']
            and receipt['prior_association_sha256'] == rt.sha(prior_path)
            and receipt['material_receipt_sha256'] == rt.sha(material_path)
            and receipt['training_frames'] == training and receipt['auxiliary_frames'] == auxiliary
            and receipt['heldout_source_images_read'] == 0,
            'R29 coverage evidence changed its source or frame roles.')
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    names = tuple(field.name for field in fields(SurfaceAssociation))
    association = SurfaceAssociation.from_arrays({name: arrays.pop(name) for name in names}).to('cuda')
    source_slots = list(range(prepared.main_count)) + list(range(aux_start, len(prepared.rows)))
    require(association.source_slots.tolist() == source_slots,
            'R29 coverage has an unregistered pose or different native column order.')
    return association, arrays, receipt


@torch.no_grad()
def load_native_coverage(prepared, path, prior_path, material_path, material,
                         prior, dc_lower, dc_upper, color_guarded, dc_constant, max_rgb_change):
    association, data, receipt = _packet(prepared, path, prior_path, material_path)
    model = prepared.model
    ids = model.densification_ids
    require(len(ids) > 0, 'R29 source has no Gaussians.')
    matched, present = association.indices(prior.stable_ids)
    require(bool(present.all())
            and all(torch.equal(getattr(association, name)[matched], getattr(prior, name))
                    for name in ('owner_faces', 'target_faces', 'barycentric', 'support_views'))
            and torch.equal(association.source_slots, prior.source_slots)
            and torch.equal(association.source_bins, prior.source_bins),
            'R29 evidence did not preserve the previously corroborated skin associations.')
    indices, included = association.indices(ids)
    require(int(included.sum()) == len(association.stable_ids)
            and torch.equal(model.binding[included], association.owner_faces[indices[included]])
            and bool((association.target_faces < len(model.faces)).all()),
            'R29 evidence did not match its live Gaussian owners.')
    expanded = included & ~prior.mask(ids)
    claimed = torch.as_tensor(data['expanded_ids'], device=ids.device)
    require(claimed.dtype == torch.long
            and torch.equal(claimed, ids[expanded].sort().values)
            and len(claimed) == receipt['expanded_rows']
            and bool((material.head_scope[expanded] & ~material.possible_protected_material[expanded]
                      & ~material.silhouette_spill[expanded]
                      & ((material.labels[expanded] == 0) | (material.labels[expanded] == 1))).all()),
            'R29 admits a protected material or an unsupported original row.')
    sorted_ids, order = ids.sort()
    promoted = order[torch.searchsorted(sorted_ids, claimed)]
    low, high, valid = (torch.as_tensor(data[name], device=ids.device)
                        for name in ('expanded_rgb_lower', 'expanded_rgb_upper', 'expanded_rgb_valid'))
    require(low.shape == high.shape == (len(claimed), 3) and valid.dtype == torch.bool
            and valid.shape == claimed.shape
            and bool(((low[valid] >= 0) & (low[valid] <= high[valid]) & (high[valid] <= 1)).all()),
            'R29 repeated native color bounds are invalid.')
    bounded = promoted[valid]
    source_rgb = model._features_dc[bounded, 0]*dc_constant + .5
    low = torch.maximum(low[valid].minimum(source_rgb), source_rgb-max_rgb_change)
    high = torch.minimum(high[valid].maximum(source_rgb), source_rgb+max_rgb_change)
    dc_lower[bounded, 0] = (low-.5)/dc_constant
    dc_upper[bounded, 0] = (high-.5)/dc_constant
    color_guarded[bounded] = True
    material.labels[expanded] = 1
    material.observed[expanded] = True
    material.geometry_eligible[expanded] = True
    material.unknown_unobserved_elongated[expanded] = False
    return association, data, receipt


@torch.no_grad()
def add_eye_seeds(prepared, association, data, receipt, material,
                  dc_lower, dc_upper, color_guarded, dc_constant):
    model = prepared.model
    ids = model.densification_ids
    bind, bary, rgb, sigma, support = (torch.as_tensor(data[name], device=ids.device)
        for name in ('seed_faces', 'seed_barycentric', 'seed_rgb', 'seed_sigma_m', 'seed_support_views'))
    size = len(bind)
    require(size == receipt['seed_count'] and bind.dtype == torch.long and bind.shape == (size,)
            and bool(((bind >= 0) & (bind < len(model.faces))).all())
            and bary.shape == rgb.shape == (size, 3) and bary.dtype == torch.float32
            and bool((bary >= 0).all()) and torch.allclose(bary.sum(1), bary.new_ones(size))
            and bool(((rgb >= 0) & (rgb <= 1)).all())
            and sigma.shape == (size,) and bool((torch.isfinite(sigma) & (sigma > 0)).all())
            and support.dtype == torch.bool and support.shape == (size, len(association.source_slots)),
            'R29 eye seed arrays are invalid.')
    if not size:
        return association, dc_lower, dc_upper, color_guarded
    slots = association.source_slots
    native_xy = data['seed_pixels_xy']
    require(native_xy.shape == (size, len(slots), 2)
            and np.isfinite(native_xy).all()
            and np.all(native_xy[support.cpu().numpy()] >= 0)
            and np.all(native_xy[~support.cpu().numpy()] == -1),
            'R29 eye seed positions disagree with signed native evidence.')
    new_ids = torch.arange(model.next_densification_id, model.next_densification_id+size, device=ids.device)
    require(bool(new_ids[0] > ids.max()) and model.optimizer is None,
            'R29 eye seeds must be added before creating the optimizer with new stable IDs.')
    joined = SurfaceAssociation.from_arrays(dict(
        stable_ids=torch.cat((association.stable_ids, new_ids)),
        owner_faces=torch.cat((association.owner_faces, bind)),
        target_faces=torch.cat((association.target_faces, bind)),
        barycentric=torch.cat((association.barycentric, bary)),
        support_views=torch.cat((association.support_views, support)),
        source_slots=association.source_slots, source_bins=association.source_bins))
    model.select_mesh_by_timestep(prepared.reference)
    triangles = model.vertices_sequence[prepared.reference][model.faces[bind]]
    anchors = (bary[:, :, None]*triangles).sum(1)
    frame_size = model.face_scaling[bind, 0]
    centers = ((model.face_orien_mat[bind].transpose(1, 2)
                @ (anchors-model.face_center[bind])[:, :, None]).squeeze(-1)/frame_size[:, None])
    radius = sigma/frame_size
    rotation = centers.new_zeros((size, 4))
    rotation[:, 0] = 1
    dc = ((rgb-.5)/dc_constant)[:, None, :]
    peak = torch.logit(centers.new_full((size, 1), COVERAGE_R29['seed_opacity']))
    added = dict(_xyz=centers, _scaling=torch.stack((radius, radius*COVERAGE_R29['seed_normal_ratio'],
                 radius), dim=1).log(), _rotation=rotation, _features_dc=dc, _opacity=peak,
                 _features_rest=model._features_rest.new_zeros((size, *model._features_rest.shape[1:])))
    for name in RAW_PARAMETERS:
        current = getattr(model, name)
        setattr(model, name, torch.nn.Parameter(torch.cat((current, added[name].to(current)), 0),
                                                  requires_grad=current.requires_grad))
    model.binding = torch.cat((model.binding, bind))
    model.binding_counter = torch.bincount(model.binding, minlength=len(model.faces)).to(torch.int32)
    model.densification_ids = torch.cat((ids, new_ids))
    model.next_densification_id += size
    for name, value in vars(material).items():
        if name == 'stable_ids':
            extra = new_ids
        elif name == 'owner_faces':
            extra = bind
        elif name == 'visible_views':
            extra = support.sum(1).to(value.dtype)
        else:
            extra = value.new_full((size,), name in ('labels', 'observed', 'head_scope', 'geometry_eligible'))
        setattr(material, name, torch.cat((value, extra)))
    require(torch.allclose(model.get_xyz[-size:], anchors, atol=2e-6, rtol=0)
            and torch.isfinite(model._scaling[-size:]).all(), 'R29 seeded centers failed binding parity.')
    margin = COVERAGE_R29['seed_rgb_margin']
    low = (((rgb-margin).clamp(0, 1)-.5)/dc_constant)[:, None, :]
    high = (((rgb+margin).clamp(0, 1)-.5)/dc_constant)[:, None, :]
    return joined, torch.cat((dc_lower, low)), torch.cat((dc_upper, high)), torch.cat((
        color_guarded, torch.ones(size, dtype=torch.bool, device=ids.device)))


def mark_eye_targets(targets, association, data):
    pixels = data['seed_pixels_xy']
    slots = {slot: column for column, slot in enumerate(association.source_slots.tolist())}
    radius = COVERAGE_R29['eye_radius_px']
    for target in targets:
        x0, y0, x1, y1 = target.box
        xy = np.floor(pixels[:, slots[target.slot]]).astype(np.int64)
        mask = torch.zeros_like(target.valid)
        for x, y in xy:
            if x0 <= x < x1 and y0 <= y < y1:
                mask[max(0, y-y0-radius):y-y0+radius+1, max(0, x-x0-radius):x-x0+radius+1] = True
        target.eye_deficit = (mask & target.eyes & target.valid & ~target.dark_eye
                              & (target.baseline_alpha < COVERAGE_R29['initial_hole_alpha']))

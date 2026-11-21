"""Preserve credible initial facial geometry while reducing unsupported splat extent."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .face_surface_geometry import FACE_SURFACE, metric_face_surface
from .surface_attachment import _geometry


FACE_SURFACE_R27 = {
    'center_excess_target_ratio': .85,
    'thickness_excess_target_ratio': .90,
    'normal_sigma_minimum_ratio': .85,
    'normal_sigma_floor_weight': .25,
    'unknown_elongation_start': 3.,
    'unknown_elongation_full': 6.,
    'unknown_major_preferred_shrink': .15,
    'unknown_unobserved_elongation_start': 6.,
    'unknown_unobserved_elongation_full': 12.,
    'unknown_unobserved_major_preferred_shrink': .07,
    'unknown_principal_minimum_ratio': .95,
    'unknown_envelope_weight': 1.,
    'unknown_elongation_weight': .35,
}


@dataclass(frozen=True)
class FaceGeometryReference:
    """Stable-ID index of the original triangle-local geometry, independent of pose."""
    stable_ids: torch.Tensor
    geometry: torch.Tensor

    def lookup(self, ids):
        positions = torch.searchsorted(self.stable_ids, ids.contiguous()).clamp_max(len(self.stable_ids)-1)
        if not bool((self.stable_ids[positions] == ids).all()):
            raise ValueError('Facial geometry reference is missing a requested stable ID.')
        return self.geometry[positions]

    @torch.no_grad()
    def with_children(self, parent_ids, child_ids):
        """Register exact split children with the matching parent reference."""
        parent_ids = torch.as_tensor(parent_ids, device=self.stable_ids.device, dtype=torch.long)
        child_ids = torch.as_tensor(child_ids, device=self.stable_ids.device, dtype=torch.long)
        if child_ids.ndim != 2 or parent_ids.ndim != 1 or child_ids.shape[0] != len(parent_ids):
            raise ValueError('Expected one stable child-ID row per split parent.')
        if not child_ids.numel():
            return self
        inherited = self.lookup(parent_ids).repeat_interleave(child_ids.shape[1], dim=0)
        ids = torch.cat((self.stable_ids, child_ids.flatten()))
        ids, order = ids.sort()
        if bool((ids[1:] == ids[:-1]).any()) or bool((child_ids < 0).any()):
            raise ValueError('A facial geometry child stable ID is invalid or already registered.')
        return FaceGeometryReference(ids, torch.cat((self.geometry, inherited))[order])


@torch.no_grad()
def snapshot_face_geometry(model):
    """Freeze original signed normal center, actual normal sigma and principal widths."""
    if model.binding is None or not len(model.binding):
        raise ValueError('Facial geometry requires a nonempty triangle-bound model.')
    stable_ids = model.densification_ids
    if stable_ids.shape != (len(model.binding),) or stable_ids.dtype != torch.long:
        raise ValueError('Facial geometry requires one int64 stable ID per row.')
    ids, order = stable_ids.sort()
    if bool((ids < 0).any()) or bool((ids[1:] == ids[:-1]).any()):
        raise ValueError('Facial geometry requires unique nonnegative stable IDs.')
    center, normal, _, _, _ = _geometry(
        model._xyz, model._scaling, model._rotation, model._xyz.new_ones(len(ids)),
        detach_tangent_scales=False)
    principal = model._scaling.sort(dim=1).values.exp()
    geometry = torch.column_stack((center, normal, principal))
    if not bool(torch.isfinite(geometry).all()) or not bool((geometry[:, 1:] > 0).all()):
        raise ValueError('Invalid baseline center or covariance for facial geometry.')
    return FaceGeometryReference(ids, geometry[order])


def _initial_geometry(model, row_ids, baseline, source_ids):
    if source_ids is None:
        source_ids = model.densification_ids[row_ids]
    if source_ids.shape != row_ids.shape:
        raise ValueError('An original stable ID is required for every selected facial row.')
    scale = model.face_scaling[model.binding[row_ids], 0]
    return baseline.lookup(source_ids)*scale[:, None], scale


def _clearance(initial, metric_dead_zone, ratio):
    return torch.minimum(initial, torch.maximum(initial*ratio, torch.full_like(initial, metric_dead_zone)))


def face_surface_limits(model, row_ids, baseline, *, source_ids=None):
    """Return current pose metrics and per-row skin targets in metres."""
    initial, scale = _initial_geometry(model, row_ids, baseline, source_ids)
    values = metric_face_surface(model._xyz[row_ids], model._scaling[row_ids], model._rotation[row_ids], scale)
    center, normal = initial[:, 0].abs(), initial[:, 1]
    cfg = FACE_SURFACE_R27
    values.update(initial_center=initial[:, 0], initial_normal_sigma=normal,
        center_cap=_clearance(center, FACE_SURFACE['center_clearance_m'], cfg['center_excess_target_ratio']),
        normal_sigma_cap=_clearance(normal, FACE_SURFACE['normal_sigma_clearance_m'], cfg['thickness_excess_target_ratio']),
        normal_sigma_floor=normal.clamp(max=FACE_SURFACE['normal_sigma_clearance_m'])*cfg['normal_sigma_minimum_ratio'])
    return values


def _penalty(excess):
    ratio = excess.relu()/FACE_SURFACE['loss_unit_m']
    return F.smooth_l1_loss(ratio, torch.zeros_like(ratio))


def face_surface_loss_r27(model, row_ids, baseline, *, source_ids=None):
    """Use one-sided original center/thickness bounds on credible skin."""
    if not len(row_ids):
        zero = model._xyz.new_zeros(())
        return zero, dict.fromkeys(('center', 'thickness', 'thickness_floor', 'tangent', 'orientation'), zero)
    values = face_surface_limits(model, row_ids, baseline, source_ids=source_ids)
    terms = dict(center=_penalty(values['center'].abs()-values['center_cap']),
                 thickness=_penalty(values['normal_sigma']-values['normal_sigma_cap']),
                 thickness_floor=_penalty(values['normal_sigma_floor']-values['normal_sigma']),
                 tangent=_penalty(values['tangent_major_sigma']-values['tangent_cap']),
                 orientation=(values['axis_excess']*values['axis_reliability']).mean())
    weights = {**FACE_SURFACE, 'thickness_floor_weight': FACE_SURFACE_R27['normal_sigma_floor_weight']}
    return sum(weights[f'{name}_weight']*loss for name, loss in terms.items()), terms


@torch.no_grad()
def project_skin_envelope(model, row_ids, baseline, *, source_ids=None):
    """Keep true normal bounds in every pose without touching other rows or rotations."""
    if not len(row_ids):
        return
    ids = model.densification_ids[row_ids] if source_ids is None else source_ids
    reference = baseline.lookup(ids)
    local = model._xyz[row_ids]
    cap = reference[:, 0].abs()
    model._xyz[row_ids, 1] = local[:, 1].clamp(-cap, cap)

    logs = model._scaling[row_ids]
    widths = logs.exp()
    _, sigma, _, _, axes = _geometry(local, logs, model._rotation[row_ids],
                                      logs.new_ones(len(logs)), detach_tangent_scales=False)
    affected = sigma > reference[:, 1]*(1.+2e-6)
    if not bool(affected.any()):
        return
    rows, widths, axes = row_ids[affected], widths[affected], axes[affected]
    # Reduce normal covariance via the axis most aligned to the normal first.
    # A tangent axis with zero normal weight cannot be modified by this projection.
    weights = (axes[:, 1]/widths).square()
    variance = widths.square()
    floor = torch.minimum(widths, reference[affected, 2:3]*.5).square()
    excess = ((weights*variance).sum(dim=1)-reference[affected, 1].square()).clamp_min(0)
    for axis in weights.argsort(dim=1, descending=True).split(1, dim=1):
        weight = weights.gather(1, axis)[:, 0]
        old = variance.gather(1, axis)[:, 0]
        capacity = weight*(old-floor.gather(1, axis)[:, 0])
        taken = torch.minimum(capacity, excess)
        safe_weight = torch.where(weight > 0, weight, 1.)
        new = (old-taken/safe_weight).maximum(floor.gather(1, axis)[:, 0])
        variance.scatter_(1, axis, new[:, None])
        excess = (excess-taken).clamp_min(0)
    projected = .5*variance.log()
    # Preserve exact raw parameters on axes the projection did not change.
    model._scaling[rows] = torch.where(variance < widths.square(), projected, logs[affected])


def unknown_support_loss(model, row_ids, baseline, *, observed=False, source_ids=None):
    """Bound unlabeled footprints to their own baseline and reduce only supported long axes."""
    if not len(row_ids):
        zero = model._xyz.new_zeros(())
        return zero, dict.fromkeys(('support_envelope', 'elongation'), zero)
    initial, scale = _initial_geometry(model, row_ids, baseline, source_ids)
    observed = torch.as_tensor(observed, device=scale.device, dtype=torch.bool)
    if observed.shape not in (torch.Size([]), row_ids.shape):
        raise ValueError('Unknown support evidence must be scalar or match selected rows.')
    initial = initial[:, 2:]
    current = model._scaling[row_ids].sort(dim=1).values.exp()*scale[:, None]
    cfg = FACE_SURFACE_R27
    original_elongation = initial[:, 2]/initial[:, 1]
    observed_support = (original_elongation-cfg['unknown_elongation_start'])/(
        cfg['unknown_elongation_full']-cfg['unknown_elongation_start'])
    weak_support = (original_elongation-cfg['unknown_unobserved_elongation_start'])/(
        cfg['unknown_unobserved_elongation_full']-cfg['unknown_unobserved_elongation_start'])
    shrink = torch.where(observed, observed_support.clamp(0, 1)*cfg['unknown_major_preferred_shrink'],
                         weak_support.clamp(0, 1)*cfg['unknown_unobserved_major_preferred_shrink'])
    target = _clearance(initial[:, 2], FACE_SURFACE['tangent_sigma_min_cap_m'], 1.-shrink)
    floor = torch.cat((initial[:, :2]*cfg['unknown_principal_minimum_ratio'],
                       (initial[:, 2]*torch.minimum(1.-shrink,
                        torch.full_like(shrink, cfg['unknown_principal_minimum_ratio'])))[:, None]), dim=1)
    envelope = _penalty(current-initial)+_penalty(floor-current)
    elongation = _penalty((current[:, 2]-target)*(shrink > 0))
    terms = dict(support_envelope=envelope, elongation=elongation)
    weighted = cfg['unknown_envelope_weight']*envelope+cfg['unknown_elongation_weight']*elongation
    return weighted, terms

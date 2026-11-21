"""Metric geometry penalties for independently selected bound facial skin rows."""

from math import radians, sin

import torch
from torch.nn import functional as F

from .splat_surface_distance import folded_normal_distance
from .surface_attachment import _geometry


FACE_SURFACE = {
    'center_clearance_m': .00075,
    'normal_sigma_clearance_m': .00085,
    'tangent_sigma_min_cap_m': .0025,
    'tangent_sigma_max_cap_m': .0045,
    'tangent_support_multiple': 1.15,
    'axis_clearance_degrees': 15.,
    'relative_axis_gap_full_weight': .30,
    'loss_unit_m': .003,
    'center_weight': 1.,
    'thickness_weight': .65,
    'tangent_weight': .35,
    'orientation_weight': .18,
}


def _tangent_major_sigma(scaled_axes):
    """Square root of the major eigenvalue of the full local XZ covariance."""
    x, z = scaled_axes[:, 0], scaled_axes[:, 2]
    xx, zz, xz = x.square().sum(1), z.square().sum(1), (x*z).sum(1)
    # Vector norm supplies a finite zero subgradient at repeated eigenvalues.
    separation = torch.linalg.vector_norm(torch.stack((xx-zz, 2*xz), dim=1), dim=1)
    return (.5*(xx+zz+separation)).clamp_min(torch.finfo(x.dtype).tiny).sqrt()


def metric_face_surface(local, log_sigma, quaternion, face_scale):
    """Evaluate the rendered 3D covariance in metric owner-face coordinates."""
    center, normal_sigma, alignment_sq, sheet, scaled_axes = _geometry(
        local, log_sigma, quaternion, face_scale, detach_tangent_scales=False)
    tangent_sigma = _tangent_major_sigma(scaled_axes)
    tangent_cap = (FACE_SURFACE['tangent_support_multiple']*face_scale).clamp(
        FACE_SURFACE['tangent_sigma_min_cap_m'], FACE_SURFACE['tangent_sigma_max_cap_m'])
    # A near-degenerate minimum eigenspace has no unique thin-axis direction.
    ordered = log_sigma.detach().sort(dim=1).values
    gap = -torch.expm1(ordered[:, 0]-ordered[:, 1])
    reliability = (gap/FACE_SURFACE['relative_axis_gap_full_weight']).clamp(0, 1)
    axis_excess = (1.-alignment_sq-sin(radians(FACE_SURFACE['axis_clearance_degrees']))**2).relu()
    return dict(center=center, normal_sigma=normal_sigma, tangent_major_sigma=tangent_sigma,
                tangent_cap=tangent_cap, axis_alignment_sq=alignment_sq, axis_excess=axis_excess,
                axis_reliability=reliability, sheet=sheet, support_scale=face_scale)


def _bound_geometry(model, row_ids):
    if model.binding is None:
        raise ValueError('Facial skin geometry requires triangle-bound Gaussians.')
    return metric_face_surface(model._xyz[row_ids], model._scaling[row_ids], model._rotation[row_ids],
                               model.face_scaling[model.binding[row_ids], 0])


def face_surface_terms(values):
    """Robust clearance losses with no pressure below the metric dead zones."""
    unit = FACE_SURFACE['loss_unit_m']

    def excess(value, limit):
        outside = ((value-limit)/unit).relu()
        return F.smooth_l1_loss(outside, torch.zeros_like(outside))

    return dict(center=excess(values['center'].abs(), FACE_SURFACE['center_clearance_m']),
                thickness=excess(values['normal_sigma'], FACE_SURFACE['normal_sigma_clearance_m']),
                tangent=excess(values['tangent_major_sigma'], values['tangent_cap']),
                orientation=(values['axis_excess']*values['axis_reliability']).mean())


def face_surface_loss(model, row_ids):
    """Return the metric objective and each unweighted component for selected rows."""
    if not len(row_ids):
        zero = model._xyz.new_zeros(())
        return zero, dict.fromkeys(('center', 'thickness', 'tangent', 'orientation'), zero)
    terms = face_surface_terms(_bound_geometry(model, row_ids))
    weighted = sum(FACE_SURFACE[f'{name}_weight']*term for name, term in terms.items())
    return weighted, terms


@torch.no_grad()
def face_surface_metrics(model, row_ids):
    """Return per-row geometry in the model's currently selected pose."""
    values = _bound_geometry(model, row_ids)
    values['folded_distance'] = folded_normal_distance(values['center'], values['normal_sigma'])
    return values

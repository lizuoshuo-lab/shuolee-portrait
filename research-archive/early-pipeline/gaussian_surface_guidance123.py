"""Optional loss interfaces; not imported by the running upstream baseline.

These are local experimental losses, not GaussianAvatars paper defaults.
The teacher is a saved Gaussian reconstruction, not measured clothing truth.
"""
import math
import torch


def directional_color_energy(features_rest):
    """Unclamped linear-RGB directional variance under a uniform sphere.

    Real orthonormal SH: E[(c(direction)-c_DC)^2] = sum(c_lm^2)/(4*pi).
    DC, opacity, position and covariance are deliberately absent.
    """
    if features_rest.ndim != 3 or features_rest.shape[-1] != 3:
        raise ValueError('Expected [Gaussians, non-DC SH coefficients, RGB]')
    return features_rest.square().sum(dim=1).mean() / (4 * math.pi)


@torch.no_grad()
def closest_bound_barycentrics(triangles, centers):
    """Freeze closest point coordinates on each existing bound triangle.

    Do not change binding identities or call the nearest different body part.
    Degenerate triangles are rejected instead of inventing a plane.
    """
    a, b, c = triangles.unbind(dim=1)
    ab, ac, ap = b-a, c-a, centers-a
    d00, d01, d11 = (ab*ab).sum(-1), (ab*ac).sum(-1), (ac*ac).sum(-1)
    d20, d21 = (ap*ab).sum(-1), (ap*ac).sum(-1)
    den = d00*d11-d01*d01
    valid = den > torch.finfo(den.dtype).eps*d00*d11
    safe = torch.where(valid, den, torch.ones_like(den))
    v, w = (d11*d20-d01*d21)/safe, (d00*d21-d01*d20)/safe
    plane = torch.stack((1-v-w, v, w), -1)
    inside = valid & (plane >= 0).all(-1)
    candidates = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, j]-triangles[:, i]
        ratio = ((centers-triangles[:, i])*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-30)
        ratio = ratio.clamp(0, 1)
        bary = torch.zeros_like(plane)
        bary[:, i], bary[:, j] = 1-ratio, ratio
        candidates.append(bary)
    bary = torch.stack(candidates, 1)
    points = (bary[..., None]*triangles[:, None]).sum(2)
    best = (points-centers[:, None]).square().sum(-1).argmin(1)
    edge_bary = bary[torch.arange(len(bary), device=bary.device), best]
    return torch.where(inside[:, None], plane, edge_bary), valid, inside


def gaussian_to_mesh_loss(vertices, faces, binding, barycentrics,
                          teacher_centers, teacher_weights, beta_m=0.005):
    """One-way metric attraction, only through the posed mesh vertices.

    Teacher positions/weights and the attachment coordinates are stop-gradient.
    Detaching is essential: moving a student Gaussian or reducing its opacity
    must not be an alternative way of satisfying this mesh teaching term.
    """
    bary = barycentrics.detach()
    target, weight = teacher_centers.detach(), teacher_weights.detach()
    if beta_m <= 0 or not bool(weight.sum() > 0):
        raise ValueError('Positive robust scale and nonempty teacher support required')
    point = (vertices[faces[binding]] * bary[..., None]).sum(1)
    value = torch.nn.functional.smooth_l1_loss(point, target, beta=beta_m, reduction='none').sum(-1)
    return (value*weight).sum()/weight.sum()

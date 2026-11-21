"""R15 native foreground losses and SH0 color projection for a bound head."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class HeadTargets:
    rgb: torch.Tensor
    alpha: torch.Tensor
    crop: tuple
    photo_pixels: torch.Tensor
    ssim_centers: torch.Tensor
    window: torch.Tensor
    inside: torch.Tensor
    outside: torch.Tensor
    face: torch.Tensor | None
    photo_weights: torch.Tensor | None = None


@torch.no_grad()
def prepare_head_targets(reference, target_alpha, *, allowed=None, face=None, confidence_weighted=False):
    """Prepare the native head matte and its reliable photometric/shape regions."""
    alpha = target_alpha.squeeze(0) if target_alpha.ndim == 3 else target_alpha
    if (alpha.ndim != 2 or reference.shape != (3, *alpha.shape)
            or reference.device != alpha.device or not reference.is_floating_point()
            or not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any()):
        raise ValueError("Expected same-device native CHW RGB and normalized HW head matte")
    for mask in (allowed, face):
        if mask is not None and (mask.shape != alpha.shape or mask.device != alpha.device or mask.dtype != torch.bool):
            raise ValueError("Expected boolean native allowed and face masks")
    alpha = alpha.to(reference.dtype)
    foreground = alpha > (.5 if confidence_weighted else .99)
    ys, xs = foreground.nonzero(as_tuple=True)
    if len(ys) < 1000:
        raise ValueError("Fewer than 1000 reliable head foreground pixels")
    crop = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
    photo = foreground[crop]
    invalid = F.pad((alpha[crop] <= .99).float()[None, None], (5, 5, 5, 5), value=1)
    centers = F.max_pool2d(invalid, 11, stride=1)[0, 0] == 0
    if int(centers.sum()) < 1000:
        raise ValueError("Fewer than 1000 complete head SSIM windows")
    inside = F.max_pool2d((~foreground).float()[None, None], 5, stride=1, padding=2)[0, 0] == 0
    outside = F.max_pool2d((alpha >= .01).float()[None, None], 7, stride=1, padding=3)[0, 0] == 0
    if allowed is not None:
        outside &= ~allowed
    core = face[crop] & photo if face is not None else None
    if core is not None and not core.any():
        core = None
    x = torch.arange(-5, 6, device=reference.device, dtype=reference.dtype)
    gaussian = (-x.square() / (2 * 1.5 ** 2)).exp()
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    sy, sx = crop
    weights = (2 * alpha[crop][photo] - 1).clamp(0, 1) if confidence_weighted else None
    return HeadTargets(reference[:, sy, sx], alpha, crop, photo, centers, window, inside, outside, core, weights)


def head_loss(premultiplied_rgb, alpha, targets, *, constraint_ramp=1.0, photo_only=False):
    """Apply R15's native appearance and alpha losses to a black-background head render."""
    if not 0 <= constraint_ramp <= 1:
        raise ValueError("The R15 constraint ramp must lie in [0, 1]")
    if alpha.ndim == 3:
        alpha = alpha.squeeze(0)
    if alpha.shape != targets.alpha.shape or premultiplied_rgb.shape != (3, *alpha.shape):
        raise ValueError("Head renderer and native image support differ")
    sy, sx = targets.crop
    head_rgb, head_alpha = premultiplied_rgb[:, sy, sx], alpha[targets.crop]
    straight = head_rgb / head_alpha.clamp_min(1e-4)
    target = targets.rgb
    difference = (straight - target).abs()[:, targets.photo_pixels]
    weights = targets.photo_weights
    l1 = difference.mean() if weights is None else (difference.mean(0)*weights).sum()/weights.sum().clamp_min(1)
    a, b = (torch.where(targets.photo_pixels, image, 0.) for image in (straight, target))
    u, v, aa, bb, ab = F.conv2d(torch.stack((a, b, a.square(), b.square(), a * b)),
                               targets.window, padding=5, groups=3)
    aa, bb, ab = aa - u.square(), bb - v.square(), ab - u * v
    ssim = ((2*u*v + .01**2) * (2*ab + .03**2)) / ((u.square() + v.square() + .01**2) * (aa + bb + .03**2))
    dssim = 1 - ssim[:, targets.ssim_centers].mean()
    photo = .8*l1 + .2*dssim
    if photo_only:
        return photo, dict(photo=photo, foreground_l1=l1, foreground_dssim=dssim)

    warmup = .05 * (alpha - targets.alpha).abs().sum() / targets.alpha.sum().clamp_min(1)
    missing = (.98 - alpha[targets.inside]).relu()
    area = targets.inside.count_nonzero().clamp_min(1)
    balanced = (.40*(missing.square() + .20*missing).sum() + .12*alpha[targets.outside].square().sum()) / area
    face_rgb = alpha.new_zeros(())
    face_alpha = alpha.new_zeros(())
    if targets.face is not None:
        face = targets.face
        face_alpha_values = head_alpha[face]
        # Both black and white composites must agree on observed opaque skin.
        black_error = head_rgb[:, face] - target[:, face]
        face_rgb = .5 * (black_error.abs() + (black_error + 1 - face_alpha_values).abs()).mean()
        face_alpha = (.98 - face_alpha_values).relu().mean()
    shape = balanced + .08*face_rgb + .10*face_alpha
    alpha_term = (1 - constraint_ramp)*warmup + constraint_ramp*shape
    return photo + alpha_term, dict(photo=photo, foreground_l1=l1, foreground_dssim=dssim,
                                    alpha=alpha_term, balanced_alpha=balanced, face_rgb=face_rgb, face_alpha=face_alpha)


@torch.no_grad()
def project_dc_(dc, optimizer, *, base_dc=None):
    """Project SH0 into [0, 1] color and clear Adam momentum pushing out of range."""
    stored_c0 = float(torch.tensor(SH_C0, dtype=dc.dtype, device="cpu"))
    real_bound = .5 / max(SH_C0, stored_c0)
    bound_value = torch.tensor(real_bound, dtype=dc.dtype, device="cpu")
    if float(bound_value) > real_bound:
        bound_value = torch.nextafter(bound_value, torch.zeros_like(bound_value))
    bound = float(bound_value)
    if base_dc is None:
        dc.clamp_(-bound, bound)
        value = dc
    else:
        value = (base_dc + dc).clamp_(-bound, bound)
        dc.copy_(value - base_dc)
    moment = optimizer.state.get(dc, {}).get("exp_avg")
    if moment is not None:
        outward = ((value <= -bound) & (moment > 0)) | ((value >= bound) & (moment < 0))
        moment.masked_fill_(outward, 0.)

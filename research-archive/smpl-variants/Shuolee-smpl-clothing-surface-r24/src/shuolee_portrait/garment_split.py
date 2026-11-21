"""Trial tangent splits in the original training cameras before changing Adam rows."""

import math

import torch
from torch.nn import functional as F

from .bound_replay import RAW_PARAMETERS
from .drivable_clothing_bundle import CombinedClothingScene


LIMITS = dict(new_holes_fraction=5e-5, max_new_holes_in_tile=3,
              lost_foreground_alpha=4e-4, added_background_alpha=2e-4,
              added_rgb_mae=2e-4, max_tile_alpha_loss=.025)
TRIALS = ((1, .50), (1, .25), (.25, .25), (.0625, .25))


def child_logits(parent_alpha, shift_in_sigma):
    """Match the parent centre under a fully projected split axis, with a local cap."""
    remaining_variance = 1-shift_in_sigma**2
    projected_falloff = math.exp(-.5*shift_in_sigma**2/remaining_variance)
    same_centre_alpha = -torch.expm1(.5*torch.log1p(-parent_alpha))
    return torch.logit((same_centre_alpha/projected_falloff).clamp(max=.985), eps=1e-7)


def _tiles(image):
    return F.avg_pool2d(image[None, None], 32, stride=32, ceil_mode=True,
                        count_include_pad=True, divisor_override=1)[0, 0]


def _measure(before, after, target):
    old_rgb, old_alpha = before
    new_rgb, new_alpha = after
    _, target_rgb, target_alpha, box, *_ = target
    truth = target_alpha[0]
    foreground, background = truth >= .99, truth <= .01
    total = foreground.sum().clamp_min(1)
    old_a, new_a = old_alpha[0], new_alpha[0]
    holes = foreground & (old_a >= .5) & (new_a < .5)
    lost = (old_a-new_a).relu()*foreground
    spilled = (new_a-old_a).relu()*background
    x0, y0, x1, y1 = box
    ys, xs = slice(y0, y1), slice(x0, x1)
    target_crop = truth[ys, xs]
    premultiplied = target_rgb*target_crop
    old_error = (old_rgb-premultiplied).abs().mean(dim=0)
    new_error = (new_rgb-premultiplied).abs().mean(dim=0)
    tile_count = _tiles(foreground[ys, xs].float())
    tile_loss = _tiles(lost[ys, xs]) / tile_count.clamp_min(1)
    values = torch.stack((holes.sum()/total, _tiles(holes.float()).max(), lost.sum()/total,
                          spilled.sum()/total, ((new_error-old_error)*target_crop).sum()/total,
                          (tile_loss*(tile_count >= 16)).max()))
    metrics = dict(zip(LIMITS, values.tolist(), strict=True))
    return metrics, bool(torch.isfinite(values).all()) and all(metrics[key] <= value for key, value in LIMITS.items())


@torch.no_grad()
def split_with_coverage(prepared, targets, draw, threshold, *, budget, preview_only=False):
    """Render all training views and commit only a complete passing trial."""
    from utils.general_utils import build_rotation

    p, cloth = prepared, prepared.cloth
    previous = cloth.timestep
    score = (cloth.xyz_gradient_accum / cloth.denom.clamp_min(1)).flatten()
    eligible = torch.where(torch.isfinite(score) & (score >= threshold) &
                           (cloth.get_opacity.flatten() >= .02))[0]
    rotations = build_rotation(cloth._rotation[eligible])
    normals = cloth.garment_normals()[eligible]
    tangent = (rotations*normals[:, :, None]).sum(dim=1).abs() <= .5
    supported = tangent.any(dim=1)
    candidates = eligible[supported]
    order = score[candidates].argsort(descending=True, stable=True)[:budget]
    selected = candidates[order]
    outcome = dict(eligible=len(eligible), tangent_eligible=len(candidates), parents=0,
                   children=0, trials=[], train_cameras=len(targets))
    if not len(selected):
        cloth.reset_densification_stats()
        return outcome
    axes = rotations[supported][order]
    tangent = tangent[supported][order]

    def evaluate(scene, target, *, copy):
        slot, _, _, (x0, y0, x1, y1), *_ = target
        scene.select_mesh_by_timestep(slot)
        rgb, alpha = draw(p.cameras[slot], scene)
        crop = rgb[:, y0:y1, x0:x1].contiguous()
        return (crop.clone(), alpha.clone()) if copy else (crop, alpha)

    try:
        before = [evaluate(p.scene, target, copy=True) for target in targets]
        used = set()
        for fraction, shift in TRIALS:
            size = max(1, int(len(selected)*fraction))
            if (size, shift) in used:
                continue
            used.add((size, shift))
            parents = selected[:size]
            scales = cloth._scaling[parents].exp()
            axis = scales.masked_fill(~tangent[:size], -torch.inf).argmax(dim=1)
            row = torch.arange(size, device=axis.device)
            offsets = shift*scales[row, axis, None]*axes[row, :, axis]
            shrunk = cloth._scaling[parents].clone()
            shrunk[row, axis] += .5*math.log1p(-shift**2)
            seed_xyz = cloth._xyz[parents]
            children = {name: torch.cat((getattr(cloth, name)[parents], getattr(cloth, name)[parents]))
                        for name in RAW_PARAMETERS}
            children.update(_xyz=torch.cat((seed_xyz-offsets, seed_xyz+offsets)),
                            _scaling=shrunk.repeat(2, 1),
                            _opacity=child_logits(cloth.get_opacity[parents], shift).repeat(2, 1))
            remove = torch.zeros(len(cloth._xyz), dtype=torch.bool, device=axis.device)
            remove[parents] = True
            kept = torch.where(~remove)[0]
            inherited = parents.repeat(2)
            attributes = {name: torch.cat((getattr(cloth, name)[kept], children[name]))
                          for name in RAW_PARAMETERS}
            preview = cloth.preview_rows(torch.cat((kept, inherited)), attributes)
            scene = CombinedClothingScene(p.fixed, preview)
            trial = dict(proposed_parents=size, shift_in_sigma=shift, views=[])
            for old, target in zip(before, targets, strict=True):
                after = evaluate(scene, target, copy=False)
                metrics, passed = _measure(old, after, target)
                trial['views'].append(dict(frame=p.frames[target[0]], passed=passed, **metrics))
            trial['passed'] = all(view['passed'] for view in trial['views'])
            outcome['trials'].append(trial)
            if not trial['passed']:
                continue
            stable_ids = cloth.densification_ids[parents].cpu().tolist()
            first_child = cloth.next_densification_id
            if not preview_only:
                cloth.densification_postfix(children['_xyz'], children['_features_dc'], children['_features_rest'],
                    children['_opacity'], children['_scaling'], children['_rotation'], constraint_parents=inherited)
                extended = F.pad(remove, (0, 2*size), value=False)
                cloth.prune_points(extended)
                cloth._opacity.requires_grad_(False)
                preview.select_mesh_by_timestep(p.reference)
                cloth.select_mesh_by_timestep(p.reference)
                for field in ('get_xyz', 'get_scaling', 'get_rotation', 'get_opacity', 'get_features'):
                    if not torch.equal(getattr(cloth, field), getattr(preview, field)):
                        raise RuntimeError(f'Committed garment split differs from guarded preview in {field}.')
            outcome.update(parents=size, children=2*size, parent_ids=stable_ids,
                           first_child_id=first_child, shift_in_sigma=shift,
                           committed=not preview_only)
            break
    finally:
        p.scene.select_mesh_by_timestep(previous)
        cloth.reset_densification_stats()
    return outcome

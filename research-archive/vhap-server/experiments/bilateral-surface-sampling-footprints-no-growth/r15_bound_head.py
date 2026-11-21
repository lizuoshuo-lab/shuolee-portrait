"""R15-style SH0 training and guarded growth on a frozen, cached FLAME head."""

import math

import torch

from r15_head_loss import project_dc_

PARAMETERS = ("head_dc", "head_offset", "head_log_scale", "head_opacity")
ROW_BUFFERS = ("binding", "scale_base", "opacity_bias", "gradient_sum", "gradient_count", "max_radii")


class R15BoundHead(torch.nn.Module):
    def __init__(self, reference_frame, protected_triangles=None, max_points=None, *, max_parents=128):
        super().__init__()
        self.reference = reference_frame
        xyz, features, opacity, scales, _ = reference_frame.head
        count = len(xyz)
        self.max_points = 2 * count if max_points is None else max_points
        if not count or self.max_points < count or max_parents < 1:
            raise ValueError("A head needs one initial Gaussian per triangle and a valid growth budget")
        self.max_parents = max_parents
        protected = (torch.zeros(count, dtype=torch.bool, device=xyz.device) if protected_triangles is None
                     else torch.as_tensor(protected_triangles, device=xyz.device, dtype=torch.bool))
        if protected.shape != (count,):
            raise ValueError("Protected triangles must have one flag per cached FLAME triangle")
        self.extent = float((xyz.amax(0) - xyz.amin(0)).norm())
        self.offset_limit = 0.004
        self.fit_footprints = False
        if not math.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("The bound head has an invalid reference extent")
        self.register_buffer("protected", protected.clone(), persistent=False)
        self.register_buffer("initial_dc", features[:, :1].detach().clone(), persistent=False)
        self.register_buffer("binding", torch.arange(count, device=xyz.device))
        self.register_buffer("scale_base", torch.ones_like(scales))
        self.register_buffer("opacity_bias", torch.zeros_like(opacity))
        self.register_buffer("gradient_sum", xyz.new_zeros(count))
        self.register_buffer("gradient_count", xyz.new_zeros(count))
        self.register_buffer("max_radii", xyz.new_zeros(count))
        self.head_dc = torch.nn.Parameter(self.initial_dc.clone())
        self.head_offset = torch.nn.Parameter(torch.zeros_like(xyz))
        self.head_log_scale = torch.nn.Parameter(torch.zeros_like(scales))
        self.head_opacity = torch.nn.Parameter(torch.zeros_like(opacity))

    def _pose(self, frame, binding, values, scale_base, opacity_bias, geometry):
        dc, offset, log_scale, opacity_update = values
        xyz, features, opacity, scales, rotations = frame.head
        if not geometry:
            offset, log_scale, opacity_update = offset.detach(), log_scale.detach(), opacity_update.detach()
        local_offset = self.offset_limit * offset.tanh() * (~self.protected[binding, None])
        xyz = xyz[binding] + (frame.axes[binding] @ local_offset[..., None]).squeeze(-1)
        log_ratio = log_scale if self.fit_footprints else 0.18 * log_scale.tanh()
        logit_delta = opacity_update if self.fit_footprints else opacity_update.tanh()
        scales = scales[binding] * scale_base * log_ratio.exp()
        opacity = (opacity[binding].clamp(1e-6, 1 - 1e-6).logit() + opacity_bias + logit_delta).sigmoid()
        # The renderer expects the scene's SH layout, but the head has no higher orders.
        sh = torch.cat((dc, dc.new_zeros((len(binding), features.shape[1] - 1, 3))), dim=1)
        return xyz, sh, opacity, scales, rotations[binding]

    def head(self, frame, geometry=True):
        return self._pose(frame, self.binding, tuple(getattr(self, name) for name in PARAMETERS),
                          self.scale_base, self.opacity_bias, geometry)

    def prior(self, *, regularize_opacity=True):
        return (0.012 * self.head_offset[~self.protected[self.binding]].tanh().square().sum() / self.head_offset.numel()
                + 0.006 * self.head_log_scale.tanh().square().mean()
                + (0.006 * self.head_opacity.tanh().square().mean() if regularize_opacity else 0.)
                + 0.004 * (self.head_dc - self.initial_dc[self.binding]).tanh().square().mean())

    @torch.no_grad()
    def project_color(self, optimizer):
        project_dc_(self.head_dc, optimizer)

    @torch.no_grad()
    def add_stats(self, viewspace_grad, radii, foreground_fraction):
        if (viewspace_grad is None or viewspace_grad.ndim != 2 or viewspace_grad.shape[0] != len(self.binding)
                or viewspace_grad.shape[1] < 2 or radii.shape != self.binding.shape
                or not 0 < foreground_fraction <= 1):
            raise ValueError("Head gradient, radii and native foreground fraction do not match")
        visible = radii > 0
        score = viewspace_grad[visible, :2].norm(dim=-1) * foreground_fraction
        if not bool(torch.isfinite(score).all()):
            raise RuntimeError("Non-finite gradient in visible head Gaussians")
        self.gradient_sum[visible] += score
        self.gradient_count[visible] += 1
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], radii[visible].to(self.max_radii))

    def _replace_parameters(self, values, optimizer, rows=None, new_rows=None, edited_opacity_rows=None):
        replacements = []
        for name, value in zip(PARAMETERS, values, strict=True):
            old = getattr(self, name)
            found = [(group, index) for group in optimizer.param_groups for index, p in enumerate(group["params"]) if p is old]
            if len(found) != 1:
                raise ValueError(f"The head optimizer must own exactly one {name} parameter")
            group, index = found[0]
            state = {}
            for key, saved in optimizer.state.get(old, {}).items():
                if isinstance(saved, torch.Tensor) and saved.shape == old.shape:
                    if rows is None:
                        continue
                    saved = saved[rows].clone()
                    edited_rows = edited_opacity_rows if name == "head_opacity" else new_rows
                    saved[edited_rows] = 0
                state[key] = saved
            replacements.append((name, old, torch.nn.Parameter(value), group, index, state))
        for name, old, new, group, index, state in replacements:
            optimizer.state.pop(old, None)
            group["params"][index] = new
            if state:
                optimizer.state[new] = state
            setattr(self, name, new)

    def _check_bindings(self, binding):
        count = len(self.protected)
        if (binding.ndim != 1 or binding.dtype != torch.long or not count <= len(binding) <= self.max_points
                or bool(((binding < 0) | (binding >= count)).any())):
            raise ValueError("Invalid head triangle binding table")
        per_triangle = torch.bincount(binding, minlength=count)
        if bool((per_triangle == 0).any()) or bool((per_triangle[self.protected] != 1).any()):
            raise ValueError("Head growth removed a triangle or changed a protected triangle's topology")

    @torch.no_grad()
    def prepare_restore(self, state, optimizer):
        """Resize trainable rows before standard module and optimizer state loading."""
        binding = state["binding"].to(self.binding.device)
        self._check_bindings(binding)
        for name in (*PARAMETERS, *ROW_BUFFERS):
            target, src = getattr(self, name), state[name]
            if src.shape != (len(binding), *target.shape[1:]) or src.dtype != target.dtype or not bool(torch.isfinite(src).all()):
                raise ValueError(f"Invalid saved head rows: {name}")
        if bool((state["scale_base"] <= 0).any()):
            raise ValueError("A saved head sigma multiplier is not positive")
        self._replace_parameters(tuple(state[name].to(getattr(self, name)).clone() for name in PARAMETERS), optimizer)
        for name in ROW_BUFFERS:
            setattr(self, name, state[name].to(getattr(self, name)).clone())

    @torch.no_grad()
    def densify(self, step, optimizer, guard):
        """Propose stock-timed growth; accept only after a caller's native coverage regression check.

        The guard receives two functions that render head tensors for an arbitrary cached frame, and event metadata.
        No topology is installed until it accepts both the lost support and exterior-spill risks.
        """
        if not 300 < step <= 7800 or step % 60:
            return None
        event = {"step": step, "before": len(self.binding), "cloned": 0, "split": 0, "accepted": False}
        score = self.gradient_sum / self.gradient_count.clamp_min(1)
        eligible = (score >= .0002) & (~self.protected[self.binding])
        ranked = torch.argsort(torch.where(eligible, score, -math.inf), descending=True, stable=True)
        parents = ranked[:min(self.max_parents, self.max_points - len(self.binding))]
        parents = parents[eligible[parents]]
        event["max_radii_px"] = float(self.max_radii.max())
        event["eligible_above_20px"] = int((eligible & (self.max_radii > 20)).sum())
        self.gradient_sum.zero_()
        self.gradient_count.zero_()
        self.max_radii.zero_()
        if not len(parents):
            return event

        reference = self.reference
        base_binding = self.binding[parents]
        _, _, alpha, sigma, _ = self.head(reference)
        split = sigma[parents].amax(dim=-1) > 0.01 * self.extent
        clones, splits = parents[~split], parents[split]
        split_binding = base_binding[split]
        local_sigma = sigma[splits] / reference.axes[split_binding].norm(dim=1).clamp_min(1e-8)
        tangent = 2 * local_sigma[:, [0, 2]].argmax(dim=1)
        split_offsets = self.head_offset[splits].tanh() * self.offset_limit
        row = torch.arange(len(splits), device=splits.device)
        major, center = local_sigma[row, tangent], split_offsets[row, tangent]
        separation = torch.minimum(0.6 * major, 0.5 * (self.offset_limit - center.abs()))
        valid = separation > 0.02 * major
        splits, split_offsets, tangent, separation = (item[valid] for item in (splits, split_offsets, tangent, separation))
        event.update(cloned=len(clones), split=len(splits))
        if not len(clones) and not len(splits):
            return event

        keep = torch.ones_like(self.binding, dtype=torch.bool)
        keep[splits] = False
        retained = keep.nonzero().flatten()
        appended = torch.cat((clones, splits, splits))
        sources = torch.cat((retained, appended))
        binding = self.binding[sources]
        self._check_bindings(binding)
        dc, offset, log_scale, opacity = (getattr(self, name)[sources].clone() for name in PARAMETERS)
        scale_base, opacity_bias = self.scale_base[sources].clone(), self.opacity_bias[sources].clone()
        new_rows = torch.arange(len(sources), device=sources.device) >= len(retained)
        opacity_rows = new_rows.clone()
        opacity_rows[:len(retained)] = torch.isin(retained, clones)
        # Share point transmittance between the two descendants rather than doubling peak alpha.
        half = 1 - (1 - alpha[sources[opacity_rows]]).sqrt()
        native = reference.head[2][binding[opacity_rows]].clamp(1e-6, 1 - 1e-6).logit()
        opacity_bias[opacity_rows] = half.clamp(1e-6, 1 - 1e-6).logit() - native
        opacity[opacity_rows] = 0
        if len(splits):
            delta = torch.zeros_like(split_offsets)
            delta.scatter_(1, tangent[:, None], separation[:, None])
            children = torch.cat((split_offsets - delta, split_offsets + delta)) / self.offset_limit
            offset[-2 * len(splits):] = children.clamp(-.99999, .99999).atanh()
            scale_base[-2 * len(splits):] /= 1.6
        values = (dc, offset, log_scale, opacity)

        def proposed(frame):
            return self._pose(frame, binding, values, scale_base, opacity_bias, True)

        event["proposed"] = len(binding)
        if not guard(self.head, proposed, event):
            return event
        self._replace_parameters(values, optimizer, rows=sources, new_rows=new_rows, edited_opacity_rows=opacity_rows)
        self.binding, self.scale_base, self.opacity_bias = binding, scale_base, opacity_bias
        self.gradient_sum = self.gradient_sum.new_zeros(len(binding))
        self.gradient_count = self.gradient_count.new_zeros(len(binding))
        self.max_radii = self.max_radii.new_zeros(len(binding))
        self.project_color(optimizer)
        event["accepted"] = True
        return event

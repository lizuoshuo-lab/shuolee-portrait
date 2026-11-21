"""Bounded synthetic CUDA check for the portable SH0 foreground/alpha objective."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checks(worktree, repeats):
    import torch

    sys.path.insert(0, str(worktree / "src"))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import camera, render_pose, _native_foreground_target, _sh0_foreground_alpha_loss
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import l1_loss, ssim
    from utils.sh_utils import RGB2SH

    torch.backends.cuda.matmul.allow_tf32 = False
    require(torch.cuda.is_available(), "CUDA is unavailable in the configured runtime.")
    device = torch.device("cuda:0")
    black, white = (torch.full((3,), value, device=device) for value in (0., 1.))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    def make_camera(w, h):
        cx, cy = .48 * w, .51 * h
        focal = .85 * h
        cam, record = camera(dict(frame="synthetic", wh=[w, h], w2c=np.eye(4).tolist(),
                                  K=[[focal, 0, cx], [0, focal, cy], [0, 0, 1]]), 0)
        return cam, record, cx - .5, cy - .5

    def make_gaussians(xyz, color, scale, opacity):
        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float32, device=device)

        model = GaussianModel(0)
        n = len(xyz)
        model._xyz = torch.nn.Parameter(tensor(xyz).clone())
        model._features_dc = torch.nn.Parameter(RGB2SH(tensor(color).expand(n, 3)).unsqueeze(1).contiguous())
        model._features_rest = torch.nn.Parameter(torch.empty((n, 0, 3), device=device))
        model._scaling = torch.nn.Parameter(tensor(scale).expand(n, 3).log().contiguous())
        model._opacity = torch.nn.Parameter(tensor(opacity).reshape(-1, 1).expand(n, 1).logit().contiguous())
        model._rotation = torch.nn.Parameter(tensor([1, 0, 0, 0]).expand(n, 4).contiguous())
        return model

    def paired(cam, model):
        rgb = render_pose(render, cam, model, pipe, black, 0)
        override = torch.ones_like(model._xyz)
        alpha = render(cam, model, pipe, black, override_color=override)["render"]
        return rgb, alpha

    def fixture(folder, tag, w, h, x, y, radius, *, colored):
        yy, xx = np.ogrid[:h, :w]
        distance = np.hypot(xx - x, yy - y)
        mask = np.rint(255 * np.clip((radius + 1.75 - distance) / 3.5, 0, 1)).astype(np.uint8)
        rgb = np.zeros((h, w, 3), np.uint8)
        if colored:
            rgb[..., 0] = 55 + xx % 139
            rgb[..., 1] = 40 + yy % 157
            rgb[..., 2] = 75 + (xx + yy) % 120
        image_path, mask_path = folder / (tag + "-rgb.png"), folder / (tag + "-alpha.png")
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(mask).save(mask_path)
        pixels, target_alpha, box = _native_foreground_target(dict(frame=tag, wh=[w, h], image=image_path, mask=mask_path))
        require(np.array_equal(np.rint(target_alpha.numpy()[0] * 255).astype(np.uint8), mask), "Native target loader altered soft mask values.")
        return pixels.to(device), target_alpha.to(device), box

    w, h = 192, 160
    cam, camera_record, cx, cy = make_camera(w, h)
    opaque = make_gaussians([[0, 0, 2]], [0, 0, 0], [.25, .21, .12], .97)
    transparent = make_gaussians([[0, 0, 2]], [0, 0, 0], [.25, .21, .12], 1e-7)
    colored = make_gaussians([[-.09, -.03, 1.9], [.11, .05, 2.1], [.03, -.17, 2.]],
        [[.12, .65, .38], [.71, .26, .19], [.24, .18, .73]], [.21, .16, .11], [.73, .81, .68])

    with TemporaryDirectory(prefix="sh0-alpha-fixture-") as temporary:
        folder = Path(temporary)
        fg = fixture(folder, "black-foreground", w, h, cx, cy, 32, colored=False)
        bg = fixture(folder, "black-background", w, h, 26, cy, 28, colored=False)
        photo_target = fixture(folder, "color", w, h, cx + 7, cy - 3, 32, colored=True)
        with torch.no_grad():
            o, oa = paired(cam, opaque)
            t, ta = paired(cam, transparent)
            black_rgb_error = float((o["render"] - t["render"]).abs().max())
            require(black_rgb_error < 1e-6 and float(oa.max()) > .9 and float(ta.max()) < 1e-4,
                    "Black opaque/transparent control lacks identical RGB and distinct native alpha.")
            opaque_fg = _sh0_foreground_alpha_loss(o["render"], oa[:1], *fg)
            transparent_fg = _sh0_foreground_alpha_loss(t["render"], ta[:1], *fg)
            opaque_bg = _sh0_foreground_alpha_loss(o["render"], oa[:1], *bg)
            transparent_bg = _sh0_foreground_alpha_loss(t["render"], ta[:1], *bg)
            require(all(float(parts[0]) < 1e-6 for parts in (opaque_fg, transparent_fg, opaque_bg, transparent_bg))
                    and float(opaque_fg[3]) < float(transparent_fg[3])
                    and float(opaque_bg[3]) > float(transparent_bg[3]),
                    "Opacity is not penalized independently on both classes.")

            pkg, alpha_rgb = paired(cam, colored)
            alpha = alpha_rgb[:1]
            white_composite = render(cam, colored, pipe, white)["render"]
            alpha_channel_error = float((alpha_rgb - alpha).abs().max())
            composite_error = float((white_composite - pkg["render"] - (1 - alpha)).abs().max())
            require(alpha_channel_error < 1e-6 and composite_error < 5e-5, "White-pass alpha does not match black/white compositing.")
            pixels, mask, box = photo_target
            x0, y0, x1, y1 = box
            valid = mask[:, y0:y1, x0:x1] > .99
            changed = torch.where(valid, pixels, 1 - pixels)
            valid_change = pixels * .65
            original_terms = _sh0_foreground_alpha_loss(pkg["render"], alpha, *photo_target)
            background_terms = _sh0_foreground_alpha_loss(pkg["render"], alpha, changed, mask, box)
            altered_terms = _sh0_foreground_alpha_loss(pkg["render"], alpha, valid_change, mask, box)
            effects = [float((a - b).abs()) for a, b in zip(original_terms, background_terms, strict=True)]
            require(max(effects) < 1e-7 and float((original_terms[2] - altered_terms[2]).abs()) > 1e-4,
                    "SSIM/RGB includes background or cannot distinguish a foreground change.")

            edge = (fg[1] > 0) & (fg[1] < 1)
            rgb = torch.zeros_like(o["render"])
            edge_target = fg[1]
            exact = _sh0_foreground_alpha_loss(rgb, edge_target, *fg)[3]
            hardened = _sh0_foreground_alpha_loss(rgb, (edge_target >= .5).to(edge_target.dtype), *fg)[3]
            require(bool(edge.any()) and float(exact) == 0 and float(hardened) > 0, "Soft target edge is discarded.")

        params = {name: getattr(colored, name) for name in ("_xyz", "_features_dc", "_scaling", "_opacity", "_rotation")}
        frozen = {name: param.detach().clone() for name, param in params.items()}
        pkg, alpha_rgb = paired(cam, colored)
        photo, _, _, alpha_loss = _sh0_foreground_alpha_loss(pkg["render"], alpha_rgb[:1], *photo_target)
        alpha_parameters = (params["_opacity"], params["_xyz"], params["_scaling"], params["_features_dc"])
        alpha_grads = torch.autograd.grad(alpha_loss, alpha_parameters, retain_graph=True, allow_unused=True)
        require(alpha_grads[-1] is None and all(grad is not None and torch.isfinite(grad).all()
                and float(grad.norm()) > 0 for grad in alpha_grads[:-1]), "Alpha loss is coupled to SH0 or has missing geometry/opacity gradient.")
        (photo + .1 * alpha_loss).backward()
        grads = {}
        for name, param in params.items():
            require(param.grad is not None and torch.isfinite(param.grad).all() and torch.equal(frozen[name], param),
                    f"Missing or nonfinite gradient, or changed parameters: {name}")
            grads[name] = float(param.grad.norm())
        require(all(grads[name] > 0 for name in ("_xyz", "_features_dc", "_opacity", "_scaling")), "The full objective lost an essential gradient.")
        require(pkg["viewspace_points"].grad is not None and torch.isfinite(pkg["viewspace_points"].grad).all(),
                "The original RGB densification gradient disappeared.")

        tw, th = 2704, 1520
        timing_cam, _, tx, ty = make_camera(tw, th)
        gx, gy = np.meshgrid(np.linspace(-.45, .45, 36), np.linspace(-.75, .75, 48))
        xyz = np.stack((gx, gy, 2 + .05 * np.sin(5 * gx)), -1).reshape(-1, 3)
        color = np.stack((.55 + .25 * np.sin(8 * gx), .5 + .3 * np.cos(5 * gy), .5 + .3 * np.sin(6 * gy)), -1).reshape(-1, 3)
        timed = make_gaussians(xyz, color, [.026, .033, .02], .45)
        timed_target = fixture(folder, "timing", tw, th, tx, ty, 380, colored=True)
        target_rgb, target_mask, crop = timed_target
        a, b, c, d = crop
        # The baseline objective sees the same known analytic synthetic RGB and alpha, composited on black.
        with Image.open(folder / "timing-rgb.png") as source:
            full = torch.from_numpy(np.array(source)).permute(2, 0, 1).float().div(255).to(device)
        baseline_target = full * target_mask
        del full
        require(torch.equal(baseline_target[:, b:d, a:c], target_rgb * target_mask[:, b:d, a:c]), "Timing arms use different synthetic targets.")
        fields = (timed._xyz, timed._features_dc, timed._scaling, timed._rotation, timed._opacity)

        def profile(foreground):
            if foreground:
                color_pkg, a = paired(timing_cam, timed)
                rgb_loss, _, _, silhouette = _sh0_foreground_alpha_loss(color_pkg["render"], a[:1], *timed_target)
                return rgb_loss + .1 * silhouette
            prediction = render_pose(render, timing_cam, timed, pipe, black, 0)["render"]
            return .8 * l1_loss(prediction, baseline_target) + .2 * (1 - ssim(prediction, baseline_target))

        def timed_backward(foreground, measure):
            for param in fields:
                param.grad = None
            torch.cuda.synchronize()
            event_start, event_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.reset_peak_memory_stats()
            allocation = torch.cuda.memory_allocated()
            start = perf_counter()
            event_start.record()
            loss = profile(foreground)
            loss.backward()
            event_end.record()
            event_end.synchronize()
            seconds = perf_counter() - start
            require(torch.isfinite(loss), "A synthetic timing pass produced a nonfinite loss.")
            if measure:
                return dict(cuda_ms=event_start.elapsed_time(event_end), wall_ms=1000 * seconds,
                            transient_peak_bytes=torch.cuda.max_memory_allocated() - allocation)

        for arm in (False, True):
            timed_backward(arm, False)
        timings = dict(baseline=[], foreground_alpha=[])
        for _ in range(repeats):
            for arm, name in ((False, "baseline"), (True, "foreground_alpha")):
                timings[name].append(timed_backward(arm, True))
        timing_medians = {arm: {name: median(x[name] for x in rows) for name in rows[0]} for arm, rows in timings.items()}

    return dict(state="synthetic_cuda_passed", optimizer_updates=0, source_images="synthetic", renderer=render.__module__,
        training_source_sha256=rt.sha(worktree / "src/shuolee_portrait/training.py"),
        script_sha256=rt.sha(Path(__file__)), camera=camera_record, gpu=torch.cuda.get_device_name(device),
        black_opaque_vs_transparent=dict(rgb_max_diff=black_rgb_error, opaque_peak_alpha=float(oa.max()),
            transparent_peak_alpha=float(ta.max()), opaque_fg_loss=float(opaque_fg[3]), transparent_fg_loss=float(transparent_fg[3]),
            opaque_bg_loss=float(opaque_bg[3]), transparent_bg_loss=float(transparent_bg[3])),
        white_background_parity_max=composite_error, alpha_channel_max=alpha_channel_error,
        outside_rgb_effects=effects, foreground_ssim_change=float((original_terms[2] - altered_terms[2]).abs()),
        soft_edge_pixels=int(edge.sum()), soft_edge_hardening_loss=float(hardened), combined_gradient_norms=grads,
        alpha_gradient_norms=dict(zip(("opacity", "xyz", "scaling"), (float(g.norm()) for g in alpha_grads[:-1]), strict=True)),
        alpha_color_gradient_none=True, timing=dict(raster=[tw, th], synthetic_points=len(xyz), repeats=repeats,
            forward_backward_medians=timing_medians, samples=timings, representative_subject_latency=False),
        quality_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.run_cuda:
        parser.error("The bounded CUDA run requires --run-cuda.")
    import fcntl

    if args.out is not None:
        require(not args.out.exists() and args.out.parent.is_dir(), "Choose a new output in an existing diagnostic directory.")
    worktree = args.worktree.resolve(strict=True)
    with Path("/tmp/shuolee-native122-gpu.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = checks(worktree, args.repeats)
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(data)
    print(data, end="")


if __name__ == "__main__":
    main()

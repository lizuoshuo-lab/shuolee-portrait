"""Use mini-splatting2's scene optimizer with a depth-sorted VHAP layer."""

from argparse import Namespace
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from fuse_scene import make_rasterizer as make_calibrated_rasterizer


class Rasterizer:
    def __init__(self, K, size, R, C):
        from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer

        self.calibration = K, size, R, C
        settings = make_calibrated_rasterizer(K, size, R, C).raster_settings
        self.native = GaussianRasterizer(GaussianRasterizationSettings(**settings._asdict()))

    def half(self):
        K, (width, height), R, C = self.calibration
        w, h = round(width / 2), round(height / 2)
        sx, sy = w / width, h / height
        pixel_map = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        return Rasterizer(pixel_map @ K, (w, h), R, C)

    @staticmethod
    def _arguments(means3D, shs=None, culling=None, **kwargs):
        if culling is None:
            culling = means3D.new_zeros(len(means3D), dtype=torch.bool)
        dc = shs[:, :1].contiguous() if shs is not None else None
        rest = shs[:, 1:].contiguous() if shs is not None else None
        return dict(means3D=means3D, dc=dc, shs=rest, culling=culling.contiguous(), **kwargs)

    def __call__(self, **kwargs):
        return self.native(**self._arguments(**kwargs))[:2]

    def training(self, **kwargs):
        return self.native(**self._arguments(**kwargs), flag_max_count=True)

    def importance(self, **kwargs):
        return self.native.render_simp(**self._arguments(**kwargs))

    def depth(self, **kwargs):
        return self.native.render_depth(**self._arguments(**kwargs))


def make_rasterizer(K, size, R, C):
    return Rasterizer(K, size, R, C)


def load_scene_model(vendor, seed_ply, extent, opt, view_count):
    """Load the pinned MSv2 class without shadowing GaussianAvatars' scene package."""
    model_file = Path(vendor) / "scene" / "gaussian_model.py"
    spec = importlib.util.spec_from_file_location("msv2_gaussian_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GaussianModel(3)
    model.load_ply(str(seed_ply))
    model.active_sh_degree = 0
    model.spatial_lr_scale = extent
    model.training_setup(opt)
    model.init_culling(view_count)
    model.max_radii2D = model.get_xyz.new_zeros(len(model.get_xyz))
    return model


def scene_tensors(model):
    features = model.get_features
    count = (model.active_sh_degree + 1) ** 2
    if count < features.shape[1]:
        features = torch.cat((features[:, :count], torch.zeros_like(features[:, count:])), dim=1)
    return model.get_xyz, features, model.get_opacity, model.get_scaling, model.get_rotation


class SceneController:
    """Keep native topology operations on the scene slots before the head slots."""

    def __init__(self, model, frames, extent, actor_tensors, opt):
        self.model, self.actor_tensors, self.opt = model, actor_tensors, opt
        self.extent = extent
        self.ids = {frame.name: i for i, frame in enumerate(frames)}
        self.views = [SimpleNamespace(frame=frame, uid=i, original_image=frame.reference)
                      for i, frame in enumerate(frames)]
        self.half_views = {}
        self.native_args = Namespace(warn_until_iter=3000, imp_metric="indoor", sampling_factor=0.6)
        self.blur = model.get_xyz.new_zeros(len(model.get_xyz), dtype=torch.bool)
        self.events = []

    def frame_for_step(self, frame, step):
        if step > self.native_args.warn_until_iter:
            return frame
        if frame.name not in self.half_views:
            camera = frame.rasterizer.half()
            _, (width, height), _, _ = camera.calibration
            refs = F.interpolate(frame.reference[None], (height, width), mode="bicubic",
                                 align_corners=False, antialias=True)[0].clamp_(0, 1)
            masks = {name: F.interpolate(mask[None, None].float(), (height, width),
                                         mode="nearest-exact")[0, 0].bool()
                     for name, mask in frame.masks.items()}
            y, x = torch.where(masks["head_allowed"])
            crop = (slice(max(0, int(y.min()) - 4), min(height, int(y.max()) + 5)),
                    slice(max(0, int(x.min()) - 4), min(width, int(x.max()) + 5)))
            self.half_views[frame.name] = replace(frame, rasterizer=camera, reference=refs,
                                                   masks=masks, crop=crop)
        return self.half_views[frame.name]

    def getTrainCameras_warn_up(self, step, *_args, **_kwargs):
        views = []
        for view in self.views:
            frame = self.frame_for_step(view.frame, step)
            views.append(SimpleNamespace(frame=frame, uid=view.uid, original_image=frame.reference))
        return views

    def culling(self, frame, head_count):
        scene = self.model._culling[:, self.ids[frame.name]]
        return torch.cat((scene, scene.new_zeros(head_count)))

    def _joint_arguments(self, view, model, culling):
        scene = scene_tensors(model)
        head = self.actor_tensors(view.frame)
        xyz, shs, opacity, scales, rotations = (torch.cat((a, b)) for a, b in zip(scene, head))
        return dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=shs, opacities=opacity,
                    scales=scales, rotations=rotations,
                    culling=torch.cat((culling, culling.new_zeros(len(head[0])))))

    def _importance(self, view, model, _pipe, _background, culling):
        _, radii, weights, area, largest = view.frame.rasterizer.importance(
            **self._joint_arguments(view, model, culling))
        n = len(model.get_xyz)
        return dict(radii=radii[:n], accum_weights=weights[:n], area_proj=area[:n], area_max=largest[:n])

    def _depth(self, view, model, _pipe, _background, culling):
        raw = view.frame.rasterizer.depth(**self._joint_arguments(view, model, culling))
        transmittance = raw["accum_alpha"]
        owner = raw["gidx"].reshape_as(transmittance)
        # The upstream function samples proportional to 1 - transmittance. Exclude
        # head support and depth owned by head slots from reseeding the scene layer.
        exclude = (view.frame.masks["head_allowed"].reshape_as(transmittance)
                   | (owner < 0) | (owner >= len(model.get_xyz))
                   | (raw["rendered_depth"].reshape_as(transmittance) <= 0))
        raw["accum_alpha"] = transmittance.masked_fill(exclude, 1.0)
        return raw

    @torch.no_grad()
    def update(self, step, frame, projected_scene, radii_scene, area_scene):
        model = self.model
        before = len(model.get_xyz)
        visible = radii_scene > 0
        if projected_scene.grad is None or radii_scene.shape != (before,) or area_scene.numel() != before:
            raise ValueError("MSv2 densification requires matching scene slots and projected gradients")
        if step < 3000:
            visible_ids = visible.nonzero()
            model.max_radii2D[visible_ids] = torch.maximum(model.max_radii2D[visible_ids], radii_scene[visible_ids])
            if model._culling[:, self.ids[frame.name]].any():
                model.add_densification_stats_culling(projected_scene, visible_ids, model.factor_culling)
            else:
                model.add_densification_stats(projected_scene, visible_ids)
            height, width = frame.reference.shape[-2:]
            self.blur |= area_scene.reshape(-1) > height * width / 5000
        model.optimizer.step(visible, before)
        model.optimizer.zero_grad(set_to_none=True)

        stages = []
        if step == 2000:
            model.interesction_preserving(self, self._importance, step, self.native_args, None, None)
            xyz, rgb = model.depth_reinit(self, self._depth, step, len(model.get_xyz), self.native_args, None, None)
            if not len(xyz) or not torch.isfinite(xyz).all() or not torch.isfinite(rgb).all():
                raise RuntimeError("Native MSv2 depth reinitialization produced invalid scene samples")
            model.reinitial_pts(xyz, rgb)
            model.training_setup(self.opt)
            model.init_culling(len(self.views))
            stages.append("scene_depth_reinitialization")
        elif step == 3000:
            model.culling_with_interesction_sampling(self, self._importance, step, self.native_args, None, None)
            model.training_setup(self.opt)
            self.half_views.clear()
            stages.append("importance_sampling")
        elif step == 8000:
            model.culling_with_interesction_preserving(self, self._importance, step, self.native_args, None, None)
            stages.append("importance_preserving")
        elif 500 <= step < 3000:
            if step > 500 and step % self.opt.densification_interval == 0:
                model.densify_and_prune_mask(self.opt.densify_grad_threshold, 0.005,
                                             self.extent, None, self.blur)
                stages.append("densify_and_prune")
            if step % 250 == 0:
                model.culling_with_clone(self, self._importance, step, self.native_args, None, None)
                stages.append("importance_clone")
        elif step == (8000 + self.opt.iterations) // 2 and step > 8000:
            model.init_culling(len(self.views))
            stages.append("clear_view_culling")
        if stages:
            if len(model.get_xyz) < 2:
                raise RuntimeError(f"MSv2 {stages} discarded the scene")
            self.blur = model.get_xyz.new_zeros(len(model.get_xyz), dtype=torch.bool)
            event = dict(step=step, stages=stages, before=before, after=len(model.get_xyz))
            self.events.append(event)
            print(f"MSV2_STAGE {event}", flush=True)

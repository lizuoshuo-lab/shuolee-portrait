"""Scheduled shape and coverage terms for the stock SH0 training loop."""

import hashlib
from pathlib import Path

import numpy as np

from . import runtime as rt
from .head_appearance import face_composite_loss
from .shape_refinement import (_balanced_alpha, _footprints, _prepare_views, _projection,
                               _reseed, _screen_visibility, _shape_loss, shape_config)


def main_schedule():
    from .world_scale import scale_config

    settings = shape_config()
    return dict(warmup_steps=2400, constraint_ramp_steps=600, stock_densify_through=7800,
                coverage_steps=[4800, 8400], face_rgb_weight=.08, face_alpha_weight=.10,
                world_scale=scale_config(),
                constraints={key: value for key, value in settings.items() if key not in {
                    'steps', 'reseed_step', 'reset_steps', 'face_rgb_anchor', 'face_alpha_anchor',
                    'position_lr', 'position_final_lr', 'scaling_lr', 'rotation_lr', 'feature_lr', 'opacity_lr'}})


def frozen_surface(root, spec, fit, layout, clothing):
    if spec is None:
        return None, None
    if not isinstance(spec, dict) or set(spec) != {'file', 'sha256'}:
        raise ValueError('Expected a mesh-only file and its SHA-256.')
    root = Path(root).resolve()
    relative = Path(spec['file'])
    source = (root/relative).resolve(strict=True)
    if relative.is_absolute() or not source.is_relative_to(root/'fit') or rt.sha(source) != spec['sha256']:
        raise ValueError('Frozen surface is outside the fitted inputs or its hash differs.')
    with np.load(source, allow_pickle=False) as mesh:
        if (set(mesh.files) != {'vertices_world', 'faces', 'reference_frame_id', 'reference_pose_index', 'static_offset'}
                or mesh['reference_frame_id'].item() != layout['reference_frame_id']
                or int(mesh['reference_pose_index']) != layout['reference_pose_index']
                or not np.array_equal(mesh['faces'], fit['faces'])):
            raise ValueError('Frozen surface does not match the fitted topology and reference pose.')
        delta = mesh['static_offset'].copy()
        if (delta.shape != (len(clothing), 3) or delta.dtype != np.float32
                or not np.isfinite(delta).all() or np.any(delta[~clothing])):
            raise ValueError('Frozen surface has an invalid shared clothing offset.')
    return source, delta


def initialization(model):
    fields = ('_xyz', '_scaling', '_rotation', '_features_dc', '_features_rest', '_opacity', 'binding')
    hashes = {name: hashlib.sha256(getattr(model, name).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
              for name in fields}
    if (len(model.binding) != len(model.faces) or not (model.binding_counter == 1).all()
            or model.optimizer.state or model._xyz.count_nonzero()):
        raise RuntimeError('SH0 initialization must be one fresh local center per triangle with an empty optimizer.')
    return dict(policy='one_per_triangle', initial_points=len(model.binding), local_parameter_sha256=hashes,
                optimizer_initial_state_entries=0, optimizer_updates=0, no_prior_gaussian_loaded=True)


def _protected_landmark_pixels(view, landmarks):
    """Native reliable foreground around observed eyes and ears in the OpenPose layout."""
    import torch

    confidence = np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] >= .5)
    inner = landmarks[67:118, :2][confidence[67:118]]
    if len(inner) < 8:
        return {}
    # Ear neighborhoods are an observation-based probe, without assuming that
    # the detector has established physical visibility through hair.
    ear_radius = max(3, round(.12*float(np.ptp(inner, axis=0).max())))
    x0, y0, x1, y1 = view.box
    regions = {}
    for name, indices in (('right_eye', range(86, 92)), ('left_eye', range(92, 98)),
                          ('right_ear', [17]), ('left_ear', [18])):
        indices = np.asarray(indices)
        supported = landmarks[indices, :2][confidence[indices]]
        ear = len(indices) == 1
        if len(supported) < (1 if ear else 3):
            continue
        pad = ear_radius if ear else max(2, round(.25*float(np.ptp(supported, axis=0).max())))
        left, top = np.maximum([x0, y0], np.floor(supported.min(axis=0)).astype(int)-pad)
        right, bottom = np.minimum([x1, y1], np.ceil(supported.max(axis=0)).astype(int)+pad+1)
        if left >= right or top >= bottom:
            continue
        pixels = view.inside[top:bottom, left:right].nonzero().T
        if pixels.shape[1] >= 8:
            regions[name] = pixels + torch.tensor([[top], [left]], device=pixels.device)
    return regions


class Constraints:
    def __init__(self, model, rows, slots, cameras, fit, pipe, black, targets, bins, extent, schedule):
        self.model, self.rows, self.slots, self.cameras = model, rows, slots, cameras
        self.pipe, self.black, self.bins, self.extent = pipe, black, bins, extent
        self.schedule = schedule
        self.options = schedule['constraints']
        self.views, self.matrices = _prepare_views(model, rows, slots, cameras, fit, pipe, black, self.options,
                                                  native_targets=targets, capture_face_anchor=False)
        self.protected_regions = [_protected_landmark_pixels(view, fit['observed_native_keypoints'][slot])
                                  for view, slot in zip(self.views, slots, strict=True)]

    def loss(self, i, rgb, alpha, visible, old_alpha, step):
        view, matrix = self.views[i], self.matrices[i]
        xy, depth = _projection(self.model.get_xyz, *matrix)
        supported, x, y = _screen_visibility(xy, depth, view, self.options)
        exterior = supported & view.outside[y, x]
        alpha_loss = _balanced_alpha(alpha[0], view.inside, view.outside, self.options)
        shape = _shape_loss(self.model, visible, view, exterior, self.extent, self.options)
        footprint = _footprints(self.model, visible, view, *matrix, self.options)
        total = alpha_loss + shape + self.options['footprint_weight']*footprint
        if view.face is not None:
            x0, y0, _, _ = view.box
            interior = alpha[0, view.face[0]+y0, view.face[1]+x0]
            total += self.schedule['face_rgb_weight']*face_composite_loss(rgb, alpha, view.rgb, view.box, view.face)
            total += self.schedule['face_alpha_weight']*(self.options['foreground_target']-interior).relu().mean()
        ramp = min(1., (step-self.schedule['warmup_steps'])/self.schedule['constraint_ramp_steps'])
        return (1-ramp)*old_alpha + ramp*total

    def reseed(self, reference, *, confirm_prune=None):
        return _reseed(self.model, self.rows, self.cameras, self.views, self.slots, self.bins, self.matrices,
                       self.pipe, self.black, self.options, reference, preserve_optimizer=True, confirm_prune=confirm_prune)

"""Attach Gaussian geometry to repeatedly observed interior facial skin."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import torch
from torch.nn import functional as F

from .face_supplement import REGIONS, _hull
from .shape_refinement import _projection
from .size_child_optimization import _preserve_pose
from .splat_surface_distance import folded_normal_distance


CONFIG = dict(landmark_confidence=.5, max_landmark_error_px=6., max_landmark_error_fraction=.08,
              min_landmark_span_px=24., min_material_pixels=20, boundary_margin_fraction=.035,
              glasses_clearance_fraction=.09, lip_clearance_fraction=.065, min_color_seed_pixels=6,
              min_color_seed_luma=.055, max_color_seed_luma=.94, max_color_seed_chroma_mad=.045,
              source_chroma_tolerance_min=.055, source_chroma_tolerance_max=.115,
              min_relative_luma=.48, max_relative_luma=1.60, min_face_cosine=.28,
              mesh_anchor_depth_slack_m=.010, center_visibility_depth_slack_m=.025,
              min_independent_source_bins=2, active_opacity=.08,
              center_clearance_m=.0025, normal_sigma_clearance_m=.0020, loss_unit_m=.003,
              sheet_thin_over_middle=.55, sheet_middle_over_major=.45,
              center_weight=1., thickness_weight=.45, alignment_weight=.30)

_GEOMETRY_PARAMETERS = ('_xyz', '_scaling', '_rotation')
_AUDIT_FIELDS = ('position', 'log_scale', 'quaternion')
_PAIR_DTYPE = np.dtype([('view', 'i4'), ('stable_id', 'i8'), ('face', 'i4'),
                        ('mean_mm', 'f4'), ('normal_sigma_mm', 'f4'), ('distance_mm', 'f4'),
                        ('thin_axis_angle_deg', 'f4')])


@dataclass(frozen=True)
class _SkinView:
    box: tuple
    mask: torch.Tensor
    faces: torch.Tensor
    evidence: dict


def _disk(radius):
    distance = np.arange(-radius, radius+1)**2
    return distance[:, None]+distance[None, :] <= radius**2


def _polygon(points, origin, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    hull = _hull(points-origin)
    if len(hull) >= 3:
        cv2.fillConvexPoly(mask, np.asarray(hull, dtype=np.int32), 1)
    return mask.astype(bool)


def _skin_pixels(view, row, observed, projected):
    """Conservative native color support in the lower internal face landmarks."""
    c = CONFIG
    eye_names = ('right_eye', 'left_eye')
    names = (*eye_names, 'nose', 'mouth')
    core = np.concatenate([REGIONS[name] for name in names])
    confidence = np.isfinite(observed).all(1) & (observed[:, 2] >= c['landmark_confidence'])
    reliable = core[confidence[core]]
    span = float(np.ptp(observed[reliable, :2], axis=0).max()) if len(reliable) else 0.
    evidence = dict(frame=row['frame'], landmark_span_px=span, status='unsupported', skin_pixels=0)
    if span < c['min_landmark_span_px']:
        return None, evidence | dict(reason='face_landmark_span')
    error = np.linalg.norm(projected-observed[:, :2], axis=1)
    usable = confidence & np.isfinite(error) & (error <= min(c['max_landmark_error_px'], c['max_landmark_error_fraction']*span))
    groups = {name: REGIONS[name][usable[REGIONS[name]]] for name in names}
    eyes = np.concatenate([ids for name in eye_names if len(ids := groups[name]) >= 3]
                          or [np.empty(0, dtype=np.int64)])
    evidence['paired_landmark_counts'] = {name: len(ids) for name, ids in groups.items()}
    if len(eyes) < 3 or len(groups['nose']) < 4 or len(groups['mouth']) < 6 or not usable[80]:
        return None, evidence | dict(reason='paired_eye_nose_mouth_landmarks')
    keep = np.concatenate((eyes, groups['nose'], groups['mouth']))
    evidence['paired_landmark_peak_px'] = float(error[keep].max())
    both = np.stack((observed[:, :2], projected))
    xy = both[:, keep].reshape(-1, 2)
    vx, vy, vr, vb = view.box
    left, top = np.maximum(np.floor(xy.min(0)).astype(int), [vx, vy])
    right, bottom = np.minimum(np.ceil(xy.max(0)).astype(int)+1, [vr, vb])
    if not (left < right and top < bottom):
        return None, evidence | dict(reason='empty_face_crop')
    box = tuple(map(int, (left, top, right, bottom)))
    origin = np.array([left, top])
    shape = (bottom-top, right-left)
    yy, xx = np.mgrid[top:bottom, left:right]
    valid = view.inside[top:bottom, left:right].detach().cpu().numpy().copy()
    for points in both:
        mouth = points[groups['mouth']]
        down = mouth.mean(0)-points[80]
        distance = np.linalg.norm(down)
        if not np.isfinite(distance) or distance < .10*span:
            return None, evidence | dict(reason='degenerate_lower_face_axis')
        down /= distance
        eye_limit = (points[eyes] @ down).max()+max(2, round(c['glasses_clearance_fraction']*span))
        valid &= (xx*down[0]+yy*down[1] >= eye_limit) & _polygon(points[keep], origin, shape)
        lips = _polygon(mouth, origin, shape)
        valid &= ~binary_dilation(lips, structure=_disk(max(2, round(c['lip_clearance_fraction']*span))))
    for left_other, top_other, right_other, bottom_other in row.get('other_person_boxes_xyxy', ()):
        valid &= ~((xx >= left_other) & (xx < right_other) & (yy >= top_other) & (yy < bottom_other))
    valid = binary_erosion(valid, structure=_disk(max(1, round(c['boundary_margin_fraction']*span))))
    evidence['interior_before_color_pixels'] = int(valid.sum())
    if int(valid.sum()) < c['min_material_pixels']:
        return None, evidence | dict(reason='insufficient_eroded_skin_interior')

    rgb = view.rgb[:, top-vy:bottom-vy, left-vx:right-vx].detach().cpu().numpy().transpose(1, 2, 0)
    rgb = rgb.astype(np.float32)/(255. if rgb.dtype == np.uint8 else 1.)
    luma = rgb @ np.asarray([.2126, .7152, .0722], np.float32)
    chroma = rgb[..., [0, 2]]/rgb.sum(-1, keepdims=True).clip(1e-5)
    seed_center = both[:, 80].mean(0)
    radius = max(2., .075*span)
    seed = valid & ((xx-seed_center[0])**2+(yy-seed_center[1])**2 <= radius**2)
    seed &= (luma >= c['min_color_seed_luma']) & (luma <= c['max_color_seed_luma'])
    evidence['color_seed_pixels'] = int(seed.sum())
    if int(seed.sum()) < c['min_color_seed_pixels']:
        return None, evidence | dict(reason='no_reliable_native_nose_color')
    center = np.median(chroma[seed], axis=0)
    chroma_distance = np.linalg.norm(chroma-center, axis=-1)
    mad = float(np.median(chroma_distance[seed]))
    if mad > c['max_color_seed_chroma_mad']:
        return None, evidence | dict(reason='mixed_nose_color_seed')
    seed_luma = float(np.median(luma[seed]))
    tolerance = float(np.clip(4*mad, c['source_chroma_tolerance_min'], c['source_chroma_tolerance_max']))
    valid &= (chroma_distance <= tolerance) & (luma >= seed_luma*c['min_relative_luma']) & (
        luma <= min(c['max_color_seed_luma'], seed_luma*c['max_relative_luma']))
    pixels = int(valid.sum())
    if pixels < c['min_material_pixels']:
        return None, evidence | dict(reason='insufficient_consistent_native_color')
    receipt = dict(evidence, status='supported', skin_pixels=pixels, box_xyxy=list(box),
                   native_material_mask_sha256=hashlib.sha256(np.packbits(valid).tobytes()).hexdigest(),
                   seed_luma=seed_luma, seed_chroma_mad=mad, chroma_tolerance=tolerance)
    return (box, torch.as_tensor(valid, device=view.inside.device)), receipt


def _in_skin(xy, depth, skin, mesh_depth, tolerance):
    if skin is None:
        return torch.zeros_like(depth, dtype=torch.bool)
    (left, top, right, bottom), mask = skin
    # K uses corner coordinates: floor(K projection) indexes the native pixel.
    x, y = xy.floor().long().unbind(-1)
    valid = (depth > .2) & (x >= left) & (x < right) & (y >= top) & (y < bottom)
    x, y = x.clamp(left, right-1), y.clamp(top, bottom-1)
    return valid & mask[y-top, x-left] & ((depth-mesh_depth[y, x]).abs() <= tolerance)


def _geometry(local, log_sigma, quaternion, face_scale, *, detach_tangent_scales):
    """Triangle-local +Y and normalized wxyz match the bound CUDA renderer."""
    from roma import quat_wxyz_to_xyzw, unitquat_to_rotmat

    rotation = unitquat_to_rotmat(quat_wxyz_to_xyzw(F.normalize(quaternion, dim=-1)))
    sigma = log_sigma.exp()*face_scale[:, None]
    ordered, axes = sigma.detach().sort(dim=-1)
    sheet = (ordered[:, 0] <= CONFIG['sheet_thin_over_middle']*ordered[:, 1]) & (
        ordered[:, 1] >= CONFIG['sheet_middle_over_major']*ordered[:, 2])
    thin = axes[:, :1]
    normal = rotation[:, 1]
    thin_alignment_sq = normal.gather(1, thin)[:, 0].square().clamp(max=1.)
    if detach_tangent_scales:
        # Keep the actual covariance value while allowing thickness loss to
        # adjust the thin sigma and orientation, leaving tangent widths free.
        sigma = sigma.detach().scatter(1, thin, sigma.gather(1, thin))
    scaled_axes = rotation*sigma[:, None, :]
    return local[:, 1]*face_scale, scaled_axes[:, 1].norm(dim=-1), thin_alignment_sq, sheet, scaled_axes


class SurfaceAttachment:
    """Use frozen triangle material evidence and update current bound Gaussians."""

    @torch.no_grad()
    def __init__(self, model, constraints, head_faces, observed, *, projected, distribution=False):
        self.model, self.constraints = model, constraints
        self.distribution = distribution
        observed, projected = np.asarray(observed), np.asarray(projected)
        if (observed.ndim != 3 or observed.shape[1:] != (118, 3) or projected.shape != observed.shape[:2]+(2,)
                or len(observed) != model.num_timesteps or head_faces.shape != (len(model.faces),)):
            raise ValueError('Surface attachment requires registered native RTMW118 points and one fixed mesh topology.')
        if model.static_offset.requires_grad:
            raise ValueError('Skin attachment requires the post-warmup frozen posed mesh.')
        head_ids = head_faces.nonzero(as_tuple=True)[0]
        head_triangles = model.faces[head_ids]
        views, masks = [], []
        for i, (view, row, slot, matrix) in enumerate(zip(constraints.views, constraints.rows,
                                                        constraints.slots, constraints.matrices, strict=True)):
            skin, evidence = _skin_pixels(view, row, observed[slot], projected[slot])
            faces = torch.zeros_like(head_faces)
            if skin is not None and len(head_ids):
                xyz = model.vertices_sequence[slot][head_triangles]
                center = xyz.mean(1)
                xy, depth = _projection(center, *matrix)
                w2c = matrix[1]
                camera_center = -w2c[:3, :3].T @ w2c[:3, 3]
                normal = F.normalize(torch.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0], dim=-1), dim=-1)
                cosine = (normal*F.normalize(camera_center-center, dim=-1)).sum(-1)
                supported = (cosine >= CONFIG['min_face_cosine']) & _in_skin(
                    xy, depth, skin, view.mesh_depth, CONFIG['mesh_anchor_depth_slack_m'])
                faces[head_ids] = supported
            evidence.update(angular_bin=constraints.bins[i], visible_skin_triangles=int(faces.count_nonzero()))
            box, mask = skin if skin is not None else ((0, 0, 0, 0), faces[:0])
            views.append(_SkinView(box, mask, faces, evidence))
            masks.append(faces)
        bins = sorted(set(constraints.bins))
        if not bins:
            raise ValueError('Surface attachment requires a fixed native angular-bin roster.')
        support_bins = torch.stack([torch.stack([mask for mask, value in zip(masks, constraints.bins, strict=True)
                                               if value == group]).any(0) for group in bins]).sum(0)
        confirmed = support_bins >= CONFIG['min_independent_source_bins']
        self.views = [_SkinView(v.box, v.mask, v.faces & confirmed, dict(v.evidence,
                     repeated_skin_triangles=int((v.faces & confirmed).count_nonzero()))) for v in views]
        self.has_supported_view = [row.evidence['repeated_skin_triangles'] > 0 for row in self.views]
        self.supported_faces = int(confirmed.count_nonzero())
        self.tally = model._xyz.new_zeros((len(views), 9 if distribution else 8), dtype=torch.float64)
        self.update_audits = []
        self._pending_update = None

    @torch.no_grad()
    def _selected(self, index, visible):
        model = self.model
        if model.timestep != self.constraints.slots[index]:
            raise ValueError('Skin loss must use the exact posed raster and registered source frame.')
        view = self.views[index]
        if not self.has_supported_view[index]:
            return model.binding[:0]
        candidate = view.faces[model.binding] & (model.get_opacity[:, 0] >= CONFIG['active_opacity'])
        if visible is not None:
            candidate &= visible.reshape(-1)
        ids = candidate.nonzero(as_tuple=True)[0]
        owner = model.binding[ids]
        xyz = (model.face_orien_mat[owner] @ model._xyz[ids, :, None]).squeeze(-1)
        xyz = xyz*model.face_scaling[owner]+model.face_center[owner]
        xy, depth = _projection(xyz, *self.constraints.matrices[index])
        valid = _in_skin(xy, depth, (view.box, view.mask), self.constraints.views[index].mesh_depth,
                         CONFIG['center_visibility_depth_slack_m'])
        return ids[valid]

    def _geometry(self, ids, *, detach_tangent_scales):
        model = self.model
        scale = model.face_scaling[model.binding[ids], 0]
        return _geometry(model._xyz[ids], model._scaling[ids], model._rotation[ids], scale,
                         detach_tangent_scales=detach_tangent_scales)

    def loss(self, index, visible, *, audit_step=None):
        """An unweighted scalar for the existing native RGB and alpha objective."""
        ids = self._selected(index, visible)
        self.tally[index, 0] += 1
        if self.distribution:
            return self._distribution_loss(index, ids, audit_step)
        if not len(ids):
            return self.model._xyz.new_zeros(())
        center, sigma, alignment_sq, sheet, _ = self._geometry(ids, detach_tangent_scales=True)
        unit = CONFIG['loss_unit_m']
        center_over = ((center.abs()-CONFIG['center_clearance_m'])/unit).relu()
        sigma_over = ((sigma-CONFIG['normal_sigma_clearance_m'])/unit).relu()
        center_penalty = F.smooth_l1_loss(center_over, torch.zeros_like(center_over), reduction='mean')
        thickness_penalty = F.smooth_l1_loss(sigma_over, torch.zeros_like(sigma_over), reduction='none')
        thickness_penalty = thickness_penalty.masked_fill(~sheet, 0).mean()
        alignment_penalty = (1-alignment_sq).masked_fill(~sheet, 0).mean()
        result = (CONFIG['center_weight']*center_penalty + CONFIG['thickness_weight']*thickness_penalty
                  + CONFIG['alignment_weight']*alignment_penalty)
        with torch.no_grad():
            self.tally[index, 1:4].add_(self.tally.new_tensor([1, len(ids), 0]))
            self.tally[index, 3].add_(sheet.count_nonzero())
            self.tally[index, 4:].add_(torch.stack((result, center_penalty, thickness_penalty, alignment_penalty)))
        return result

    def _distribution_loss(self, index, ids, audit_step):
        model = self.model
        if self._pending_update is not None:
            raise RuntimeError('A sampled skin backward pass has no recorded optimizer step.')
        if not len(ids):
            if audit_step is not None:
                self.update_audits.append(dict(step=audit_step, frame=self.views[index].evidence['frame'],
                                               eligible_rows=0, state='no_visible_skin'))
            return model._xyz.new_zeros(())

        parameters = tuple(getattr(model, name)[ids] for name in _GEOMETRY_PARAMETERS)
        face_scale = model.face_scaling[model.binding[ids], 0]
        center, sigma, alignment_sq, sheet, _ = _geometry(*parameters, face_scale, detach_tangent_scales=False)
        distance = folded_normal_distance(center, sigma)
        unit = CONFIG['loss_unit_m']
        distribution_penalty = F.smooth_l1_loss(distance/unit, torch.zeros_like(distance), reduction='mean')
        alignment_penalty = (1-alignment_sq).masked_fill(~sheet, 0).mean()
        result = distribution_penalty + CONFIG['alignment_weight']*alignment_penalty
        if audit_step is not None:
            self._prepare_update_audit(audit_step, index, ids, face_scale, parameters)
        with torch.no_grad():
            self.tally[index, 1] += 1
            self.tally[index, 2] += len(ids)
            self.tally[index, 3] += sheet.count_nonzero()
            self.tally[index, 4:].add_(torch.stack((result, distribution_penalty, sigma.sum()*1000,
                                                  alignment_penalty, distance.sum()*1000)))
        return result

    def _prepare_update_audit(self, step, index, ids, face_scale, parameters):
        gradients = {}

        def capture(field):
            def record(gradient):
                gradients[field] = torch.linalg.vector_norm(gradient.detach(), dim=-1)
            return record

        for field, parameter in zip(_AUDIT_FIELDS, parameters, strict=True):
            parameter.register_hook(capture(field))
        self._pending_update = dict(step=step, frame=self.views[index].evidence['frame'], ids=ids,
            stable_ids=self.model.densification_ids[ids], face_scale=face_scale.detach(),
            before=tuple(parameter.detach() for parameter in parameters), gradients=gradients)

    @torch.no_grad()
    def record_optimizer_step(self):
        """Read a sampled main Adam step before any parameter projection or topology change."""
        audit = self._pending_update
        if audit is None:
            return
        model, ids = self.model, audit['ids']
        if (set(audit['gradients']) != set(_AUDIT_FIELDS)
                or not torch.equal(model.densification_ids[ids], audit['stable_ids'])):
            raise RuntimeError(f"Missing skin prior gradients or changed audit rows at step {audit['step']}.")
        before = audit['before']
        after = tuple(getattr(model, name)[ids] for name in _GEOMETRY_PARAMETERS)
        xyz_before, log_scale_before, quat_before = before
        xyz_after, log_scale_after, quat_after = after
        scale_mm = audit['face_scale']*1000
        xyz_update = (xyz_after-xyz_before).norm(dim=-1)*scale_mm
        scale_update = (log_scale_after.exp()-log_scale_before.exp()).norm(dim=-1)*scale_mm
        q0, q1 = F.normalize(quat_before, dim=-1), F.normalize(quat_after, dim=-1)
        chord = torch.minimum((q1-q0).norm(dim=-1), (q1+q0).norm(dim=-1))
        rotation_update = torch.rad2deg(4*(chord/2).clamp(max=1).asin())
        stats = {}
        for field, update in zip(_AUDIT_FIELDS, (xyz_update, scale_update, rotation_update), strict=True):
            gradient = audit['gradients'][field]
            if not (torch.isfinite(gradient).all() and torch.isfinite(update).all()):
                raise RuntimeError(f"Nonfinite skin prior gradient or Adam update at step {audit['step']}.")
            active = gradient > 0
            stats[field] = dict(prior_gradient_rows=int(active.sum()), prior_gradient_l2=float(gradient.norm()),
                updated_rows=int((update > 0).sum()), updated_rows_with_prior_gradient=int(((update > 0) & active).sum()),
                update_mean=float(update.mean()), update_max=float(update.max()))
        self.update_audits.append(dict(step=audit['step'], frame=audit['frame'], eligible_rows=len(ids),
                                       state='recorded', parameters=stats))
        self._pending_update = None

    def report(self):
        """Report source eligibility and actual prior participation."""
        if self._pending_update is not None:
            raise RuntimeError('A sampled skin backward pass has no recorded optimizer step.')
        reports = []
        for row, tally in zip(self.views, self.tally.tolist(), strict=True):
            calls, supported, points, sheets, loss, first, second, align = tally[:8]
            divisor = max(supported, 1.)
            record = dict(row.evidence, loss_calls=int(calls), supported_calls=int(supported),
                supported_row_draws=int(points), flattened_row_draws=int(sheets),
                mean_loss_on_supported_calls=loss/divisor, mean_alignment_penalty=align/divisor)
            if self.distribution:
                record.update(distribution_row_draws=int(points), distribution_nonsheet_row_draws=int(points-sheets),
                    mean_distribution_penalty=first/divisor,
                    mean_normal_sigma_mm=second/points if points else None,
                    mean_expected_abs_distance_mm=tally[8]/points if points else None)
            else:
                record.update(mean_center_penalty=first/divisor, mean_thickness_penalty=second/divisor)
            reports.append(record)
        receipt = dict(schema='shuolee-visible-skin-surface-attachment/1', config=CONFIG.copy(),
            repeated_skin_triangles=self.supported_faces,
            supported_views=sum(self.has_supported_view), views=reports,
            material_basis='Lower interior paired native landmarks, eroded independent alpha, excluded eye/eyewear '
                           'band and lips, relative per-view nose color, front-facing frozen mesh visibility. '
                           'This conservative support is a material proxy.',
            geometry_basis='Bound +Y local normal, normalized wxyz rotation and physical one-sigma axes. '
                           'Center uses normal displacement. Covariance gradients leave tangent scales free.')
        if self.distribution:
            eligible = sum(row['supported_row_draws'] for row in reports)
            entered = sum(row['distribution_row_draws'] for row in reports)
            receipt.update(schema='shuolee-visible-skin-surface-attachment/2',
                config=dict(CONFIG, loss='fixed_bound_plane_folded_normal_expected_abs_smooth_l1',
                            uses_sheet_gate_for_distribution=False),
                geometry_basis='Analytic E|d| to the fixed bound triangle plane. The folded normal includes the '
                               'signed center mean and full projected covariance n^T Sigma n. Every eligible '
                               'row enters the distribution term. Only the thin-axis angle term requires a sheet.',
                coverage=dict(eligible_row_draws=eligible, distribution_row_draws=entered,
                    fraction=entered/eligible if eligible else None,
                    distribution_nonsheet_row_draws=sum(row['distribution_nonsheet_row_draws'] for row in reports)),
                gradient_update_audit=dict(
                    gradient_scope='Weighted face prior only, captured on gathered parameters during the main backward pass.',
                    update_scope='Observed total main Adam step for the same rows, before projection or densification.',
                    update_units=dict(position='mm in the native posed frame', log_scale='mm L2 of one-sigma axis lengths',
                                      quaternion='degrees of physical relative rotation'),
                    steps=self.update_audits.copy()))
        return receipt

    @torch.no_grad()
    def measure(self, *, pairs_path=None):
        """Measure current exact bound covariance on the same source skin support."""
        rows, pairs = [], []
        with _preserve_pose(self.model):
            for index, view in enumerate(self.views):
                self.model.select_mesh_by_timestep(self.constraints.slots[index])
                ids = self._selected(index, None)
                result = dict(frame=view.evidence['frame'], supported_rows=len(ids), flattened_rows=0)
                if len(ids):
                    center, normal, alignment_sq, sheet, axes = self._geometry(ids, detach_tangent_scales=False)
                    tangent_axes = axes[:, [0, 2]]
                    tangent = torch.linalg.eigvalsh(tangent_axes @ tangent_axes.transpose(-1, -2))[:, 1].clamp_min(0).sqrt()
                    angle = torch.rad2deg(alignment_sq.sqrt().acos())
                    if self.distribution or pairs_path is not None:
                        distance = folded_normal_distance(center, normal)
                        result['expected_abs_distance_mm'] = _quantiles(distance*1000)
                    if pairs_path is not None:
                        pair = np.empty(len(ids), dtype=_PAIR_DTYPE)
                        pair['view'] = index
                        pair['stable_id'] = self.model.densification_ids[ids].cpu().numpy()
                        pair['face'] = self.model.binding[ids].cpu().numpy()
                        pair['mean_mm'] = (center*1000).cpu().numpy()
                        pair['normal_sigma_mm'] = (normal*1000).cpu().numpy()
                        pair['distance_mm'] = (distance*1000).cpu().numpy()
                        pair['thin_axis_angle_deg'] = angle.masked_fill(~sheet, float('nan')).cpu().numpy()
                        pairs.append(pair)
                    result.update(flattened_rows=int(sheet.count_nonzero()),
                                  abs_normal_center_mm=_quantiles(center.abs()*1000),
                                  signed_normal_center_mm=_quantiles(center*1000),
                                  normal_sigma_mm=_quantiles(normal*1000),
                                  tangent_major_sigma_mm=_quantiles(tangent*1000),
                                  flattened_thin_axis_angle_deg=_quantiles(angle[sheet]))
                rows.append(result)
        receipt = dict(visibility_basis='Projected center within repeated native skin and frozen z buffer', views=rows)
        if pairs_path is not None:
            path = Path(pairs_path)
            values = np.concatenate(pairs) if pairs else np.empty(0, dtype=_PAIR_DTYPE)
            with path.open('xb') as stream:
                np.savez_compressed(stream, pairs=values,
                    frames=[row.evidence['frame'] for row in self.views],
                    skin_mask_sha256=[row.evidence.get('native_material_mask_sha256', '') for row in self.views])
            receipt['stable_pairs'] = dict(file=path.name, rows=len(values), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                key='Same native frame and material mask SHA-256, stable Gaussian ID and binding face.')
        return receipt


def compare_skin_pairs(first_path, last_path):
    """Compare retained stable rows only where the native skin region is the same."""
    paths = [Path(first_path), Path(last_path)]
    snapshots = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            snapshots.append((archive['frames'], archive['skin_mask_sha256'], archive['pairs']))
    (frames, masks, first), (last_frames, last_masks, last) = snapshots
    if not np.array_equal(frames, last_frames) or not np.array_equal(masks, last_masks):
        raise ValueError('Paired skin comparison requires the same native frame and material masks.')
    key_dtype = np.dtype([('view', 'i4'), ('stable_id', 'i8')])

    def keys(values):
        result = np.empty(len(values), dtype=key_dtype)
        result['view'], result['stable_id'] = values['view'], values['stable_id']
        if len(np.unique(result)) != len(values):
            raise ValueError('Duplicate stable skin observation within a native view.')
        return result

    _, first_idx, last_idx = np.intersect1d(keys(first), keys(last), assume_unique=True, return_indices=True)
    paired_first, paired_last = first[first_idx], last[last_idx]
    if not np.array_equal(paired_first['face'], paired_last['face']):
        raise ValueError('A stable skin Gaussian changed its binding face.')

    def summary(left, right):
        available = np.isfinite(left) & np.isfinite(right)
        left, right = left[available], right[available]

        def percentiles(values):
            return dict(zip(('p05', 'p50', 'p95'), np.quantile(values, (.05, .50, .95)).tolist()))

        if not len(left):
            return dict(paired_rows=0)
        return dict(paired_rows=len(left), first=percentiles(left), last=percentiles(right),
                    paired_delta=percentiles(right-left))

    return dict(schema='shuolee-stable-skin-pairs/1',
        first_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        last_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(), first_rows=len(first), last_rows=len(last),
        paired_rows=len(paired_first),
        paired_rows_by_frame={str(frame): int(np.count_nonzero(paired_first['view'] == i))
                              for i, frame in enumerate(frames)},
        expected_abs_distance_mm=summary(paired_first['distance_mm'], paired_last['distance_mm']),
        abs_center_distance_mm=summary(np.abs(paired_first['mean_mm']), np.abs(paired_last['mean_mm'])),
        normal_sigma_mm=summary(paired_first['normal_sigma_mm'], paired_last['normal_sigma_mm']),
        thin_axis_angle_deg=summary(paired_first['thin_axis_angle_deg'], paired_last['thin_axis_angle_deg']))


def _quantiles(value):
    if not len(value):
        return None
    median, high = value.quantile(value.new_tensor([.5, .95])).tolist()
    return dict(p50=median, p95=high)

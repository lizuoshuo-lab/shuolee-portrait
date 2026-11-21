"""Explicit anatomical 2D -> SMPL-X surface bindings and differentiable loss.

This module does not estimate cameras, infer visibility from detector confidence,
or attach texture features to the closest bare-body vertex. Image observations
stay in their recorded native integer-centre pixel coordinate system.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def face_bindings(model):
    """IBUG 17..67 / RTMW wholebody 40..90, fixed surface points only."""
    from smplx.joint_names import JOINT_NAMES
    if model.use_face_contour or model.joint_mapper is not None:
        raise ValueError('Use the fixed, unremapped SMPL-X landmark layout.')
    faces = model.faces_tensor[model.lmk_faces_idx].detach().cpu().numpy()
    bary = model.lmk_bary_coords.detach().cpu().numpy()
    if faces.shape != (51, 3) or bary.shape != (51, 3):
        raise ValueError('Unsupported SMPL-X face landmark topology.')
    if not np.allclose(bary.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid barycentric surface binding.')
    rows = []
    for slot in range(51):
        ibug = slot + 17
        region = ('brows' if ibug < 27 else 'nose' if ibug < 36
                  else 'eyes' if ibug < 48 else 'mouth')
        rows.append(dict(anchor_id=f'ibug68:{ibug}', ibug68_index=ibug,
                         rtmw133_index=slot + 40, fixed_face_slot=slot,
                         smplx_name=JOINT_NAMES[76 + slot], region=region,
                         triangle_index=int(model.lmk_faces_idx[slot]),
                         vertex_indices=faces[slot].tolist(),
                         barycentric=bary[slot].tolist(),
                         binding='fixed_anatomical_surface_not_nearest_vertex'))
    return rows


def face_world(vertices_world, model):
    """Same attached material location moves with the posed SMPL-X mesh."""
    ids = model.faces_tensor[model.lmk_faces_idx]
    return (vertices_world[..., ids, :] * model.lmk_bary_coords[..., None]).sum(-2)


def project_integer_centres(xyz_world, world_to_camera, K_integer):
    camera = xyz_world @ world_to_camera[..., :3, :3].transpose(-1, -2)
    camera = camera + world_to_camera[..., None, :3, 3]
    h = camera @ K_integer.transpose(-1, -2)
    # Prevent NaNs while reporting behind-camera points separately. No half-
    # pixel correction here: K is explicitly integer-centre, not renderer K.
    uv = h[..., :2] / h[..., 2:].clamp_min(1e-7)
    return uv, camera[..., 2]


class DirectFaceObjective:
    """Fixed observation states, immutable FIT/HELD, no residual-based pruning."""
    def __init__(self, uv, sigma_px, reliability, eligible, split, *, device):
        shape = np.asarray(uv).shape
        if len(shape) != 3 or shape[1:] != (51, 2):
            raise ValueError('Expected frame x fixed_face51 x xy.')
        n = shape[0]
        for value in (sigma_px, reliability, eligible, split):
            if np.asarray(value).shape != (n, 51):
                raise ValueError('Face observation shape mismatch.')
        if not np.isfinite(uv).all() or not np.isfinite(sigma_px).all():
            raise ValueError('Nonfinite native observations or uncertainty.')
        if np.any(np.asarray(sigma_px) <= 0):
            raise ValueError('Uncertainty must be positive, not zero confidence.')
        if not np.isfinite(reliability).all() or np.any(np.asarray(reliability) < 0):
            raise ValueError('Invalid detector reliability.')
        if not set(np.asarray(split).ravel()).issubset({'FIT', 'HELD'}):
            raise ValueError('Invalid immutable observation split.')
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.uv, self.sigma = tensor(uv), tensor(sigma_px)
        self.fit_weight = tensor(np.asarray(reliability) * np.asarray(eligible)
                                 * (np.asarray(split) == 'FIT'))
        self.held = torch.as_tensor(np.asarray(eligible) & (np.asarray(split) == 'HELD'),
                                    dtype=torch.bool, device=device)
        if not bool((self.fit_weight > 0).any()):
            raise ValueError('No observed, semantically verified FIT face points.')
        # Normalise each anatomical group, so eye support is not overwhelmed by
        # the 20 mouth points. No brow/eye range is silently zeroed.
        self.regions = {'brows': slice(0, 10), 'nose': slice(10, 19),
                        'eyes': slice(19, 31), 'mouth': slice(31, 51)}

    def __call__(self, projected):
        e = torch.linalg.vector_norm(projected - self.uv, dim=-1) / self.sigma
        rho = torch.sqrt(1 + e.square()) - 1
        terms = {}
        for name, idx in self.regions.items():
            w = self.fit_weight[:, idx]
            if bool((w > 0).any()):
                terms[name] = (rho[:, idx] * w).sum() / w.sum()
        return torch.stack(list(terms.values())).mean(), terms

    def residuals(self, projected):
        return torch.linalg.vector_norm(projected - self.uv, dim=-1)


def load_reviewed_face(root, rows, obs, model, *, device, synthetic=False):
    """Read observed visibility/identity. Unknown remains inactive and recorded.

    The sidecar is a reviewed input, not auto-produced by this loader. Model
    self-visibility, mask inclusion, or RTMW scores cannot assert 'observed'.
    """
    root = Path(root)
    path = root / 'data/smplx-face-observations.json'
    review = json.loads(path.read_text(encoding='utf-8-sig'))
    if review.get('synthetic_control', False) != synthetic:
        raise ValueError('Synthetic and measured observations cannot be mixed.')
    if review.get('coordinate_domain') != 'native_integer_pixel_centres':
        raise ValueError('Unknown image coordinate contract.')
    if review['observations_sha256'] != sha(root / 'data/observations.npz'):
        raise ValueError('2D observations changed since review.')
    if review['inputs_sha256'] != sha(root / 'data/inputs.json'):
        raise ValueError('Frame/camera inputs changed since review.')
    bindings = face_bindings(model)
    frame_ids = [str(r['frame']) for r in rows]
    if len(set(frame_ids)) != len(frame_ids) or list(obs['frame_ids'].astype(str)) != frame_ids:
        raise ValueError('Frame identity mismatch.')
    if obs['joint_names'][40:91].tolist() != [f'face-{i}' for i in range(17, 68)]:
        raise ValueError('RTMW semantic layout mismatch.')
    n = len(rows)
    eligible = np.zeros((n, 51), bool)
    sigma = np.ones((n, 51), float)
    reliability = np.zeros((n, 51), float)
    split = np.full((n, 51), 'FIT', dtype='<U4')
    seen = set()
    records = []
    frame_lookup = {v: i for i, v in enumerate(frame_ids)}
    allowed_states = {'observed', 'occluded', 'outside', 'unknown'}
    family_split = {}
    for record in review['records']:
        i = frame_lookup[str(record['frame'])]
        slot = int(record['ibug68_index']) - 17
        if not 0 <= slot < 51 or (i, slot) in seen:
            raise ValueError('Invalid/duplicate anatomical observation.')
        seen.add((i, slot))
        b = bindings[slot]
        if record['anchor_id'] != b['anchor_id']:
            raise ValueError('2D-to-3D anatomical identity mismatch.')
        state = record['visibility']
        if state not in allowed_states:
            raise ValueError('Visibility must be observed/occluded/outside/unknown.')
        if record['split'] not in ('FIT', 'HELD'):
            raise ValueError('Invalid split.')
        family = record['physical_family_id']
        if family in family_split and family_split[family] != record['split']:
            raise ValueError('Physical-family FIT/HELD leakage.')
        family_split[family] = record['split']
        split[i, slot] = record['split']
        active = state == 'observed' and record['semantic_identity_verified'] is True
        reason = 'active' if active else state + '_or_semantic_identity_unverified'
        if active:
            if not record.get('review_evidence') or record.get('visibility_source') not in (
                    'image_review', 'independent_annotation', 'synthetic_known'):
                raise ValueError('Observed visibility requires image evidence.')
            if not synthetic and record['visibility_source'] == 'synthetic_known':
                raise ValueError('Synthetic visibility in real data.')
            xy = obs['keypoints'][i, b['rtmw133_index']]
            if not np.array_equal(xy, np.asarray(record['uv_native'], dtype=xy.dtype)):
                raise ValueError('Reviewed native pixel coordinate changed.')
            w, h = rows[i]['wh']
            if not (0 <= xy[0] < w and 0 <= xy[1] < h):
                raise ValueError('Observed point outside original frame.')
            sigma[i, slot] = record['localization_sigma_px']
            reliability[i, slot] = record['reliability']
            eligible[i, slot] = True
        records.append({**record, 'active_for_fit': bool(active and record['split'] == 'FIT'),
                        'reason': reason, 'binding': b})
    if len(seen) != n * 51:
        raise ValueError('Every face observation needs a recorded state, including unknown.')
    objective = DirectFaceObjective(obs['keypoints'][:, 40:91], sigma, reliability,
                                    eligible, split, device=device)
    return objective, dict(source_sha256=sha(path), bindings=bindings,
                           records=records, no_error_based_exclusion=True,
                           missing_visibility_default='unknown_inactive')

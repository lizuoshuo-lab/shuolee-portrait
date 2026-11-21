"""Shared-identity SMPL-X fitting in explicitly labelled camera coordinates.

This is a local sequence adapter, not an official SMPLify-X sequence release.
It owns one beta tensor. Articulation is indexed by real frame IDs; the camera
matrices are buffers and are never optimized. Unknown/held face observations
cannot enter the objective. Body detector weights remain labelled heuristics.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .camera_world_units import normalize_camera_w2c, validate_camera_length_contract


@dataclass
class SequenceObservations:
    frame_ids: tuple
    targets: torch.Tensor
    confidence: torch.Tensor
    face_visibility: np.ndarray
    face_split: np.ndarray
    face_semantic_verified: np.ndarray
    physical_face_verified: np.ndarray
    face_family_ids: np.ndarray

    def __post_init__(self):
        n = len(self.frame_ids)
        if n == 0 or len(set(self.frame_ids)) != n:
            raise ValueError('Real frame IDs must be nonempty and unique.')
        if self.targets.shape != (n, 118, 2) or self.confidence.shape != (n, 118):
            raise ValueError('Explicit official coco25+hands+fixedface118 layout required.')
        if not torch.isfinite(self.targets).all() or not torch.isfinite(self.confidence).all():
            raise ValueError('Nonfinite observation.')
        if not ((self.confidence >= 0) & (self.confidence <= 1)).all():
            raise ValueError('Pass labelled relative confidence, not raw detector response.')
        arrays = [self.face_visibility, self.face_split, self.face_semantic_verified,
                  self.physical_face_verified, self.face_family_ids]
        if any(np.shape(a) != (n, 51) for a in arrays):
            raise ValueError('Every face target needs an explicit review state.')
        if not set(self.face_visibility.ravel()) <= {'observed', 'unknown', 'occluded', 'outside'}:
            raise ValueError('Unrecognized visibility state.')
        if not set(self.face_split.ravel()) <= {'FIT', 'HELD', 'UNASSIGNED'}:
            raise ValueError('Unrecognized split.')
        families = {}
        for family, split in zip(self.face_family_ids.ravel(), self.face_split.ravel()):
            if not isinstance(family, (str, np.str_)) or not family:
                raise ValueError('Explicit observation family identity required.')
            if family in families and families[family] != split:
                raise ValueError('A face family crosses the fitting/holdout boundary.')
            families[family] = split

    def face_active(self, split='FIT'):
        mask = ((self.face_visibility == 'observed') &
                self.face_semantic_verified & (self.face_split == split))
        return torch.as_tensor(mask, dtype=torch.bool, device=self.targets.device)

    def fitting_confidence(self, *, hand_policy='prior_only'):
        if hand_policy not in ('prior_only', 'fit_without_duplicate_wrists'):
            raise ValueError('Explicit hand policy required.')
        c = self.confidence.clone()
        # Do not create an alternate fitting route through COCO face channels.
        c[:, [0, 15, 16, 17, 18]] = 0
        c[:, [1, 8, 9, 12]] = 0  # Missing neck/pelvis and upstream ignored hips.
        c[:, 67:] *= self.face_active().to(c.dtype)
        if hand_policy == 'prior_only':
            c[:, 25:67] = 0
        else:
            c[:, [25, 46]] = 0  # Hand roots duplicate body wrist anchors.
        return c


def project_complete_k(points, K):
    """Zero-based integer pixel centres, full K including cx/cy and skew."""
    if points.ndim != 3 or K.shape != (points.shape[0], 3, 3):
        raise ValueError('Frame-wise complete K required.')
    q = torch.einsum('nij,nkj->nki', K, points)
    return q[..., :2] / q[..., 2:]


def camera_to_world(points, w2c, *, translation_unit=None):
    if translation_unit != 'meters':
        raise ValueError('camera_to_world requires explicit translation_unit="meters" matching model vertices.')
    if points.ndim != 3 or points.shape[-1] != 3 or w2c.shape != (points.shape[0], 4, 4):
        raise ValueError('Expected model-meter NxVx3 points and matching meter Nx4x4 w2c.')
    if not torch.isfinite(points).all() or not torch.isfinite(w2c).all():
        raise ValueError('Nonfinite camera-to-world input.')
    # Row-vector inverse of Xc=R Xw+t. Applies to vertices, not naively to a
    # SMPL root translation (SMPL rotates around a shape-dependent pelvis).
    return torch.einsum('nki,nij->nkj', points - w2c[:, None, :3, 3], w2c[:, :3, :3])


class SharedIdentitySequence(nn.Module):
    def __init__(self, body, vposer, frame_ids, K, w2c, initial, *,
                 camera_length_contract=None, dtype=torch.float64):
        super().__init__()
        # Fail before touching body/VPoser when source units are undeclared.
        # These cameras are fixed inputs; only the model parameters are learned.
        source_w2c = w2c.detach().cpu().numpy() if isinstance(w2c, torch.Tensor) else w2c
        camera_units = normalize_camera_w2c(source_w2c, camera_length_contract)
        # Store only pickle-safe immutable primitives, never a MappingProxy.
        self._camera_length_contract = tuple(camera_units.contract.items())
        self.frame_ids = tuple(map(str, frame_ids))
        n = len(self.frame_ids)
        if n == 0 or len(set(self.frame_ids)) != n:
            raise ValueError('Duplicate/missing frame identity.')
        self.body = body.to(dtype=dtype)
        self.vposer = vposer.to(dtype=dtype)
        self.body.requires_grad_(False)
        self.vposer.requires_grad_(False)
        self.vposer.eval()
        device = next(self.body.buffers()).device
        K = torch.as_tensor(K, dtype=dtype, device=device)
        # Keep source values at float64 even if the chosen model dtype is lower.
        source_w2c = torch.tensor(camera_units.source_w2c.copy(), dtype=torch.float64, device=device)
        w2c = torch.tensor(camera_units.w2c_meters.copy(), dtype=dtype, device=device)
        if K.shape != (n, 3, 3) or w2c.shape != (n, 4, 4):
            raise ValueError('Exact per-frame cameras required; no interpolation.')
        if not torch.isfinite(K).all() or not torch.isfinite(w2c).all():
            raise ValueError('Nonfinite camera.')
        if not torch.allclose(K[:, 2], K.new_tensor([0., 0., 1.]).expand(n, 3)):
            raise ValueError('Not a complete rectified perspective K.')
        if not torch.all(K[:, [0, 1], [0, 1]] > 0):
            raise ValueError('Invalid focal length.')
        if not torch.allclose(w2c[:, 3], w2c.new_tensor([0., 0., 0., 1.]).expand(n, 4)):
            raise ValueError('Invalid homogeneous extrinsics.')
        rot = w2c[:, :3, :3]
        eye = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3)
        if not torch.allclose(rot @ rot.transpose(1, 2), eye, atol=1e-6, rtol=0) or not torch.allclose(torch.linalg.det(rot), torch.ones(n, dtype=dtype, device=device), atol=1e-6, rtol=0):
            raise ValueError('Extrinsic rotation must be proper, with no scale/shear.')
        self.register_buffer('K', K.detach().clone())
        self.register_buffer('source_w2c', source_w2c)
        self.register_buffer('w2c', w2c.clone())
        self._camera_constructor_frame_ids = self.frame_ids
        self._camera_constructor_values = tuple(
            (name, self._camera_tensor_signature(getattr(self, name)))
            for name in ('K', 'source_w2c', 'w2c'))
        # There is exactly one fitted identity, initialized from zero.
        self.state = nn.ParameterDict({'betas': nn.Parameter(torch.zeros((1, 10), dtype=dtype, device=device))})
        shapes = {'global_orient': (n, 3), 'transl': (n, 3), 'latent': (n, 32),
                  'jaw_pose': (n, 3), 'expression': (n, 10)}
        if set(initial) != set(shapes):
            raise ValueError('Explicit per-frame pose/appearance initialization required; no per-frame betas.')
        for key, shape in shapes.items():
            x = torch.as_tensor(initial[key], dtype=dtype, device=device)
            if x.shape != shape or not torch.isfinite(x).all():
                raise ValueError('Invalid initial ' + key)
            self.state[key] = nn.Parameter(x.clone())
        self.register_buffer('hand_pca', torch.zeros((n, body.num_pca_comps), dtype=dtype, device=device))
        self.register_buffer('eye_pose', torch.zeros((n, 3), dtype=dtype, device=device))

    @staticmethod
    def _camera_tensor_signature(value):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise RuntimeError('Camera checkpoint values must be dense tensors.')
        cpu = value.detach().cpu().contiguous()
        return (tuple(cpu.shape), str(cpu.dtype), cpu.view(torch.uint8).numpy().tobytes())

    def _assert_camera_binding(self):
        if self.frame_ids != self._camera_constructor_frame_ids:
            raise RuntimeError('Frozen camera frame identities/order changed since construction.')
        for name, expected in self._camera_constructor_values:
            value = getattr(self, name)
            if self._camera_tensor_signature(value) != expected:
                raise RuntimeError('Frozen camera buffer changed since construction: ' + name)
            if value.requires_grad:
                raise RuntimeError('Frozen camera buffer cannot require gradients: ' + name)

    def _apply(self, fn, recurse=True):
        # Probe before nn.Module visits any body/VPoser/parameter tensors.
        # Device-only transfers preserve the constructor values and dtypes;
        # changing precision would break their exact provenance binding.
        self._assert_camera_binding()
        for name, _ in self._camera_constructor_values:
            value = getattr(self, name)
            probe = fn(torch.empty_like(value, requires_grad=False))
            if probe.dtype != value.dtype:
                raise RuntimeError('Camera dtype is fixed at construction; select dtype in the constructor: ' + name)
            if probe.device.type == 'meta':
                raise RuntimeError('Frozen camera values cannot move to a data-free meta device.')
        result = super()._apply(fn, recurse=recurse)
        self._assert_camera_binding()
        return result

    def _validate_camera_extra_state(self, state):
        required = {'schema', 'frame_ids', 'contract'}
        if not isinstance(state, dict) or set(state) != required:
            raise RuntimeError('Camera checkpoint requires schema, frame_ids and contract metadata; legacy units are not inferred.')
        if state['schema'] != 'shuolee-sequence-camera-length/1':
            raise RuntimeError('Unsupported sequence camera checkpoint schema.')
        frames = state['frame_ids']
        if (not isinstance(frames, (list, tuple)) or
                not all(isinstance(item, str) for item in frames) or
                tuple(frames) != self._camera_constructor_frame_ids or
                self.frame_ids != self._camera_constructor_frame_ids):
            raise RuntimeError('Camera checkpoint frame identities/order differ from the constructor.')
        try:
            contract = validate_camera_length_contract(state['contract'])
        except (ValueError, TypeError) as error:
            raise RuntimeError('Invalid camera checkpoint length contract.') from error
        if dict(contract) != self.camera_length_contract:
            raise RuntimeError('Camera checkpoint length factor, unit or provenance differs from the constructor.')

    def _validate_camera_checkpoint(self, state_dict, prefix):
        self._assert_camera_binding()
        extra_key = prefix + '_extra_state'
        if extra_key not in state_dict:
            raise RuntimeError('Missing camera length checkpoint metadata: ' + extra_key)
        self._validate_camera_extra_state(state_dict[extra_key])
        values = {}
        for name, expected in self._camera_constructor_values:
            key = prefix + name
            if key not in state_dict:
                raise RuntimeError('Missing frozen camera checkpoint buffer: ' + key)
            value = state_dict[key]
            if self._camera_tensor_signature(value) != expected:
                raise RuntimeError('Frozen camera checkpoint buffer differs from the constructor: ' + key)
            values[name] = value
        # Check the explicit unit relationship as well as the constructor bind.
        # Round only to the originally selected model dtype, never infer a scale.
        try:
            units = normalize_camera_w2c(values['source_w2c'].detach().cpu().numpy(),
                                         state_dict[extra_key]['contract'])
        except (ValueError, TypeError) as error:
            raise RuntimeError('Invalid source camera units in checkpoint.') from error
        expected_meter = torch.tensor(units.w2c_meters.copy(), dtype=values['w2c'].dtype)
        if not torch.equal(values['w2c'].detach().cpu(), expected_meter):
            raise RuntimeError('Camera checkpoint source_w2c and w2c violate the declared common scale.')

    def get_extra_state(self):
        # Pure JSON-compatible values support normal state_dict serialization.
        self._assert_camera_binding()
        return dict(schema='shuolee-sequence-camera-length/1',
                    frame_ids=list(self._camera_constructor_frame_ids),
                    contract=self.camera_length_contract)

    def set_extra_state(self, state):
        # Loading weights must never replace the constructor's camera provenance.
        self._validate_camera_extra_state(state)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # This path is called for both top-level and parent-module loads. Reject
        # missing/conflicting units even under strict=False, before copying any
        # of this module's buffers or recursing into its learned body state.
        self._validate_camera_checkpoint(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward_state(self, state=None):
        self._assert_camera_binding()
        state = self.state if state is None else state
        n = len(self.frame_ids)
        out = self.body(betas=state['betas'].expand(n, -1),
                        body_pose=self.vposer.decode(state['latent'])['pose_body'].reshape(n, 63),
                        global_orient=state['global_orient'], transl=state['transl'],
                        jaw_pose=state['jaw_pose'], expression=state['expression'],
                        left_hand_pose=self.hand_pca, right_hand_pose=self.hand_pca,
                        leye_pose=self.eye_pose, reye_pose=self.eye_pose,
                        return_verts=True, return_full_pose=True)
        return out

    def project(self, out):
        self._assert_camera_binding()
        return project_complete_k(out.joints, self.K)

    def world_vertices(self, out):
        self._assert_camera_binding()
        return camera_to_world(out.vertices, self.w2c, translation_unit='meters')

    @property
    def camera_length_contract(self):
        return dict(self._camera_length_contract)

    def layout(self):
        offset = 0
        slices = {}
        for key, value in self.state.items():
            slices[key] = (offset, offset + value.numel(), tuple(value.shape))
            offset += value.numel()
        return slices

    def flatten(self):
        return torch.cat([v.detach().reshape(-1) for v in self.state.values()])

    def unpack(self, vector):
        return {k: vector[a:b].reshape(shape) for k, (a, b, shape) in self.layout().items()}

    def restore(self, vector):
        with torch.no_grad():
            for key, value in self.unpack(vector).items():
                self.state[key].copy_(value)

    def native_snapshot(self, reference_frame):
        self._assert_camera_binding()
        if str(reference_frame) not in self.frame_ids:
            raise ValueError('Reference frame is not in the observed roster.')
        i = self.frame_ids.index(str(reference_frame))
        with torch.no_grad():
            out = self.forward_state()
            world = self.world_vertices(out)
        return dict(frame_ids=list(self._camera_constructor_frame_ids),
                    reference_frame=self.frame_ids[i], shared_betas=self.state['betas'].detach().clone(),
                    reference_vertices_world=world[i].detach().clone(),
                    vertices_world_sequence=world.detach().clone(),
                    K=self.K.detach().clone(), w2c=self.w2c.detach().clone(),
                    w2c_meters=self.w2c.detach().clone(),
                    source_w2c=self.source_w2c.detach().clone(),
                    camera_length_contract=dict(self.camera_length_contract),
                    coordinate_system='opencv_world_meters',
                    normalized_translation_unit='meters',
                    scale_source_basis=self.camera_length_contract['basis'],
                    scale_source_evidence=self.camera_length_contract['evidence'],
                    common_sfm_gauge_validated=False,
                    root_parameter_domain='camera coordinates in model meters, per frame',
                    physical_camera_accepted=False, training_admitted=False, gaussian_export=False)


def staged_weights(stage, tensor):
    if stage not in range(5):
        raise ValueError('Expected one of five registered upstream stages.')
    bp = [404., 404., 57.4, 4.78, 4.78][stage]
    shape = [100., 50., 10., 5., 5.][stage]
    jaw = [[4040., 40400., 40400.], [4040., 40400., 40400.],
           [574., 5740., 5740.], [47.8, 478., 478.], [47.8, 478., 478.]][stage]
    return dict(body=bp, shape=shape, bending=3.17 * bp, hand=bp, expression=shape,
                jaw=tensor.new_tensor(jaw), face=[0., 0., 0., 0., 2.][stage])


def sequence_residuals(model, vector, observations, stage, image_heights):
    """Mean per-frame official objective, one shared shape prior.

    The algebra matches the pinned single-image SMPLify-X robustifier/priors.
    Hand PCA is fixed at its prior mean; no unreviewed hand target is optimized.
    Selection is entirely from the frozen observation review, never residuals.
    """
    if model.frame_ids != observations.frame_ids:
        raise ValueError('Observation/model frame order mismatch.')
    state = model.unpack(vector)
    out = model.forward_state(state)
    weights = staged_weights(stage, vector)
    confidence = observations.fitting_confidence()
    joint_weights = torch.ones_like(confidence)
    joint_weights[:, 67:] *= weights['face']
    delta = observations.targets - model.project(out)
    heights = torch.as_tensor(image_heights, device=vector.device, dtype=vector.dtype)
    if heights.shape != (len(model.frame_ids),) or not (heights > 0).all():
        raise ValueError('Native image heights required.')
    data = (confidence * joint_weights)[..., None] * (1000. / heights[:, None, None]) * 100. * delta / torch.sqrt(delta.square() + 10000.)
    bend = out.full_pose[:, 3:66][:, [52, 55, 9, 12]] * vector.new_tensor([1., -1., -1., -1.])
    n = len(model.frame_ids)
    terms = [data.reshape(-1), (state['latent'] * weights['body']).reshape(-1),
             (torch.exp(bend) * weights['bending'] ** .5).reshape(-1),
             (out.left_hand_pose * weights['hand']).reshape(-1),
             (out.right_hand_pose * weights['hand']).reshape(-1),
             (state['expression'] * weights['expression']).reshape(-1),
             (state['jaw_pose'] * weights['jaw']).reshape(-1)]
    return torch.cat([torch.cat(terms) / n ** .5,
                      (state['betas'] * weights['shape']).reshape(-1)]), out

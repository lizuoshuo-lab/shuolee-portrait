"""Fixed-table Torch consumer of the validated free-clothing CPU contract.

Construction copies/validates geometry and cameras once. The per-frame path is
Torch-only on its construction device, with no NumPy, CPU copy, eigendecomposition
or file reads. It owns no learned parameters or optimizer and never rebinds a
Gaussian. It is not a trainer or an assertion of physical camera calibration.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import torch

from .camera_world_units import normalize_camera_w2c, validate_camera_length_contract
from .free_clothing_transport import FreeClothingTransport, SCHEMA as SOURCE_SCHEMA


SCHEMA = 'shuolee-free-clothing-torch/1'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _async_need(condition, message):
    # Unlike bool(CUDA tensor), this does not copy a scalar to the host. A failed
    # CUDA device assertion requires stopping that run, not continuing training.
    torch._assert_async(condition, message)


def quaternion_product(a, b):
    """Hamilton product, wxyz: R(product) = R(a) @ R(b)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def normalized_quaternion_matrix(q):
    """Rotation matrix for a normalized wxyz quaternion; differentiable."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def _normalize_quaternion_stably(q):
    # Scaling before squaring avoids norm underflow/overflow without an
    # epsilon clamp. Keep the scale in autograd so parameter gradients retain
    # the ordinary normalization derivative at every representable amplitude.
    amplitude = q.abs().amax(dim=-1, keepdim=True)
    _async_need(torch.isfinite(amplitude).all() & (amplitude > 0).all(),
                'Zero/nonfinite quaternion amplitude.')
    _async_need(torch.isfinite(amplitude.reciprocal()).all(),
                'Quaternion amplitude is too small for finite parameter gradients.')
    scaled = q / amplitude
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)


def _matrix_quaternion_once(matrix):
    """Constant-table conversion, largest component branch, including pi turns."""
    m = matrix
    diagonal = torch.stack((1+m[..., 0, 0]+m[..., 1, 1]+m[..., 2, 2],
                            1+m[..., 0, 0]-m[..., 1, 1]-m[..., 2, 2],
                            1-m[..., 0, 0]+m[..., 1, 1]-m[..., 2, 2],
                            1-m[..., 0, 0]-m[..., 1, 1]+m[..., 2, 2]), -1)
    a, b, c = m[..., 2, 1]-m[..., 1, 2], m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]-m[..., 0, 1]
    d, e, f = m[..., 1, 0]+m[..., 0, 1], m[..., 2, 0]+m[..., 0, 2], m[..., 2, 1]+m[..., 1, 2]
    candidates = torch.stack((torch.stack((diagonal[..., 0], a, b, c), -1),
                              torch.stack((a, diagonal[..., 1], d, e), -1),
                              torch.stack((b, d, diagonal[..., 2], f), -1),
                              torch.stack((c, e, f, diagonal[..., 3]), -1)), -2)
    selected = diagonal.argmax(-1)
    denominator = diagonal.gather(-1, selected[..., None]).sqrt() * 2
    _need(bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all()),
          'Invalid constant motion rotation.')
    chosen = candidates.gather(-2, selected[..., None, None].expand(*selected.shape, 1, 4)).squeeze(-2)
    q = chosen / denominator
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FreeClothingTorch:
    frame_ids: tuple
    reference_frame_id: str
    gaussian_ids: tuple
    _rotations: torch.Tensor
    _quaternions: torch.Tensor
    _scales: torch.Tensor
    _centroids: torch.Tensor
    _residuals: torch.Tensor
    _support: torch.Tensor
    _camera_centers: torch.Tensor
    _metadata_json: str
    _buffer_guards: tuple

    @classmethod
    @torch.inference_mode(False)
    def from_cpu(cls, source, *, camera_length_contract, frame_source_w2c,
                 device='cpu', dtype=torch.float32):
        """Freeze source tables and explicit source-camera-to-metre conversion.

        Source geometry is already in its declared common metre coordinates;
        it is not scaled again. The length contract converts source camera
        translations exactly once. Consistency is a caller declaration, not a
        newly estimated scale or independently measured physical calibration.
        """
        _need(type(source) is FreeClothingTransport, 'A validated FreeClothingTransport source is required.')
        _need(dtype in (torch.float32, torch.float64), 'Only float32/float64 tables are supported.')
        length = dict(validate_camera_length_contract(camera_length_contract))
        _need(isinstance(frame_source_w2c, Mapping) and set(frame_source_w2c) == set(source.frame_ids),
              'Frozen source-camera map must exactly cover the named source frames.')
        source.assert_unchanged()
        source_report = source.report()
        _need(source_report['schema'] == SOURCE_SCHEMA and source.translation_unit == 'meters',
              'Source schema or normalized geometry unit differs.')
        # This is the sole compatibility boundary with the immutable CPU source
        # tables. Geometry is not reimplemented or recomputed on every step.
        rotations = torch.tensor(source._rotations.copy(), dtype=torch.float64, device='cpu')
        motion_q = _matrix_quaternion_once(rotations)
        residual = torch.tensor(source._residuals.copy(), dtype=torch.float64, device='cpu')
        ref_index = source.frame_ids.index(source.reference_frame_id)
        motion_q[ref_index] = torch.tensor([1., 0., 0., 0.], dtype=torch.float64, device='cpu')
        raw_cameras, metric_cameras, camera_centers = [], [], []
        for frame in source.frame_ids:
            converted = normalize_camera_w2c(frame_source_w2c[frame], length)
            _need(converted.source_w2c.shape == (4, 4),
                  'Each named frame requires exactly one 4x4 source w2c; camera batches are forbidden.')
            raw_cameras.append(converted.source_w2c.tolist())
            metric_cameras.append(converted.w2c_meters.tolist())
            E = torch.tensor(converted.w2c_meters.copy(), dtype=torch.float64, device='cpu')
            camera_centers.append(-E[:3, :3].T @ E[:3, 3])
        source.assert_unchanged()
        source_json = json.dumps(source_report, sort_keys=True, separators=(',', ':'), allow_nan=False)
        metadata = dict(schema=SCHEMA, source_schema=SOURCE_SCHEMA,
                        source_report_sha256=hashlib.sha256(source_json.encode()).hexdigest(),
                        source_report=source_report, frame_ids=list(source.frame_ids),
                        gaussian_ids=list(source.gaussian_ids), reference_frame_id=source.reference_frame_id,
                        motion_support_triangle_ids=source.motion_support_triangle_ids.tolist(),
                        camera_length_contract=length, source_w2c=raw_cameras, normalized_w2c=metric_cameras,
                        geometry_unit='meters', geometry_already_normalized=True,
                        camera_translation_scale_applications=1, fixed_point_count=len(source.gaussian_ids),
                        point_mapping_policy='fixed explicit identities/support; no clone, split, prune or nearest rebind',
                        input_contract_checked_during_preparation=True,
                        source_frame_files_independently_verified=False, files_read_in_forward=False,
                        source_camera_metric_relationship_independently_verified=False,
                        physical_camera_accepted=False, training_admitted=False, optimizer_present=False,
                        runtime_validation='Torch async assertions; failed CUDA assertions require ending the run')
        target = torch.device(device)
        _need(target.type in ('cpu', 'cuda'), 'Only CPU or CUDA consumers are supported.')
        arrays = dict(_rotations=rotations, _quaternions=motion_q,
                      _scales=torch.tensor(source._scales.copy(), dtype=torch.float64, device='cpu'),
                      _centroids=torch.tensor(source._centroids.copy(), dtype=torch.float64, device='cpu'),
                      _residuals=residual, _camera_centers=torch.stack(camera_centers))
        for name, value in arrays.items():
            arrays[name] = value.to(dtype=dtype)
            _need(bool(torch.isfinite(arrays[name]).all()), 'Table dtype conversion overflow: ' + name)
        _need(bool((arrays['_scales'] > 0).all()), 'Motion scale underflow during table preparation.')
        obj = object.__new__(cls)
        for name, value in arrays.items():
            object.__setattr__(obj, name, value.to(device=target))
        object.__setattr__(obj, '_support', torch.tensor(source.motion_support_triangle_ids.copy(), dtype=torch.long, device=target))
        for name, value in dict(frame_ids=tuple(source.frame_ids), reference_frame_id=source.reference_frame_id,
                                gaussian_ids=tuple(source.gaussian_ids),
                                _metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False)).items():
            object.__setattr__(obj, name, value)
        buffer_names = tuple(arrays) + ('_support',)
        object.__setattr__(obj, '_buffer_guards', tuple(
            (name, id(getattr(obj, name)), getattr(obj, name)._version) for name in buffer_names))
        return obj

    @property
    def device(self):
        return self._rotations.device

    @property
    def dtype(self):
        return self._rotations.dtype

    def report(self):
        data = json.loads(self._metadata_json)
        data.update(device=str(self.device), dtype=str(self.dtype),
                    triangle_table_shape=list(self._scales.shape),
                    tensor_parameters_owned=0, CUDA_parity_validated=False)
        return data

    def forward(self, frame_id, xyz_ref, quats_ref, log_scales_ref, *, gaussian_ids,
                return_covariance=False):
        """Gather one known frame; all numerical work stays on the table device.

        Caller parameters retain gradients. The returned SH reference direction
        must go into the existing SH evaluator, then its RGB goes to the renderer.
        No learned parameters, camera parameters or layouts are updated here.
        Full covariance is optional for audits, avoiding a large covariance
        allocation/decomposition in the native quaternion/scale training path.
        Ambient autocast never silently changes the declared numerical dtype.
        """
        # Constant-size host metadata checks, with no device scalar reads or
        # content hashing. Guards supported Tensor mutations/replacements;
        # bypassing Torch with .data/storage writes is outside this API.
        for name, identity, version in self._buffer_guards:
            value = getattr(self, name)
            _need(isinstance(value, torch.Tensor) and id(value) == identity and value._version == version and
                  not value.requires_grad and value.grad_fn is None,
                  'Frozen motion/camera table was mutated or replaced, or made trainable: ' + name)
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._forward_tensors(frame_id, xyz_ref, quats_ref, log_scales_ref,
                                         gaussian_ids=gaussian_ids, return_covariance=return_covariance)

    def _forward_tensors(self, frame_id, xyz_ref, quats_ref, log_scales_ref, *, gaussian_ids,
                         return_covariance):
        _need(type(return_covariance) is bool, 'return_covariance must be explicit bool.')
        _need(isinstance(frame_id, str) and frame_id in self.frame_ids, 'Unknown frame identity.')
        _need(isinstance(gaussian_ids, (tuple, list)) and tuple(gaussian_ids) == self.gaussian_ids,
              'Gaussian identities/order differ; this consumer has a fixed point layout.')
        n = len(self.gaussian_ids)
        for tensor, shape, label in ((xyz_ref, (n, 3), 'XYZ'), (quats_ref, (n, 4), 'quaternion'),
                                      (log_scales_ref, (n, 3), 'log scale')):
            _need(isinstance(tensor, torch.Tensor) and tensor.device == self.device and tensor.dtype == self.dtype and
                  tuple(tensor.shape) == shape, label + ' must match the fixed device, dtype and Gaussian layout.')
            _async_need(torch.isfinite(tensor).all(), 'Nonfinite reference ' + label)
        frame = self.frame_ids.index(frame_id)
        reference = self.frame_ids.index(self.reference_frame_id)
        support = self._support
        Q, q_motion = self._rotations[frame, support], self._quaternions[frame, support]
        a = self._scales[frame, support]
        q_ref = _normalize_quaternion_stably(quats_ref)
        if frame_id == self.reference_frame_id:
            centers, quaternions, log_scales = xyz_ref.clone(), q_ref, log_scales_ref.clone()
        else:
            centers = a[:, None] * torch.einsum('nij,nj->ni', Q, xyz_ref-self._centroids[reference, support]) + self._centroids[frame, support]
            quaternions = quaternion_product(q_motion, q_ref)
            quaternions = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
            log_scales = log_scales_ref + a.log()[:, None]
        scales = log_scales.exp()
        _async_need(torch.isfinite(scales).all() & (scales > 0).all(), 'Transported scale overflow/underflow.')
        variances = scales.square()
        _async_need(torch.isfinite(variances).all() & (variances > 0).all(),
                    'Transported covariance eigenvalue overflow/underflow.')
        _async_need(torch.isfinite(scales.reciprocal()).all(), 'Transported inverse scale overflow.')
        covariance = None
        if return_covariance:
            rotation = normalized_quaternion_matrix(quaternions)
            factor = rotation * scales[:, None, :]
            covariance = factor @ factor.transpose(-1, -2)
            _async_need(torch.isfinite(covariance).all(), 'Nonfinite transported covariance.')
            # Optional audit path only: no eigendecomposition or host read.
            _, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
            _async_need((info == 0).all(), 'Transported covariance is not positive definite.')
        ray = centers-self._camera_centers[frame]
        ray_norm = torch.linalg.vector_norm(ray, dim=-1, keepdim=True)
        _async_need(torch.isfinite(centers).all() & torch.isfinite(ray_norm).all() & (ray_norm > 0).all(),
                    'Nonfinite center or camera coincides with a Gaussian.')
        directions = torch.einsum('nji,nj->ni', Q, ray/ray_norm)
        return dict(frame_id=frame_id, reference_frame_id=self.reference_frame_id,
                    gaussian_ids=self.gaussian_ids, centers=centers, quats=quaternions,
                    scales=scales, log_scales=log_scales, covariance=covariance,
                    reference_view_directions=directions, rotations=Q, motion_quats=q_motion,
                    motion_scales=a, support_vertex_residual=self._residuals[frame, support],
                    motion_support_triangle_ids=support.clone(), training_admitted=False)

    __call__ = forward

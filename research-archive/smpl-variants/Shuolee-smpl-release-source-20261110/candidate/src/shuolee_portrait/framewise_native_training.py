"""Bounded shared-reference Gaussian training with injected native rendering.

One free reference field is transported into each named, frozen frame. This is
an executable trainer, not a physical-calibration verdict. It intentionally has
no densification, support rebinding, old-subject initialization or image warp.
The first objective is foreground-normalized L1 plus valid-pixel alpha L1; it
is explicitly not equivalent to the older L1+SSIM implementation.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from avatar_core.utils.sh_utils import C0, eval_sh
from .frame_pose_contract import Bundle
from .free_clothing_transport import _array_sha
from .free_clothing_torch import FreeClothingTorch

SCHEMA = 'shuolee-framewise-native-training/1'
PARAMETERS = ('xyz', 'quats', 'log_scales', 'opacity_logits', 'sh')


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def validate_reference_representation(quats, log_scales):
    """Reject numeric underflow/overflow, including the final saved step."""
    norm = torch.linalg.vector_norm(quats, dim=-1)
    need(bool(torch.isfinite(norm).all()) and bool((norm > 0).all()), 'Quaternion norm must be finite and positive.')
    scales = log_scales.exp()
    variance = scales.square()
    need(bool(torch.isfinite(scales).all()) and bool((scales > 0).all()) and
         bool(torch.isfinite(variance).all()) and bool((variance > 0).all()) and
         bool(torch.isfinite(scales.reciprocal()).all()), 'Scale/covariance representation overflow or underflow.')


def save_exclusive(path, payload):
    path = Path(path)
    need(path.parent.is_dir(), 'Output parent directory must already exist.')
    need(not path.exists(), 'Refuse to overwrite a checkpoint/reference state.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix='.tmp') as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    initial_points: int = 200000
    sh_degree: int = 3
    xyz_lr: float = .00016
    quaternion_lr: float = .001
    log_scale_lr: float = .005
    opacity_lr: float = .05
    sh_lr: float = .0025
    adam_eps: float = 1e-15
    alpha_weight: float = .1
    rgb_foreground_threshold: float = .99

    def validate(self, purpose):
        need(type(self.max_steps) is int and 0 < self.max_steps <= 18000, 'Explicit bounded max_steps in 1..18000 required.')
        need(type(self.initial_points) is int and self.initial_points > 0, 'Positive fixed initial_points required.')
        need(purpose == 'synthetic_contract_test' or self.initial_points == 200000,
             'Real-person runs require the declared 200000 points; no silent reduction.')
        need(type(self.sh_degree) is int and 0 <= self.sh_degree <= 3, 'SH degree must be 0..3.')
        for key in ('xyz_lr', 'quaternion_lr', 'log_scale_lr', 'opacity_lr', 'sh_lr', 'adam_eps'):
            value = getattr(self, key)
            need(type(value) in (int, float) and np.isfinite(value) and value > 0, 'Invalid ' + key)
        need(np.isfinite(self.alpha_weight) and self.alpha_weight >= 0, 'Invalid alpha weight.')
        need(np.isfinite(self.rgb_foreground_threshold) and 0 < self.rgb_foreground_threshold <= 1,
             'Explicit RGB foreground threshold in (0,1] required.')


def fresh_reference_parameters(xyz_ref, *, initial_scale, initial_opacity, initial_rgb,
                               sh_degree=3, dtype=torch.float32, device='cpu'):
    """Build fresh parameters from free clothing seeds, without loading weights.

    Scale is an explicit scalar world radius or N-by-3 array. This helper does
    not infer scene scale, bind centers to SMPL or allocate per-face quotas.
    """
    need(dtype in (torch.float32, torch.float64), 'Float32/64 required.')
    xyz = torch.as_tensor(xyz_ref, dtype=dtype, device=device).detach().clone()
    need(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0 and bool(torch.isfinite(xyz).all()), 'Finite free XYZ required.')
    n = len(xyz)
    radius = torch.as_tensor(initial_scale, dtype=dtype, device=device)
    if radius.ndim == 0:
        radius = radius.expand(n, 3)
    need(radius.shape == (n, 3) and bool(torch.isfinite(radius).all()) and bool((radius > 0).all()), 'Positive explicit scale required.')
    need(np.isfinite(initial_opacity) and 0 < initial_opacity < 1, 'Initial opacity must be strictly between zero and one.')
    rgb = torch.as_tensor(initial_rgb, dtype=dtype, device=device)
    if rgb.shape == (3,):
        rgb = rgb.expand(n, 3)
    need(rgb.shape == (n, 3) and bool(torch.isfinite(rgb).all()) and bool(((rgb >= 0) & (rgb <= 1)).all()), 'Initial RGB must be in [0,1].')
    need(type(sh_degree) is int and 0 <= sh_degree <= 3, 'SH degree must be 0..3.')
    q = torch.zeros((n, 4), dtype=dtype, device=device); q[:, 0] = 1.
    sh = torch.zeros((n, 3, (sh_degree + 1)**2), dtype=dtype, device=device)
    sh[:, :, 0] = (rgb - .5) / C0
    return dict(xyz=xyz, quats=q, log_scales=radius.log().clone(),
                opacity_logits=torch.full((n,), float(np.log(initial_opacity / (1-initial_opacity))), dtype=dtype, device=device), sh=sh)


def foreground_native_loss(rgb, alpha, target_rgb, target_alpha, rgb_valid, alpha_valid, *, alpha_weight):
    """HWC/HW L1, normalized by valid foreground/alpha counts independently.

    RGB outside rgb_valid has exactly zero effect and gradient. Valid background
    may supervise alpha; explicitly occluded pixels do not supervise either.
    """
    need(rgb.ndim == 3 and rgb.shape[-1] == 3 and alpha.shape == rgb.shape[:2], 'Expected HWC RGB and HW alpha.')
    need(target_rgb.shape == rgb.shape and target_alpha.shape == alpha.shape, 'Target dimensions differ.')
    need(rgb_valid.shape == alpha.shape and alpha_valid.shape == alpha.shape and
         rgb_valid.dtype == torch.bool and alpha_valid.dtype == torch.bool, 'Explicit boolean validity masks required.')
    for t in (alpha, target_rgb, target_alpha, rgb_valid, alpha_valid):
        need(t.device == rgb.device, 'All loss inputs must share a device.')
    need(alpha.dtype == target_rgb.dtype == target_alpha.dtype == rgb.dtype, 'Loss dtypes differ.')
    need(not target_rgb.requires_grad and not target_alpha.requires_grad, 'Targets must be frozen.')
    need(bool(rgb_valid.any()) and bool(alpha_valid.any()), 'No valid RGB foreground or alpha supervision.')
    need(bool((~rgb_valid | alpha_valid).all()), 'RGB supervision cannot include occluded/invalid alpha pixels.')
    need(bool((~rgb_valid | (target_alpha > 0)).all()), 'RGB supervision must be inside target foreground.')
    rgb_loss = (rgb[rgb_valid] - target_rgb[rgb_valid]).abs().mean()
    alpha_loss = (alpha[alpha_valid] - target_alpha[alpha_valid]).abs().mean()
    return dict(loss=rgb_loss + alpha_weight * alpha_loss, rgb_l1=rgb_loss, alpha_l1=alpha_loss,
                rgb_valid_pixels=rgb_valid.sum(), alpha_valid_pixels=alpha_valid.sum())


@dataclass(frozen=True)
class PreparedObservation:
    frame_id: str
    binding: str
    rgb: torch.Tensor
    foreground: torch.Tensor
    rgb_valid: torch.Tensor
    alpha_valid: torch.Tensor
    _guards: tuple

    def assert_unchanged(self):
        for name, identity, version in self._guards:
            value = getattr(self, name)
            need(isinstance(value, torch.Tensor) and id(value) == identity and value._version == version,
                 'Observation tensor mutated or replaced: ' + name)


class ReferenceGaussians(nn.Module):
    def __init__(self, values, *, count, degree, device, dtype):
        super().__init__()
        need(isinstance(values, Mapping) and set(values) == set(PARAMETERS), 'Complete fresh reference parameters required.')
        shapes = dict(xyz=(count, 3), quats=(count, 4), log_scales=(count, 3), opacity_logits=(count,), sh=(count, 3, (degree+1)**2))
        for name in PARAMETERS:
            raw = values[name]
            need(not isinstance(raw, torch.Tensor) or not raw.requires_grad, 'Initialization must be fresh detached values, not old trainable parameters.')
            t = torch.as_tensor(raw, dtype=dtype, device=device).detach().clone()
            need(tuple(t.shape) == shapes[name] and bool(torch.isfinite(t).all()), 'Invalid initial ' + name)
            self.register_parameter(name, nn.Parameter(t))
        validate_reference_representation(self.quats, self.log_scales)


class FramewiseNativeTrainer:
    def __init__(self, bundle, consumer, initial, *, gaussian_ids, frame_specs,
                 render_native, render_contract, renderer_provenance,
                 initialization_provenance, run_id, config):
        need(type(bundle) is Bundle and type(consumer) is FreeClothingTorch, 'Validated bundle and fixed Torch consumer required.')
        need(isinstance(config, TrainingConfig), 'Explicit TrainingConfig required.')
        config.validate(bundle.purpose)
        need(isinstance(run_id, str) and bool(run_id.strip()), 'Explicit run_id required.')
        need(callable(render_native) and callable(getattr(render_contract, 'manifest', None)), 'Native callable and explicit render contract required.')
        render_manifest = render_contract.manifest()
        need(render_manifest.get('pixels') == 'pixel_corners' and render_manifest.get('gaussian_evaluation') == 'ray-3d',
             'The injected renderer must declare pixel-corner ray-3D evaluation.')
        need(isinstance(renderer_provenance, Mapping) and bool(renderer_provenance), 'Renderer version/source provenance required.')
        need(isinstance(initialization_provenance, Mapping) and
             initialization_provenance.get('kind') in ('fresh_clothing_surface', 'synthetic_known') and
             initialization_provenance.get('old_subject_checkpoint_loaded') is False,
             'Fresh initialization provenance must explicitly exclude old-subject checkpoints.')
        need(bundle.purpose == 'synthetic_contract_test' or initialization_provenance['kind'] == 'fresh_clothing_surface',
             'Real input requires fresh clothing-surface initialization.')
        bundle.assert_unchanged()
        b_report, c_report = bundle.report(), consumer.report()
        ids = tuple(gaussian_ids)
        need(ids == consumer.gaussian_ids and len(ids) == config.initial_points, 'Gaussian identity/order/count mismatch.')
        need(set(b_report['frame_ids']) == set(consumer.frame_ids) and bundle.reference_frame_id == consumer.reference_frame_id,
             'Bundle and motion frames/reference differ.')
        need(isinstance(frame_specs, Mapping) and set(frame_specs) == set(consumer.frame_ids), 'Frame specs must exactly cover registered frames.')
        vertices = np.stack([bundle.frame(f).vertices_world for f in consumer.frame_ids])
        source_hash = c_report['source_report']['input_array_sha256']
        need(_array_sha(vertices) == source_hash['vertices_world'] and _array_sha(bundle.faces) == source_hash['faces'],
             'Consumer source mesh/topology differs from the frame bundle.')
        self.bundle, self.consumer, self.config = bundle, consumer, config
        self.frame_ids, self.gaussian_ids = tuple(consumer.frame_ids), ids
        self.reference_frame_id = bundle.reference_frame_id
        self._render_native, self._render_contract = render_native, copy.deepcopy(render_contract)
        self._render_manifest_json = canonical(render_manifest)
        self._sources = {str(Path(r['path']).resolve()): r['sha256'] for r in b_report['sources']}
        self._frames, camera_records = {}, {}
        for i, frame_id in enumerate(self.frame_ids):
            frame, spec = bundle.frame(frame_id), frame_specs[frame_id]
            need(isinstance(spec, Mapping) and set(spec) == {'target_wh', 'radial', 'tangential', 'thin_prism', 'alpha_valid_path', 'alpha_valid_sha256', 'validity_note'},
                 'Frame spec requires size, explicit 6+2+4 coefficients and a hashed validity mask/note.')
            need(isinstance(spec['validity_note'], str) and bool(spec['validity_note'].strip()), 'Validity/occlusion semantics must be described.')
            # The current Bundle and initializer are rectified-pinhole. A Brown
            # residual here would change the training rays without changing the
            # initializer's projection. A future residual bundle must be explicit.
            for name, count in (('radial', 6), ('tangential', 2), ('thin_prism', 4)):
                raw_coeffs = np.asarray(spec[name])
                need(raw_coeffs.shape == (count,) and raw_coeffs.dtype.kind in 'iuf' and
                     np.isfinite(raw_coeffs).all() and np.all(raw_coeffs == 0),
                     'Current rectified-pinhole bundle requires explicit zero residual coefficients: ' + name)
            pix = frame.for_size(spec['target_wh'])
            need(np.array_equal(frame.w2c, np.asarray(c_report['normalized_w2c'][i])), 'Bundle E and normalized motion camera differ.')
            need(pix['K_integer_centers'][0, 1] == 0 and pix['K_integer_centers'][1, 0] == 0,
                 'Current native backend requires zero skew; do not silently discard it.')
            path = Path(spec['alpha_valid_path']).resolve()
            need(path.is_file() and sha_file(path) == spec['alpha_valid_sha256'], 'Validity-mask file/hash mismatch.')
            self._sources[str(path)] = spec['alpha_valid_sha256']
            camera_records[frame_id] = dict(frame_id=frame_id, source_frame=frame.source_frame, pose_index=frame.pose_index,
                image=str(frame.image), mask=str(frame.mask), native_wh=list(frame.wh), target_wh=list(pix['wh']),
                K_integer_centers=pix['K_integer_centers'].tolist(), K_renderer_corners=pix['K_renderer_corners'].tolist(),
                native_to_output=pix['native_to_output'].tolist(), w2c=frame.w2c.tolist(),
                radial=list(spec['radial']), tangential=list(spec['tangential']), thin_prism=list(spec['thin_prism']),
                alpha_valid_path=str(path), alpha_valid_sha256=spec['alpha_valid_sha256'], validity_note=spec['validity_note'],
                validity_resampling='All BILINEAR footprint contributors must be valid; float invalid coverage equals zero.')
            record = camera_records[frame_id]
            data = {}
            for name, shape, raw in (('K', (1, 3, 3), [record['K_renderer_corners']]), ('E', (1, 4, 4), [record['w2c']]),
                    ('radial', (1, 6), [spec['radial']]), ('tangential', (1, 2), [spec['tangential']]), ('thin_prism', (1, 4), [spec['thin_prism']])):
                value = torch.tensor(raw, dtype=consumer.dtype, device=consumer.device)
                need(tuple(value.shape) == shape and bool(torch.isfinite(value).all()), 'Invalid frozen camera tensor: ' + name)
                data[name] = value
            data['guards'] = tuple((name, id(value), value._version) for name, value in data.items())
            data['record'] = canonical(record)
            self._frames[frame_id] = data
        self.field = ReferenceGaussians(initial, count=config.initial_points, degree=config.sh_degree, device=consumer.device, dtype=consumer.dtype)
        self._initial_hashes = {name: tensor_sha(getattr(self.field, name)) for name in PARAMETERS}
        self.optimizer = self._new_optimizer()
        need(not self.optimizer.state, 'A new run must have empty Adam state.')
        self._optimizer_options = tuple((g['name'], g['lr'], g['eps'], g['betas']) for g in self.optimizer.param_groups)
        self._parameter_guards = tuple((name, id(getattr(self.field, name)), tuple(getattr(self.field, name).shape)) for name in PARAMETERS)
        self._steps, self._failed, self.render_calls = 0, False, 0
        self._view_counts = {frame_id: 0 for frame_id in self.frame_ids}
        binding = dict(schema=SCHEMA, run_id=run_id, bundle=b_report, motion_consumer=c_report,
            gaussian_ids=list(ids), reference_frame_id=self.reference_frame_id,
            config=asdict(config), cameras=camera_records, renderer_contract=render_manifest,
            renderer_provenance=dict(renderer_provenance), initialization_provenance=dict(initialization_provenance),
            initial_parameter_sha256=self._initial_hashes, parameter_order=list(PARAMETERS),
            device_type=consumer.device.type, dtype=str(consumer.dtype), objective='foreground-normalized L1 plus valid-alpha L1; no SSIM',
            physical_camera_accepted=False, formal_training_admitted=False, execution_kind='bounded_research',
            fixed_layout=True, automatic_densification=False, nearest_rebinding=False, optimizer='Adam')
        self._binding_json = canonical(binding)
        self.binding_sha256 = hashlib.sha256(self._binding_json.encode()).hexdigest()

    @property
    def step_count(self):
        return self._steps

    def _new_optimizer(self):
        rates = (self.config.xyz_lr, self.config.quaternion_lr, self.config.log_scale_lr, self.config.opacity_lr, self.config.sh_lr)
        return torch.optim.Adam([dict(params=[getattr(self.field, name)], lr=lr, name=name) for name, lr in zip(PARAMETERS, rates)],
                                lr=0., eps=self.config.adam_eps, foreach=False)

    def _guards(self):
        need(not self._failed, 'This trainer failed; inspect its evidence before a new run.')
        need(canonical(self._render_contract.manifest()) == self._render_manifest_json, 'Renderer contract changed.')
        for name, identity, shape in self._parameter_guards:
            t = getattr(self.field, name)
            need(id(t) == identity and tuple(t.shape) == shape and t.device == self.consumer.device and t.dtype == self.consumer.dtype,
                 'Trainable parameter identity/layout/device changed: ' + name)
        need([g['name'] for g in self.optimizer.param_groups] == list(PARAMETERS) and
             all(len(g['params']) == 1 and g['params'][0] is getattr(self.field, name) for g, name in zip(self.optimizer.param_groups, PARAMETERS)),
             'Optimizer and shared parameter identity differ.')
        need(tuple((g['name'], g['lr'], g['eps'], g['betas']) for g in self.optimizer.param_groups) == self._optimizer_options,
             'Optimizer options changed outside the bounded configuration.')

    def _camera(self, frame_id):
        need(isinstance(frame_id, str) and frame_id in self._frames, 'Unknown frame identity.')
        data = self._frames[frame_id]
        for name, identity, version in data['guards']:
            value = data[name]
            need(id(value) == identity and value._version == version, 'Frozen camera tensor changed: ' + name)
        return data, json.loads(data['record'])

    def load_observation(self, frame_id):
        """Read hashed source PNG/JPEG and explicit validity, then resize once."""
        _, record = self._camera(frame_id)
        arrays = {}
        size = tuple(record['target_wh'])
        for name, path_key, mode in (('rgb', 'image', 'RGB'), ('foreground', 'mask', 'L'), ('alpha_valid', 'alpha_valid_path', 'L')):
            path = str(Path(record[path_key]).resolve())
            need(path in self._sources and sha_file(path) == self._sources[path], 'Observation source changed: ' + path)
            with Image.open(path) as image:
                image.load()
                need(image.size == tuple(record['native_wh']), 'Observation source dimensions differ.')
                raw = np.asarray(image.convert(mode))
                if name == 'alpha_valid':
                    need(np.isin(raw, [0, 255]).all(), 'Validity image must contain exactly binary 0/255 decisions.')
                    # Nearest-valid is unsafe when RGB/alpha are filtered: one
                    # occluded input can contribute to a different valid output.
                    invalid = Image.fromarray((raw == 0).astype(np.float32))
                    raw = np.array(invalid.resize(size, Image.Resampling.BILINEAR), copy=True) == 0
                else:
                    raw = np.array(image.convert(mode).resize(size, Image.Resampling.BILINEAR), copy=True)
            tensor = torch.tensor(raw, device=self.consumer.device)
            arrays[name] = tensor if name == 'alpha_valid' else tensor.to(dtype=self.consumer.dtype) / 255.
        arrays['rgb_valid'] = (arrays['foreground'] >= self.config.rgb_foreground_threshold) & arrays['alpha_valid']
        need(bool(arrays['rgb_valid'].any()) and bool(arrays['alpha_valid'].any()), 'No valid foreground/alpha at target size.')
        guards = tuple((name, id(tensor), tensor._version) for name, tensor in arrays.items())
        return PreparedObservation(frame_id, self.binding_sha256, **arrays, _guards=guards)

    def render(self, frame_id):
        self._guards()
        data, record = self._camera(frame_id)
        with torch.autocast(device_type=self.consumer.device.type, enabled=False):
            posed = self.consumer(frame_id, self.field.xyz, self.field.quats, self.field.log_scales,
                                  gaussian_ids=self.gaussian_ids, return_covariance=False)
            colors = (eval_sh(self.config.sh_degree, self.field.sh, posed['reference_view_directions']) + .5).clamp_min(0)
            self.render_calls += 1
            rgb, alpha, info = self._render_native(posed['centers'], posed['quats'], posed['scales'], self.field.opacity_logits.sigmoid(), colors,
                data['E'], data['K'], *record['target_wh'], contract=self._render_contract,
                radial_coeffs=data['radial'], tangential_coeffs=data['tangential'], thin_prism_coeffs=data['thin_prism'], sh_degree=None)
        width, height = record['target_wh']
        need(rgb.shape == (1, height, width, 3) and alpha.shape == (1, height, width, 1), 'Native renderer output shape mismatch.')
        need(rgb.dtype == alpha.dtype == self.consumer.dtype and rgb.device == alpha.device == self.consumer.device,
             'Native renderer output dtype/device mismatch.')
        need(bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(alpha).all()), 'Nonfinite native image/coverage.')
        return dict(rgb=rgb[0], alpha=alpha[0, :, :, 0], native_info=info, frame_id=frame_id,
                    reference_frame_id=self.reference_frame_id)

    def step(self, frame_id, observation=None):
        self._guards()
        need(self._steps < self.config.max_steps, 'Registered step budget exhausted.')
        observation = self.load_observation(frame_id) if observation is None else observation
        need(type(observation) is PreparedObservation and observation.frame_id == frame_id and
             observation.binding == self.binding_sha256, 'Observation/frame/run binding mismatch.')
        observation.assert_unchanged()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            image = self.render(frame_id)
            with torch.autocast(device_type=self.consumer.device.type, enabled=False):
                losses = foreground_native_loss(image['rgb'], image['alpha'], observation.rgb, observation.foreground,
                    observation.rgb_valid, observation.alpha_valid, alpha_weight=self.config.alpha_weight)
            need(bool(torch.isfinite(losses['loss'])), 'Nonfinite training loss.')
            losses['loss'].backward()
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.field.parameters()), 'Missing/nonfinite shared-parameter gradient.')
            self.optimizer.step()
            need(all(bool(torch.isfinite(p).all()) for p in self.field.parameters()), 'Nonfinite post-update parameter.')
            validate_reference_representation(self.field.quats, self.field.log_scales)
            self._steps += 1; self._view_counts[frame_id] += 1
            return dict(step=self._steps, frame_id=frame_id, reference_frame_id=self.reference_frame_id,
                        **{key: value.detach() for key, value in losses.items()},
                        rgb=image['rgb'].detach(), alpha=image['alpha'].detach(), quality_accepted=False)
        except BaseException:
            self._failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    def report(self):
        return dict(schema=SCHEMA, binding_sha256=self.binding_sha256, run_id=json.loads(self._binding_json)['run_id'],
            frame_ids=list(self.frame_ids), reference_frame_id=self.reference_frame_id, steps=self._steps,
            max_steps=self.config.max_steps, view_counts=dict(self._view_counts), render_calls=self.render_calls,
            gaussian_count=len(self.gaussian_ids), failed=self._failed, optimizer_state_entries=len(self.optimizer.state),
            objective='foreground-normalized L1 plus valid-alpha L1; no SSIM',
            physical_camera_accepted=False, quality_accepted=False, formal_training_admitted=False)

    def reference_state(self):
        self._guards()
        with torch.no_grad():
            validate_reference_representation(self.field.quats, self.field.log_scales)
            q = self.field.quats / torch.linalg.vector_norm(self.field.quats, dim=-1, keepdim=True)
            values = dict(xyz=self.field.xyz, quats=q, scales=self.field.log_scales.exp(),
                          opacity=self.field.opacity_logits.sigmoid(), sh=self.field.sh)
            need(all(bool(torch.isfinite(v).all()) for v in values.values()), 'Invalid named-reference export.')
            return dict(schema=SCHEMA + '/reference', binding_sha256=self.binding_sha256, frame_id=self.reference_frame_id,
                gaussian_ids=list(self.gaussian_ids), step=self._steps, coordinate_system='opencv_world_meters',
                parameters=cpu_copy(values), provenance=json.loads(self._binding_json),
                posed_clouds_averaged=False, training_quality_accepted=False)

    def save_reference_state(self, path):
        return save_exclusive(path, self.reference_state())

    def save_checkpoint(self, path):
        self._guards(); self.bundle.assert_unchanged()
        for name, digest in self._sources.items():
            need(sha_file(name) == digest, 'Checkpoint source changed: ' + name)
        payload = dict(schema=SCHEMA + '/checkpoint', binding_json=self._binding_json,
            binding_sha256=self.binding_sha256, step=self._steps, view_counts=dict(self._view_counts),
            field=cpu_copy(self.field.state_dict()), optimizer=cpu_copy(self.optimizer.state_dict()))
        self._validate_checkpoint(payload)
        return save_exclusive(path, payload)

    def _validate_checkpoint(self, payload):
        need(isinstance(payload, dict) and set(payload) == {'schema', 'binding_json', 'binding_sha256', 'step', 'view_counts', 'field', 'optimizer'},
             'Unknown or incomplete training checkpoint; do not infer missing units/frames.')
        need(payload['schema'] == SCHEMA + '/checkpoint' and payload['binding_json'] == self._binding_json and
             payload['binding_sha256'] == self.binding_sha256, 'Checkpoint frame/camera/scale/layout/config/run binding differs.')
        step = payload['step']
        need(type(step) is int and 0 <= step <= self.config.max_steps, 'Invalid checkpoint step count.')
        counts = payload['view_counts']
        need(isinstance(counts, dict) and set(counts) == set(self.frame_ids) and
             all(type(v) is int and v >= 0 for v in counts.values()) and sum(counts.values()) == step, 'Checkpoint view counts differ.')
        need(isinstance(payload['field'], dict) and set(payload['field']) == set(PARAMETERS), 'Checkpoint parameter names differ.')
        for name, expected in self.field.state_dict().items():
            value = payload['field'][name]
            need(isinstance(value, torch.Tensor) and value.dtype == expected.dtype and value.shape == expected.shape and
                 bool(torch.isfinite(value).all()), 'Invalid checkpoint field: ' + name)
        validate_reference_representation(payload['field']['quats'], payload['field']['log_scales'])
        optim = payload['optimizer']
        need(isinstance(optim, dict) and set(optim) == {'state', 'param_groups'}, 'Invalid Adam checkpoint.')
        expected_groups = self._new_optimizer().state_dict()['param_groups']
        need(optim['param_groups'] == expected_groups, 'Adam parameter groups/rates/order changed.')
        state = optim['state']
        need(isinstance(state, dict) and set(state) == (set() if step == 0 else set(range(len(PARAMETERS)))), 'Adam state count/layout differs.')
        for index, item in state.items():
            need(isinstance(item, dict) and set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unknown Adam state fields.')
            tick = item['step']
            need(isinstance(tick, torch.Tensor) and tick.numel() == 1 and bool(torch.isfinite(tick).all()) and float(tick) == step,
                 'Adam step differs from trainer step.')
            parameter = payload['field'][PARAMETERS[index]]
            for key in ('exp_avg', 'exp_avg_sq'):
                value = item[key]
                need(isinstance(value, torch.Tensor) and value.shape == parameter.shape and value.dtype == parameter.dtype and
                     bool(torch.isfinite(value).all()), 'Invalid Adam tensor: ' + key)
            need(bool((item['exp_avg_sq'] >= 0).all()), 'Adam variance cannot be negative.')

    def load_checkpoint(self, path, *, resume_same_run=False):
        """Explicit same-run recovery only, never initialize from another subject."""
        self._guards()
        need(resume_same_run is True and self._steps == 0 and not self.optimizer.state, 'Recovery requires an untouched fresh instance and explicit same-run request.')
        need({name: tensor_sha(getattr(self.field, name)) for name in PARAMETERS} == self._initial_hashes, 'Fresh initialization was changed before recovery.')
        self.bundle.assert_unchanged()
        for name, digest in self._sources.items():
            need(sha_file(name) == digest, 'Recovery source changed: ' + name)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        self._validate_checkpoint(payload)
        replacement = self._new_optimizer()
        replacement.load_state_dict(payload['optimizer'])
        self.field.load_state_dict(payload['field'], strict=True)
        self.optimizer = replacement
        self._steps, self._view_counts = payload['step'], dict(payload['view_counts'])
        return self.report()

"""Explicit common camera-length conversion; no scale estimation or admission.

SMPL-X camera-space vertices are in model metres. An SfM translation cannot
be subtracted from those vertices until its common unit has been declared.
This module converts a declared length unit; it cannot establish that all
camera baselines share a valid gauge or that the supplied factor is measured.
The same factor must also be applied to any SfM points used elsewhere.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np


def validate_camera_length_contract(contract):
    """Return an immutable copy of the explicit source-unit declaration.

    ``evidence`` is a nonempty provenance string, not a calibration verdict.
    A scalar factor is required even for sfm_units with an explicit factor 1.
    No implicit unit, per-frame scale, or acceptance flag is supported.
    """
    if not isinstance(contract, Mapping):
        raise ValueError('Explicit camera_length_contract is required before using model metres.')
    fields = {'translation_unit', 'meters_per_camera_unit', 'basis', 'evidence'}
    if set(contract) != fields:
        raise ValueError('camera_length_contract requires exactly translation_unit, '
                         'meters_per_camera_unit, basis and evidence; no per-frame scales.')
    unit = contract['translation_unit']
    if unit not in ('meters', 'sfm_units'):
        raise ValueError('Camera translation_unit must be meters or sfm_units.')
    factor = contract['meters_per_camera_unit']
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, Real):
        raise ValueError('meters_per_camera_unit must be one explicit common scalar, not a per-frame array.')
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('meters_per_camera_unit must be positive and finite.')
    if unit == 'meters' and factor != 1.:
        raise ValueError('Camera translations declared meters require meters_per_camera_unit=1.')
    basis = contract['basis']
    if basis not in ('synthetic_known', 'measured', 'model_conditioned'):
        raise ValueError('Camera length basis must be synthetic_known, measured or model_conditioned.')
    evidence = contract['evidence']
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Camera length evidence must be a nonempty provenance string.')
    return MappingProxyType(dict(translation_unit=unit, meters_per_camera_unit=factor,
                                 basis=basis, evidence=evidence))


def _readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True, eq=False)
class CameraWorldUnits:
    source_w2c: np.ndarray
    w2c_meters: np.ndarray
    contract: Mapping

    def report(self):
        return dict(camera_length_contract=dict(self.contract),
                    normalized_translation_unit='meters',
                    coordinate_system='opencv_world_meters',
                    conversion='source w2c translation multiplied by one common meters_per_camera_unit; R unchanged',
                    sfm_points_require_same_scale=True,
                    physical_camera_accepted=False, training_admitted=False,
                    scale_estimated_here=False, common_sfm_gauge_validated=False)


def normalize_camera_w2c(w2c, camera_length_contract):
    """Convert only translation to model metres, retaining the source E.

    Accept one 4x4 E or a nonempty Nx4x4 batch. Arrays use immutable bytes
    backing; mutating the caller's input or declaration cannot change them.
    Rotations must already be proper rigid rotations. K is never consumed or
    changed. This is a unit conversion, not re-estimation of extrinsics.
    """
    contract = validate_camera_length_contract(camera_length_contract)
    try:
        source = np.asarray(w2c)
        if source.dtype.kind not in 'iuf' or not (
                source.shape == (4, 4) or
                (source.ndim == 3 and source.shape[0] > 0 and source.shape[1:] == (4, 4))):
            raise ValueError('Expected one numeric 4x4 w2c or nonempty Nx4x4 batch.')
        source = source.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Invalid camera w2c for explicit length conversion.') from error
    if not np.isfinite(source).all():
        raise ValueError('Camera w2c contains nonfinite values.')
    R = source[..., :3, :3]
    if (not np.allclose(source[..., 3, :], [0., 0., 0., 1.], atol=1e-12, rtol=0) or
            not np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-6, rtol=0) or
            not np.allclose(np.linalg.det(R), 1., atol=1e-6, rtol=0)):
        raise ValueError('Camera w2c must be proper rigid extrinsics before length conversion.')
    normalized = source.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        normalized[..., :3, 3] *= contract['meters_per_camera_unit']
    if not np.isfinite(normalized).all():
        raise ValueError('Camera translation overflows during conversion to meters.')
    return CameraWorldUnits(_readonly(source), _readonly(normalized), contract)

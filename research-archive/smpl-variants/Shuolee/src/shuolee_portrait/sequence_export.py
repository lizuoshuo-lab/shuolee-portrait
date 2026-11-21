"""Standard 3DGS export of one named reference pose and same-world cameras.

Camera records retain the original sequence viewpoints. Other frames have
different subject poses; their photographs are not multiview ground truth for
this static reference model. No axis conversion or pose averaging is performed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .frame_pose_contract import Bundle, _camera


SCHEMA = 'shuolee-reference-export/1'
REFERENCE_SCHEMA = 'shuolee-framewise-native-training/1/reference'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _array(raw, shape, name):
    device = getattr(raw, 'device', None)
    _need(device is None or str(device) == 'cpu', 'Export requires CPU state: ' + name)
    _need(not getattr(raw, 'requires_grad', False), 'Export requires detached state: ' + name)
    value = np.asarray(raw)
    _need(value.dtype in (np.dtype('float32'), np.dtype('float64')) and
          value.shape == shape and np.isfinite(value).all(), 'Invalid reference ' + name)
    return value.copy()


def _prepare(state, bundle):
    _need(type(bundle) is Bundle, 'A validated frame-pose Bundle is required.')
    bundle.assert_unchanged()
    _need(isinstance(state, dict) and state.get('schema') == REFERENCE_SCHEMA,
          'Expected a trainer reference_state result.')
    _need(state.get('coordinate_system') == 'opencv_world_meters' and
          state.get('frame_id') == bundle.reference_frame_id,
          'Reference frame/coordinate system differs from the bundle.')
    _need(type(state.get('step')) is int and state['step'] > 0,
          'Reference export requires at least one completed training step.')
    _need(state.get('posed_clouds_averaged') is False, 'Posed clouds cannot be averaged into a reference export.')
    ids = state.get('gaussian_ids')
    _need(isinstance(ids, (list, tuple)) and bool(ids) and
          all(isinstance(x, str) and bool(x.strip()) for x in ids) and len(set(ids)) == len(ids),
          'Unique ordered Gaussian identities are required.')
    provenance = state.get('provenance')
    _need(isinstance(provenance, dict) and
          hashlib.sha256(_canonical(provenance).encode()).hexdigest() == state.get('binding_sha256'),
          'Trainer provenance/binding hash differs.')
    source_bundle = provenance.get('bundle', {})
    _need(isinstance(source_bundle, dict) and source_bundle == bundle.report() and
          provenance.get('reference_frame_id') == bundle.reference_frame_id and
          provenance.get('gaussian_ids') == list(ids),
          'Trainer sequence/video/reference identity differs from the bundle.')
    values = state.get('parameters')
    _need(isinstance(values, dict) and set(values) == {'xyz', 'quats', 'scales', 'opacity', 'sh'},
          'Complete reference parameters are required.')
    n = len(ids)
    sh_shape = tuple(getattr(values['sh'], 'shape', ()))
    _need(len(sh_shape) == 3 and sh_shape[:2] == (n, 3) and sh_shape[2] in (1, 4, 9, 16),
          'SH must have shape N x 3 x (degree+1)^2 for degree 0..3.')
    shapes = dict(xyz=(n, 3), quats=(n, 4), scales=(n, 3), opacity=(n,), sh=sh_shape)
    parameters = {key: _array(values[key], shape, key) for key, shape in shapes.items()}
    scales, opacity, q = parameters['scales'], parameters['opacity'], parameters['quats']
    _need(np.all(scales > 0), 'Reference scales must be positive.')
    _need(np.all((opacity > 0) & (opacity < 1)), 'Reference opacity must lie strictly inside (0,1).')
    tolerance = 32 * np.finfo(q.dtype).eps
    _need(np.all(np.abs(np.linalg.norm(q.astype(np.float64), axis=1) - 1) <= tolerance),
          'Reference quaternions must already be unit wxyz rotations.')
    with np.errstate(over='ignore', under='ignore'):
        stored_scales, stored_opacity = scales.astype(np.float32), opacity.astype(np.float32)
    _need(np.isfinite(stored_scales).all() and np.all(stored_scales > 0) and
          np.all((stored_opacity > 0) & (stored_opacity < 1)),
          'Physical scale/opacity cannot be represented by float32 Gaussian readers.')
    return parameters, int(np.sqrt(sh_shape[2])) - 1


def _ply_fields(parameters):
    xyz, sh = parameters['xyz'], parameters['sh']
    fields = {axis: xyz[:, i] for i, axis in enumerate(('x', 'y', 'z'))}
    fields.update({axis: np.zeros(len(xyz)) for axis in ('nx', 'ny', 'nz')})
    fields.update({f'f_dc_{i}': sh[:, i, 0] for i in range(3)})
    rest = sh[:, :, 1:].reshape(len(xyz), -1)
    fields.update({f'f_rest_{i}': rest[:, i] for i in range(rest.shape[1])})
    opacity = parameters['opacity'].astype(np.float64)
    fields['opacity'] = np.log(opacity) - np.log1p(-opacity)
    scale = np.log(parameters['scales'].astype(np.float64))
    fields.update({f'scale_{i}': scale[:, i] for i in range(3)})
    fields.update({f'rot_{i}': parameters['quats'][:, i] for i in range(4)})
    records = np.empty(len(xyz), dtype=[(key, '<f4') for key in fields])
    with np.errstate(over='ignore', under='ignore'):
        for key, value in fields.items():
            records[key] = value
    _need(all(np.isfinite(records[key]).all() for key in fields),
          'PLY attributes overflow float32 storage.')
    return records, fields


def _camera_records(bundle):
    records = []
    for i, name in enumerate(bundle.report()['frame_ids']):
        frame = bundle.frame(name)
        K, E = _camera(frame.K_integer_centers, frame.w2c, name)
        corner = K.copy(); corner[:2, 2] += .5
        inverse = np.eye(4)
        inverse[:3, :3] = E[:3, :3].T
        inverse[:3, 3] = -E[:3, :3].T @ E[:3, 3]
        _need(np.allclose(E @ inverse, np.eye(4), atol=2e-6, rtol=0), 'Camera inversion failed: ' + name)
        records.append(dict(id=i, img_name=f'camera_{i:06d}', width=frame.wh[0], height=frame.wh[1],
            position=inverse[:3, 3].tolist(), rotation=inverse[:3, :3].tolist(),
            fx=float(corner[0, 0]), fy=float(corner[1, 1]), cx=float(corner[0, 2]), cy=float(corner[1, 2]),
            K_integer_centers=K.tolist(), K_renderer_corners=corner.tolist(),
            w2c=E.tolist(), c2w=inverse.tolist(), pixelConvention='pixel_corners', projectionModel='pinhole',
            frame_id=name, source_frame=frame.source_frame, pose_index=frame.pose_index,
            sequence_id=bundle.sequence_id, video_sha256=bundle.video_sha256,
            reference_frame_id=bundle.reference_frame_id,
            is_reference_frame=name == bundle.reference_frame_id,
            camera_role='original_sequence_viewpoint',
            static_reference_multiview_ground_truth=False))
    return records


def _write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def export_reference(state, bundle, output_dir):
    """Write standard PLY/cameras and a receipt after independent file readback.

    All source geometry is already in the named reference world. PLY stores
    float32 xyz, wxyz rotation, log radii, opacity logits and channel-major SH.
    Successful serialization does not establish reconstruction quality.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    _need(not output_dir.exists(), 'Export output must be new; existing results are preserved.')
    parameters, degree = _prepare(state, bundle)
    records, fields = _ply_fields(parameters)
    cameras = _camera_records(bundle)
    output_dir.mkdir(parents=True, exist_ok=False)
    ply_path, camera_path = output_dir/'point_cloud.ply', output_dir/'cameras.json'
    with ply_path.open('xb') as stream:
        PlyData([PlyElement.describe(records, 'vertex')], text=False, byte_order='<').write(stream)
    _write_json(camera_path, cameras)
    loaded = PlyData.read(ply_path, mmap=False)
    _need(not loaded.text and loaded.byte_order == '<' and len(loaded.elements) == 1 and
          loaded.elements[0].name == 'vertex', 'Expected binary little-endian vertex PLY.')
    vertex = loaded['vertex'].data
    _need(vertex.dtype.names == records.dtype.names and len(vertex) == len(records), 'PLY attribute layout differs.')
    errors = {}
    for key, source in fields.items():
        _need(np.array_equal(vertex[key], records[key]) and np.isfinite(vertex[key]).all(),
              'PLY field roundtrip failed: ' + key)
        errors[key] = float(np.max(np.abs(vertex[key].astype(np.float64) - np.asarray(source, dtype=np.float64))))
    decoded_cameras = json.loads(camera_path.read_text(encoding='utf-8'))
    _need(decoded_cameras == cameras, 'Camera JSON numerical roundtrip failed.')
    bundle.assert_unchanged()
    restored_scales = np.exp(np.column_stack([vertex[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    logits = vertex['opacity'].astype(np.float64)
    restored_opacity = np.exp(-np.logaddexp(0, -logits))
    _need(np.isfinite(restored_scales).all() and np.all(restored_scales > 0) and
          np.all((restored_opacity > 0) & (restored_opacity < 1)), 'PLY physical representation roundtrip failed.')
    receipt = dict(schema=SCHEMA, state='export_complete', reference_frame_id=bundle.reference_frame_id,
        sequence_id=bundle.sequence_id, video_sha256=bundle.video_sha256, step=state['step'],
        gaussian_count=len(records), sh_degree=degree, camera_count=len(cameras),
        coordinate_system='opencv_world_meters', coordinate_transform_applied=False, posed_clouds_averaged=False,
        quaternion_order='wxyz', scale_storage='log_standard_deviation', opacity_storage='logit',
        sh_storage='DC RGB, then channel-major remaining coefficients',
        binding_sha256=state['binding_sha256'],
        gaussian_order_sha256=hashlib.sha256(_canonical(list(state['gaussian_ids'])).encode()).hexdigest(),
        ply=dict(path=ply_path.name, sha256=_sha(ply_path)), cameras=dict(path=camera_path.name, sha256=_sha(camera_path)),
        format_passed=True, roundtrip_max_abs_by_field=errors,
        scale_roundtrip_max_relative=float(np.max(np.abs(restored_scales / parameters['scales'] - 1))),
        opacity_roundtrip_max_abs=float(np.max(np.abs(restored_opacity - parameters['opacity']))),
        camera_role='original_sequence_viewpoints; other frames contain different subject poses',
        static_reference_multiview_ground_truth=False, quality_accepted=False, physical_calibration_accepted=False)
    _write_json(output_dir/'result.json', receipt)
    return receipt

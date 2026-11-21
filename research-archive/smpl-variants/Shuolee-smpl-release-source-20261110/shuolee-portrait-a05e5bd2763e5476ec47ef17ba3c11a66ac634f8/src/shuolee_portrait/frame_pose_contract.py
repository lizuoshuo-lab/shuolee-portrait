"""Hash-bound native frame/camera/pose association; no fitting or rendering.

The manifest paths are relative to the manifest. Image and mask paths are
relative to the frame-roster JSON. Absolute paths are also supported. Pose
rows are joined by explicit frame identity, never by their array position.
Successful loading proves this data contract, not physical calibration or
permission to train. In particular, this module does not construct the old
zero-skew CameraProj2 or connect a renderer backend.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from .alignment_inputs import validate_camera
from .camera_pixel_contract import resize_matrix


SCHEMA = 'shuolee-frame-pose/1'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, name):
    _need(isinstance(value, str) and bool(value.strip()),
          name + ' must be a nonempty string.')
    return value


def _digest(value, name):
    _need(isinstance(value, str) and len(value) == 64 and
          all(c in '0123456789abcdef' for c in value),
          name + ' must be a lowercase SHA256 digest.')
    return value


def _sha(path):
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
    except OSError as error:
        raise ValueError('Cannot read bound input: ' + str(path)) from error
    return digest.hexdigest()


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError('Cannot read JSON input: ' + str(path)) from error
    _need(isinstance(value, dict), 'Expected a JSON object: ' + str(path))
    return value


def _required(value, keys, name):
    _need(isinstance(value, dict), name + ' must be an object.')
    missing = sorted(set(keys) - value.keys())
    _need(not missing, name + ' is missing: ' + ', '.join(missing))


def _path(base, value, name):
    _text(value, name)
    try:
        result = (base / value).resolve()
    except (OSError, ValueError) as error:
        raise ValueError('Invalid ' + name) from error
    _need(result.is_file(), 'Missing ' + name + ': ' + str(result))
    return result


def _numeric(value, shape, name):
    try:
        result = np.asarray(value)
        _need(result.dtype.kind in 'iuf' and result.shape == shape,
              name + ' must be a numeric array of shape ' + str(shape))
        result = result.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Invalid ' + name + ': expected ' + str(shape)) from error
    _need(np.isfinite(result).all(), name + ' contains nonfinite values.')
    return result


def _readonly(value):
    """Bytes backing also prevents callers from re-enabling write access."""
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _size(value, name):
    try:
        array = np.asarray(value)
        valid = (array.shape == (2,) and array.dtype.kind in 'iu' and
                 np.all(array > 0))
    except (ValueError, TypeError):
        valid = False
    _need(valid, name + ' must contain two positive integer width/height values.')
    return tuple(int(x) for x in array)


def _camera(K, E, name):
    K = _numeric(K, (3, 3), name + ' K_integer_centers')
    E = _numeric(E, (4, 4), name + ' w2c')
    # Reuse the shared convention check, then enforce this stricter schema.
    validate_camera(dict(K_integer_centers=K, w2c=E))
    _need(np.allclose(K[2], [0., 0., 1.], atol=1e-12, rtol=0) and
          abs(K[1, 0]) <= 1e-12 and K[0, 0] > 0 and K[1, 1] > 0,
          name + ' K must be upper-triangular pinhole K with positive fx/fy; skew is allowed.')
    R = E[:3, :3]
    _need(np.allclose(E[3], [0., 0., 0., 1.], atol=1e-12, rtol=0) and
          np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0) and
          abs(np.linalg.det(R) - 1.) <= 1e-7,
          name + ' w2c must be a proper rigid transform, without scale or reflection.')
    return K, E


def _string_array(value, shape, name):
    _need(isinstance(value, np.ndarray) and value.shape == shape and
          value.dtype.kind in 'US', name + ' must be a string array of shape ' + str(shape))
    try:
        values = [x.decode('utf-8') if isinstance(x, bytes) else str(x)
                  for x in value.reshape(-1).tolist()]
    except UnicodeError as error:
        raise ValueError(name + ' must contain UTF-8 strings.') from error
    for item in values:
        _text(item, name)
    return values


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    frame_id: str
    source_frame: int
    pose_index: int
    image: Path
    mask: Path
    wh: tuple
    K_integer_centers: np.ndarray
    w2c: np.ndarray
    vertices_world: np.ndarray

    def for_size(self, target_wh):
        """Return explicit half-pixel resize and renderer-corner conventions.

        This computes coordinates only. It does not resize the image/mask or
        instantiate a renderer. Full K (including skew) is retained.
        """
        wh = _size(target_wh, 'target_wh')
        transform = resize_matrix(self.wh, wh)
        K = transform @ self.K_integer_centers
        corners = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
        return dict(frame_id=self.frame_id, source_frame=self.source_frame,
                    pose_index=self.pose_index, wh=wh,
                    native_to_output=_readonly(transform),
                    K_integer_centers=_readonly(K),
                    K_renderer_corners=_readonly(corners @ K),
                    w2c=self.w2c, pixel_convention='integer_centers',
                    renderer_pixel_convention='pixel_corners',
                    renderer_backend_connected=False)


@dataclass(frozen=True, slots=True, eq=False)
class Bundle:
    sequence_id: str
    video_sha256: str
    purpose: str
    reference_frame_id: str
    shared_betas: np.ndarray
    faces: np.ndarray
    _frames: object = field(repr=False)
    _bound_files: tuple = field(repr=False)

    def frame(self, frame_id):
        _text(frame_id, 'frame_id')
        _need(frame_id in self._frames, 'Unknown frame identity: ' + frame_id)
        return self._frames[frame_id]

    @property
    def reference(self):
        return self.frame(self.reference_frame_id)

    def assert_unchanged(self):
        for path, expected in self._bound_files:
            _need(_sha(path) == expected, 'Bound input changed: ' + str(path))

    def report(self):
        return dict(schema=SCHEMA, contract_valid=True, purpose=self.purpose,
                    sequence_id=self.sequence_id, video_sha256=self.video_sha256,
                    coordinate_system='opencv_world_meters',
                    pixel_domain='rectified_pinhole', pixel_convention='integer_centers',
                    reference_frame_id=self.reference_frame_id,
                    frames=len(self._frames), frame_ids=list(self._frames),
                    pose_indices={key: value.pose_index for key, value in self._frames.items()},
                    vertices_per_frame=int(self.reference.vertices_world.shape[0]),
                    faces=int(self.faces.shape[0]), shared_beta_shape=list(self.shared_betas.shape),
                    sources=[dict(path=str(path), sha256=digest)
                             for path, digest in self._bound_files],
                    association='explicit frame_id; exact source frame, image digest, K and w2c',
                    video_file_verified=False,
                    physical_calibration_validated=False, training_admitted=False,
                    renderer_backend_connected=False, image_pixels_modified=False)


def load_frame_pose_bundle(manifest_path):
    """Validate and load a complete immutable frame/pose roster on CPU."""
    try:
        manifest_path = Path(manifest_path).resolve()
    except (TypeError, OSError, ValueError) as error:
        raise ValueError('Invalid frame-pose manifest path.') from error
    manifest_hash = _sha(manifest_path)
    manifest = _read_json(manifest_path)
    _required(manifest, ('schema', 'purpose', 'sequence_id', 'video_sha256',
                        'coordinate_system', 'pixel_domain', 'pixel_convention',
                        'reference_frame_id', 'frames', 'poses'), 'Manifest')
    _need(manifest['schema'] == SCHEMA, 'Unsupported frame-pose schema.')
    _need(manifest['purpose'] in ('real', 'synthetic_contract_test'),
          'Manifest purpose must be real or synthetic_contract_test.')
    _need(manifest['coordinate_system'] == 'opencv_world_meters',
          'Declare opencv_world_meters; implicit axis or scale conversion is forbidden.')
    _need(manifest['pixel_domain'] == 'rectified_pinhole' and
          manifest['pixel_convention'] == 'integer_centers',
          'Require rectified_pinhole integer_centers; no implicit distortion conversion.')
    sequence = _text(manifest['sequence_id'], 'sequence_id')
    video = _digest(manifest['video_sha256'], 'video_sha256')
    reference_id = _text(manifest['reference_frame_id'], 'reference_frame_id')
    bound = {manifest_path: manifest_hash}

    def bind(path, expected, name):
        expected = _digest(expected, name + ' sha256')
        _need(path not in bound or bound[path] == expected,
              'Conflicting hashes for the same input: ' + str(path))
        actual = _sha(path)
        _need(actual == expected, name + ' SHA256 mismatch: ' + str(path))
        bound[path] = actual

    def source(desc, name):
        _required(desc, ('path', 'sha256'), name)
        path = _path(manifest_path.parent, desc['path'], name + ' path')
        bind(path, desc['sha256'], name)
        return path

    frames_desc = manifest['frames']
    _required(frames_desc, ('path', 'sha256', 'rows_key'), 'frames')
    _need(frames_desc['rows_key'] == 'records', 'frames.rows_key must be records.')
    frames_path = source(frames_desc, 'frames')
    frames_json = _read_json(frames_path)
    rows = frames_json.get('records')
    _need(isinstance(rows, list) and len(rows) > 0, 'Frame records must be a nonempty list.')
    row_data, source_indices = {}, set()
    for row in rows:
        _required(row, ('frame', 'source_frame', 'sequence_id', 'video_sha256', 'wh',
                        'image', 'source_sha256', 'mask', 'mask_sha256',
                        'K_integer_centers', 'w2c'), 'Frame record')
        _need(not any('beta' in key.lower() for key in row),
              'Frame records cannot supply per-frame betas; use only shared_betas in poses.')
        frame_id = _text(row['frame'], 'Frame record frame')
        _need(frame_id not in row_data, 'Duplicate frame identity: ' + frame_id)
        index = row['source_frame']
        _need(type(index) is int and index >= 0, 'source_frame must be a nonnegative integer.')
        _need(index not in source_indices, 'Duplicate source_frame in one video: ' + str(index))
        source_indices.add(index)
        _need(row['sequence_id'] == sequence and row['video_sha256'] == video,
              'Frame sequence/video provenance differs: ' + frame_id)
        wh = _size(row['wh'], 'Frame ' + frame_id + ' wh')
        K, E = _camera(row['K_integer_centers'], row['w2c'], 'Frame ' + frame_id)
        if 'K' in row:
            validate_camera(row)  # Optional legacy renderer K must be exactly +0.5.
        image = _path(frames_path.parent, row['image'], 'image')
        mask = _path(frames_path.parent, row['mask'], 'mask')
        for path, digest, kind in ((image, row['source_sha256'], 'image'),
                                  (mask, row['mask_sha256'], 'mask')):
            bind(path, digest, frame_id + ' ' + kind)
            try:
                with Image.open(path) as opened:
                    actual_wh = tuple(opened.size)
                    # Dimensions alone can come from an intact header even
                    # when the bound pixel payload is truncated or corrupt.
                    opened.load()
            except (OSError, ValueError) as error:
                raise ValueError('Cannot decode ' + kind + ' input: ' + str(path)) from error
            _need(actual_wh == wh, frame_id + ' ' + kind + ' dimensions differ from native wh.')
        row_data[frame_id] = (index, image, mask, wh, K, E, row['source_sha256'])
    _need(reference_id in row_data, 'reference_frame_id is absent from the frame roster.')

    poses_path = source(manifest['poses'], 'poses')
    try:
        with np.load(poses_path, allow_pickle=False) as archive:
            poses = {key: archive[key].copy() for key in archive.files}
    except (OSError, TypeError, ValueError, EOFError, BadZipFile) as error:
        raise ValueError('Invalid poses NPZ (object arrays/pickle are forbidden).') from error
    _required(poses, ('frame_ids', 'source_frames', 'image_sha256', 'sequence_id',
                      'video_sha256', 'vertices_world', 'faces', 'shared_betas',
                      'K_integer_centers', 'w2c'), 'poses NPZ')
    _need(not any('beta' in key.lower() and key != 'shared_betas' for key in poses),
          'Per-frame/alternative betas are forbidden; require only shared_betas (1,10).')
    n = len(row_data)
    pose_ids = _string_array(poses['frame_ids'], (n,), 'poses frame_ids')
    _need(len(set(pose_ids)) == n and set(pose_ids) == set(row_data),
          'Pose and frame identities must be unique and have exactly the same complete set.')
    _need(_string_array(poses['sequence_id'], (), 'poses sequence_id')[0] == sequence and
          _string_array(poses['video_sha256'], (), 'poses video_sha256')[0] == video,
          'Pose sequence/video provenance differs from the manifest.')
    pose_hashes = _string_array(poses['image_sha256'], (n,), 'poses image_sha256')
    indices = poses['source_frames']
    _need(indices.shape == (n,) and indices.dtype.kind in 'iu' and np.all(indices >= 0),
          'poses source_frames must contain nonnegative integer indices.')
    vertices = poses['vertices_world']
    _need(vertices.ndim == 3 and vertices.shape[0] == n and
          vertices.shape[1] >= 3 and vertices.shape[2] == 3,
          'vertices_world must have shape N x V x 3 with V >= 3.')
    vertices = _numeric(vertices, vertices.shape, 'vertices_world')
    faces = poses['faces']
    _need(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0 and
          faces.dtype.kind in 'iu', 'faces must be a nonempty integer F x 3 array.')
    _need(np.all(faces >= 0) and np.all(faces < vertices.shape[1]) and
          np.all(faces[:, 0] != faces[:, 1]) and np.all(faces[:, 0] != faces[:, 2]) and
          np.all(faces[:, 1] != faces[:, 2]),
          'faces contain out-of-range or degenerate vertex indices.')
    betas = _numeric(poses['shared_betas'], (1, 10), 'shared_betas')
    pose_K = _numeric(poses['K_integer_centers'], (n, 3, 3), 'poses K_integer_centers')
    pose_E = _numeric(poses['w2c'], (n, 4, 4), 'poses w2c')
    lookup = {frame_id: i for i, frame_id in enumerate(pose_ids)}
    frames = {}
    for frame_id, (index, image, mask, wh, K, E, image_hash) in row_data.items():
        i = lookup[frame_id]
        _digest(pose_hashes[i], 'poses image_sha256')
        _need(int(indices[i]) == index and pose_hashes[i] == image_hash,
              'Pose source frame/image digest differs for frame ' + frame_id)
        _camera(pose_K[i], pose_E[i], 'Pose ' + frame_id)
        _need(np.array_equal(pose_K[i], K) and np.array_equal(pose_E[i], E),
              'Pose K/w2c differs for frame ' + frame_id + '; no silent camera replacement.')
        frames[frame_id] = Frame(frame_id, index, i, image, mask, wh,
                                 _readonly(K), _readonly(E), _readonly(vertices[i]))
    result = Bundle(sequence, video, manifest['purpose'], reference_id,
                    _readonly(betas), _readonly(faces), MappingProxyType(frames),
                    tuple(bound.items()))
    result.assert_unchanged()
    return result

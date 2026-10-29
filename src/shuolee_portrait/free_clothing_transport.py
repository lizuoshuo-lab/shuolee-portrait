"""CPU motion support for one free reference-world Gaussian field.

The directed triangle basis and square-root area ratio define a local
similarity field. This field is NOT an exact interpolant of general sheared
triangles. Its vertex residual is exposed, not hidden by a quality threshold.
Support faces describe motion, not a position target or material identity.
There is no nearest-face refresh, per-face point quota, optimizer or renderer.
Clone, split and prune require a newly explicit Gaussian-ID/support mapping.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch


SCHEMA = 'shuolee-free-clothing-transport/1'
ASSUMPTION = 'directed_triangle_basis_and_sqrt_area_ratio_local_similarity'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(value, name):
    _need(isinstance(value, (list, tuple)), name + ' must be an explicit list or tuple.')
    result = tuple(value)
    _need(bool(result) and all(isinstance(x, str) and bool(x.strip()) for x in result),
          name + ' must contain nonempty string identities.')
    _need(len(set(result)) == len(result), 'Duplicate ' + name)
    return result


def _sha_text(value, name):
    _need(isinstance(value, str) and len(value) == 64 and
          all(c in '0123456789abcdef' for c in value), name + ' must be a lowercase SHA256.')
    return value


def _readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _numeric(value, shape, name):
    try:
        a = np.asarray(value)
        _need(a.shape == shape and a.dtype.kind in 'iuf', 'Invalid ' + name + ' shape/type.')
        a = a.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Invalid numeric ' + name) from error
    _need(np.isfinite(a).all(), 'Nonfinite ' + name)
    return a


def _array_sha(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps([a.dtype.str, list(a.shape)], separators=(',', ':')).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
    except OSError as error:
        raise ValueError('Cannot read transport input: ' + str(path)) from error
    return digest.hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _geometry(vertices, faces):
    triangle = vertices[:, faces]
    edge = triangle[:, :, 1] - triangle[:, :, 0]
    other = triangle[:, :, 2] - triangle[:, :, 0]
    cross = np.cross(edge, other)
    length = np.linalg.norm(edge, axis=-1)
    twice_area = np.linalg.norm(cross, axis=-1)
    _need(np.isfinite(length).all() and np.isfinite(twice_area).all() and
          np.all(length > 0) and np.all(twice_area > 0),
          'Degenerate/nonfinite triangle geometry; no epsilon repair is allowed.')
    u = edge / length[..., None]
    normal = cross / twice_area[..., None]
    v = np.cross(normal, u)
    # Ordered vertex identities define orientation. A 180-degree pose rotation
    # is valid; the sign of a cross-frame normal dot product is not a gate.
    basis = np.stack((u, v, normal), axis=-1)
    _need(np.isfinite(basis).all() and
          np.allclose(np.swapaxes(basis, -1, -2) @ basis, np.eye(3), rtol=0, atol=1e-10) and
          np.allclose(np.linalg.det(basis), 1., rtol=0, atol=1e-10),
          'Triangle basis is not numerically a proper oriented frame.')
    centroid = triangle.mean(axis=2)
    _need(np.isfinite(centroid).all(), 'Nonfinite triangle centroid.')
    return triangle, basis, twice_area * .5, centroid


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FreeClothingTransport:
    frame_ids: tuple
    reference_frame_id: str
    gaussian_ids: tuple
    translation_unit: str
    vertices_world: np.ndarray
    faces: np.ndarray
    motion_support_triangle_ids: np.ndarray
    _source_items: tuple
    _rotations: np.ndarray
    _scales: np.ndarray
    _centroids: np.ndarray
    _residuals: np.ndarray
    _bound_files: tuple

    def __init__(self, *, frame_ids, reference_frame_id, vertices_world, faces,
                 motion_support_triangle_ids, gaussian_ids, frame_source_sha256,
                 translation_unit=None):
        _need(translation_unit == 'meters', 'Explicit translation_unit="meters" is required; units are not estimated.')
        ids = _ids(frame_ids, 'frame_ids')
        gids = _ids(gaussian_ids, 'gaussian_ids')
        _need(isinstance(reference_frame_id, str) and reference_frame_id in ids,
              'An existing named reference_frame_id is required.')
        _need(isinstance(frame_source_sha256, Mapping) and set(frame_source_sha256) == set(ids),
              'Source SHA256 map must exactly cover the frame identities.')
        sources = tuple((key, _sha_text(frame_source_sha256[key], 'frame source')) for key in ids)
        raw = np.asarray(vertices_world)
        _need(raw.ndim == 3 and raw.shape[0] == len(ids) and raw.shape[1] >= 3 and raw.shape[2] == 3,
              'vertices_world must have shape frame x vertex x 3.')
        vertices = _numeric(raw, raw.shape, 'vertices_world')
        raw_faces = np.asarray(faces)
        _need(raw_faces.ndim == 2 and raw_faces.shape[1] == 3 and len(raw_faces) > 0 and
              raw_faces.dtype.kind in 'iu', 'faces must be a nonempty integer F x 3 array.')
        _need(np.all(raw_faces >= 0) and np.all(raw_faces < vertices.shape[1]) and
              np.all(raw_faces[:, 0] != raw_faces[:, 1]) and
              np.all(raw_faces[:, 0] != raw_faces[:, 2]) and
              np.all(raw_faces[:, 1] != raw_faces[:, 2]), 'Invalid or repeated triangle vertex indices.')
        face = raw_faces.astype(np.int64, copy=True)
        support = np.asarray(motion_support_triangle_ids)
        _need(support.shape == (len(gids),) and support.dtype.kind in 'iu' and
              np.all(support >= 0) and np.all(support < len(face)),
              'Each named Gaussian requires one valid explicit motion-support triangle ID.')
        support = support.astype(np.int64, copy=True)
        triangle, basis, area, centroids = _geometry(vertices, face)
        reference = ids.index(reference_frame_id)
        rotations = basis @ np.swapaxes(basis[reference], -1, -2)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            scales = np.sqrt(area / area[reference])
        _need(np.isfinite(scales).all() and np.all(scales > 0), 'Nonpositive/nonfinite local similarity scale.')
        # Preserve exact identity at the selected reference, rather than relying
        # on the floating point result of F_ref @ F_ref.T.
        rotations[reference] = np.eye(3)
        scales[reference] = 1.
        offset = triangle[reference] - centroids[reference, :, None]
        predicted = (scales[:, :, None, None] *
                     np.einsum('nfij,fkj->nfki', rotations, offset) + centroids[:, :, None])
        difference = predicted - triangle
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            residuals = np.sqrt(np.mean(np.sum(difference * difference, axis=-1), axis=-1)) / np.sqrt(area)
        _need(np.isfinite(residuals).all(), 'Nonfinite local-similarity vertex residual.')
        residuals[reference] = 0.
        for name, value in dict(frame_ids=ids, reference_frame_id=reference_frame_id,
                                gaussian_ids=gids, translation_unit='meters',
                                vertices_world=_readonly(vertices), faces=_readonly(face),
                                motion_support_triangle_ids=_readonly(support), _source_items=sources,
                                _rotations=_readonly(rotations), _scales=_readonly(scales),
                                _centroids=_readonly(centroids), _residuals=_readonly(residuals),
                                _bound_files=()).items():
            object.__setattr__(self, name, value)

    @property
    def frame_source_sha256(self):
        return dict(self._source_items)

    def _selection(self, frame_id, gaussian_ids):
        _need(isinstance(frame_id, str) and frame_id in self.frame_ids, 'Unknown frame identity.')
        _need(_ids(gaussian_ids, 'gaussian_ids') == self.gaussian_ids,
              'Gaussian identity/order differs from the fixed support mapping; rebuild it explicitly.')
        self.assert_unchanged()
        frame = self.frame_ids.index(frame_id)
        reference = self.frame_ids.index(self.reference_frame_id)
        support = self.motion_support_triangle_ids
        return (frame, self._rotations[frame, support], self._scales[frame, support],
                self._centroids[reference, support], self._centroids[frame, support],
                self._residuals[frame, support])

    def _metadata(self, frame_id):
        return dict(frame_id=frame_id, reference_frame_id=self.reference_frame_id,
                    source_sha256=self.frame_source_sha256[frame_id],
                    gaussian_ids=self.gaussian_ids, translation_unit='meters',
                    transport_assumption=ASSUMPTION, physical_quality_accepted=False,
                    training_admitted=False, support_is_material_correspondence=False)

    def transport_numpy(self, frame_id, centers_ref, covariance_ref, camera_center, *, gaussian_ids):
        _, Q, a, ref, posed, residual = self._selection(frame_id, gaussian_ids)
        n = len(self.gaussian_ids)
        centers = _numeric(centers_ref, (n, 3), 'centers_ref')
        covariance = _numeric(covariance_ref, (n, 3, 3), 'covariance_ref')
        camera = _numeric(camera_center, (3,), 'camera_center')
        tolerance = 32 * np.finfo(covariance.dtype).eps
        _need(np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=0, rtol=tolerance),
              'Reference covariance must be symmetric.')
        _need(np.all(np.linalg.eigvalsh(covariance) > 0), 'Reference covariance must be positive definite.')
        if frame_id == self.reference_frame_id:
            moved, transported = centers.copy(), covariance.copy()
        else:
            moved = a[:, None] * np.einsum('nij,nj->ni', Q, centers - ref) + posed
            transported = a[:, None, None] ** 2 * (Q @ covariance @ np.swapaxes(Q, -1, -2))
        direction = moved - camera
        length = np.linalg.norm(direction, axis=-1)
        _need(np.isfinite(moved).all() and np.isfinite(transported).all() and
              np.isfinite(length).all() and np.all(length > 0),
              'Nonfinite transport or camera coincides with a Gaussian center.')
        _need(np.all(np.linalg.eigvalsh(transported) > 0),
              'Transported covariance is not positive definite; numerical underflow is not repaired.')
        view_ref = np.einsum('nji,nj->ni', Q, direction / length[:, None])
        return dict(**self._metadata(frame_id), centers=_readonly(moved), covariance=_readonly(transported),
                    reference_view_directions=_readonly(view_ref), rotations=_readonly(Q), scales=_readonly(a),
                    support_vertex_residual=_readonly(residual),
                    motion_support_triangle_ids=self.motion_support_triangle_ids)

    # Caller autocast and default-device contexts must not change the declared
    # float32/float64 CPU transport. The decorator restores their prior state.
    @torch.autocast(device_type='cpu', enabled=False)
    def transport_torch(self, frame_id, centers_ref, covariance_ref, camera_center, *, gaussian_ids):
        _, Q, a, ref, posed, residual = self._selection(frame_id, gaussian_ids)
        n = len(self.gaussian_ids)
        for tensor, shape, name in ((centers_ref, (n, 3), 'centers_ref'),
                                     (covariance_ref, (n, 3, 3), 'covariance_ref')):
            _need(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and
                  tensor.dtype in (torch.float32, torch.float64) and tuple(tensor.shape) == shape,
                  name + ' must be a float32/float64 CPU tensor with the declared Gaussian order.')
            _need(bool(torch.isfinite(tensor).all()), 'Nonfinite ' + name)
        _need(centers_ref.dtype == covariance_ref.dtype, 'Center and covariance tensor dtypes must match.')
        if isinstance(camera_center, torch.Tensor):
            _need(camera_center.device.type == 'cpu' and camera_center.dtype == centers_ref.dtype and
                  tuple(camera_center.shape) == (3,), 'camera_center must match the CPU tensor dtype and have shape (3,).')
            camera = camera_center
        else:
            camera = torch.tensor(_numeric(camera_center, (3,), 'camera_center'), dtype=centers_ref.dtype, device='cpu')
        _need(bool(torch.isfinite(camera).all()), 'Nonfinite camera_center.')
        tolerance = 32 * torch.finfo(covariance_ref.dtype).eps
        _need(torch.allclose(covariance_ref, covariance_ref.transpose(-1, -2), atol=0, rtol=tolerance),
              'Reference covariance must be symmetric.')
        _need(bool((torch.linalg.eigvalsh(covariance_ref) > 0).all()), 'Reference covariance must be positive definite.')
        convert = lambda x: torch.tensor(np.array(x, copy=True), dtype=centers_ref.dtype, device='cpu')
        qt, at, rt, pt = map(convert, (Q, a, ref, posed))
        if frame_id == self.reference_frame_id:
            moved, transported = centers_ref.clone(), covariance_ref.clone()
        else:
            moved = at[:, None] * torch.einsum('nij,nj->ni', qt, centers_ref - rt) + pt
            transported = at[:, None, None].square() * (qt @ covariance_ref @ qt.transpose(-1, -2))
        direction = moved - camera
        length = torch.linalg.vector_norm(direction, dim=-1)
        _need(bool(torch.isfinite(moved).all()) and bool(torch.isfinite(transported).all()) and
              bool(torch.isfinite(length).all()) and bool((length > 0).all()),
              'Nonfinite transport or camera coincides with a Gaussian center.')
        _need(bool((torch.linalg.eigvalsh(transported) > 0).all()),
              'Transported covariance is not positive definite; numerical underflow is not repaired.')
        view_ref = torch.einsum('nji,nj->ni', qt, direction / length[:, None])
        return dict(**self._metadata(frame_id), centers=moved, covariance=transported,
                    reference_view_directions=view_ref, rotations=qt, scales=at,
                    support_vertex_residual=convert(residual),
                    motion_support_triangle_ids=torch.tensor(self.motion_support_triangle_ids.copy(), dtype=torch.long, device='cpu'))

    def _arrays(self):
        return dict(vertices_world=self.vertices_world, faces=self.faces,
                    motion_support_triangle_ids=self.motion_support_triangle_ids,
                    frame_ids=np.asarray(self.frame_ids), gaussian_ids=np.asarray(self.gaussian_ids),
                    reference_frame_id=np.asarray(self.reference_frame_id), translation_unit=np.asarray('meters'),
                    frame_source_sha256=np.asarray([value for _, value in self._source_items]))

    def report(self):
        return dict(schema=SCHEMA, frame_ids=list(self.frame_ids), gaussian_ids=list(self.gaussian_ids),
                    reference_frame_id=self.reference_frame_id, translation_unit='meters',
                    frame_source_sha256=self.frame_source_sha256,
                    input_array_sha256={key: _array_sha(value) for key, value in self._arrays().items()},
                    transport_assumption=ASSUMPTION,
                    support_vertex_residual_definition='RMS of three mapped-minus-posed vertex distances divided by sqrt(posed triangle area)',
                    max_support_vertex_residual_by_frame={frame: float(self._residuals[i, self.motion_support_triangle_ids].max())
                                                         for i, frame in enumerate(self.frame_ids)},
                    per_triangle_quota=False, body_attraction=False,
                    learned_coordinates='free reference-world meters',
                    support_update_policy='explicit new Gaussian-ID/support mapping after clone/split/prune; no automatic nearest refresh',
                    view_direction_convention='camera to posed Gaussian center, unit length, then Q transpose to reference direction',
                    physical_quality_accepted=False, physical_camera_accepted=False, training_admitted=False,
                    source_frame_files_verified=False, common_metric_scale_verified=False,
                    independent_material_identity_verified=False, renderer_connected=False,
                    serialization_scope='geometry/support metadata only; no learned Gaussian parameters or optimizer',
                    bound_files=[dict(path=str(path), sha256=digest) for path, digest in self._bound_files])

    def assert_unchanged(self):
        for path, expected in self._bound_files:
            _need(_file_sha(path) == expected, 'Transport package changed: ' + str(path))

    def save(self, manifest_path):
        self.assert_unchanged()
        path = Path(manifest_path).resolve()
        _need(path.suffix.lower() == '.json', 'Transport manifest path must end in .json.')
        payload = path.with_suffix('.npz')
        _need(not path.exists() and not payload.exists(), 'Refuse to overwrite an existing transport package.')
        _need(path.parent.is_dir(), 'Transport destination directory must already exist.')
        arrays = self._arrays()
        with payload.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        metadata = self.report()
        metadata['payload'] = dict(path=payload.name, sha256=_file_sha(payload))
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        return path

    @classmethod
    def load(cls, manifest_path):
        path = Path(manifest_path).resolve()
        manifest_sha = _file_sha(path)
        try:
            metadata = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_json_object)
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError('Invalid transport JSON manifest.') from error
        _need(isinstance(metadata, dict) and metadata.get('schema') == SCHEMA, 'Unsupported transport package schema.')
        descriptor = metadata.get('payload')
        _need(isinstance(descriptor, dict) and set(descriptor) == {'path', 'sha256'}, 'Explicit transport payload/hash required.')
        name = descriptor['path']
        _need(isinstance(name, str) and name and Path(name).name == name and Path(name).suffix.lower() == '.npz',
              'Transport payload must be a sibling NPZ filename.')
        payload = path.parent / name
        payload_sha = _file_sha(payload)
        _need(payload_sha == _sha_text(descriptor['sha256'], 'payload sha256'), 'Transport payload SHA256 mismatch.')
        try:
            with np.load(payload, allow_pickle=False) as archive:
                _need(len(archive.files) == len(set(archive.files)),
                      'Duplicate transport NPZ fields are forbidden.')
                arrays = {key: archive[key].copy() for key in archive.files}
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
            raise ValueError('Invalid transport NPZ; pickle/object arrays are forbidden: ' + str(error)) from error
        required = {'vertices_world', 'faces', 'motion_support_triangle_ids', 'frame_ids', 'gaussian_ids',
                    'reference_frame_id', 'translation_unit', 'frame_source_sha256'}
        _need(set(arrays) == required, 'Transport NPZ fields differ from the schema.')
        def strings(key, scalar=False):
            value = arrays[key]
            _need(value.dtype.kind == 'U' and (value.shape == () if scalar else value.ndim == 1),
                  'Explicit Unicode identities required: ' + key)
            return value.item() if scalar else value.tolist()
        frame_ids = strings('frame_ids')
        sources = strings('frame_source_sha256')
        _need(len(frame_ids) == len(sources), 'Frame/source hash array lengths differ.')
        loaded = cls(frame_ids=frame_ids, gaussian_ids=strings('gaussian_ids'),
                     reference_frame_id=strings('reference_frame_id', True),
                     translation_unit=strings('translation_unit', True),
                     frame_source_sha256=dict(zip(frame_ids, sources)),
                     vertices_world=arrays['vertices_world'], faces=arrays['faces'],
                     motion_support_triangle_ids=arrays['motion_support_triangle_ids'])
        expected = loaded.report()
        _need(set(metadata) == set(expected) | {'payload'}, 'Unexpected/missing transport metadata fields.')
        # Bound-file lists are provenance for the source package only. They do
        # not substitute for the actual files being loaded and checked here.
        _need(isinstance(metadata['bound_files'], list), 'Invalid bound-file provenance.')
        for key, value in expected.items():
            if key != 'bound_files':
                _need(metadata[key] == value, 'Transport JSON/NPZ metadata mismatch: ' + key)
        object.__setattr__(loaded, '_bound_files', ((path, manifest_sha), (payload, payload_sha)))
        loaded.assert_unchanged()
        return loaded

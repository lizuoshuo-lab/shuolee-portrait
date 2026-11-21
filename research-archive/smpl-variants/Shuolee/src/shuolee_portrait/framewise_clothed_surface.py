"""CPU silhouette-envelope initialization in a named reference-world domain.

Queries are transported by fixed motion support before *each* native-mask
lookup. Known background is negative evidence; unobserved/occluded pixels,
out-of-view points and points behind the camera are not negative evidence.
The result is an observed outer envelope, not measured clothing or a material
correspondence. Free sample centres are never projected onto the body mesh.
This module owns no optimizer, renderer, body forward pass or GPU operation.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from .camera_world_units import validate_camera_length_contract
from .frame_pose_contract import Bundle, _camera
from .free_clothing_transport import FreeClothingTransport


SCHEMA = 'shuolee-framewise-clothed-surface/1'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _array_sha(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([a.dtype.str, list(a.shape)]).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    _need(type(value) is int and value >= minimum, name + ' must be an explicit integer >= ' + str(minimum))
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ReferenceGrid:
    """Inclusive regular sample axes, in the named frame's world metres."""
    reference_frame_id: str
    axes: tuple
    shape: tuple
    spacing: tuple
    size: int

    def __init__(self, axes, *, reference_frame_id, max_voxels=5_000_000):
        _need(isinstance(reference_frame_id, str) and bool(reference_frame_id.strip()),
              'An explicit named reference frame is required.')
        _integer(max_voxels, 'max_voxels')
        _need(isinstance(axes, (tuple, list)) and len(axes) == 3, 'Three regular reference axes required.')
        checked, spacing = [], []
        for raw in axes:
            a = np.asarray(raw)
            _need(a.ndim == 1 and len(a) >= 3 and a.dtype.kind in 'iuf', 'Each axis needs >=3 numeric samples.')
            a = a.astype(np.float64, copy=True)
            steps = np.diff(a)
            _need(np.isfinite(a).all() and np.all(steps > 0) and
                  np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12),
                  'Axes must be finite, increasing and uniformly spaced.')
            checked.append(_readonly(a))
            spacing.append(float(steps[0]))
        shape = tuple(len(a) for a in checked)
        count = int(np.prod(shape, dtype=np.int64))
        _need(count <= max_voxels, 'Reference grid exceeds the explicitly registered voxel budget.')
        for key, value in dict(reference_frame_id=reference_frame_id, axes=tuple(checked),
                               shape=shape, spacing=tuple(spacing), size=count).items():
            object.__setattr__(self, key, value)

    @classmethod
    def from_bounds(cls, lower, upper, *, spacing_m, reference_frame_id, max_voxels=5_000_000):
        _integer(max_voxels, 'max_voxels')
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        _need(lo.shape == hi.shape == (3,) and np.isfinite(lo).all() and np.isfinite(hi).all() and
              np.all(hi > lo), 'Finite three-dimensional increasing bounds required.')
        _need(np.isscalar(spacing_m) and np.isfinite(spacing_m) and spacing_m > 0, 'Positive spacing_m required.')
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            extent = np.ceil((hi - lo) / spacing_m) + 1
        _need(np.isfinite(extent).all() and np.all(extent >= 3) and np.all(extent <= max_voxels),
              'Bounds exceed the voxel budget or provide fewer than three samples per axis.')
        sizes = extent.astype(np.int64)
        _need(int(np.prod(sizes, dtype=object)) <= max_voxels,
              'Bounds exceed the voxel budget or provide fewer than three samples per axis.')
        return cls(tuple(lo[j] + np.arange(int(sizes[j])) * spacing_m for j in range(3)),
                   reference_frame_id=reference_frame_id, max_voxels=max_voxels)

    def chunk(self, begin, end):
        ids = np.unravel_index(np.arange(begin, end, dtype=np.int64), self.shape)
        return np.column_stack([self.axes[j][ids[j]] for j in range(3)])


@dataclass(frozen=True, slots=True, eq=False)
class VoteVolume:
    grid: ReferenceGrid
    positive_counts: np.ndarray
    valid_counts: np.ndarray
    query_support_triangle_ids: np.ndarray
    min_valid_views: int
    vote_fraction: float
    _report_json: str

    @property
    def eligible(self):
        return self.valid_counts >= self.min_valid_views

    @property
    def fractions(self):
        result = np.full(self.grid.shape, np.nan, dtype=np.float64)
        np.divide(self.positive_counts, self.valid_counts, out=result, where=self.valid_counts > 0)
        return result

    @property
    def occupied(self):
        return self.eligible & (self.fractions > self.vote_fraction)

    def report(self):
        return json.loads(self._report_json)


@dataclass(frozen=True, slots=True, eq=False)
class FramewiseSurfaceInitialization:
    xyz_ref: np.ndarray
    gaussian_ids: tuple
    transport: FreeClothingTransport
    surface_vertices_ref: np.ndarray
    surface_faces: np.ndarray
    surface_sample_face_ids: np.ndarray
    sample_positive_counts: np.ndarray
    sample_valid_counts: np.ndarray
    votes: VoteVolume
    _report_json: str

    def report(self):
        return json.loads(self._report_json)


def _prepare(bundle, grid, reference_frame_id, valid_masks, validity_notes, camera_length_contract):
    _need(type(bundle) is Bundle, 'A loaded frame-pose Bundle is required.')
    _need(type(grid) is ReferenceGrid, 'An explicit ReferenceGrid is required.')
    _need(reference_frame_id == bundle.reference_frame_id == grid.reference_frame_id,
          'Named reference identity differs between bundle, grid and requested initialization.')
    bundle.assert_unchanged()
    frame_ids = tuple(bundle.report()['frame_ids'])
    _need(isinstance(valid_masks, Mapping) and set(valid_masks) == set(frame_ids),
          'valid_masks must explicitly and exactly cover all named frames.')
    _need(isinstance(validity_notes, Mapping) and set(validity_notes) == set(frame_ids),
          'validity_notes must exactly cover all named frames with mask/occlusion provenance.')
    length = dict(validate_camera_length_contract(camera_length_contract))
    _need(length['translation_unit'] == 'meters' and length['meters_per_camera_unit'] == 1.,
          'Frame-pose Bundle already contains normalized world/metre cameras; require meters/1, '
          'never apply the upstream SfM scale to Bundle E again.')
    frames, sources, vertices = {}, {}, []
    for name in frame_ids:
        frame = bundle.frame(name)
        _need(frame.frame_id == name, 'Frame key and contained identity differ.')
        K, metric_E = _camera(frame.K_integer_centers, frame.w2c, name)
        _need(isinstance(validity_notes[name], str) and bool(validity_notes[name].strip()),
              'Each validity mask requires a nonempty provenance note.')
        valid = np.asarray(valid_masks[name])
        _need(valid.dtype == np.bool_ and valid.shape == (frame.wh[1], frame.wh[0]),
              name + ' valid mask must be native-size bool H x W; no implicit resizing.')
        with Image.open(frame.mask) as opened:
            opened.load()
            _need(opened.size == frame.wh, name + ' silhouette dimensions differ from native frame.')
            raw = np.array(opened.convert('L'))
        _need(np.isin(raw, [0, 255]).all(),
              name + ' silhouette must be an explicit binary 0/255 mask; no hidden thresholding.')
        record = dict(frame_id=name, source_frame=frame.source_frame, wh=list(frame.wh),
                      image_sha256=_file_sha(frame.image), mask_sha256=_file_sha(frame.mask),
                      valid_mask_sha256=_array_sha(valid), validity_note=validity_notes[name],
                      K_integer_centers=K.tolist(), bundle_w2c_role='already_normalized_model_units',
                      normalized_w2c=metric_E.tolist(), vertices_sha256=_array_sha(frame.vertices_world))
        sources[name] = _json_sha(record)
        frames[name] = dict(record=record, mask=_readonly(raw == 255), valid=_readonly(valid),
                            K=K, E=metric_E, wh=frame.wh)
        vertices.append(frame.vertices_world)
    # One sentinel only exposes the validated per-triangle motion tables. Grid
    # and final-Gaussian support are selected explicitly below, never rebound.
    motion = FreeClothingTransport(frame_ids=frame_ids, reference_frame_id=reference_frame_id,
        vertices_world=np.stack(vertices), faces=bundle.faces, motion_support_triangle_ids=np.array([0]),
        gaussian_ids=('motion-table-sentinel',), frame_source_sha256=sources, translation_unit='meters')
    reference_index = frame_ids.index(reference_frame_id)
    tree = cKDTree(motion._centroids[reference_index])
    bundle.assert_unchanged()
    return motion, tree, frames, length


def _move_queries(motion, frame_id, points_ref, support):
    """Centres-only evaluation of the same frozen local-similarity field."""
    if frame_id == motion.reference_frame_id:
        return points_ref.copy()
    i = motion.frame_ids.index(frame_id)
    r = motion.frame_ids.index(motion.reference_frame_id)
    moved = motion._scales[i, support, None] * np.einsum(
        'nij,nj->ni', motion._rotations[i, support], points_ref - motion._centroids[r, support])
    moved += motion._centroids[i, support]
    _need(np.isfinite(moved).all(), 'Nonfinite transported reference query.')
    return moved


def _mask_observation(points, frame):
    """Native integer-centre nearest-pixel lookup, preserving the entire K."""
    E, K = frame['E'], frame['K']
    cam = points @ E[:3, :3].T + E[:3, 3]
    q = cam @ K.T
    positive = np.zeros(len(points), dtype=np.bool_)
    observed = np.zeros(len(points), dtype=np.bool_)
    uv = np.full((len(points), 2), np.nan)
    front = np.isfinite(cam).all(1) & np.isfinite(q).all(1) & (cam[:, 2] > 0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        uv[front] = q[front, :2] / q[front, 2:]
    width, height = frame['wh']
    inside = front & np.isfinite(uv).all(1) & (uv[:, 0] >= -.5) & (uv[:, 0] < width-.5) & \
        (uv[:, 1] >= -.5) & (uv[:, 1] < height-.5)
    ids = np.flatnonzero(inside)
    # +.5 is solely nearest-index rounding, NOT conversion of K to corners.
    pix = np.floor(uv[ids] + .5).astype(np.int64)
    known = frame['valid'][pix[:, 1], pix[:, 0]]
    observed[ids] = known
    positive[ids] = known & frame['mask'][pix[:, 1], pix[:, 0]]
    return positive, observed


def _vote_queries(points_for_chunk, size, support, motion, frames, chunk_size):
    positive, valid = np.zeros(size, np.uint32), np.zeros(size, np.uint32)
    per_frame = []
    for name in motion.frame_ids:
        seen, hits = 0, 0
        for begin in range(0, size, chunk_size):
            end = min(begin + chunk_size, size)
            moved = _move_queries(motion, name, points_for_chunk(begin, end), support[begin:end])
            hit, known = _mask_observation(moved, frames[name])
            positive[begin:end] += hit
            valid[begin:end] += known
            hits += int(hit.sum())
            seen += int(known.sum())
        per_frame.append(dict(frame_id=name, observed_queries=seen, foreground_queries=hits,
                              background_queries=seen-hits, unknown_queries=size-seen))
    return positive, valid, per_frame


def _evaluate(bundle, grid, *, reference_frame_id, valid_masks, validity_notes, camera_length_contract,
              min_valid_views, vote_fraction, chunk_size):
    _integer(min_valid_views, 'min_valid_views')
    _integer(chunk_size, 'chunk_size')
    _need(type(vote_fraction) in (int, float) and np.isfinite(vote_fraction) and 0 < vote_fraction < 1,
          'vote_fraction must be explicitly fixed between zero and one; equality is outside.')
    motion, tree, frames, length = _prepare(bundle, grid, reference_frame_id, valid_masks,
                                           validity_notes, camera_length_contract)
    _need(min_valid_views <= len(motion.frame_ids), 'min_valid_views exceeds the frame count.')
    support = np.empty(grid.size, dtype=np.int64)
    for begin in range(0, grid.size, chunk_size):
        end = min(begin + chunk_size, grid.size)
        support[begin:end] = tree.query(grid.chunk(begin, end), workers=1)[1]
    positive, valid, by_frame = _vote_queries(grid.chunk, grid.size, support, motion, frames, chunk_size)
    report = dict(schema=SCHEMA, reference_frame_id=reference_frame_id, frame_ids=list(motion.frame_ids),
        geometry_unit='meters', pixel_domain='rectified_pinhole', pixel_convention='integer_centers',
        camera_length_contract=length, camera_translation_scale_applications=0,
        camera_input_role='Bundle already normalized to model units; no SfM conversion here',
        geometry_scaled_again=False, full_K_preserved=True, image_pixels_warped=False,
        grid_shape=list(grid.shape), grid_spacing_m=list(grid.spacing), grid_voxels=grid.size,
        grid_axes_sha256=[_array_sha(axis) for axis in grid.axes],
        min_valid_views=min_valid_views, vote_fraction=float(vote_fraction),
        fraction_comparison='strictly_greater', query_support_selection_count=1,
        support_policy='nearest_reference_triangle_centroid_once; motion only, no surface projection',
        query_support_sha256=_array_sha(support), per_frame=by_frame,
        source_bundle_report_sha256=_json_sha(bundle.report()),
        source_bundle_files=bundle.report()['sources'], motion_faces_sha256=_array_sha(motion.faces),
        input_frames=[frames[name]['record'] for name in motion.frame_ids],
        unknown_policy='exclude from numerator and denominator; never a negative vote',
        validity_policy='caller-supplied explicit native masks; no automatic occlusion inference',
        eligible_queries=int((valid >= min_valid_views).sum()),
        motion_vertex_residual_max=float(motion._residuals.max()),
        physical_calibration_validated=False, material_correspondence_validated=False,
        training_admitted=False, quality_accepted=False)
    volume = VoteVolume(grid, _readonly(positive.reshape(grid.shape)), _readonly(valid.reshape(grid.shape)),
                        _readonly(support), min_valid_views, float(vote_fraction),
                        json.dumps(report, sort_keys=True, allow_nan=False))
    bundle.assert_unchanged()
    return volume, motion, tree, frames


def evaluate_framewise_votes(bundle, grid, *, reference_frame_id, valid_masks, validity_notes,
                             camera_length_contract, min_valid_views, vote_fraction, chunk_size=100_000):
    """Inspect finite-budget volume votes without extracting or sampling a surface."""
    return _evaluate(bundle, grid, reference_frame_id=reference_frame_id, valid_masks=valid_masks,
        validity_notes=validity_notes, camera_length_contract=camera_length_contract,
        min_valid_views=min_valid_views, vote_fraction=vote_fraction, chunk_size=chunk_size)[0]


def extract_supported_surface(volume):
    """Historical vote-fraction proxy, retained for diagnosis, not initialization.

    Production initialization uses extract_boundary_surface below. This helper
    cannot infer pixel-square boundary locations from discrete votes alone.
    """
    _need(type(volume) is VoteVolume, 'A VoteVolume is required.')
    occupied = volume.occupied
    _need(bool(occupied.any()), 'No supported silhouette envelope; do not relax thresholds automatically.')
    _need(not any(np.take(occupied, [0, -1], axis=j).any() for j in range(3)),
          'Supported envelope reaches the grid boundary; refuse a clipped surface.')
    neighborhood = binary_dilation(occupied, structure=np.ones((3, 3, 3), dtype=bool))
    _need(not np.any(neighborhood & ~volume.eligible),
          'Unknown/under-supported voxels touch the envelope; refuse an invented closure.')
    field = np.where(volume.eligible, volume.fractions, 0.)
    verts, faces, _, _ = marching_cubes(field, level=volume.vote_fraction,
        spacing=volume.grid.spacing, allow_degenerate=False)
    verts = verts.astype(np.float64) + np.array([a[0] for a in volume.grid.axes])
    _need(len(faces) > 0, 'No extractable surface.')
    # Every output edge must have two incident faces. Unknown background must
    # not quietly generate an open mesh which is then sampled as complete.
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, multiplicity = np.unique(edges, axis=0, return_counts=True)
    _need(np.all(multiplicity == 2), 'Extracted envelope is not closed/manifold at its edges.')
    return _readonly(verts), _readonly(faces.astype(np.int64))


class NativePixelSquareBoundary:
    """Exact distances to known FG/BG interfaces, in native pixel units.

    The KD tree indexes unit segment centres. Any segment closer than a current
    distance d has its centre within d + 1/2, so a radius query gives the complete
    candidate set. Neither a centre EDT nor interpolated mask labels are used.
    Unknown and image-exterior interfaces are deliberately not added as walls.
    """
    def __init__(self, mask, valid):
        mask, valid = np.asarray(mask), np.asarray(valid)
        _need(mask.dtype == valid.dtype == np.bool_ and mask.ndim == 2 and mask.shape == valid.shape,
              'Boundary requires same-size native boolean mask and validity.')
        y0, x0 = np.nonzero(valid[:, :-1] & valid[:, 1:] & (mask[:, :-1] != mask[:, 1:]))
        y1, x1 = np.nonzero(valid[:-1, :] & valid[1:, :] & (mask[:-1, :] != mask[1:, :]))
        self.centres = np.concatenate((np.column_stack((x0 + .5, y0)),
                                       np.column_stack((x1, y1 + .5)))).astype(np.float64)
        # Vertical segments extend in y; horizontal segments extend in x.
        self.extended_axis = np.concatenate((np.ones(len(x0), np.int64), np.zeros(len(x1), np.int64)))
        self.tree = cKDTree(self.centres) if len(self.centres) else None
        self.mixed_known_labels = bool((mask & valid).any() and (~mask & valid).any())

    def distance(self, uv):
        uv = np.asarray(uv, dtype=np.float64)
        _need(uv.ndim == 2 and uv.shape[1] == 2 and np.isfinite(uv).all(), 'Finite native 2D queries required.')
        if not len(uv):
            return np.empty(0, dtype=np.float64)
        _need(self.tree is not None or not self.mixed_known_labels,
              'Foreground/background are separated only by unknown pixels; no observed boundary to interpolate.')
        if self.tree is None:
            return np.full(len(uv), np.inf)
        result = np.empty(len(uv), dtype=np.float64)
        for begin in range(0, len(uv), 4096):
            points = uv[begin:begin + 4096]
            _, nearest = self.tree.query(points, k=1, workers=1)
            delta = np.abs(points - self.centres[nearest])
            delta[np.arange(len(points)), self.extended_axis[nearest]] = np.maximum(
                delta[np.arange(len(points)), self.extended_axis[nearest]] - .5, 0.)
            best = np.linalg.norm(delta, axis=1)
            candidates = self.tree.query_ball_point(points, np.nextafter(best + .5, np.inf), workers=1)
            lengths = np.fromiter((len(item) for item in candidates), dtype=np.int64, count=len(points))
            owners = np.repeat(np.arange(len(points)), lengths)
            ids = np.concatenate(candidates).astype(np.int64)
            delta = np.abs(points[owners] - self.centres[ids])
            delta[np.arange(len(ids)), self.extended_axis[ids]] = np.maximum(
                delta[np.arange(len(ids)), self.extended_axis[ids]] - .5, 0.)
            np.minimum.at(best, owners, np.linalg.norm(delta, axis=1))
            result[begin:begin + len(points)] = best
        return result


def _boundary_margins(points, frame, boundary):
    hit, known = _mask_observation(points, frame)
    result = np.full(len(points), np.nan, dtype=np.float64)
    ids = np.flatnonzero(known)
    if len(ids):
        cam = points[ids] @ frame['E'][:3, :3].T + frame['E'][:3, 3]
        q = cam @ frame['K'].T
        distance = boundary.distance(q[:, :2] / q[:, 2:])
        result[ids] = np.where(hit[ids], distance, -distance)
    return result, known


def _required_votes(view_count, vote_fraction):
    """First integer p satisfying the original floating p/n > fraction rule.

    Computing only floor(fraction*n)+1 can disagree at floating exact ties.
    This lookup preserves the exact existing division/comparison semantics.
    """
    return np.array([0] + [int(np.searchsorted(np.arange(n+1, dtype=float) / n,
                                               vote_fraction, side='right'))
                           for n in range(1, view_count+1)], dtype=np.int64)


def _quorum_margin(margins, known, *, min_valid_views, vote_fraction):
    _need(margins.shape == known.shape and known.dtype == np.bool_, 'Margin and known arrays differ.')
    count = known.sum(axis=0)
    required = _required_votes(len(margins), vote_fraction)[count]
    eligible = count >= min_valid_views
    ordered = np.sort(np.where(known, margins, -np.inf), axis=0)
    result = np.full(margins.shape[1], np.nan)
    ids = np.flatnonzero(eligible)
    result[ids] = ordered[len(margins)-required[ids], ids]
    return result


def _vote_xyz(xyz, motion, tree, frames, volume):
    # A new geometric query gets its support once. No existing Gaussian moves
    # between supports and the per-triangle motions are never blended.
    support = tree.query(xyz, workers=1)[1].astype(np.int64)
    positive, valid, _ = _vote_queries(lambda b, e: xyz[b:e], len(xyz), support, motion, frames, 100_000)
    fraction = np.divide(positive, valid, out=np.zeros(len(xyz), dtype=float), where=valid > 0)
    eligible = valid >= volume.min_valid_views
    return eligible & (fraction > volume.vote_fraction), eligible, support


def _refine_native_edge_vertices(vertices, volume, motion, tree, frames):
    """Finite observed-crossing search; not a global topology certification."""
    origin = np.array([axis[0] for axis in volume.grid.axes])
    spacing = np.array(volume.grid.spacing)
    index = (vertices-origin) / spacing
    nearest = np.rint(index)
    # Only recognize grid-edge coordinates within float32 MC arithmetic error.
    arithmetic_bound = 8*np.finfo(np.float32).eps*max(volume.grid.shape)
    fixed = np.abs(index-nearest) <= arithmetic_bound
    eligible_vertices = np.flatnonzero(fixed.sum(axis=1) == 2)
    corrected = vertices.copy()
    audit = dict(scan_intervals=16, bisection_rounds=16,
        candidate_grid_edge_vertices=len(eligible_vertices), non_edge_or_grid_node_vertices=int(len(vertices)-len(eligible_vertices)),
        refined_vertices=0, multiple_observed_crossing_edges=0, observed_crossings=0,
        scan_edges_with_unknown=0, scan_edges_with_support_change=0,
        missing_known_bracket_vertices=0, bisection_unknown_vertices=0,
        final_inside_verified=False, subscan_crossings_certified_absent=False,
        crossing_selection='nearest registered MC proxy position; not a topology proof')
    if not len(eligible_vertices):
        return corrected, audit
    axis = np.argmin(fixed[eligible_vertices], axis=1)
    lo = nearest[eligible_vertices].astype(np.int64)
    lo[np.arange(len(lo)), axis] = np.floor(index[eligible_vertices, axis]).astype(np.int64)
    hi = lo.copy(); hi[np.arange(len(hi)), axis] += 1
    inside_grid = ((lo >= 0) & (hi < np.array(volume.grid.shape))).all(axis=1)
    eligible_vertices, axis, lo, hi = eligible_vertices[inside_grid], axis[inside_grid], lo[inside_grid], hi[inside_grid]
    begin, end = origin + lo*spacing, origin + hi*spacing
    t = np.linspace(0., 1., 17)
    scan = begin[:, None, :] + (end-begin)[:, None, :]*t[None, :, None]
    occupied, observed, support = _vote_xyz(scan.reshape(-1, 3), motion, tree, frames, volume)
    occupied, observed, support = (x.reshape(-1, 17) for x in (occupied, observed, support))
    transition = (occupied[:, :-1] != occupied[:, 1:]) & observed[:, :-1] & observed[:, 1:]
    counts = transition.sum(axis=1)
    audit.update(multiple_observed_crossing_edges=int((counts > 1).sum()), observed_crossings=int(counts.sum()),
                 scan_edges_with_unknown=int((~observed).any(axis=1).sum()),
                 scan_edges_with_support_change=int((support[:, 1:] != support[:, :-1]).any(axis=1).sum()),
                 missing_known_bracket_vertices=int((counts == 0).sum()))
    keep = np.flatnonzero(counts > 0)
    if not len(keep):
        return corrected, audit
    original_t = index[eligible_vertices, axis] - lo[np.arange(len(lo)), axis]
    distance = np.abs((t[:-1]+t[1:])[None, :]/2 - original_t[:, None])
    crossing = np.argmin(np.where(transition, distance, np.inf), axis=1)[keep]
    left, right = scan[keep, crossing], scan[keep, crossing+1]
    left_inside = occupied[keep, crossing]
    inner = np.where(left_inside[:, None], left, right)
    outer = np.where(left_inside[:, None], right, left)
    active = np.ones(len(keep), dtype=bool)
    for _ in range(16):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        mid = (inner[ids]+outer[ids])/2
        hit, seen, _ = _vote_xyz(mid, motion, tree, frames, volume)
        active[ids[~seen]] = False
        inner[ids[seen & hit]] = mid[seen & hit]
        outer[ids[seen & ~hit]] = mid[seen & ~hit]
    good = np.flatnonzero(active)
    if len(good):
        hit, seen, _ = _vote_xyz(inner[good], motion, tree, frames, volume)
        out_hit, out_seen, _ = _vote_xyz(outer[good], motion, tree, frames, volume)
        _need(bool((hit & seen & ~out_hit & out_seen).all()), 'Refined native edge lost its known inside/outside bracket.')
        corrected[eligible_vertices[keep[good]]] = inner[good]
    audit.update(refined_vertices=int(len(good)), bisection_unknown_vertices=int((~active).sum()),
                 final_inside_verified=bool(len(good)),
                 maximum_vertex_displacement_m=float(np.linalg.norm(corrected-vertices, axis=1).max()))
    return corrected, audit


def extract_boundary_surface(volume, motion, tree, frames):
    """Construct a pixel-boundary-aware finite surface; preserve original votes."""
    _need(type(volume) is VoteVolume, 'A VoteVolume is required.')
    occupied = volume.occupied
    _need(bool(occupied.any()), 'No supported silhouette envelope; do not relax thresholds automatically.')
    _need(not any(np.take(occupied, [0, -1], axis=j).any() for j in range(3)),
          'Supported envelope reaches the grid boundary; refuse a clipped surface.')
    neighborhood = binary_dilation(occupied, structure=np.ones((3, 3, 3), dtype=bool))
    _need(not np.any(neighborhood & ~volume.eligible),
          'Unknown/under-supported voxels touch the envelope; refuse an invented closure.')
    boundaries = {name: NativePixelSquareBoundary(frame['mask'], frame['valid']) for name, frame in frames.items()}
    field = np.full(volume.grid.size, np.nan)
    for begin in range(0, volume.grid.size, 100_000):
        end = min(begin+100_000, volume.grid.size)
        points = volume.grid.chunk(begin, end)
        support = volume.query_support_triangle_ids[begin:end]
        margins, known = [], []
        for name in motion.frame_ids:
            moved = _move_queries(motion, name, points, support)
            margin, seen = _boundary_margins(moved, frames[name], boundaries[name])
            margins.append(margin); known.append(seen)
        field[begin:end] = _quorum_margin(np.array(margins), np.array(known),
            min_valid_views=volume.min_valid_views, vote_fraction=volume.vote_fraction)
    field = field.reshape(volume.grid.shape)
    nonzero = volume.eligible & (field != 0)
    _need(np.array_equal((field > 0)[nonzero], occupied[nonzero]), 'Boundary field changed the frozen occupancy rule.')
    # All finite native distances fit inside the image diagonal. Infinity means
    # no finite constraint in that view; this cap preserves signs, not a margin.
    cap = 1 + max(float(np.hypot(*frame['wh'])) for frame in frames.values())
    finite = np.nan_to_num(field, nan=-cap, posinf=cap, neginf=-cap)
    verts, faces, _, _ = marching_cubes(finite, level=0., spacing=volume.grid.spacing, allow_degenerate=False)
    verts = verts.astype(np.float64) + np.array([a[0] for a in volume.grid.axes])
    verts, edge_audit = _refine_native_edge_vertices(verts, volume, motion, tree, frames)
    _need(len(faces) > 0 and np.isfinite(verts).all(), 'No finite extractable boundary surface.')
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, multiplicity = np.unique(edges, axis=0, return_counts=True)
    _need(np.all(multiplicity == 2), 'Boundary envelope is not closed/manifold at its edges.')
    tri = verts[faces]
    _need(np.all(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) > 0),
          'Boundary refinement produced a degenerate triangle; no automatic face deletion.')
    vertex_hits, vertex_known, _ = _vote_xyz(verts, motion, tree, frames, volume)
    audit = dict(method='exact_native_known_pixel_square_boundary_margin_and_bounded_oracle_edge_intersections',
        distance_units='native pixels', required_votes='first integer p whose floating p/n is strictly above vote_fraction',
        masks_or_votes_changed=False, unknown_boundary_segments_added=False,
        per_frame_known_boundary_segments={name:len(b.centres) for name,b in boundaries.items()},
        finite_grid_proxy=True, zero_margin_grid_nodes=int((volume.eligible & (field == 0)).sum()),
        vertices_meeting_original_rule=int(vertex_hits.sum()), vertices_with_sufficient_views=int(vertex_known.sum()),
        vertices_total=len(verts), edge_refinement=edge_audit,
        all_samples_pass_required_for_training=False)
    return _readonly(verts), _readonly(faces.astype(np.int64)), audit


def sample_surface_by_area(vertices, faces, *, count, seed):
    """IID area sampling; no per-face quota, attraction or point reprojection."""
    _integer(count, 'count')
    _integer(seed, 'seed', 0)
    verts, fs = np.asarray(vertices), np.asarray(faces)
    _need(verts.ndim == 2 and verts.shape[1] == 3 and np.isfinite(verts).all(), 'Invalid surface vertices.')
    _need(fs.ndim == 2 and fs.shape[1] == 3 and fs.dtype.kind in 'iu' and len(fs) > 0 and
          np.all(fs >= 0) and np.all(fs < len(verts)), 'Invalid surface face indices.')
    triangles = verts[fs]
    areas = .5 * np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                        triangles[:, 2]-triangles[:, 0]), axis=1)
    _need(np.isfinite(areas).all() and np.all(areas > 0) and np.isfinite(areas.sum()),
          'Surface has degenerate or nonfinite triangles.')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(fs), size=count, p=areas/areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1-root, root*(1-uv[:, 1]), root*uv[:, 1]))
    xyz = np.sum(triangles[chosen] * bary[:, :, None], axis=1)
    return _readonly(xyz), _readonly(chosen), float(areas.sum())


def initialize_framewise_surface(bundle, grid, *, reference_frame_id, valid_masks, validity_notes,
                                 camera_length_contract, min_valid_views, vote_fraction,
                                 seed, initial_points=200_000, chunk_size=100_000):
    """Return free reference XYZ plus final fixed-ID motion transport for training.

    Bundle E and mesh already share normalized model units. The required
    meters/1 declaration must NOT be replaced by an upstream SfM conversion;
    that conversion and its original source E belong in upstream provenance.
    A model-conditioned unit basis is allowed, without claiming a measured
    physical scale. The caller still owns covariance, opacity and SH, as well
    as physical/visual acceptance. Sample vote coverage is reported, not used
    to prune, resample, tune thresholds or invent invisible clothing surfaces.
    """
    _integer(initial_points, 'initial_points')
    _integer(seed, 'seed', 0)
    volume, motion, tree, frames = _evaluate(bundle, grid, reference_frame_id=reference_frame_id,
        valid_masks=valid_masks, validity_notes=validity_notes, camera_length_contract=camera_length_contract,
        min_valid_views=min_valid_views, vote_fraction=vote_fraction, chunk_size=chunk_size)
    vertices, faces, boundary_audit = extract_boundary_surface(volume, motion, tree, frames)
    xyz, selected, area = sample_surface_by_area(vertices, faces, count=initial_points, seed=seed)
    support = tree.query(xyz, workers=1)[1].astype(np.int64)
    gaussian_ids = tuple('surface-%08d' % i for i in range(initial_points))
    transport = FreeClothingTransport(frame_ids=motion.frame_ids, reference_frame_id=reference_frame_id,
        vertices_world=motion.vertices_world, faces=motion.faces, motion_support_triangle_ids=support,
        gaussian_ids=gaussian_ids, frame_source_sha256=motion.frame_source_sha256, translation_unit='meters')
    positive, valid, sample_frames = _vote_queries(lambda begin, end: xyz[begin:end], initial_points,
                                                  support, transport, frames, chunk_size)
    ratios = np.divide(positive, valid, out=np.zeros(initial_points, dtype=float), where=valid > 0)
    report = dict(volume.report(), initial_points=initial_points, seed=seed,
        boundary_geometry=boundary_audit,
        final_gaussian_support_selection_count=1, final_support_sha256=_array_sha(support),
        xyz_ref_sha256=_array_sha(xyz), surface_vertices=len(vertices), surface_faces=len(faces),
        surface_area_m2=area, occupied_voxels=int(volume.occupied.sum()),
        sampling='IID triangle area probability with uniform barycentric sampling',
        per_triangle_quota=False, bare_mesh_attraction=False, points_projected_onto_body=False,
        training_rebinding_allowed=False, sampled_point_votes_by_frame=sample_frames,
        sampled_points_with_sufficient_views=int((valid >= min_valid_views).sum()),
        sampled_points_meeting_vote_rule=int(((valid >= min_valid_views) & (ratios > vote_fraction)).sum()),
        limitations=['Silhouette envelope cannot recover unobserved concavities or true skin/clothing depth.',
            'Nearest-triangle-centroid support is a motion approximation, not a physical material match.',
            'Piecewise motion can be discontinuous between support regions; shear residual is reported.',
            'Per-frame validity requires external visibility/occlusion evidence; this module cannot infer it.',
            'Surface interpolation and final support selection can change sampled-point votes; these are reported.',
            'Unit contract is declared evidence, not a newly estimated physical scale.'])
    bundle.assert_unchanged()
    return FramewiseSurfaceInitialization(xyz, gaussian_ids, transport, vertices, faces, selected,
        _readonly(positive), _readonly(valid), volume, json.dumps(report, sort_keys=True, allow_nan=False))

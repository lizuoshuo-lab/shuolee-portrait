"""Small known-geometry CPU controls; no body model, native renderer or GPU."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shuolee_portrait.frame_pose_contract import Bundle, Frame, load_frame_pose_bundle
from shuolee_portrait.framewise_clothed_surface import (
    ReferenceGrid, VoteVolume, evaluate_framewise_votes, extract_supported_surface,
    initialize_framewise_surface, sample_surface_by_area, _mask_observation)


def length(factor=1.):
    return dict(translation_unit='meters' if factor == 1 else 'sfm_units',
                meters_per_camera_unit=factor, basis='synthetic_known',
                evidence='Known synthetic scale; not physical calibration.')


def look_at(center, target):
    forward = target-center
    forward /= np.linalg.norm(forward)
    up = np.array([0., 1., 0.]) if abs(forward[1]) < .9 else np.array([0., 0., 1.])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    E = np.eye(4)
    E[:3, :3] = np.stack((right, up, forward))
    E[:3, 3] = -E[:3, :3] @ center
    return E


def cube_silhouette(K, E, center, half=.45, size=65):
    """Independent ray/slab intersection with a known translated solid cube."""
    y, x = np.indices((size, size))
    pixels = np.stack((x, y, np.ones_like(x)), -1)
    rays = (pixels @ np.linalg.inv(K).T) @ E[:3, :3]
    origin = -E[:3, :3].T @ E[:3, 3] - center
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = (-half-origin)/rays, (half-origin)/rays
    near = np.minimum(a, b).max(-1)
    far = np.maximum(a, b).min(-1)
    return (far >= np.maximum(near, 0)).astype(np.uint8)*255


def make_bundle(directory, many=False):
    center = np.array([0., 0., 4.])
    mesh = np.array([[-.3, -.3, 4.], [.4, -.2, 4.], [.1, .4, 4.]])
    K = np.array([[45., 3., 32.], [0., 45., 32.], [0., 0., 1.]])
    frames, bound = {}, []
    names = ('right', 'reference', 'left', 'above', 'below', 'back') if many else ('moved', 'reference')
    cameras = ([np.array([3., 0., 4.]), np.zeros(3), np.array([-3., 0., 4.]),
                np.array([0., 3., 4.]), np.array([0., -3., 4.]), np.array([0., 0., 8.])]
               if many else [np.zeros(3), np.zeros(3)])
    for i, (name, camera) in enumerate(zip(names, cameras)):
        shift = np.zeros(3) if name == 'reference' else np.array([.6, .15, .1]) if many else np.array([1.2, 0., 0.])
        E = look_at(camera, center)
        image, mask = directory/(name+'.png'), directory/(name+'-mask.png')
        Image.fromarray(np.full((65, 65, 3), i*13, dtype=np.uint8)).save(image)
        Image.fromarray(cube_silhouette(K, E, center+shift)).save(mask)
        for path in (image, mask):
            bound.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
        frames[name] = Frame(name, 10+i*7, i, image, mask, (65, 65), K.copy(), E, mesh+shift)
    bundle = Bundle('known-moving-cube', 'b'*64, 'synthetic_contract_test', 'reference',
                    np.zeros((1, 10)), np.array([[0, 1, 2]]), MappingProxyType(frames), tuple(bound))
    valid = {name: np.ones((65, 65), dtype=bool) for name in names}
    notes = {name: 'Analytic synthetic visibility; entire camera frame is known, no occluder.' for name in names}
    return bundle, valid, notes


def grid_small():
    return ReferenceGrid(([-.1, 0., .1], [-.1, 0., .1], [3.9, 4., 4.1]), reference_frame_id='reference')


def load_file_bundle(bundle, directory):
    """Exercise the production loader, including declared normalized K/E role."""
    ids = list(bundle._frames)
    frames = [bundle.frame(name) for name in ids]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [dict(frame=f.frame_id, source_frame=f.source_frame, sequence_id=bundle.sequence_id,
                 video_sha256=bundle.video_sha256, wh=list(f.wh), image=str(f.image),
                 source_sha256=digest(f.image), mask=str(f.mask), mask_sha256=digest(f.mask),
                 K_integer_centers=f.K_integer_centers.tolist(), w2c=f.w2c.tolist()) for f in frames]
    roster = directory/'frames.json'
    roster.write_text(json.dumps(dict(records=rows)), encoding='utf-8')
    poses = directory/'poses.npz'
    np.savez(poses, frame_ids=np.array(ids), source_frames=np.array([f.source_frame for f in frames]),
             image_sha256=np.array([digest(f.image) for f in frames]),
             sequence_id=np.array(bundle.sequence_id), video_sha256=np.array(bundle.video_sha256),
             vertices_world=np.stack([f.vertices_world for f in frames]), faces=bundle.faces,
             shared_betas=bundle.shared_betas, K_integer_centers=np.stack([f.K_integer_centers for f in frames]),
             w2c=np.stack([f.w2c for f in frames]))
    manifest = directory/'bundle.json'
    manifest.write_text(json.dumps(dict(schema='shuolee-frame-pose/1', purpose='synthetic_contract_test',
        sequence_id=bundle.sequence_id, video_sha256=bundle.video_sha256,
        coordinate_system='opencv_world_meters', pixel_domain='rectified_pinhole',
        pixel_convention='integer_centers', reference_frame_id='reference',
        frames=dict(path=roster.name, sha256=digest(roster), rows_key='records'),
        poses=dict(path=poses.name, sha256=digest(poses)))), encoding='utf-8')
    return load_frame_pose_bundle(manifest)


def vote(bundle, valid, notes, grid=None, **changes):
    args = dict(reference_frame_id='reference', valid_masks=valid, validity_notes=notes,
                camera_length_contract=length(), min_valid_views=1, vote_fraction=.75, chunk_size=7)
    args.update(changes)
    return evaluate_framewise_votes(bundle, grid_small() if grid is None else grid, **args)


class FramewiseSurfaceControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.bundle, self.valid, self.notes = make_bundle(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_moving_silhouette_requires_warp_before_vote(self):
        result = vote(self.bundle, self.valid, self.notes)
        self.assertEqual(int(result.positive_counts[1, 1, 1]), 2)
        self.assertEqual(int(result.valid_counts[1, 1, 1]), 2)
        frame = self.bundle.frame('moved')
        old = dict(K=frame.K_integer_centers, E=frame.w2c, wh=frame.wh,
                   mask=np.array(Image.open(frame.mask)) == 255, valid=self.valid['moved'])
        hit, seen = _mask_observation(np.array([[0., 0., 4.]]), old)
        self.assertTrue(seen[0])
        self.assertFalse(hit[0], 'Negative control: old static-world lookup must miss moved body.')
        self.assertFalse(result.report()['training_admitted'])

    def test_unknown_excluded_but_known_background_negative(self):
        # Test an existing known-background reference-world query by moving the
        # silhouette only, with a re-bound file: geometry is intentionally wrong.
        moved = self.bundle.frame('moved')
        Image.fromarray(np.zeros((65, 65), np.uint8)).save(moved.mask)
        bound = tuple((path, hashlib.sha256(path.read_bytes()).hexdigest()) for path, _ in self.bundle._bound_files)
        bundle = replace(self.bundle, _bound_files=bound)
        unknown = dict(self.valid, moved=np.zeros((65, 65), bool))
        result = vote(bundle, unknown, self.notes)
        self.assertEqual(int(result.valid_counts[1, 1, 1]), 1)
        self.assertEqual(float(result.fractions[1, 1, 1]), 1.)
        self.assertTrue(result.occupied[1, 1, 1])
        negative = vote(bundle, self.valid, self.notes)
        self.assertEqual(int(negative.valid_counts[1, 1, 1]), 2)
        self.assertEqual(float(negative.fractions[1, 1, 1]), .5)
        self.assertFalse(negative.occupied[1, 1, 1])

    def test_frame_identity_order_and_reference_not_index_zero(self):
        expected = vote(self.bundle, self.valid, self.notes)
        reordered = replace(self.bundle, _frames=MappingProxyType(dict(reversed(list(self.bundle._frames.items())))))
        actual = vote(reordered, self.valid, self.notes)
        np.testing.assert_array_equal(expected.positive_counts, actual.positive_counts)
        np.testing.assert_array_equal(expected.valid_counts, actual.valid_counts)
        with self.assertRaisesRegex(ValueError, 'reference identity'):
            vote(self.bundle, self.valid, self.notes, reference_frame_id='moved')
        wrong_grid = ReferenceGrid(([-.1, 0., .1], [-.1, 0., .1], [3.9, 4., 4.1]), reference_frame_id='moved')
        with self.assertRaisesRegex(ValueError, 'reference identity'):
            vote(self.bundle, self.valid, self.notes, grid=wrong_grid)

    def test_full_skew_K_integer_centres_and_unknown_outside(self):
        K = np.array([[20., 7., 10.2], [0., 30., 11.1], [0., 0., 1.]])
        mask = np.zeros((32, 32), bool)
        mask[23, 17] = True  # Independent equation: (u,v)=(17,23.1).
        frame = dict(K=K, E=np.eye(4), wh=(32, 32), mask=mask, valid=np.ones_like(mask))
        points = np.array([[1., 2., 5.], [100., 0., 1.], [0., 0., -1.]])
        hit, seen = _mask_observation(points, frame)
        np.testing.assert_array_equal(hit, [True, False, False])
        np.testing.assert_array_equal(seen, [True, False, False])
        wrongK = K.copy()
        wrongK[:2, 2] += .5
        self.assertFalse(_mask_observation(points, dict(frame, K=wrongK))[0][0])

    def test_production_bundle_camera_not_scaled_again(self):
        # File loading guarantees that image/pose records bind exactly to the
        # same declared normalized E. The initializer must not reinterpret it.
        grid = ReferenceGrid.from_bounds([-.8, -.1, 3.9], [.8, .1, 4.1], spacing_m=.1,
                                         reference_frame_id='reference')
        frames = {}
        for name, f in self.bundle._frames.items():
            E = f.w2c.copy()
            E[0, 3] = -.5
            frames[name] = replace(f, w2c=E)
        metric = load_file_bundle(replace(self.bundle, _frames=MappingProxyType(frames)), self.directory)
        baseline = vote(metric, self.valid, self.notes, grid=grid)
        self.assertEqual(baseline.report()['camera_translation_scale_applications'], 0)
        self.assertFalse(baseline.report()['geometry_scaled_again'])
        for item in baseline.report()['input_frames']:
            np.testing.assert_array_equal(item['normalized_w2c'], metric.frame(item['frame_id']).w2c)
        for invalid in (length(2.), dict(length(), translation_unit='sfm_units')):
            with self.assertRaisesRegex(ValueError, 'already contains normalized'):
                vote(metric, self.valid, self.notes, camera_length_contract=invalid)
        with self.assertRaises(ValueError):
            vote(self.bundle, self.valid, self.notes, camera_length_contract=None)

    def test_model_conditioned_basis_does_not_claim_physical_scale(self):
        loaded = load_file_bundle(self.bundle, self.directory)
        contract = dict(length(), basis='model_conditioned', evidence='Shared model gauge, not measured metres.')
        actual = vote(loaded, self.valid, self.notes, camera_length_contract=contract)
        self.assertFalse(actual.report()['physical_calibration_validated'])
        self.assertEqual(actual.report()['camera_length_contract']['basis'], 'model_conditioned')
        self.assertEqual(actual.report()['camera_translation_scale_applications'], 0)

    def test_validity_contract_missing_wrong_size_or_non_bool_rejected(self):
        for masks in ({'reference': self.valid['reference']},
                      dict(self.valid, moved=np.ones((12, 12), bool)),
                      dict(self.valid, moved=np.ones((65, 65), np.uint8))):
            with self.assertRaises(ValueError):
                vote(self.bundle, masks, self.notes)
        with self.assertRaises(ValueError):
            vote(self.bundle, self.valid, dict(self.notes, moved=''))

    def test_input_mutation_and_nonbinary_silhouette_fail_closed(self):
        path = self.bundle.frame('moved').mask
        Image.fromarray(np.full((65, 65), 128, np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'Bound input changed'):
            vote(self.bundle, self.valid, self.notes)
        bound = tuple((p, hashlib.sha256(p.read_bytes()).hexdigest()) for p, _ in self.bundle._bound_files)
        with self.assertRaisesRegex(ValueError, 'binary'):
            vote(replace(self.bundle, _bound_files=bound), self.valid, self.notes)

    def test_surface_unknown_neighbor_or_grid_clip_rejected(self):
        grid = grid_small()
        positive = np.zeros(grid.shape, np.uint32)
        positive[1, 1, 1] = 1
        valid = np.ones(grid.shape, np.uint32)
        valid[0, 0, 0] = 0
        volume = VoteVolume(grid, positive, valid, np.zeros(grid.size, np.int64), 1, .5, '{}')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            extract_supported_surface(volume)
        positive[0, 1, 1] = 1
        with self.assertRaisesRegex(ValueError, 'grid boundary'):
            extract_supported_surface(replace(volume, valid_counts=np.ones(grid.shape, np.uint32)))

    def test_moving_cube_extracts_free_points_and_fixed_transport(self):
        folder = self.directory/'six'
        folder.mkdir()
        bundle, valid, notes = make_bundle(folder, many=True)
        grid = ReferenceGrid.from_bounds([-1., -1., 3.], [1., 1., 5.], spacing_m=.125,
                                         reference_frame_id='reference', max_voxels=6000)
        result = initialize_framewise_surface(bundle, grid, reference_frame_id='reference',
            valid_masks=valid, validity_notes=notes, camera_length_contract=length(),
            min_valid_views=6, vote_fraction=.9, seed=713, initial_points=256, chunk_size=157)
        self.assertEqual(result.xyz_ref.shape, (256, 3))
        self.assertEqual(result.gaussian_ids, result.transport.gaussian_ids)
        self.assertTrue(np.all(result.transport.motion_support_triangle_ids == 0))
        # Bare support triangle lies in z=4; sampled enclosure must not snap there.
        self.assertGreater(np.count_nonzero(abs(result.xyz_ref[:, 2]-4.) > .15), 128)
        self.assertFalse(result.report()['points_projected_onto_body'])
        self.assertEqual(result.report()['final_gaussian_support_selection_count'], 1)
        np.testing.assert_array_equal(result.transport.vertices_world[result.transport.frame_ids.index('reference')],
                                      bundle.reference.vertices_world)
        covariance = np.tile(np.eye(3)*.01, (256, 1, 1))
        moved = result.transport.transport_numpy('right', result.xyz_ref, covariance, np.zeros(3),
                                                 gaussian_ids=result.gaussian_ids)
        np.testing.assert_allclose(moved['centers'], result.xyz_ref+[.6, .15, .1], atol=1e-14, rtol=0)
        with self.assertRaises(ValueError):
            result.xyz_ref[0, 0] = 100

    def test_area_sampling_not_triangle_quota_and_reproducible(self):
        verts = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.],
                          [10., 0., 0.], [16., 0., 0.], [10., 3., 0.]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        xyz, chosen, area = sample_surface_by_area(verts, faces, count=5000, seed=123)
        self.assertEqual(area, 10.)
        self.assertLess(abs(np.mean(chosen == 1)-.9), .02)
        np.testing.assert_array_equal(xyz, sample_surface_by_area(verts, faces, count=5000, seed=123)[0])
        small = sample_surface_by_area(verts, faces, count=1, seed=123)[1]
        self.assertEqual(len(np.unique(small)), 1)  # No mandatory sample for every face.

    def test_budget_and_unobserved_queries_are_not_empty_space(self):
        with self.assertRaises(ValueError):
            ReferenceGrid.from_bounds([0, 0, 0], [2, 2, 2], spacing_m=.01,
                                      reference_frame_id='reference', max_voxels=100)
        all_unknown = {name: np.zeros_like(value) for name, value in self.valid.items()}
        result = vote(self.bundle, all_unknown, self.notes)
        self.assertTrue(np.isnan(result.fractions).all())
        self.assertFalse(result.eligible.any())
        with self.assertRaisesRegex(ValueError, 'No supported'):
            extract_supported_surface(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)

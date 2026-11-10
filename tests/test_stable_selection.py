"""CPU fixtures for the stable SMPL capture and selection contract."""
from itertools import islice
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shuolee_portrait import official_sequence, runtime, stable_selection


VIDEO_SHA = "a" * 64


def fixture(count):
    ids = [f"frame_{i:03}" for i in range(count)]
    rows = [dict(frame=name, source_frame_index=i, wh=[200, 100], initial_box_xyxy=[0, 0, 150, 100])
            for i, name in enumerate(ids)]
    targets = np.zeros((count, 118, 3), np.float32)
    targets[..., 2] = 1.
    return dict(rows=rows, ids=ids, targets=targets, K=np.tile(np.eye(3), (count, 1, 1)),
                source_manifest=dict(video_sha256=VIDEO_SHA), source_reliability=np.ones(count))


def fitted_winners(inputs, scope):
    index = {frame: i for i, frame in enumerate(inputs["ids"])}
    winners = {}
    for frame in scope["fit_frame_ids"]:
        pose = {field: np.zeros((1, 3), np.float32) for field in stable_selection.POSE_FIELDS}
        pose.update(frame_id=np.asarray(frame), betas=np.zeros((1, 10), np.float32),
                    K_integer_centers=inputs["K"][index[frame]],
                    vertices_camera=np.ones((1, 3, 3), np.float32),
                    faces=np.array([[0, 1, 2]]), joints_projected_native=np.ones((1, 118, 2), np.float32))
        winners[frame] = pose
    return winners


class StableSelectionTests(unittest.TestCase):
    def test_missing_bin_seed_is_ranked_once_and_excludes_holdouts(self):
        inputs = fixture(60)
        scope = stable_selection.selection_scope(inputs)
        winners = fitted_winners(inputs, scope)
        for pose in winners.values():
            pose['pose_embedding'] = np.ones((1, 32), np.float32)
        missing = 3
        members = scope['source_view_bins'][missing]
        first, second, *_ = [frame for frame in members if frame in winners]
        self.assertTrue(set(members) & set(scope['retained_holdouts']))
        for frame in members:
            if frame in winners:
                winners[frame]['joints_projected_native'][0, 2, 0] = 5.
        winners[second]['joints_projected_native'][0, 2, 0] = 4.
        selection = stable_selection.select_frames(inputs, winners, scope=scope)
        self.assertIn(missing, selection['coverage']['front']['missing_source_bins'])
        items = list(official_sequence._missing_bin_seeds(inputs, selection, winners, {}))
        item, = [item for item in items if item['source_bin'] == missing]
        self.assertEqual(item['frame_id'], second)
        self.assertEqual(item['seed_kind'], 'nearby_baseline_admitted')
        self.assertNotIn(item['seed_frame'], scope['retained_holdouts'])
        self.assertLessEqual(abs(selection['reasons'][item['seed_frame']]['source_bin']-missing), 2)
        anchors = {second: (np.ones((1, 32), np.float32), {})}
        item, = [item for item in official_sequence._missing_bin_seeds(inputs, selection, winners, anchors)
                 if item['source_bin'] == missing]
        self.assertEqual((item['frame_id'], item['seed_kind'], item['seed_frame']),
                         (second, 'same_frame_shape_anchor', second))
        self.assertNotEqual(first, item['frame_id'])

    def test_recovery_checks_each_supported_point_in_native_pixels(self):
        observed = np.zeros((118, 3), np.float32)
        observed[[2, 25, 67], 2] = 1.
        before = dict(joints_projected_native=np.zeros((1, 118, 2), np.float32))
        after = dict(joints_projected_native=before['joints_projected_native'].copy())
        after['joints_projected_native'][0, [2, 25, 67, 68], 0] = [1., .75, .51, 500.]
        protect = official_sequence._recovery_protection(observed, before, after)
        self.assertEqual(protect['head']['points'], 1)
        self.assertFalse(protect['head']['passed'])
        self.assertEqual(protect['nonhead']['points'], 2)
        self.assertTrue(protect['nonhead']['passed'])
        after['joints_projected_native'][0, 25, 0] = 1.01
        self.assertFalse(official_sequence._recovery_protection(observed, before, after)['nonhead']['passed'])

    def test_half_budget_original_pose_gate_and_reference(self):
        inputs = fixture(60)
        scope = stable_selection.selection_scope(inputs)
        expected_holdouts = ["frame_010", "frame_028", "frame_043", "frame_055"]
        self.assertEqual(scope["retained_holdouts"], expected_holdouts)
        inputs["source_reliability"][[10, 28, 43, 55]] = 100.
        anchors = stable_selection.shape_anchors(inputs, scope)
        self.assertEqual((len(anchors["front"]), len(anchors["rear"])), (24, 0))
        self.assertTrue(set(expected_holdouts).isdisjoint(anchors["front"]))

        winners = fitted_winners(inputs, scope)
        winners["frame_003"]["joints_projected_native"][0, 2, 0] = 4.
        winners["frame_006"]["joints_projected_native"][0, 19, 0] = 4.
        winners["frame_009"]["joints_projected_native"][0, 9, 0] = 400.
        winners["frame_025"]["joints_projected_native"][:] = 0.
        selected = stable_selection.select_frames(inputs, winners, scope=scope)
        self.assertEqual(len(selected["selected_frames"]), 24)
        self.assertEqual(selected["coverage"]["rear"]["selected"], 0)
        self.assertEqual(selected["reference_frame"], "frame_025")
        self.assertIn("frame_009", selected["selected_frames"])
        self.assertEqual(selected["reasons"]["frame_003"]["screen_flags"],
                         ["original_supported_body_reprojection_over_existing3percent_height"])
        self.assertEqual(selected["reasons"]["frame_006"]["screen_flags"],
                         ["feet_reprojection_over_existing3percent_height"])
        self.assertTrue(set(expected_holdouts).isdisjoint(selected["selected_frames"]))

    def test_full_budget_and_protected_gap_frame(self):
        inputs = fixture(72)
        layout = dict(video_sha256=VIDEO_SHA, front_source_ranges=[[0, 20], [42, 57]],
                      rear_source_ranges=[[24, 41], [60, 71]], held_source_indices=[2, 22, 30, 47, 69])
        scope = stable_selection.selection_scope(inputs, arc="full", capture_layout=layout)
        winners = fitted_winners(inputs, scope)
        selected = stable_selection.select_frames(inputs, winners, scope=scope)
        self.assertEqual((selected["coverage"]["front"]["selected"], selected["coverage"]["rear"]["selected"]),
                         (24, 12))
        self.assertEqual(selected["selected_frames"], sorted(selected["selected_frames"]))
        self.assertTrue(selected["full_roster"])
        self.assertEqual(selected["reasons"]["frame_022"]["status"], "protected_holdout")
        self.assertTrue(set(scope["retained_holdouts"]).isdisjoint(scope["fit_frame_ids"]))
        with self.assertRaisesRegex(ValueError, "frozen source roster"):
            stable_selection.select_frames(inputs, dict(winners, frame_022=winners["frame_000"]), scope=scope)
        with self.assertRaisesRegex(ValueError, "different original video"):
            stable_selection.selection_scope(inputs, arc="full", capture_layout=dict(layout, video_sha256="b"*64))

    def test_quota_extras_spread_native_time_and_report_edge_gaps(self):
        inputs = fixture(72)
        for index, row in enumerate(inputs["rows"]):
            row["time_seconds"] = index + (20 if index >= 68 else 0)
        scope = stable_selection.selection_scope(inputs)
        winners = fitted_winners(inputs, scope)
        for bin_id in (0, 1, 7, 8, 23):
            for frame in scope["source_view_bins"][bin_id]:
                winners.pop(frame, None)
        winners["frame_067"]["joints_projected_native"][0, 2, 0] = 4.
        winners["frame_068"]["joints_projected_native"][0, 2, 0] = 2.75
        selection = stable_selection.select_frames(inputs, winners, scope=scope)
        coverage = selection["coverage"]["front"]
        self.assertTrue(selection["full_roster"])
        self.assertFalse(selection["full_bin_coverage"])
        self.assertEqual(coverage["missing_bin_runs"], [[0, 1], [7, 8], [23, 23]])
        self.assertEqual(coverage["max_consecutive_uncovered_bins"], 2)
        self.assertEqual(selection["coverage"]["rear"]["max_consecutive_uncovered_bins"], 0)
        self.assertEqual(coverage["quota_fill_spacing_axis"], "native_pts_microseconds")
        self.assertEqual(selection["reasons"]["frame_068"]["status"], "selected_quota_fill")
        self.assertEqual(selection["reasons"]["frame_067"]["status"], "pose_screen_rejection")
        self.assertTrue(set(scope["retained_holdouts"]).isdisjoint(selection["selected_frames"]))

    def test_sampler_balances_occupied_bins_and_rotates_within_each_bin(self):
        ids = ["frame_010", "frame_002", "frame_000", "frame_001"]
        rows = [dict(frame=frame, source_frame_index=int(frame[6:])) for frame in ids]
        reasons = {frame: dict(source_bin=bin_id, status=status) for frame, bin_id, status in zip(
            ids, (3, 0, 0, 0), ("selected_bin_leader", "selected_quota_fill",
                                 "selected_bin_leader", "selected_quota_fill"))}
        selection = dict(selected_frames=sorted(ids), reasons=reasons, retained_holdouts=["frame_009"],
                         coverage={"front": {"occupied_source_bins": [0, 3]}})
        bins, order = stable_selection.source_bin_training_order(rows, selection, seed=0)
        self.assertEqual(bins, [3, 0, 0, 0])
        picked = list(islice(order, 8))
        _, repeat = stable_selection.source_bin_training_order(rows, selection, seed=0)
        self.assertEqual(picked, list(islice(repeat, 8)))
        for i in range(0, len(picked), 2):
            self.assertEqual(sorted(bins[j] for j in picked[i:i+2]), [0, 3])
        self.assertEqual([ids[j] for j in picked if bins[j] == 0],
                         ["frame_000", "frame_001", "frame_002", "frame_000"])
        with self.assertRaisesRegex(ValueError, "protected selection"):
            stable_selection.source_bin_training_order(
                rows, dict(selection, retained_holdouts=["frame_000"]), seed=0)

    def test_q90_excludes_protected_and_unassigned_native_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image, mask = root / "rgb.png", root / "mask.png"
            Image.new("RGB", (8, 6)).save(image)
            Image.new("L", (8, 6), 255).save(mask)
            K = np.array([[8., 0., 3.], [0., 8., 2.], [0., 0., 1.]])
            corner = K.copy()
            corner[:2, 2] += .5
            ids = [f"frame_{i:03}" for i in range(5)]
            rows = [dict(frame=frame, image=str(image), mask=str(mask), wh=[8, 6],
                         source_sha256=runtime.sha(image), mask_sha256=runtime.sha(mask), source_frame_index=i*10,
                         K_integer_centers=K.tolist(), K=corner.tolist(), w2c=np.eye(4).tolist())
                    for i, frame in enumerate(ids)]
            manifest, observations = root / "inputs.json", root / "observations.npz"
            runtime.save(manifest, dict(records=rows, video_sha256=VIDEO_SHA))
            points = np.full((5, 133, 2), 2., np.float32)
            points[2, 0] = [-.5, 2.]
            points[2, 1] = [7.5, 2.]
            scores = np.tile(np.array([.1, 100., .4, .8, 80.])[:, None], (1, 133))
            np.savez(observations, frame_ids=np.array(ids[::-1]), keypoints=points[::-1], scores=scores[::-1],
                     boxes=np.tile([0, 0, 8, 6], (5, 1)), flip_disagreement=np.zeros((5, 133)))
            layout = dict(arc="half", video_sha256=VIDEO_SHA, front_source_ranges=[[0, 30]],
                          rear_source_ranges=[], held_source_indices=[10])
            inputs = official_sequence.load_inputs(
                manifest, observations, reference_frame=ids[0], capture_layout=layout,
                camera_length_contract=dict(translation_unit="meters", meters_per_camera_unit=1.,
                                            basis="synthetic_known", evidence="CPU fixture"))

        self.assertEqual(inputs["scope"]["fit_frame_ids"], [ids[i] for i in (0, 2, 3)])
        np.testing.assert_allclose(inputs["confidence"]["per_joint_q90"], .72, rtol=0, atol=1e-12)
        targets = inputs["targets"]
        self.assertFalse(targets[[1, 4]].any())
        np.testing.assert_array_equal(targets[2, [0, 16], :2], points[2, :2])
        self.assertAlmostEqual(float(targets[2, 0, 2]), .4 / .72)
        self.assertEqual(float(targets[2, 16, 2]), 0.)


if __name__ == "__main__":
    unittest.main()


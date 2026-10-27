"""Reproject real serialized camera files through independent reader formulas."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.camera_exchange import camera_documents, write_camera_exchange


def sample_rows():
    rows = []
    for i in range(3):
        E = np.eye(4)
        E[:3, :3] = Rotation.from_euler('xyz', [13 + i * 2, -17 + i * 4, 19 + i * 5], degrees=True).as_matrix()
        E[:3, 3] = [.23 * i, -.17, 1.1 + .3 * i]
        rows.append(dict(frame=f'frame_{i}', name=f'frame_{i}.png', wh=[640, 480],
            K=[[710.4 + i, 0., 137.25], [0., 694.1 - i, 311.4], [0., 0., 1.]], w2c=E.tolist()))
    return rows


def pinhole(K, camera_points):
    q = camera_points @ K.T
    return q[:, :2] / q[:, 2:]


def hamilton_rotation(q):
    # Explicit COLMAP/Hamilton world-to-camera formula. This reader does not use
    # the exporter's scipy quaternion conversion or any exporter helper.
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


class CameraExchangeTests(unittest.TestCase):
    def test_common_json_and_colmap_text_reproject_offcenter_cameras(self):
        rows = sample_rows()
        points = np.array([[-.4, -.3, 4.1], [.7, -.2, 6.2], [.13, .8, 5.4], [1.2, 1.3, 8.]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = write_camera_exchange(root, rows, world_to_viewer=dict(rotation=np.eye(3).tolist()))
            cameras = json.loads((root / 'cameras.json').read_text())
            self.assertIsInstance(cameras, list)
            camera_lines = [line.split() for line in (root / 'colmap/cameras.txt').read_text().splitlines() if line and not line.startswith('#')]
            intrinsics = {}
            for values in camera_lines:
                self.assertEqual(values[1], 'PINHOLE')
                fx, fy, cx, cy = map(float, values[4:])
                intrinsics[int(values[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
            lines = [line for line in (root / 'colmap/images.txt').read_text().splitlines() if not line.startswith('#')]
            self.assertEqual(len(lines), 2 * len(rows))
            for i, (source, common) in enumerate(zip(rows, cameras)):
                E, K = np.array(source['w2c']), np.array(source['K'])
                expected = pinhole(K, points @ E[:3, :3].T + E[:3, 3])
                c2w = np.asarray(common['rotation'])
                camera_points = (points - common['position']) @ c2w
                json_K = np.array([[common['fx'], 0, common['cx']], [0, common['fy'], common['cy']], [0, 0, 1.]])
                np.testing.assert_allclose(pinhole(json_K, camera_points), expected, atol=2e-10)
                values = lines[2 * i].split()
                self.assertEqual(lines[2 * i + 1], '')  # No invented image observations.
                R = hamilton_rotation(np.array(values[1:5], float))
                t = np.array(values[5:8], float)
                colmap_K = intrinsics[int(values[8])]
                np.testing.assert_allclose(pinhole(colmap_K, points @ R.T + t), expected, atol=2e-10)
                np.testing.assert_allclose(-R.T @ t, common['position'], atol=1e-12)
                np.testing.assert_array_equal(colmap_K, K)
                self.assertEqual(common['cx'], 137.25)
                self.assertEqual(common['cy'], 311.4)
                self.assertEqual(common['pixelConvention'], 'pixel_corners')
                self.assertEqual(common['projectionModel'], 'pinhole')
            self.assertEqual(report['point_count'], 0)
            sparse = [line for line in (root / 'colmap/points3D.txt').read_text().splitlines() if line and not line.startswith('#')]
            self.assertEqual(sparse, [])
            self.assertEqual(report['records'], rows)

    def test_duplicate_spaces_paths_and_reserved_names_have_lossless_mapping(self):
        rows = sample_rows()
        rows += [copy.deepcopy(rows[0]), copy.deepcopy(rows[1])]
        originals = ['same frame.png', 'same frame.png', '../same frame.png', 'CON.png', '人像 侧面.png']
        for row, name in zip(rows, originals):
            row['frame'], row['name'] = 'duplicate source frame', name
        docs = camera_documents(rows)
        names = [camera['img_name'] for camera in docs['cameras']]
        self.assertEqual(len(set(name.casefold() for name in names)), len(rows))
        self.assertTrue(all(re.fullmatch(r'[A-Za-z0-9_.-]+', name) and not name.startswith('.') for name in names))
        self.assertEqual([mapping['original_name'] for mapping in docs['name_mapping']], originals)
        self.assertTrue(all(mapping['original_frame'] == 'duplicate source frame' for mapping in docs['name_mapping']))
        self.assertEqual(docs, camera_documents(rows))

    def test_empty_reflected_and_skew_cameras_fail_before_creating_files(self):
        with self.assertRaises(ValueError):
            camera_documents([])
        with tempfile.TemporaryDirectory() as directory:
            for field in ('reflection', 'skew'):
                rows = sample_rows()
                if field == 'reflection':
                    rows[0]['w2c'] = np.diag([-1., 1., 1., 1.]).tolist()
                else:
                    rows[0]['K'][0][1] = 3.
                target = Path(directory) / field
                with self.assertRaises(ValueError):
                    write_camera_exchange(target, rows)
                self.assertFalse(target.exists())

    def test_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_camera_exchange(root, sample_rows())
            before = hashlib.sha256((root / 'cameras.json').read_bytes()).hexdigest()
            self.assertEqual(first['files_sha256']['cameras.json'], before)
            with self.assertRaises(FileExistsError):
                write_camera_exchange(root, sample_rows())
            self.assertEqual(hashlib.sha256((root / 'cameras.json').read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()

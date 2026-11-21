"""Opt-in DriveStudio AV2 camera poses at actual image exposure timestamps.

Leaves the author's repository and LiDAR-time instance annotations unchanged.
Call install(ArgoVerseCameraData) before constructing an AV2 PixelSource. Both
camera rays and the camera-to-world rotations used by the SMPL loader then use
the same exposure-time poses. This does not interpolate moving object poses.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


def camera_to_reference(data_path, camera_id, start, end):
    root = Path(data_path)
    records = json.loads((root/'capture_timestamps.json').read_text())['frames']
    if not (0 <= start < end <= len(records)):
        raise ValueError('Camera interval is outside the capture manifest')
    reference = np.loadtxt(root/'ego_pose'/f'{start:03d}.txt')
    inverse_reference = np.linalg.inv(reference)
    matrices, stamps = [], []
    for index in range(start, end):
        frame = records[index]
        assert frame['frame_idx'] == index
        candidates = [c for c in frame['cameras'] if int(c['camera_id']) == int(camera_id)]
        if len(candidates) != 1:
            raise ValueError(f'Missing/ambiguous camera {camera_id}, frame {index}')
        entry = candidates[0]
        path = root/entry['capture_pose']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Capture pose must stay inside the scene directory')
        matrix = np.loadtxt(path)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f'Invalid capture pose {path}')
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1], atol=1e-10)
        np.testing.assert_allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-7)
        if abs(np.linalg.det(matrix[:3, :3])-1) > 1e-7:
            raise ValueError('Capture rotation must have determinant +1')
        matrices.append(inverse_reference@matrix)
        stamps.append(int(entry['timestamp_ns']))
    return torch.from_numpy(np.stack(matrices)).float(), torch.tensor(stamps, dtype=torch.int64)


def install(camera_class):
    """Patch only the two camera calibration entry points, in this process."""
    if camera_class.__dict__.get('_gs_lab_capture_time_installed', False):
        return
    original_load = camera_class.load_calibrations
    original_get = camera_class.__dict__['get_camera2worlds']

    def load_calibrations(self):
        # Preserve author intrinsics, load-size scaling and distortion handling.
        original_load(self)
        self.cam_to_worlds, self.capture_timestamps_ns = camera_to_reference(
            self.data_path, self.cam_id, self.start_timestep, self.end_timestep)
        self.camera_pose_time_basis = 'image_exposure_time'

    def get_camera2worlds(cls, data_path, cam_id, start_timestep, end_timestep):
        matrices, _ = camera_to_reference(data_path, cam_id, start_timestep, end_timestep)
        return matrices

    camera_class._gs_lab_original_load_calibrations = original_load
    camera_class._gs_lab_original_get_camera2worlds = original_get
    camera_class.load_calibrations = load_calibrations
    camera_class.get_camera2worlds = classmethod(get_camera2worlds)
    camera_class._gs_lab_capture_time_installed = True


def receipt(data_path):
    root = Path(data_path)
    return dict(camera_pose_time_basis='image_exposure_time',
        coordinate_reference='inverse ego-to-city at requested interval start, same reference as author objects and LiDAR',
        capture_manifest_sha256=hashlib.sha256((root/'capture_timestamps.json').read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pending='Object translations/rotations remain at LiDAR timestamps. No object-motion or rolling-shutter interpolation; no quality-improvement claim.')

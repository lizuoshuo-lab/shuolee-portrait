"""Opt-in AV2 initialization: heldout RGB and LiDAR do not seed Gaussians.

The camera projection follows DriveStudio. Sparse depth uses the nearest point
at each pixel instead of unordered duplicate-pixel assignment. Only training
timesteps may mark LiDAR points visible or assign their initial RGB. Camera
calibration and annotated object trajectories at all times remain known priors.
"""
import hashlib
import inspect

import torch
from tqdm import tqdm


def project_training_only(self, delete_out_of_view_points=True):
    if self.type != 'argoverse':
        raise ValueError('This adapter has only been validated for Argoverse 2')
    train, test, _, _ = self.split_train_test()
    train = set(map(int, train))
    test = set(map(int, test))
    assert train and not train & test
    assert train | test == set(range(self.num_img_timesteps))
    for cam in self.pixel_source.camera_data.values():
        if cam.undistort:
            raise ValueError('AV2 adapter expects the already undistorted source images')
        depth_maps = []
        for frame in tqdm(range(len(cam)), desc=f'Train-only initialization {cam.cam_name}', dynamic_ncols=True):
            time = self.pixel_source.normalized_time[frame]
            closest = self.lidar_source.find_closest_timestep(time)
            # Guard the synchronized-frame assumption before selecting train data.
            assert int(closest) == frame
            rays = self.lidar_source.get_lidar_rays(closest)
            points = rays['lidar_origins'] + rays['lidar_viewdirs'] * rays['lidar_ranges']
            intrinsic = torch.nn.functional.pad(cam.intrinsics[frame], (0, 1, 0, 1))
            intrinsic[3, 3] = 1.0
            projection = intrinsic @ cam.cam_to_worlds[frame].inverse()
            projected = (projection[:3, :3] @ points.T + projection[:3, 3:4]).T
            depth = projected[:, 2]
            pixels = projected[:, :2] / (depth[:, None] + 1e-6)
            valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < cam.WIDTH)
                     & (pixels[:, 1] >= 0) & (pixels[:, 1] < cam.HEIGHT) & (depth > 0))
            pixels = pixels[valid]
            depth_map = torch.full((cam.HEIGHT * cam.WIDTH,), float('inf'), device=self.device)
            flat_indices = pixels[:, 1].long() * cam.WIDTH + pixels[:, 0].long()
            depth_map.scatter_reduce_(0, flat_indices, depth[valid], reduce='amin', include_self=True)
            depth_map[~torch.isfinite(depth_map)] = 0
            depth_map = depth_map.reshape(cam.HEIGHT, cam.WIDTH)
            depth_maps.append(depth_map)
            # Preserve test-time depth for evaluation, but never read test RGB
            # or admit test-time LiDAR points to the initialization population.
            if frame in train:
                indices = torch.where(rays['lidar_mask'])[0][valid]
                self.lidar_source.visible_masks[indices] = True
                self.lidar_source.colors[indices] = cam.images[frame][pixels[:, 1].long(), pixels[:, 0].long()]
        cam.load_depth(torch.stack(depth_maps).to(self.device).float())
    self.gs_lab_initialization = dict(training_timesteps=sorted(train), heldout_timesteps=sorted(test),
        policy='training_RGB_and_training_timestep_visible_LiDAR_only',
        evaluation_depth='projected_before_filtering_and_retained_for_all_images',
        sparse_depth_collision='deterministic_nearest_positive_depth_per_pixel',
        known_priors='all camera calibrations, exposure poses and annotated object trajectories')
    if delete_out_of_view_points:
        self.lidar_source.delete_invisible_pts()
        assert not any(bool((self.lidar_source.timesteps == t).any()) for t in test)


def install(dataset_class):
    if dataset_class.__dict__.get('_gs_lab_train_only_installed', False):
        return
    dataset_class._gs_lab_original_project_lidar = dataset_class.project_lidar_pts_on_images
    dataset_class._gs_lab_original_projection_sha256 = hashlib.sha256(
        inspect.getsource(dataset_class.project_lidar_pts_on_images).encode()).hexdigest()
    dataset_class.project_lidar_pts_on_images = project_training_only
    dataset_class._gs_lab_train_only_installed = True

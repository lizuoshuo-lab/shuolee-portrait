"""Real AV2 heldout-image intervention, author negative control, and seed audit."""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path('/opt/gs-slam-lab/dynamic-human')
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB/'outputs/dynamic-human-street-005'
sys.path.insert(0, str(ROOT/'repos/drivestudio'))
os.chdir(ROOT/'repos/drivestudio')
from datasets.driving_dataset import DrivingDataset
from datasets.argoverse.argoverse_sourceloader import ArgoVerseCameraData
from street_capture_time_adapter import install as install_capture
from street_training_only_adapter import project_training_only


def digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def dataset_fingerprint(dataset):
    result = dict(lidar_xyz=digest(dataset.lidar_source.pts_xyz),
        lidar_rgb=digest(dataset.lidar_source.colors), lidar_time=digest(dataset.lidar_source.timesteps),
        train_images={str(c):digest(cam.images[dataset.train_timesteps])
                      for c,cam in dataset.pixel_source.camera_data.items()},
        depth_maps={str(c):digest(cam.lidar_depth_maps) for c,cam in dataset.pixel_source.camera_data.items()},
        aabb=digest(dataset.aabb))
    torch.manual_seed(73)
    xyz,rgb,_=dataset.get_lidar_samples(num_samples=100000,return_color=True)
    result['background_sample'] = [digest(xyz),digest(rgb)]
    result['objects'] = {}
    for kind in ('RigidNodes','DeformableNodes'):
        torch.manual_seed(73)
        nodes=dataset.get_init_objects(kind,instance_max_pts=5000,only_moving=True,
            traj_length_thres=0.5,exclude_smpl=False)
        result['objects'][kind]={str(i):{k:digest(v) if isinstance(v,torch.Tensor) else v
                                 for k,v in values.items()} for i,values in nodes.items()}
    return result


def audit_nearest_depth(dataset):
    """Separate NumPy reduction for every camera/frame, before LiDAR filtering."""
    changed = 0
    for cam in dataset.pixel_source.camera_data.values():
        for frame in range(len(cam)):
            rays = dataset.lidar_source.get_lidar_rays(frame)
            xyz = rays['lidar_origins'] + rays['lidar_viewdirs'] * rays['lidar_ranges']
            homogeneous = torch.nn.functional.pad(cam.intrinsics[frame], (0,1,0,1))
            homogeneous[3,3]=1
            matrix=homogeneous@cam.cam_to_worlds[frame].inverse()
            # Reuse projected coordinates to isolate the independently implemented collision reduction.
            p=(matrix[:3,:3]@xyz.T+matrix[:3,3:4]).T.numpy()
            uv=p[:,:2]/(p[:,2,None]+1e-6)
            keep=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cam.WIDTH)&(uv[:,1]>=0)&(uv[:,1]<cam.HEIGHT)
            ij=uv[keep].astype(np.int64)
            expected=np.full(cam.HEIGHT*cam.WIDTH,np.inf,dtype=np.float32)
            np.minimum.at(expected,ij[:,1]*cam.WIDTH+ij[:,0],p[keep,2])
            expected[~np.isfinite(expected)]=0
            expected=expected.reshape(cam.HEIGHT,cam.WIDTH)
            actual=cam.lidar_depth_maps[frame].numpy()
            np.testing.assert_array_equal(actual,expected)
            changed+=int(np.count_nonzero(cam._gs_lab_author_depth[frame].numpy()!=expected))
        del cam._gs_lab_author_depth
    return changed


def main():
    path=OUT/'training-only-initialization-audit.json'
    assert not path.exists(), 'Do not overwrite a completed audit'
    torch.set_num_threads(6)
    install_capture(ArgoVerseCameraData)
    cfg=OmegaConf.load(OUT/'full-scene-data-config.yaml')
    cfg.start_timestep=120
    cfg.end_timestep=140
    cfg.pixel_source.load_smpl=False
    original_project=DrivingDataset.project_lidar_pts_on_images
    control=[]
    fingerprints=[]
    depth_changes=[]
    started=time.perf_counter()
    for perturb in (False, True):
        def project(self, delete_out_of_view_points=True):
            _, test, _, _ = self.split_train_test()
            if perturb:
                for cam in self.pixel_source.camera_data.values():
                    for frame in test:
                        cam.images[frame] = torch.tensor([1.,0.,1.],device=self.device)
            # Negative control: the unmodified author method must reveal the intervention.
            original_project(self, delete_out_of_view_points=False)
            control.append(dict(rgb=digest(self.lidar_source.colors),
                training_rgb=digest(self.lidar_source.colors[~torch.isin(
                    self.lidar_source.timesteps,torch.as_tensor(test,device=self.device))]),
                depth={str(c):digest(cam.lidar_depth_maps) for c,cam in self.pixel_source.camera_data.items()}))
            for cam in self.pixel_source.camera_data.values():
                cam._gs_lab_author_depth=cam.lidar_depth_maps.clone()
            self.lidar_source.colors.fill_(1.)
            self.lidar_source.visible_masks.zero_()
            project_training_only(self,False)
            depth_changes.append(audit_nearest_depth(self))
            if delete_out_of_view_points:
                self.lidar_source.delete_invisible_pts()
        DrivingDataset.project_lidar_pts_on_images=project
        torch.manual_seed(73)
        np.random.seed(73)
        try:
            dataset=DrivingDataset(cfg)
        finally:
            DrivingDataset.project_lidar_pts_on_images=original_project
        assert list(map(int,dataset.test_timesteps)) == [10,20]
        fingerprints.append(dataset_fingerprint(dataset))
        for t in dataset.test_timesteps:
            assert dataset.lidar_source.get_lidar_rays(int(t))['lidar_ranges'].numel()==0
        dimensions=dict(frames=21,cameras=7,images=len(dataset.full_image_set),
            training_images=len(dataset.train_image_set),heldout_images=len(dataset.test_image_set),
            retained_lidar_points=dataset.lidar_source.num_points,
            rigid_instances=len(fingerprints[-1]['objects']['RigidNodes']),
            deformable_instances=len(fingerprints[-1]['objects']['DeformableNodes']))
        del dataset
        gc.collect()
    assert control[0]['rgb'] != control[1]['rgb'], 'Intervention must expose author initialization leakage'
    assert control[0]['training_rgb'] == control[1]['training_rgb']
    assert fingerprints[0] == fingerprints[1], 'Heldout RGB affected training initialization'
    report=dict(status='passed_real_heldout_RGB_noninterference_with_author_negative_control',
        source_global_frames=[120,140],heldout_global_frames=[130,140],
        perturbation='Replace all pixels in both heldout frames of all seven cameras with magenta',
        dimensions=dimensions, fingerprints=fingerprints,author_negative_control=control,
        author_depth_pixels_changed_by_nearest_hit=depth_changes,
        checks=['training RGB bitwise identical','filtered LiDAR geometry/colors/times bitwise identical',
            'sampled background seeds bitwise identical','rigid and deformable object seeds bitwise identical',
            'all sparse depth maps match independent NumPy minimum.at reduction','heldout LiDAR absent from initialization',
            'unmodified author initialization changes under the same heldout intervention'],
        seconds=time.perf_counter()-started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).with_name('street_training_only_adapter.py').read_bytes()).hexdigest(),
        limitations=['Initialization audit, not proof of a whole training/evaluation pipeline',
            'Local depth collision policy keeps the nearest hit; differs from unordered author pixel overwrite',
            'Camera/box trajectory priors remain known at evaluation times; not motion forecasting'])
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],dimensions=dimensions,seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    main()

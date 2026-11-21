"""Full 156-frame DriveStudio data construction; no SMPL mesh or training."""
from pathlib import Path
import hashlib
import io
import json
import os
import resource
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path('/opt/gs-slam-lab/dynamic-human')
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-street-005'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'
sys.path.insert(0,str(ROOT/'repos/drivestudio'))
os.chdir(ROOT/'repos/drivestudio')
from datasets.driving_dataset import DrivingDataset
from datasets.argoverse.argoverse_sourceloader import ArgoVerseCameraData
from pytorch3d.transforms import quaternion_to_matrix
from street_capture_time_adapter import install
from audit_street_inputs import PoseReader

def main():
    assert not (OUT/'full-scene-dataset-audit.json').exists()
    torch.set_num_threads(6)
    torch.manual_seed(73)
    np.random.seed(73)
    install(ArgoVerseCameraData)
    cfg=OmegaConf.load(ROOT/'repos/drivestudio/configs/datasets/argoverse/7cams.yaml').data
    cfg.data_root=str(SCENE.parent)
    cfg.start_timestep=0
    cfg.end_timestep=155
    cfg.pixel_source.downscale_when_loading=[4]*7
    cfg.pixel_source.test_image_stride=10
    cfg.preload_device='cpu'
    OmegaConf.save(cfg,OUT/'full-scene-data-config.yaml')
    source_pose=PoseReader(io.BytesIO((SCENE/'humanpose/smpl.pkl').read_bytes())).load()
    # The only joblib input is the previously audited author pose archive.
    # Keep author pose conversion, replacing pickle decoding with the restricted reader.
    import joblib
    original=joblib.load
    def read_pose(path,*args,**kwargs):
        if Path(path).resolve()!=(SCENE/'humanpose/smpl.pkl').resolve():
            raise ValueError(f'Unexpected pickle input: {path}')
        return source_pose
    joblib.load=read_pose
    started=time.perf_counter()
    try:
        dataset=DrivingDataset(cfg)
    finally:
        joblib.load=original
    assert dataset.num_img_timesteps==156
    assert len(dataset.full_image_set)==1092
    assert len(dataset.train_image_set)==987 and len(dataset.test_image_set)==105
    assert not set(dataset.train_indices)&set(dataset.test_indices)
    assert sorted(dataset.train_indices+dataset.test_indices)==list(range(1092))
    assert dataset.pixel_source.num_cams==7
    poses=dataset.pixel_source.smpl_human_all
    assert len(poses)==27
    instances=json.loads((SCENE/'instances/instances_info.json').read_text())
    reference_inverse=np.linalg.inv(np.loadtxt(SCENE/'ego_pose/000.txt'))
    max_rotation_error=0.
    max_translation_error=0.
    valid_count=0
    for instance,actual in poses.items():
        expected=source_pose[instance]
        assert torch.equal(actual['frame_valid'],expected['valid_mask'])
        for frame in torch.where(actual['frame_valid'])[0].tolist():
            camera=int(expected['selected_cam_idx'][frame])
            c2w=dataset.pixel_source.camera_data[camera].cam_to_worlds[frame]
            rotation=c2w[:3,:3]@expected['smpl']['global_orient'][frame,0]
            predicted=quaternion_to_matrix(actual['smpl_quats'][frame])
            error=max(float((predicted[0]-rotation).abs().max()),
                float((predicted[1:]-expected['smpl']['body_pose'][frame]).abs().max()))
            assert error<2e-6
            max_rotation_error=max(max_rotation_error,error)
            annotation=instances[str(instance)]['frame_annotations']
            index=annotation['frame_idx'].index(frame)
            translation=(reference_inverse@np.array(annotation['obj_to_world'][index]))[:3,3]
            error=float(np.max(np.abs(actual['smpl_trans'][frame].numpy()-translation)))
            assert error<1e-4
            max_translation_error=max(max_translation_error,error)
            valid_count+=1
    assert valid_count==2366
    assert torch.isfinite(dataset.aabb).all()
    assert dataset.lidar_source.num_points>0
    report=dict(status='full_scene_images_lidar_objects_and_body_pose_interface_verified',
        scene_frames=156,cameras=7,loaded_images=1092,train_images=987,test_images=105,
        heldout_timesteps=list(map(int,dataset.test_timesteps)),
        pose_instances=27,valid_pose_observations=valid_count,
        main_person_79_valid_frames=int(poses[79]['frame_valid'].sum()),
        max_body_rotation_error=max_rotation_error,max_body_translation_error=max_translation_error,
        visible_lidar_points=dataset.lidar_source.num_points,aabb=dataset.aabb.tolist(),
        seconds=time.perf_counter()-started,max_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((OUT/'full-scene-data-config.yaml').read_bytes()).hexdigest(),
        overrides=['capture-time camera adapter','restricted decoder for previously validated author pose archive'],
        initialization_boundary='Unmodified author projection colors LiDAR using all loaded images before train/test splitting. Do not claim heldout RGB independence for this audit. Training needs an explicit training-only color initialization adapter.',
        pending=['Official neutral template and mesh-to-image alignment','Training-only color initialization and noninterference test',
            'Actual joint street training and heldout person evaluation'],
        limitation='Dataset construction only; no body mesh, optimized street model or reconstruction accuracy verified')
    (OUT/'full-scene-dataset-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

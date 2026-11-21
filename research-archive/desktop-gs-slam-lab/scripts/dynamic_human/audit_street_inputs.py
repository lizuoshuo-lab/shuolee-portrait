"""Validate the exported AV2 / author human-pose interface without SMPL geometry."""
from pathlib import Path
from collections import OrderedDict
import io
import json
import pickle
import hashlib
import numpy as np
import torch

LAB = Path(__file__).resolve().parents[2]
OUT = LAB/'outputs/dynamic-human-street-005'
SCENE = LAB/'data/dynamic-human/argoverse2/processed/training/000'


class PoseReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return torch._utils._rebuild_tensor_v2
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        raise ValueError(f'Unexpected pickle global: {module}.{name}')


def main():
    receipt = json.loads((OUT/'preprocessing-receipt.json').read_text())
    frames = receipt['frames']
    instances = json.loads((SCENE/'instances/instances_info.json').read_text())
    timestamps = json.loads((SCENE/'capture_timestamps.json').read_text())['frames']
    assert len(timestamps) == frames == 156
    previous = -1
    for frame in timestamps:
        ts = int(frame['lidar_timestamp_ns'])
        assert ts > previous
        previous = ts
        assert len(frame['cameras']) == 7
        for cam in frame['cameras']:
            dt_ms = (int(cam['timestamp_ns'])-ts)*1e-6
            assert abs(dt_ms) <= 25 and dt_ms == cam['dt_from_lidar_ms']
            matrix = np.loadtxt(SCENE/cam['capture_pose'])
            np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
    for directory, suffix, count in [('images', '*.jpg', 1092), ('lidar', '*.bin', 156),
                                     ('ego_pose', '*.txt', 156), ('camera_pose_capture', '*.txt', 1092),
                                     ('intrinsics', '*.txt', 7), ('extrinsics', '*.txt', 7),
                                     ('dynamic_masks/all', '*.png', 1092),
                                     ('dynamic_masks/human', '*.png', 1092),
                                     ('dynamic_masks/vehicle', '*.png', 1092)]:
        assert len(list((SCENE/directory).glob(suffix))) == count, directory
    p = SCENE/'humanpose/smpl.pkl'
    poses = PoseReader(io.BytesIO(p.read_bytes())).load()
    assert isinstance(poses, dict)
    rows, archive, max_orthogonality_error, max_det_error = [], {}, 0., 0.
    for instance_id, data in poses.items():
        entry = instances[str(instance_id)]
        assert entry['class_name'] == 'PEDESTRIAN'
        valid = data['valid_mask'].numpy()
        cameras = data['selected_cam_idx'].numpy()
        assert valid.dtype == np.bool_ and valid.shape == (frames,)
        assert np.all((cameras[valid] >= 0) & (cameras[valid] <= 6))
        observed = set(entry['frame_annotations']['frame_idx'])
        assert set(np.flatnonzero(valid)).issubset(observed)
        arrays = {k: v.numpy() for k, v in data['smpl'].items()}
        assert arrays['betas'].shape == (frames, 10)
        assert arrays['global_orient'].shape == (frames, 1, 3, 3)
        assert arrays['body_pose'].shape == (frames, 23, 3, 3)
        for name, values in arrays.items():
            assert np.isfinite(values).all(), name
            archive[f'{instance_id}_{name}'] = values
        rotations = np.concatenate([arrays['global_orient'][valid], arrays['body_pose'][valid]], axis=1)
        error = float(np.abs(rotations.swapaxes(-1, -2) @ rotations-np.eye(3)).max())
        det_error = float(np.abs(np.linalg.det(rotations)-1).max())
        max_orthogonality_error = max(max_orthogonality_error, error)
        max_det_error = max(max_det_error, det_error)
        assert error < 1e-4 and det_error < 1e-4
        archive[f'{instance_id}_valid_mask'] = valid
        archive[f'{instance_id}_selected_cam_idx'] = cameras
        rows.append(dict(instance_id=int(instance_id), track_uuid=entry['id'], valid_frames=int(valid.sum()),
            annotated_frames=len(observed), selected_cameras=sorted(set(cameras[valid].tolist())),
            max_rotation_orthogonality_error=error, max_rotation_determinant_error=det_error))
    np.savez_compressed(OUT/'humanpose-validated-arrays.npz', **archive)
    target = 'ec986b20-004d-49f2-b9b0-d0d017bdeb1e'
    selected = next(row for row in rows if row['track_uuid'] == target)
    report = dict(status='structural_interface_passed_visual_body_alignment_requires_template',
        scene_frames=frames, images=1092, coarse_mask_files=3276,
        pose_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        body_pose_instances=len(rows), valid_pose_observations=sum(row['valid_frames'] for row in rows),
        annotated_pedestrian_tracks=receipt['pedestrian_instances'],
        max_rotation_orthogonality_error=max_orthogonality_error, max_rotation_determinant_error=max_det_error,
        instances=rows, target_pedestrian=selected,
        body_pose_convention='Global orientation in selected camera coordinates; body_pose 23 local rotation matrices; 10 shape coefficients; no model template or texture in this archive.',
        validation_scope='Every valid pose refers to an exported pedestrian ID and annotated frame; exact frame count and camera indices checked. IDs are author numeric IDs without embedded UUIDs. Structural agreement is not yet a visual person-to-mesh alignment test.',
        pending=['SMPL template and posed mesh projection', 'precise person and sky masks', 'street training and held-out evaluation'])
    (OUT/'input-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'instances'}, indent=2))


if __name__ == '__main__':
    main()

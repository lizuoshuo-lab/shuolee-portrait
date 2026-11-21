"""Execute actual author calibration methods on all exported street cameras.

AST isolation excludes unrelated GPU and SMPL imports. It is a calibration
interface test, not proof of a complete DriveStudio training environment.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image
from street_capture_time_adapter import install, receipt

LAB = Path(__file__).resolve().parents[2]
SCENE = LAB/'data/dynamic-human/argoverse2/processed/training/000'
OUT = LAB/'outputs/dynamic-human-street-005'
SOURCE = LAB/'repos/drivestudio/datasets/argoverse/argoverse_sourceloader.py'


def author_camera_class():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArgoVerseCameraData')
    node.bases = [ast.Name(id='object', ctx=ast.Load())]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ['load_calibrations', 'get_camera2worlds']]
    assert len(node.body) == 2
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(np=np, torch=torch, os=os, OPENCV2DATASET=np.eye(4))
    exec(compile(module, str(SOURCE), 'exec'), scope)
    return scope['ArgoVerseCameraData']


def main():
    cls = author_camera_class()
    original_load = cls.load_calibrations
    original_get = cls.get_camera2worlds
    install(cls)
    install(cls)  # Repeated launcher imports must not wrap recursively.
    rows = []
    for start, end in [(0, 156), (31, 61), (120, 150)]:
        reference = np.loadtxt(SCENE/'ego_pose'/f'{start:03d}.txt')
        for cam in range(7):
            with Image.open(SCENE/'images'/f'{start:03d}_{cam}.jpg') as image:
                width, height = image.size
            for factor in [1, 2]:
                item = cls()
                item.data_path = str(SCENE)
                item.cam_id = cam
                item.start_timestep, item.end_timestep = start, end
                item.original_size = [height, width]
                item.load_size = [height//factor, width//factor]
                original_load(item)
                old_intrinsics = item.intrinsics.clone()
                old_distortions = item.distortions.clone()
                old_poses = item.cam_to_worlds.clone()
                torch.testing.assert_close(original_get(str(SCENE), cam, start, end), old_poses, rtol=0, atol=0)
                item.load_calibrations()
                torch.testing.assert_close(item.intrinsics, old_intrinsics, rtol=0, atol=0)
                torch.testing.assert_close(item.distortions, old_distortions, rtol=0, atol=0)
                helper = cls.get_camera2worlds(str(SCENE), cam, start, end)
                torch.testing.assert_close(helper, item.cam_to_worlds, rtol=0, atol=0)
                assert item.capture_timestamps_ns.dtype == torch.int64
                assert torch.all(item.capture_timestamps_ns[1:] > item.capture_timestamps_ns[:-1])
                roundtrip_error = 0.
                for offset, frame in enumerate(range(start, end)):
                    # Restore city coordinates, independently compare the export.
                    restored = reference@item.cam_to_worlds[offset].numpy().astype(np.float64)
                    expected = np.loadtxt(SCENE/'camera_pose_capture'/f'{frame:03d}_{cam}.txt')
                    np.testing.assert_allclose(restored, expected, atol=2e-5, rtol=0)
                    roundtrip_error = max(roundtrip_error, float(np.abs(restored-expected).max()))
                rows.append(dict(camera=cam, interval=[start, end], downscale=factor,
                    frames=end-start, roundtrip_max_abs_error=roundtrip_error,
                    max_translation_change_m=float(torch.linalg.norm(item.cam_to_worlds[:, :3, 3]-old_poses[:, :3, 3], dim=1).max())))
    # Missing exposure poses must fail, never silently revert to LiDAR time.
    with tempfile.TemporaryDirectory(prefix='gs-lab-missing-capture-') as missing:
        try:
            cls.get_camera2worlds(missing, 0, 0, 1)
        except FileNotFoundError:
            missing_fails = True
        else:
            raise AssertionError('Missing capture data silently accepted')
    report = dict(status='calibration_entry_points_verified_full_training_not_run',
        source_file_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        adapter=receipt(SCENE), camera_interval_scale_cases=len(rows),
        actual_author_methods_executed=['load_calibrations', 'get_camera2worlds'],
        all_1092_capture_poses_checked=True, intrinsics_and_distortion_preserved=True,
        camera_ray_and_smpl_orientation_pose_entry_points_identical=True,
        missing_capture_data_fails=missing_fails, cases=rows,
        scope='AST isolation executes author methods and adapter with real calibration files, not full image loading or street training.')
    (OUT/'capture-time-adapter-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()

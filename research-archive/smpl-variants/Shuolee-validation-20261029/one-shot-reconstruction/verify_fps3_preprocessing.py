"""Read real demo inputs, verify fresh preprocessing, remove only temporary copies."""
from pathlib import Path
import datetime
import json
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

REPO = Path('/mnt/d/Projects/Shuolee')
sys.path.insert(0, str(REPO/'src'))
from shuolee_portrait import preprocessing as p, runtime as rt

EVIDENCE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/preprocess-fps3')
OUT = Path('/mnt/d/Projects/Shuolee-validation-20261029/one-shot-reconstruction/fps3-preprocessing-check')
VIDEO = Path('/mnt/d/00/工作/数据集/demo.mp4')
SOURCE = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
OUT.mkdir(exist_ok=True)
expected = rt.read(EVIDENCE/'plan.json')
source_inputs = rt.read(SOURCE/'data/inputs.json')
source_pose = rt.read(SOURCE/'data/pose-inputs.json')
expected_pixel = rt.read(EVIDENCE/'pixel-contract.json')
assert rt.sha(VIDEO) == expected['video_sha256']

with tempfile.TemporaryDirectory(prefix='smplx-fps3-check-') as directory:
    root = Path(directory); (root/'logs').mkdir(); (root/'data').mkdir()
    contract = p.extract_dual_resolution(VIDEO, root, dict(sample_fps=3,min_frames=12,max_frames=300))
    decoded, decoded_pose = rt.read(root/'decoded.json'), rt.read(root/'decoded-pose.json')
    assert decoded['source_indices'] == expected['source_indices']
    assert len(decoded['records']) == expected['frame_count'] == 59
    source_by_name = {r['frame']:r for r in source_inputs['records']}
    pose_by_name = {r['frame']:r for r in source_pose['records']}
    native_exact = pose_exact = 0
    rows, small_rows = [], []
    for actual, small in zip(decoded['records'], decoded_pose['records']):
        native, pose = source_by_name[actual['frame']], pose_by_name[small['frame']]
        for key in ('source_frame_index','time_seconds','target_time_seconds'):
            assert actual[key] == small[key] == native[key] == pose[key]
        assert actual['source_sha256'] == native['source_sha256'] == rt.sha(native['image'])
        assert small['source_sha256'] == pose['source_sha256'] == rt.sha(pose['image'])
        assert rt.sha(native['mask']) == native['mask_sha256']
        native_exact += 1; pose_exact += 1
        rows.append(dict(native, image=actual['image'], source_sha256=actual['source_sha256']))
        small_rows.append(dict(pose, image=small['image'], source_sha256=small['source_sha256'],
                               mask=None,mask_sha256=None))
    contract = p.camera_intrinsics(contract, expected_pixel['pose_K_integer_centers'])
    np.testing.assert_allclose(contract['GS_K_integer_centers'],expected_pixel['K_integer_centers'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(contract['GS_K_corner'],expected_pixel['K_corner'],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(contract['pose_to_GS_integer'],expected_pixel['pose_to_GS_integer'])
    rt.save(root/'pixel-contract.json',contract)
    rt.save(root/'data/inputs.json',dict(source_inputs,records=rows))
    rt.save(root/'data/pose-inputs.json',dict(source_pose,records=small_rows))
    shutil.copyfile(SOURCE/'data/observations.npz',root/'data/observations.npz')
    p.derive_pose_observations(root)
    with np.load(root/'data/pose-observations.npz',allow_pickle=False) as actual, \
         np.load(SOURCE/'data/pose-observations.npz',allow_pickle=False) as original, \
         np.load(SOURCE/'data/observations.npz',allow_pickle=False) as native:
        errors={k:float(np.max(abs(actual[k]-original[k]))) for k in ('keypoints','boxes','flip_disagreement')}
        assert max(errors.values()) < .001
        np.testing.assert_array_equal(actual['frame_ids'],native['frame_ids'])
        np.testing.assert_array_equal(actual['scores'],native['scores'])
        roundtrip=float(np.max(abs((actual['keypoints']+.5)*2-.5-native['keypoints'])))
        assert roundtrip < .001
    masks_exact = 0
    for actual in rt.read(root/'data/pose-inputs.json')['records']:
        original=pose_by_name[actual['frame']]
        with Image.open(actual['mask']) as a, Image.open(original['mask']) as b:
            np.testing.assert_array_equal(np.array(a),np.array(b))
        masks_exact += 1
    # Exercise the actual CPU camera matrix builder with no second image resize.
    sys.path.insert(0,str(REPO/'src/avatar_core'))
    from shuolee_portrait.training import camera
    from shuolee_portrait.sequence_pipeline import GAUSSIAN_CONFIG
    camera_errors=[]
    for i,row in enumerate(rows):
        _, record=camera(row,i,GAUSSIAN_CONFIG['training_height'],pose_index=i)
        assert record['wh']==row['wh']==[2704,1520]
        np.testing.assert_array_equal(record['K'],row['K'])
        camera_errors.append(record['projection_error_px'])
    import torch
    assert not torch.cuda.is_initialized()
    result=dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        video_sha256=rt.sha(VIDEO),source_pts_indices=decoded['source_indices'],frames=59,
        sample_fps=3,pose_wh=[1352,760],GS_wh=[2704,1520],
        exact_GS_image_files=native_exact,exact_pose_image_files=pose_exact,exact_pose_mask_pixels=masks_exact,
        pose_observation_difference_from_research_px=errors,GS_pose_roundtrip_max_px=roundtrip,
        actual_training_camera_projection_max_px=max(camera_errors),GS_training_resize_applied=False,
        detector_inference_repeated=False,camera_estimation_repeated=False,fitting_started=False,
        Gaussian_training_started=False,CUDA_initialized=False,VHAP_integrated=False,
        camera_calibration_accepted=False,quality_accepted=False,server_verified=False,
        source_inputs_read_only=True,temporary_frame_copies_removed_after_check=True,
        source_hashes={str(f.relative_to(REPO)):rt.sha(f) for f in
            [REPO/'src/shuolee_portrait'/name for name in
             ['preprocessing.py','video119.py','reconstruction.py','sequence_pipeline.py']]})
    rt.save(OUT/'result.json',result)
    rt.save(OUT/'pixel-contract.json',contract)
    print(json.dumps(result),flush=True)
assert not root.exists()

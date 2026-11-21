"""Independent PLY/COLMAP readers for the two real full orientation runs."""
from pathlib import Path
import hashlib, json
import numpy as np
import pycolmap
from plyfile import PlyData

art=Path('/mnt/d/Projects/Shuolee-validation-20261028')
receipt=json.loads((art/'orientation-validation.json').read_text())
assert receipt['state']=='passed'
reports={}
for orientation,job in receipt['jobs'].items():
    root=Path(job['run'])
    result=json.loads((root/'result.json').read_text())
    validation=json.loads((root/'export/validation.json').read_text())
    training=json.loads((root/'gaussian-run/result.json').read_text())
    fit=json.loads((root/'fit/progress.json').read_text())
    fit_gate=json.loads((root/'fit/geometry-gate.json').read_text())
    initialization=json.loads((root/'clothing-init/audit.json').read_text())
    run=json.loads((root/'run.json').read_text())
    inputs=json.loads((root/'data/training-inputs.json').read_text())['records']
    intrinsic=json.loads((root/'intrinsic-contract.json').read_text())
    preparation=json.loads((root/'preparation.json').read_text())
    assert preparation['from_scratch'] and preparation['observations_complete']
    assert not any(preparation[key] for key in ('old_subject_state_loaded','old_frames_loaded',
        'old_cameras_loaded','old_masks_loaded','old_observations_loaded'))
    ply=PlyData.read(root/'export/point_cloud.ply',mmap=False)['vertex'].data
    for field in ply.dtype.names:assert np.isfinite(ply[field]).all(),field
    sha=hashlib.sha256((root/'export/point_cloud.ply').read_bytes()).hexdigest()
    assert sha==result['sha256']==validation['sha256']
    assert result['state']=='completed' and result['steps']==training['steps']==18000
    assert fit['step']==fit['steps']==1200
    assert fit_gate['coarse_initialization_gate_passed'] is True
    assert initialization['points']==200000 and training['binding'] is False
    assert sum(training['per_frame_updates'])==18000
    assert min(training['per_frame_updates'])>0
    assert not any(run[key] for key in ('old_subject_loaded','old_fitted_person_loaded','old_gaussian_loaded','old_optimizer_loaded'))
    reconstruction=pycolmap.Reconstruction(str(root/'export/colmap'))
    cameras=json.loads((root/'export/cameras.json').read_text())
    expected_wh=[899,1600] if orientation=='portrait' else [1600,899]
    assert len(cameras)==len(inputs)==len(reconstruction.images)==result['frames']
    assert intrinsic['mode']=='rectified_center_crop' and intrinsic['orientation']==orientation
    K=np.array(intrinsic['raw_K_integer']);K[:2,2]+=.5
    for camera in cameras:
        assert [camera['width'],camera['height']]==expected_wh
        np.testing.assert_allclose([camera['fx'],camera['fy'],camera['cx'],camera['cy']],K[[0,1,0,1],[0,1,2,2]],atol=1e-9)
        R=np.array(camera['rotation'])
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-8)
        assert abs(np.linalg.det(R)-1)<1e-8
    for camera in reconstruction.cameras.values():
        assert [camera.width,camera.height]==expected_wh
        np.testing.assert_allclose(camera.calibration_matrix(),K,atol=1e-9)
    assert validation['format_passed'] is True
    assert all(p.suffix in ('.ply','.json','.txt') for p in (root/'export').rglob('*') if p.is_file())
    reports[orientation]=dict(state='passed',run=str(root),readers=['plyfile','pycolmap '+pycolmap.__version__],
        points=len(ply),properties=len(ply.dtype.names),sha256=sha,finite=True,frames=len(inputs),
        decoded_wh=expected_wh,exported_K=K.tolist(),steps=18000,fit_steps=1200,initial_points=200000,
        free_world_gaussians=True,fresh_subject_and_optimizer=True,
        full_camera_intrinsics_preserved=True,max_native_export_error=max(c['max_abs'] for c in validation['native_render_checks']),
        physical_quality_accepted=False,viewer_required=False)
out=dict(state='passed',source_git=receipt['source_git'],results=reports)
(art/'orientation-independent-exports.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

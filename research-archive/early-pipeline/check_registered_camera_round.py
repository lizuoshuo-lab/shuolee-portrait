from pathlib import Path
import hashlib,json,sys
import numpy as np
W=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
L=W if W.exists() else Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee/current'
A=B/'person-static-mask-foreground-fresh-070';R=B/'person-static-registered-camera-fresh-072'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(L/'repos/PGSR-shuolee'))
from camera_contract import require_registered_cameras
source=read(R/'source-inputs.json');assert source['from_scratch']
assert not any(source[k] for k in ['old_subject_fit_loaded','old_gaussian_loaded','old_optimizer_loaded'])
for item in source['inputs']:assert sha(R/item['file'])==sha(A/item['file'])==item['sha256']
frozen=read(R/'source-final/manifest.json')
for rel,digest in frozen['files'].items():
    assert sha(L/'repos'/rel)==sha(R/'source-final'/Path(rel).name)==digest,rel
fit_inputs=read(R/'data/inputs.json');train_inputs=read(R/'data/training-inputs.json')
old_fit=read(A/'data/inputs.json')['records']
counts=require_registered_cameras(train_inputs);require_registered_cameras(fit_inputs)
for r,s,a,f in zip(fit_inputs['records'],train_inputs['records'],read(A/'data/training-inputs.json')['records'],old_fit):
    for k in ['frame','source_frame_index','image','source_sha256','K','w2c','wh','camera_kind']:
        assert r[k]==s[k],(r['frame'],k)
    # The historical rough-body fit has masks only on original59; the static
    # Gaussian stage uses all175 masks. Preserve each stage's own070 inputs.
    for k in f:
        if k not in ('w2c','camera_kind'):assert r[k]==f[k],(r['frame'],'fit',k)
    for k in ['frame','image','mask','source_sha256','mask_sha256','K','wh','time_seconds','official_prediction_slot']:
        assert s[k]==a[k],(r['frame'],k)
assert sha(R/'data/training-inputs.json')==source['camera_manifest_sha256']
fit=read(R/'fit/protocol.json');init=read(R/'clothing-init/audit.json')
assert fit['steps']==1200 and fit['frames']==175 and fit['all_frames_in_each_update']
assert fit['calibrated_cameras']==59 and fit['registered_cameras']==116 and fit['interpolated_cameras']==0
assert fit['non_original_sfm_observation_weight']==.5
assert init['source_fit_sha256']==sha(R/'fit/fitted.npz')
points=np.load(R/'clothing-init/points.npz')['xyz']
assert points.shape==(200000,3) and np.isfinite(points).all()
assert init['from_scratch'] and init['training_steps']==0 and init['frames']==175
assert init['representation']=='shared_static_world' and not init['body_attraction']
assert not init['old_gaussian_loaded'] and not init['body_motion_loaded']
for name in ['xyz','point_ids','observations','rgb']:
    np.testing.assert_array_equal(np.load(A/'clothing-support/surface.npz')[name],np.load(R/'clothing-support/surface.npz')[name])
cfg=R/'gaussian-run/config.json';final=(R/'gaussian-run/result.json').exists()
if cfg.exists():
    x=read(A/'gaussian-run/config.json');y=read(cfg)
    assert x['options']==y['options'] and x['foreground_photometric']==y['foreground_photometric']
    assert x['alpha_composite']==y['alpha_composite'] and not y['alpha_composite']['enabled']
    assert x['steps']==y['steps']==18000 and x['max_height']==y['max_height']==1536
    assert y['camera_source']==counts and y['adapter_sha256']==sha(R/'source-final/shuolee_static.py')
    assert y['from_scratch'] and not y['old_optimizer_loaded'] and not y['old_gaussian_loaded']
    for raw,c in zip(train_inputs['records'],read(R/'gaussian-run/training-cameras.json')['records']):
        assert raw['w2c']==c['w2c']
        k=np.asarray(raw['K']).copy();k[0]*=c['width']/raw['wh'][0];k[1]*=c['height']/raw['wh'][1]
        np.testing.assert_allclose(k,c['K'],atol=1e-10,rtol=0)
if final:
    assert read(R/'gaussian-run/result.json')['steps']==18000
    np.testing.assert_array_equal(np.load(A/'gaussian-run/sampling-schedule.npy')[:,0],np.load(R/'gaussian-run/sampling-schedule.npy')[:,0])
proof=dict(reference='070',candidate='072',camera_counts=counts,
    raw_observations_and_official_predictions_unchanged=True,
    fit_and_training_camera_manifests_match=True,original59_sparse_clothing_support_unchanged=True,
    stage_masks_unchanged_from070=True,
    fit_mask_count=sum(r.get('mask') is not None for r in fit_inputs['records']),
    gaussian_mask_count=sum(r.get('mask') is not None for r in train_inputs['records']),
    fresh_fit_steps=1200,fresh_points=200000,new_initialization_point_sha256=sha(R/'clothing-init/points.npz'),
    initial_arrays_changed_downstream_of_camera_input=not np.array_equal(points,np.load(A/'clothing-init/points.npz')['xyz']),
    training_options_unchanged=cfg.exists(),actual_camera_projection_checked=cfg.exists(),
    final_primary_view_sampling_order_checked=final,
    note='Neighbor selections and fresh initialization depend on new extrinsics; this is a camera-input pipeline comparison.',
    quality_accepted=False)
(R/'controlled-protocol.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
print(json.dumps(proof),flush=True)

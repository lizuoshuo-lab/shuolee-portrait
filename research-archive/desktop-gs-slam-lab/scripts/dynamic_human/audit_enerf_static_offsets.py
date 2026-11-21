"""Verify the separate-time calibration and test its projection on nearby data."""
from pathlib import Path
import json,numpy as np
from audit_enerf_input import digest
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-static-calibration-037'
NEAR=LAB/'outputs/dynamic-human-enerf-static-camera-036'
ROOT=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
report=json.loads((OUT/'calibration.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
assert len(report['rows'])==18 and sorted(r['camera'] for r in report['rows'])==[f'{i:02d}' for i in range(18)]
for n,h in protocol['calibration_sha256'].items():assert digest(ROOT/'optimized'/n)==h
def stats(e):return dict(n=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)))
def project(X,K,w):
    a=np.c_[X,np.ones(len(X))]@w[:3].T@K.T
    return a[:,:2]/a[:,2:]
records=[];near=[]
for row in report['rows']:
    camera=row['camera'];shift=np.array(row['shift_xy_px'])
    assert digest(ROOT/'bkgd'/f'{camera}.jpg')==row['background_sha256']
    z=np.load(OUT/f'camera{camera}-frame000700-matches.npz')
    assert np.array_equal(shift,np.median((z['source_xy']-z['reference_xy'])[~z['check']],axis=0))
    for source in row['sources']:
        frame=source['frame'];assert frame in [700,710,720]
        assert digest(ROOT/'images'/camera/f'{frame:06d}.jpg')==source['rgb_sha256']
        assert digest(ROOT/'bgmtv2'/camera/f'{frame:06d}.jpg')==source['matte_sha256']
        z=np.load(OUT/f'camera{camera}-frame{frame:06d}-matches.npz')
        cells=np.floor(z['reference_xy']/[240,180]).astype(int)
        assert np.array_equal(z['check'],(cells[:,0]+3*cells[:,1])%5==0)
        choose=z['check'] if frame==700 else np.ones(len(z['check']),bool)
        diff=z['source_xy'][choose]-z['reference_xy'][choose]
        before=np.linalg.norm(diff,axis=1);after=np.linalg.norm(diff-shift,axis=1)
        expected=next(c for c in row['checks'] if c['frame']==frame)
        for values,key in [(before,'identity'),(after,'translation')]:
            assert len(values)==expected[key]['n'] and abs(np.median(values)-expected[key]['median_px'])<1e-10
            assert abs(np.quantile(values,.9)-expected[key]['p90_px'])<1e-10
        records.append(dict(camera=camera,frame=frame,before=stats(before),after=stats(after)))
    if camera not in ['00','06','12']:
        p=np.load(NEAR/f'camera{camera}-candidate.npz')
        for frame in [740,750,760]:
            z=np.load(NEAR/f'camera{camera}-frame{frame:06d}-matches.npz')
            uv=project(z['X'],p['K'],p['original_w2c']);target=z['y']
            near.append(dict(camera=camera,frame=frame,points=len(uv),
                pose_original=stats(np.linalg.norm(uv-target,axis=1)),pose_plus_frozen_offset=stats(np.linalg.norm(uv+shift-target,axis=1)),
                reference_identity=stats(np.linalg.norm(z['reference_xy']-target,axis=1)),reference_plus_frozen_offset=stats(np.linalg.norm(z['reference_xy']+shift-target,axis=1))))
aggregate={}
for frame in [700,710,720]:
    values=[r for r in records if r['frame']==frame]
    aggregate[str(frame)]=dict(before_median_of18_camera_medians_px=float(np.median([r['before']['median_px'] for r in values])),
        after_median_of18_camera_medians_px=float(np.median([r['after']['median_px'] for r in values])),
        cameras_improved=sum(r['after']['median_px']<r['before']['median_px'] for r in values))
near_aggregate={}
for frame in [740,750,760]:
    values=[r for r in near if r['frame']==frame]
    near_aggregate[str(frame)]={k:float(np.median([r[k]['median_px'] for r in values])) for k in ['pose_original','pose_plus_frozen_offset','reference_identity','reference_plus_frozen_offset']}
result=dict(status='54calibration_checks_and45nearby_static_checks_verified',calibration_checks=records,nearby_checks=near,
    aggregate=aggregate,nearby_median_of15_camera_medians_px=near_aggregate,
    fit_scope='Offsets fit only frame700non-check spatial cells. No refit using710/720or740/750/760. Nearby checks reuse036frozen static correspondences.',
    appearance_holdout_qualification='Static calibration photos from00/06/12 used at700/710/720. Future human training still excludes their appearance textures and masks.',
    interpretation='A stable local image-registration correction is supported; physical camera changes, moving-body alignment and training improvement are not established.',
    applied_to_training=False,goal_complete=False)
(OUT/'independent-calibration-audit.json').write_text(json.dumps(result,indent=2)+'\n')
candidate=dict(status='frozen_candidate_for_matched_model_control_not_applied',offsets={r['camera']:r['shift_xy_px'] for r in report['rows']},
    native_resolution=[1920,1080],apply='Add to decoded/cropped projectionKprincipalpoint only. Keep originalundistortionK/D, RGB/mattes, extrinsics, initialpointcloud and source-selection unchanged.',
    fit_frame=700,calibration_sha256=digest(OUT/'calibration.json'),protocol_sha256=digest(OUT/'protocol.json'),
    qualification='Static calibration photographs from all18cameras; original15/3appearance split retained. Empirical pixel offsets, not physically identified calibration.')
(OUT/'candidate-projection-offsets.json').write_text(json.dumps(candidate,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],aggregate=aggregate,nearby=near_aggregate)))

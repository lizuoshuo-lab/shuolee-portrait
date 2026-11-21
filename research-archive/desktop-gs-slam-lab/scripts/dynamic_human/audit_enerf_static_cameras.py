"""Recompute geometry metrics and compare simpler pixel-offset explanations."""
from pathlib import Path
import json,numpy as np,cv2
from audit_enerf_input import read_cameras,digest
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-static-camera-036'
ROOT=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
report=json.loads((OUT/'summary.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
assert len(report['rows'])==15 and all(r['status']=='candidate_measured_not_applied' for r in report['rows'])
cal=read_cameras(ROOT/'optimized');seed=LAB/'outputs/dynamic-human-enerf-background-032'
assert digest(seed/'background-init.npz')==protocol['source_geometry_sha256']
xyz=np.load(seed/'background-init.npz')['points'].astype(np.float64)
for name,h in protocol['calibration_sha256'].items():assert digest(ROOT/'optimized'/name)==h
def project(X,K,w):
    q=np.c_[X,np.ones(len(X))]@w[:3].T@K.T
    return q[:,:2]/q[:,2:]
def stats(e):return dict(n=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)))
def equal_stats(e,s):
    assert len(e)==s['n'] and abs(np.median(e)-s['median_px'])<1e-8 and abs(np.quantile(e,.9)-s['p90_px'])<1e-8
shifts={};rows=[]
for r in report['rows']:
    name=r['camera'];z=np.load(OUT/f'camera{name}-frame000750-matches.npz');p=np.load(OUT/f'camera{name}-candidate.npz')
    fit=~z['check'];shifts[name]=np.median(z['y'][fit]-project(z['X'][fit],p['K'],p['original_w2c']),axis=0)
for r in report['rows']:
    name=r['camera'];p=np.load(OUT/f'camera{name}-candidate.npz');K=p['K'];w0=p['original_w2c'];w1=p['candidate_w2c']
    assert np.array_equal(w0,np.array(cal[name]['w2c'])) and np.array_equal(K,np.array(cal[name]['K']))
    shared=np.median([s for c,s in shifts.items() if c!=name],axis=0)
    tests=[]
    for case in r['evaluations']:
        frame=case['frame'];source=next(x for x in r['sources'] if x['frame']==frame)
        assert digest(ROOT/'images'/name/f'{frame:06d}.jpg')==source['rgb_sha256']
        assert digest(ROOT/'bgmtv2'/name/f'{frame:06d}.jpg')==source['matte_sha256']
        z=np.load(OUT/f'camera{name}-frame{frame:06d}-matches.npz')
        assert np.array_equal(xyz[z['point_ids']],z['X'])
        cells=np.floor(z['reference_xy']/[240,180]).astype(int)
        assert np.array_equal(z['check'],(cells[:,0]+3*cells[:,1])%5==0)
        select=z['check'] if frame==750 else np.ones(len(z['X']),bool)
        uv=project(z['X'][select],K,w0);target=z['y'][select]
        e0=np.linalg.norm(uv-target,axis=1);e1=np.linalg.norm(project(z['X'][select],K,w1)-target,axis=1)
        equal_stats(e0,case['original_pose']);equal_stats(e1,case['candidate_pose'])
        # Descriptor matches are qualified before fitting; no candidate-based test filtering.
        shift_error=np.linalg.norm(uv+shifts[name]-target,axis=1)
        shared_error=np.linalg.norm(uv+shared-target,axis=1)
        tests.append(dict(frame=frame,original=stats(e0),PnP_candidate=stats(e1),per_camera_pixel_translation=stats(shift_error),leave_camera_out_shared_translation=stats(shared_error)))
    rows.append(dict(camera=name,pixel_translation_xy=shifts[name].tolist(),other_camera_median_translation_xy=shared.tolist(),checks=tests,
        rotation_change_deg=r['rotation_change_deg'],center_change_released_units=r['center_change_released_units'],geometry_smallest_largest_ratio=r['point_smallest_largest_ratio']))
aggregate={}
for frame in [750,740,760]:
    cases=[next(c for c in r['checks'] if c['frame']==frame) for r in rows]
    aggregate[str(frame)]={kind:dict(median_of_camera_medians_px=float(np.median([c[kind]['median_px'] for c in cases])),range_of_camera_medians_px=[min(c[kind]['median_px'] for c in cases),max(c[kind]['median_px'] for c in cases)]) for kind in ['original','PnP_candidate','per_camera_pixel_translation','leave_camera_out_shared_translation']}
result=dict(status='45case_source_and_geometry_metrics_verified_simpler_models_measured',rows=rows,aggregate=aggregate,
    qualification='Posthoc comparison of pose vs two-parameter image-offset models on identical predeclared fit/check splits. Not a physical calibration identification, and not a trained-person quality result.',
    heldout_appearance_images_read=0,applied_to_training=False,goal_complete=False)
(OUT/'independent-audit-and-offset-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],aggregate=aggregate,shifts={c:s.tolist() for c,s in shifts.items()})))

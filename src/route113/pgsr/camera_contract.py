"""Fail before GPU initialization if any training camera is unregistered."""
import numpy as np


def require_registered_cameras(document):
    rows=document['records']
    assert document.get('camera_registration_complete') is True, 'Per-frame camera registration is incomplete'
    assert document.get('accepted_for_training') is True, 'Camera registration has not passed its observation review'
    experimental=document.get('training_use')=='explicit_user_requested_full_scene_comparison'
    if experimental:
        assert document.get('user_requested_full_scene_person_pipeline') is True
        assert document.get('camera_accuracy_accepted') is False, 'Do not silently promote an experimental calibration'
        assert document.get('registration_audit',{}).get('registered')==len(rows)
    ids=[r['source_frame_index'] for r in rows]
    assert len(set(ids))==len(ids) and ids==sorted(ids), 'Duplicate or reordered source-frame mapping'
    counts={}
    for r in rows:
        kind=r['camera_kind'];counts[kind]=counts.get(kind,0)+1
        assert kind in ('calibrated','registered_background_pnp','fresh_full_scene_sfm'), 'Interpolated or unknown camera is not valid training calibration'
        assert r['frame']==f"raw_{r['source_frame_index']:06d}", 'Image frame and camera frame disagree'
        if kind=='calibrated':
            assert r['original_calibrated_index'] is not None, 'Missing original SfM camera index'
        elif kind=='registered_background_pnp':
            proof=r.get('camera_registration',{})
            assert proof.get('valid') is True and proof.get('interpolation_used') is False
            assert proof.get('method')=='background_EPNP_RANSAC_LM'
            assert proof.get('raw_source_frame')==r['source_frame_index']
            assert proof.get('fit_background_observations',0)>=80 and proof.get('held_observations',0)>=20
        else:
            proof=r.get('camera_registration',{})
            assert experimental and proof.get('valid') is True and proof.get('interpolation_used') is False
            assert proof.get('method')=='full_frame_sift_exhaustive_sfm_global_ba'
            assert proof.get('raw_source_frame')==r['source_frame_index']
            assert proof.get('registered_observations',0)>=80
        E=np.asarray(r['w2c']);K=np.asarray(r['K'])
        assert E.shape==(4,4) and K.shape==(3,3) and np.isfinite(E).all() and np.isfinite(K).all()
        assert np.max(np.abs(E[3]-[0,0,0,1]))<1e-8
        assert np.max(np.abs(E[:3,:3]@E[:3,:3].T-np.eye(3)))<1e-6 and abs(np.linalg.det(E[:3,:3])-1)<1e-6
        assert K[0,0]>0 and K[1,1]>0 and K[2,2]==1
    return dict(frames=len(rows),kinds=counts,interpolated_cameras=0,
        observation_review_accepted=not experimental,
        experimental_training_authorized=experimental,registered_views_verified=True)

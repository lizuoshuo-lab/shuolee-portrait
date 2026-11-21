"""Consume the existing N3 observations in a fixed-camera LOO evaluation.

This validates a material/geometry component, not complete camera calibration
or training readiness. No solve, coordinate correction, censoring, or GPU use.
"""
from collections import defaultdict
import math
import numpy as np
from material_observations122 import (
    validate_observation, point_reasons, camera_fit_permission,
    independent_face_admission, audit_family_splits,
)
from shared_background122 import project, triangulate


def checked_collection(observations):
    """One native measurement per source/track/frame, immutable family labels."""
    keys={}; tracks=defaultdict(list)
    for o in observations:
        validate_observation(o)
        if o.key in keys:
            raise ValueError('Duplicate observation identity: '+str(o.key))
        keys[o.key]=o
        tracks[(o.source_sha256,o.track_id)].append(o)
    if not audit_family_splits(observations)['passed']:
        raise ValueError('A frozen physical family crosses fit/holdout')
    for track, rows in tracks.items():
        if len({(o.frozen_family_id,o.frozen_split,o.group,o.width,o.height) for o in rows})!=1:
            raise ValueError('One track changes family, split, group or native size: '+str(track))
    return keys,tracks


def checked_cameras(K,cameras):
    K=np.asarray(K,dtype=float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Invalid complete native K')
    if not np.array_equal(K[2],[0.,0.,1.]) or K[0,1]!=0 or K[1,0]!=0:
        raise ValueError('This consumer supports complete zero-skew pinhole K only')
    out={}
    for frame, raw in cameras.items():
        E=np.asarray(raw,dtype=float)
        if type(frame) is not int or frame<0 or E.shape!=(4,4) or not np.isfinite(E).all():
            raise ValueError('Invalid camera frame or matrix')
        if not np.array_equal(E[3],[0.,0.,0.,1.]) or not np.allclose(E[:3,:3]@E[:3,:3].T,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(E[:3,:3])-1)>1e-8:
            raise ValueError('Camera must be a proper world-to-camera rigid pose')
        out[frame]=E.copy()
    return K.copy(),out


def canonical_triangulate(K,cameras,support):
    """Algebraic DLT in a deterministic local gauge, not a metric optimizer.

    The earliest supporting camera is the coordinate frame; the largest
    supporting baseline sets unit scale. This removes input-world Sim(3)
    dependence of homogeneous DLT with noisy measurements. No camera changes
    escape this function and no target observation determines this gauge.
    """
    reference=min(o.frame_id for o in support)
    E=cameras[reference]; inv=np.linalg.inv(E)
    local={o.frame_id:cameras[o.frame_id]@inv for o in support}
    baseline=max(float(np.linalg.norm(M[:3,3])) for M in local.values())
    if not math.isfinite(baseline) or baseline<=0:
        return np.full(3,np.nan)
    for M in local.values(): M[:3,3]/=baseline
    point=triangulate(K,local,[(o.frame_id,o.raw_uv) for o in support])*baseline
    return (point-E[:3,3])@E[:3,:3]


def predict_target(K,cameras,target,track,camera_fit_families,camera_evidence):
    """All other registered observations form XYZ. Never fit to the target."""
    support=[o for o in track if o.frame_id!=target.frame_id]
    row=dict(target_key=list(target.key),uv=None,
        triangulation_observation_keys=[list(o.key) for o in support],
        camera_fit_families=list(camera_fit_families),frozen_camera_evidence=camera_evidence,
        target_used_in_triangulation=False,numerical_reasons=[])
    missing=[o.frame_id for o in [target]+support if o.frame_id not in cameras]
    if missing: row['numerical_reasons'].append('missing_camera_frames:'+','.join(map(str,sorted(set(missing)))))
    if len(support)<2: row['numerical_reasons'].append('insufficient_other_views')
    if any(not o.tracker_numeric_validity or not np.isfinite(o.raw_uv).all() for o in [target]+support):
        row['numerical_reasons'].append('invalid_native_measurement')
    if row['numerical_reasons']: return row
    X=canonical_triangulate(K,cameras,support)
    if not np.isfinite(X).all(): row['numerical_reasons'].append('invalid_triangulation');return row
    depths=[float(project(K,cameras[o.frame_id],X[None])[1][0]) for o in [target]+support]
    if any(z<=0 or not math.isfinite(z) for z in depths):
        row['numerical_reasons'].append('nonpositive_depth');return row
    uv=project(K,cameras[target.frame_id],X[None])[0][0]
    if not np.isfinite(uv).all(): row['numerical_reasons'].append('nonfinite_projection');return row
    row.update(uv=uv.tolist(),positive_depth=True,point_world_diagnostic=X.tolist(),
               triangulation='DLT in earliest-support-camera and max-support-baseline gauge; not nonlinear geometric optimum')
    return row


def evaluate_material_geometry(K,cameras,observations,camera_fit_families,camera_evidence,
                               required_frames,source_sha256):
    """Recompute every target; material support checks apply to its whole family.

    `camera_evidence` binds a candidate file. It is not a physical camera pass.
    Evidence claims still require independent acquisition/review; code cannot
    manufacture their truth. Positive synthetic cases are component controls.
    """
    observations=list(observations)
    if any(o.source_sha256!=source_sha256 for o in observations):
        raise ValueError('Mixed source videos in one camera contract')
    keys,tracks=checked_collection(observations);K,cameras=checked_cameras(K,cameras)
    required=list(required_frames)
    if any(type(f) is not int or f<0 for f in required) or len(required)!=len(set(required)):
        raise ValueError('Required frame identities must be unique integers')
    fitset=set(camera_fit_families)
    if any(not isinstance(f,str) or not f for f in fitset): raise ValueError('Camera fit family identities required')
    contradictions=defaultdict(list)
    for o in observations:
        if o.material_identity_evidence=='contradicted': contradictions[o.frozen_family_id].append(list(o.key))
    results=[]
    for track in tracks.values():
        for target in track:
            pred=predict_target(K,cameras,target,track,sorted(fitset),camera_evidence)
            reasons=list(point_reasons(target))+list(pred['numerical_reasons'])
            if target.frozen_split not in {'heldout_background','heldout_human'}: reasons.append('not_heldout_observation')
            if target.frozen_family_id in fitset: reasons.append('heldout_family_used_for_camera_fit')
            if target.development_exposure: reasons.append('method_development_exposed')
            if target.static_evidence!='bounded_static': reasons.append('target_staticness_'+target.static_evidence)
            if not camera_evidence: reasons.append('camera_evidence_missing')
            if contradictions[target.frozen_family_id]: reasons.append('family_has_material_contradiction')
            support_issues=[]
            for key in pred['triangulation_observation_keys']:
                support=keys[tuple(key)]
                why=list(point_reasons(support))
                if support.static_evidence!='bounded_static': why.append('staticness_'+support.static_evidence)
                if support.development_exposure: why.append('method_development_exposed')
                if why: support_issues.append(dict(key=key,reasons=sorted(set(why))))
            if support_issues: reasons.append('triangulation_support_not_physically_validated')
            error=None if pred['uv'] is None else math.hypot(pred['uv'][0]-target.raw_uv[0],pred['uv'][1]-target.raw_uv[1])
            if error is None or not math.isfinite(error) or error>=1.: reasons.append('reprojection_not_strictly_below_1px')
            # Preserve the existing N3 necessary face checks, plus checks on all support.
            face=None
            if target.group=='face_candidate':
                face=independent_face_admission(target,pred if pred['uv'] is not None else None)
                reasons+=face['reasons']
            results.append(dict(observation=target.record(),prediction=pred,error_px=error,
                material_geometry_component_accepted=not reasons,reasons=sorted(set(reasons)),
                support_issues=support_issues,contradictions=contradictions[target.frozen_family_id],
                necessary_face_check=face,camera_fit_permission=camera_fit_permission(target)))
    missing=sorted(set(required)-set(cameras)); errors=[r['error_px'] for r in results if r['error_px'] is not None]
    return dict(scope='Material/geometry component only; not camera calibration, region coverage, body/staticity or training acceptance',
        required_frames=required,actual_registered_frames=sorted(cameras),missing_camera_frames=missing,
        observations=results,targets=len(results),numerical_predictions=len(errors),
        all_numerical_errors_strictly_lt1=bool(errors) and len(errors)==len(results) and max(errors)<1.,
        numerical_max_px=max(errors) if errors else None,
        material_geometry_component_accepted=bool(required and results) and not missing and all(r['material_geometry_component_accepted'] for r in results),
        physical_camera_accepted=False,training_admitted=False,training_started=False,
        all_raw_observations_retained=True)

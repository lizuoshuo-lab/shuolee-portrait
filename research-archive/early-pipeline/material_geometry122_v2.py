"""Fixed complete output cameras for support-only material prediction.

No fitting, pixel/image warping or physical-evidence promotion. The camera is
the exact corner-coordinate K/E/D record consumed by the native renderer.
"""
from collections import defaultdict
import copy, math
import numpy as np
from material_observations122 import point_reasons, camera_fit_permission, independent_face_admission
from material_geometry122 import checked_collection
from native_residual_camera122 import validate_camera, undistort_normalized, project_world
from native_residual_domain122 import sampled_domain_check

def checked_cameras(cameras):
    out={}
    for frame,c in cameras.items():
        if type(frame)is not int or frame<0 or c.get('id')!=str(frame):
            raise ValueError('Explicit integer frame and matching camera identity required')
        validate_camera(c)
        E=np.asarray(c['w2c'],dtype=float)
        if not np.allclose(E[:3,:3]@E[:3,:3].T,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(E[:3,:3])-1)>1e-8:
            raise ValueError('Admission requires a proper world-to-camera rigid pose')
        sampled_domain_check(c)
        out[frame]=copy.deepcopy(c)
    return out

def canonical_triangulate(cameras,support):
    reference=min(o.frame_id for o in support)
    E=np.asarray(cameras[reference]['w2c'],dtype=float); inv=np.linalg.inv(E)
    local={o.frame_id:np.asarray(cameras[o.frame_id]['w2c'],dtype=float)@inv for o in support}
    baseline=max(float(np.linalg.norm(M[:3,3])) for M in local.values())
    if not math.isfinite(baseline) or baseline<=0:return np.full(3,np.nan)
    rows=[]
    for o in support:
        c=cameras[o.frame_id]; K=np.asarray(c['K'],dtype=float)
        uv=np.asarray(o.raw_uv,dtype=float)+.5
        xy=undistort_normalized(((uv-K[:2,2])/[K[0,0],K[1,1]])[None],c['residual_projection'])[0]
        M=local[o.frame_id].copy();M[:3,3]/=baseline
        rows.extend([K[0,0]*(xy[0]*M[2]-M[0]),K[1,1]*(xy[1]*M[2]-M[1])])
    _,_,vt=np.linalg.svd(np.asarray(rows));q=vt[-1]
    if abs(q[3])<=1e-12:return np.full(3,np.nan)
    point=q[:3]/q[3]*baseline
    return (point-E[:3,3])@E[:3,:3]

def predict_target(cameras,target,track,camera_fit_families,camera_evidence):
    support=[o for o in track if o.frame_id!=target.frame_id]
    row=dict(target_key=list(target.key),uv=None,
        triangulation_observation_keys=[list(o.key) for o in support],
        camera_fit_families=list(camera_fit_families),frozen_camera_evidence=camera_evidence,
        target_used_in_triangulation=False,numerical_reasons=[],
        output_projection='opencv_brown12_output',reported_pixels='integer_centers',
        explicit_pixel_bridge='observed+0.5 to exact rendererK/E/D; projected-0.5 to raw observation')
    missing=[o.frame_id for o in [target]+support if o.frame_id not in cameras]
    if missing:row['numerical_reasons'].append('missing_camera_frames:'+','.join(map(str,sorted(set(missing)))))
    if len(support)<2:row['numerical_reasons'].append('insufficient_other_views')
    if any(not o.tracker_numeric_validity or not np.isfinite(o.raw_uv).all() for o in [target]+support):
        row['numerical_reasons'].append('invalid_native_measurement')
    if row['numerical_reasons']:return row
    try:
        X=canonical_triangulate(cameras,support)
        if not np.isfinite(X).all():raise ValueError('invalid_triangulation')
        depths=[]
        for o in [target]+support:
            E=np.asarray(cameras[o.frame_id]['w2c']);depths.append(float((E[:3,:3]@X+E[:3,3])[2]))
        if any(z<=0 or not math.isfinite(z) for z in depths):raise ValueError('nonpositive_depth')
        uv=project_world(X[None],cameras[target.frame_id])[0]-.5
        if not np.isfinite(uv).all():raise ValueError('nonfinite_projection')
    except (ValueError,np.linalg.LinAlgError) as e:
        row['numerical_reasons'].append(str(e));return row
    row.update(uv=uv.tolist(),positive_depth=True,point_world_diagnostic=X.tolist(),
        triangulation='Inverse residual rays, fx/fy weighted algebraicDLT in earliest-support-camera/max-support-baseline gauge; not nonlinear geometric optimum')
    return row

def evaluate_material_geometry(cameras,observations,camera_fit_families,camera_evidence,
                               required_frames,source_sha256):
    """Recompute every target; material support checks apply to its whole family.

    `camera_evidence` binds a candidate file. It is not a physical camera pass.
    Evidence claims still require independent acquisition/review; code cannot
    manufacture their truth. Positive synthetic cases are component controls.
    """
    observations=list(observations)
    if any(o.source_sha256!=source_sha256 for o in observations):
        raise ValueError('Mixed source videos in one camera contract')
    keys,tracks=checked_collection(observations);cameras=checked_cameras(cameras)
    if any(o.frame_id in cameras and (o.width,o.height)!=(cameras[o.frame_id]["width"],cameras[o.frame_id]["height"]) for o in observations):
        raise ValueError("Observation and camera raster mismatch")
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
            pred=predict_target(cameras,target,track,sorted(fitset),camera_evidence)
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

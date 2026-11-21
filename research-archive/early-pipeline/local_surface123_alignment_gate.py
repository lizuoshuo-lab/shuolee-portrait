"""Fail-closed admission for a *new* local static-person training run.

This validates a reviewed evidence contract and recomputes its face errors.
It cannot establish physical material identity from a boolean or a hash. Those
judgements remain an actual-review responsibility, bound to the exact inputs.
Historical coarse/body-only research decisions are deliberately insufficient.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math
import numpy as np

SCHEMA = 'shuolee-alignment-admission/1'
CAMERA_SCHEMA = 'shuolee-camera-admission/1'
ARTIFACTS = ('decoded.json', 'data/training-inputs.json',
             'camera/solver-output.npz', 'fit/fitted.npz', 'alignment-preregistration.json')
UNCERTAINTY_METHODS = {'independent_repeat_annotation', 'controlled_physical_target',
                       'independent_physical_measurement'}


class Rejected(ValueError):
    pass


def need(condition, code):
    if not condition:
        raise Rejected(code)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array(value, shape, code):
    a = np.asarray(value, dtype=np.float64)
    need(a.shape == shape and np.isfinite(a).all(), code)
    return a


def unique_map(rows, field, code):
    need(isinstance(rows, list) and len(rows) > 0, code+'_empty')
    keys = [row[field] for row in rows]
    need(all(isinstance(k, str) and k for k in keys) and len(set(keys)) == len(keys), code+'_duplicate_or_invalid')
    return dict(zip(keys, rows))


def input_path(root, value):
    """Read the same run through its Windows WSL mount for read-only audits."""
    path=Path(value)
    if path.is_file():
        return path
    linux_prefix='/opt/gs-slam-lab/shuolee-local-surface-123/'+root.name+'/'
    if value.startswith(linux_prefix):
        return root/value[len(linux_prefix):]
    return path


def evaluate(root, *, phase='fit', allow_synthetic_contract_test=False):
    """Read only; synthetic tests can never receive training_allowed=True."""
    root = Path(root).resolve()
    need(phase in {'camera','fit'}, 'unknown_gate_phase')
    schema=CAMERA_SCHEMA if phase=='camera' else SCHEMA
    artifacts=tuple(name for name in ARTIFACTS if phase=='fit' or name!='fit/fitted.npz')
    result = dict(schema=schema, phase=phase, created=datetime.now(timezone.utc).isoformat(),
                  subject=root.name, state='denied', contract_passed=False,
                  fitting_allowed=False, training_allowed=False, geometry_quality_proven_by_this_tool=False,
                  reasons=[], files_read=[], face_observation_errors=[])
    bound_files = []
    try:
        receipt_path = root/('reviews/camera-alignment-admission.json' if phase=='camera' else 'reviews/alignment-admission.json')
        need(receipt_path.is_file(), 'camera_alignment_receipt_missing' if phase=='camera' else 'fine_alignment_receipt_missing')
        receipt = read(receipt_path)
        need(receipt.get('schema') == schema, 'unsupported_admission_schema')
        need(receipt.get('subject') == root.name, 'subject_identity_mismatch')
        synthetic = receipt.get('purpose') == 'synthetic_contract_test'
        need(receipt.get('purpose') in {'physical_alignment', 'synthetic_contract_test'}, 'unknown_evidence_purpose')
        need(not synthetic or allow_synthetic_contract_test, 'synthetic_evidence_not_training_admission')
        bound_files.append((receipt_path, sha(receipt_path)))
        expected_hashes = receipt['artifacts']
        need(set(expected_hashes) == set(artifacts), 'artifact_binding_incomplete')
        for name in artifacts:
            path = root/name
            need(path.is_file(), 'artifact_missing:'+name)
            digest = sha(path)
            need(digest == expected_hashes[name], 'artifact_hash_mismatch:'+name)
            bound_files.append((path, digest))

        def evidence(key):
            link = receipt['evidence'][key]
            path = (root/link['path']).resolve()
            need(path.is_relative_to(root/'reviews'), 'evidence_outside_reviews:'+key)
            need(path.is_file(), 'evidence_missing:'+key)
            digest = sha(path)
            need(digest == link['sha256'], 'evidence_hash_mismatch:'+key)
            bound_files.append((path, digest))
            return read(path), digest

        roster, roster_sha = evidence('roster')
        measurements, measurements_sha = evidence('measurements')
        review, _ = evidence('review')
        prereg=read(root/'alignment-preregistration.json')
        need(prereg.get('roster_sha256') == roster_sha, 'preregistered_roster_changed')
        need(prereg.get('subject') == root.name and prereg.get('frozen_before_error_evaluation') is True,
             'invalid_preregistration_identity_or_order')
        prereg_time=datetime.fromisoformat(prereg['created'])
        evaluation_time=datetime.fromisoformat(measurements['evaluation_started'])
        need(prereg_time.tzinfo is not None and evaluation_time.tzinfo is not None and prereg_time < evaluation_time,
             'preregistration_not_before_evaluation')
        need(prereg.get('camera_sha256') == expected_hashes['camera/solver-output.npz']
             and prereg.get('training_inputs_sha256') == expected_hashes['data/training-inputs.json'],
             'preregistration_camera_or_input_changed')
        for key in ['actual_visual_review', 'identity_checked_before_errors',
                    'held_family_independence_reviewed', 'uncertainty_adequate_for_subpixel_claim',
                    'all_training_frames_registered_reviewed']:
            need(review.get(key) is True, 'review_missing_or_failed:'+key)
        need(review.get('decision') == 'approved', 'fine_alignment_review_not_approved')
        need(review.get('artifacts') == expected_hashes, 'review_artifact_binding_mismatch')
        need(review.get('roster_sha256') == roster_sha and review.get('measurements_sha256') == measurements_sha,
             'review_evidence_binding_mismatch')
        need(roster.get('pixel_domain') == 'native_video_integer_centers', 'non_native_or_ambiguous_pixel_domain')
        need(roster.get('camera_model') == 'rectified_pinhole', 'camera_model_not_supported_by_this_gate')
        need(roster.get('frozen_before_projection_evaluation') is True, 'roster_not_frozen_before_errors')
        need(roster.get('no_error_based_exclusions') is True, 'error_based_exclusion_or_unknown')

        decoded = read(root/'decoded.json')['records']
        rows = read(root/'data/training-inputs.json')['records']
        frames = [row['frame'] for row in rows]
        need(len(frames) == len(set(frames)) and frames == [r['frame'] for r in decoded], 'training_frame_identity_or_order_mismatch')
        need(frames == roster['all_registered_frame_ids'], 'roster_frame_coverage_mismatch')
        need(frames == prereg['all_registered_frame_ids'], 'preregistered_frame_roster_mismatch')
        frame_map = dict(zip(frames, range(len(rows))))
        with np.load(root/'camera/solver-output.npz', allow_pickle=False) as camera:
            K = array(camera['K'], (3,3), 'invalid_camera_K')
            E = array(camera['w2c'], (len(rows),4,4), 'invalid_camera_array')
            fit_tracks = {str(x) for x in camera['track_ids'].tolist()}
        need(K[0,0] > 0 and K[1,1] > 0 and np.allclose(K[2], [0,0,1]), 'invalid_pinhole_K')
        renderer_K = K.copy();renderer_K[:2,2] += .5
        for i, (row, dec) in enumerate(zip(rows, decoded)):
            need(all(row[k] == dec[k] for k in ['source_frame_index','source_sha256','wh']), 'source_frame_binding_mismatch:'+frames[i])
            need(row['camera_registration'].get('interpolation_used') is False, 'interpolated_or_unverified_camera:'+frames[i])
            need(np.allclose(array(row['K_integer_centers'],(3,3),'invalid_row_K'), K, atol=1e-9,rtol=0), 'row_K_mismatch:'+frames[i])
            need(np.allclose(array(row['K'],(3,3),'invalid_renderer_K'), renderer_K,atol=1e-9,rtol=0), 'pixel_center_contract_mismatch:'+frames[i])
            need(np.allclose(array(row['w2c'],(4,4),'invalid_row_pose'), E[i],atol=1e-9,rtol=0), 'row_pose_mismatch:'+frames[i])
            need(np.allclose(E[i,3],[0,0,0,1]) and np.allclose(E[i,:3,:3].T@E[i,:3,:3],np.eye(3),atol=1e-5)
                 and abs(np.linalg.det(E[i,:3,:3])-1) < 1e-5, 'invalid_w2c_rotation:'+frames[i])
            for path_key,hash_key in [('image','source_sha256'),('mask','mask_sha256')]:
                path=input_path(root,row[path_key])
                need(path.is_file(), 'input_file_missing:'+frames[i]+':'+path_key)
                digest=sha(path)
                need(digest==row[hash_key], 'input_file_hash_mismatch:'+frames[i]+':'+path_key)
                bound_files.append((path,digest))

        families = unique_map(roster['families'], 'id', 'family_roster')
        checks = unique_map(roster['checks'], 'id', 'held_observation_roster')
        points = unique_map(measurements['points'], 'family_id', 'static_points')
        need(set(points) == set(families), 'static_point_family_coverage_mismatch')
        declared = measurements['evaluated_observation_ids']
        need(len(declared) == len(set(declared)) and set(declared) == set(checks), 'omitted_or_duplicate_held_observations')
        need(measurements.get('camera_artifact_sha256') == expected_hashes['camera/solver-output.npz'], 'point_fit_camera_binding_mismatch')
        need(measurements.get('point_estimation_used_check_observations') is False, 'held_used_for_point_estimation')
        alias_fit = fit_tracks | set(review.get('camera_fit_family_aliases', []))
        for ident, family in families.items():
            need(family.get('physical_identity_reviewed') is True and family.get('static_material_reviewed') is True, 'unverified_material:'+ident)
            need(family.get('used_for_camera_fit') is False, 'held_family_used_for_camera_fit:'+ident)
            identifiers = {ident, *family['source_track_ids'], *family['alias_family_ids']}
            need(not identifiers.intersection(alias_fit), 'fit_held_family_leakage:'+ident)
            point = points[ident]
            need(point.get('estimation') in {'independent_physical_measurement','triangulated_from_registered_anchors'}, 'unknown_point_estimation:'+ident)
            anchors = point.get('anchor_observation_ids', [])
            need(not set(anchors).intersection(checks), 'check_observation_is_anchor:'+ident)
            if point['estimation'] == 'triangulated_from_registered_anchors':
                need(len(anchors) >= 2 and len(set(anchors)) == len(anchors), 'insufficient_or_duplicate_anchors:'+ident)
            array(point['xyz_world'], (3,), 'nonfinite_point:'+ident)

        observed_parts, observed_frames = set(), set()
        for ident, obs in checks.items():
            family = families[obs['family_id']]
            need(family['part'] in {'eyes','nose','mouth'}, 'non_face_check_in_face_roster:'+ident)
            frame = obs['frame'];need(frame in frame_map, 'unknown_observation_frame:'+ident)
            i = frame_map[frame]
            xy = array(obs['xy'],(2,),'invalid_observation:'+ident)
            w,h = rows[i]['wh'];need(0<=xy[0]<w and 0<=xy[1]<h, 'observation_outside_native_image:'+ident)
            uncertainty = obs['localization_uncertainty_px']
            need(isinstance(uncertainty,(float,int)) and math.isfinite(uncertainty) and uncertainty>=0, 'unknown_localization_uncertainty:'+ident)
            need(obs.get('uncertainty_method') in UNCERTAINTY_METHODS, 'uncertainty_not_independent_physical_evidence:'+ident)
            xyz = E[i,:3,:3]@np.asarray(points[obs['family_id']]['xyz_world']) + E[i,:3,3]
            need(xyz[2] > 0, 'nonpositive_camera_depth:'+ident)
            p = K@xyz;uv=p[:2]/p[2]
            error=float(np.linalg.norm(uv-xy))
            result['face_observation_errors'].append(dict(id=ident,frame=frame,family=obs['family_id'],
                error_px=error,projected=uv.tolist(),localization_uncertainty_px=uncertainty))
            observed_parts.add(family['part']);observed_frames.add(frame)
        errors=[x['error_px'] for x in result['face_observation_errors']]
        result.update(face_count=len(errors),face_max_error_px=max(errors),
                      face_median_error_px=float(np.median(errors)),all_face_errors_strictly_lt1=all(e<1. for e in errors))
        need(result['all_face_errors_strictly_lt1'], 'face_observation_at_or_above_1px')
        need(observed_parts == {'eyes','nose','mouth'}, 'eyes_nose_mouth_coverage_incomplete')
        coverage=review['face_coverage']
        need(coverage.get('actual_review') is True and coverage.get('adequate') is True, 'face_view_coverage_not_accepted')
        need(set(coverage['observed_frame_ids']) == observed_frames, 'face_coverage_frames_mismatch')
        unobserved=set(coverage['unobserved_frame_ids'])
        need(unobserved.isdisjoint(observed_frames) and unobserved|observed_frames==set(frames), 'unobserved_views_not_reported')
        need(coverage['view_groups'] and all(v and set(v).issubset(observed_frames) for v in coverage['view_groups'].values()), 'view_group_coverage_missing')

        for scope in ['full_body','background']:
            guard,_ = evidence(scope)
            need(guard.get('actual_visual_review') is True and guard.get('passed') is True, 'protection_failed:'+scope)
            need(guard.get('artifacts') == expected_hashes, 'protection_hash_mismatch:'+scope)
            need(guard.get('independent_validation') is True and guard.get('no_error_based_exclusions') is True, 'protection_not_independent:'+scope)
            need(guard.get('protocol_and_limits') and guard.get('observations_or_report') and guard.get('reviewed_frame_ids')
                 and set(guard['reviewed_frame_ids']).issubset(frames), 'protection_evidence_incomplete:'+scope)
            if scope=='full_body':
                need(set(guard['assessed_regions']) >= {'torso','hands','feet','occlusions','eyes','nose','mouth'}, 'fullbody_region_coverage_incomplete')
                if phase=='fit':
                    need(guard.get('current_smplx_overlay_reviewed') is True, 'current_smplx_overlay_not_reviewed')
        need(all(sha(p)==h for p,h in bound_files), 'evidence_changed_during_validation')
        result.update(state='synthetic_contract_passed_not_training' if synthetic else 'reviewed_contract_passed',
                      contract_passed=True,fitting_allowed=not synthetic and phase=='camera',
                      training_allowed=not synthetic and phase=='fit')
    except (Rejected, KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        result['reasons'].append(str(exc) if isinstance(exc,Rejected) else 'malformed_or_unreadable_evidence:'+repr(exc))
    result['files_read']=[dict(path=str(p),sha256=h) for p,h in bound_files]
    return result


def require_alignment_admission(root):
    result=evaluate(root)
    if not result['training_allowed']:
        raise RuntimeError('Fine alignment not admitted: '+'; '.join(result['reasons']))
    return result


def require_camera_admission(root):
    result=evaluate(root,phase='camera')
    if not result['fitting_allowed']:
        raise RuntimeError('Camera alignment not admitted before fitting: '+'; '.join(result['reasons']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--phase',choices=['camera','fit'],default='fit')
    args=parser.parse_args();result=evaluate(args.root,phase=args.phase)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    raise SystemExit(0 if (result['fitting_allowed'] if args.phase=='camera' else result['training_allowed']) else 3)

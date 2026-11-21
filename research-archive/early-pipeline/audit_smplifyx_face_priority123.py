"""Read-only saved-file, seed, regional and original-score checks.

Reuse the existing whole-sequence criteria. The extra check reconstructs the
body/feet-only objective difference independently from saved 2D projections.
"""
from datetime import datetime, timezone
import inspect
import json
import numpy as np
import audit_gaussianavatars_params123 as a
import audit_smplifyx_forward_guard123 as forward
from audit_smplifyx_data_terms123 import image_terms, load


def invoke(module):
    source = inspect.getsource(module.main)
    assert 'guarded-root-sequence' in source
    source = source.replace('guarded-root-sequence', 'face-priority-guarded-sequence')
    scope = dict(module.main.__globals__)
    exec(compile(source, '<unchanged-selection-and-regional-criteria>', 'exec'), scope)
    scope['main']()


def main():
    source = inspect.getsource(forward.main)
    assert 'forward-guarded-sequence' in source
    source = source.replace('forward-guarded-sequence', 'face-priority-guarded-sequence')
    scope = dict(forward.main.__globals__, invoke=invoke)
    exec(compile(source, '<same-chronological-seed-audit>', 'exec'), scope)
    scope['main']()
    body = a.REC/'demo-vhap-initialization/fullbody-smplifyx'
    root = body/'face-priority-guarded-sequence'
    plan_path = body/'face-priority-guarded-sequence-plan.json'
    plan = a.read(plan_path)
    base = load(body/'full-sequence/fit/fitted.npz')
    ids = list(base['frame_ids'].astype(str))
    inputs = {r['frame']: r for r in a.read(body/'full-sequence/data/training-inputs.json')['records']}
    rows = []
    for frame in plan['processing_order']:
        folder = root/'frames'/frame
        if not a.lp(folder/'result.json').exists():
            break
        r = a.read(folder/'result.json')
        d = a.read(folder/'objective-selection.json')
        anchor = frame == plan['anchor']
        assert r['original_stage_indices'] == (list(range(5)) if anchor else [3, 4])
        assert r['body_observation_amplitude'] == (1. if anchor else 2.)
        candidate = load(folder/'candidate-00-temporal.npz')
        before = load(body/'full-sequence/frames'/frame/'winner-camera-space.npz')
        for key in ('faces', 'betas', 'K_integer_centers', 'camera_translation'):
            assert np.array_equal(candidate[key], before[key]), (frame, key)
        # Raw candidates are camera-space arrays; w2c is attached by export.
        # Check the saved export separately, without inventing a raw w2c field.
        saved = load(folder/'winner-camera-space.npz')
        assert np.array_equal(saved['w2c'], before['w2c'])
        homogeneous = candidate['joints_camera'] @ candidate['K_compute_float32'].T
        projected = homogeneous[..., :2]/homogeneous[..., 2:]
        projection_error = float(np.max(np.abs(projected-candidate['joints_projected_native'])))
        assert projection_error < .01, (frame, projection_error)
        target = base['observed_native_keypoints'][ids.index(frame)]
        terms = image_terms(candidate['joints_projected_native'], target, inputs[frame]['wh'][1])
        weighted = d['seeded_upstream_reported_objective']
        reconstructed_original = weighted if anchor else weighted-3*(terms['body_data']+terms['feet_data']+terms['face_data'])
        error = reconstructed_original-d['original_saved_objectives'][0]
        # Same established float32 replay tolerance as the independent-cache oracle.
        assert abs(error) < .01, (frame, error)
        if not anchor:
            assert r['selection_uses_original_body_and_face_amplitudes']
            assert r['fit_weighted_objective'] == weighted
            assert r['hand_stage_values_unchanged'] and not r['face_stage_values_unchanged']
            assert r['face_observation_amplitude']==4. and r['face_schedule_fit']==[0.,4.]
            assert r['original_body_numeric_weights_unchanged'] is False
        rows.append(dict(frame=frame,body_amplitude=r['body_observation_amplitude'],
            original_stages=r['original_stage_indices'],weighted_fit_objective=weighted,
            reconstructed_original_objective=reconstructed_original,
            saved_original_objective=d['original_saved_objectives'][0],error=error,
            original_body_data=terms['body_data'],original_feet_data=terms['feet_data'],original_face_data=terms['face_data'],face_amplitude=r['face_observation_amplitude'],
            projection_max_abs_px=projection_error,export_w2c_unchanged=True,
            candidate_sha256=a.sha(folder/'candidate-00-temporal.npz')))
    result = dict(updated=datetime.now(timezone.utc).isoformat(),
        state='all59_original_score_normalization_verified' if len(rows)==59 else 'partial_original_score_normalization_verified',
        frames=len(rows),rows=rows,max_abs_error=max((abs(r['error']) for r in rows),default=None),
        plan_sha256=a.sha(plan_path),source_sha256=a.sha(__file__),
        no_candidate_or_source_changes=True,no_gpu=True,quality_accepted=False,goal_complete=False)
    a.save(root/'original-score-normalization-audit.json',result)
    print(json.dumps({k:result[k] for k in ('state','frames','max_abs_error')}))


if __name__=='__main__':
    main()

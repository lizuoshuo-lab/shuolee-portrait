"""One registered CPU/source audit. Does not import the fitter or run inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

TOP = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = TOP / 'shuolee/current/local-clothed-surface-123'
ROOT = Path('/opt/gs-slam-lab/shuolee-local-surface-123')
SRC = ROOT / 'runtime/src/shuolee_portrait'
H = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
OUT = REC / 'observation-visibility-consumer-audit'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    if OUT.exists():
        raise RuntimeError('Registered audit already has an output directory; inspect it instead of rerunning.')
    plan = REC / 'observation-visibility-consumer-plan.json'
    assert read(plan)['budget']['audits'] == 1
    fitting = SRC / 'fitting.py'
    source_hash = sha(fitting)
    snapshot = read(REC / 'runtime-source-manifest.json')
    manifest_hash = next(x['sha256'] for x in snapshot['files'] if x['path'] == 'src/shuolee_portrait/fitting.py')
    assert source_hash == manifest_hash
    historical = []
    for subject in ['winterhat', 'whiteT', 'grayT']:
        protocol = ROOT / subject / 'fit/protocol.json'
        p = read(protocol)
        assert p['source_sha256'] == source_hash
        prior = REC / 'smplx-mapping-audit' / (subject + '-result.json')
        s = read(prior)
        historical.append(dict(subject=subject, protocol_sha256=sha(protocol),
            actual_source_same_as_current=True, anatomical_visibility=p['anatomical_visibility'],
            saved_face_visibility_all_one=s['saved_face_surface_visibility_all_one'],
            reused_audit_sha256=sha(prior), numerical_projection_error=s['independent_projection_max_abs_px'],
            nonzero_face_detector_residual=s['face_nonzero_weight_native_error'],
            new_fit=False, new_projection_or_occlusion_computation=False))
    coverage = REC / 'field-preflight/coverage'
    info = read(coverage / 'observations.json')
    with np.load(coverage / 'observations.npz', allow_pickle=False) as arr:
        fields = {k: list(arr[k].shape) for k in arr.files}
        slot = next(i for i, r in enumerate(info['rows']) if r['source_frame_index'] == 440)
        face = arr['keypoints'][slot, 23:91]
        face_inside = int(((face[:, 0] >= 0) & (face[:, 0] < 2704) & (face[:, 1] >= 0) & (face[:, 1] < 1520)).sum())
    assert 'flip_disagreement' not in fields
    production = ROOT / 'field'
    missing = [str(p.relative_to(production)) for p in [
        production / 'data/observations.npz', production / 'data/inputs.json',
        production / 'fit/protocol.json', production / 'fit/observation-weights.npz'
    ] if not p.exists()]
    assert len(missing) == 4
    evidence = []
    needles = ['q90=', 'rel=', 'flip=', 'face_weight', 'anatomy_mode=',
               'torch.ones_like(facw)', 'face=wl', 'loss=body', 'groups=',
               'body=model(', 'face=(v', "return q[...,:2]"]
    for p in [fitting, SRC / 'observations.py', SRC / 'observation_support.py', H / 'local_surface123.py']:
        lines = p.read_text(encoding='utf-8').splitlines()
        selected = [dict(line=i + 1, text=line) for i, line in enumerate(lines)
                    if any(x in line for x in needles)]
        evidence.append(dict(path=str(p), sha256=sha(p), relevant_lines=selected))
    result = dict(updated=datetime.now(timezone.utc).isoformat(), state='complete_read_only',
        registered_plan_sha256=sha(plan), actual_fitter_source_sha256=source_hash,
        frozen_manifest_match=True, actual_historical_consumers=historical,
        projection_semantics=dict(detector='Independent 2D RTMW detections on the source image.',
            model='Barycentric landmarks of the fitted SMPL-X vertices, projected using each saved K and w2c, minus 0.5.',
            same_mesh_reprojection_agrees_with_checkpoint=True,
            residual_is_detector_to_model_not_two_renderers=True,
            verified_scope='Frozen three subject diagnostic projections only; not arbitrary viewer/export paths.'),
        active_face_contract=dict(template_landmarks=51, active_landmarks=29,
            excluded_zero_based_slots=list(range(10)) + list(range(19, 31)),
            remaining_regions='nose and mouth', jaw_and_expression_optimized=False,
            shared_static_pose=True, soft_objective_coefficient=0.15,
            image_mask_support_is_anatomical_visibility=False,
            coefficient_alone_proves_cause=False),
        field_negative_control=dict(source_frame=440, slot=slot,
            observation_sha256=sha(coverage / 'observations.npz'), observation_fields=fields,
            face_predictions_inside_raster=face_inside,
            physical_visibility='Back of head: facial surface not visible in the previously reviewed source image.',
            visibility_review_sha256=sha(coverage / 'visual-review.json'),
            production_files_missing=missing,
            actual_fitting_weight=None,
            weight_replay_executed=False,
            reason='ROI-only unflipped pass lacks production flip disagreement, camera/mask inputs and fitted weights; no fabricated defaults.'),
        conclusions=[
            'Historical visibility deficiency is confirmed on the identical actual consumer, not newly discovered or newly repaired.',
            'Field source440 predictions have not entered an SMPL-X fit; future exposure to the same policy is a risk, not observed training causality.',
            'Detected and projected points are different quantities; their discrepancy is the actual fitting residual.',
            'Physical K/E, observational correspondence, static-person assumption and model/optimization effects remain non-unique causes.'
        ], no_inference=True, no_camera_solve=True, no_parameters_changed=True,
        no_training=True, quality_accepted=False,
        next='Integrate explicit observed visibility/material evidence before a future fit; do not repeat this audit, infer visibility from a wrong mesh, or manufacture missing field weights.')
    OUT.mkdir()
    save(OUT / 'source-evidence.json', evidence)
    save(OUT / 'result.json', result)
    print(json.dumps(dict(result=str(OUT / 'result.json'), source=source_hash,
        old_consumers=len(historical), field_weight_replay=False, no_training=True), ensure_ascii=False))

if __name__ == '__main__':
    main()

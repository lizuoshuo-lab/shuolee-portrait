"""Register one bounded, read-only cross-person intake of existing observations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

R = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
O = R/'generic-2d3d-research/adapter-a0b'
L = '/opt/gs-slam-lab/shuolee-local-surface-123'
M = '/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')

assert not O.exists(), 'Read the registered experiment; do not register it twice.'
previous = read(R/'generic-2d3d-research/adapter-a0/result.json')
assert all(sha(P/'src/shuolee_portrait'/name) == value for name, value in previous['actual_modules'].items())
rows = read(R/'queue-state.json')['subjects']
specs = []
for row in rows:
    ident = row['id']
    spec = dict(id=ident, label=row['label'], expected_frames=row['selected_frames'])
    if ident in ('winterhat', 'whiteT', 'grayT'):
        spec.update(frames=f'{L}/{ident}/data/inputs.json', rows_key='records',
                    wholebody=f'{L}/{ident}/data/observations.npz', names_key='joint_names',
                    face=f'{L}/{ident}/data/observations.npz', face_convention='coco_wholebody_133',
                    coordinates_key='keypoints', metadata=f'{L}/{ident}/data/inputs.json',
                    metadata_rows_key='records', reviews=None,
                    camera_usage='Historical camera records for format inspection only; never a new candidate or initialization.')
    elif ident in ('pattern', 'field', 'indoor'):
        base = f'{M}/{ident}-preflight'
        spec.update(frames=f'{base}/coverage/observations.json', rows_key='rows',
                    wholebody=f'{base}/coverage/observations.npz', names_key='keypoint_names')
        if ident in ('field', 'indoor'):
            spec.update(face=f'{base}/semantic-face-all/observations.npz', face_convention='ibug68',
                        coordinates_key='fan_native_continuous', metadata=f'{base}/semantic-face-all/observations.json',
                        metadata_rows_key='rows', reviews=f'{base}/semantic-face-all/real-2d-smplx-bindings.json')
        else:
            spec.update(face=spec['wholebody'], face_convention='coco_wholebody_133', coordinates_key='keypoints',
                        metadata=spec['frames'], metadata_rows_key='rows', reviews=None)
        spec['camera_usage'] = 'No new camera computation or pose substitution.'
    else:
        spec.update(state='not_available_for_this_intake',
                    reason='Full registered 94-frame observation stream absent from this round. Historical 24-frame diagnostics cannot replace it.',
                    prior_source_audit=f'{M}/remaining-native-input-audit/result.json')
    specs.append(spec)
O.mkdir()
plan = dict(created=datetime.now(timezone.utc).isoformat(), stage='A0b',
    hypothesis='The frozen generic consumer can preserve existing native pixels and named SMPL-X correspondences across subjects; missing camera or observation evidence must remain explicit.',
    variable='Only declared input files/person; one unchanged production source and one unchanged fitter configuration.',
    baseline_source=previous['actual_modules'],
    config=dict(fit_backend='smplx_direct_v1', fit_steps=1200, alignment_inputs='alignment-inputs.json', direct_face_binding_policy='reviewed_anatomical_v1'),
    source_policy='Use already complete FAN68 when explicitly present, otherwise the existing named COCO133 face slice. No provider is selected by fit errors. Provider differences are reported and this is not an accuracy comparison.',
    missing_review_policy='Generate an explicit UNKNOWN/UNASSIGNED inventory, never a positive review, confidence, physical identity or uncertainty. Every generated record remains inactive.',
    camera_policy='Historical cameras are inspected only. No new camera or training inputs are constructed; no old geometry or optimizer is loaded.',
    budget=dict(subjects=len(specs), cpu_seconds_per_subject=120, new_detector_runs=0, camera_solves=0, fitting_runs=0, GPU_jobs=0, training_runs=0),
    stop_conditions=['Any hash/frame/order/semantic/pixel inconsistency stops that subject without repairing coordinates.',
                     'Do not create absent boxes, flip disagreement, cameras, confidence or annotation evidence.',
                     'Any production source mutation or training artifact creation fails the whole intake.',
                     'No real fitting unless the existing independent admission gate passes.'],
    subjects=specs, acceptance='Input/call-path compatibility only; cannot certify physical calibration or reconstruction quality.')
save(O/'plan.json', plan)
print(json.dumps(dict(plan=str(O/'plan.json'), subjects=len(specs), frozen_modules=len(previous['actual_modules'])), ensure_ascii=False))

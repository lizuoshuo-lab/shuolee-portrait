"""Read-only source/data-flow audit; no detector, fitting or GPU imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

H = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
T = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = T/'shuolee/current/local-clothed-surface-123'
O = REC/'generic-2d3d-research'
P = Path('/opt/gs-slam-lab/shuolee-local-surface-123/runtime')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

spec = [
 ('runtime_cli', P/'src/shuolee_portrait/cli.py', ["elif name=='fit'", 'from .fitting import main']),
 ('runtime_fitter', P/'src/shuolee_portrait/fitting.py', ['face_weight=', 'face_weight[:,19:31]', 'body=model(', 'def project(', 'clamp_min(.01)', 'facw*face_vis', 'anatomy_mode=', 'face_vis=torch.ones']),
 ('runtime_pixel_contract', P/'src/shuolee_portrait/camera_pixel_contract.py', ['adding 0.5', 'corner =', 'K_integer_centers=', 'K_renderer_corners=']),
 ('main_orchestrator', H/'local_surface123.py', ["PROJECT/'run.py'", "for name in ['fit','fit-gate']"]),
 ('camera_output', H/'local_surface123_camera_reviewed.py', ['Kc=K.copy()', 'K_integer_centers=', 'coordinate_contract=']),
 ('preparation_entry', H/'prepare_native_person122.py', ['def build_fit_inputs', 'native_camera(', 'from shuolee_portrait.fitting import main as fit']),
 ('candidate_fitter', REC/'direct-smplx-bindings/candidate_fitting.py', ['load_reviewed_face', 'jaw=param', 'expression=param', 'jaw_pose=jaw', 'direct_face(', 'face_world(']),
 ('candidate_entry', REC/'direct-smplx-bindings/run_candidate.py', ['require_camera_admission', 'candidate_fitting.py', 'm.main(']),
 ('direct_binding_module', H/'smplx_direct_correspondence123.py', ['def face_bindings', 'lmk_faces_idx', 'lmk_bary_coords', 'def project_integer_centres', 'def load_reviewed_face', 'observations.npz', "obs['keypoints']", 'localization_sigma_px']),
 ('manual_visibility_review', H/'review_indoor_semantic_all123.py', ['ranges=[', '(0,11,', '(68,99,', 'fan_native_continuous', "split='UNASSIGNED'", 'training_eligible=False']),
 ('camera_observation_entry', H/'local_surface123_frozen_observations.py', ['def require_fit_support', "expected=list(range(0,subject['source_frames'],5))", "if s['development_only']", 'all_nodes_actually_reviewed', 'localization_uncertainty_px']),
 ('material_sampling', H/'field_material_topology123.py', ["seg['development_only']=seg['family_id'] in selected"]),
]
files = []
for name,p,patterns in spec:
    lines = p.read_text(encoding='utf-8').splitlines()
    evidence=[]
    for pattern in patterns:
        hits=[i for i,line in enumerate(lines) if pattern in line]
        evidence.append(dict(pattern=pattern, found=bool(hits), locations=[dict(line=i+1, text=lines[i], following=lines[i+1:i+3]) for i in hits]))
    files.append(dict(id=name,path=str(p),sha256=sha(p),lines=len(lines),evidence=evidence))

manifest=read(REC/'direct-smplx-bindings/candidate-manifest.json')
consumer=read(REC/'observation-visibility-consumer-audit/result.json')
face=read(REC/'indoor-preflight/semantic-face-all/review-result.json')
bindings=read(REC/'indoor-preflight/semantic-face-all/real-2d-smplx-bindings.json')
records=bindings.get('records',bindings.get('bindings',[]))
assert records, list(bindings)
actual_runtime_sha=sha(P/'src/shuolee_portrait/fitting.py')
assert actual_runtime_sha==manifest['baseline_sha256']==consumer['actual_fitter_source_sha256']
assert sha(REC/'direct-smplx-bindings/candidate_fitting.py')==manifest['candidate_sha256']

result=dict(updated=datetime.now(timezone.utc).isoformat(),state='read_only_source_audit_complete',
    files=files, production_source_modified=False, GPU_imported=False,
    fitting_executed=False, feature_extraction=False, new_training=False,
    candidate=dict(**manifest, separate_entry=True, main_pipeline_calls_legacy_fitter=True),
    current_indoor_bindings=dict(records=len(records), eligible=sum(bool(x.get('training_eligible')) for x in records),
       splits=sorted({x.get('split','missing') for x in records}), detector_field='fan_native_continuous',
       main_fitter_field='data/observations.npz:keypoints[:,40:91]',
       automatic_connection_verified=False),
    known_historical_consumers=consumer['actual_historical_consumers'],
    exclusions=['Source analysis does not prove the unique cause of ghosting.',
      'Legacy K corner +0.5 and projected -0.5 are a paired convention, not an established bug.',
      'The separate candidate has synthetic/input validation, not a real-person result.',
      'Manual visibility review is diagnostic annotation; it must not become per-subject production logic.',
      'Current admission constraints are recorded, not weakened by this audit.'])
out=O/'local-code-audit.json'
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(files),runtime_same=True,bindings=result['current_indoor_bindings'],new_training=False),ensure_ascii=False))

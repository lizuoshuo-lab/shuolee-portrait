"""Bounded CPU A0 acceptance; no fitting or detector inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys, time
import numpy as np

P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=R/'generic-2d3d-research/adapter-a0'
sys.path.insert(0,str(P/'src'))
from shuolee_portrait.alignment_inputs import load_alignment_inputs, sha, read
from shuolee_portrait.fit_backend import inspect_inputs
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(p,**kwargs):return dict(path=str(p),sha256=sha(p),**kwargs)
start=time.monotonic()
log=O/'cpu-tests.log'
assert not log.exists(), 'Inspect the existing result instead of repeating tests.'
with log.open('x') as stream:
    run=subprocess.run([sys.executable,str(P/'scripts/check_alignment_inputs.py')],stdout=stream,stderr=subprocess.STDOUT,timeout=120)
save(O/'test-process.json',dict(updated=datetime.now(timezone.utc).isoformat(),exit_code=run.returncode,log=str(log),seconds=time.monotonic()-start))
print(log.read_text(),flush=True)
if run.returncode:raise SystemExit(run.returncode)

# Reuse the previous known-geometry observations; no new optimization.
old=R/'direct-smplx-bindings/synthetic-input-fixture/data'
synthetic=dict(schema='shuolee-alignment-input/1',purpose='synthetic_contract_test',camera_model='rectified_pinhole',
    frames=link(old/'inputs.json'),
    face=link(old/'observations.npz',convention='coco_wholebody_133',coordinates_key='keypoints',coordinate_domain='native_integer_pixel_centres'),
    reviews=link(old/'smplx-face-observations.json',format='reviewed_anatomical_v1'))
save(O/'synthetic-adapter-inputs.json',synthetic)
b=load_alignment_inputs(O/'synthetic-adapter-inputs.json')
import torch, smplx
torch.set_num_threads(4)
modelpath=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
model=smplx.SMPLX(str(modelpath),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False)
objective,receipt=b.make_face_objective(model,device='cpu')
pred=(objective.uv+2.).detach().requires_grad_(True)
loss,regions=objective(pred);loss.backward()
g=pred.grad.detach().numpy();fit=objective.fit_weight.detach().numpy()>0
assert np.all(g[~fit]==0) and np.all(np.linalg.norm(g[fit],axis=-1)>0)
assert len(receipt['bindings'])==51 and set(regions)=={'brows','nose','eyes','mouth'}
for slot,anchor in enumerate(receipt['bindings']):assert anchor['ibug68_index']==slot+17 and anchor['rtmw133_index']==slot+40
save(O/'synthetic-consumer-result.json',dict(known_observations_reused=True,template_sha256=sha(modelpath),
    frames=len(b.rows),exact_target_consumed=bool(np.array_equal(objective.uv.detach().numpy(),b.face_uv.astype(np.float32))),
    bindings=51,active_regions=list(regions),held_gradient_zero=True,active_fit_gradient_nonzero=True,
    new_optimization_steps=0,GPU_used=False,physical_accuracy_evidence=False))

# Intake the real complete native stream. Missing cameras and review fields
# remain missing; this creates a tiny manifest, not a fabricated run dataset.
pre=R/'indoor-preflight';face=pre/'semantic-face-all'
intake=O/'real-native-intake';intake.mkdir(exist_ok=True)
manifest=dict(schema='shuolee-alignment-input/1',purpose='real',camera_model='rectified_pinhole',
    frames=link(pre/'coverage/observations.json',rows_key='rows'),
    wholebody=link(pre/'coverage/observations.npz',convention='coco_wholebody_133',names_key='keypoint_names',coordinate_domain='native_integer_pixel_centres'),
    face=link(face/'observations.npz',convention='ibug68',coordinates_key='fan_native_continuous',coordinate_domain='native_integer_pixel_centres',
              metadata=link(face/'observations.json',rows_key='rows')),
    reviews=link(face/'real-2d-smplx-bindings.json',format='native_face_bindings_v1'))
save(intake/'alignment-inputs.json',manifest)
save(intake/'config.json',dict(fit_backend='smplx_direct_v1',fit_steps=1200,alignment_inputs='alignment-inputs.json',direct_face_binding_policy='reviewed_anatomical_v1'))
bundle,report=inspect_inputs(intake)
with np.load(face/'observations.npz') as z:assert np.array_equal(bundle.face_uv,z['fan_native_continuous'][:,17:])
with np.load(pre/'coverage/observations.npz') as z:assert np.array_equal(bundle.body['keypoints'],z['keypoints'])
assert report['frames']==177 and report['fixed_face_observations']==9027
assert report['active_fit']==0 and report['active_held']==0 and not report['fitting_allowed']
assert report['camera_records']==0 and not (intake/'fit').exists()
save(O/'real-intake-result.json',report)

before=read(O/'before-source-hashes.json')
preserved={name:sha(P/Path(name.replace('\\','/')))==digest for name,digest in before.items() if name.replace('\\','/')!='src/shuolee_portrait/cli.py'}
assert all(preserved.values())
# Inspect the actual call path and candidate's use of the loaded bundle.
import ast
module=ast.parse((P/'src/shuolee_portrait/fitting_direct.py').read_text())
src=(P/'src/shuolee_portrait/fitting_direct.py').read_text()
assert 'bundle.make_face_objective(model' in src and 'bundle.rows' in src and 'obs=bundle.body' in src
assert 'load_reviewed_face(' not in src and 'smplx_direct_correspondence123' not in src
assert 'torch=c.runtime()' not in src.split('def _fit_prepared')[0]
for token in ('indoor177','winterhat112','field177','grass94','range(0,source_frames,5)'):
    for name in ('alignment_inputs.py','fit_backend.py','fitting_direct.py','smplx_correspondence.py','reviewed_admission.py'):
        assert token not in (P/'src/shuolee_portrait'/name).read_text(),(name,token)
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='A0_interface_checks_passed_real_fitting_not_admitted',
    unit_and_CLI_tests=15,test_exit_code=0,actual_CLI_preflight_checked=True,actual_fit_dispatch_denied_before_assets=True,
    direct_candidate_uses_selected_stream=True,body_and_face_source_files_unchanged=True,
    previous_source_files_preserved=preserved,synthetic_loss_consumer_checked=True,
    real_frames=177,real_fixed_face_observations=9027,real_active_fit_observations=0,
    real_cameras=0,real_fitting_started=False,Gaussian_training_started=False,
    source_code_scanned_for_subject_rules=True,seconds=time.monotonic()-start,
    quality_accepted=False,scope='Portable input/dispatch adaptation only; no real geometry or unique root-cause claim.')
assert result['seconds']<120,'A0 CPU wall-time budget exceeded'
save(O/'result.json',result)
print(json.dumps(result,ensure_ascii=False),flush=True)

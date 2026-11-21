"""Build a separately identified real fitter candidate; never edit the frozen runtime."""
from pathlib import Path
import hashlib,json,ast
H=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'direct-smplx-bindings'
BASE=Path('/opt/gs-slam-lab/shuolee-local-surface-123/runtime/src/shuolee_portrait/fitting.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((OUT/'plan.json').read_text())
assert sha(BASE)==plan['baseline_fitter_sha256']
target=OUT/'candidate_fitting.py'
assert not target.exists()
s=BASE.read_text()
def replace(a,b):
    global s
    if s.count(a)!=1:raise ValueError('Ambiguous/missing source edit: '+a[:70])
    s=s.replace(a,b)
replace('from .runtime import c, ASSETS','from .runtime import c, ASSETS\nfrom smplx_direct_correspondence123 import load_reviewed_face, face_world')
replace("cfg=c.read(root/'config.json')", "cfg=c.read(root/'config.json')\n    if cfg.get('direct_face_binding_policy') != 'reviewed_anatomical_v1':\n        raise ValueError('This candidate requires explicit reviewed anatomical observations.')")
old="""    face_weight=rel[:,40:91].copy();face_weight[:,:10]=0 # exclude brow/eye and glasses areas
    face_weight[:,19:31]=0
    face_weight*=camw[:,None]"""
replace(old,"""    direct_face,binding_review=load_reviewed_face(root,rows,obs,model,device='cuda')
    # Model visibility may be logged, but cannot rewrite independently reviewed
    # observation states or turn unknown detections into measured constraints.
    face_weight=direct_face.fit_weight.detach().cpu().numpy()
    c.save(out/'effective-face-bindings.json',binding_review)""")
replace("feet_weight*=support[:,17:23];hands_weight*=support[:,91:133];face_weight*=support[:,40:91]",
        "feet_weight*=support[:,17:23];hands_weight*=support[:,91:133]\n    # No post-review mask/error-based deletion of facial observations.")
replace("hd=param((1,2,45))", "hd=param((1,2,45))\n    jaw=param((1,3));expression=param((1,10))")
replace("[shared,bd,hd,latent]", "[shared,bd,hd,latent,jaw,expression]")
replace("body=model(body_pose=pose,betas=beta,left_hand_pose=lh,right_hand_pose=rh)",
        "body=model(body_pose=pose,betas=beta,left_hand_pose=lh,right_hand_pose=rh,jaw_pose=jaw,expression=expression)")
replace("face=(v[:,face_ids]*face_bary[None,:,:,None]).sum(2);uf,_=project(face)",
        "face=face_world(v,model);uf,facecam=project(face)")
replace("corr=corr)","corr=corr,facecam=facecam,jaw=jaw,expression=expression)")
replace("face=wl(st['uf'],target[:,40:91],facw*face_vis,6*scale)",
        "face,face_regions=direct_face(st['uf'])")
replace("positive=100*torch.relu(.05-st['cam'][...,2]).square().mean()",
        "positive=100*torch.relu(.05-st['cam'][...,2]).square().mean()\n        positive=positive+100*(torch.relu(.05-st['facecam'][...,2]).square()*facw).sum()/facw.sum().clamp_min(1)")
replace("+.15*face+", "+face+")
replace("+.03*hd.tanh().square().mean()+positive",
        "+.03*hd.tanh().square().mean()+.01*jaw.square().mean()+.001*expression.square().mean()+positive")
replace("contact=contact)", "contact=contact,**{'face_'+k:v for k,v in face_regions.items()})")
replace("shared_identity=True,per_frame_visibility_hardcoding=False,", "shared_identity=True,per_frame_visibility_hardcoding=False,\n        direct_face_binding_policy='reviewed_anatomical_v1',\n        direct_face_review_sha256=binding_review['source_sha256'],\n        face_loss='Native recorded uncertainty; group-normalized; FIT only; no blanket eye/brow removal',\n        shared_jaw_and_expression_optimized=True,\n        candidate_not_official_smplifyx=True,")
replace("left_hand_pose=cpu(st['lh']),right_hand_pose=cpu(st['rh']),pelvis=cpu(st['pelvis']),flat_hand_mean=False,faces=model.faces)",
        "left_hand_pose=cpu(st['lh']),right_hand_pose=cpu(st['rh']),pelvis=cpu(st['pelvis']),flat_hand_mean=False,faces=model.faces,\n            jaw_pose=cpu(jaw),expression=cpu(expression))\n        np.savez_compressed(out/'direct-face-residuals.npz',projected=cpu(st['uf']),target=cpu(direct_face.uv),\n            error_px=cpu(direct_face.residuals(st['uf'])),fit_weight=cpu(direct_face.fit_weight),\n            held=cpu(direct_face.held),sigma_px=cpu(direct_face.sigma))")
replace("face_surface_visibility=cpu(face_vis)","face_surface_visibility=np.ones_like(cpu(face_vis)),face_model_visibility_diagnostic_only=cpu(face_vis)")
ast.parse(s)
target.write_text(s,encoding='utf-8')
# Candidate is intentionally not silently inserted into an already registered
# production run. The entry verifies camera admission and a fresh fit directory.
entry=OUT/'run_candidate.py'
entry.write_text('''from pathlib import Path
import argparse,sys,importlib.util
H=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
PROJECT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/runtime')
sys.path[:0]=[str(H),str(PROJECT/'src')]
from local_surface123_alignment_gate import require_camera_admission
from shuolee_portrait import runtime
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--steps',type=int,default=1200)
a=p.parse_args()
if (a.root/'fit').exists():raise RuntimeError('Fresh fitting only; preserve historical fits.')
require_camera_admission(a.root)
runtime.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',a.root)
spec=importlib.util.spec_from_file_location('shuolee_portrait.direct_fitting123',Path(__file__).with_name('candidate_fitting.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.main(a.root,a.steps)
''',encoding='utf-8')
(OUT/'candidate-manifest.json').write_text(json.dumps(dict(baseline_sha256=sha(BASE),
    candidate_sha256=sha(target),module_sha256=sha(H/'smplx_direct_correspondence123.py'),
    entry_sha256=sha(entry),production_runtime_unchanged=True,
    actual_objective_calls_direct_binding=True,real_fitting_run=False),indent=2)+'\n')
assert sha(BASE)==plan['baseline_fitter_sha256']
print(json.dumps(dict(candidate=str(target),real_fitting_run=False)))

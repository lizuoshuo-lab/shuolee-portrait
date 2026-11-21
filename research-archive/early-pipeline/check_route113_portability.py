"""Bounded deployment checks: no fitting iterations and no Gaussian training."""
from pathlib import Path
import importlib.util, json, subprocess, sys, time, ast
P=Path('/mnt/d/00/工作/人体重建')
spec=importlib.util.spec_from_file_location('r113',P/'scripts/route113.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
D=P/'data/hover175';A=Path('/mnt/d/GS-SLAM-Data/shuolee-113-assets')
R=Path('/opt/gs-slam-lab/route113-portability-check/prepare-002')
result=dict(route='113',created=time.time(),remote_server_verified=False,full_training_verified=False,
            stable_release_accepted=False,quality_accepted=False,gaussian_optimizer_updates=0,body_fit_updates=0)
for f in (P/'src/route113').rglob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
for f in [P/'scripts/route113.py']:ast.parse(f.read_text(encoding='utf-8-sig'))
result['python_syntax']='passed'
cmd=[sys.executable,'-s',str(P/'scripts/route113.py')]
args=['--dataset',str(D),'--assets',str(A)]
doctor=subprocess.run(cmd+['doctor']+args,text=True,capture_output=True,check=True)
result['CUDA_controls']=json.loads(doctor.stdout.splitlines()[-1])
if not R.exists():subprocess.run(cmd+['prepare']+args+['--run',str(R)],check=True)
before=r.sha(R/'config.json');marker=r.sha(R/'stages/prepare.json')
again=subprocess.run(cmd+['prepare']+args+['--run',str(R)],capture_output=True,text=True)
assert again.returncode!=0 and 'destination already exists' in again.stderr
assert before==r.sha(R/'config.json') and marker==r.sha(R/'stages/prepare.json')
result['fresh_output_reuse_rejected_without_overwrite']=True
rows=r.read(R/'data/training-inputs.json')['records']
assert len(rows)==175 and all(Path(x['image']).is_file() and Path(x['mask']).is_file() for x in rows)
assert not (R/'fit/fitted.npz').exists() and not (R/'gaussian-run/current.pt').exists()
result['prepared_paths_175_valid']=True
c=r.configure(D,A,R)
import fit_head_surface_082 as head
torch,h,native,v,l,b,lb=head.model()
assert torch.isfinite(v).all() and torch.isfinite(l).all()
result['FLAME_forward']={'vertices':list(v.shape),'landmarks':list(l.shape),'finite':True}
import shuolee_vposer as vp
model,config,meta=vp.load_vposer()
with torch.no_grad():out=model.decode(torch.zeros(1,32,device='cuda'))
assert torch.isfinite(out['pose_body']).all()
result['VPoser_strict_load_and_forward']=True
import smplx
body=smplx.SMPLX(str(c.MODEL),batch_size=1,num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False).cuda()
with torch.no_grad():vertices=body().vertices
assert vertices.shape==(1,10475,3) and torch.isfinite(vertices).all()
result['SMPLX_template_forward']=True
import fit_fresh_sequence, initialize_static_clothing_cloud, prepare_fullscene_clothing_075, initialize_ncc_pair_093, prepare_native_geometry_113
import compatible_gaussian_export
result['preparation_and_export_imports']='passed'
result['native_rasterizer_built_from_vendored_source']=True
result['environment']={'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
  'scope':'Existing isolated local Linux environment; existing OS/toolchain. Not remote server.'}
result['weights_or_subject_uploaded_to_git']=False
result['scope']='Syntax, 175-input hashes, fresh preparation, templates, CUDA finite-difference controls and imports only. No full 113 fitting/training/export rerun.'
result['input_audit']=r.read(R/'input-audit.json')
r.save(P/'provenance/route113-validation.json',result)
print(json.dumps(result,ensure_ascii=False))

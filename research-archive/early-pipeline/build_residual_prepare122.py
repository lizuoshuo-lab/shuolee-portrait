from pathlib import Path
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-fit'
dest=H/'prepare_native_person122_v2.py'
if dest.exists():raise FileExistsError(dest)
s=(H/'prepare_native_person122.py').read_text(encoding='utf-8')
s=s.replace('from native_training_admission122 import','from native_training_admission122_v2 import').replace('from native_surface122 import','from native_surface122_v2 import')
s=s.replace("anatomical_visibility='mesh_surface_zbuffer_v1'","anatomical_visibility='continuous_mesh_rays_v2'")
s=s.replace("wh=[w,h],K=camera['K'],w2c=camera['w2c'],","wh=[w,h],K=camera['K'],w2c=camera['w2c'],camera=camera,")
s=s.replace('fullK_integer_to_fitter_corner_only=True','complete_camera_KED_retained=True')
s=s.replace("from shuolee_portrait.fitting import main as fit","from fitting_output_residual122 import main as fit")
s=s.replace("from shuolee_portrait.fit_gate import evaluate","from fit_gate_output_residual122 import evaluate")
s=s.replace("PROJECT/'src/shuolee_portrait/fitting.py'","Path(__file__).parent/'fitting_output_residual122.py'")
s=s.replace("initializer_source=ref(Path(__file__).parent/'native_surface122.py')","initializer_source=ref(Path(__file__).parent/'native_surface122_v2.py')")
s=s.replace("doc=read(checked_asset(path.parent,m['cameras']));cam={c['frame_id']:c for c in doc['cameras']}","doc=read(checked_asset(path.parent,m['cameras']));cam={c['frame_id']:c['camera'] for c in doc['cameras']}")
s=s.replace("rows=[dict(w2c=cam[f['frame_id']]['w2c'],alpha=f['alpha'],observation_valid=f['observation_valid']) for f in m['frames']]","rows=[dict(frame_id=f['frame_id'],alpha=f['alpha'],observation_valid=f['observation_valid']) for f in m['frames']]")
s=s.replace("initialize_surface(fit,rows,doc['K'],path.parent","initialize_surface(fit,rows,cam,path.parent")
a=s.index('def dispatch_training(output):');b=s.index('\ndef main():',a)
s=s[:a]+'''def dispatch_training(output):
    raise RuntimeError('Residual preparation is versioned, but person optimizer orchestration is not yet connected. Do not fall back to the old pinhole entry.')

'''+s[b:]
s=s.replace("external_occluders_in_fitter='Legacy fit containment/model-visibility remains coarse; actual fit review must reject occlusion artifacts before surface. Surface separately uses observed-valid mask.'","external_occluders_in_fitter='Continuous self-visibility only; scene occluders unknown. Anatomical prior is coarse. Surface uses observed-valid mask.',visibility_policy='continuous_mesh_rays_v2',prior_is_coarse=True")
compile(s,str(dest),'exec');dest.write_text(s,encoding='utf-8')
p=O/'plan.json';plan=json.loads(p.read_text(encoding='utf-8'))
plan['source_discovery_before_execution']='Legacy fit_gate review overlay also had pinhole projection. Versioned coarse-gate metrics stay unchanged; overlay now traces all native observed pixels with the same complete camera. Added under original all-upstream projection consistency scope before test execution.'
p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
data=json.loads((O/'implementation.json').read_text(encoding='utf-8'))
for n in ['native_residual_fit122.py','fit_gate_output_residual122.py','prepare_native_person122_v2.py']:
    data['files'][str(H/n)]=hashlib.sha256((H/n).read_bytes()).hexdigest()
data.update(person_preparation_entry_wired=True,person_optimizer_entry_wired=False,remaining='New prepared inputs and actual materials still rejected; versioned optimizer dispatch pending. No positive personfit/18k execution claimed.')
(O/'implementation.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print('R6_PREPARE_FIT_SURFACE_CHAIN_READY_WITH_TRAIN_DISPATCH_BLOCKED')

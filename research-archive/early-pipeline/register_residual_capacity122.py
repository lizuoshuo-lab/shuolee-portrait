from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json
from native_residual_camera122 import projection,with_projection
from native_residual_domain122 import sampled_domain_check
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-residual-capacity'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists(),'Already registered; inspect instead'
assert read(R/'queue-state.json')['running'] is None
assert read(R/'native-residual-camera/result.json')['state']=='completed'
cvo=read(R/'native-cvo-control/plan.json');small=read(R/'native-residual-camera/cameras-fixture.json');cams=[]
for i,old in enumerate(cvo['native_cameras']):
    c=copy.deepcopy(old);c['id']='r3-'+['portrait','landscape'][i];c['image_name']=c['id']+'.png'
    c=with_projection(c,small[i]['residual_projection']);sampled_domain_check(c);cams.append(c)
inputs=[H/n for n in ['native_residual_domain122.py','native_residual_camera122.py','native_portrait_renderer122_v3.py','native_training122_v4.py','native_scene_package122_v5.py',
 'native_linear_archive122.py','native_gaussian_ply122.py','native_training122.py','native_training122_v2.py','native_scene_package122.py',
 'native_portrait_renderer122_v2.py','native_interactive122.py','native_interactive122_v2.py','native_interactive122_v3.py']]
inputs += [R/'native-training-integration/subpixel-culling-diagnosis/frozen-large-arrays.npz',R/'native-cvo-control/plan.json',R/'native-residual-camera/cameras-fixture.json']
inputs += [H/'probe_residual_capacity122.py',H/'run_residual_capacity122.sh']
stamp=datetime.now(timezone.utc).isoformat();O.mkdir()
plan=dict(stage='R3_full_native_residual_capacity',registered=stamp,
 hypothesis='The explicit residual K/E/D implementation must train and replay at actual portrait/landscape raster sizes and 200k points, not only tiny synthetic fixtures.',
 change='Same frozen synthetic geometry and full-size K/E fixtures as T8; explicit nonzero output Brown 6+2+4 coefficients are the new renderer variable. One fresh optimizer update over both cameras.',
 cameras=cams,optimizer=cvo['optimizer'],
 limits=dict(wall_seconds=300,points=200000,optimizer_steps=1,view_exposures=2,GPU_jobs=1,native_float_max=2e-5),
 controls=['Full native shapes, finite nonempty target, gradients to all parameter classes, actual free-world center update.',
 'Record exact linear arrays and pre-export float references before package encoding; native arrays and each K/E/D record exact.',
 'Both native replay errors measured independently; complete covariance and SH preview checked separately.',
 'No resolution/point fallback, no new camera estimation or image warp, no person training; preserve live v1/v2/v3.'],
 scope='Known synthetic coefficients only. Not real calibration, 18k stability, density adaptation, arbitrary coefficient bounds, or person-quality acceptance.',
 stop='Preserve first failure. No rerun or threshold sweep except a separately evidenced implementation repair under the same fixture.',
 input_sha256={str(p):sha(p) for p in inputs},physical_training_admitted=False)
save(O/'plan.json',plan);save(O/'execution-registration.json',dict(plan_sha256=sha(O/'plan.json'),sources=plan['input_sha256']))
q=read(R/'queue-state.json');q.update(updated=stamp,phase='R3_full_native_residual_registered',running=None)
q['last_goal_turn']=dict(classification='progress',completed='R3 registered after inspecting live CPU/GPU processes; previous R2 publication complete.',remaining='Execute once, then inspect full native results and versioned recorded-camera import.',publication_complete=False)
save(R/'queue-state.json',q)
print(json.dumps(dict(registered=True,cameras=[c['id'] for c in cams],points=200000)))

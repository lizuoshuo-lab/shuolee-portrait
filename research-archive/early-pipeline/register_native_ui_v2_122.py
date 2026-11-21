"""Freeze T2 UI controls and run request/CPU loader checks before starting GPU work."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import hashlib,json,socket,urllib.request,zipfile
import numpy as np
from native_interactive122_v2 import request_pair
from native_scene_package122_v2 import load_scene
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';A=R/'native-renderer-v2';O=R/'native-ui-v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
assert read(R/'queue-state.json')['running'] is None and read(A/'result-completed.json')['all_registered_controls_passed']
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8875))!=0,'Port already occupied'
assert not O.exists()
specs={'pinhole':('normal-pinhole','6点合成 · 针孔完整K'), 'kb4':('normal-opencv_fisheye4','6点合成 · 已知KB4鱼眼'), 'large':('large-package','20万点合成 · 修复后单步')}
scenes={name:load_scene(A/folder/'scene.json',check_runtime=False) for name,(folder,label) in specs.items()}
controls=[dict(scene_id='pinhole',camera_id='0'),dict(scene_id='pinhole',camera_id='1'),dict(scene_id='kb4',camera_id='0'),dict(scene_id='large',camera_id='0')]
checks=[]
def ok(name,value):
    checks.append(dict(name=name,passed=bool(value)))
    if not value:raise AssertionError(name)
def reject(name,fn,text):
    try:fn()
    except (ValueError,RuntimeError) as error:ok(name,text in str(error))
    else:ok(name,False)
for c in controls:ok('Allow registered '+str(c),request_pair(c,scenes,controls,[])==(c['scene_id'],c['camera_id']))
reject('No arbitrary path',lambda:request_pair(dict(scene_id='pinhole',camera_id='0',path='elsewhere'),scenes,controls,[]),'overrides')
reject('No projection substitution',lambda:request_pair(dict(scene_id='pinhole',camera_id='0',projection='fisheye'),scenes,controls,[]),'overrides')
reject('No camera override',lambda:request_pair(dict(scene_id='pinhole',camera_id='0',K=np.eye(3).tolist()),scenes,controls,[]),'overrides')
reject('Unknown package',lambda:request_pair(dict(scene_id='wrong',camera_id='0'),scenes,controls,[]),'Unknown')
reject('Unknown camera',lambda:request_pair(dict(scene_id='large',camera_id='1'),scenes,controls,[]),'Unknown')
reject('Outside once-only budget',lambda:request_pair(dict(scene_id='kb4',camera_id='2'),scenes,controls,[]),'Outside')
reject('No concurrent job',lambda:request_pair(controls[0],scenes,controls,[dict(state='running')]),'already running')
reject('Failure requires inspection',lambda:request_pair(controls[0],scenes,controls,[dict(state='failed')]),'Previous render failed')
reject('No duplicate',lambda:request_pair(controls[0],scenes,controls,[dict(controls[0],state='complete')]),'Already executed')
reject('Old schema cannot enter v2 UI',lambda:load_scene(R/'native-training-integration/existing-backend-path/final-native-package/scene.json',check_runtime=False),'v2 schema')
for sid,scene in scenes.items():
    ok(sid+' explicit projection',len(scene.manifest['projections'])==1)
    ok(sid+' complete covariance SH',scene.parameters['sh_degree']==3 and len(scene.parameters['means'])==scene.manifest['model']['vertices'])
O.mkdir();(O/'packages').mkdir()
immutable={};packages={}
for name in ['native_interactive122_v2.py','native_interactive122_v2.html','native_interactive_worker122_v2.sh','native_interactive122.py',
             'native_interactive122.html','native_interactive_worker122.sh','native_portrait_renderer122_v2.py','native_scene_package122_v2.py',
             'native_portrait_renderer122.py','native_scene_package122.py','native_gaussian_ply122.py','register_native_ui_v2_122.py']:
    immutable[str(H/name)]=sha(H/name)
for sid,scene in scenes.items():
    path=O/'packages'/('T2_'+sid+'_native_v2.zip')
    assets=[scene.path,scene.path.parent/scene.manifest['model']['asset']['path'],scene.path.parent/scene.manifest['cameras']['path']]
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
        for file in assets:archive.write(file,file.relative_to(scene.path.parent).as_posix());immutable[str(file)]=sha(file)
        archive.writestr('README.txt','Synthetic control only. Contains the original standard PLY, complete same-coordinate cameras and scene manifest.\nUse native_scene_package122_v2.py with matching backend/source hashes. Legacy v1 and Spark/SIBR preview are not native-equivalent.\nNo human calibration or quality acceptance.\n')
    with zipfile.ZipFile(path) as archive:
        for file in assets:assert hashlib.sha256(archive.read(file.relative_to(scene.path.parent).as_posix())).hexdigest()==sha(file)
    packages[sid]=dict(scene=str(scene.path),label=specs[sid][1],zip=str(path),zip_sha256=sha(path),zip_bytes=path.stat().st_size)
    immutable[str(path)]=sha(path)
with urllib.request.urlopen('http://127.0.0.1:8874/api/state',timeout=10) as response:old=json.load(response)
old={k:old[k] for k in ['scene_sha256','renderer','backend','jobs','remaining']}
save(O/'old-v1-state.json',old)
plan=dict(stage='T2',registered=now(),port=8875,
    hypothesis='Explicit per-package v2 loading, fixed recorded cameras and native worker must preserve T1 response and metadata through the actual UI and downloads.',
    scope='New isolated loopback GUI, no current v1 source/service change, no training/optimization/new calibration or input reprocessing.',
    packages=packages,controls=controls,budget=dict(native_renders=4,per_render_seconds=120,simultaneous_GPU=1,optimizer_steps=0),
    references={'pinhole:0':str(A/'cli-pinhole/native-rgb-alpha.npz'),'pinhole:1':str(A/'pinhole_1.png'),
                'kb4:0':str(A/'cli-opencv_fisheye4/native-rgb-alpha.npz'),'large:0':str(A/'large_updated.png')},
    tolerance=dict(npz_response_max_abs=3e-6,png_max_u8_difference=1),
    UI_checks=['Select pinhole camera0 and1, KB4camera0 andlargecamera0; fullK/w2c/size/projection shown.',
        'After package/camera change, clear incompatible result; completed results selectable without new GPU execution.',
        'Show actual RGB and Alpha output; download a native float array and a complete PLY/camera/manifest ZIP; compare bytes and decode saved camera identities.',
        'Retain old v1 response/service and119 historical preview, never claim actualperson orSpark/SIBRequivalence.'],
    limitations='Registered catalog/recorded views only, not arbitrary filesystem import, continuous orbit, full-resolution capacity or human quality.',
    stopping='On failure preserve process/log/source; inspect exact job. No duplicate task, threshold/camera changes or optimized initialization.',
    immutable_inputs=immutable,prior_T1_review_sha256=sha(A/'review.json'),previous_goal_turn='progress: T1 completed code/CUDA/CLI/review andpublication',
    no_physical_admission=True)
for p in plan['references'].values():plan['immutable_inputs'][p]=sha(p)
save(O/'plan.json',plan);save(O/'cpu-checks.json',dict(updated=now(),checks=checks,all_passed=True,count=len(checks),GPU_started=False))
q=read(R/'queue-state.json');q.update(updated=now(),phase='T2_native_v2_UI_registered',running=None,next='Run isolated8875 v2service; complete exactly4actualUIrenders anddownload byte/metadata checks, no training.');save(R/'queue-state.json',q)
print(json.dumps(dict(registered=True,CPU_checks=len(checks),packages={k:v['zip_bytes'] for k,v in packages.items()},port=8875)))

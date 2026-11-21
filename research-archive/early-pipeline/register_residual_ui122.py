"""Register two UI inference jobs on the completed R3 full-size package."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,socket,urllib.request,zipfile,warnings
from native_interactive122_v4 import inspect_zip,CORE
from native_interactive122_v2 import request_pair
from native_scene_package122_v5 import load_scene
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-residual-ui';A=R/'native-residual-capacity';S=A/'native-package'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists(),'Already registered; inspect instead'
assert read(A/'progress.json')['state']=='completed'
assert all(c['passed'] for c in read(A/'result.json')['checks'])
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8877))!=0
scene=load_scene(S/'scene.json',check_runtime=False)
O.mkdir();native=O/'R4_residual_native.zip'
with zipfile.ZipFile(native,'x',zipfile.ZIP_STORED) as z:
    for name in sorted(CORE):z.write(S/name,name)
    z.writestr('README.txt','Known synthetic control only. No physical portrait acceptance.\nNative arrays: native-parameters.npz; preview: point_cloud.ply.\nFull recorded K/E and per-camera Brown 6+2+4 output-residual coefficients: cameras.json.\nDo not silently discard coefficients for a pinhole preview. No raw fisheye image or vendor FISHEYE_8 model.\nUse the exact renderer and scene-v5 source hashes in scene.json.\nNo optimizer checkpoint or resumed training.\n')
checks=[]
def ok(name,v):
    checks.append(dict(name=name,passed=bool(v)))
    if not v:raise AssertionError(name)
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError,zipfile.BadZipFile):ok(name,True)
    else:ok(name,False)
blob=native.read_bytes();digest=sha(native);files=inspect_zip(blob,digest)
ok('Archive preserves every full native asset',all(hashlib.sha256(files[n]).hexdigest()==sha(S/n) for n in CORE))
ok('Complete per-camera coefficient records',len(scene.cameras)==2 and all(set(c['residual_projection'])=={'schema','input_domain','pixels','mapping','units','radial','tangential','thin_prism'} for c in scene.cameras.values()))
reject('Wrong hash',lambda:inspect_zip(blob,'0'*64))
reject('Empty file',lambda:inspect_zip(b'',digest))
reject('Non ZIP',lambda:inspect_zip(b'x',digest))
old=R/'native-ui-v3/packages/T9_exact_native.zip'
reject('Old zero-distortion package is not residual package',lambda:inspect_zip(old.read_bytes(),digest))
def badzip(names=None,compression=zipfile.ZIP_STORED):
    b=io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(b,'w',compression) as z:
            for n in names or sorted(CORE|{'README.txt'}):z.writestr(n,b'{}')
    return b.getvalue()
reject('Missing native arrays',lambda:inspect_zip(badzip(sorted((CORE|{'README.txt'})-{'native-parameters.npz'})),digest))
reject('Path traversal',lambda:inspect_zip(badzip(sorted(CORE|{'README.txt','../bad'})),digest))
reject('Duplicate entry',lambda:inspect_zip(badzip(sorted(CORE|{'README.txt'})+['scene.json']),digest))
reject('Unexpected compression',lambda:inspect_zip(badzip(compression=zipfile.ZIP_DEFLATED),digest))
controls=[dict(scene_id='exact',camera_id=c) for c in scene.cameras];scenes={'exact':scene}
for c in controls:ok(c['camera_id']+' recorded request',request_pair(c,scenes,controls,[])==('exact',c['camera_id']))
reject('Import first',lambda:request_pair(controls[0],{},controls,[]))
reject('No camera or coefficient override',lambda:request_pair(dict(controls[0],radial=[0]*6),scenes,controls,[]))
reject('No concurrent GPU jobs',lambda:request_pair(controls[0],scenes,controls,[dict(state='running')]))
reject('No retry after failure',lambda:request_pair(controls[0],scenes,controls,[dict(state='failed')]))
reject('No rerender cached camera',lambda:request_pair(controls[0],scenes,controls,[dict(controls[0],state='complete')]))
snapshots={}
for version,port in [('v1',8874),('v2',8875),('v3',8876)]:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state',timeout=10) as f:d=json.load(f)
    snapshots[version]=dict(jobs=d['jobs'],remaining=d['remaining'])
save(O/'previous-ui-state.json',snapshots)
sources=['native_interactive122_v4.py','native_interactive122_v4.html','native_interactive_worker122_v4.sh','native_residual_render_cli122.py',
 'native_scene_package122_v5.py','native_residual_domain122.py','native_residual_camera122.py','native_portrait_renderer122_v3.py',
 'native_linear_archive122.py','native_scene_package122.py','native_scene_package122_v2.py','native_gaussian_ply122.py',
 'native_interactive122.py','native_interactive122_v2.py','native_interactive122_v3.py','register_residual_ui122.py']
immutable={str(H/n):sha(H/n) for n in sources}
for name in CORE:immutable[str(S/name)]=sha(S/name)
references={c:str(A/(c+'-preexport-float.npz')) for c in scene.cameras}
for p in [native,O/'previous-ui-state.json',*map(Path,references.values())]:immutable[str(p)]=sha(p)
stamp=datetime.now(timezone.utc).isoformat()
plan=dict(stage='R4_residual_actual_UI',registered=stamp,port=8877,
 hypothesis='Actual downloaded/imported native package must retain nonlinear recorded K/E/D and exact arrays through real UI rendering, camera switch and float download.',
 scope='Known synthetic package only; two new inference jobs, no optimization or camera estimation; old live services unchanged.',
 package=dict(zip=str(native),zip_sha256=digest,assets={n:sha(S/n) for n in CORE}),camera_ids=list(scene.cameras),controls=controls,
 budget=dict(native_renders=2,per_render_seconds=120,GPU_jobs=1,optimizer_steps=0),references=references,
 tolerance=dict(native_float_response_max=2e-5),immutable_inputs=immutable,
 UI_checks=['Download package through actual UI and import downloaded ZIP via file chooser.',
 'Read full K/E/D; render portrait and landscape once, inspect RGB and Alpha with original aspect ratio.',
 'Download full float response and verify against reference; switch/reload cached view without new GPU.',
 'Reject old zero-distortion native package and preserve loaded residual scene; do not claim generic pinhole viewers preserve D.'],
 stop='Preserve first failed render; no repeat or threshold change. Physical calibration and person quality remain unaccepted.')
save(O/'plan.json',plan);save(O/'cpu-checks.json',dict(checks=checks,count=len(checks),passed=True))
q=read(R/'queue-state.json');q.update(updated=stamp,phase='R4_residual_UI_registered',running=None);save(R/'queue-state.json',q)
print(json.dumps(dict(registered=True,checks=len(checks),zip_bytes=len(blob),port=8877)))

"""Freeze import/download and two recorded-camera UI checks; never optimize."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib, io, json, socket, urllib.request, warnings, zipfile
import numpy as np
from native_interactive122_v3 import inspect_zip, CORE, MAX_ZIP
from native_interactive122_v2 import request_pair
from native_scene_package122_v3 import load_scene

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-ui-v3';A=R/'native-cvo-control';S=A/'exact-native-package'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
assert not (O/'plan.json').exists(),'Already registered; do not rerun'
assert read(R/'queue-state.json')['running'] is None
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8876))!=0,'New service port occupied'
scene=load_scene(S/'scene.json',check_runtime=False)
native=O/'packages/T9_exact_native.zip'
with zipfile.ZipFile(native,'x',zipfile.ZIP_STORED) as archive:
    for name in sorted(CORE):archive.write(S/name,name)
    archive.writestr('README.txt','Synthetic control only; no human calibration acceptance.\nCanonical inference parameters: native-parameters.npz, linear float32 with no renormalization.\nRecorded cameras: cameras.json, full K and OpenCV world-to-camera, pixel_corners.\nUse native_scene_package122_v3.py with the exact backend/source hashes declared in scene.json.\nNo optimizer checkpoint or authorization to resume training.\npoint_cloud.ply is a standard preview only; its response is not identical.\nFor Spark/SIBR-compatible layout and standard cameras, use the separately labelled preview ZIP.\n')
checks=[]
def ok(name,value):
    checks.append(dict(name=name,passed=bool(value)))
    if not value:raise AssertionError(name)
def reject(name,call):
    try:call()
    except (ValueError,RuntimeError,zipfile.BadZipFile):ok(name,True)
    else:ok(name,False)
blob=native.read_bytes();digest=sha(native);files=inspect_zip(blob,digest)
ok('Native ZIP preserves every original asset',all(hashlib.sha256(files[n]).hexdigest()==sha(S/n) for n in CORE))
ok('Linear fields retain full covariance and SH3',scene.parameters['sh'].shape==(200000,16,3))
reject('Wrong package digest',lambda:inspect_zip(blob,'0'*64))
reject('Empty file',lambda:inspect_zip(b'',digest))
reject('Non ZIP bytes',lambda:inspect_zip(b'not a package',digest))
def smallzip(names=None,compression=zipfile.ZIP_STORED,schema='wrong'):
    b=io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(b,'w',compression) as z:
            for name in names or sorted(CORE|{'README.txt'}):z.writestr(name,json.dumps({'schema':schema}) if name=='scene.json' else b'x')
    return b.getvalue()
reject('Wrong scene schema',lambda:inspect_zip(smallzip(),digest))
reject('Missing native arrays',lambda:inspect_zip(smallzip(sorted((CORE|{'README.txt'})-{'native-parameters.npz'})),digest))
reject('Archive path traversal',lambda:inspect_zip(smallzip(sorted(CORE|{'README.txt','../other'})),digest))
reject('Duplicate archive entry',lambda:inspect_zip(smallzip(sorted(CORE|{'README.txt'})+['scene.json']),digest))
reject('Unexpected compressed input',lambda:inspect_zip(smallzip(compression=zipfile.ZIP_DEFLATED),digest))
preview=O/'packages/T9_standard_preview.zip'
reject('Preview is not native package',lambda:inspect_zip(preview.read_bytes(),digest))
controls=[dict(scene_id='exact',camera_id=c) for c in scene.cameras]
scenes={'exact':scene}
for c in controls:ok('Recorded '+c['camera_id'],request_pair(c,scenes,controls,[])==('exact',c['camera_id']))
reject('Import required first',lambda:request_pair(controls[0],{},controls,[]))
reject('Camera override forbidden',lambda:request_pair(dict(controls[0],K=[]),scenes,controls,[]))
reject('No unknown view',lambda:request_pair(dict(scene_id='exact',camera_id='unknown'),scenes,controls,[]))
reject('No concurrent GPU render',lambda:request_pair(controls[0],scenes,controls,[dict(state='running')]))
reject('Failure is not silently retried',lambda:request_pair(controls[0],scenes,controls,[dict(state='failed')]))
reject('Recorded result reused, not re-rendered',lambda:request_pair(controls[0],scenes,controls,[dict(controls[0],state='complete')]))
with zipfile.ZipFile(preview) as z:
    m=json.loads(z.read('scene.json')); cameras=json.loads(z.read('cameras.json'))
    ok('Preview original PLY byte identity',hashlib.sha256(z.read(m['model']['file'])).hexdigest()==sha(S/'point_cloud.ply'))
    ok('Preview rendering scope explicit',m['rendererContract']['nativeImageEquivalence']=='known-not-equivalent')
    ok('SIBR layout and full-K text model exist',all(n in z.namelist() for n in ['cfg_args','input.ply','cameras.txt','images.txt','camera_provenance.json']))
    for c,n in zip(cameras,scene.cameras.values()):
        ok('Standard K preserved '+n['id'],[c['fx'],c['fy'],c['cx'],c['cy']]==[n['K'][0][0],n['K'][1][1],n['K'][0][2],n['K'][1][2]])
        rot=np.array(c['rotation']).T; t=-rot@np.array(c['position'])
        ok('Standard world pose preserved '+n['id'],np.max(np.abs(np.c_[rot,t]-np.array(n['w2c'])[:3]))<1e-6)
snapshots={}
for version,port in [('v1',8874),('v2',8875)]:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state',timeout=10) as response:data=json.load(response)
    snapshots[version]=dict(jobs=data['jobs'],remaining=data['remaining'])
save(O/'previous-ui-state.json',snapshots)
sources=['native_interactive122_v3.py','native_interactive122_v3.html','native_interactive_worker122_v3.sh',
 'native_interactive122_v2.py','native_interactive122.py','native_scene_package122_v3.py','native_linear_archive122.py',
 'native_scene_package122_v2.py','native_scene_package122.py','native_portrait_renderer122_v2.py','native_portrait_renderer122.py',
 'native_gaussian_ply122.py','register_native_ui_v3_122.py','package_native_ui_v3_preview122.mjs']
immutable={str(H/n):sha(H/n) for n in sources}
for n in ['scene-package.js','camera-formats.js','camera-io.js','render-contract.js','point-cloud-loader.js','spark-camera-template.js']:
    immutable[str(T/'viewer'/n)]=sha(T/'viewer'/n)
for n in CORE:immutable[str(S/n)]=sha(S/n)
for file in [native,preview,O/'previous-ui-state.json']:immutable[str(file)]=sha(file)
references={c:str(A/(c+'-preexport-float.npz')) for c in scene.cameras}
for file in references.values():immutable[file]=sha(file)
plan=dict(stage='T9',registered=now(),port=8876,
 hypothesis='Exact canonical arrays, recorded full cameras and backend contract survive real ZIP download, OpenFile import, two native UI renders and floating output download without conversion.',
 scope='Synthetic inference only, isolated v3 loopback service. V1/v2 services and119 model stay unchanged. No calibration or optimization.',
 package=dict(zip=str(native),zip_sha256=digest,assets={n:sha(S/n) for n in CORE}),
 preview=dict(path=str(preview),sha256=sha(preview)),camera_ids=list(scene.cameras),controls=controls,
 budget=dict(native_renders=2,per_render_seconds=120,simultaneous_GPU=1,optimizer_steps=0,imports=1),
 references=references,tolerance=dict(native_float_response_max=2e-5),immutable_inputs=immutable,
 UI_checks=['Actual browser native ZIP download then file chooser import; two recorded full K/E cameras read.',
 'Render portrait and landscape once, actual RGB and alpha review, no aspect deformation.',
 'Switch to cached camera without new job; download floating NPZ and standard preview ZIP, decode bytes and cameras.',
 'Preview cannot import as exact native; preserve current loaded exact model after rejection.',
 'Old v1/v2 state unchanged; no historical119 model replacement or official externalGUI claim.'],
 stopping='On mismatch/failure preserve evidence; no automatic rerender, parameter/threshold change or person training.',
 limitations='Registered synthetic package only; not arbitrary third-party import, continuous orbit or physical portrait acceptance.')
save(O/'plan.json',plan);save(O/'cpu-checks.json',dict(updated=now(),checks=checks,count=len(checks),all_passed=True,training_started=False,
 preparation_errors=['Initial new Node packaging script encoded query as literal filename, corrected before output.',
 'Caller passed dot-prefixed extension; existing scenePackage API correctly rejected it. Corrected to ply before output.']))
q=read(R/'queue-state.json');q.update(updated=now(),phase='T9_exact_native_UI_registered',running=None,
 next='Run v3 service8876; actual import/download and exactly two recorded-camera renders. No new training.');save(R/'queue-state.json',q)
print(json.dumps(dict(registered=True,checks=len(checks),native_bytes=len(blob),preview_bytes=preview.stat().st_size)))

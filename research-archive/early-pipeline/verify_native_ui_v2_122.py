"""Verify the actual UI jobs/downloads after the registered four renders."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request,zipfile
import numpy as np
from PIL import Image
from native_scene_package122_v2 import load_scene
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-ui-v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'result.json').exists(),'Verification already recorded'
plan=read(O/'plan.json');state=read(O/'ui-state.json');checks=[]
def check(name,value,**details):
    checks.append(dict(name=name,passed=bool(value),**details))
    if not value:raise AssertionError(name)
result=dict(stage='T2',started=datetime.now(timezone.utc).isoformat(),checks=checks,physical_calibration_accepted=False,human_training_started=False)
try:
    check('Exactly registered four UI jobs completed',len(state['jobs'])==4 and all(j['state']=='complete' for j in state['jobs']))
    check('Actual job identities equal registered controls',[dict(scene_id=j['scene_id'],camera_id=j['camera_id']) for j in state['jobs']]==plan['controls'])
    for name,digest in plan['immutable_inputs'].items():check('Frozen '+Path(name).name,sha(name)==digest)
    for j in state['jobs']:
        scene=load_scene(plan['packages'][j['scene_id']]['scene'],check_runtime=False)
        check('Same recorded camera '+j['id'],j['camera']==scene.cameras[j['camera_id']] and j['scene_sha256']==scene.manifest_sha256)
        check('Explicit version and projection '+j['id'],j['adapter_revision']==scene.manifest['backend']['adapter_revision'] and j['projection']==next(iter(scene.manifest['projections'])))
        check('Native worker and no training '+j['id'],j['native_linux_pid']>0 and j['exit_code']==0 and j['training_started'] is False)
        path=O/j['id'];ref=Path(plan['references'][j['scene_id']+':'+j['camera_id']]);actual=np.load(path/'native-rgb-alpha.npz')
        check('Native size and finite output '+j['id'],actual['rgb'].shape==(j['camera']['height'],j['camera']['width'],3) and np.isfinite(actual['rgb']).all() and np.isfinite(actual['alpha']).all())
        if ref.suffix=='.npz':
            expected=np.load(ref);difference={k:float(np.abs(actual[k]-expected[k]).max()) for k in ['rgb','alpha']}
            check('Independent CLI reference '+j['id'],max(difference.values())<plan['tolerance']['npz_response_max_abs'],max_abs=difference)
        else:
            delta=int(np.abs(np.asarray(Image.open(path/'rgb.png')).astype('int16')-np.asarray(Image.open(ref)).astype('int16')).max())
            check('Previous direct-render PNG '+j['id'],delta<=plan['tolerance']['png_max_u8_difference'],max_u8_difference=delta)
    downloaded=Path('C:/Users/Administrator/Downloads/T2_pinhole_native_v2.zip')
    check('Actual browser ZIP bytes equal published package',sha(downloaded)==plan['packages']['pinhole']['zip_sha256'],download=str(downloaded))
    extracted=O/'browser-downloaded-package';extracted.mkdir(exist_ok=False)
    original=Path(plan['packages']['pinhole']['scene']).parent
    with zipfile.ZipFile(downloaded) as archive:
        check('ZIP exact members',set(archive.namelist())=={'scene.json','point_cloud.ply','cameras.json','README.txt'})
        for name in ['scene.json','point_cloud.ply','cameras.json']:
            data=archive.read(name);check('Downloaded member '+name,hashlib.sha256(data).hexdigest()==sha(original/name));(extracted/name).write_bytes(data)
    loaded=load_scene(extracted/'scene.json',check_runtime=False)
    check('Downloaded package fullK and covariance SH decode',len(loaded.cameras)==3 and loaded.parameters['sh_degree']==3 and loaded.cameras['0']['K']==state['jobs'][0]['camera']['K'])
    floatdownload=Path('C:/Users/Administrator/Downloads/native-rgb-alpha (1).npz')
    check('Actual browser float arrays byte match',sha(floatdownload)==state['jobs'][0]['files']['native-rgb-alpha.npz'],download=str(floatdownload))
    with urllib.request.urlopen('http://127.0.0.1:8874/api/state',timeout=10) as response:old=json.load(response)
    previous=read(O/'old-v1-state.json');check('Existing v1 service and completed results unchanged',{k:old[k] for k in previous}==previous)
    for name in ['ui-pinhole0.png','ui-pinhole1.png','ui-kb4.png','ui-large.png','ui-alpha.png']:check('Actual UI screenshot '+name,(O/name).stat().st_size>5000)
    result.update(state='completed',all_controls_passed=True,native_UI_scope='Registered package selection,4fixedcamera renders,RGB/Alpha anddownload byte/metadata verification',
        broad_interactive_viewer_accepted=False,arbitrary_file_import_tested=False,continuous_orbit_tested=False,
        actual_visual_review='Six anisotropic blobs remain visible in pinhole0/1 andKB4. The64x64 large200k image has a dim small ellipsoid andcoloredoutline, consistent withpriorone-stepfixture. These are synthetic wiring controls, not a humanmodel.',
        UI_source='Cua IAB tab42',human_calibration=False,company00_tested=False,old_v1_preserved=True,
        next='Keep matched v2 source/package rules. Integrate physical material/full-frame/camera admission into an actual training orchestration entry before allowing new person runs. Do not repeat camera experiments without new evidence.')
except Exception as error:
    result.update(state='failed',error=repr(error));raise
finally:
    result['updated']=datetime.now(timezone.utc).isoformat();save(O/'result.json',result)
print(json.dumps(dict(state=result['state'],checks=len(checks),UI_scope=result.get('native_UI_scope'))))

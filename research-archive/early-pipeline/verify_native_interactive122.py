from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,numpy as np

R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\robust-portrait-122')
U=R/'native-interactive';D=U/'launcher-repair';O=R/'native-package-loader'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
assert not (U/'result.json').exists()
p=read(D/'plan.json');s=read(D/'ui-state.json');checks=[]
assert len(s['jobs'])==4 and all(j['state']=='complete' for j in s['jobs'])
for j,c in zip(s['jobs'],p['controls']):
    assert {k:j[k] for k in ['camera_id','projection']}==c
    output=D/j['id'];ref=O/(j['projection']+'_'+j['camera_id'])
    a=np.load(output/'native-rgb-alpha.npz');b=np.load(ref/'native-rgb-alpha.npz')
    errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['rgb','alpha']}
    assert all(v<=3e-6 for v in errors.values())
    for name,digest in j['files'].items():assert sha(output/name)==digest
    assert j['camera']==next(c for c in read(O/'fixture-package/cameras.json') if c['id']==j['camera_id'])
    assert j['native_receipt']['scene_sha256']==sha(O/'fixture-package/scene.json')
    checks.append(dict(id=j['id'],camera=j['camera_id'],projection=j['projection'],native_linux_pid=j['native_linux_pid'],
        raw_max_abs=errors,exact_camera_and_package=True,passed=True))
download=Path(r'C:\Users\Administrator\Downloads\native-rgb-alpha.npz')
assert sha(download)==s['jobs'][-1]['files']['native-rgb-alpha.npz']
for path,digest in p['immutable_inputs'].items():assert sha(Path(path))==digest
screens=['01-pinhole0-ui.png','02-pinhole2-ui.png','03-fisheye0-ui.png','04-fisheye2-alpha-ui.png','04-fisheye2-final-ui.png']
assert all((D/x).is_file() for x in screens)
controls=read(U/'request-controls.json');assert controls['passed']
result=dict(stage='R1e',updated=datetime.now(timezone.utc).isoformat(),passed=True,scope='synthetic recorded-camera native UI',
    renders=checks,request_boundary_checks=len(controls['checks']),actual_UI_render_buttons=4,actual_download_SHA=sha(download),
    render_sources_unchanged=True,continuous_orbit_verified=False,physical_camera_or_face_admission=False,training_started=False,
    repairs=[dict(kind='display_hidden_CSS',before_GPU=True),dict(kind='Windows_to_WSL_inline_arguments',
        evidence='failed-launch-ui-state.json',native_renders_before_repair=0,raw_failure_preserved=True)],
    screenshots=[dict(file='launcher-repair/'+x,sha256=sha(D/x)) for x in screens],
    scope_note='Core headless receipt retains native_interactive_UI=false; this separate receipt records actual browser controls and native output. No SIBR/Spark/company00 claim.')
save(U/'result.json',result)
save(U/'viewer-ui-review.json',dict(updated=result['updated'],tab='41',url='http://127.0.0.1:8874/',
    actual_four_controls_reviewed=True,RGB_and_alpha_reviewed=True,download_roundtrip_verified=True,
    selector_change_clears_previous_frame=True,reload_no_new_job=True,completed_budget_disables_render=True,
    new_model_published=False,tab15_untouched=True,remote00=False))
print(json.dumps(dict(passed=True,raw_max_abs=[x['raw_max_abs'] for x in checks],download_sha256=sha(download))))

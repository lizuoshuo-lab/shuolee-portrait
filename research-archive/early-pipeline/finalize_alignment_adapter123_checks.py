from pathlib import Path
from datetime import datetime,timezone
import ast,copy,json,subprocess,sys,time
import numpy as np
P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
H=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=R/'generic-2d3d-research/adapter-a0'
sys.path.insert(0,str(P/'src'))
from shuolee_portrait.alignment_inputs import read,sha,load_alignment_inputs
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
start=time.monotonic();log=O/'cpu-tests-final.log'
assert not log.exists()
with log.open('x') as f:
    run=subprocess.run([sys.executable,str(P/'scripts/check_alignment_inputs.py')],stdout=f,stderr=subprocess.STDOUT,timeout=100)
print(log.read_text(),flush=True)
assert run.returncode==0
import torch,smplx
torch.set_num_threads(4)
model=smplx.SMPLX('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz',num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False)
b=load_alignment_inputs(O/'synthetic-adapter-inputs.json');objective,receipt=b.make_face_objective(model,device='cpu')
bad=copy.deepcopy(b);bad.records[0]['original_review']['anchor']=copy.deepcopy(receipt['bindings'][0]);bad.records[0]['original_review']['anchor']['vertex_indices'][0]+=1
try:bad.make_face_objective(model,device='cpu')
except ValueError as e:assert 'surface binding changed' in str(e)
else:raise AssertionError('Wrong anatomical surface binding was accepted.')

intake=O/'real-native-intake'
run=subprocess.run([sys.executable,str(P/'run.py'),'fit-inputs','--output',str(intake)],capture_output=True,text=True,timeout=30)
assert run.returncode==0,run.stderr
report=json.loads(run.stdout)
assert report['frames']==177 and report['fixed_face_observations']==9027 and report['camera_records']==0
assert not report['fitting_allowed'] and report['selected_implementation']=='shuolee_portrait.fitting_direct._fit_prepared'
save(O/'real-cli-preflight.json',report)

def function(path,name):
    return next(ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
gate=H/'local_surface123_alignment_gate.py';new=P/'src/shuolee_portrait/reviewed_admission.py'
assert function(gate,'evaluate')==function(new,'evaluate')
assert function(gate,'require_camera_admission')==function(new,'require_camera_admission')
before=read(O/'before-source-hashes.json')
for name,h in before.items():
    if name.replace('\\','/')!='src/shuolee_portrait/cli.py':assert sha(P/name.replace('\\','/'))==h,name
modules=['alignment_inputs.py','fit_backend.py','fitting_direct.py','smplx_correspondence.py','reviewed_admission.py','cli.py']
for name in modules:ast.parse((P/'src/shuolee_portrait'/name).read_text())
result=read(O/'result.json')
save(O/'result-initial.json',result)
result.update(updated=datetime.now(timezone.utc).isoformat(),unit_and_CLI_tests=16,
    rerun_reason='Added fail-closed native image hash and recorded surface-binding checks; corrected per-call asset lookup after interface review.',
    template_binding_negative_control_passed=True,admission_function_AST_identical=True,
    real_actual_CLI_checked=True,actual_modules={name:sha(P/'src/shuolee_portrait'/name) for name in modules},
    cumulative_CPU_seconds=result['seconds']+time.monotonic()-start,
    test_script_sha256=sha(P/'scripts/check_alignment_inputs.py'),
    package_construction_note='Source port completed before a Windows path-separator KeyError in its final print; no source rollback or repeated generation. Corrected the print and verified all source files and preserved hashes.')
assert result['cumulative_CPU_seconds']<120
save(O/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='previous_source_files_preserved'},ensure_ascii=False),flush=True)

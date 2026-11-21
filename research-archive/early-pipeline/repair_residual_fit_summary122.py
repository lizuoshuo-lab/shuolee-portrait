"""Correct one summary field using the immutable already-recorded check."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-fit')
H=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'result-completed.json').exists()
original=read(O/'result.json');check=read(O/'checks-progress.json')[0]
assert original['checks'][0]==check and check['name']=='Four varyingKED Torch projections match independent OpenCV' and check['passed']
assert original['CPU_projection_max_px']==9922 and check['max_coordinate_error_px']<2e-10
assert sha(O/'executed-check-script.py')==read(O/'execution-registration.json')['script_sha256']
corrected=dict(original);corrected['CPU_projection_max_px']=check['max_coordinate_error_px'];corrected['summary_repair']='serialization-repair.json'
save(O/'result-completed.json',corrected)
save(O/'serialization-repair.json',dict(updated=datetime.now(timezone.utc).isoformat(),field='CPU_projection_max_px',old=9922,new=check['max_coordinate_error_px'],
    cause='Driver reused local e for a sphere triangle vertex index after recording the actual CPU projectioncheck; only the final summary used overwritten e.',
    original_result_sha256=sha(O/'result.json'),check_record_sha256=sha(O/'checks-progress.json'),executed_script_sha256=sha(O/'executed-check-script.py'),
    corrected_script_sha256=sha(H/'check_residual_fit122.py'),CPU_or_GPU_rerun=False,component_sources_changed=False,all_checks_retained=True))
print('R6_SUMMARY_CORRECTED_FROM_RECORDED_CHECK_NO_RERUN')

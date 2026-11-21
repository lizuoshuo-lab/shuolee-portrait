"""Record the unexecuted R7 source boundary; never run or relabel old controls."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-pipeline'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(O/'plan.json');progress=read(O/'progress.json')
assert plan['stage']=='R7' and progress['test_executed'] is False
files=['prepare_native_person122_v3.py','train_native_person122_v2.py','native_training122_v5.py','native_pipeline_binding122.py','native_training_policy122.py']
source=(H/'train_native_person122_v2.py').read_text(encoding='utf-8')
start=source.index('class PixelCache:');end=source.index('\nclass DiskView:',start);fragment=source[start:end]
assert "key=(frame['frame_id'],digest(camera),digest(frame))" in fragment and "for key in ['rgb','alpha','rgb_valid','observation_valid']" in fragment and 'self.current=key' in fragment
stamp=datetime.now(timezone.utc).isoformat()
data=dict(updated=stamp,registered_plan=str(O/'plan.json'),plan_sha256=sha(O/'plan.json'),previous_goal_turn='progress',
 source_drafts={str(H/n):sha(H/n) for n in files},test_executed=False,source_modified=False,human_training=False,
 actual_current_gpu_command='PID309 train_head_motion.py --motion fan --steps18000; observed alive with elapsed969seconds in this turn',
 source_findings=[dict(file=str(H/'train_native_person122_v2.py'),symbol='PixelCache.get',finding='The requested frame/camera digest tuple is stored in key, but the pixel-channel loop reuses key; self.current receives the last channel name instead of the request tuple. Repeated same-frame property requests cannot hit this cache.',runtime_reproduced=False,not_a_proven_alignment_cause=True)],
 next='Use the existing single R7 suite and budgets: first exercise the actual CPU PixelCache with known small RGB/alpha/masks, preserve the initial failure, repair only the demonstrated key-shadowing defect, then finish source/preparation/review/policy controls. GPU portion may run only after the live00 job finishes and the shared lock is free; no real person fit or training. Do not regenerate or rerun earlier R1-R6 fixtures.',
 goal_or_quality_complete=False)
save(O/'source-preflight-20261029.json',data)
print(json.dumps(dict(path=str(O/'source-preflight-20261029.json'),test_executed=False,source_modified=False)))

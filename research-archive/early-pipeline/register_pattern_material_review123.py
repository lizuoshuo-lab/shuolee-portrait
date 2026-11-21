from pathlib import Path
from datetime import datetime,timezone
import json
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123';PRE=REC/'pattern-preflight';O=PRE/'native-material'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
assert not (O/'result.json').exists() and not (O/'review-plan.json').exists()
progress=read(O/'progress.json');assert progress['pid']==322
save(O/'review-plan.json',dict(registered=stamp,before_completed_tracks=True,source_slots=[0,24,49,74,98],
    regions=['head','person','background'],splits=['FIT','HELD'],queries_per_slot_region_split=1,maximum_families=30,
    selection='At each predeclared slot/region/split, eligible unquarantined appearance tracks containing the slot are sorted by familySHA256 then trackSHA256. Select the first family not selected in an earlier row. No residual, confidence, pose or camera used.',
    order='For each slot ascending, regions head/person/background then FIT/HELD.',
    views='Use chronological first/query/last observations; if query equals first or last, use middle observation as third. Always3distinctframes from the same frozen track.',
    crops='192x192 native crops centered floor(uv), no resize or image text. Black beyond source bound retained. Source and feature hashes checked.',
    visual_labels='coarse_material_consistent, wrong_material, occlusion_or_identity_unknown, no_available_track. Review any position ambiguity explicitly; coarse resemblance never certifies subpixel identity.',
    followup='Representative samples diagnose support, not whole-family or all-frame accuracy. Review all observations of any proposed physical validation family before admitting camera/fit evidence.',
    exclusion_policy='Do not change saved observations or FIT/HELD. Invalid/unknown physical labels remain evidence; no error-based pruning.',
    cameras_started=0,SMPLX_started=0,training_started=False))
for root in [R,REC]:
    p=root/'queue-state.json';q=read(p);q.update(updated=stamp,current_stage='pattern_native_material_inventory',phase='native_material_support',
        running=dict(pid=322,platform='WSL',command=progress['command'],exec_session=64315,kind='CPU_native_feature_inventory'),worker_pid=322,
        training_running=False,local_GPU_task_active=False,latest_evidence=str(O/'progress.json'),
        previous_goal_turn_classification='progress: completed99framevisibility,4nativefootmislabels,updatedexistingviewer/report/PPT')
    save(p,q)
print(json.dumps(dict(review_plan=str(O/'review-plan.json'),verified_live_PID=322,previous_turn='progress',training=False),ensure_ascii=False))

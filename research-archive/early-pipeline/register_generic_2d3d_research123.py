"""Apply the user's universal-pipeline priority; no experiment execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123';O=REC/'generic-2d3d-research'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not O.exists();O.mkdir();(O/'sources').mkdir()
stamp=datetime.now(timezone.utc).isoformat()
inventory=read(REC/'indoor-preflight/native-material/result.json')
assert inventory['state']=='complete_physical_material_review_pending'
assert not (REC/'indoor-preflight/native-material/topology-review').exists()
decision='按用户最新要求，先完成通用2D到3D人体对应方案及现有工程整体问题的源码调研；不继续逐人物专用修补。室内原生匹配已结束，保留原始证据，未启动拓扑审阅、相机、SMPL-X或高斯训练。调研完成后才登记冻结配置的跨人物适应性测试。'
plan=dict(updated=stamp,state='research_in_progress',user_priority='generic_engineering_research_before_adaptive_tests',
    requirements=['Single reusable image/camera/keypoint/body-model contracts.','Use established upstream anatomical mappings and fitting methods where applicable.',
    'No subject/frame-index conditions in production core; per-video metadata and optimized person parameters remain allowed.',
    'Separate observation semantics, camera calibration, 3D fitting and clothed surface reconstruction.',
    'Research first; no new fitting, topology, detector, training or weight downloads during this review.'],
    upstream_scope=['SMPL-X and SMPLify-X','MMHuman3D conventions and SMPLify-X','EasyMocap calibrated multiview','Multi-HMR initializer','DensePose surface correspondence','Two previously authorized Gaussian avatar repositories'],
    local_scope=['Direct correspondence implementation','Actual runtime fitter','Camera observation admission','Per-person visibility helpers'],
    evidence_policy='Primary upstream source and local code/actual existing results. Separate implementation gap from experimentally established ghosting cause.',
    carried_inventory=dict(result=str(REC/'indoor-preflight/native-material/result.json'),sha256=sha(REC/'indoor-preflight/native-material/result.json'),frames=177,training=False),
    next='Write source-backed comparison, system-level findings and preregistered universal adaptation plan; publish existing viewer and log.')
save(O/'plan.json',plan)
for root in [R,REC]:
    p=root/'queue-state.json';q=read(p);q.update(updated=stamp,current_stage='generic_2d3d_research_before_adaptation',active_subject=None,current_subject=None,
        running=None,worker_pid=None,worker_command=None,training_running=False,training_started=False,local_GPU_task_active=False,
        next=decision,next_action=decision,next_plan=str(O/'plan.json'),latest_evidence=str(O/'plan.json'),
        user_latest_priority='No dataset-specific tuning; investigate open-source2D-to3D methods before adaptation tests')
    save(p,q)
for name in ['protocol.json','next-action.json']:
    p=REC/name;q=read(p);q.update(updated=stamp,current_stage='generic_2d3d_research_before_adaptation',next=decision,next_action=decision,next_plan=str(O/'plan.json'));save(p,q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,current_stage='generic_2d3d_research_before_adaptation',next=decision,next_plan=str(O/'plan.json'),training_running=False);save(p,q)
save(REC/'indoor-preflight/material-deferred-by-user.json',dict(updated=stamp,inventory_complete=True,topology_not_started=True,reason=decision))
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8');s+='\n\n### 2026-10-29 通用工程优先\n\n'+decision+'\n';p.write_text(s,encoding='utf-8')
print(json.dumps(dict(state=plan['state'],inventory_completed=inventory['frames'],new_test_started=False),ensure_ascii=False))

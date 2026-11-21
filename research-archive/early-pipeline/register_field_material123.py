"""Register the first local field texture inventory, with unchanged source pixels."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = T/'shuolee/current/local-clothed-surface-123'
F = REC/'field-preflight'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (F/'native-material').exists() and not (F/'native-material-plan.json').exists()
assert read(REC/'observation-visibility-consumer-audit/result.json')['state']=='complete_read_only'
old=H/'pattern_native_material123.py'
s=old.read_text(encoding='utf-8-sig')
replacements={
    "PRE=REC/'pattern-preflight'":"PRE=REC/'field-preflight'",
    "preflight/pattern')":"preflight/field')",
    'n==99==len(decoded':'n==177==len(decoded',
    '/tmp/shuolee-pattern-material123.lock':'/tmp/shuolee-field-material123.lock',
    'all464temporalpairs':'all854temporalpairs',
    'len(pairs)==464':'len(pairs)==854',
    "    if any(x['regions'][0]['kept']==0 for x in manifest):raise RuntimeError('At least one headROI lacks native texture support; no second extraction or threshold retry.')":
    "    # A detector ROI on an invisible back-face need not contain facial texture.\n"
    "    # Keep every frame and record zero support; do not pretend that it is a visible face.\n"
    "    save(O/'empty-head-roi.json',dict(slots=[x['slot'] for x in manifest if x['regions'][0]['kept']==0],\n"
    "        no_second_extraction=True,visible_face_support_not_established=True))"
}
for a,b in replacements.items():
    assert s.count(a)==1,(a,s.count(a))
    s=s.replace(a,b)
new=H/'field_native_material123.py'
assert not new.exists()
new.write_text(s,encoding='utf-8')
compile(s,str(new),'exec')
stamp=datetime.now(timezone.utc).isoformat()
plan=read(REC/'pattern-preflight/native-material-plan.json')
dec=read(F/'decoded.json')
assert len(dec['records'])==177
plan.update(registered=stamp,state='registered_not_started',subject='field',
    source='D:/00/工作/数据集/video7.mp4',source_sha256=dec['video_sha256'],
    decoded_manifest=str(F/'decoded.json'),decoded_manifest_sha256=sha(F/'decoded.json'),
    ROI_observations_sha256=sha(F/'coverage/observations.json'),frames=177,
    hypothesis='The first field native texture inventory can separate repeatable appearance support from semantic back-face predictions before any geometric solve.',
    not_a_camera_or_K_comparison=True,all_predeclared_source_indices=[r['source_frame_index'] for r in dec['records']],
    preceding_consumer_audit_sha256=sha(REC/'observation-visibility-consumer-audit/result.json'),
    implementation=dict(path=str(new),sha256=sha(new),reference_path=str(old),reference_sha256=sha(old),
        changes=['Subject/frame count and 854 temporal pairs.',
                 'Record empty detector head ROIs instead of globally failing on an invisible back-face; no quota/threshold changes.']),
    stop='One 1200s CPU pass only. Stop on source/frame identity mismatch, absent native image, timeout or existing output. Record empty ROI support rather than inventing observations. Do not retry or solve cameras automatically.')
plan['method']['roi_semantics']='Face/body boxes only localize searches; back-face boxes are not visible facial material.'
plan['method']['topology_policy']='After immutable extraction, apply the previously checked bridge isolation once before review; preserve original family identity and split. Cycles are not physical truth.'
plan['review']=dict(source_slots=[0,44,88,132,176],regions=['head','person','background'],splits=['FIT','HELD'],
    max_queries=30,selection='First topologically eligible family sorted by family SHA then segment SHA containing each predeclared slot/region/split, excluding already selected families.',
    views='First/query/last; replace repeated endpoint by chronological middle; three distinct original frames.',
    crops='192x192 original pixels, floor(query), no resampling or text.',
    scope='Diagnostic sampled physical identity only. All sampled HELD families marked development; no promotion to FIT.',
    camera_or_fit_admission=False)
save(F/'native-material-plan.json',plan)
for root in [REC,T/'shuolee/current/robust-portrait-122']:
    q=read(root/'queue-state.json');q.update(updated=stamp,current_stage='field_first_native_material_registered',
        current_subject='field',active_subject='field',next_plan=str(F/'native-material-plan.json'),
        previous_goal_turn_classification='progress: registered source audit completed; exact historical consumer identity verified; field missing weights preserved as unknown',
        training_running=False,training_allowed=False,local_GPU_task_active=False)
    save(root/'queue-state.json',q)
print(''.join(difflib.unified_diff(old.read_text(encoding='utf-8-sig').splitlines(True),s.splitlines(True),fromfile=old.name,tofile=new.name)))
print(json.dumps(dict(plan=str(F/'native-material-plan.json'),script=str(new),frames=177,pairs=854),ensure_ascii=False))

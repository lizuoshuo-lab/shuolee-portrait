"""Freeze one full-view review of the ten already opened background families."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
F=REC/'field-preflight';O=F/'native-material';B=O/'topology-review'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
out=F/'background-material-review-plan.json';assert not out.exists()
index=read(B/'native-review/index.json');rows=[q for q in index['rows'] if q['query_region']=='background']
assert len(rows)==10 and len({q['family_id'] for q in rows})==10
segments={q['id']:q for q in read(B/'segments.json')['records']}
selected=[]
for row in rows:
    s=segments[row['segment_id']]
    assert s['family_id']==row['family_id'] and s['split']==row['split'] and s['development_only']
    selected.append(dict(review_row=row['row'],query_slot=row['query_slot'],family_id=s['family_id'],segment_id=s['id'],split=s['split'],development_only=True,nodes=s['nodes']))
assert sum(len(q['nodes']) for q in selected)==102
stamp=datetime.now(timezone.utc).isoformat()
plan=dict(registered=stamp,state='registered_not_executed',
    question='已有背景候选的整段观测，是否确为同一个静止物理点；首中尾三张检查是否遗漏中间错误连接？',
    change_from_previous_review='Review every original view of the exact ten previously opened background segments, rather than only three views. No new selection or matching.',
    inputs={str(p):sha(p) for p in [B/'segments.json',B/'native-review/index.json',B/'actual-review.json',O/'feature-manifest.json']},
    selected=selected,observations=102,output=str(B/'background-full-review'),
    representation=dict(pixel_domain='native decoded MP4 integer-centre coordinates',crop_size=192,raw_context_and_center_crop=True,resampling=False,burned_text=False,source_sha_required=True),
    budget=dict(passes=1,CPU_seconds=120,CPU_threads=2,GPU_tasks=0,new_feature_extraction=0,new_matches=0,new_camera=0,SMPLX_fit=0,training=0),
    review=dict(required=['same_material_identity','static_scene_support','ambiguity_or_occlusion','whole_segment_views_checked'],
        uncertainty='Unknown unless measured independently; native visual review alone cannot certify1px.',
        roi='Background ROI is a query label, not physical staticity.',
        splits='Keep original FIT/HELD labels and development-only status. Opened HELD is not a new independent final benchmark.',
        exclusions='No fit-error-based point removal. Preserve every observation and label conflict at the supported scope.'),
    stop=['Do not add families or tune descriptors when ambiguous. Save unknown/conflict evidence.',
          'If full-view physical support remains inadequate, do not launch camera or fitting on these ten tracks.',
          'This bounded review alone cannot authorize a177-frame camera candidate.'],
    downstream='Only a reviewed native observation adapter may consume material evidence. The legacy camera function re-extracts and does not read these segments, so it must not be launched unchanged.')
save(out,plan)
nexttext='执行已登记的background-material-review-plan：仅展开此前已选10条背景轨迹的全部102个原生观测，检查整段材料身份和静止性，保持FIT/HELD与development标记。无新特征、无相机求解或训练。随后据实际证据决定原生观测消费接口，禁止直接启动会重新提取的旧camera函数。'
for root in [R,REC]:
    p=root/'queue-state.json';q=read(p);q.update(updated=stamp,next=nexttext,next_action=nexttext,next_plan=str(out));save(p,q)
p=REC/'next-action.json';q=read(p);q.update(updated=stamp,next=nexttext,next_action=nexttext,next_plan=str(out));save(p,q)
p=REC/'protocol.json';q=read(p);q.update(updated=stamp,next=nexttext,next_action=nexttext,next_plan=str(out));save(p,q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,next=nexttext,next_plan=str(out));save(p,q)
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8');s+='\n\n下一项已登记：'+nexttext+'\n';p.write_text(s,encoding='utf-8')
print(json.dumps(dict(plan=str(out),families=10,observations=102,executed=False),ensure_ascii=False))

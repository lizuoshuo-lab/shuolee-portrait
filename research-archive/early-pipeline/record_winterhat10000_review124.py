from pathlib import Path
from datetime import datetime, timezone
import json
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
OUT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/winterhat-selected-sh0-pipeline124'
snap=OUT/'gaussian-run/step-010000'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
result=read(snap/'result.json');assert result['step']==10000
save(snap/'visual-review.json',dict(updated=now,step=10000,actual_visual_review=True,
    overview_body_face_feet_reviewed=[13,16,41,56,57,71,96,109],
    finding='All eight native-view comparisons were reviewed. Face, eyes, lips and hat texture are clearer than at1000; the yellow skirt and dark boots remain soft and spotty. Source mesh arm/foot depth uncertainties remain. The clothing mesh offset is still small (maximum0.000365m), so photometric clothing coverage is not mesh-surface recovery. No gross new visual collapse; continue the same bounded run. Novel views remain unreviewed at this checkpoint.',
    offset=read(snap/'offset-update.json'),novel_views_reviewed=False,
    quality_accepted=False,source_pose_quality_accepted=False,decision='continue_same_optimizer_to18000'))
coord=H.parent/'shuolee-coordination/remy00-delivery-20261103.json'
d=read(coord);d.update(updated=now,
    company00_failure_stage='after selecting Remy in model selector',
    zip_bug_explains_reported_failure=False,company00_verified=False,
    page_version='2026.11.03.3',
    local_model_select_verified=True,
    model_select_diagnostic='404 now shows requested file and current launcher root, plus direct paired ZIP import steps.',
    exact_model_label='Remy · 沉浸空间261103_112806 · 外部参考',
    older_separate_label='Remy 分享模型 · 外部参考',
    root_cause='00 requested URL/root and synchronized file presence still unverified')
save(coord,d)
print('WINTERHAT10000_REVIEW_AND_REMY404_SCOPE_RECORDED')

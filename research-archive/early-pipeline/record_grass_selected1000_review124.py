from pathlib import Path
from datetime import datetime,timezone
import json
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
OUT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/grass-selected-sh0-pipeline124'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
snap=OUT/'gaussian-run/step-001000';now=datetime.now(timezone.utc).isoformat()
request=read(OUT/'review-1000-request.json');geometry=read(OUT/'gaussian-run/geometry-progress.json')
assert geometry['step']==1000 and geometry['new_relative_normal_reversals']==0
assert not (OUT/'review-1000-decision.json').exists()
review=dict(updated=now,step=1000,actual_visual_review=True,
    overview_body_face_feet_reviewed=[4,12,30,46,49,58,70],
    native_crops_reviewed=['frame_000046 face','frame_000070 feet'],
    finding='已审阅7个训练视角的原图／mesh／高斯及原生面部、脚部裁片。早期高斯已覆盖白衣、头发、脸和鞋，衣服仍有黑色孔隙和模糊，手指及脚趾细节不足。未发现新增网格翻折或整体崩坏，可以继续同一轮至18,000步。源拟合的手臂前后深度尚未被证实修复，本轮仅检验已冻结流程的高斯效果；不把训练视角投影当成侧面几何验收。',
    geometry=geometry,novel_views_reviewed=False,quality_accepted=False,
    source_pose_quality_accepted=False,decision='continue_same_optimizer_to18000')
save(snap/'visual-review.json',review)
save(OUT/'review-1000-decision.json',dict(updated=now,decision='continue',
    world_ply_sha256=request['world_ply_sha256'],same_live_optimizer=True,maximum_step=18000,
    review=str(snap/'visual-review.json')))
for p in [T/'shuolee/current/robust-portrait-122'/n for n in ['protocol.json','queue-state.json','evidence-ledger.json']]+[Path(__file__).parent.parent/'shuolee-coordination/local-research-status.json']:
    d=read(p);d['updated']=now;entry=d['grass_selected_sh0_pipeline124']
    entry.update(step_reviewed=1000,actual_visual_review=True,decision='continue_same_optimizer_to18000',quality_accepted=False)
    d['current_stage']='grass_selected_sh0_pipeline124_after1000_review';save(p,d)
print('GRASS_SELECTED1000_ACTUAL_REVIEW_CONTINUE_SAME_OPTIMIZER')

"""Finalize from saved review and the observed positive three-view case."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';Q=R/'video10-correspondence-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (Q/'decision.json').exists()
v=read(Q/'visual-review.json');stats=v['statistics'];stamp=datetime.now(timezone.utc).isoformat()
save(Q/'decision-draft-failure.json',dict(updated=stamp,phase='CPU decision drafting only',reason='Draft incorrectly assumed the selected eye-region cycle failed. The saved numerical result actually passes in bothdirections. Assertion prevented publishing that false claim.',
    saved_actual_cycles=stats['candidate58_cycle_checks'],inference_repeated=False,visual_review_preserved=True))
v['row58_three_view_review']=dict(image='review/contexts/row_058_threeview.png',actual_review=True,
    observation='Three saved native crops671/644/698 plus returned671 show the same coarse glasses/eye-region. No independently annotated exact physical point or HELD measurement; lens/reflection stability not certified.',
    no_new_context_family=True,no_new_inference=True)
save(Q/'visual-review.json',v)
summary='同一查询位置的七组RoMa对照完成，72个预登记位置和16个扩大上下文全部审阅；15个通过固定置信度/往返筛查，其中8个粗背景、1个粗人物、6个身份仍有歧义。12个旧粗背景控制只有4个通过该筛查。可见遮挡误配仍可能有0.08px往返误差，但本轮置信度门槛拦下该例。唯一眼镜/眼周粗候选为FIT，三视角循环通过；循环误差0.075至0.174px只是自洽误差，独立面部HELD仍为0。没有证据准入新相机或SMPL-X重拟合。'
d=dict(updated=stamp,state='bounded_matcher_contrast_complete_not_admitted',summary=summary,stats=stats,
    cycles_are_self_consistency_only=True,selected_probe_not_all_video=True,old_SIFT_not_rerun=True,
    RoMa_attempt_consumed=True,thresholds_changed=False,new_correspondence_inference_allowed=False,
    new_camera=False,new_fit=False,new_training=False,quality_accepted=False,
    next='Read-only inventory of remaining original background families at graph gaps; review existing evidence before any further feature extraction, matcher inference or camera estimate. Do not conclude all636candidates were unusable from a selected subset.')
save(Q/'decision.json',d)
prog=read(Q/'progress.json');prog.update(updated=stamp,state='complete_reviewed_not_admitted',pid=None,quality_accepted=False);save(Q/'progress.json',prog)
print(summary)

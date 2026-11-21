"""Persist inspected source/mask, fit/surface and first-1000-step evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
p=REC/'winterhat-input-000000-review/actual-review.json';d=read(p)
d['previous_review_index_sha256']=d['index_sha256'];d['index_sha256']=sha(p.parent/'index.json')
d.update(updated=stamp,person_masks_actually_reviewed=True,
    mask_findings='六个原图/遮罩对照已实际审阅。帽子、上衣、裙摆、靴子均保留；红色挎包在部分帧被漏分割，正面frame_000067尤明显。',
    face_ROI_caveat='frame_000001检测器ROI是后侧帽子，不能计入有效人脸覆盖。')
save(p,d)
entry=dict(kind='winterhat_actual_1000step_review',date=stamp,steps=1000,initial_points=200000,
    inputs_fresh=True,old_model_loaded=False,old_optimizer_loaded=False,
    reviews=[str(REC/f'winterhat-{s}-review/actual-review.json') for s in ['input-000000','fit-000000','surface-000000','train-001000']],
    finding='衣服/裙面/靴子有自由点支持；鼻嘴仍有明显重影、包和鞋边拉丝。有限同一轮继续18k，质量未通过。',
    current_quality_accepted=False,physical_calibration_accepted=False,
    screenshot=str(REC/'progress-training-ui.png'),screenshot_sha256=sha(REC/'progress-training-ui.png'))
p=REC/'evidence-ledger.json';d=read(p)
if not any(x.get('kind')==entry['kind'] for x in d['entries']):d['entries'].append(entry)
d['updated']=stamp;save(p,d)
p=R/'queue-state.json';d=read(p);d.update(updated=stamp,phase='LOCAL_SURFACE123_actual_training',training_started=True,
    formal_training_admitted=False,local_surface_round=str(REC),
    next='Continue finite local surface123 queue; each person has actual fit/surface/1000-step reviews and completed export/UI quality checks.',
    last_goal_turn=dict(classification='progress',current=entry['finding'],publication_complete=True));save(p,d)
p=T/'shuolee/long-term-plan.md'
if '本地表面自由点训练123' not in p.read_text(encoding='utf-8-sig'):
    with p.open('a',encoding='utf-8') as f:f.write('\n\n## 2026-10-28 本地表面自由点训练123\n用户优先本地数据集训练与查看器交付，复用沙发119的静态自由衣物表面方法。沙发保留，其余7人物逐个从头拟合、20万初始点、空优化器，1000步实际审阅后同轮最多18k，不修改正在运行的源码。第一段112个预登记真实帧已注册，1000步实际审阅衣物中心分布成立，脸和脚仍重影，质量未通过。目标继续，不恢复定时跟进。每人导出world/+Y完整协方差/SH和配套camera，实际OpenFile Gaussian/points通过再发布。\n')
save(REC/'progress-ui-review.json',dict(updated=stamp,actual_UI_checked=True,tab_id='40',
    URL='http://127.0.0.1:8870/local-surface-123/index.html',observed_training_step=500,
    historical_sofa_preserved=True,screenshot=entry['screenshot'],sha256=entry['screenshot_sha256'],
    new_model_UI_verified=False))
print(json.dumps(entry,ensure_ascii=False))
